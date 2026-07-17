// =============================================================================
// mnist_utils.rs — shared MNIST dataset helpers
// =============================================================================
//
// Both `train_diffusion.rs` and `train_diffusion_cond.rs` need identical
// logic for downloading, parsing, and normalising MNIST binary files.
// Putting it here keeps the training binaries focused on model logic and
// avoids copy-paste drift between files.
//
// MNIST binary format reference:
//   http://yann.lecun.com/exdb/mnist/
// =============================================================================

use anyhow::{bail, Result};
use candle_core::{DType, Device, Tensor};
use flate2::read::GzDecoder;
use std::fs::{create_dir_all, File};
use std::io::{BufReader, Read};
use std::path::Path;

// Preserve the long-standing `mnist_utils::save_png` import path while the
// encoder lives in a focused module with its own validation tests.
pub use super::image_io::save_png;

// =============================================================================
// acquire_mnist_images — download (once) and load the training image file
// =============================================================================
//
// WHY a separate function?
//   Keeps training entrypoints focused on model logic.
//   Download is idempotent: if the file already exists the HTTP request is
//   skipped entirely, so re-running training is fast.
//
// The MNIST image file is distributed as a GZIP-compressed IDX3-ubyte binary.
// We download the compressed form and decompress it on-the-fly to disk.
pub fn acquire_mnist_images(device: &Device) -> Result<Tensor> {
    let dest_path = "mnist/MNIST/raw/train-images-idx3-ubyte";
    let dest_dir = Path::new("mnist/MNIST/raw");

    if !Path::new(dest_path).exists() {
        println!("MNIST images not found locally. Downloading...");

        // `mkdir -p` equivalent: create every missing directory in the path.
        create_dir_all(dest_dir)?;

        // Public GitHub mirror of the original MNIST files.
        // The original Yann LeCun server is rate-limited, so this mirror is
        // more reliable for automated training pipelines.
        let url = "https://raw.githubusercontent.com/fgnt/mnist/master/train-images-idx3-ubyte.gz";
        println!("Downloading images from: {}", url);

        // Blocking HTTP GET — we don't need async I/O here.
        let response = reqwest::blocking::get(url)?;

        // Any non-2xx status is treated as a hard error.
        // WHY? A silent 404 would write an HTML error page to disk and cause a
        // confusing magic-number failure at parse time.
        if !response.status().is_success() {
            bail!(
                "Failed to download MNIST images. HTTP status: {}",
                response.status()
            );
        }

        println!("Decompressing GZIP archive to {}...", dest_path);

        // GzDecoder wraps the HTTP response body and decompresses it
        // as bytes are read — no need to buffer the full compressed archive.
        let mut gz_decoder = GzDecoder::new(response);
        let mut out_file = File::create(dest_path)?;
        std::io::copy(&mut gz_decoder, &mut out_file)?;
        println!("Image download and extraction complete!");
    }

    load_mnist_images(dest_path, device)
}

// =============================================================================
// acquire_mnist_labels — download (once) and load the training label file
// =============================================================================
//
// Parallel to `acquire_mnist_images` but for the IDX1-ubyte label file.
// Labels are needed by the class-conditioned model to build one-hot vectors.
pub fn acquire_mnist_labels() -> Result<Vec<u8>> {
    let dest_path = "mnist/MNIST/raw/train-labels-idx1-ubyte";
    let dest_dir = Path::new("mnist/MNIST/raw");

    if !Path::new(dest_path).exists() {
        println!("MNIST labels not found locally. Downloading...");
        create_dir_all(dest_dir)?;

        let url = "https://raw.githubusercontent.com/fgnt/mnist/master/train-labels-idx1-ubyte.gz";
        println!("Downloading labels from: {}", url);

        let response = reqwest::blocking::get(url)?;
        if !response.status().is_success() {
            bail!(
                "Failed to download MNIST labels. HTTP status: {}",
                response.status()
            );
        }

        println!("Decompressing labels...");
        let mut gz_decoder = GzDecoder::new(response);
        let mut out_file = File::create(dest_path)?;
        std::io::copy(&mut gz_decoder, &mut out_file)?;
        println!("Label download and extraction complete!");
    }

    load_mnist_labels(dest_path)
}

// =============================================================================
// acquire_mnist — convenience wrapper that loads both images and labels
// =============================================================================
//
// Used by `train_diffusion_cond.rs` which needs both modalities.
// Returns `(images_tensor, labels_vec)`.
pub fn acquire_mnist(device: &Device) -> Result<(Tensor, Vec<u8>)> {
    let images = acquire_mnist_images(device)?;
    let labels = acquire_mnist_labels()?;
    Ok((images, labels))
}

// =============================================================================
// load_mnist_images — raw IDX3-ubyte binary parser → Candle Tensor
// =============================================================================
//
// The IDX3 format (magic = 0x0803 = 2051) layout:
//
//   Offset | Bytes | Value
//   -------| ----- | -----
//   0      | 4     | Magic number (big-endian u32): must equal 2051
//   4      | 4     | Number of images  (big-endian u32)
//   8      | 4     | Rows per image    (big-endian u32) → 28
//   12     | 4     | Cols per image    (big-endian u32) → 28
//   16     | N*R*C | Raw pixel bytes, one u8 per pixel, row-major order
//
// WHY big-endian?
//   The IDX format was designed before modern little-endian conventions.
//   `u32::from_be_bytes` handles the byte-order swap transparently.
//
// WHY affine(1/127.5, -1.0)?
//   Raw pixel values are integers in [0, 255].
//   After `x * (1/127.5) - 1.0` they land in [-1.0, 1.0].
//   This zero-centred range matches the Gaussian prior N(0, I) used in the
//   DDPM forward process and stabilises training.
//
// Output tensor shape: (num_images, rows*cols) — already flattened for the MLP.
pub fn load_mnist_images(path: &str, device: &Device) -> Result<Tensor> {
    let file = File::open(path)?;
    // BufReader batches small reads into larger OS calls, which matters when
    // reading many small header fields.
    let mut reader = BufReader::new(file);

    // --- Step 1: Validate the magic number ----------------------------------
    // If this check fails we probably opened the wrong file.  Fail loudly
    // rather than silently producing garbage tensors.
    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic)?;
    let magic_num = u32::from_be_bytes(magic);
    if magic_num != 2051 {
        bail!(
            "Invalid MNIST image magic number: {} (expected 2051)",
            magic_num
        );
    }

    // --- Step 2: Read dimension metadata (12 bytes = 3 × u32) ---------------
    let mut meta = [0u8; 12];
    reader.read_exact(&mut meta)?;

    let num_images = u32::from_be_bytes([meta[0], meta[1], meta[2], meta[3]]) as usize;
    let rows = u32::from_be_bytes([meta[4], meta[5], meta[6], meta[7]]) as usize;
    let cols = u32::from_be_bytes([meta[8], meta[9], meta[10], meta[11]]) as usize;
    println!("Loaded {} images of size {}×{}", num_images, rows, cols);

    // --- Step 3: Read the full pixel buffer ---------------------------------
    let mut buffer = vec![0u8; num_images * rows * cols];
    reader.read_exact(&mut buffer)?;

    // Build tensor: cast u8 → f32, then normalise [0,255] → [-1,1].
    // We flatten to (N, rows*cols) immediately because the MLP operates on
    // 1-D image vectors, not 2-D spatial grids.
    let tensor = Tensor::from_vec(buffer, (num_images, rows * cols), device)?
        .to_dtype(DType::F32)?
        .affine(1.0 / 127.5, -1.0)?;

    Ok(tensor)
}

// =============================================================================
// load_mnist_labels — raw IDX1-ubyte binary parser → Vec<u8>
// =============================================================================
//
// IDX1 format (magic = 0x0801 = 2049) layout:
//
//   Offset | Bytes | Value
//   -------| ----- | -----
//   0      | 4     | Magic number (big-endian u32): must equal 2049
//   4      | 4     | Number of labels (big-endian u32)
//   8      | N     | One byte per label, value in {0, …, 9}
pub fn load_mnist_labels(path: &str) -> Result<Vec<u8>> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    // Validate magic number for the label file (2049, not 2051).
    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic)?;
    let magic_num = u32::from_be_bytes(magic);
    if magic_num != 2049 {
        bail!(
            "Invalid MNIST label magic number: {} (expected 2049)",
            magic_num
        );
    }

    // Read the item count.
    let mut meta = [0u8; 4];
    reader.read_exact(&mut meta)?;
    let num_items = u32::from_be_bytes(meta) as usize;

    // Read all label bytes.  Each is a digit class in {0, …, 9}.
    let mut buffer = vec![0u8; num_items];
    reader.read_exact(&mut buffer)?;

    Ok(buffer)
}

// =============================================================================
// make_one_hot — convert class labels to one-hot encoded tensor
// =============================================================================
//
// WHY one-hot encoding?
//   The class-conditioned MLP receives the digit class as part of its input.
//   One-hot encoding gives each class its own dedicated input dimension,
//   preventing the model from treating class 9 as "more" than class 1 (which
//   would happen with a raw integer feature).
//
// Output shape: (labels.len(), num_classes=10)
// Each row is all zeros except for a single 1.0 at the class index.
pub fn make_one_hot(labels: &[u8], device: &Device) -> Result<Tensor> {
    let num_classes = 10usize; // MNIST has digits 0–9
    let mut one_hot_vec = Vec::with_capacity(labels.len() * num_classes);

    for &label in labels {
        // Allocate a zero row.
        let mut row = vec![0.0f32; num_classes];
        // Set the correct class position to 1.0.
        // Guard against out-of-range labels (shouldn't happen with clean MNIST).
        if (label as usize) < num_classes {
            row[label as usize] = 1.0;
        }
        one_hot_vec.extend_from_slice(&row);
    }

    // Build a 2-D tensor from the flat buffer.
    Ok(Tensor::from_vec(
        one_hot_vec,
        (labels.len(), num_classes),
        device,
    )?)
}
