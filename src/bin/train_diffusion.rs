// =============================================================================
// train_diffusion.rs — DDPM (Denoising Diffusion Probabilistic Model) trainer
// =============================================================================
//
// This binary implements a minimal DDPM pipeline on the MNIST digit dataset.
// A two-layer MLP is trained to denoise images, then the trained model is used
// to generate a fresh digit image via the reverse diffusion (sampling) process.
//
// Paper reference: "Denoising Diffusion Probabilistic Models" (Ho et al., 2020)
//   https://arxiv.org/abs/2006.11239
// =============================================================================

// `bail!` lets us return early with a descriptive error string.
// `Result` is anyhow's alias for std::result::Result<T, anyhow::Error>.
use anyhow::{bail, Result};

// Candle is a lightweight ML framework (similar to PyTorch) written in Rust.
// `DType`  — data type of tensor elements (F32, U32, …)
// `Device` — where tensors live (CPU or CUDA GPU)
// `Tensor` — the core n-dimensional array type
use candle_core::{DType, Device, Tensor};

// The MNIST file we download is GZIP-compressed.
// GzDecoder streams the decompressed bytes without allocating the full archive.
use flate2::read::GzDecoder;

// Local model components for the diffusion pipeline:
//   `get_time_embedding`  — sinusoidal positional encoding of the timestep
//   `BetaScheduler`       — pre-computes beta/alpha/sigma schedules
//   `MlpAdamOptimizer`    — Adam optimizer wrapper for the MLP weights
//   `sample_ddpm`          — reverse diffusion sampler used after training
//   `SimpleDenoisingMlp`  — two-layer MLP: input → hidden → output
use llm_scratch_rs::models::diffusion::{
    get_time_embedding, sample_ddpm_from_noise, BetaScheduler, MlpAdamOptimizer, SimpleDenoisingMlp,
};

// Standard filesystem helpers used for caching the downloaded dataset.
use std::fs::{create_dir_all, File};
use std::io::{BufReader, Read};
use std::path::Path;

// =============================================================================
// acquire_mnit — dataset acquisition (download + cache)
// =============================================================================
//
// WHY a separate function?
//   Keeps `main` focused on the training loop rather than data logistics.
//   Also lets us skip the download on subsequent runs (idempotent).
//
// The MNIST IDX3-ubyte format stores images as raw big-endian bytes.
// We download the GZIP-compressed version from a public GitHub mirror
// and decompress it directly to disk.
fn acquire_mnit(device: &Device) -> Result<Tensor> {
    let dest_path = "mnist/MNIST/raw/train-images-idx3-ubyte";
    let dest_dir = Path::new("mnist/MNIST/raw");

    // Only download if the file is not already cached locally.
    // This avoids re-downloading on every training run.
    if !Path::new(dest_path).exists() {
        println!("MNIST dataset not found locally. Preparing programmatic download...");

        // Recursively create all parent directories (like `mkdir -p`).
        create_dir_all(dest_dir)?;

        // Public GZIP mirror of the original MNIST binary files.
        // WHY this URL? The original Yann LeCun site is rate-limited; this
        // GitHub mirror is more reliable for automated downloads.
        let url = "https://raw.githubusercontent.com/fgnt/mnist/master/train-images-idx3-ubyte.gz";
        println!("Downloading from: {}", url);

        // Perform a synchronous (blocking) HTTP GET request.
        // We use the blocking variant because we don't need async here.
        let response = reqwest::blocking::get(url)?;

        // Treat any non-2xx HTTP status as a hard failure — we don't want to
        // silently write a corrupted file (e.g. a 404 HTML page) to disk.
        if !response.status().is_success() {
            bail!(
                "Failed to download dataset. HTTP Status: {}",
                response.status()
            );
        }

        println!("Decompressing GZIP archive to {}...", dest_path);

        // Wrap the HTTP response body in a GzDecoder so we decompress on-the-fly
        // as bytes stream in, rather than buffering the entire archive in memory.
        let mut gz_decoder = GzDecoder::new(response);
        let mut out_file = File::create(dest_path)?;

        // Stream decompressed bytes from the decoder directly into the file.
        std::io::copy(&mut gz_decoder, &mut out_file)?;
        println!("Download and extraction complete!");
    }

    // Delegate to the raw binary parser to produce a Candle Tensor.
    load_mnist_images(dest_path, device)
}

// =============================================================================
// load_mnist_images — binary IDX3 parser → Candle Tensor
// =============================================================================
//
// The MNIST IDX3 format (magic number 0x0803 = 2051) has this layout:
//
//   Offset  | Bytes | Meaning
//   ------  | ----- | -------
//   0       | 4     | Magic number (big-endian u32): must be 2051
//   4       | 4     | Number of images (big-endian u32)
//   8       | 4     | Number of rows   (big-endian u32)  → 28
//   12      | 4     | Number of cols   (big-endian u32)  → 28
//   16      | N*R*C | Raw pixel bytes, one u8 per pixel, row-major
//
// WHY big-endian? The IDX format pre-dates modern hardware conventions.
// Rust's `u32::from_be_bytes` handles the byte-order conversion for us.
//
// WHY affine(1/127.5, -1.0)?
//   Pixel values are in [0, 255] as u8.
//   After casting to f32 and applying x * (1/127.5) - 1.0, they land in [-1, 1].
//   This symmetric normalisation is standard for generative models because it
//   centres the data around zero, which stabilises training and matches the
//   Gaussian prior used in the forward diffusion process.
/// Parses the MNIST IDX3-ubyte binary file and returns a Tensor of shape
/// `(num_images, 1, rows, cols)` with pixel values normalised to `[-1, 1]`.
fn load_mnist_images(path: &str, device: &Device) -> Result<Tensor> {
    let file = File::open(path)?;

    // BufReader wraps the file with an internal read buffer so small reads
    // (e.g. 4-byte header fields) don't each become a syscall.
    let mut reader = BufReader::new(file);

    // --- Step 1: validate the magic number -----------------------------------
    // The magic number acts as a file-type signature.  If it's wrong, we
    // probably opened the wrong file (e.g. the labels file) — fail loudly.
    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic)?;
    let magic_num = u32::from_be_bytes(magic);
    if magic_num != 2051 {
        bail!("Invalid magic number for MNIST images: {}", magic_num);
    }

    // --- Step 2: read the three dimension fields (num, rows, cols) -----------
    // All three are packed contiguously so we read them in one shot (12 bytes).
    let mut meta = [0u8; 12];
    reader.read_exact(&mut meta)?;

    let num_images = u32::from_be_bytes([meta[0], meta[1], meta[2], meta[3]]) as usize;
    let rows = u32::from_be_bytes([meta[4], meta[5], meta[6], meta[7]]) as usize;
    let cols = u32::from_be_bytes([meta[8], meta[9], meta[10], meta[11]]) as usize;
    println!("Loading {} images of size {}x{}...", num_images, rows, cols);

    // --- Step 3: read the raw pixel buffer -----------------------------------
    // Total bytes = images × rows × cols.  Pixels are u8 in row-major order.
    let mut buffer = vec![0u8; num_images * rows * cols];
    reader.read_exact(&mut buffer)?;

    // Build a Tensor, then cast and normalise:
    //   1. from_vec  → shape (N, 1, H, W) with u8 elements
    //   2. to_dtype  → cast to f32 so arithmetic is floating-point
    //   3. affine    → scale by 1/127.5 and shift by -1.0  →  range [-1, 1]
    // The channel dimension (1) is kept so the shape is compatible with
    // standard vision convention (N, C, H, W), even though we later flatten
    // for the MLP.
    let tensor = Tensor::from_vec(buffer, (num_images, 1, rows, cols), device)?
        .to_dtype(DType::F32)?
        .affine(1.0 / 127.5, -1.0)?;

    Ok(tensor)
}

// =============================================================================
// main — training loop and reverse-diffusion sampling
// =============================================================================
//
// High-level flow:
//   1. Load MNIST images.
//   2. Build the beta schedule, the MLP, and the Adam optimizer.
//   3. Run the forward-diffusion training loop (epochs × mini-batches).
//   4. After training, run the reverse-diffusion sampling loop to generate an image.
//   5. Save PNGs: original reference, initial noise, and generated output.
// =============================================================================

// Entry point: orchestrates training & sampling for the DDPM pipeline.
fn main() -> Result<()> {
    // --- Device selection ----------------------------------------------------
    // `Device::Cpu` runs all tensor ops on the CPU.
    // Swap to `Device::Cuda(0)` to use the first NVIDIA GPU (requires the
    // `cuda` Candle feature flag and a CUDA-capable GPU).
    let device = Device::Cpu;
    println!("==DDPM MNIST Model Training");

    // --- Dataset loading -----------------------------------------------------
    // `acquire_mnit` handles download + caching automatically.
    // `dataset` has shape (60000, 1, 28, 28) with pixel values in [-1, 1].
    let dataset = acquire_mnit(&device)?;

    // `dim(0)` returns the size of the first dimension = number of images.
    // We use this later to generate random indices into the dataset.
    let total_samples = dataset.dim(0)?;
    println!("Total samples: {}", total_samples);

    // =========================================================================
    // Hyper-parameter configuration
    // =========================================================================
    //
    // steps (T)        — number of diffusion timesteps.
    //                    More steps = smoother denoising, but slower inference.
    //                    100 is enough for MNIST; typical DDPM uses 1000.
    //
    // time_emb_dim     — size of the sinusoidal time embedding vector.
    //                    Larger = more expressive conditioning signal.
    //                    16 is lightweight; production models use 128–512.
    //
    // img_dim          — 28 × 28 = 784 pixels, flattened into a 1-D vector.
    //                    The MLP operates on flat vectors, not 2-D images.
    //
    // hidden_dim       — number of neurons in the single hidden layer.
    //                    512 gives enough capacity for MNIST patterns.
    //
    // batch_size       — number of training examples per gradient update.
    //                    128 balances GPU memory, gradient variance, and speed.
    //
    // epochs           — total number of gradient update steps.
    //                    With batch_size=128 and 60 000 samples, each "epoch"
    //                    here is really one mini-batch step (not a full epoch
    //                    in the traditional sense).
    //
    // lr               — Adam learning rate.
    //                    0.001 is the standard Adam default (Kingma & Ba 2015).
    // =========================================================================
    let steps = 100; // T: total diffusion timesteps
    let time_emb_dim = 16; // dimensionality of sinusoidal time encoding
    let img_dim = 784; // 28×28 flattened image size
    let hidden_dim = 512; // hidden layer width of the denoising MLP
    let batch_size = 128; // mini-batch size per gradient step
    let epochs = 20000; // total number of gradient update steps
    let lr = 0.001; // Adam learning rate (α)

    // --- Build the beta schedule ---------------------------------------------
    // `BetaScheduler` pre-computes a *linear* beta schedule from β_1=0.0001
    // to β_T=0.02 (as in the original DDPM paper).
    //
    // WHY pre-compute?  α_t, ᾱ_t, and σ_t are deterministic functions of β_t
    // and never change during training, so computing them once is efficient.
    //
    // Key quantities the scheduler stores:
    //   betas         β_t  — noise variance added at each step
    //   alphas        α_t  = 1 - β_t
    //   alphas_cumprod ᾱ_t = ∏_{s=1}^{t} α_s  (cumulative product)
    //   sigmas        σ_t  = sqrt(β_t)   used in the reverse process
    let scheduler = BetaScheduler::new(steps, 0.0001, 0.02, &device)?;

    // --- Build the denoising MLP ---------------------------------------------
    // Architecture: Linear(784+16, 512) → ReLU → Linear(512, 784)
    //
    // WHY 784+16 as input?
    //   We concatenate the flattened noisy image (784 dims) with the time
    //   embedding (16 dims) so the model knows *when* (i.e., how noisy) the
    //   input is and can adjust its prediction accordingly.
    //
    // WHY 784 as output?
    //   The model predicts the noise vector ε that was added to x_0.
    //   ε has the same shape as the image, so the output dimension = img_dim.
    let mut mlp = SimpleDenoisingMlp::new(img_dim + time_emb_dim, hidden_dim, img_dim, &device)?;

    // --- Build the Adam optimizer --------------------------------------------
    // Adam maintains per-parameter first and second moment estimates (m, v).
    // This allows adaptive learning rates and is more stable than plain SGD
    // for diffusion models.
    let mut optimizer = MlpAdamOptimizer::new(&mlp, lr)?;

    // Log model configuration so we can sanity-check before a long training run.
    println!("Scheduler: Linear, {} steps", steps);
    println!(
        "MLP: input_dim={} (784+16), hidden_dim={}, output_dim=784",
        img_dim + time_emb_dim,
        hidden_dim
    );
    println!("Starting training for {} epochs...\n", epochs);

    // =========================================================================
    // Training loop
    // =========================================================================
    //
    // Each iteration corresponds to one mini-batch gradient update.
    // The DDPM training objective (simplified) is:
    //
    //   L = E_{t, x_0, ε} [ || ε - ε_θ(x_t, t) ||² ]
    //
    // In plain English: the model ε_θ takes a noisy image x_t and a timestep t
    // and must predict the Gaussian noise ε that was added to the clean x_0.
    // Minimising this MSE loss is equivalent to maximising the ELBO of the
    // variational lower bound on log p(x_0).
    // =========================================================================
    for epoch in 1..=epochs {
        // ---------------------------------------------------------------------
        // Step 1 — Sample a random mini-batch of clean images (x_0)
        // ---------------------------------------------------------------------
        // WHY random sampling instead of sequential batches?
        //   Stochastic gradient descent requires i.i.d. samples.  Random
        //   sampling ensures each batch covers a variety of digit classes and
        //   avoids gradient bias from correlated sequential samples.
        //
        // We generate uniform floats in [0, total_samples) and truncate to u32
        // indices.  Subtracting 1e-4 prevents the float from rounding up to
        // `total_samples` (which would be out of bounds).
        let index_tensor = Tensor::rand(
            0.0f32,
            total_samples as f32 - 1e-4f32,
            (batch_size,),
            &device,
        )?
        .to_dtype(DType::U32)?;

        // `index_select` gathers rows at the sampled indices → shape (B, 1, 28, 28).
        // `reshape` flattens each image to a 1-D vector → shape (B, 784).
        // This flat representation is required by the MLP.
        let x0 = dataset
            .index_select(&index_tensor, 0)?
            .reshape((batch_size, img_dim))?;

        // ---------------------------------------------------------------------
        // Step 2 — Sample a random timestep t ∈ {0, …, T-1} for each example
        // ---------------------------------------------------------------------
        // WHY a different t per example?
        //   The model must learn to denoise at *all* noise levels simultaneously.
        //   Randomising t across the batch ensures even coverage of the schedule
        //   without needing to iterate through every t on every mini-batch.
        //
        // WHY float first, then cast to U32?
        //   `Tensor::rand` generates continuous uniform floats.  Casting to U32
        //   truncates (floor) to integer timestep indices.
        let t_float = Tensor::rand(0.0f32, steps as f32 - 1e-4f32, (batch_size,), &device)?;
        let t = t_float.to_dtype(DType::U32)?;

        // ---------------------------------------------------------------------
        // Step 3 — Draw Gaussian noise ε and create the noisy image x_t
        // ---------------------------------------------------------------------
        // WHY standard normal noise (mean=0, std=1)?
        //   The DDPM forward process is defined as:
        //     x_t = sqrt(ᾱ_t) * x_0 + sqrt(1 - ᾱ_t) * ε,  ε ~ N(0, I)
        //
        //   This is a *closed-form* one-shot corruption that jumps from x_0
        //   directly to any noise level t, avoiding the need to iterate t steps.
        //   `scheduler.add_noise` applies this formula using the pre-computed ᾱ_t.
        //
        // The target label for the model is precisely this noise vector ε.
        let noise = Tensor::randn(0.0f32, 1.0f32, (batch_size, img_dim), &device)?;
        let xt = scheduler.add_noise(&x0, &noise, &t)?;

        // ---------------------------------------------------------------------
        // Step 4 — Compute sinusoidal time embedding and build the model input
        // ---------------------------------------------------------------------
        // WHY sinusoidal embeddings?
        //   The model needs to know the noise level (timestep t) to calibrate
        //   its prediction.  Sinusoidal embeddings (like in Transformers) map
        //   each integer t to a continuous vector where nearby timesteps have
        //   similar representations, giving the model a smooth sense of "time".
        //
        // WHY concatenation instead of addition?
        //   For a simple two-layer MLP, concatenation is the easiest way to
        //   fuse two modalities.  Attention-based U-Nets use cross-attention or
        //   additive conditioning instead.
        //
        // v shape: (batch_size, img_dim + time_emb_dim) = (128, 800)
        let time_emb = get_time_embedding(&t, time_emb_dim)?;
        let v = Tensor::cat(&[&xt, &time_emb], 1)?;

        // ---------------------------------------------------------------------
        // Step 5 — Forward pass: predict the noise ε̂ = ε_θ(x_t, t)
        // ---------------------------------------------------------------------
        // The MLP returns:
        //   pred — the predicted noise vector ε̂  (shape: batch_size × img_dim)
        //   a1   — post-activation of the hidden layer (needed for backward pass)
        //   z1   — pre-activation of the hidden layer (needed for backward pass)
        //
        // WHY return intermediate activations?
        //   Our manual backprop implementation needs the cached activations to
        //   compute gradients efficiently (chain rule).  Autograd frameworks
        //   (e.g., PyTorch) do this automatically in their computation graph.
        let (pred, a1, z1) = mlp.forward(&v)?;

        // ---------------------------------------------------------------------
        // Step 6 — Compute MSE loss between predicted and true noise
        // ---------------------------------------------------------------------
        // Loss = (1/N) Σ (ε̂ - ε)²   (Mean Squared Error)
        //
        // WHY MSE?
        //   Ho et al. showed that optimising this simple objective is equivalent
        //   to a re-weighted ELBO of the diffusion model likelihood.  MSE is
        //   differentiable everywhere and numerically stable.
        //
        // Note: we compute the scalar loss only for *logging*; the actual
        // gradient is computed by `mlp.backward` in the next step.
        let diff = pred.sub(&noise)?;
        let loss = diff.sqr()?.mean_all()?.to_scalar::<f32>()?;

        // ---------------------------------------------------------------------
        // Step 7 — Back-propagation: compute gradients ∂L/∂W for all weights
        // ---------------------------------------------------------------------
        // WHY manual backward?
        //   Candle does not yet have full autograd support for all ops in this
        //   custom MLP.  We hand-rolled the backward pass using the chain rule.
        //
        // `grads` contains:
        //   dw1  — gradient w.r.t. first layer weights  (input→hidden)
        //   db1  — gradient w.r.t. first layer biases
        //   dw2  — gradient w.r.t. second layer weights (hidden→output)
        //   db2  — gradient w.r.t. second layer biases
        let grads = mlp.backward(&v, &a1, &z1, &pred, &noise)?;

        // ---------------------------------------------------------------------
        // Step 8 — Adam optimizer step: update weights using the gradients
        // ---------------------------------------------------------------------
        // Adam computes adaptive per-parameter learning rates:
        //   m_t = β1 * m_{t-1} + (1-β1) * g_t          (biased 1st moment)
        //   v_t = β2 * v_{t-1} + (1-β2) * g_t²         (biased 2nd moment)
        //   m̂_t = m_t / (1 - β1^t)                      (bias-corrected)
        //   v̂_t = v_t / (1 - β2^t)
        //   θ_t = θ_{t-1} - lr * m̂_t / (sqrt(v̂_t) + ε)
        //
        // This adapts the step size for each parameter individually and handles
        // sparse or noisy gradients better than plain gradient descent.
        optimizer.step(&mut mlp, &grads)?;

        // ---------------------------------------------------------------------
        // Step 9 — Periodic logging: print loss and gradient norms
        // ---------------------------------------------------------------------
        // WHY gradient norms?
        //   The L2 norm of the gradient (||∇W||₂) indicates how strongly the
        //   weights are being updated.  Very small norms → vanishing gradients
        //   (training stalled).  Very large norms → exploding gradients (unstable).
        //   Watching both the loss and the norms together helps diagnose training
        //   health without running a full validation loop.
        //
        // WHY log only every 100 epochs?
        //   Printing to stdout every step would dominate wall-clock time and
        //   flood the terminal.  Every 100 steps gives a smooth curve without
        //   overhead.
        if epoch % 100 == 0 || epoch == 1 {
            let dw1_norm = grads.dw1.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
            let dw2_norm = grads.dw2.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
            println!(
                "Epoch {:4}/{} - MSE Loss: {:.6} | dw1 norm: {:.4}, dw2 norm: {:.4}",
                epoch, epochs, loss, dw1_norm, dw2_norm
            );
        }
    }

    // =========================================================================
    // Reverse diffusion sampling (image generation)
    // =========================================================================
    //
    // Now that the model is trained we run the *reverse* process to generate
    // a new image from pure Gaussian noise.
    //
    // The reverse process is (from the DDPM paper):
    //
    //   x_{t-1} = (1/sqrt(α_t)) * (x_t - β_t/sqrt(1-ᾱ_t) * ε_θ(x_t, t))
    //             + σ_t * z        where z ~ N(0, I)  if t > 0
    //                                    z = 0         if t = 0
    //
    // Starting from x_T ~ N(0, I) we iterate t = T-1 down to t = 0 to recover
    // a sample x_0 from the learned data distribution.
    // =========================================================================
    println!("\n=== Starting Reverse Diffusion Sampling (Generation) ===");

    // Save image #0 from the training set as a visual reference baseline.
    // This lets us compare the generated image quality against a real digit.
    let original_sample = dataset
        .index_select(&Tensor::new(&[0u32], &device)?, 0)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    save_png("mnist_original.png", &original_sample)?;

    // Generate one image by running the reusable DDPM reverse sampler.
    let num_samples = 1; // generate one image at a time
    let initial_noise = Tensor::randn(0.0f32, 1.0f32, (num_samples, img_dim), &device)?;
    save_png(
        "mnist_noisy.png",
        &initial_noise.flatten_all()?.to_vec1::<f32>()?,
    )?;

    let xt = sample_ddpm_from_noise(
        &mlp,
        &scheduler,
        initial_noise,
        img_dim,
        time_emb_dim,
        &device,
    )?;

    // After the loop, `xt` is x_0 — our generated image.
    // Flatten to a 1-D slice and save as PNG.
    let final_pixels = xt.flatten_all()?.to_vec1::<f32>()?;
    save_png("mnist_generated.png", &final_pixels)?;

    Ok(())
}

// =============================================================================
// save_png — convert model output tensor to a grayscale PNG file
// =============================================================================
//
// Input pixel values come from the model in the range [-1, 1] (because we
// trained with normalised data in that range).
//
// PNG requires u8 values in [0, 255], so we apply:
//   pixel_u8 = round( ((val + 1.0) / 2.0).clamp(0, 1) * 255 )
//
// The clamp guards against out-of-range predictions (the model is not
// constrained to output exactly [-1, 1]).
//
/// Writes a 28×28 grayscale image to disk as a PNG file.
///
/// # Arguments
/// * `path`       — destination file path (e.g. "mnist_generated.png")
/// * `image_flat` — 784 pixel values in the range **[-1, 1]**
fn save_png(path: &str, image_flat: &[f32]) -> Result<()> {
    use std::io::BufWriter;

    // Create or truncate the destination file.
    let file = File::create(path)?;

    // BufWriter accumulates small writes into larger OS-level writes,
    // which is important for the PNG encoder that writes in small chunks.
    let ref mut w = BufWriter::new(file);

    // Configure the PNG encoder: 28×28 pixels, single channel (grayscale), 8-bit.
    let mut encoder = png::Encoder::new(w, 28, 28);
    encoder.set_color(png::ColorType::Grayscale);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header()?;

    // Map each f32 pixel in [-1, 1] to a u8 in [0, 255].
    // The `clamp` call prevents artefacts if the model output slightly exceeds
    // the training data range (which is common for diffusion model outputs).
    let mut data = vec![0u8; 784];
    for (i, &val) in image_flat.iter().enumerate() {
        let norm = ((val + 1.0) / 2.0).clamp(0.0, 1.0); // map [-1,1] → [0,1]
        data[i] = (norm * 255.0).round() as u8; // scale to [0, 255]
    }

    // Write the raw pixel bytes; the encoder handles PNG chunk framing and CRC.
    writer.write_image_data(&data)?;
    println!("Saved generated image as PNG to: {}", path);
    Ok(())
}
