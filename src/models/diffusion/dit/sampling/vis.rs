//! # Image Visualization, Stitching, and Dataset Metadata
//!
//! ## Dynamic Range Remapping: $[-1.0, 1.0] \to [0, 255]$
//! Diffusion models predict and generate images whose pixel activations follow a standard
//! distribution scaled into $[-1.0, 1.0]$. To render valid 8-bit grayscale PNGs, values are
//! linearly remapped, clamped, and quantized:
//!
//! $$\text{pixel}_{(u, v)} = \text{round}\left( \text{clamp}\left( \frac{x_{(u, v)} + 1.0}{2.0},\, 0.0,\, 1.0 \right) \times 255.0 \right)$$
//!
//! ## Spatial Grid Composition
//! - **Lookbook ($2 \times 5$ Grid)**: Stitches 10 class samples into an organized 56x140 image.
//! - **Filmstrip ($1 \times K$ Horizon)**: Aligns sequential generation steps (CFG sweep or morphing) into a 28x(28*K) panorama.

use std::fs::File;
use std::io::BufWriter;

/// Canonical class names for the 10 categories in Fashion-MNIST.
pub const FASHION_CLASSES: [&str; 10] = [
    "t_shirt",
    "trouser",
    "pullover",
    "dress",
    "coat",
    "sandal",
    "shirt",
    "sneaker",
    "bag",
    "ankle_boot",
];

/// Encodes an arbitrary 2D grayscale float slice into an 8-bit PNG file.
pub fn save_image_grid(
    path: &str,
    image_flat: &[f32],
    width: u32,
    height: u32,
) -> anyhow::Result<()> {
    crate::utils::ensure_parent_dir(path)?;
    let file = File::create(path)?;
    let writer = BufWriter::new(file);
    let mut encoder = png::Encoder::new(writer, width, height);
    encoder.set_color(png::ColorType::Grayscale);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header()?;

    let bytes: Vec<u8> = image_flat
        .iter()
        .map(|&p| {
            let norm = ((p + 1.0) / 2.0).clamp(0.0, 1.0);
            (norm * 255.0).round() as u8
        })
        .collect();

    writer.write_image_data(&bytes)?;
    Ok(())
}

/// Stitches a 2x5 grid of 10 fashion category images into a single lookbook collage.
///
/// Output resolution: $56 \times 140$ pixels.
pub fn save_lookbook_collage(path: &str, class_images: &[Vec<f32>]) -> anyhow::Result<()> {
    let width = 5 * 28; // 140 px
    let height = 2 * 28; // 56 px
    let mut collage = vec![0.0f32; width * height];

    for (c, img) in class_images.iter().enumerate() {
        let row = c / 5;
        let col = c % 5;
        for y in 0..28 {
            for x in 0..28 {
                let dst_y = row * 28 + y;
                let dst_x = col * 28 + x;
                collage[dst_y * width + dst_x] = img[y * 28 + x];
            }
        }
    }

    save_image_grid(path, &collage, width as u32, height as u32)
}

/// Stitches a horizontal sequence of images into a filmstrip (for CFG sweeps or morphing transitions).
///
/// Output resolution: $28 \times (28 \cdot K)$ pixels, where $K$ is the number of frames.
pub fn save_filmstrip(path: &str, frames: &[Vec<f32>]) -> anyhow::Result<()> {
    let k = frames.len();
    let width = k * 28;
    let height = 28;
    let mut filmstrip = vec![0.0f32; width * height];

    for (i, frame) in frames.iter().enumerate() {
        let col_offset = i * 28;
        for y in 0..28 {
            for x in 0..28 {
                filmstrip[y * width + col_offset + x] = frame[y * 28 + x];
            }
        }
    }

    save_image_grid(path, &filmstrip, width as u32, height as u32)
}
