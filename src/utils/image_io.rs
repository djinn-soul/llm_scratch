// =============================================================================
// image_io.rs — validated conversion from model tensors to MNIST PNG files
// =============================================================================

use anyhow::{bail, Result};
use std::fs::File;
use std::io::BufWriter;

const MNIST_SIDE: u32 = 28;
const MNIST_PIXELS: usize = (MNIST_SIDE * MNIST_SIDE) as usize;

/// Write one normalized MNIST image as an 8-bit grayscale PNG.
///
/// Diffusion inputs and outputs use `[-1, 1]`, whereas PNG stores `[0, 255]`.
/// Values slightly outside the model domain are clamped because overshoot is a
/// normal numerical effect. NaN and infinity are rejected instead: casting them
/// would hide a broken sampling trajectory inside an apparently valid file.
pub fn save_png(path: &str, image_flat: &[f32]) -> Result<()> {
    // The encoder accepts any byte count, so validate shape before a truncated
    // or oversized buffer becomes a corrupt/misleading diagnostic image.
    if image_flat.len() != MNIST_PIXELS {
        bail!(
            "expected {} pixels for a {}x{} MNIST image, got {}",
            MNIST_PIXELS,
            MNIST_SIDE,
            MNIST_SIDE,
            image_flat.len()
        );
    }
    if let Some((index, value)) = image_flat
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        bail!("cannot save non-finite pixel at index {index}: {value}");
    }

    crate::utils::ensure_parent_dir(path)?;
    let file = File::create(path)?;
    let writer = BufWriter::new(file);
    let mut encoder = png::Encoder::new(writer, MNIST_SIDE, MNIST_SIDE);
    encoder.set_color(png::ColorType::Grayscale);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header()?;

    // (x + 1) / 2 maps [-1, 1] to [0, 1]; scaling then maps to one grayscale
    // byte. Clamp before the cast so overshoot cannot wrap or saturate oddly.
    let data: Vec<u8> = image_flat
        .iter()
        .map(|value| (((value + 1.0) / 2.0).clamp(0.0, 1.0) * 255.0).round() as u8)
        .collect();
    writer.write_image_data(&data)?;
    println!("Saved image to: {path}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_non_finite_pixels() {
        // Validation must happen before File::create, hence this test can use a
        // sentinel path without leaving a file behind.
        let mut pixels = vec![0.0f32; MNIST_PIXELS];
        pixels[37] = f32::NAN;
        let error = save_png("unused-non-finite.png", &pixels).unwrap_err();
        assert!(error.to_string().contains("non-finite pixel at index 37"));
    }

    #[test]
    fn rejects_wrong_pixel_count() {
        let error = save_png("unused-wrong-size.png", &[0.0]).unwrap_err();
        assert!(error.to_string().contains("expected 784 pixels"));
    }
}
