// =============================================================================
// resample_diffusion_unet.rs — inspect a saved U-Net checkpoint without training
// =============================================================================
//
// This binary exists so sampling bugs can be fixed and re-tested independently
// of the expensive training loop. It restores both the model parameters and the
// original fixed noise tensor, then writes every reverse-diffusion frame. Using
// the same checkpoint and starting noise makes before/after comparisons useful;
// DDPM's per-step Gaussian noise can still make separate runs differ.

use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use candle_core::{safetensors, Device};

use llm_scratch_rs::models::diffusion::sampling::sample_ddpm_cfg_from_timestep_with_callback;
use llm_scratch_rs::models::diffusion::{
    load_model_checkpoint, one_hot_class, BetaScheduler, SimpleDenoisingUNet,
};
use llm_scratch_rs::utils::mnist_utils::save_png;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

const IMAGE_SIZE: usize = 784;
const TIME_EMBEDDING_SIZE: usize = 16;
const CLASS_COUNT: usize = 10;
const SAMPLE_CLASS: usize = 3;
const GUIDANCE_SCALE: f64 = 1.0;
const DEFAULT_START_TIMESTEP: usize = 99;

// Match the trainer's device preference so a checkpoint can be inspected on the
// same CUDA setup, while keeping CPU-only machines usable.
fn active_device() -> Device {
    if candle_core::utils::cuda_is_available() {
        Device::new_cuda(1)
            .or_else(|_| Device::new_cuda(0))
            .unwrap_or(Device::Cpu)
    } else {
        Device::Cpu
    }
}

// The trainer stores fixed noise beside its model checkpoints. Resolving it
// relative to the requested checkpoint keeps custom checkpoint directories
// self-contained instead of silently reading the repository default.
fn load_fixed_noise(checkpoint: &Path, device: &Device) -> Result<candle_core::Tensor> {
    let directory = checkpoint
        .parent()
        .ok_or_else(|| anyhow!("checkpoint path has no parent directory"))?;
    let noise_path = directory.join("fixed_noise.safetensors");
    let tensors = safetensors::load(&noise_path, device)
        .with_context(|| format!("failed to load {}", noise_path.display()))?;
    tensors
        .get("fixed_noise")
        .cloned()
        .ok_or_else(|| anyhow!("{} does not contain fixed_noise", noise_path.display()))
}

// Usage:
//   resample_diffusion_unet [checkpoint] [start_timestep]
//
// Both arguments are optional to keep the common epoch-8000, full-chain check
// convenient. Parsing errors are surfaced rather than replaced with defaults.
fn parse_arguments() -> Result<(PathBuf, usize)> {
    let mut arguments = std::env::args().skip(1);
    let checkpoint = arguments
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("unet_checkpoints/epoch_8000.safetensors"));
    let start_timestep = arguments
        .next()
        .map(|value| value.parse::<usize>())
        .transpose()
        .context("start timestep must be a non-negative integer")?
        .unwrap_or(DEFAULT_START_TIMESTEP);
    Ok((checkpoint, start_timestep))
}

fn main() -> Result<()> {
    let (checkpoint, start_timestep) = parse_arguments()?;
    let device = active_device();
    println!("Active device: {device:?}");
    println!("Loading checkpoint: {}", checkpoint.display());

    // Recreate the exact architecture before assigning tensors from disk. The
    // checkpoint loader validates parameter names and shapes, so architecture
    // drift fails here instead of producing a misleading image.
    let model = SimpleDenoisingUNet::new(IMAGE_SIZE, TIME_EMBEDDING_SIZE + CLASS_COUNT, &device)?;
    load_model_checkpoint(&model, &checkpoint, &device)?;
    let initial_noise = load_fixed_noise(&checkpoint, &device)?;
    let class = one_hot_class(SAMPLE_CLASS, CLASS_COUNT, &device)?;
    let scheduler = BetaScheduler::new_cosine(100, &device)?;

    // Include the start timestep in the directory name so shortened diagnostic
    // chains cannot overwrite the full t=99 result.
    let output_dir = PathBuf::from(format!("unet_resampled_t{start_timestep:03}"));
    std::fs::create_dir_all(&output_dir)?;
    let generated = sample_ddpm_cfg_from_timestep_with_callback(
        &model,
        &scheduler,
        initial_noise,
        start_timestep,
        IMAGE_SIZE,
        TIME_EMBEDDING_SIZE,
        &class,
        GUIDANCE_SCALE,
        &device,
        |frame_index, sample| {
            // Log the raw model domain before save_png clamps [-1, 1] to bytes.
            // This exposes numerical explosions that may look merely black or
            // white after image encoding.
            let pixels = sample.flatten_all()?.to_vec1::<f32>()?;
            let minimum = pixels.iter().copied().fold(f32::INFINITY, f32::min);
            let maximum = pixels.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            if frame_index % 10 == 0 || frame_index == start_timestep {
                println!("Frame {frame_index:03} | raw range [{minimum:.4}, {maximum:.4}]");
            }
            // Save every step for visual inspection, but print only every tenth
            // range to keep a normal run readable.
            save_png(
                &output_dir
                    .join(format!("frame_{frame_index:03}.png"))
                    .to_string_lossy(),
                &pixels,
            )
        },
    )?;

    let final_path = output_dir.join("sample_class3_scale1.png");
    save_png(
        &final_path.to_string_lossy(),
        &generated.flatten_all()?.to_vec1::<f32>()?,
    )?;
    println!("Saved corrected sample to {}", final_path.display());
    Ok(())
}
