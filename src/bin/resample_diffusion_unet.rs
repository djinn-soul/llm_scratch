// =============================================================================
// resample_diffusion_unet.rs — Plug-and-Play Reverse Sampling for Saved U-Net
// =============================================================================
//
// Supports modular, plug-and-play reverse samplers with zero retraining:
//   1. DPM-Solver++ (2M) — 2nd-order exponential ODE integrator (8–10 steps)
//   2. DDIM               — 1st-order deterministic ODE sampler (20–50 steps)
//   3. DDPM               — 1st-order stochastic Markovian SDE sampler (100 steps)
//   4. ALL                — Comparative benchmark running all samplers side-by-side
//
// Usage:
//   cargo run --release --bin resample_diffusion_unet
//   cargo run --release --bin resample_diffusion_unet -- --sampler dpm --steps 8
//   cargo run --release --bin resample_diffusion_unet -- --sampler ddim --steps 20
//   cargo run --release --bin resample_diffusion_unet -- --sampler ddpm --steps 100
//   cargo run --release --bin resample_diffusion_unet -- --sampler all

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use candle_core::{Device, Tensor};

use llm_scratch_rs::models::diffusion::sampling::{sample_diffusion_cfg, SamplerKind};
use llm_scratch_rs::models::diffusion::{
    load_model_checkpoint, one_hot_class, BetaScheduler, SimpleDenoisingUNet,
};
use llm_scratch_rs::utils::mnist_utils::save_png;

// Use mimalloc for high-throughput tensor allocations during inference.
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

// ── Model and Sampling Hyperparameters ────────────────────────────────────────
const IMAGE_SIZE: usize = 784; // 28x28 flattened MNIST image area (pixels)
const TIME_EMBEDDING_SIZE: usize = 16; // Sinusoidal timestep embedding vector dimension
const CLASS_COUNT: usize = 10; // Total MNIST digit classes (0..9)
const DEFAULT_SAMPLE_CLASS: usize = 3; // Target digit class to generate (0..9)
const DEFAULT_GUIDANCE_SCALE: f64 = 2.5; // CFG scale: s=2.5 for sharp, high-contrast strokes
const DEFAULT_START_TIMESTEP: usize = 99; // Starting noise level (T-1 = 99 for 100-step schedule)

/// Selects the execution device: prefers CUDA GPU if available, otherwise falls back to CPU.
fn active_device() -> Device {
    if candle_core::utils::cuda_is_available() {
        Device::new_cuda(1)
            .or_else(|_| Device::new_cuda(0))
            .unwrap_or(Device::Cpu)
    } else {
        Device::Cpu
    }
}

/// Assembles multiple 28×28 grayscale images into a single grid PNG.
fn save_grid_png(path: &str, images_flat: &[f32], rows: usize, cols: usize) -> Result<()> {
    use std::fs::File;
    use std::io::BufWriter;

    let file = File::create(path)?;
    let ref mut w = BufWriter::new(file);

    let img_h = 28;
    let img_w = 28;
    let grid_h = rows * img_h;
    let grid_w = cols * img_w;

    let mut encoder = png::Encoder::new(w, grid_w as u32, grid_h as u32);
    encoder.set_color(png::ColorType::Grayscale);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header()?;

    let mut data = vec![0u8; grid_h * grid_w];

    for r in 0..rows {
        for c in 0..cols {
            let img_idx = r * cols + c;
            let img_offset = img_idx * img_h * img_w;

            for y in 0..img_h {
                for x in 0..img_w {
                    let val = images_flat[img_offset + y * img_w + x];
                    let norm = ((val + 1.0) / 2.0).clamp(0.0, 1.0);
                    let pixel_val = (norm * 255.0).round() as u8;

                    let grid_y = r * img_h + y;
                    let grid_x = c * img_w + x;
                    data[grid_y * grid_w + grid_x] = pixel_val;
                }
            }
        }
    }

    writer.write_image_data(&data)?;
    println!("Saved grid image to: {}", path);
    Ok(())
}

#[derive(Debug, Clone)]
enum Mode {
    Single(SamplerKind, usize),
    BenchmarkAll,
}

#[derive(Debug, Clone)]
struct Config {
    checkpoint: PathBuf,
    start_timestep: usize,
    sample_class: usize,
    guidance_scale: f64,
    mode: Mode,
}

fn parse_cli() -> Result<Config> {
    let mut args = std::env::args().skip(1);
    let mut checkpoint: Option<PathBuf> = None;
    let mut start_timestep: Option<usize> = None;
    let mut sampler_str: Option<String> = None;
    let mut steps_opt: Option<usize> = None;
    let mut sample_class = DEFAULT_SAMPLE_CLASS;
    let mut guidance_scale = DEFAULT_GUIDANCE_SCALE;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--sampler" | "-s" => {
                sampler_str = Some(args.next().context("missing value for --sampler")?);
            }
            "--steps" | "-n" => {
                steps_opt = Some(
                    args.next()
                        .context("missing value for --steps")?
                        .parse::<usize>()
                        .context("invalid integer for --steps")?,
                );
            }
            "--guidance" | "-g" => {
                guidance_scale = args
                    .next()
                    .context("missing value for --guidance")?
                    .parse::<f64>()
                    .context("invalid float for --guidance")?;
            }
            "--class" | "-c" => {
                sample_class = args
                    .next()
                    .context("missing value for --class")?
                    .parse::<usize>()
                    .context("invalid integer for --class")?;
            }
            "--start" | "-t" => {
                start_timestep = Some(
                    args.next()
                        .context("missing value for --start")?
                        .parse::<usize>()
                        .context("invalid integer for --start")?,
                );
            }
            "--checkpoint" | "-k" => {
                checkpoint = Some(PathBuf::from(
                    args.next().context("missing value for --checkpoint")?,
                ));
            }
            "--help" | "-h" => {
                println!(
                    "Usage: cargo run --release --bin resample_diffusion_unet [OPTIONS]\n\n\
                     Options:\n  \
                       --sampler <dpm|ddim|ddpm|all>  Sampler algorithm (default: dpm)\n  \
                       --steps <N>                   Inference steps (default: 8 for dpm, 20 for ddim, 100 for ddpm)\n  \
                       --guidance <float>            Classifier-free guidance scale (default: 2.5)\n  \
                       --class <0..9>                Digit class to preview (default: 3)\n  \
                       --start <0..99>               Starting timestep (default: 99)\n  \
                       --checkpoint <path>           Model checkpoint file\n  \
                       --help                        Print help information"
                );
                std::process::exit(0);
            }
            val if !val.starts_with('-') => {
                if checkpoint.is_none() {
                    checkpoint = Some(PathBuf::from(val));
                } else if start_timestep.is_none() {
                    start_timestep = Some(val.parse::<usize>().context("invalid start timestep")?);
                }
            }
            unknown => bail!("unknown option '{}'. Run with --help for usage.", unknown),
        }
    }

    let checkpoint = checkpoint.unwrap_or_else(|| {
        let mut latest_epoch = 0;
        let mut latest_path = PathBuf::from("unet_checkpoints/ema_epoch_25000.safetensors");
        if let Ok(entries) = std::fs::read_dir("unet_checkpoints") {
            for entry in entries.flatten() {
                let path = entry.path();
                if let Some(filename) = path.file_name().and_then(|f| f.to_str()) {
                    if filename.starts_with("ema_epoch_") && filename.ends_with(".safetensors") {
                        let epoch_str = &filename["ema_epoch_".len()..filename.len() - ".safetensors".len()];
                        if let Ok(epoch) = epoch_str.parse::<usize>() {
                            if epoch >= latest_epoch {
                                latest_epoch = epoch;
                                latest_path = path.clone();
                            }
                        }
                    }
                }
            }
        }
        latest_path
    });
    let start_timestep = start_timestep.unwrap_or(DEFAULT_START_TIMESTEP);

    let mode = match sampler_str.as_deref() {
        Some("all") | Some("compare") | Some("benchmark") => Mode::BenchmarkAll,
        Some(s) => {
            let sampler: SamplerKind = s.parse()?;
            let steps = steps_opt.unwrap_or_else(|| sampler.default_inference_steps());
            Mode::Single(sampler, steps)
        }
        None => {
            let sampler = SamplerKind::DpmSolver2m;
            let steps = steps_opt.unwrap_or_else(|| sampler.default_inference_steps());
            Mode::Single(sampler, steps)
        }
    };

    Ok(Config {
        checkpoint,
        start_timestep,
        sample_class,
        guidance_scale,
        mode,
    })
}

fn run_single_sampler(
    sampler: SamplerKind,
    num_steps: usize,
    config: &Config,
    model: &SimpleDenoisingUNet,
    scheduler: &BetaScheduler,
    device: &Device,
) -> Result<Vec<f32>> {
    let output_dir = PathBuf::from(format!(
        "unet_resampled_{}_t{:03}",
        match sampler {
            SamplerKind::DpmSolver2m => "dpm",
            SamplerKind::Ddim => "ddim",
            SamplerKind::Ddpm => "ddpm",
        },
        config.start_timestep
    ));
    std::fs::create_dir_all(&output_dir)?;

    println!(
        "\n=== {} Sampling ({} steps, Class = {}, Guidance = {:.1}) ===",
        sampler, num_steps, config.sample_class, config.guidance_scale
    );

    let initial_noise = Tensor::randn(0.0f32, 1.0f32, (1, IMAGE_SIZE), device)?;
    let target_class_one_hot = one_hot_class(config.sample_class, CLASS_COUNT, device)?;

    let t0 = Instant::now();
    let generated = sample_diffusion_cfg(
        sampler,
        model,
        scheduler,
        initial_noise,
        config.start_timestep,
        num_steps,
        IMAGE_SIZE,
        TIME_EMBEDDING_SIZE,
        &target_class_one_hot,
        config.guidance_scale,
        device,
        |frame_index, sample| {
            let pixels = sample.flatten_all()?.to_vec1::<f32>()?;
            let minimum = pixels.iter().copied().fold(f32::INFINITY, f32::min);
            let maximum = pixels.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            println!("Frame {frame_index:02} | raw range [{minimum:.4}, {maximum:.4}]");
            save_png(
                &output_dir
                    .join(format!("frame_{frame_index:02}.png"))
                    .to_string_lossy(),
                &pixels,
            )
        },
    )?;
    let single_duration = t0.elapsed();

    let single_path = output_dir.join(format!(
        "sample_class{}_scale{:.1}.png",
        config.sample_class, config.guidance_scale
    ));
    save_png(
        &single_path.to_string_lossy(),
        &generated.flatten_all()?.to_vec1::<f32>()?,
    )?;
    println!(
        "Saved preview to: {} (took {:.2?})",
        single_path.display(),
        single_duration
    );

    println!("\nGenerating Full Digits (0 to 9) Grid...");
    let mut all_digits_flat: Vec<f32> = Vec::with_capacity(10 * IMAGE_SIZE);
    let t_grid = Instant::now();
    for digit in 0..10 {
        let digit_noise = Tensor::randn(0.0f32, 1.0f32, (1, IMAGE_SIZE), device)?;
        let digit_one_hot = one_hot_class(digit, CLASS_COUNT, device)?;
        let sample = sample_diffusion_cfg(
            sampler,
            model,
            scheduler,
            digit_noise,
            config.start_timestep,
            num_steps,
            IMAGE_SIZE,
            TIME_EMBEDDING_SIZE,
            &digit_one_hot,
            config.guidance_scale,
            device,
            |_, _| Ok(()),
        )?;
        let pixels = sample.flatten_all()?.to_vec1::<f32>()?;
        save_png(
            &output_dir
                .join(format!("digit_{}.png", digit))
                .to_string_lossy(),
            &pixels,
        )?;
        all_digits_flat.extend(pixels);
    }
    let grid_duration = t_grid.elapsed();

    let grid_filename = match sampler {
        SamplerKind::DpmSolver2m => "all_digits_0_to_9_dpm2m.png",
        SamplerKind::Ddim => "all_digits_0_to_9_ddim.png",
        SamplerKind::Ddpm => "all_digits_0_to_9_ddpm.png",
    };
    let grid_path = output_dir.join(grid_filename);
    save_grid_png(&grid_path.to_string_lossy(), &all_digits_flat, 2, 5)?;
    println!(
        "Grid generation finished in {:.2?}. Outputs in '{}'",
        grid_duration,
        output_dir.display()
    );

    Ok(all_digits_flat)
}

fn main() -> Result<()> {
    let config = parse_cli()?;
    let device = active_device();

    println!("============================================================");
    println!("  Diffusion U-Net Plug-and-Play Resampler");
    println!("============================================================");
    println!("Active device: {device:?}");
    println!("Loading checkpoint: {}", config.checkpoint.display());

    let cond_dim = TIME_EMBEDDING_SIZE + CLASS_COUNT;
    let model = SimpleDenoisingUNet::new(IMAGE_SIZE, cond_dim, &device)?;
    load_model_checkpoint(&model, &config.checkpoint, &device)?;
    let scheduler = BetaScheduler::new_cosine(100, &device)?;

    match config.mode {
        Mode::Single(sampler, steps) => {
            run_single_sampler(sampler, steps, &config, &model, &scheduler, &device)?;
        }
        Mode::BenchmarkAll => {
            println!("\n=== Running Comparative Sampler Benchmark ===");
            let output_dir = PathBuf::from("unet_resampled_comparison");
            std::fs::create_dir_all(&output_dir)?;

            let benchmarks = [
                (SamplerKind::Ddpm, 100, "DDPM (100 steps)"),
                (SamplerKind::Ddim, 20, "DDIM (20 steps)"),
                (SamplerKind::DpmSolver2m, 8, "DPM-Solver++ 2M (8 steps)"),
            ];

            let mut all_samplers_flat: Vec<f32> = Vec::with_capacity(3 * 10 * IMAGE_SIZE);

            for (sampler, steps, label) in benchmarks {
                println!("\n--- Running {} ---", label);
                let digits =
                    run_single_sampler(sampler, steps, &config, &model, &scheduler, &device)?;
                all_samplers_flat.extend(digits);
            }

            // Assemble a 3-row comparison grid (Row 1: DDPM, Row 2: DDIM, Row 3: DPM-Solver++)
            let comparison_path = output_dir.join("sampler_comparison_3rows.png");
            save_grid_png(&comparison_path.to_string_lossy(), &all_samplers_flat, 3, 10)?;

            println!("\n============================================================");
            println!("  Benchmark Complete!");
            println!("  Saved 3-Row Comparison Grid to: {}", comparison_path.display());
            println!("  - Row 1: DDPM (100 steps)");
            println!("  - Row 2: DDIM (20 steps)");
            println!("  - Row 3: DPM-Solver++ 2M (8 steps)");
            println!("============================================================");
        }
    }

    Ok(())
}
