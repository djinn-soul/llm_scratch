// =============================================================================
// train_diffusion_unet.rs — U-Net Classifier-Free Guidance DDPM trainer
// =============================================================================

use anyhow::Result;
use candle_core::{DType, Device, Tensor};

use llm_scratch_rs::models::diffusion::sampling::{sample_ddpm_cfg, sample_ddpm_cfg_with_callback};

// Model components:
use llm_scratch_rs::models::diffusion::{
    get_time_embedding, make_one_hot_cfg, one_hot_class, BetaScheduler, DenoisingModel,
    MlpAdamOptimizer, SimpleDenoisingUNet,
};

// Shared MNIST dataset loader and PNG writer.
use llm_scratch_rs::utils::mnist_utils::{acquire_mnist, save_png};

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

// =============================================================================
// save_cfg_sample — helper: generate one image at a given guidance scale
// =============================================================================
fn save_cfg_sample(
    model: &dyn DenoisingModel,
    scheduler: &BetaScheduler,
    target_one_hot: &Tensor,
    guidance_scale: f64,
    img_dim: usize,
    time_emb_dim: usize,
    filename: &str,
    device: &Device,
) -> Result<()> {
    let initial_noise = Tensor::randn(0.0f32, 1.0f32, (1, img_dim), device)?;
    let generated = sample_ddpm_cfg(
        model,
        scheduler,
        initial_noise,
        img_dim,
        time_emb_dim,
        target_one_hot,
        guidance_scale,
        device,
    )?;

    let final_pixels = generated.flatten_all()?.to_vec1::<f32>()?;
    save_png(filename, &final_pixels)?;
    Ok(())
}

// =============================================================================
// save_cfg_sample_frames — helper: save all 100 frames during reverse sampling
// =============================================================================
fn save_cfg_sample_frames(
    model: &dyn DenoisingModel,
    scheduler: &BetaScheduler,
    img_dim: usize,
    time_emb_dim: usize,
    class_label: u32,
    guidance_scale: f64,
    output_dir: &str,
    device: &Device,
) -> Result<()> {
    std::fs::create_dir_all(output_dir)?;

    let initial_noise = Tensor::randn(0.0f32, 1.0f32, (1, img_dim), device)?;
    let class_one_hot = one_hot_class(class_label as usize, 10, device)?;

    sample_ddpm_cfg_with_callback(
        model,
        scheduler,
        initial_noise,
        img_dim,
        time_emb_dim,
        &class_one_hot,
        guidance_scale,
        device,
        |frame_idx, xt| {
            let final_pixels = xt.flatten_all()?.to_vec1::<f32>()?;
            let frame_filename = format!("{}/frame_{:03}.png", output_dir, frame_idx);
            save_png(&frame_filename, &final_pixels)
        },
    )?;

    Ok(())
}

fn main() -> Result<()> {
    // --- Device selection & batch size matching ---
    let device = if candle_core::utils::cuda_is_available() {
        Device::new_cuda(1)
            .or_else(|_| Device::new_cuda(0))
            .unwrap_or(Device::Cpu)
    } else {
        Device::Cpu
    };
    println!("Active Device: {:?}", device);

    let batch_size = if let Device::Cuda(_) = device {
        256
    } else {
        128
    };
    println!("Selected Batch Size: {}", batch_size);

    println!("Loading MNIST dataset...");
    let (images, train_labels) = acquire_mnist(&device)?;
    let (total_samples, _) = images.dims2()?;
    println!("Loaded {} images of size 28x28", total_samples);

    let img_dim = 784;
    let time_emb_dim = 16;
    let cond_dim = time_emb_dim + 10;
    let mut model = SimpleDenoisingUNet::new(img_dim, cond_dim, &device)?;
    let scheduler = BetaScheduler::new(100, 1e-4, 0.02, &device)?;
    let mut optimizer = MlpAdamOptimizer::new(&model, 1e-4)?;

    let num_epochs = 1000;
    println!("Starting U-Net training for {} epochs...", num_epochs);

    let start_time = std::time::Instant::now();

    for epoch in 1..=num_epochs {
        let index_tensor =
            Tensor::rand(0.0f32, total_samples as f32 - 1e-4, (batch_size,), &device)?
                .to_dtype(DType::U32)?;

        let indices = index_tensor.to_vec1::<u32>()?;
        let x0 = images.index_select(&index_tensor, 0)?;

        let batch_labels: Vec<u8> = indices.iter().map(|&x| train_labels[x as usize]).collect();
        let label_one_hot = make_one_hot_cfg(&batch_labels, 10, 0.15f32, &device)?;

        let t_float = Tensor::rand(0.0f32, 100.0f32 - 1e-4, (batch_size,), &device)?;
        let t_tensor = t_float.to_dtype(DType::U32)?;

        let noise = Tensor::randn(0.0f32, 1.0f32, x0.shape(), &device)?;
        let xt = scheduler.add_noise(&x0, &noise, &t_tensor)?;

        let t_emb = get_time_embedding(&t_tensor, time_emb_dim)?;
        let cond = Tensor::cat(&[t_emb, label_one_hot], 1)?;
        let input_v = Tensor::cat(&[xt, cond], 1)?;

        let (pred, intermediates) = model.forward(&input_v)?;
        let grads = model.backward(&input_v, &intermediates, &pred, &noise)?;
        optimizer.step(&mut model, &grads)?;

        if epoch % 10 == 0 || epoch == 1 {
            let elapsed = start_time.elapsed().as_secs_f64();
            let loss = pred.sub(&noise)?.sqr()?.mean_all()?.to_scalar::<f32>()?;
            println!(
                "Epoch {:>5}/{} | Loss: {:.6} | Elapsed: {:.2}s | Speed: {:.1} epochs/s",
                epoch,
                num_epochs,
                loss,
                elapsed,
                epoch as f64 / elapsed
            );
        }
    }

    println!("Generating guided sampling comparison images...");
    let sample_class = 3u32;

    let class_one_hot = one_hot_class(sample_class as usize, 10, &device)?;

    save_cfg_sample(
        &model,
        &scheduler,
        &class_one_hot,
        0.0,
        img_dim,
        time_emb_dim,
        "mnist_cfg_unet_generated_s0.png",
        &device,
    )?;
    save_cfg_sample(
        &model,
        &scheduler,
        &class_one_hot,
        1.0,
        img_dim,
        time_emb_dim,
        "mnist_cfg_unet_generated_s1.png",
        &device,
    )?;
    save_cfg_sample(
        &model,
        &scheduler,
        &class_one_hot,
        3.0,
        img_dim,
        time_emb_dim,
        "mnist_cfg_unet_generated_s3.png",
        &device,
    )?;
    println!("Saved sample PNGs matching s=0, s=1, and s=3.");

    println!("Saving reverse sampling frames to folder 'unet_frames'...");
    save_cfg_sample_frames(
        &model,
        &scheduler,
        img_dim,
        time_emb_dim,
        sample_class,
        3.0,
        "unet_frames",
        &device,
    )?;
    println!("Denoising frames saved to 'unet_frames/'.");

    Ok(())
}
