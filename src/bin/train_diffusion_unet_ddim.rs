// =============================================================================
// train_diffusion_unet_ddim.rs — U-Net Classifier-Free Guidance DDPM/DDIM trainer
// =============================================================================
//
// This binary trains a SimpleDenoisingUNet on MNIST with Classifier-Free
// Guidance (CFG) and a cosine noise schedule, then evaluates using both DDPM
// and DDIM sampling.
//
// Training procedure (each epoch):
//   1. Sample a random minibatch of images + labels from MNIST.
//   2. Stochastically drop 15% of labels → all-zeros (CFG label dropout).
//   3. Sample random timesteps t ~ Uniform(0, T) for each sample.
//   4. Forward-diffuse: x_t = sqrt(ᾱ_t)*x_0 + sqrt(1-ᾱ_t)*ε.
//   5. Model predicts ε from concat(x_t, time_emb, class_one_hot).
//   6. MSE loss between predicted noise and actual noise.
//   7. Manual backward pass + Adam optimizer step.
//
// Post-training diagnostics:
//   - Reconstruction: corrupt a known image at t=20,50,80 and reconstruct.
//   - CFG sweep: generate at guidance scales s=0, 1, 3 to compare.
//   - Frame dump: save every reverse step as a PNG for animation.
//   - Periodic checkpoints with DDIM-previewed grid images.
//

use anyhow::Result;
use candle_core::{Device, Tensor};
use rand::RngExt;

use llm_scratch_rs::common::Ema;
use llm_scratch_rs::models::diffusion::sampling::{
    sample_ddim_cfg_strided_with_call_back, sample_ddpm_cfg_from_timestep_with_callback,
    sample_ddpm_cfg_strided_with_callback,
};

// Model components:
use llm_scratch_rs::models::diffusion::{
    get_time_embedding, load_model_checkpoint, make_one_hot_cfg, one_hot_class,
    save_model_checkpoint, BetaScheduler, DenoisingModel, MlpAdamOptimizer, Parameterized,
    SimpleDenoisingUNet,
};

// Shared MNIST dataset loader and PNG writer.
use llm_scratch_rs::utils::mnist_utils::{acquire_mnist, save_png};

// Use mimalloc as the global allocator for faster allocation patterns in
// tensor-heavy workloads (many small allocations during forward/backward).
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

// The cosine scheduler has 100 entries indexed 0..99. Keeping the diagnostic
// start explicit makes it difficult to accidentally pass the step count (100)
// as an index, and lets reconstruction helpers share the partial-chain API.
const SAMPLING_START_TIMESTEP: usize = 99;

// Assembles multiple 28×28 grayscale images into a single grid PNG.
//
// Images arrive as a flat f32 slice in [-1, 1] (the training normalization).
// Each image is denormalized to [0, 255] and placed into the grid at its
// (row, col) position. This is used for checkpoint preview grids.
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
                    // Denormalize: [-1, 1] → [0, 1] → [0, 255]
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
    // Strided to ~20 steps instead of the full 100 — respaced DDPM stays
    // faithful to the stochastic posterior at this stride, so diagnostic
    // samples cost ~5x fewer forward passes with no visible quality loss.
    let generated = sample_ddpm_cfg_strided_with_callback(
        model,
        scheduler,
        initial_noise,
        SAMPLING_START_TIMESTEP,
        20,
        img_dim,
        time_emb_dim,
        target_one_hot,
        guidance_scale,
        device,
        |_, _| Ok(()),
    )?;

    let final_pixels = generated.flatten_all()?.to_vec1::<f32>()?;
    save_png(filename, &final_pixels)?;
    Ok(())
}

fn save_fixed_noise_checkpoint(
    model: &dyn DenoisingModel,
    scheduler: &BetaScheduler,
    fixed_noise: &Tensor,
    class_one_hot: &Tensor,
    guidance_scale: f64,
    epoch: usize,
    img_dim: usize,
    time_emb_dim: usize,
    device: &Device,
) -> Result<()> {
    std::fs::create_dir_all("unet_checkpoints")?;
    fixed_noise.save_safetensors("fixed_noise", "unet_checkpoints/fixed_noise.safetensors")?;

    // Deterministic DDIM preview, strided to ~20 reverse steps instead of the
    // full 100. DDIM's non-Markovian formulation stays accurate at large
    // strides, so this is a ~5x cut in forward passes for a preview image with
    // no visible quality loss.
    let generated = sample_ddim_cfg_strided_with_call_back(
        model,
        scheduler,
        fixed_noise.clone(),
        SAMPLING_START_TIMESTEP,
        10,
        img_dim,
        time_emb_dim,
        class_one_hot,
        guidance_scale,
        device,
        |_, _| Ok(()),
    )?;
    let pixels = generated.flatten_all()?.to_vec1::<f32>()?;
    // Save as a 4x4 grid (16 images total)
    save_grid_png(
        &format!(
            "unet_checkpoints/epoch_{:04}_s{:.0}_grid.png",
            epoch, guidance_scale
        ),
        &pixels,
        4,
        4,
    )?;
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

    sample_ddpm_cfg_from_timestep_with_callback(
        model,
        scheduler,
        initial_noise,
        SAMPLING_START_TIMESTEP,
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

// =============================================================================
// save_reconstruction_diagnostics — single-step noise prediction quality check
// =============================================================================
//
// HOW IT WORKS:
//   1. Take the first training image x_0 and its label.
//   2. Forward-diffuse it to several noise levels (t=20, 50, 80).
//   3. Ask the model to predict the noise at each level.
//   4. Reconstruct x_0_hat from the prediction using the closed-form:
//        x0_hat = (x_t - sqrt(1 - ᾱ_t) * eps_hat) / sqrt(ᾱ_t)
//   5. Save the noisy image and reconstruction side-by-side.
//
// WHY these timesteps?
//   t=20 is low noise (easy), t=50 is medium, t=80 is heavy noise (hard).
//   Comparing reconstructions at these levels shows whether the model has
//   learned to predict noise accurately across the full diffusion range.
//
// WHY single-step reconstruction and not full reverse sampling?
//   Single-step isolates the model's per-step noise prediction quality
//   without compounding errors across the entire reverse chain. If the
//   single-step reconstruction is good but full sampling is bad, the issue
//   is likely error accumulation, not the model itself.
fn save_reconstruction_diagnostics(
    model: &dyn DenoisingModel,
    scheduler: &BetaScheduler,
    images: &Tensor,
    labels: &[u8],
    time_emb_dim: usize,
    device: &Device,
) -> Result<()> {
    std::fs::create_dir_all("unet_recon")?;

    // Use the very first training image as a fixed reference.
    let x0 = images.narrow(0, 0, 1)?;
    let class_one_hot = one_hot_class(labels[0] as usize, 10, device)?;
    save_png(
        "unet_recon/original.png",
        &x0.flatten_all()?.to_vec1::<f32>()?,
    )?;

    let alphas_cumprod = scheduler.alphas_cumprod.to_vec1::<f32>()?;
    for &t_step in &[20usize, 50, 80] {
        // Forward-diffuse x_0 to noise level t.
        let t_tensor = Tensor::new(&[t_step as u32], device)?;
        let noise = Tensor::randn(0.0f32, 1.0f32, x0.shape(), device)?;
        let xt = scheduler.add_noise(&x0, &noise, &t_tensor)?;

        // Build model input: concat(x_t, time_emb, class_one_hot)
        let time_emb = get_time_embedding(&t_tensor, time_emb_dim)?;
        let input_v = Tensor::cat(&[&xt, &time_emb, &class_one_hot], 1)?;
        let (pred_noise, _) = model.forward(&input_v)?;

        // Reconstruct x_0 from the noisy image and predicted noise.
        //
        // ── DERIVATION ──
        //
        // From the forward process:
        //   x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1-alpha_bar_t) * epsilon
        //
        // Rearranging to solve for x_0:
        //   x_0 = (x_t - sqrt(1-alpha_bar_t) * epsilon) / sqrt(alpha_bar_t)
        //
        // Substituting the model's prediction epsilon_theta for the true epsilon:
        //   x0_hat = (x_t - sqrt(1-alpha_bar_t) * eps_theta) / sqrt(alpha_bar_t)
        //
        // If the model perfectly predicted the noise, x0_hat = x_0 exactly.
        // At low t (little noise), the prediction is easy and x0_hat is sharp.
        // At high t (heavy noise), the prediction is hard and x0_hat is blurry.
        let alpha_bar = alphas_cumprod[t_step] as f64;
        let sqrt_alpha_bar = alpha_bar.sqrt();
        let sqrt_one_minus_alpha_bar = (1.0 - alpha_bar).sqrt();
        let x0_hat = xt
            .sub(&pred_noise.affine(sqrt_one_minus_alpha_bar, 0.0)?)?
            .affine(1.0 / sqrt_alpha_bar, 0.0)?;

        save_png(
            &format!("unet_recon/noisy_t{:03}.png", t_step),
            &xt.flatten_all()?.to_vec1::<f32>()?,
        )?;
        save_png(
            &format!("unet_recon/recon_t{:03}.png", t_step),
            &x0_hat.flatten_all()?.to_vec1::<f32>()?,
        )?;
    }

    Ok(())
}

fn main() -> Result<()> {
    // --- Device selection ---
    // Prefer CUDA GPU 1 (if multi-GPU), fall back to GPU 0, then CPU.
    let device = if candle_core::utils::cuda_is_available() {
        Device::new_cuda(1)
            .or_else(|_| Device::new_cuda(0))
            .unwrap_or(Device::Cpu)
    } else {
        Device::Cpu
    };
    println!("Active Device: {:?}", device);

    // WHY different batch sizes?
    //   GPU can handle larger batches efficiently; CPU is memory-constrained.
    let batch_size = if let Device::Cuda(_) = device {
        256
    } else {
        128
    };
    println!("Selected Batch Size: {}", batch_size);

    // Load MNIST: 60k images normalized to [-1, 1], flattened to (N, 784).
    println!("Loading MNIST dataset...");
    let (images, train_labels) = acquire_mnist(&device)?;
    let (total_samples, _) = images.dims2()?;
    println!("Loaded {} images of size 28x28", total_samples);

    // --- Model & schedule setup ---
    let img_dim = 784; // 28×28 pixels, flattened
    let time_emb_dim = 16; // Sinusoidal time embedding dimension
                           // cond_dim = time_emb_dim (16) + num_classes (10) = 26
                           // The model receives concat(x_t, time_emb, class_one_hot) as input.
    let cond_dim = time_emb_dim + 10;
    let model = SimpleDenoisingUNet::new(img_dim, cond_dim, &device)?;

    // WHY cosine schedule instead of linear?
    //   The cosine schedule (Nichol & Dhariwal, 2021) distributes noise more
    //   evenly across timesteps. Linear schedules destroy too much signal in
    //   early steps and waste capacity on near-clean images in late steps.
    //   Cosine gives the model useful gradients throughout the full chain.
    let scheduler = BetaScheduler::new_cosine(100, &device)?;

    let mut optimizer = MlpAdamOptimizer::new(&model, 1e-4)?;
    // Add EMA instance here (0.9999 decay rate):
    let mut ema = Ema::new(&model, 0.9999)?;
    // Fixed class and noise for deterministic checkpoint comparisons.
    // Using the same noise across epochs lets us visually track how the
    // model's output evolves during training.
    let sample_class = 3u32;
    let class_one_hot = one_hot_class(sample_class as usize, 10, &device)?;
    let checkpoint_noise = Tensor::randn(0.0f32, 1.0f32, (16, img_dim), &device)?;

    // --- Resume checkpoint support ---
    let args: Vec<String> = std::env::args().collect();
    let mut start_epoch = 1;

    if args.len() > 1 {
        if let Ok(resume_epoch) = args[1].parse::<usize>() {
            let model_path = format!("unet_checkpoints/epoch_{resume_epoch:04}.safetensors");
            let opt_path = format!("unet_checkpoints/opt_epoch_{resume_epoch:04}.safetensors");
            let ema_path = format!("unet_checkpoints/ema_epoch_{resume_epoch:04}.safetensors");

            if std::path::Path::new(&model_path).exists()
                && std::path::Path::new(&opt_path).exists()
            {
                println!("Resuming training from epoch {}...", resume_epoch);
                load_model_checkpoint(&model, &model_path, &device)?;
                optimizer.load_checkpoint(&opt_path, &device)?;

                if std::path::Path::new(&ema_path).exists() {
                    let temp_ema_model = SimpleDenoisingUNet::new(img_dim, cond_dim, &device)?;
                    load_model_checkpoint(&temp_ema_model, &ema_path, &device)?;
                    ema.shadow_params = temp_ema_model
                        .params()
                        .iter()
                        .map(|p| (*p).copy().map_err(Into::into))
                        .collect::<Result<Vec<_>>>()?;
                    ema.num_samples = optimizer.t;
                }

                start_epoch = resume_epoch + 1;
                println!(
                    "Successfully resumed model & optimizer state at step t = {}",
                    optimizer.t
                );
            } else {
                println!(
                    "Warning: Requested resume epoch {} but checkpoint files were not found.",
                    resume_epoch
                );
            }
        }
    }

    // Create the checkpoint directory up front, not lazily at the first save.
    //
    // The periodic block below writes the model weights and optimizer state
    // before it reaches `save_fixed_noise_checkpoint`, which used to be the only
    // caller of `create_dir_all`. On a clean tree that ordering made the very
    // first checkpoint fail with "The system cannot find the path specified"
    // and take the whole run down with it — after the full interval of training
    // had already been spent. Creating the directory before the loop means a
    // missing path can no longer surface hours in.
    std::fs::create_dir_all("unet_checkpoints")?;

    let num_epochs = 25000;
    println!(
        "Starting U-Net training for {} epochs (from epoch {})...",
        num_epochs, start_epoch
    );

    let start_time = std::time::Instant::now();

    // =========================================================================
    // TRAINING LOOP
    // =========================================================================
    //
    // Each epoch performs one minibatch gradient step. The training objective
    // follows Ho et al. (2020) "Denoising Diffusion Probabilistic Models".
    //
    // ── THE TRAINING OBJECTIVE (mathematical derivation) ──
    //
    // The true objective is to maximize the Evidence Lower Bound (ELBO):
    //
    //   log p(x_0) >= E_q[ log p(x_0|x_1) - KL(q(x_T|x_0) || p(x_T))
    //                      - sum_t KL(q(x_{t-1}|x_t,x_0) || p_theta(x_{t-1}|x_t)) ]
    //
    // Ho et al. showed that the KL terms simplify to a weighted sum of:
    //
    //   L_t = E_{x_0, epsilon} [ || epsilon - epsilon_theta(x_t, t) ||^2 ]
    //
    // where x_t = sqrt(alpha_bar_t)*x_0 + sqrt(1-alpha_bar_t)*epsilon.
    //
    // They further showed that DROPPING the per-timestep weighting and using
    // a SIMPLE (unweighted) MSE loss works better in practice:
    //
    //   L_simple = E_{t, x_0, epsilon} [ || epsilon - epsilon_theta(x_t, t) ||^2 ]
    //
    // This is what we compute below: sample t uniformly, sample epsilon,
    // construct x_t, predict epsilon_theta, and minimize MSE.
    //
    // ── WHY PREDICT NOISE INSTEAD OF x_0 DIRECTLY? ──
    //
    // Three equivalent parameterizations exist:
    //   (a) Predict x_0 directly → works but gradients are noisy at high t
    //   (b) Predict noise epsilon → more stable gradients across all t
    //   (c) Predict the score function → equivalent to (b) up to scaling
    //
    let mut rng = rand::rng();

    // Noise prediction (b) gives the most uniform gradient magnitudes
    // across timesteps, leading to faster and more stable training.
    for epoch in start_epoch..=num_epochs {
        // --- Step 1: Sample a random minibatch of training images. ---
        // Sample random indices on CPU directly, avoiding float-to-u32 casting
        // and CPU-GPU synchronization roundtrips.
        let indices: Vec<u32> = (0..batch_size)
            .map(|_| rng.random_range(0..total_samples as u32))
            .collect();
        let index_tensor = Tensor::new(indices.as_slice(), &device)?;
        let x0 = images.index_select(&index_tensor, 0)?;

        // --- Step 2: Build class conditioning with CFG label dropout. ---
        //
        // make_one_hot_cfg stochastically replaces 15% of labels with all-zeros.
        //
        // WHY 15% dropout?
        //   This is the CFG label dropout rate. During training, setting some
        //   labels to the null vector teaches the model to produce both
        //   conditional (with label) and unconditional (without label) noise
        //   predictions. At inference, CFG blends these two to amplify class
        //   signal. 10-20% dropout is standard; 15% is a reasonable middle.
        let batch_labels: Vec<u8> = indices.iter().map(|&x| train_labels[x as usize]).collect();
        let label_one_hot = make_one_hot_cfg(&batch_labels, 10, 0.15f32, &device)?;

        // --- Step 3: Sample random timesteps uniformly from [0, T). ---
        //
        // WHY uniform sampling?
        //   The simplified training objective from Ho et al. treats all timesteps
        //   equally (L_simple sums over t with uniform weight). Uniform t sampling
        //   is the Monte Carlo estimator for this sum. Non-uniform strategies
        //   exist (e.g. importance sampling by loss magnitude) but uniform is the
        //   standard baseline and works well for MNIST.
        let t_vec: Vec<u32> = (0..batch_size)
            .map(|_| rng.random_range(0..100u32))
            .collect();
        let t_tensor = Tensor::new(t_vec.as_slice(), &device)?;

        // --- Step 4: Forward diffusion — corrupt x_0 to x_t. ---
        //
        // ── THE REPARAMETERIZATION TRICK ──
        //
        // Instead of iterating the forward chain x_0 -> x_1 -> ... -> x_t
        // (which would require t sequential operations), we jump directly:
        //
        //   x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
        //
        // This works because the composition of Gaussian transitions is itself
        // Gaussian. The coefficients come from:
        //   E[x_t | x_0] = sqrt(alpha_bar_t) * x_0   (signal component)
        //   Var[x_t | x_0] = (1 - alpha_bar_t) * I    (noise component)
        //
        // The two coefficient squares sum to 1:
        //   (sqrt(alpha_bar_t))^2 + (sqrt(1-alpha_bar_t))^2 = alpha_bar_t + 1 - alpha_bar_t = 1
        //
        // This "variance-preserving" property means x_t has roughly unit variance
        // regardless of t, which helps the model see consistent input magnitudes.
        let noise = Tensor::randn(0.0f32, 1.0f32, x0.shape(), &device)?;
        let xt = scheduler.add_noise(&x0, &noise, &t_tensor)?;

        // --- Step 5: Build model input. ---
        //   input_v = concat(x_t, time_embedding, class_one_hot)
        //   Shape: (batch, img_dim + time_emb_dim + num_classes)
        let t_emb = get_time_embedding(&t_tensor, time_emb_dim)?;
        let input_v = Tensor::cat(&[&xt, &t_emb, &label_one_hot], 1)?;

        // --- Step 6: Forward + backward pass. ---
        //
        // Forward: model predicts epsilon_hat = epsilon_theta(concat(x_t, t_emb, class), theta)
        // The model takes the concatenated input and outputs a tensor of the
        // same shape as the noise (batch, img_dim). The intermediates (hidden
        // layer activations) are cached for the backward pass.
        let (pred, intermediates) = model.forward(&input_v)?;
        //
        // Backward: compute gradients of L_simple = MSE(epsilon_hat, epsilon)
        //
        // The MSE gradient w.r.t. the prediction is:
        //   dL/d(pred) = (2/N) * (pred - noise)
        //
        // where N = batch_size * img_dim. This gradient is then backpropagated
        // through the network using the chain rule and the cached intermediates.
        let grads = model.backward(&input_v, &intermediates, &pred, &noise)?;

        // --- Gradient Clipping (max_norm = 1.0) & Norm Computation ---
        let max_norm = 1.0f32;
        let is_log_epoch = epoch % 100 == 0 || epoch == 1;

        // Compute squared norm of each parameter gradient on-device without blocking CPU sync.
        let norm_sq_tensors: Vec<Tensor> = grads
            .iter()
            .map(|g| g.sqr()?.sum_all().map_err(Into::into))
            .collect::<Result<Vec<Tensor>>>()?;
        let stacked_norms = Tensor::stack(&norm_sq_tensors, 0)?;

        let (global_norm, grad_norms) = if is_log_epoch {
            let norms_sq_vec = stacked_norms.to_vec1::<f32>()?;
            let total_norm_sq: f32 = norms_sq_vec.iter().sum();
            let grad_norms: Vec<f32> = norms_sq_vec.into_iter().map(|n| n.sqrt()).collect();
            (total_norm_sq.sqrt(), grad_norms)
        } else {
            let total_norm_sq = stacked_norms.sum_all()?.to_scalar::<f32>()?;
            (total_norm_sq.sqrt(), Vec::new())
        };
        let grads = if global_norm > max_norm {
            let scale = (max_norm / (global_norm + 1e-6)) as f64;
            grads
                .iter()
                .map(|g| g.affine(scale, 0.0).map_err(Into::into))
                .collect::<Result<Vec<Tensor>>>()?
        } else {
            grads
        };

        // --- Logging (every 100 epochs) ---
        // Print MSE loss and per-parameter gradient L2 norms for monitoring.
        if is_log_epoch {
            let elapsed = start_time.elapsed().as_secs_f64();
            let loss = pred.sub(&noise)?.sqr()?.mean_all()?.to_scalar::<f32>()?;
            let param_names = model.param_names();
            let norms_str: Vec<String> = param_names
                .iter()
                .zip(grad_norms.iter())
                .map(|(name, norm)| format!("{} norm(pre-clip): {:.4}", name, norm))
                .collect();
            println!(
                "Epoch {:>5}/{} | Loss: {:.6} | Elapsed: {:.2}s | Speed: {:.1} epochs/s | {}",
                epoch,
                num_epochs,
                loss,
                elapsed,
                epoch as f64 / elapsed,
                norms_str.join(", ")
            );
        }

        // --- Learning Rate Schedule: Linear Warmup (500 epochs) + Cosine Decay ---
        let base_lr = 1e-4f64;
        let min_lr = 1e-6f64;
        let warmup_epochs = 500.0f64;
        let total_epochs_f = num_epochs as f64;

        let current_lr = if (epoch as f64) < warmup_epochs {
            base_lr * (epoch as f64 / warmup_epochs)
        } else {
            let progress = (epoch as f64 - warmup_epochs) / (total_epochs_f - warmup_epochs);
            min_lr + 0.5 * (base_lr - min_lr) * (1.0 + (std::f64::consts::PI * progress).cos())
        };
        optimizer.lr = current_lr;
        // --- Step 7: Adam optimizer update (uses LR set above). ---
        optimizer.step(&model, &grads)?;
        // 2. Update EMA shadow weights with the new model weights
        ema.update(&model)?;
        if epoch % 500 == 0 {
            println!(
                "Saving fixed-noise checkpoint sample for epoch {}...",
                epoch
            );
            // Save online model weights
            save_model_checkpoint(
                &model,
                format!("unet_checkpoints/epoch_{epoch:04}.safetensors"),
            )?;
            // Optimizer state is saved under the same epoch tag as the online
            // weights above, and must stay paired with them: resuming these
            // weights with zeroed Adam moments (or a reset `t`, which re-applies
            // bias correction) produces a loss spike at restart.
            //
            // Deliberately saved before the EMA swap below — the EMA weights are
            // an evaluation artifact, and Adam's moments belong to the online
            // weights being trained.
            optimizer
                .save_checkpoint(format!("unet_checkpoints/opt_epoch_{epoch:04}.safetensors"))?;

            let online_generated = sample_ddim_cfg_strided_with_call_back(
                &model,
                &scheduler,
                checkpoint_noise.clone(),
                SAMPLING_START_TIMESTEP,
                10, // 10 strided DDIM steps
                img_dim,
                time_emb_dim,
                &class_one_hot,
                1.0,
                &device,
                |_, _| Ok(()),
            )?;
            save_grid_png(
                &format!("unet_checkpoints/epoch_{epoch:04}_online_grid.png"),
                &online_generated.flatten_all()?.to_vec1::<f32>()?,
                4,
                4,
            )?;
            // --- NON-DESTRUCTIVE EMA EVALUATION ---
            // A. Backup live training weights
            ema.store(&model)?;

            // B. Copy EMA shadow weights into the live model
            ema.copy_to_model(&model)?;
            // C. Save EMA checkpoint and generate preview image using EMA weights
            save_model_checkpoint(
                &model,
                format!("unet_checkpoints/ema_epoch_{epoch:04}.safetensors"),
            )?;
            save_fixed_noise_checkpoint(
                &model,
                &scheduler,
                &checkpoint_noise,
                &class_one_hot,
                // Standard CFG semantics: s=1 is the conditional prediction.
                // Larger values intentionally extrapolate and are a separate
                // quality/fidelity experiment, not the neutral checkpoint view.
                1.0,
                epoch,
                img_dim,
                time_emb_dim,
                &device,
            )?;
            // D. Restore live training weights so training continues cleanly
            ema.restore(&model)?;
        }
    }

    // --- Post-training diagnostics ---
    // These run once after training completes to assess model quality.

    // Diagnostic 1: Single-step reconstruction at t=20, 50, 80.
    // Shows how well the model predicts noise at different corruption levels.
    println!("Saving reconstruction diagnostics to folder 'unet_recon'...");
    save_reconstruction_diagnostics(
        &model,
        &scheduler,
        &images,
        &train_labels,
        time_emb_dim,
        &device,
    )?;
    println!("Reconstruction diagnostics saved to 'unet_recon/'.");

    // Diagnostic 2: CFG guidance scale comparison.
    //   s=0 → unconditional (no class guidance — shows what "random digit" looks like)
    //   s=1 → standard conditional (ordinary class-conditioned prediction)
    //   s=3 → amplified guidance (sharper class features, less diversity)
    println!("Generating guided sampling comparison images...");

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

    // Diagnostic 3: Save every reverse step as a separate PNG frame.
    // These frames can be assembled into an animation showing the full
    // denoising trajectory from pure noise → recognizable digit.
    println!("Saving reverse sampling frames to folder 'unet_frames'...");
    save_cfg_sample_frames(
        &model,
        &scheduler,
        img_dim,
        time_emb_dim,
        sample_class,
        1.0,
        "unet_frames",
        &device,
    )?;
    println!("Denoising frames saved to 'unet_frames/'.");

    Ok(())
}
