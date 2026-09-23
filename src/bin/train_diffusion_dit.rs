//! # Class-Conditional Diffusion Transformer (DiT) Training
//!
//! ## Mathematical Foundations of the Training Loop:
//! 1. **Data Normalization**: Raw uint8 pixels $[0, 255]$ are scaled to the zero-mean domain $[-1.0, 1.0]$.
//! 2. **Forward Perturbation ($q$-sampling)**:
//!    $$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})$$
//! 3. **Classifier-Free Dropout ($p_{\text{uncond}} = 0.15$)**:
//!    Randomly masks class IDs to token 10, training the network as both a conditioned model and unconditioned prior.
//! 4. **Variational Surrogate Loss ($L_{\text{simple}}$)**:
//!    $$L = \frac{1}{B} \sum_{i=1}^B \| \epsilon_i - \epsilon_\theta(x_{t, i}, t_i, c_i) \|_2^2$$
//! 5. **Polyak EMA Parameter Stabilization**:
//!    $$\theta_{\text{EMA}} \leftarrow 0.999 \cdot \theta_{\text{EMA}} + 0.001 \cdot \theta_{\text{live}}$$

use burn::module::{AutodiffModule, Module};
use burn::optim::AdamWConfig;
use burn::record::CompactRecorder;
use burn::tensor::{Distribution, Tensor};
use candle_core::Device as CandleDevice;
use rand::RngExt;

use llm_scratch_rs::models::diffusion::dit::{
    find_latest_checkpoint, get_learning_rate, get_time_step_embeddings, q_sample,
    sample_ddim_with_cfg, save_lookbook_collage, train_steps, update_ema, DiTConfig,
    DiffusionTransformer, SimpleNoiseScheduler, FASHION_CLASSES,
};
use llm_scratch_rs::utils::mnist_utils::{acquire_mnist, save_png};

// ============================================================================
// Backend Configuration (ROCm / WGPU / CPU)
// ============================================================================
// Option 1: Native ROCm (AMD HIP hardware acceleration)
#[cfg(feature = "rocm")]
pub type MyBackend = burn::backend::Rocm;

// Option 2: WGPU (Vulkan on AMD GPU) — Default when ROCm is not requested
#[cfg(all(not(feature = "rocm"), not(feature = "cpu")))]
pub type MyBackend = burn::backend::Wgpu;

// Option 3: CPU fallback (NdArray on CPU)
#[cfg(feature = "cpu")]
pub type MyBackend = burn::backend::ndarray::NdArray;

/// Autodiff wrapper enabling automatic differentiation and gradient tracking
pub type MyAutodiffBackend = burn::backend::Autodiff<MyBackend>;

pub fn main() -> anyhow::Result<()> {
    let device = Default::default();
    #[cfg(feature = "rocm")]
    println!("Initializing DiT (Diffusion Transformer) on Native ROCm (AMD HIP)...");
    #[cfg(all(not(feature = "rocm"), not(feature = "cpu")))]
    println!("Initializing DiT (Diffusion Transformer) on WGPU (Vulkan / GPU)...");
    #[cfg(feature = "cpu")]
    println!("Initializing DiT (Diffusion Transformer) on CPU (NdArray)...");

    // Collect optional command line arguments (e.g. `cargo run -- resume` or `-- checkpoints/...`)
    let args: Vec<String> = std::env::args().collect();
    let config = DiTConfig {
        img_size: 28,
        in_channels: 1,
        patch_size: 2,
        num_classes: 11, // 10 clothing classes + 1 null token for CFG
        hidden_dim: 256,
        depth: 6,
        num_heads: 8,
        mlp_ratio: 4.0,
    };
    let num_timesteps = 500;

    let mut model: DiffusionTransformer<MyAutodiffBackend> =
        DiffusionTransformer::new(config.clone(), &device);

    // Initialize EMA shadow model (inference backend without autodiff graph)
    let mut ema_model: DiffusionTransformer<MyBackend> =
        DiffusionTransformer::new(config.clone(), &device);

    let mut optimizer = AdamWConfig::new().init();
    let scheduler = SimpleNoiseScheduler::new_cosine(num_timesteps);

    // 2. Load MNIST / Fashion-MNIST Dataset (60k images normalized to [-1, 1])
    println!("Loading Fashion-MNIST dataset...");
    let (candle_images, labels_vec) = acquire_mnist(&CandleDevice::Cpu)?;
    let images_flat = candle_images.to_vec2::<f32>()?; // 60,000 x 784
    let total_samples = images_flat.len();
    println!("Successfully loaded {} samples.", total_samples);

    let batch_size = 64;
    let num_steps = std::env::var("STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(26000);
    let checkpoint_interval = 1000;

    println!(
        "Starting DiT training for {} steps with batch size {} (saving checkpoint tensors every {} steps)...",
        num_steps, batch_size, checkpoint_interval
    );

    // Ensure output directories exist
    std::fs::create_dir_all("checkpoints")?;
    std::fs::create_dir_all("samples")?;

    // Scan checkpoints/ for resuming
    let mut start_step = 1;
    let resume_env = std::env::var("RESUME").ok();
    let resume_arg = args
        .get(1)
        .map(|s| s.as_str())
        .or(resume_env.as_deref());

    if let Some(resume_val) = resume_arg {
        let checkpoint_info = if resume_val == "resume"
            || resume_val == "auto"
            || resume_val == "1"
            || resume_val == "true"
        {
            find_latest_checkpoint("checkpoints")
        } else {
            let step = llm_scratch_rs::models::diffusion::dit::extract_step_from_path(resume_val)
                .unwrap_or(0);
            Some((resume_val.to_string(), step))
        };

        if let Some((cp_path, step_num)) = checkpoint_info {
            println!("🔄 Resuming training from checkpoint: {}", cp_path);
            let recorder = CompactRecorder::new();
            let stem = cp_path.trim_end_matches(".mpk");
            model = model.load_file(stem, &recorder, &device)?;
            ema_model = ema_model.load_file(stem, &recorder, &device)?;
            start_step = step_num + 1;
            println!(
                "   Continuing from step {} up to {}...",
                start_step, num_steps
            );
        } else {
            println!("⚠️ No checkpoint found to resume from. Starting fresh from step 1.");
        }
    }

    let mut rng = rand::rng();
    for step in start_step..=num_steps {
        // --- Step A: Sample Random Mini-batch & Apply CFG Dropout ---
        // During training, Classifier-Free Guidance requires teaching the model to predict
        // noise BOTH with and without conditioning. We randomly replace the class label
        // with the null token (Class 10) 15% of the time (p_uncond = 0.15).
        let mut batch_images = Vec::with_capacity(batch_size * 784);
        let mut batch_labels = Vec::with_capacity(batch_size);
        for _ in 0..batch_size {
            let idx = rng.random_range(0..total_samples);
            batch_images.extend_from_slice(&images_flat[idx]);
            if rng.random_range(0.0..1.0) < 0.15 {
                batch_labels.push(10); // Null class token for unconditional learning
            } else {
                batch_labels.push(labels_vec[idx] as i32); // Ground truth class (0..9)
            }
        }

        // Reshape into Burn tensors: Images [B, 1, 28, 28] and Labels [B]
        let x_0: Tensor<MyAutodiffBackend, 4> =
            Tensor::<MyAutodiffBackend, 1>::from_floats(batch_images.as_slice(), &device)
                .reshape([batch_size, 1, 28, 28]);
        let class_labels: Tensor<MyAutodiffBackend, 1, burn::tensor::Int> =
            Tensor::from_ints(batch_labels.as_slice(), &device);

        // --- Step B: Sample Diffusion Timesteps t ~ Uniform(0, T) ---
        let timesteps: Vec<usize> = (0..batch_size)
            .map(|_| rng.random_range(0..num_timesteps))
            .collect();

        // --- Step C: Sample Standard Gaussian Noise ε ~ N(0, I) ---
        let noise: Tensor<MyAutodiffBackend, 4> = Tensor::random(
            [batch_size, 1, 28, 28],
            Distribution::Normal(0.0, 1.0),
            &device,
        );

        // --- Step D: Forward Noising (Closed-Form q-sampling) ---
        // Computes x_t = sqrt(α_bar_t) * x_0 + sqrt(1 - α_bar_t) * ε
        let x_t = q_sample(x_0, noise.clone(), &timesteps, &scheduler, &device);

        // --- Step E: Compute Continuous Sinusoidal Timestep Embeddings ---
        let t_emb =
            get_time_step_embeddings::<MyAutodiffBackend>(&timesteps, config.hidden_dim, &device);

        // --- Step F: Learning Rate Schedule (Warmup + Cosine Decay) ---
        let current_lr = get_learning_rate(
            step, num_steps, 500,  /* 500-step linear warmup */
            2e-4, /* peak learning rate */
            1e-5, /* minimum floor learning rate */
        );

        // --- Step G: Forward Pass, MSE Loss, Backpropagation, and AdamW Step ---
        let (updated_model, loss_val) = train_steps(
            model,
            x_t,
            noise,
            t_emb,
            class_labels,
            &mut optimizer,
            current_lr,
        );
        model = updated_model;

        // --- Step H: Update Exponential Moving Average (EMA) Weights ---
        // Keeps a low-pass filtered copy of model weights: θ_ema ← 0.999 * θ_ema + 0.001 * θ_live
        ema_model = update_ema(ema_model, &model.valid(), 0.999);

        // Periodic training loss logging
        if step % 50 == 0 || step == 1 {
            println!("Step {:5}/{}: MSE Loss = {:.6}", step, num_steps, loss_val);
        }

        // --- Step I: Periodic Evaluation & Lookbook Checkpointing ---
        if step % checkpoint_interval == 0 || step == num_steps {
            println!(
                "\n>>> [Step {}/{}] Generating EMA Lookbook & Saving Checkpoint...",
                step, num_steps
            );
            let valid_model = ema_model.clone();

            // 1. Save Model Weights Checkpoint (Compact binary .mpk format)
            let checkpoint_path = format!("checkpoints/dit_fashion_step_{:05}", step);
            let recorder = CompactRecorder::new();
            if let Err(e) = valid_model.clone().save_file(&checkpoint_path, &recorder) {
                eprintln!(
                    "Warning: Failed to save checkpoint to {}: {:?}",
                    checkpoint_path, e
                );
            } else {
                println!(
                    "    Saved model weights checkpoint to: {}.mpk",
                    checkpoint_path
                );
            }

            // 2. Generate preview samples for all 10 fashion classes using EMA model
            let mut class_samples = Vec::with_capacity(10);
            for class_id in 0..10 {
                let sample = sample_ddim_with_cfg(
                    &valid_model,
                    &scheduler,
                    1,
                    class_id,
                    50,
                    1.8,
                    config.hidden_dim,
                    &device,
                );
                let sample_pixels = sample.into_data().as_slice::<f32>().unwrap().to_vec();
                class_samples.push(sample_pixels);
            }

            // 3. Save 2x5 stitched lookbook collage
            let lookbook_path = format!("samples/dit_step_{:05}_lookbook.png", step);
            if let Err(e) = save_lookbook_collage(&lookbook_path, &class_samples) {
                eprintln!(
                    "Warning: Failed to save lookbook {}: {:?}",
                    lookbook_path, e
                );
            }

            // 4. Also save individual class images
            for (class_id, pixels) in class_samples.iter().enumerate() {
                let filename = format!(
                    "samples/dit_step_{:05}_class_{}_{}.png",
                    step, class_id, FASHION_CLASSES[class_id]
                );
                let _ = save_png(&filename, pixels);
            }
            println!("    Generated lookbook collage & 10 class previews in samples/\n");
        }
    }

    println!("DiT training finished successfully!");
    Ok(())
}
