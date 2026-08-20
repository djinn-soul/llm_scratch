use burn::module::AutodiffModule;
use burn::optim::{GradientsParams, Optimizer};
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{Distribution, Tensor};

use burn::optim::AdamWConfig;
use candle_core::Device as CandleDevice;
use rand::RngExt;

use llm_scratch_rs::models::diffusion::dit::DiTConfig;
use llm_scratch_rs::models::diffusion::dit::DiffusionTransformer;
use llm_scratch_rs::utils::mnist_utils::{acquire_mnist, save_png};

// --- Option 1: Native ROCm (AMD HIP) ---
#[cfg(feature = "rocm")]
pub type MyBackend = burn::backend::Rocm;
// --- Option 2: WGPU (Vulkan on AMD RX 9070) ---
#[cfg(feature = "wgpu")]
pub type MyBackend = burn::backend::Wgpu;
// --- Option 3: CPU (Ryzen 9 9900X) ---
#[cfg(not(any(feature = "rocm", feature = "wgpu")))]
pub type MyBackend = burn::backend::ndarray::NdArray;
// Autodiff wrapper for training
pub type MyAutodiffBackend = burn::backend::Autodiff<MyBackend>;

pub fn train_steps<B: AutodiffBackend>(
    model: DiffusionTransformer<B>,
    x_t: Tensor<B, 4>,
    target_noise: Tensor<B, 4>,
    t_emb: Tensor<B, 2>,
    class_labels: Tensor<B, 1, burn::tensor::Int>,
    optimizer: &mut impl Optimizer<DiffusionTransformer<B>, B>,
    lr: f64,
) -> (DiffusionTransformer<B>, f32) {
    let pred_noise = model.forward(x_t, t_emb, class_labels);

    let diff = pred_noise - target_noise;
    let loss = diff.powf_scalar(2.0).mean();

    let loss_val = loss.clone().into_data().as_slice::<f32>().unwrap()[0];

    let grads = GradientsParams::from_grads(loss.backward(), &model);

    let model = optimizer.step(lr, model, grads);
    (model, loss_val)
}

pub fn get_time_step_embeddings<B: Backend>(
    timesteps: &[usize],
    dim: usize,
    device: &B::Device,
) -> Tensor<B, 2> {
    let half_dim = dim / 2;
    let mut data = Vec::with_capacity(timesteps.len() * dim);

    for &t in timesteps {
        for i in 0..half_dim {
            let freq = (-(i as f32) * (10000.0f32.ln()) / (half_dim as f32)).exp();
            let arg = t as f32 * freq;
            data.push(arg.sin());
            data.push(arg.cos());
        }
    }
    Tensor::<B, 1>::from_floats(data.as_slice(), device).reshape([timesteps.len(), dim])
}

pub struct SimpleNoiseScheduler {
    pub alphas_cumprod: Vec<f32>,
    pub sqrt_alphas_cumprod: Vec<f32>,
    pub sqrt_one_minus_alphas_cumprod: Vec<f32>,
}

impl SimpleNoiseScheduler {
    pub fn new_linear(timesteps: usize, beta_start: f32, beta_end: f32) -> Self {
        let mut alphas_cumprod = Vec::with_capacity(timesteps);
        let mut cumprod = 1.0f32;

        for t in 0..timesteps {
            let beta =
                beta_start + (beta_end - beta_start) * (t as f32) / ((timesteps - 1) as f32);
            let alpha = 1.0 - beta;
            cumprod *= alpha;

            alphas_cumprod.push(cumprod);
        }

        let sqrt_alphas_cumprod = alphas_cumprod.iter().map(|&a| a.sqrt()).collect();
        let sqrt_one_minus_alphas_cumprod =
            alphas_cumprod.iter().map(|&a| (1.0 - a).sqrt()).collect();

        Self {
            alphas_cumprod,
            sqrt_alphas_cumprod,
            sqrt_one_minus_alphas_cumprod,
        }
    }
}

pub fn q_sample<B: Backend>(
    x_0: Tensor<B, 4>,
    noise: Tensor<B, 4>,
    timesteps: &[usize],
    scheduler: &SimpleNoiseScheduler,
    device: &B::Device,
) -> Tensor<B, 4> {
    let b = timesteps.len();
    let mut sqrt_alpha_data = Vec::with_capacity(b);
    let mut sqrt_one_minus_alpha_data = Vec::with_capacity(b);

    for &t in timesteps {
        sqrt_alpha_data.push(scheduler.sqrt_alphas_cumprod[t]);
        sqrt_one_minus_alpha_data.push(scheduler.sqrt_one_minus_alphas_cumprod[t]);
    }

    let sqrt_alpha =
        Tensor::<B, 1>::from_floats(sqrt_alpha_data.as_slice(), device).reshape([b, 1, 1, 1]);
    let sqrt_one_minus_alpha =
        Tensor::<B, 1>::from_floats(sqrt_one_minus_alpha_data.as_slice(), device)
            .reshape([b, 1, 1, 1]);

    x_0 * sqrt_alpha + noise * sqrt_one_minus_alpha
}

pub fn sample_ddim<B: Backend>(
    model: &DiffusionTransformer<B>,
    scheduler: &SimpleNoiseScheduler,
    batch_size: usize,
    class_label: usize,
    steps: usize,
    hidden_dim: usize,
    device: &B::Device,
) -> Tensor<B, 4> {
    // 1. Start from pure random noise x_T ~ N(0, I)
    let mut x = Tensor::random(
        [batch_size, 1, 28, 28],
        Distribution::Normal(0.0, 1.0),
        device,
    );
    let class_labels = Tensor::from_ints(vec![class_label as i32; batch_size].as_slice(), device);

    let total_steps = scheduler.alphas_cumprod.len();
    let step_size = total_steps / steps;
    let timesteps: Vec<usize> = (0..steps).map(|i| (steps - 1 - i) * step_size).collect();

    for (i, &t) in timesteps.iter().enumerate() {
        let t_emb = get_time_step_embeddings::<B>(&vec![t; batch_size], hidden_dim, device);
        let pred_noise = model.forward(x.clone(), t_emb, class_labels.clone());

        let alpha_bar_t = scheduler.alphas_cumprod[t];
        let alpha_bar_prev = if i + 1 < timesteps.len() {
            scheduler.alphas_cumprod[timesteps[i + 1]]
        } else {
            1.0
        };

        // Predict clean image x_0
        let pred_x0 = (x - pred_noise.clone() * (1.0 - alpha_bar_t).sqrt()) / alpha_bar_t.sqrt();

        // Direction pointing to x_t
        let dir = pred_noise * (1.0 - alpha_bar_prev).sqrt();

        // Assemble x_{t-prev}
        x = pred_x0 * alpha_bar_prev.sqrt() + dir;
    }

    x
}

pub fn main() -> anyhow::Result<()> {
    let device = Default::default();
    println!("Initializing DiT (Diffusion Transformer) on Burn/NdArray/ROCm...");

    let config = DiTConfig {
        img_size: 28,
        in_channels: 1,
        patch_size: 4,
        num_classes: 10,
        hidden_dim: 128,
        depth: 4,
        num_heads: 4,
        mlp_ratio: 4.0,
    };
    let mut model: DiffusionTransformer<MyAutodiffBackend> =
        DiffusionTransformer::new(config.clone(), &device);
    let mut optimizer = AdamWConfig::new().init();
    let scheduler = SimpleNoiseScheduler::new_linear(100, 1e-4, 0.02);
    let lr = 2e-4;

    // 2. Load MNIST Dataset (60k images normalized to [-1, 1])
    println!("Loading MNIST dataset...");
    let (candle_images, labels_vec) = acquire_mnist(&CandleDevice::Cpu)?;
    let images_flat = candle_images.to_vec2::<f32>()?; // 60,000 x 784
    let total_samples = images_flat.len();
    println!("Successfully loaded {} MNIST samples.", total_samples);

    let batch_size = 64;
    let num_steps = 1000;
    println!(
        "Starting DiT training for {} steps with batch size {}...",
        num_steps, batch_size
    );

    let mut rng = rand::rng();
    for step in 1..=num_steps {
        // --- Step A: Sample Random Batch ---
        let mut batch_images = Vec::with_capacity(batch_size * 784);
        let mut batch_labels = Vec::with_capacity(batch_size);
        for _ in 0..batch_size {
            let idx = rng.random_range(0..total_samples);
            batch_images.extend_from_slice(&images_flat[idx]);
            batch_labels.push(labels_vec[idx] as i32);
        }
        // Convert to Burn Tensors: [B, 1, 28, 28] and [B]
        let x_0: Tensor<MyAutodiffBackend, 4> =
            Tensor::<MyAutodiffBackend, 1>::from_floats(batch_images.as_slice(), &device)
                .reshape([batch_size, 1, 28, 28]);
        let class_labels: Tensor<MyAutodiffBackend, 1, burn::tensor::Int> =
            Tensor::from_ints(batch_labels.as_slice(), &device);

        // --- Step B: Sample Random Timesteps t in [0, 100) ---
        let timesteps: Vec<usize> = (0..batch_size)
            .map(|_| rng.random_range(0..100))
            .collect();

        // --- Step C: Sample Gaussian Noise epsilon ---
        let noise: Tensor<MyAutodiffBackend, 4> = Tensor::random(
            [batch_size, 1, 28, 28],
            Distribution::Normal(0.0, 1.0),
            &device,
        );

        // --- Step D: Forward Diffusion x_t = sqrt(alpha_bar)*x_0 + sqrt(1 - alpha_bar)*noise ---
        let x_t = q_sample(x_0, noise.clone(), &timesteps, &scheduler, &device);

        // --- Step E: Sinusoidal Timestep Embeddings ---
        let t_emb =
            get_time_step_embeddings::<MyAutodiffBackend>(&timesteps, config.hidden_dim, &device);

        // --- Step F: Optimization Step (Forward -> Loss -> Backward -> AdamW Step) ---
        let (updated_model, loss_val) =
            train_steps(model, x_t, noise, t_emb, class_labels, &mut optimizer, lr);
        model = updated_model;

        // Logging
        if step % 50 == 0 || step == 1 {
            println!("Step {:4}/{}: MSE Loss = {:.6}", step, num_steps, loss_val);
        }

        // Periodic DDIM Image Sampling Preview
        if step % 200 == 0 || step == num_steps {
            println!("Sampling DDIM preview at step {}...", step);
            let valid_model = model.valid();
            let sample =
                sample_ddim(&valid_model, &scheduler, 1, 3, 20, config.hidden_dim, &device);
            let sample_pixels = sample.into_data().as_slice::<f32>().unwrap().to_vec();
            let filename = format!("dit_sample_step_{:04}.png", step);
            save_png(&filename, &sample_pixels)?;
            println!("Saved preview sample to {}", filename);
        }
    }

    println!("DiT training finished successfully!");
    Ok(())
}
