use burn::module::{AutodiffModule, Module};
use burn::module::{ModuleMapper, ModuleVisitor, Param};
use burn::optim::AdamWConfig;
use burn::optim::{GradientsParams, Optimizer};
use burn::record::CompactRecorder;
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{Distribution, Tensor};
use candle_core::Device as CandleDevice;
use rand::RngExt;
use std::collections::VecDeque;

use llm_scratch_rs::models::diffusion::dit::DiTConfig;
use llm_scratch_rs::models::diffusion::dit::DiffusionTransformer;
use llm_scratch_rs::utils::mnist_utils::{acquire_mnist, save_png};
const FASHION_CLASSES: [&str; 10] = [
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

// ============================================================================
// Heterogeneous Tensor Container for Module Traversal
// ============================================================================

/// Heterogeneous wrapper holding parameter tensors of varying dimensionality (1D to 4D).
///
/// ### Why is this needed?
/// In Burn, tensor dimensionality `D` is a compile-time const generic (`Tensor<B, D>`).
/// Neural network layers contain tensors of different ranks:
/// - **1D**: Biases and normalization scale/shift vectors ($\beta, \gamma$)
/// - **2D**: Linear projection weight matrices ($W_{\text{qkv}}, W_{\text{proj}}$)
/// - **3D**: Positional embeddings or sequence representations
/// - **4D**: Patch embedding convolution filters / spatial weights
///
/// To collect all parameters across the entire DiT model into a single ordered FIFO
/// queue (`VecDeque`), we wrap them in this enum.
pub enum AnyTensor<B: Backend> {
    D1(Tensor<B, 1>),
    D2(Tensor<B, 2>),
    D3(Tensor<B, 3>),
    D4(Tensor<B, 4>),
}
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

/// Performs one optimization step on a mini-batch.
///
/// ### Training Objective (DDPM / DiT):
/// Predict the noise $\epsilon \sim \mathcal{N}(0, \mathbf{I})$ that was added to clean image $x_0$:
/// $$\mathcal{L}_{\text{simple}} = \frac{1}{B} \sum_{i=1}^B \|\hat{\epsilon}_\theta(x_{t,i}, t_i, c_i) - \epsilon_i\|^2$$
///
/// # Arguments
/// * `model` - Active DiT model
/// * `x_t` - Noisy input images $[B, 1, 28, 28]$
/// * `target_noise` - Ground truth added Gaussian noise $\epsilon \sim \mathcal{N}(0, \mathbf{I})$
/// * `t_emb` - Sinusoidal timestep embeddings $[B, \text{hidden\_dim}]$
/// * `class_labels` - Conditioning class integer labels $[B]$
/// * `optimizer` - AdamW optimizer instance
/// * `lr` - Learning rate
///
/// # Returns
/// Updated model and scalar MSE loss value.
pub fn train_steps<B: AutodiffBackend>(
    model: DiffusionTransformer<B>,
    x_t: Tensor<B, 4>,
    target_noise: Tensor<B, 4>,
    t_emb: Tensor<B, 2>,
    class_labels: Tensor<B, 1, burn::tensor::Int>,
    optimizer: &mut impl Optimizer<DiffusionTransformer<B>, B>,
    lr: f64,
) -> (DiffusionTransformer<B>, f32) {
    // 1. Forward pass: predict noise added to x_t
    let pred_noise = model.forward(x_t, t_emb, class_labels);

    // 2. Compute Mean Squared Error (MSE) loss
    let diff = pred_noise - target_noise;
    let loss = diff.powf_scalar(2.0).mean();

    // Extract loss value as f32 for logging
    let loss_val = loss.clone().into_data().as_slice::<f32>().unwrap()[0];

    // 3. Backward pass: compute gradients w.r.t. all parameters
    let grads = GradientsParams::from_grads(loss.backward(), &model);

    // 4. Optimizer update: step weights using AdamW
    let model = optimizer.step(lr, model, grads);
    (model, loss_val)
}

/// Generates continuous sinusoidal positional embeddings for diffusion timesteps $t \in [0, T)$.
///
/// Follows the standard Transformer / DDPM frequency formulation:
/// $$\text{PE}(t, 2i) = \sin\left(\frac{t}{10000^{2i / d}}\right), \quad \text{PE}(t, 2i+1) = \cos\left(\frac{t}{10000^{2i / d}}\right)$$
///
/// # Arguments
/// * `timesteps` - Slice of integer timesteps for each batch item
/// * `dim` - Target embedding dimension ($D$)
/// * `device` - Compute device
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

/// Linear Variance Noise Scheduler for the forward diffusion process.
///
/// Computes pre-calculated variance schedules:
/// - $\beta_t \in [\beta_{\text{start}}, \beta_{\text{end}}]$ (linear schedule)
/// - $\alpha_t = 1 - \beta_t$
/// - $\bar{\alpha}_t = \prod_{s=0}^t \alpha_s$ (cumulative product `alphas_cumprod`)
/// - $\sqrt{\bar{\alpha}_t}$ (signal scaling factor)
/// - $\sqrt{1 - \bar{\alpha}_t}$ (noise scaling factor)
pub struct SimpleNoiseScheduler {
    pub alphas_cumprod: Vec<f32>,
    pub sqrt_alphas_cumprod: Vec<f32>,
    pub sqrt_one_minus_alphas_cumprod: Vec<f32>,
}

impl SimpleNoiseScheduler {
    /// Creates a linear beta schedule across discrete timesteps.
    pub fn new_linear(timesteps: usize, beta_start: f32, beta_end: f32) -> Self {
        let mut alphas_cumprod = Vec::with_capacity(timesteps);
        let mut cumprod = 1.0f32;

        for t in 0..timesteps {
            let beta = beta_start + (beta_end - beta_start) * (t as f32) / ((timesteps - 1) as f32);
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

    pub fn new_cosine(timesteps: usize) -> Self {
        let s = 0.008f32;
        let mut alphas_cumprod: Vec<f32> = Vec::with_capacity(timesteps);
        let f0 = ((s / (1.0 + s)) * (std::f32::consts::PI / 2.0))
            .cos()
            .powi(2);
        for t in 0..timesteps {
            let t_frac = t as f32 / timesteps as f32;

            let ft = (((t_frac + s) / (1.0 + s)) * (std::f32::consts::PI / 2.0))
                .cos()
                .powi(2);
            let alpha_bar = ft / f0;
            let alpha_bar = alpha_bar.clamp(0.0001, 0.9999);
            alphas_cumprod.push(alpha_bar);
        }

        let sqrt_alphas_cumprod: Vec<f32> = alphas_cumprod.iter().map(|&a| a.sqrt()).collect();
        let sqrt_one_minus_alphas_cumprod =
            alphas_cumprod.iter().map(|&a| (1.0 - a).sqrt()).collect();

        Self {
            alphas_cumprod,
            sqrt_alphas_cumprod,
            sqrt_one_minus_alphas_cumprod,
        }
    }
}

/// Forward Diffusion Process ($q$-sampling):
/// Directly noises clean image $x_0$ to arbitrary timestep $t$ in closed form:
///
/// $$q(x_t \mid x_0) = \mathcal{N}\left(x_t; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t) \mathbf{I}\right)$$
/// Using the reparameterization trick with $\epsilon \sim \mathcal{N}(0, \mathbf{I})$:
/// $$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$$
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

// ============================================================================
// Learning Rate Schedule: Linear Warmup + Cosine Decay
// ============================================================================

/// Calculates the learning rate for a given training step using Linear Warmup
/// followed by Cosine Annealing decay.
///
/// ### Schedule Stages:
/// 1. **Warmup Phase** (`step < warmup_steps`):
///    Ramps learning rate linearly from `lr_min` to `lr_max`:
///    $$\eta_t = \eta_{\min} + (\eta_{\max} - \eta_{\min}) \cdot \frac{t}{t_{\text{warmup}}}$$
///    *Why*: Self-attention heads in Vision Transformers produce high-variance gradients early on;
///    warming up prevents initial gradient spikes from destabilizing the weights.
///
/// 2. **Cosine Decay Phase** (`warmup_steps <= step <= total_steps`):
///    Smoothly decays learning rate along a cosine curve down to `lr_min`:
///    $$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min}) \left(1 + \cos\left(\frac{t - t_{\text{warmup}}}{t_{\text{total}} - t_{\text{warmup}}} \cdot \pi\right)\right)$$
///    *Why*: Anneals the step size smoothly into a flat local minimum without sharp drops.
pub fn get_learning_rate(
    step: usize,
    total_steps: usize,
    warmups_steps: usize,
    lr_max: f64,
    lr_min: f64,
) -> f64 {
    if step < warmups_steps {
        // Stage 1: Linear warmup from lr_min to lr_max
        lr_min + (lr_max - lr_min) * (step as f64 / warmups_steps.max(1) as f64)
    } else if step > total_steps {
        // Post-training: maintain minimum floor rate
        lr_min
    } else {
        // Stage 2: Smooth cosine decay down to lr_min
        let progress = (step - warmups_steps) as f64 / (total_steps - warmups_steps).max(1) as f64;
        let progress = progress.clamp(0.0, 1.0);
        lr_min + 0.5 * (lr_max - lr_min) * (1.0 + (std::f64::consts::PI * progress).cos())
    }
}

// ============================================================================
// Exponential Moving Average (EMA) Weight Management
// ============================================================================
//
// ### What is EMA?
// In generative diffusion models, the live model weights oscillate from batch to batch
// due to stochastic gradient noise. EMA maintains a "shadow" copy of the weights using
// Polyak averaging:
//
//     θ_ema ← β · θ_ema + (1 - β) · θ_live   (typically β = 0.999)
//
// Sampling images with EMA weights produces significantly higher visual quality,
// removes high-frequency speckle artifacts, and prevents mode collapse.
//
// ### Burn Implementation Architecture:
// 1. `ParamCollector` (Visitor Pattern): Traverses `live_model` and stores copies of all
//    parameter tensors into a FIFO queue (`VecDeque`).
// 2. `EmaBlender` (Mapper Pattern): Traverses `ema_model` in the identical order, pops
//    each live tensor, and blends it into the EMA tensor in-place.
// ============================================================================

/// **Step 1: Visitor** that traverses all parameters in the live model and collects them
/// into an ordered FIFO queue.
pub struct ParamCollector<B: Backend> {
    pub queue: VecDeque<AnyTensor<B>>,
}

impl<B: Backend> ModuleVisitor<B> for ParamCollector<B> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        // Extract a cloned tensor value (detached from autodiff graph)
        let val = param.val();
        let dims = val.dims();

        // Wrap the tensor in AnyTensor to erase const-generic rank D into a homogeneous queue
        match D {
            1 => self.queue.push_back(AnyTensor::D1(val.reshape([dims[0]]))),
            2 => self.queue.push_back(AnyTensor::D2(val.reshape([dims[0], dims[1]]))),
            3 => self.queue.push_back(AnyTensor::D3(val.reshape([dims[0], dims[1], dims[2]]))),
            4 => self.queue.push_back(AnyTensor::D4(val.reshape([dims[0], dims[1], dims[2], dims[3]]))),
            _ => {}
        }
    }
}

/// **Step 2: Mapper** that traverses the shadow EMA model and updates each parameter:
///
///     θ_ema = (θ_ema * decay) + (θ_live * (1 - decay))
pub struct EmaBlender<B: Backend> {
    /// EMA smoothing factor β (e.g. 0.999). Keeps 99.9% of past weight history.
    pub decay: f32,
    /// FIFO queue containing live model parameters collected by `ParamCollector`.
    pub live_queue: VecDeque<AnyTensor<B>>,
}

impl<B: Backend> ModuleMapper<B> for EmaBlender<B> {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        // Deconstruct the parameter into its ID, tensor data, and inner mapper state
        let (id, ema_tensor, mapper) = param.consume();
        let dims = ema_tensor.dims();

        // Pop the corresponding live tensor from the front of the queue
        if let Some(live_any) = self.live_queue.pop_front() {
            // Apply the EMA blend formula for matching rank D:
            // updated = ema * decay + live * (1 - decay)
            let updated = match (D, live_any) {
                (1, AnyTensor::D1(live)) => {
                    ema_tensor * self.decay + live.reshape(dims) * (1.0 - self.decay)
                }
                (2, AnyTensor::D2(live)) => {
                    ema_tensor * self.decay + live.reshape(dims) * (1.0 - self.decay)
                }
                (3, AnyTensor::D3(live)) => {
                    ema_tensor * self.decay + live.reshape(dims) * (1.0 - self.decay)
                }
                (4, AnyTensor::D4(live)) => {
                    ema_tensor * self.decay + live.reshape(dims) * (1.0 - self.decay)
                }
                _ => panic!("Shape/rank mismatch between live and EMA model parameter!"),
            };
            return Param::from_mapped_value(id, updated, mapper);
        }

        // Fallback: return unchanged if no matching parameter was found in queue
        Param::from_mapped_value(id, ema_tensor, mapper)
    }
}

/// Coordinates one EMA step: transfers live model weights to shadow EMA weights.
///
/// # Arguments
/// * `ema_model` - Current shadow EMA model (consumed and transformed)
/// * `live_model` - Active training model being optimized by AdamW
/// * `decay` - EMA decay rate $\beta$ (e.g. 0.999)
///
/// # Returns
/// The updated `DiffusionTransformer` with blended shadow weights.
pub fn update_ema<B: Backend>(
    ema_model: DiffusionTransformer<B>,
    live_model: &DiffusionTransformer<B>,
    decay: f32,
) -> DiffusionTransformer<B> {
    // 1. Extract all live parameter tensors into a queue
    let mut collector = ParamCollector {
        queue: VecDeque::new(),
    };
    live_model.visit(&mut collector);

    // 2. Map and blend each parameter into the EMA shadow model
    let mut blender = EmaBlender {
        decay,
        live_queue: collector.queue,
    };
    ema_model.map(&mut blender)
}

/// Deterministic Denoising Diffusion Implicit Models (DDIM) Reverse Sampling.
///
/// Generates images starting from pure Gaussian noise $x_T \sim \mathcal{N}(0, \mathbf{I})$
/// by iteratively stepping backwards along a deterministic trajectory.
///
/// ### At each step $t \to t_{\text{prev}}$:
/// 1. Predict clean image estimate:
///    $$\hat{x}_0 = \frac{x_t - \sqrt{1 - \bar{\alpha}_t} \hat{\epsilon}}{\sqrt{\bar{\alpha}_t}}$$
/// 2. Direction pointing towards $x_t$:
///    $$\text{dir} = \sqrt{1 - \bar{\alpha}_{t_{\text{prev}}}} \hat{\epsilon}$$
/// 3. Assemble previous sample:
///    $$x_{t_{\text{prev}}} = \sqrt{\bar{\alpha}_{t_{\text{prev}}}} \hat{x}_0 + \text{dir}$$
pub fn sample_ddim<B: Backend>(
    model: &DiffusionTransformer<B>,
    scheduler: &SimpleNoiseScheduler,
    batch_size: usize,
    class_label: usize,
    steps: usize,
    hidden_dim: usize,
    device: &B::Device,
) -> Tensor<B, 4> {
    // 1. Start from pure random Gaussian noise x_T ~ N(0, I)
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

        // Predict clean image estimate x_0
        let pred_x0 = (x - pred_noise.clone() * (1.0 - alpha_bar_t).sqrt()) / alpha_bar_t.sqrt();

        // Direction pointing to x_{t_prev}
        let dir = pred_noise * (1.0 - alpha_bar_prev).sqrt();

        // Assemble x_{t-prev}
        x = pred_x0 * alpha_bar_prev.sqrt() + dir;
    }

    x
}

/// Stitches 10 28x28 images into a single 2x5 lookbook image and saves as PNG.
pub fn save_lookbook_collage(path: &str, images: &[Vec<f32>]) -> anyhow::Result<()> {
    if images.len() != 10 {
        anyhow::bail!(
            "Expected 10 images for a 2x5 lookbook, got {}",
            images.len()
        );
    }
    let rows = 2;
    let cols = 5;
    let side = 28;
    let total_w = cols * side; // 140 pixels
    let total_h = rows * side; // 56 pixels
    let mut collage = vec![0.0f32; total_w * total_h];
    for (k, img) in images.iter().enumerate() {
        let r = k / cols; // Row: 0 or 1
        let c = k % cols; // Col: 0..4
        for y in 0..side {
            for x in 0..side {
                let dst_y = r * side + y;
                let dst_x = c * side + x;
                collage[dst_y * total_w + dst_x] = img[y * side + x];
            }
        }
    }
    // Convert from [-1, 1] to [0, 255]
    let bytes: Vec<u8> = collage
        .iter()
        .map(|&v| (((v + 1.0) / 2.0).clamp(0.0, 1.0) * 255.0).round() as u8)
        .collect();
    llm_scratch_rs::utils::ensure_parent_dir(path)?;
    let file = std::fs::File::create(path)?;
    let writer = std::io::BufWriter::new(file);
    let mut encoder = png::Encoder::new(writer, total_w as u32, total_h as u32);
    encoder.set_color(png::ColorType::Grayscale);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header()?;
    writer.write_image_data(&bytes)?;
    println!("Saved lookbook collage to: {path}");
    Ok(())
}
pub fn sample_ddim_with_cfg<B: Backend>(
    model: &DiffusionTransformer<B>,
    scheduler: &SimpleNoiseScheduler,
    batch_size: usize,
    class_label: usize,
    steps: usize,
    guidance_scale: f32, // e.g., 3.0
    hidden_dim: usize,
    device: &B::Device,
) -> Tensor<B, 4> {
    // 1. Start from pure random Gaussian noise x_T ~ N(0, I)
    let mut x = Tensor::random(
        [batch_size, 1, 28, 28],
        Distribution::Normal(0.0, 1.0),
        device,
    );

    // Prepare label vectors: class_label for conditional, 10 for unconditional
    let mut labels_vec = vec![class_label as i32; batch_size];
    labels_vec.extend(vec![10; batch_size]); //null class for unconditional generation

    let class_labels = Tensor::from_ints(labels_vec.as_slice(), device);

    let total_steps = scheduler.alphas_cumprod.len();
    let step_size = total_steps / steps;
    let timesteps: Vec<usize> = (0..steps).map(|i| (steps - 1 - i) * step_size).collect();

    for (i, &t) in timesteps.iter().enumerate() {
        // concate x with iteself
        let x_in = Tensor::cat(vec![x.clone(), x.clone()], 0);
        // 1. Pass doubled batch size for t_emb
        let t_emb = get_time_step_embeddings::<B>(&vec![t; batch_size * 2], hidden_dim, device);

        // 1. Predict with CLASS label
        let pred_both = model.forward(x_in, t_emb, class_labels.clone());
        // pred conditonal
        let pred_cond = pred_both.clone().slice([0..batch_size, 0..1, 0..28, 0..28]);
        // cfg formula eps =eps_uncond + s*(epc_cond - eps_uncond)
        let pred_uncond = pred_both.slice([batch_size..batch_size * 2, 0..1, 0..28, 0..28]);

        // cfg formula eps =eps_uncond + s*(epc_cond - eps_uncond)
        let pred_noise = pred_uncond.clone() + (pred_cond - pred_uncond) * guidance_scale;

        let alpha_bar_t = scheduler.alphas_cumprod[t];
        let alpha_bar_prev = if i + 1 < timesteps.len() {
            scheduler.alphas_cumprod[timesteps[i + 1]]
        } else {
            1.0
        };

        // 2. Predict clean estimate AND CLAMP to [-1.0, 1.0]
        let pred_x0 = (x - pred_noise.clone() * (1.0 - alpha_bar_t).sqrt()) / alpha_bar_t.sqrt();

        let pred_x0 = pred_x0.clamp(-1.0, 1.0); // <-- Eliminates the speckle noise!
                                                // Direction pointing to x_{t_prev}
        let dir = pred_noise * (1.0 - alpha_bar_prev).sqrt();

        // Assemble x_{t-prev}
        x = pred_x0 * alpha_bar_prev.sqrt() + dir;
    }

    x
}

pub fn main() -> anyhow::Result<()> {
    let device = Default::default();
    #[cfg(feature = "rocm")]
    println!("Initializing DiT (Diffusion Transformer) on Native ROCm (AMD HIP)...");
    #[cfg(all(not(feature = "rocm"), not(feature = "cpu")))]
    println!("Initializing DiT (Diffusion Transformer) on WGPU (Vulkan / GPU)...");
    #[cfg(feature = "cpu")]
    println!("Initializing DiT (Diffusion Transformer) on CPU (NdArray)...");

    let config = DiTConfig {
        img_size: 28,
        in_channels: 1,
        patch_size: 2,
        num_classes: 11, //-> cfg thing with null token class
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

    // 2. Load MNIST Dataset (60k images normalized to [-1, 1])
    println!("Loading MNIST dataset...");
    let (candle_images, labels_vec) = acquire_mnist(&CandleDevice::Cpu)?;
    let images_flat = candle_images.to_vec2::<f32>()?; // 60,000 x 784
    let total_samples = images_flat.len();
    println!("Successfully loaded {} MNIST samples.", total_samples);

    let batch_size = 64;
    // Increased total training steps (e.g. 5,000 steps) with checkpoints every 1,000 steps
    let num_steps = std::env::var("STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(25000);
    let checkpoint_interval = 1000;

    println!(
        "Starting DiT training for {} steps with batch size {} (saving checkpoint tensors every {} steps)...",
        num_steps, batch_size, checkpoint_interval
    );

    // Ensure output directories exist
    std::fs::create_dir_all("checkpoints")?;
    std::fs::create_dir_all("samples")?;

    let mut rng = rand::rng();
    for step in 1..=num_steps {
        // --- Step A: Sample Random Batch ---
        let mut batch_images = Vec::with_capacity(batch_size * 784);
        let mut batch_labels = Vec::with_capacity(batch_size);
        for _ in 0..batch_size {
            let idx = rng.random_range(0..total_samples);
            batch_images.extend_from_slice(&images_flat[idx]);
            if rng.random_range(0.0..1.0) < 0.15 {
                batch_labels.push(10); // null class
            } else {
                batch_labels.push(labels_vec[idx] as i32); // 0..9 classes
            }
        }
        // Convert to Burn Tensors: [B, 1, 28, 28] and [B]
        let x_0: Tensor<MyAutodiffBackend, 4> =
            Tensor::<MyAutodiffBackend, 1>::from_floats(batch_images.as_slice(), &device)
                .reshape([batch_size, 1, 28, 28]);
        let class_labels: Tensor<MyAutodiffBackend, 1, burn::tensor::Int> =
            Tensor::from_ints(batch_labels.as_slice(), &device);

        // --- Step B: Sample Random Timesteps t in [0, 100) ---
        let timesteps: Vec<usize> = (0..batch_size)
            .map(|_| rng.random_range(0..num_timesteps))
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

        // --- Step F: Optimization Step with Warmup + Cosine Decay ---
        let current_lr = get_learning_rate(
            step, num_steps, 500,  /* warmup */
            2e-4, /* max */
            1e-5, /* min */
        );
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

        // Step EMA weights with 0.999 decay rate
        ema_model = update_ema(ema_model, &model.valid(), 0.999);
        // Logging every 50 steps
        if step % 50 == 0 || step == 1 {
            println!("Step {:5}/{}: MSE Loss = {:.6}", step, num_steps, loss_val);
        }

        // --- Step G: Checkpoint Tensors & Sample Images Every 1,000 Steps ---
        if step % checkpoint_interval == 0 || step == num_steps {
            // println!(
            //     "\n>>> [Step {}/{}] Saving Model Tensors & Generating Digit Previews...",
            //     step, num_steps
            // );
            // println!(
            //     "\n>>> [Step {}/{}] Saving Model Tensors & Generating Fashion Previews...",
            //     step, num_steps
            // );

            println!(
                "\n>>> [Step {}/{}] Generating EMA Lookbook & Saving Checkpoint...",
                step, num_steps
            );
            // 1. Convert to validation mode for evaluation and saving
            let valid_model = ema_model.clone();

            // 2. Save Model Tensor Weights (Compact format)
            // let checkpoint_path = format!("checkpoints/dit_mnist_step_{:05}", step);
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

            // 3. Generate preview samples for all 10 fashion classes using EMA model
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

            // 4. Save 2x5 stitched lookbook collage
            let lookbook_path = format!("samples/dit_step_{:05}_lookbook.png", step);
            if let Err(e) = save_lookbook_collage(&lookbook_path, &class_samples) {
                eprintln!("Warning: Failed to save lookbook {}: {:?}", lookbook_path, e);
            }

            // 5. Also save individual class images
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
