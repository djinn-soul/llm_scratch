// =============================================================================
// train_diffusion_cnn_5layers.rs — Deeper 5-layer CNN CFG-DDPM trainer
// =============================================================================
//
// This binary is the natural evolution of `train_diffusion_cnn.rs`:
//   • Same CFG training strategy (15% label dropout, guidance scale sweep)
//   • Same DDPM noise schedule and Adam optimizer
//   • Replaced model: SimpleDenoisingCNN5Layers instead of SimpleDenoisingCNN
//
// WHY a deeper model?
//   `SimpleDenoisingCNN` (2 layers, 3×3 kernels, 16/1 channels) has a
//   receptive field of only 5×5 pixels after 2 conv layers.  It can learn
//   local texture but struggles with the global digit strokes that span a
//   significant portion of a 28×28 image.
//
//   `SimpleDenoisingCNN5Layers` (5 layers, 5×5 kernels, 64/128/128/64/1
//   channels) has a receptive field of ~21×21 after 5 layers, covering the
//   full digit area while still operating efficiently on 28×28 inputs.
//
// Key differences from train_diffusion_cnn.rs:
//   1. Model: SimpleDenoisingCNN5Layers (12 parameters, richer capacity).
//   2. Device: auto-selects CUDA if available, falls back to CPU.
//   3. Batch size: 256 on GPU (better utilisation), 128 on CPU (memory safe).
//   4. Timing: wall-clock speed logged at every 100-epoch checkpoint.
//   5. Output files: separate png names (mnist_cfg_5layers_generated_*.png).
//
// Training strategy (CFG with label dropout) — same as train_diffusion_cnn.rs:
//   • 15% of labels are zeroed per sample, training both conditional and
//     unconditional denoising branches in one network.
//   • At inference, CFG blends the two predictions:
//       ε̂_guided = ε_cond + s * (ε_cond − ε_uncond)
//   • Three scales are saved: s=0 (uncond), s=1 (cond), s=3 (guided).
//
// Output images:
//   mnist_cfg_5layers_generated_s0.png — unconditional baseline
//   mnist_cfg_5layers_generated_s1.png — standard conditional
//   mnist_cfg_5layers_generated_s3.png — CFG-amplified (best class fidelity)
// =============================================================================

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use rand::RngExt;
use std::time::Instant;

// CFG-aware reverse diffusion sampler — shared with the 2-layer CNN binary.
use llm_scratch_rs::models::diffusion::sampling::sample_ddpm_cfg;

// 5-layer CNN model components.
use llm_scratch_rs::models::diffusion::{
    get_time_embedding, BetaScheduler, DenoisingModel, MlpAdamOptimizer, SimpleDenoisingCNN5Layers,
};

// Shared MNIST I/O utilities.
use llm_scratch_rs::utils::mnist_utils::{acquire_mnist, save_png};

// Use mimalloc for faster heap allocation during training (same rationale as
// train_diffusion_cnn.rs: many small tensors are created and destroyed each step).
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

// =============================================================================
// save_cfg_sample — helper: one guided image at a given guidance scale
// =============================================================================
//
// WHY a separate helper?
//   We generate 3 images at different guidance scales after training.
//   Extracting the logic avoids repeating noise tensor construction,
//   `sample_ddpm_cfg` invocation, and PNG saving three times.
//
// Flow:
//   1. Sample x_T ~ N(0, I) — the pure noise starting point.
//   2. Run T-step CFG reverse diffusion with the trained CNN.
//   3. Flatten and save the result as a 28×28 PNG.
//
// Arguments:
//   model          — trained `SimpleDenoisingCNN5Layers` (via DenoisingModel)
//   scheduler      — pre-computed DDPM noise schedule
//   target_one_hot — one-hot conditioning vector (1, 10) for the target digit
//   guidance_scale — CFG strength: 0=uncond, 1=cond, 3+=guided
//   img_dim        — 784 (28×28 MNIST)
//   time_emb_dim   — 16 (sinusoidal time embedding size)
//   filename       — output PNG path
//   device         — CPU or CUDA
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
    // x_T ~ N(0, I): pure Gaussian noise, shape (1, 784).
    let initial_noise = Tensor::randn(0.0f32, 1.0f32, (1, img_dim), device)?;

    // CFG-guided reverse diffusion: T steps, two forward passes per step.
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

    // Flatten (1, 784) → Vec<f32> and write as a 28×28 grayscale PNG.
    let final_pixels = generated.flatten_all()?.to_vec1::<f32>()?;
    save_png(filename, &final_pixels)?;
    Ok(())
}

// =============================================================================
// make_one_hot_cfg — one-hot encoder with stochastic CFG label dropout
// =============================================================================
//
// Identical to the version in `train_diffusion_cnn.rs`.  Each label is
// zeroed independently with probability `drop_rate` (15% during training,
// 0% during inference).  All-zeros rows represent the "null class" ∅ used
// for the unconditional branch of CFG guidance.
//
// Arguments:
//   labels    — slice of u8 digit labels in {0, …, 9}
//   drop_rate — label dropout probability (0.15 for CFG training, 0.0 for inference)
//   device    — tensor device
fn make_one_hot_cfg(labels: &[u8], drop_rate: f32, device: &Device) -> Result<Tensor> {
    let n = labels.len();
    let num_classes = 10;
    let mut rng = rand::rng();
    let mut hot = vec![0.0f32; n * num_classes];

    for (i, &label) in labels.iter().enumerate() {
        // Keep the label with probability (1 - drop_rate); otherwise leave as zeros.
        if rng.random::<f32>() > drop_rate {
            let idx = (i * num_classes) + label as usize;
            hot[idx] = 1.0f32;
        }
    }
    Ok(Tensor::from_vec(hot, (n, num_classes), device)?)
}

// =============================================================================
// main — 5-layer CNN CFG training loop + multi-scale image generation
// =============================================================================
pub fn main() -> Result<()> {
    // --- Device auto-selection -----------------------------------------------
    // Attempt to use CUDA GPU 0; fall back silently to CPU if unavailable.
    // WHY try CUDA first? The 5-layer CNN with 128-channel conv layers has
    // significantly more parameters than the 2-layer version.  GPU matmul
    // throughput is 10–50× faster for the large (128, 128, 5, 5) kernels.
    let device = Device::new_cuda(0).unwrap_or(Device::Cpu);
    println!("Active Device: {:?}", device);

    // --- Device-adaptive batch size ------------------------------------------
    // GPU VRAM allows larger batches, reducing gradient variance per update.
    // CPU memory is more constrained: 128 × 784 × float × 128 features ≈ 50 MB.
    let batch_size = match device {
        Device::Cuda(_) => 256, // CUDA: larger batch → better GPU utilisation
        _               => 128, // CPU: smaller batch → fits in RAM
    };
    println!("Selected Batch Size: {}", batch_size);

    // =========================================================================
    // Hyper-parameters
    // =========================================================================
    //
    // epochs       — 20 000 steps.  The 5-layer model converges slower than
    //                the 2-layer variant because it has more parameters and a
    //                deeper gradient path, requiring more updates to settle.
    //
    // lr           — 0.001 (Adam default).
    //
    // img_dim      — 784 (28×28 MNIST, flattened).
    //
    // class_dim    — 10 (one-hot digit label size).
    //
    // time_emb_dim — 16 (sinusoidal time embedding, must match sampling calls).
    //
    // steps (T)    — 100 diffusion timesteps.
    //
    // label_dropout— 0.15: drops 15% of labels per step to enable CFG.
    //
    // cond_dim     — time_emb_dim + class_dim = 26.
    //                The CNN's first layer projects this into a 784-dim spatial
    //                conditioning map.
    let epochs        = 20000;
    let lr            = 0.001f64;
    let img_dim       = 784;
    let class_dim     = 10;
    let time_emb_dim  = 16;
    let steps         = 100;
    let label_dropout = 0.15f32;
    let cond_dim      = time_emb_dim + class_dim; // 26

    // --- Dataset loading -----------------------------------------------------
    // `acquire_mnist` downloads and parses MNIST on first run (cached after).
    // images: (60000, 784) in [-1, 1]; train_labels: 60000 u8 in {0..9}.
    println!("Loading MNIST dataset...");
    let (images, train_labels) = acquire_mnist(&device)?;
    let (total_samples, _) = images.dims2()?;

    // --- Noise schedule -------------------------------------------------------
    // Linear beta schedule from 0.0001 → 0.02 over 100 steps (DDPM paper).
    let scheduler = BetaScheduler::new(steps, 0.0001, 0.02, &device)?;

    // --- Model initialisation -------------------------------------------------
    // 5-layer encoder-decoder CNN with He-initialised weights.
    // Total trainable parameters (rough count):
    //   w_cond:  784 × 26         ≈  20K
    //   w1-w5: 64×2×25 + 128×64×25 + 128×128×25 + 64×128×25 + 1×64×25
    //         ≈ 3K + 205K + 410K + 205K + 2K ≈ 825K
    //   biases: negligible
    //   Total: ~845K parameters
    let mut cnn = SimpleDenoisingCNN5Layers::new(img_dim, cond_dim, &device)?;

    // --- Adam optimizer -------------------------------------------------------
    // Generic optimizer: works with any DenoisingModel via params()/params_mut().
    let mut optimizer = MlpAdamOptimizer::new(&cnn, lr)?;

    println!("Starting 5-layer CNN training for {} epochs...\n", epochs);

    // Start wall-clock timer for training speed logging.
    let start_time = Instant::now();

    // =========================================================================
    // Training loop — CFG noise prediction with label dropout
    // =========================================================================
    //
    // Identical structure to train_diffusion_cnn.rs.  The only model-specific
    // differences are:
    //   - The DenoisingModel forward/backward dispatch uses SimpleDenoisingCNN5Layers.
    //   - The gradient vector has 12 entries (vs 6 for the 2-layer model).
    //   - The logging includes elapsed time and steps-per-second.
    for epoch in 1..=epochs {
        // --- Step 1: Random mini-batch ----------------------------------------
        // Sample `batch_size` random image indices uniformly from [0, N).
        let index_tensor = Tensor::rand(
            0.0f32,
            total_samples as f32 - 1e-4,
            (batch_size,),
            &device,
        )?
        .to_dtype(DType::U32)?;

        let indices = index_tensor.to_vec1::<u32>()?;
        let x0 = images.index_select(&index_tensor, 0)?;

        // Gather labels aligned with the selected images.
        let batch_labels: Vec<u8> = indices.iter().map(|&x| train_labels[x as usize]).collect();

        // --- Step 2: CFG label dropout ----------------------------------------
        // Each label independently zeroed with probability label_dropout (15%).
        let label_one_hot = make_one_hot_cfg(&batch_labels, label_dropout, &device)?;

        // --- Step 3: Forward diffusion (corrupt x_0 → x_t) ------------------
        // Random timestep t ∈ {0, …, T-1} per example.
        // x_t = sqrt(ᾱ_t) * x_0 + sqrt(1 - ᾱ_t) * ε
        let t_float = Tensor::rand(0.0f32, steps as f32 - 1e-4, (batch_size,), &device)?;
        let t = t_float.to_dtype(DType::U32)?;
        let noise = Tensor::randn(0.0f32, 1.0f32, (batch_size, img_dim), &device)?;
        let xt = scheduler.add_noise(&x0, &noise, &t)?;

        // --- Step 4: Build the conditioned model input -----------------------
        // v = concat(x_t, time_emb, label_one_hot)
        // shape: (batch_size, 784 + 16 + 10) = (batch_size, 810)
        let time_emb = get_time_embedding(&t, time_emb_dim)?;
        let v = Tensor::cat(&[&xt, &time_emb, &label_one_hot], 1)?;

        // --- Step 5: Forward pass + MSE loss ---------------------------------
        // The CNN splits v into image and conditioning, runs 5 conv layers,
        // and returns the predicted noise and cached intermediates.
        let (pred, intermediates) = DenoisingModel::forward(&cnn, &v)?;
        let diff = pred.sub(&noise)?;
        let loss = diff.sqr()?.mean_all()?.to_scalar::<f32>()?;

        // --- Step 6: Backward pass -------------------------------------------
        // Manual chain-rule through all 5 conv layers + conditioning projection.
        // Returns 12 gradient tensors matching the order in params().
        let grads = DenoisingModel::backward(&cnn, &v, &intermediates, &pred, &noise)?;

        // --- Step 7: Adam optimizer step -------------------------------------
        optimizer.step(&mut cnn, &grads)?;

        // --- Step 8: Periodic logging ----------------------------------------
        // Log every 100 steps and at step 1 (to confirm training started).
        // Includes: epoch, loss, training speed (steps/s), elapsed time,
        // and L2 gradient norms for each of the 12 parameter tensors.
        if epoch % 100 == 0 || epoch == 1 {
            let elapsed = start_time.elapsed().as_secs_f32();
            let speed   = epoch as f32 / elapsed;  // steps per second

            // Compute L2 norm for each gradient tensor.
            let param_names = cnn.param_names();
            let grad_norms: Vec<f32> = grads
                .iter()
                .map(|g| -> Result<f32> { Ok(g.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt()) })
                .collect::<Result<_>>()?;

            let norms_str: Vec<String> = param_names
                .iter()
                .zip(grad_norms.iter())
                .map(|(name, norm)| format!("{} norm: {:.4}", name, norm))
                .collect();

            println!(
                "Epoch {:5}/{} - MSE Loss: {:.6} | Speed: {:.1} steps/s | Elapsed: {:.1}s | {}",
                epoch, epochs, loss, speed, elapsed, norms_str.join(", ")
            );
        }
    }

    // =========================================================================
    // CFG-guided image generation at three guidance scales
    // =========================================================================
    //
    // After training we generate three images for digit "3" at guidance scales
    // s=0, s=1, s=3.  Compare these PNGs to evaluate:
    //   • s=0: unconditional baseline — is the model well-trained enough to
    //          produce coherent MNIST-like images without class guidance?
    //   • s=1: conditional baseline — how well has the model learned digit 3?
    //   • s=3: CFG-amplified — does stronger guidance sharpen digit fidelity
    //          at the cost of sample diversity?
    println!("\n=== Starting Classifier-Free Guided Reverse Sampling ===");

    // Build the one-hot target for digit "3" (drop_rate=0 at inference).
    let mut target_vec = vec![0.0f32; class_dim];
    target_vec[3] = 1.0f32; // digit index 3
    let target_one_hot = Tensor::from_vec(target_vec, (1, class_dim), &device)?;

    // s = 0.0: unconditional — guidance term is zero, class label ignored.
    save_cfg_sample(
        &cnn,
        &scheduler,
        &target_one_hot,
        0.0,
        img_dim,
        time_emb_dim,
        "mnist_cfg_5layers_generated_s0.png",
        &device,
    )?;

    // s = 1.0: standard conditional — pure class-conditioned (no extrapolation).
    save_cfg_sample(
        &cnn,
        &scheduler,
        &target_one_hot,
        1.0,
        img_dim,
        time_emb_dim,
        "mnist_cfg_5layers_generated_s1.png",
        &device,
    )?;

    // s = 3.0: CFG-boosted — amplified class signal for stronger fidelity.
    save_cfg_sample(
        &cnn,
        &scheduler,
        &target_one_hot,
        3.0,
        img_dim,
        time_emb_dim,
        "mnist_cfg_5layers_generated_s3.png",
        &device,
    )?;

    println!("Generated 5-layer CNN images saved successfully.");
    Ok(())
}
