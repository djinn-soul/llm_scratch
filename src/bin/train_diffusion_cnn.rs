// =============================================================================
// train_diffusion_cnn.rs — CNN-backed Classifier-Free Guidance DDPM trainer
// =============================================================================
//
// This binary replaces the MLP denoiser from `train_diffusion_cfg.rs` with a
// small convolutional neural network (`SimpleDenoisingCNN`).  Everything else
// — the DDPM forward/reverse process, CFG training strategy, and guided
// sampling — is identical.
//
// WHY switch from MLP to CNN?
//   An MLP treats the flattened 784-pixel image as a 1-D vector and loses all
//   spatial structure.  A CNN preserves the 2-D grid (28×28) and processes
//   neighbouring pixels together through its kernel windows.  This gives the
//   model a natural inductive bias for images: local patterns (edges, strokes)
//   can be detected and reconstructed more efficiently than by an MLP.
//
// Architecture overview (SimpleDenoisingCNN):
//   1. Conditioning projection: Linear(cond_dim=26, img_dim=784) → reshaped to
//      (B, 1, 28, 28).  Projects the 26-dim conditioning vector (time_emb 16 +
//      class_one_hot 10) into a spatial "guidance map" the same size as the image.
//   2. Channel cat: concatenate [noisy image, conditioning map] → (B, 2, 28, 28)
//   3. Conv1: Conv2d(in=2, out=16, kernel=3×3, pad=1) + Leaky-ReLU(0.01)
//   4. Conv2: Conv2d(in=16, out=1, kernel=3×3, pad=1) → reshape to (B, 784)
//
// CFG training strategy (label dropout):
//   Same as `train_diffusion_cfg.rs`.  Each label is independently zeroed with
//   probability `label_dropout` (15%).  This teaches the model both the
//   conditional and unconditional noise distributions simultaneously.
//
// CFG sampling strategy (guidance scale sweep):
//   After training, three images are generated at guidance scales 0, 1, and 3
//   so the effect of guidance strength can be visually compared:
//     s=0 → unconditional sampling (label ignored at inference)
//     s=1 → standard conditional sampling (no extrapolation)
//     s=3 → CFG-boosted: class signal amplified 3×
//
// Memory allocator:
//   `mimalloc` replaces the system allocator for faster small-allocation
//   throughput, which is important when building many small tensors per step.
//
// Paper reference: "Classifier-Free Diffusion Guidance" (Ho & Salimans, 2022)
//   https://arxiv.org/abs/2207.12598
// =============================================================================

use anyhow::Result;
use candle_core::{DType, Device, Tensor};

// `sample_ddpm_cfg` — shared CFG-aware reverse diffusion sampler.
// Imported from the sampling submodule because it requires the generic
// `DenoisingModel` trait, added after the top-level re-export was frozen.
use llm_scratch_rs::models::diffusion::sampling::sample_ddpm_cfg;

// Model components:
//   get_time_embedding  — sinusoidal timestep encoding
//   BetaScheduler       — pre-computes beta/alpha/sigma schedule
//   DenoisingModel      — shared trait for forward/backward/params
//   MlpAdamOptimizer    — generic Adam optimizer (works with any DenoisingModel)
//   SimpleDenoisingCNN  — 2-layer CNN denoiser
use llm_scratch_rs::models::diffusion::{
    get_time_embedding, BetaScheduler, DenoisingModel, MlpAdamOptimizer, SimpleDenoisingCNN,
};

// Shared MNIST dataset loader and PNG writer.
use llm_scratch_rs::utils::mnist_utils::{acquire_mnist, save_png};

// `rand::RngExt` provides `.random::<T>()` on the thread-local RNG, used
// to decide whether to drop each class label during training.
use rand::RngExt;

// =============================================================================
// Global memory allocator: mimalloc
// =============================================================================
//
// WHY replace the system allocator?
//   Training creates and destroys thousands of small temporary tensors per step.
//   mimalloc is optimised for this allocation pattern: it uses thread-local
//   heaps and avoids global lock contention, giving measurably lower allocation
//   overhead on multi-producer workloads.
//
// The `#[global_allocator]` attribute tells the Rust runtime to use
// `mimalloc::MiMalloc` for every heap allocation in this binary.
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

// =============================================================================
// save_cfg_sample — helper: generate one image at a given guidance scale
// =============================================================================
//
// WHY a standalone helper?
//   After training we call this three times with different `guidance_scale`
//   values (0, 1, 3).  Extracting to a function avoids repeating the boilerplate
//   noise tensor construction and PNG save call.
//
// Algorithm:
//   1. Sample x_T ~ N(0, I) as the starting noise.
//   2. Run `sample_ddpm_cfg` to denoise over T steps with CFG.
//   3. Flatten x_0 and write as a 28×28 grayscale PNG.
//
// Arguments:
//   model          — the trained CNN (as a `DenoisingModel` trait object)
//   scheduler      — pre-computed noise schedule
//   target_one_hot — class conditioning vector, shape (1, 10)
//   guidance_scale — CFG guidance strength (0 = unconditional, 3+ = guided)
//   img_dim        — 784 for MNIST
//   time_emb_dim   — 16 (sinusoidal embedding dimension)
//   filename       — output PNG file path
//   device         — CPU or GPU
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
    // Start from pure Gaussian noise: x_T ~ N(0, I), shape (1, 784).
    let initial_noise = Tensor::randn(0.0f32, 1.0f32, (1, img_dim), device)?;

    // Run T steps of CFG-guided reverse diffusion.
    // The model is called twice per step (once conditioned, once unconditioned)
    // and the outputs are blended using the guidance formula:
    //   eps_guided = eps_cond + scale * (eps_cond - eps_uncond)
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

    // Flatten the (1, 784) tensor to a 784-element Vec<f32>.
    // `save_png` maps [-1, 1] → [0, 255] and writes a 28×28 grayscale PNG.
    let final_pixels = generated.flatten_all()?.to_vec1::<f32>()?;
    save_png(filename, &final_pixels)?;
    Ok(())
}

// =============================================================================
// make_one_hot_cfg — one-hot encoder with stochastic CFG label dropout
// =============================================================================
//
// WHY label dropout?
//   CFG requires the model to learn both:
//     - ε_θ(x_t, t, c)   when a class label c is provided
//     - ε_θ(x_t, t, ∅)   when no label is provided (null / unconditional)
//
//   By randomly zeroing out each label with probability `drop_rate`, training
//   examples are split between the conditional and unconditional regimes.
//   After training, guidance interpolates between these two predictions.
//
// WHY all-zeros for the null label?
//   The model sees an all-zeros class vector for dropped samples, so it learns
//   to treat all-zeros as the "no label" signal.  At inference, the same
//   all-zeros vector is passed to obtain the unconditional noise prediction.
//
// Arguments:
//   labels    — slice of u8 digit labels in {0, …, 9}
//   drop_rate — probability that each label is zeroed (0.15 = 15%)
//   device    — tensor target device
//
// Returns: Tensor of shape (labels.len(), 10) with values in {0.0, 1.0}
fn make_one_hot_cfg(labels: &[u8], drop_rate: f32, device: &Device) -> Result<Tensor> {
    let n = labels.len();
    let num_classes = 10;

    // Initialise the RNG from thread-local state (cheap, no global lock).
    let mut rng = rand::rng();

    // Start with a flat all-zeros buffer representing the null class for all rows.
    let mut hot = vec![0.0f32; n * num_classes];

    for (i, &label) in labels.iter().enumerate() {
        // `rng.random::<f32>()` draws from uniform [0, 1).
        // We keep the label when the draw exceeds `drop_rate`, i.e. with
        // probability (1 - drop_rate).  At drop_rate=0.15 this keeps ~85% of
        // labels intact and zeroes out ~15%.
        if rng.random::<f32>() > drop_rate {
            // The flat index for row i, column label.
            let idx = (i * num_classes) + label as usize;
            hot[idx] = 1.0f32;
        }
        // If the draw ≤ drop_rate: row stays all-zeros (null label).
    }

    // Build the 2-D one-hot tensor from the flat buffer.
    Ok(Tensor::from_vec(hot, (n, num_classes), device)?)
}

// =============================================================================
// main — CNN-based CFG-DDPM training loop + multi-scale image generation
// =============================================================================
//
// High-level flow:
//   1. Load MNIST images and labels.
//   2. Configure hyper-parameters.
//   3. Build the beta schedule, CNN model, and Adam optimizer.
//   4. Run the training loop: forward-diffuse → predict noise → backprop → Adam.
//   5. After training, generate one image at each of three guidance scales
//      and save them as separate PNGs for visual comparison.
// =============================================================================
pub fn main() -> Result<()> {
    // --- Device selection ----------------------------------------------------
    // Automatically selects CUDA GPU if available, otherwise defaults to CPU.
    let device = Device::new_cuda(0).unwrap_or(Device::Cpu);
    println!("Active Device: {:?}", device);

    let batch_size = match device {
        Device::Cuda(_) => 256,
        _ => 128,
    };
    println!("Selected Batch Size: {}", batch_size);

    // =========================================================================
    // Hyper-parameter configuration
    // =========================================================================
    //
    // epochs       — 20 000 gradient steps.  More than the MLP variants because
    //                convolutions have fewer parameters per channel but the
    //                gradients flow through 2D spatial operations that need more
    //                iterations to converge from random initialisation.
    //
    // lr           — 0.001: standard Adam learning rate.
    //
    // batch_size   — 128: balances gradient variance and memory usage.
    //
    // img_dim      — 784: 28×28 MNIST images flattened for I/O, then internally
    //                reshaped to (B, 1, 28, 28) inside the CNN forward pass.
    //
    // class_dim    — 10: one-hot label size (MNIST digits 0–9).
    //
    // time_emb_dim — 16: sinusoidal timestep embedding dimension.  Must match
    //                the value used inside `get_time_embedding`.
    //
    // steps (T)   — 100: diffusion timesteps.  Kept low for MNIST; larger models
    //                typically use T=1000 for smoother denoising trajectories.
    //
    // label_dropout— 0.15: 15% of training labels are zeroed per batch.
    //                This creates the unconditional branch used by CFG at
    //                inference time.
    //
    // cond_dim     — time_emb_dim + class_dim = 16 + 10 = 26.
    //                This is the dimension of the conditioning vector fed to
    //                the CNN's projection layer.
    // =========================================================================
    let epochs        = 20000;
    let lr            = 0.001f64;
    let img_dim       = 784;   // 28×28 pixels, flattened
    let class_dim     = 10;    // number of MNIST digit classes
    let time_emb_dim  = 16;    // sinusoidal time embedding size
    let steps         = 100;   // T: diffusion timesteps
    let label_dropout = 0.15f32;

    // The CNN receives a single concatenated conditioning vector of size
    // (time_emb_dim + class_dim) = 26.  It projects this into a spatial
    // conditioning map of size (B, 1, 28, 28) before channel-concatenating
    // with the noisy image.
    let cond_dim = time_emb_dim + class_dim; // 26

    // --- Dataset loading -----------------------------------------------------
    // `acquire_mnist` downloads and caches both the image and label binary files
    // if they are not already present.  After that it parses the IDX binary
    // format and returns a normalised tensor + raw label vector.
    //
    // `images`      shape: (60000, 784), pixel values in [-1, 1]
    // `train_labels` length: 60000, values in {0, …, 9}
    println!("Loading MNIST dataset...");
    let (images, train_labels) = acquire_mnist(&device)?;

    // `dims2` unpacks the two dimensions of the 2-D image tensor.
    // We only need `total_samples`; the trailing `_` discards img_dim (already
    // set above as a constant).
    let (total_samples, _) = images.dims2()?;

    // --- Beta schedule -------------------------------------------------------
    // Linear schedule from β_1=0.0001 to β_T=0.02 (original DDPM paper).
    // Pre-computes α_t, ᾱ_t, and σ_t for all t so the training loop can
    // look them up in O(1) rather than recomputing at every step.
    let scheduler = BetaScheduler::new(steps, 0.0001, 0.02, &device)?;

    // --- CNN model -----------------------------------------------------------
    // SimpleDenoisingCNN layout:
    //   w_cond: (img_dim=784, cond_dim=26)  — conditioning projection weights
    //   b_cond: (img_dim=784,)             — conditioning projection bias
    //   w1:     (16, 2, 3, 3)              — Conv1 weights (2 in-channels → 16)
    //   b1:     (16,)                      — Conv1 bias
    //   w2:     (1, 16, 3, 3)             — Conv2 weights (16 → 1 out-channel)
    //   b2:     (1,)                       — Conv2 bias
    //
    // WHY 2 input channels for Conv1?
    //   Channel 0: the noisy image x_t, reshaped to (B, 1, 28, 28).
    //   Channel 1: the conditioning map (projected from time+class), shape (B, 1, 28, 28).
    //   Concatenating them gives the CNN access to both content (noisy image)
    //   and context (what class, what noise level) in each spatial location.
    let mut cnn = SimpleDenoisingCNN::new(img_dim, cond_dim, &device)?;

    // --- Adam optimizer ------------------------------------------------------
    // `MlpAdamOptimizer::new` accepts any `DenoisingModel` and initialises
    // per-parameter moment vectors (m, v) to zero.  The generic interface
    // means we do not need a separate optimizer implementation for the CNN.
    let mut optimizer = MlpAdamOptimizer::new(&cnn, lr)?;

    println!("Starting CNN training for {} epochs...\n", epochs);

    let start_time = std::time::Instant::now();

    // =========================================================================
    // Training loop — CFG noise prediction with label dropout
    // =========================================================================
    //
    // Objective (DDPM MSE loss, same as MLP variant):
    //
    //   L = E_{t, x_0, c̃, ε} [ || ε − ε_θ(x_t, t, c̃) ||² ]
    //
    // c̃ is the (possibly dropped) class conditioning signal.
    // Minimising L trains both the conditional and unconditional branches
    // of the model simultaneously, enabling CFG at inference.
    // =========================================================================
    for epoch in 1..=epochs {
        // ---------------------------------------------------------------------
        // Step 1 — Random mini-batch
        // ---------------------------------------------------------------------
        // Sample `batch_size` random image indices uniformly from [0, N).
        // Uniform sampling ensures class balance and removes sequential
        // correlation between consecutive mini-batches.
        //
        // WHY subtract 1e-4 from the upper bound?
        //   `Tensor::rand` generates floats in [low, high).  Subtracting a tiny
        //   epsilon prevents the rare edge case where the float rounds to exactly
        //   `total_samples`, which would be an out-of-bounds index.
        let index_tensor = Tensor::rand(
            0.0f32,
            total_samples as f32 - 1e-4,
            (batch_size,),
            &device,
        )?
        .to_dtype(DType::U32)?;

        // Copy indices to a Rust Vec so we can index the plain Vec<u8> labels.
        let indices = index_tensor.to_vec1::<u32>()?;

        // Gather batch images: x0 shape (batch_size, 784).
        let x0 = images.index_select(&index_tensor, 0)?;

        // Gather labels matching the selected images.
        // This keeps images and labels perfectly aligned — same index, same row.
        let batch_labels: Vec<u8> = indices.iter().map(|&x| train_labels[x as usize]).collect();

        // ---------------------------------------------------------------------
        // Step 2 — CFG label dropout: build the (possibly nulled) one-hot tensor
        // ---------------------------------------------------------------------
        // Each label is independently zeroed with probability `label_dropout`.
        // The resulting tensor has shape (batch_size, 10).  Rows where the
        // label was dropped are all-zeros (the "null class" signal).
        //
        // WHY per-example dropout rather than per-batch?
        //   Per-example gives the model a mix of conditional and unconditional
        //   examples in every mini-batch, producing stable gradients for both
        //   branches simultaneously.
        let label_one_hot = make_one_hot_cfg(&batch_labels, label_dropout, &device)?;

        // ---------------------------------------------------------------------
        // Step 3 — Forward diffusion: corrupt x_0 → x_t
        // ---------------------------------------------------------------------
        // Sample a random timestep t ∈ {0, …, T-1} for each image in the batch.
        // WHY random t per example?
        //   The model must learn to denoise at all noise levels.  Randomising t
        //   ensures uniform training coverage across the full noise schedule.
        let t_float = Tensor::rand(0.0f32, steps as f32 - 1e-4, (batch_size,), &device)?;
        let t = t_float.to_dtype(DType::U32)?;

        // Ground-truth noise ε ~ N(0, I).  This is the training target.
        let noise = Tensor::randn(0.0f32, 1.0f32, (batch_size, img_dim), &device)?;

        // Apply the closed-form forward diffusion:
        //   x_t = sqrt(ᾱ_t) * x_0 + sqrt(1 − ᾱ_t) * ε
        // `add_noise` uses the pre-computed ᾱ_t from the scheduler.
        let xt = scheduler.add_noise(&x0, &noise, &t)?;

        // ---------------------------------------------------------------------
        // Step 4 — Build the conditioned model input vector v
        // ---------------------------------------------------------------------
        // The CNN's forward pass splits `v` into two parts:
        //   v[:, 0:784]   = x_t   — the noisy image (reshaped to 28×28 inside)
        //   v[:, 784:810] = concat(time_emb, label_one_hot) — conditioning
        //
        // WHY concatenate conditioning into a single flat vector?
        //   The CNN's first layer (w_cond) projects the entire 26-dim conditioning
        //   vector into a 784-dim spatial map.  Concatenating time and class into
        //   one vector gives the projection layer the freedom to mix them as
        //   needed, rather than treating them as separate additive signals.
        let time_emb = get_time_embedding(&t, time_emb_dim)?;

        // v shape: (batch_size, 784 + 16 + 10) = (batch_size, 810)
        let v = Tensor::cat(&[&xt, &time_emb, &label_one_hot], 1)?;

        // ---------------------------------------------------------------------
        // Step 5 — Forward pass through the CNN + MSE loss
        // ---------------------------------------------------------------------
        // `DenoisingModel::forward` calls the CNN's forward implementation:
        //   1. Project cond_vec (26 dims) → spatial conditioning map (1, 28, 28)
        //   2. Concatenate with xt_img   → (2, 28, 28) per sample
        //   3. Conv1(2→16) + Leaky-ReLU
        //   4. Conv2(16→1) + reshape back to (B, 784)
        //
        // Returns:
        //   pred          — predicted noise ε̂, shape (batch_size, 784)
        //   intermediates — [input_cat, z1, a1]: cached activations for backprop
        let (pred, intermediates) = DenoisingModel::forward(&cnn, &v)?;

        // MSE loss — used only for logging; the gradient comes from backward().
        let diff = pred.sub(&noise)?;
        let loss = diff.sqr()?.mean_all()?.to_scalar::<f32>()?;

        // ---------------------------------------------------------------------
        // Step 6 — Backward pass: compute per-parameter gradients
        // ---------------------------------------------------------------------
        // `DenoisingModel::backward` implements the manual chain rule through:
        //   Conv2 backward → Leaky-ReLU backward → Conv1 backward →
        //   Conditioning projection backward
        //
        // Returns one gradient tensor per parameter, in the same order as
        // `params()`: [dw_cond, db_cond, dw1, db1, dw2, db2].
        let grads = DenoisingModel::backward(&cnn, &v, &intermediates, &pred, &noise)?;

        // ---------------------------------------------------------------------
        // Step 7 — Adam optimizer step
        // ---------------------------------------------------------------------
        // Adam applies per-parameter adaptive learning rates using first and
        // second gradient moment estimates.  The generic optimizer zips `grads`
        // with `params_mut()` to update each parameter tensor in place.
        optimizer.step(&mut cnn, &grads)?;

        // ---------------------------------------------------------------------
        // Step 8 — Periodic logging: loss + gradient norms
        // ---------------------------------------------------------------------
        // WHY log gradient norms alongside loss?
        //   The L2 norm of each layer's gradient reveals training health:
        //     - Very small norms → gradient vanishing (learning stalled).
        //     - Very large norms → gradient explosion (instability risk).
        //   Having per-layer norms (w_cond, b_cond, w1, b1, w2, b2) makes it
        //   easy to pinpoint which layer is misbehaving.
        //
        // WHY every 100 epochs?
        //   Printing every step would flood the terminal and add measurable
        //   overhead (~microseconds per print × 20000 steps).
        if epoch % 100 == 0 || epoch == 1 {
            let elapsed = start_time.elapsed().as_secs_f32();
            let speed = epoch as f32 / elapsed;

            // `param_names()` returns ["w_cond", "b_cond", "w1", "b1", "w2", "b2"].
            let param_names = cnn.param_names();

            // Compute L2 norm = sqrt(sum(g^2)) for each gradient tensor.
            let grad_norms: Vec<f32> = grads
                .iter()
                .map(|g| -> Result<f32> { Ok(g.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt()) })
                .collect::<Result<_>>()?;

            // Zip names with norms for a readable log line.
            let norms_str: Vec<String> = param_names
                .iter()
                .zip(grad_norms.iter())
                .map(|(name, norm)| format!("{} norm: {:.4}", name, norm))
                .collect();

            println!(
                "Epoch {:5}/{} - MSE Loss: {:.6} | Speed: {:.1} steps/s | Elapsed: {:.1}s | {}",
                epoch,
                epochs,
                loss,
                speed,
                elapsed,
                norms_str.join(", ")
            );
        }
    }

    // =========================================================================
    // CFG-guided image generation at multiple guidance scales
    // =========================================================================
    //
    // Now that the CNN is trained, we run reverse diffusion three times for the
    // same target digit (3) at increasing guidance scales to compare quality:
    //
    //   s = 0.0 — unconditional sampling.
    //     The class one-hot is supplied but guidance_scale = 0, so the guidance
    //     term (eps_cond - eps_uncond) is multiplied by zero and has no effect.
    //     The output should look like a random (not necessarily digit-3) image.
    //
    //   s = 1.0 — standard conditional sampling.
    //     The guided prediction equals eps_cond: we simply use the conditional
    //     noise prediction without any extrapolation.  This is equivalent to
    //     the class-conditioned DDPM from `train_diffusion_cond.rs`.
    //
    //   s = 3.0 — CFG-amplified sampling.
    //     The guidance direction (eps_cond - eps_uncond) is scaled by 3 and
    //     added to eps_cond.  This pushes the sample more aggressively toward
    //     digit-3 characteristics, typically producing sharper, more recognisable
    //     digit strokes at the cost of some diversity.
    //
    // Saving all three PNGs lets the user see the quality-diversity trade-off
    // introduced by the guidance scale parameter at a glance.
    // =========================================================================
    println!("\n=== Starting Classifier-Free Guided Reverse Sampling ===");

    // Target class: digit "3".
    // Build the one-hot vector manually (drop_rate=0: never drop at inference).
    let target_digit = 3;
    let mut target_vec = vec![0.0f32; class_dim];
    target_vec[target_digit] = 1.0f32;
    // Shape: (1, 10) — single sample, 10 classes.
    let target_one_hot = Tensor::from_vec(target_vec, (1, class_dim), &device)?;

    // Generate and save images at three guidance scales for comparison.
    // Each call to `save_cfg_sample` runs the full T-step reverse loop.
    save_cfg_sample(
        &cnn,
        &scheduler,
        &target_one_hot,
        0.0, // s=0: unconditional — baseline (ignores the class label)
        img_dim,
        time_emb_dim,
        "mnist_cfg_generated_s0.png",
        &device,
    )?;

    save_cfg_sample(
        &cnn,
        &scheduler,
        &target_one_hot,
        1.0, // s=1: standard conditional — class-conditioned without extrapolation
        img_dim,
        time_emb_dim,
        "mnist_cfg_generated_s1.png",
        &device,
    )?;

    save_cfg_sample(
        &cnn,
        &scheduler,
        &target_one_hot,
        3.0, // s=3: CFG-boosted — amplified class guidance (sharpest class fidelity)
        img_dim,
        time_emb_dim,
        "mnist_cfg_generated_s3.png",
        &device,
    )?;

    println!("Generated images saved successfully.");
    Ok(())
}
