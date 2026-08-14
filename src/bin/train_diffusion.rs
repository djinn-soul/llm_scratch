// =============================================================================
// train_diffusion.rs — DDPM (Denoising Diffusion Probabilistic Model) trainer
// =============================================================================
//
// This binary implements a minimal *unconditional* DDPM pipeline on the MNIST
// digit dataset.  A two-layer MLP is trained to predict the noise that was
// added to a clean image, then the trained model is run in reverse to generate
// a new digit image from pure Gaussian noise.
//
// Paper reference: "Denoising Diffusion Probabilistic Models" (Ho et al., 2020)
//   https://arxiv.org/abs/2006.11239
//
// For the class-conditioned variant see `train_diffusion_cond.rs`.
// Shared dataset and PNG utilities live in `src/mnist_utils.rs`.
// =============================================================================

// `bail!` lets us return early with a descriptive error string.
// `Result` is anyhow's alias for std::result::Result<T, anyhow::Error>.
use anyhow::Result;

// Candle is a lightweight ML framework (similar to PyTorch) written in Rust.
// `DType`  — data type of tensor elements (F32, U32, …)
// `Device` — where tensors live (CPU or CUDA GPU)
// `Tensor` — the core n-dimensional array type
use candle_core::{DType, Device, Tensor};

// Local model components for the diffusion pipeline:
//   `get_time_embedding`    — sinusoidal positional encoding of the timestep
//   `BetaScheduler`         — pre-computes beta/alpha/sigma schedules
//   `MlpAdamOptimizer`      — Adam optimizer wrapper for the MLP weights
//   `SimpleDenoisingMlp`    — two-layer MLP: input → hidden → output
//   `sample_ddpm_from_noise`— shared reverse diffusion sampler (unconditional)
use llm_scratch_rs::models::diffusion::{
    get_time_embedding, sample_ddpm_from_noise, BetaScheduler, DenoisingModel, MlpAdamOptimizer,
    Parameterized, SimpleDenoisingMlp,
};

// Shared MNIST utilities: download + cache + parse binary files, save PNGs.
// Lives in src/utils/mnist_utils.rs so both diffusion binaries share the code.
use llm_scratch_rs::utils::mnist_utils::{acquire_mnist_images, save_png};

// =============================================================================
// main — training loop and reverse-diffusion sampling
// =============================================================================
//
// High-level flow:
//   1. Load MNIST images.
//   2. Build the beta schedule, the MLP, and the Adam optimizer.
//   3. Run the forward-diffusion training loop (epochs × mini-batches).
//   4. After training, run the reverse-diffusion sampling loop to generate an image.
//   5. Save PNGs: original reference, initial noise, and generated output.
// =============================================================================

// Entry point: orchestrates training & sampling for the DDPM pipeline.
fn main() -> Result<()> {
    // --- Device auto-selection ------------------------------------------------
    // Attempt to use CUDA GPU 0; fall back silently to CPU if unavailable.
    // Requires the `cuda` Cargo feature flag (`cargo run --features cuda`).
    let device = Device::new_cuda(0).unwrap_or(Device::Cpu);
    println!("== DDPM MNIST Model Training");
    println!("Active Device: {:?}", device);

    // --- Dataset loading -----------------------------------------------------
    // `acquire_mnist_images` handles download + caching automatically.
    // After download it delegates to the IDX3-ubyte binary parser.
    // `dataset` has shape (60000, 784) with pixel values in [-1, 1].
    // WHY flattened to 784? The MLP operates on 1-D vectors, not 2-D grids.
    let dataset = acquire_mnist_images(&device)?;

    // `dim(0)` returns the size of the first dimension = number of images.
    // We use this later to generate random indices into the dataset.
    let total_samples = dataset.dim(0)?;
    println!("Total samples: {}", total_samples);

    // =========================================================================
    // Hyper-parameter configuration
    // =========================================================================
    //
    // steps (T)        — number of diffusion timesteps.
    //                    More steps = smoother denoising, but slower inference.
    //                    100 is enough for MNIST; typical DDPM uses 1000.
    //
    // time_emb_dim     — size of the sinusoidal time embedding vector.
    //                    Larger = more expressive conditioning signal.
    //                    16 is lightweight; production models use 128–512.
    //
    // img_dim          — 28 × 28 = 784 pixels, flattened into a 1-D vector.
    //                    The MLP operates on flat vectors, not 2-D images.
    //
    // hidden_dim       — number of neurons in the single hidden layer.
    //                    512 gives enough capacity for MNIST patterns.
    //
    // batch_size       — number of training examples per gradient update.
    //                    128 balances GPU memory, gradient variance, and speed.
    //
    // epochs           — total number of gradient update steps.
    //                    With batch_size=128 and 60 000 samples, each "epoch"
    //                    here is really one mini-batch step (not a full epoch
    //                    in the traditional sense).
    //
    // lr               — Adam learning rate.
    //                    0.001 is the standard Adam default (Kingma & Ba 2015).
    // =========================================================================
    let steps = 100; // T: total diffusion timesteps
    let time_emb_dim = 16; // dimensionality of sinusoidal time encoding
    let img_dim = 784; // 28×28 flattened image size
    let hidden_dim = 512; // hidden layer width of the denoising MLP
    let batch_size = match device {
        Device::Cuda(_) => 256, // CUDA: larger batch → better GPU utilisation
        _ => 128,               // CPU: smaller batch → fits in RAM
    };
    let epochs = 20000; // total number of gradient update steps
    let lr = 0.001; // Adam learning rate (α)

    // --- Build the beta schedule ---------------------------------------------
    // `BetaScheduler` pre-computes a *linear* beta schedule from β_1=0.0001
    // to β_T=0.02 (as in the original DDPM paper).
    //
    // WHY pre-compute?  α_t, ᾱ_t, and σ_t are deterministic functions of β_t
    // and never change during training, so computing them once is efficient.
    //
    // Key quantities the scheduler stores:
    //   betas          β_t  — noise variance added at each step
    //   alphas         α_t  = 1 - β_t
    //   alphas_cumprod ᾱ_t  = ∏_{s=1}^{t} α_s  (cumulative product)
    //   sigmas         σ_t  = sqrt(β_t)  — used in the reverse process
    let scheduler = BetaScheduler::new(steps, 0.0001, 0.02, &device)?;

    // --- Build the denoising MLP ---------------------------------------------
    // Architecture: Linear(784+16, 512) → ReLU → Linear(512, 784)
    //
    // WHY 784+16 as input?
    //   We concatenate the flattened noisy image (784 dims) with the time
    //   embedding (16 dims) so the model knows *when* (i.e., how noisy) the
    //   input is and can adjust its prediction accordingly.
    //
    // WHY 784 as output?
    //   The model predicts the noise vector ε that was added to x_0.
    //   ε has the same shape as the image, so the output dimension = img_dim.
    let mlp = SimpleDenoisingMlp::new(img_dim + time_emb_dim, hidden_dim, img_dim, &device)?;

    // --- Build the Adam optimizer --------------------------------------------
    // Adam maintains per-parameter first and second moment estimates (m, v).
    // This allows adaptive learning rates and is more stable than plain SGD
    // for diffusion models.
    let mut optimizer = MlpAdamOptimizer::new(&mlp, lr)?;

    // Log model configuration so we can sanity-check before a long training run.
    println!("Scheduler: Linear, {} steps", steps);
    println!(
        "MLP: input_dim={} (784+16), hidden_dim={}, output_dim=784",
        img_dim + time_emb_dim,
        hidden_dim
    );
    println!("Starting training for {} epochs...\n", epochs);

    // =========================================================================
    // Training loop
    // =========================================================================
    //
    // Each iteration corresponds to one mini-batch gradient update.
    // The DDPM training objective (simplified) is:
    //
    //   L = E_{t, x_0, ε} [ || ε − ε_θ(x_t, t) ||² ]
    //
    // In plain English: the model ε_θ takes a noisy image x_t and a timestep t
    // and must predict the Gaussian noise ε that was added to the clean x_0.
    // Minimising this MSE loss is equivalent to maximising the ELBO of the
    // variational lower bound on log p(x_0).
    // =========================================================================
    for epoch in 1..=epochs {
        // ---------------------------------------------------------------------
        // Step 1 — Sample a random mini-batch of clean images (x_0)
        // ---------------------------------------------------------------------
        // WHY random sampling instead of sequential batches?
        //   Stochastic gradient descent requires i.i.d. samples.  Random
        //   sampling ensures each batch covers a variety of digit classes and
        //   avoids gradient bias from correlated sequential samples.
        //
        // We generate uniform floats in [0, total_samples) and truncate to u32
        // indices.  Subtracting 1e-4 prevents the float from rounding up to
        // `total_samples` (which would be out of bounds).
        let index_tensor = Tensor::rand(
            0.0f32,
            total_samples as f32 - 1e-4f32,
            (batch_size,),
            &device,
        )?
        .to_dtype(DType::U32)?;

        // `index_select` gathers rows at the sampled indices → shape (B, 784).
        // The dataset is already flat (N, 784) from `load_mnist_images`.
        let x0 = dataset.index_select(&index_tensor, 0)?;

        // ---------------------------------------------------------------------
        // Step 2 — Sample a random timestep t ∈ {0, …, T-1} for each example
        // ---------------------------------------------------------------------
        // WHY a different t per example?
        //   The model must learn to denoise at *all* noise levels simultaneously.
        //   Randomising t across the batch ensures even coverage of the schedule
        //   without needing to iterate through every t on every mini-batch.
        //
        // WHY float first, then cast to U32?
        //   `Tensor::rand` generates continuous uniform floats.  Casting to U32
        //   truncates (floor) to integer timestep indices.
        let t_float = Tensor::rand(0.0f32, steps as f32 - 1e-4f32, (batch_size,), &device)?;
        let t = t_float.to_dtype(DType::U32)?;

        // ---------------------------------------------------------------------
        // Step 3 — Draw Gaussian noise ε and create the noisy image x_t
        // ---------------------------------------------------------------------
        // The DDPM forward process is defined as:
        //   x_t = sqrt(ᾱ_t) * x_0 + sqrt(1 - ᾱ_t) * ε,  ε ~ N(0, I)
        //
        // This is a *closed-form* one-shot corruption that jumps from x_0
        // directly to any noise level t, avoiding the need to iterate t steps.
        // `scheduler.add_noise` applies this formula using the pre-computed ᾱ_t.
        //
        // The target label for the model is precisely this noise vector ε.
        let noise = Tensor::randn(0.0f32, 1.0f32, (batch_size, img_dim), &device)?;
        let xt = scheduler.add_noise(&x0, &noise, &t)?;

        // ---------------------------------------------------------------------
        // Step 4 — Compute sinusoidal time embedding and build the model input
        // ---------------------------------------------------------------------
        // WHY sinusoidal embeddings?
        //   The model needs to know the noise level (timestep t) to calibrate
        //   its prediction.  Sinusoidal embeddings (like in Transformers) map
        //   each integer t to a continuous vector where nearby timesteps have
        //   similar representations, giving the model a smooth sense of "time".
        //
        // WHY concatenation instead of addition?
        //   For a simple two-layer MLP, concatenation is the easiest way to
        //   fuse two modalities.  Attention-based U-Nets use cross-attention or
        //   additive conditioning instead.
        //
        // v shape: (batch_size, img_dim + time_emb_dim) = (128, 800)
        let time_emb = get_time_embedding(&t, time_emb_dim)?;
        let v = Tensor::cat(&[&xt, &time_emb], 1)?;

        // ---------------------------------------------------------------------
        // Step 5 — Forward pass: predict the noise ε̂ = ε_θ(x_t, t)
        // ---------------------------------------------------------------------
        // The MLP returns:
        //   pred — the predicted noise vector ε̂  (shape: batch_size × img_dim)
        //   a1   — post-activation of the hidden layer (needed for backward pass)
        //   z1   — pre-activation of the hidden layer (needed for backward pass)
        //
        // WHY return intermediate activations?
        //   Our manual backprop implementation needs the cached activations to
        //   compute gradients efficiently (chain rule).  Autograd frameworks
        //   (e.g., PyTorch) do this automatically in their computation graph.
        let (pred, intermediates) = DenoisingModel::forward(&mlp, &v)?;

        // ---------------------------------------------------------------------
        // Step 6 — Compute MSE loss between predicted and true noise
        // ---------------------------------------------------------------------
        // Loss = (1/N) Σ (ε̂ - ε)²   (Mean Squared Error)
        //
        // WHY MSE?
        //   Ho et al. showed that optimising this simple objective is equivalent
        //   to a re-weighted ELBO of the diffusion model likelihood.  MSE is
        //   differentiable everywhere and numerically stable.
        //
        // Note: we compute the scalar loss only for *logging*; the actual
        // gradient is computed by `mlp.backward` in the next step.
        let diff = pred.sub(&noise)?;
        let loss = diff.sqr()?.mean_all()?.to_scalar::<f32>()?;

        // ---------------------------------------------------------------------
        // Step 7 — Back-propagation: compute gradients ∂L/∂W for all weights
        // ---------------------------------------------------------------------
        // WHY manual backward?
        //   Candle does not yet have full autograd support for all ops in this
        //   custom MLP.  We hand-rolled the backward pass using the chain rule.
        //
        // `grads` contains:
        //   dw1  — gradient w.r.t. first layer weights  (input→hidden)
        //   db1  — gradient w.r.t. first layer biases
        //   dw2  — gradient w.r.t. second layer weights (hidden→output)
        //   db2  — gradient w.r.t. second layer biases
        let grads = DenoisingModel::backward(&mlp, &v, &intermediates, &pred, &noise)?;

        // ---------------------------------------------------------------------
        // Step 8 — Adam optimizer step: update weights using the gradients
        // ---------------------------------------------------------------------
        // Adam computes adaptive per-parameter learning rates:
        //   m_t = β1 * m_{t-1} + (1-β1) * g_t          (biased 1st moment)
        //   v_t = β2 * v_{t-1} + (1-β2) * g_t²         (biased 2nd moment)
        //   m̂_t = m_t / (1 - β1^t)                      (bias-corrected)
        //   v̂_t = v_t / (1 - β2^t)
        //   θ_t = θ_{t-1} - lr * m̂_t / (sqrt(v̂_t) + ε)
        //
        // This adapts the step size for each parameter individually and handles
        // sparse or noisy gradients better than plain gradient descent.
        optimizer.step(&mlp, &grads)?;

        let param_names = mlp.param_names();

        // ---------------------------------------------------------------------
        // Step 9 — Periodic logging: print loss and gradient norms
        // ---------------------------------------------------------------------
        // WHY gradient norms?
        //   The L2 norm of the gradient (||∇W||₂) indicates how strongly the
        //   weights are being updated.  Very small norms → vanishing gradients
        //   (training stalled).  Very large norms → exploding gradients (unstable).
        //   Watching both the loss and the norms together helps diagnose training
        //   health without running a full validation loop.
        //
        // WHY log only every 100 epochs?
        //   Printing to stdout every step would dominate wall-clock time and
        //   flood the terminal.  Every 100 steps gives a smooth curve without
        //   overhead.
        if epoch % 100 == 0 || epoch == 1 {
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
                "Epoch {:4}/{} - MSE Loss: {:.6} | {}",
                epoch,
                epochs,
                loss,
                norms_str.join(", ")
            );
        }
    }

    // =========================================================================
    // Reverse diffusion sampling (image generation)
    // =========================================================================
    //
    // Now that the model is trained we run the *reverse* process to generate
    // a new image from pure Gaussian noise.
    //
    // The reverse process is (from the DDPM paper):
    //
    //   x_{t-1} = (1/sqrt(α_t)) * (x_t - β_t/sqrt(1-ᾱ_t) * ε_θ(x_t, t))
    //             + σ_t * z        where z ~ N(0, I)  if t > 0
    //                                    z = 0         if t = 0
    //
    // Starting from x_T ~ N(0, I) we iterate t = T-1 down to t = 0 to recover
    // a sample x_0 from the learned data distribution.
    // =========================================================================
    println!("\n=== Starting Reverse Diffusion Sampling (Generation) ===");

    // Save image #0 from the training set as a visual reference baseline.
    // This lets us compare the generated image quality against a real digit.
    let original_sample = dataset
        .index_select(&Tensor::new(&[0u32], &device)?, 0)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    save_png("mnist_original.png", &original_sample)?;

    // Initialise x_T as pure Gaussian noise — the starting point of the
    // reverse diffusion chain.  Shape: (1, 784) for a single image.
    let num_samples = 1; // generate one image at a time
    let initial_noise = Tensor::randn(0.0f32, 1.0f32, (num_samples, img_dim), &device)?;

    // Persist the initial pure Gaussian noise for debugging / visualisation.
    save_png(
        "mnist_noisy.png",
        &initial_noise.flatten_all()?.to_vec1::<f32>()?,
    )?;

    // Run the shared unconditional reverse diffusion sampler.
    // `sample_ddpm_from_noise` iterates t from T-1 → 0, predicting and
    // subtracting noise at each step.  The implementation lives in
    // `src/models/diffusion/sampling.rs`.
    let generated = sample_ddpm_from_noise(
        &mlp,
        &scheduler,
        initial_noise,
        img_dim,
        time_emb_dim,
        &device,
    )?;

    // Flatten the generated tensor to a 1-D pixel slice and encode as PNG.
    // `save_png` (from mnist_utils) converts [-1, 1] floats to u8 [0, 255].
    let final_pixels = generated.flatten_all()?.to_vec1::<f32>()?;
    save_png("mnist_generated.png", &final_pixels)?;

    Ok(())
}
