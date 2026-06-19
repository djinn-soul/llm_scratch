// =============================================================================
// train_diffusion_cond.rs — Class-Conditioned DDPM trainer
// =============================================================================
//
// This binary extends the unconditional DDPM (`train_diffusion.rs`) with
// *class conditioning*: the digit label (0–9) is fed to the model alongside
// the noisy image and the timestep, so at inference time we can request a
// specific digit class.
//
// The key difference from the unconditional variant:
//   Unconditional input: concat(x_t, time_embedding)          → shape (784+16)
//   Conditional input:   concat(x_t, time_embedding, one_hot) → shape (784+16+10)
//
// Because the model input is wider, we instantiate the MLP with
// input_dim = img_dim + time_emb_dim + class_dim = 784 + 16 + 10 = 810.
//
// Paper reference: "Classifier-Free Diffusion Guidance" (Ho & Salimans, 2022)
//   https://arxiv.org/abs/2207.12598
// For the simpler *classifier-guided* / label-conditioned variant used here
// see the original DDPM paper (Ho et al., 2020):
//   https://arxiv.org/abs/2006.11239
// =============================================================================

// anyhow provides ergonomic error propagation with the `?` operator.
use anyhow::Result;

// Candle tensor library: DType for type casts, Device for CPU/GPU placement,
// Tensor for all numerical operations.
use candle_core::{DType, Device, Tensor};

// Diffusion model components:
//   get_time_embedding  — sinusoidal timestep positional encoding
//   BetaScheduler       — pre-computes the noise schedule (beta, alpha, sigma)
//   MlpAdamOptimizer    — Adam optimiser for the MLP parameters
//   SimpleDenoisingMlp  — two-layer fully-connected denoising network
//   sample_ddpm_cond    — shared reverse diffusion sampler with class conditioning
use llm_scratch_rs::models::diffusion::{
    get_time_embedding, sample_ddpm_cond, BetaScheduler, DenoisingModel, MlpAdamOptimizer,
    SimpleDenoisingMlp,
};

// Shared MNIST helpers: download, parse binary files, build one-hot tensors,
// and save 28x28 grayscale PNGs. Lives in src/utils/mnist_utils.rs.
use llm_scratch_rs::utils::mnist_utils::{acquire_mnist, make_one_hot, save_png};

// =============================================================================
// main — class-conditioned training loop and conditional image generation
// =============================================================================
//
// High-level flow:
//   1. Load MNIST images *and* labels (labels drive the conditioning).
//   2. Build the beta schedule, the wider MLP (810-dim input), and Adam.
//   3. Training loop: at each step, corrupt a batch, feed label one-hot to MLP.
//   4. After training, run reverse diffusion conditioned on a target digit.
//   5. Save the generated image as PNG.
// =============================================================================
fn main() -> Result<()> {
    // --- Device selection ----------------------------------------------------
    // `Device::Cpu` runs every tensor op on the host CPU.
    // Swap to `Device::Cuda(0)` for GPU training (requires the Candle `cuda`
    // feature and a compatible NVIDIA GPU).
    let device = Device::Cpu;
    println!("== Class-Conditioned DDPM MNIST Model Training");

    // --- Dataset loading -----------------------------------------------------
    // `acquire_mnist` downloads and caches both the image and label binary files.
    // `images` shape: (60000, 784), pixel values in [-1, 1]
    // `labels_raw`:   60000 u8 values in {0, …, 9}
    //
    // WHY do we need labels here but not in `train_diffusion.rs`?
    //   The unconditional model only sees (x_t, t).
    //   The conditional model also sees the digit class, so it needs labels.
    let (images, labels_raw) = acquire_mnist(&device)?;
    let total_samples = images.dim(0)?;
    println!("Total samples: {}", total_samples);

    // =========================================================================
    // Hyper-parameter configuration
    // =========================================================================
    //
    // steps (T)        — number of diffusion timesteps.
    //                    Kept at 100 for MNIST; typical DDPM uses T=1000.
    //
    // time_emb_dim     — dimension of the sinusoidal time embedding.
    //                    16 is lightweight; larger models use 128–512.
    //
    // class_dim        — number of digit classes (10 for MNIST: digits 0–9).
    //                    Each class is represented by a one-hot vector of this
    //                    length, appended to the model's input.
    //
    // img_dim          — 28×28 = 784 pixels, flattened.
    //
    // hidden_dim       — width of the single hidden layer.
    //                    512 is enough capacity for conditional MNIST generation.
    //
    // batch_size       — examples per gradient update.
    //                    128 gives a good bias/variance trade-off for Adam.
    //
    // epochs           — total gradient update steps (not traditional epochs;
    //                    each step processes one random mini-batch).
    //                    8000 is fewer than the unconditional run because the
    //                    conditional model converges faster with more signal.
    //
    // lr               — Adam learning rate (Kingma & Ba default: 0.001).
    // =========================================================================
    let steps = 100; // T: total diffusion timesteps
    let time_emb_dim = 16; // sinusoidal time embedding dimension
    let class_dim = 10; // one-hot label dimension (digits 0–9)
    let img_dim = 784; // 28×28 flattened image size
    let hidden_dim = 512; // hidden layer width
    let batch_size = 128; // mini-batch size per gradient step
    let epochs = 8000; // total gradient update steps
    let lr = 0.001; // Adam learning rate (α)

    // --- Build the beta schedule ---------------------------------------------
    // Linear schedule from β_1=0.0001 to β_T=0.02 (original DDPM paper).
    // Pre-computes β_t, α_t = 1−β_t, ᾱ_t = ∏α_s, σ_t = sqrt(β_t) for all t.
    // WHY pre-compute? These values are constant during training, so computing
    // them once is much cheaper than recomputing inside every training step.
    let scheduler = BetaScheduler::new(steps, 0.0001, 0.02, &device)?;

    // --- Build the class-conditioned denoising MLP ---------------------------
    // Architecture: Linear(810, 512) → ReLU → Linear(512, 784)
    //
    // WHY input_dim = 784 + 16 + 10 = 810?
    //   The model receives three sources of information concatenated:
    //     1. x_t          (784 dims) — the noisy image at timestep t
    //     2. time_emb     (16  dims) — sinusoidal encoding of t
    //     3. class_one_hot(10  dims) — one-hot encoding of the digit class
    //
    // WHY output_dim = 784?
    //   The model predicts the noise vector ε ~ N(0, I) that was added to x_0.
    //   ε has the same shape as the image, so output_dim = img_dim = 784.
    let mut mlp = SimpleDenoisingMlp::new(
        img_dim + time_emb_dim + class_dim, // 810
        hidden_dim,                         // 512
        img_dim,                            // 784
        &device,
    )?;

    // --- Build the Adam optimiser --------------------------------------------
    // Adam maintains per-parameter running estimates of the gradient mean (m)
    // and variance (v), giving it adaptive learning rates that work well even
    // when gradients are noisy or sparse.
    let mut optimizer = MlpAdamOptimizer::new(&mlp, lr)?;

    println!(
        "MLP input_dim={} (784+16+10), hidden={}, output=784",
        img_dim + time_emb_dim + class_dim,
        hidden_dim
    );
    println!("Starting training for {} epochs...\n", epochs);

    // =========================================================================
    // Training loop — class-conditioned noise prediction
    // =========================================================================
    //
    // The training objective is identical to unconditional DDPM except the
    // model also receives the class label:
    //
    //   L = E_{t, x_0, c, ε} [ || ε − ε_θ(x_t, t, c) ||² ]
    //
    // where c is the one-hot class vector.
    //
    // By seeing many (noisy image, timestep, label) triples, the model learns
    // to predict noise in a *class-aware* way — the noise for a "3" looks
    // different from the noise for a "7" because the signal underneath differs.
    // =========================================================================
    for epoch in 1..=epochs {
        // ---------------------------------------------------------------------
        // Step 1 — Sample a random mini-batch of images and their labels
        // ---------------------------------------------------------------------
        // WHY random sampling?
        //   SGD requires i.i.d. samples; random indices ensure the batch
        //   covers diverse digit classes and avoids sequential correlation.
        //
        // We also need the corresponding labels so we can build the one-hot
        // conditioning vectors.  For the unconditional model we only need
        // the images.
        let index_tensor = Tensor::rand(
            0.0f32,
            total_samples as f32 - 1e-4f32,
            (batch_size,),
            &device,
        )?
        .to_dtype(DType::U32)?;

        // `to_vec1` copies the index tensor to CPU memory so we can use it
        // to index into the plain Vec<u8> label array.
        let indices = index_tensor.to_vec1::<u32>()?;

        // Gather batch images: shape (batch_size, 784).
        let x0 = images.index_select(&index_tensor, 0)?;

        // Gather matching labels from the Rust Vec using the same indices.
        // This keeps images and labels perfectly aligned.
        let batch_labels: Vec<u8> = indices
            .iter()
            .map(|&idx| labels_raw[idx as usize])
            .collect();

        // Convert labels to one-hot: shape (batch_size, 10).
        // Each row has exactly one 1.0 at the digit's class position.
        let label_one_hot = make_one_hot(&batch_labels, &device)?;

        // ---------------------------------------------------------------------
        // Step 2 — Sample random diffusion timesteps t ∈ {0, …, T-1}
        // ---------------------------------------------------------------------
        // WHY a different t per sample?
        //   The model must learn to denoise at every noise level simultaneously.
        //   Random t ensures uniform coverage of the noise schedule per batch.
        let t_float = Tensor::rand(0.0f32, steps as f32 - 1e-4f32, (batch_size,), &device)?;
        let t = t_float.to_dtype(DType::U32)?;

        // ---------------------------------------------------------------------
        // Step 3 — Forward diffusion: corrupt x_0 into x_t
        // ---------------------------------------------------------------------
        // The DDPM forward process is:
        //   x_t = sqrt(ᾱ_t) * x_0 + sqrt(1 − ᾱ_t) * ε,   ε ~ N(0, I)
        //
        // `add_noise` applies this closed-form formula using the pre-computed ᾱ_t.
        // WHY closed-form? We can jump to any noise level directly without
        // iterating t individual steps.
        //
        // The noise ε is the *target* the model must predict.
        let noise = Tensor::randn(0.0f32, 1.0f32, (batch_size, img_dim), &device)?;
        let xt = scheduler.add_noise(&x0, &noise, &t)?;

        // ---------------------------------------------------------------------
        // Step 4 — Build the class-conditioned model input
        // ---------------------------------------------------------------------
        // Time embedding maps integer t → a smooth 16-dim vector.
        // WHY sinusoidal? Nearby timesteps have similar embeddings, giving the
        // model a differentiable sense of "noise level progression".
        let time_emb = get_time_embedding(&t, time_emb_dim)?;

        // v = concat(x_t, time_emb, label_one_hot)
        // shape: (batch_size, 784 + 16 + 10) = (128, 810)
        //
        // WHY concat rather than add?
        //   Concatenation gives each modality its own weight slice in the first
        //   linear layer, preventing any information from being inadvertently
        //   cancelled out before the model can use it.
        let v = Tensor::cat(&[&xt, &time_emb, &label_one_hot], 1)?;

        // ---------------------------------------------------------------------
        // Step 5 — Forward pass: ε̂ = ε_θ(x_t, t, c)
        // ---------------------------------------------------------------------
        // The MLP returns:
        //   pred — predicted noise ε̂,  shape (batch_size, 784)
        //   a1   — post-ReLU hidden activations (cached for backprop chain rule)
        //   z1   — pre-ReLU hidden activations  (cached for backprop chain rule)
        let (pred, intermediates) = DenoisingModel::forward(&mlp, &v)?;

        // ---------------------------------------------------------------------
        // Step 6 — Compute MSE loss: L = (1/N) Σ (ε̂ − ε)²
        // ---------------------------------------------------------------------
        // WHY MSE?  Ho et al. proved this simplified objective is a
        // re-weighted ELBO of the diffusion model's variational lower bound.
        // It's differentiable everywhere and numerically stable.
        //
        // The scalar `loss` is only used for logging; gradients come from `backward`.
        let diff = pred.sub(&noise)?;
        let loss = diff.sqr()?.mean_all()?.to_scalar::<f32>()?;

        // ---------------------------------------------------------------------
        // Step 7 — Backpropagation: compute ∂L/∂W for all parameters
        // ---------------------------------------------------------------------
        // WHY manual backward?
        //   This MLP uses a hand-rolled backward pass because Candle's autograd
        //   was not yet complete for all ops used here.  The chain rule gives us
        //   gradients for both linear layers (dw1, db1, dw2, db2).
        let grads = DenoisingModel::backward(&mlp, &v, &intermediates, &pred, &noise)?;

        // ---------------------------------------------------------------------
        // Step 8 — Adam optimiser step: θ ← θ − lr * Adam(∇θ L)
        // ---------------------------------------------------------------------
        // Adam computes bias-corrected first and second moment estimates, then
        // applies per-parameter adaptive learning rates.  This is more robust
        // to noisy gradients than SGD with a fixed step size.
        optimizer.step(&mut mlp, &grads)?;

        let param_names = mlp.param_names();

        // ---------------------------------------------------------------------
        // Step 9 — Periodic logging every 100 steps
        // ---------------------------------------------------------------------
        // Gradient norms complement the loss as training diagnostics:
        //   Very small norms → vanishing gradients (training stalled).
        //   Very large norms → exploding gradients (instability risk).
        // Logging every 100 steps avoids stdout overhead while still giving
        // a smooth loss curve.
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
                "Epoch {:5}/{} - MSE Loss: {:.6} | {}",
                epoch,
                epochs,
                loss,
                norms_str.join(", ")
            );
        }
    }

    // =========================================================================
    // Conditional image generation via reverse diffusion
    // =========================================================================
    //
    // We now run the *reverse* DDPM process starting from pure Gaussian noise
    // x_T ~ N(0, I) and iterating down to x_0, with the *fixed* class label
    // concatenated to the model input at every reverse step.
    //
    // WHY fix the label during sampling?
    //   We want the model to generate a specific digit (e.g. "3").  Providing
    //   the same one-hot label at every step consistently steers the denoising
    //   trajectory toward that class.
    // =========================================================================
    println!("\n=== Starting Class-Conditioned Sampling ===");

    // Choose the digit to generate (change this to any class 0–9).
    let target_digit = 3u8;
    println!("Generating digit: {}", target_digit);

    let num_samples = 1; // generate one image at a time

    // x_T ~ N(0, I): starting point of the reverse diffusion chain.
    let initial_noise = Tensor::randn(0.0f32, 1.0f32, (num_samples, img_dim), &device)?;

    // Build the one-hot conditioning vector for the target digit.
    // Shape: (num_samples=1, class_dim=10)
    // e.g. for digit 3: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    let target_one_hot = make_one_hot(&[target_digit], &device)?;

    // Run the shared class-conditioned reverse diffusion sampler.
    // This iterates from t = T-1 down to t = 0, predicting and subtracting
    // noise at each step while keeping the class vector fixed.
    let generated = sample_ddpm_cond(
        &mlp,
        &scheduler,
        initial_noise,
        img_dim,
        time_emb_dim,
        &target_one_hot,
        &device,
    )?;

    // Flatten the generated tensor to a 1-D pixel slice and encode as PNG.
    // The shared `save_png` helper converts [-1, 1] float values to u8 [0, 255].
    let final_pixels = generated.flatten_all()?.to_vec1::<f32>()?;
    save_png("mnist_cond_generated.png", &final_pixels)?;

    Ok(())
}
