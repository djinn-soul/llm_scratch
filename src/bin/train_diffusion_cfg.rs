// =============================================================================
// train_diffusion_cfg.rs — Classifier-Free Guidance (CFG) DDPM trainer
// =============================================================================
//
// This binary extends the class-conditioned DDPM (`train_diffusion_cond.rs`)
// with **Classifier-Free Guidance (CFG)** — a technique that amplifies the
// model's sensitivity to the class label at inference time, producing sharper
// and more class-faithful generated images without needing a separate
// classifier model.
//
// KEY IDEA — Label Dropout during training:
//   During training, each label is randomly *dropped* (zeroed out) with a
//   probability `label_dropout` (15% here).  This forces the model to learn
//   two conditional distributions simultaneously in a single network:
//
//     ε_θ(x_t, t, c)   — conditioned on class label c
//     ε_θ(x_t, t, ∅)   — unconditional (label replaced by all-zeros)
//
//   Because the model has seen both, at inference we can interpolate between
//   them to amplify the class signal.
//
// KEY IDEA — Guided noise prediction at inference:
//   The CFG-modified noise prediction is:
//
//     ε̂_guided = ε_cond + s * (ε_cond − ε_uncond)
//
//   where s is the guidance_scale.  A scale of 1.0 equals standard
//   conditional sampling; larger values (e.g. 3–10) push the output harder
//   toward the requested class at the cost of diversity.
//
// Paper reference: "Classifier-Free Diffusion Guidance" (Ho & Salimans, 2022)
//   https://arxiv.org/abs/2207.12598
// =============================================================================

use core::f32;

// anyhow provides ergonomic error propagation via the `?` operator.
use anyhow::Result;

// Candle tensor types used throughout.
use candle_core::{DType, Device, Tensor};

// `sample_ddpm_cfg` — the CFG-aware reverse diffusion sampler.
// Imported from the sampling submodule directly because the top-level
// re-export does not yet include it (it was added after the initial API).
use llm_scratch_rs::models::diffusion::sampling::sample_ddpm_cfg;

// Standard diffusion model components.
use llm_scratch_rs::models::diffusion::{
    get_time_embedding, BetaScheduler, DenoisingModel, MlpAdamOptimizer, SimpleDenoisingMlp,
};

// Shared MNIST download + parse helpers and PNG writer.
use llm_scratch_rs::utils::mnist_utils::{acquire_mnist, save_png};

// `rand::RngExt` provides the `.random::<T>()` method on the default RNG,
// used in `make_one_hot_cfg` to decide whether to drop each label.
use rand::RngExt;

// =============================================================================
// make_one_hot_cfg — one-hot encoder with stochastic label dropout
// =============================================================================
//
// WHY is this different from `make_one_hot` in mnist_utils?
//   Standard one-hot encoding always sets the class bit.
//   CFG training requires *randomly zeroing out* the label for a fraction of
//   training examples (`drop_rate` fraction).  When a label is dropped, the
//   model receives an all-zeros conditioning vector, which it learns to
//   interpret as "unconditional" — identical to the null class ∅.
//
// WHY 15% dropout rate?
//   Ho & Salimans found that ~10–20% gives a good trade-off between:
//     - Unconditional quality (needs enough dropped samples to learn from)
//     - Conditional quality  (needs enough labelled samples for guidance)
//
// Arguments:
//   labels    — slice of u8 class labels in {0, …, 9}
//   drop_rate — probability in [0, 1] that each label is zeroed out
//   device    — where to create the output tensor
//
// Output shape: (labels.len(), 10)
fn make_one_hot_cfg(labels: &[u8], drop_rate: f32, device: &Device) -> Result<Tensor> {
    let n = labels.len();
    let num_classes = 10;

    // Initialise the RNG once per call (cheap — just reads thread-local state).
    let mut rng = rand::rng();

    // Start with all zeros — the "null class" representation.
    let mut hot = vec![0.0f32; n * num_classes];

    for (i, &label) in labels.iter().enumerate() {
        // Draw a uniform float in [0, 1).
        // If it exceeds `drop_rate` we keep the label; otherwise we leave the
        // row as all-zeros (the label is "dropped").
        //
        // WHY > drop_rate rather than < drop_rate?
        //   `rng.random::<f32>()` returns a value in [0, 1).
        //   We want to KEEP the label with probability (1 - drop_rate), i.e.
        //   when the random value is greater than the drop threshold.
        if rng.random::<f32>() > drop_rate {
            let idx = (i * num_classes) + label as usize;
            hot[idx] = 1.0f32;
        }
        // If random <= drop_rate: label is dropped; row stays all-zeros.
    }

    Ok(Tensor::from_vec(hot, (n, num_classes), device)?)
}

// =============================================================================
// main — CFG training loop and guided image generation
// =============================================================================
//
// High-level flow:
//   1. Load MNIST images + labels.
//   2. Build the beta schedule, MLP (810-dim input), and Adam.
//   3. Training loop: corrupt image, drop labels stochastically, train.
//   4. After training, run CFG-guided reverse diffusion for a target digit.
//   5. Save generated + reference PNGs.
// =============================================================================
pub fn main() -> Result<()> {
    // --- Device selection ----------------------------------------------------
    // CPU for portability. Swap to Device::Cuda(0) for GPU acceleration.
    let device = Device::Cpu;
    println!("== Classifier-Free Guidance (CFG) DDPM Training ==");

    // --- Dataset loading -----------------------------------------------------
    // Both images and labels are required:
    //   images     — shape (60000, 784), pixel values in [-1, 1]
    //   labels_raw — 60000 u8 class labels in {0, …, 9}
    let (images, labels_raw) = acquire_mnist(&device)?;
    let total_samples = images.dim(0)?;
    println!("Total samples: {}", total_samples);

    // =========================================================================
    // Hyper-parameter configuration
    // =========================================================================
    //
    // steps (T)       — diffusion timesteps (100 for MNIST, 1000 in the paper).
    //
    // time_emb_dim    — sinusoidal time embedding size (16).
    //
    // class_dim       — one-hot label size (10 for MNIST digits 0-9).
    //
    // img_dim         — 28x28 = 784, flattened.
    //
    // hidden_dim      — MLP hidden layer width (512).
    //
    // batch_size      — 128 examples per gradient step.
    //
    // epochs          — 12 000 gradient steps. More than the unconditional model
    //                   because CFG requires learning both conditional and
    //                   unconditional behaviour simultaneously.
    //
    // lr              — Adam learning rate (0.001, standard default).
    //
    // label_dropout   — Fraction of labels zeroed during training (0.15 = 15%).
    //                   Creates the unconditional branch ε_θ(x_t, t, ∅) that
    //                   CFG guidance relies on at inference.
    // =========================================================================
    let steps = 100; // T: total diffusion timesteps
    let time_emb_dim = 16; // sinusoidal time embedding dimension
    let class_dim = 10; // one-hot label vector size (digits 0-9)
    let img_dim = 784; // 28x28 flattened image size
    let hidden_dim = 512; // MLP hidden layer width
    let batch_size = 128; // mini-batch size per gradient step
    let epochs = 12000; // total gradient update steps
    let lr = 0.001; // Adam learning rate
    let label_dropout = 0.15f32; // label drop probability for CFG training

    // --- Beta schedule -------------------------------------------------------
    // Linear schedule from beta_1=0.0001 to beta_T=0.02 (DDPM paper).
    // Pre-computes beta_t, alpha_t, alpha_bar_t, sigma_t for all t once.
    let scheduler = BetaScheduler::new(steps, 0.0001, 0.02, &device)?;

    // --- MLP architecture ---------------------------------------------------
    // Input = concat(x_t, time_embedding, label_one_hot)
    //       = 784 + 16 + 10 = 810 dimensions
    //
    // WHY the same architecture as the non-CFG conditional model?
    //   CFG is purely a *training procedure* and *sampling strategy* change.
    //   The network architecture (including input dimension) is identical.
    //   The model implicitly learns two behaviours from the same weights:
    //     - When the label slot has a 1: conditional denoising
    //     - When the label slot is all-zeros: unconditional denoising
    let mut mlp = SimpleDenoisingMlp::new(
        img_dim + time_emb_dim + class_dim, // 810
        hidden_dim,                         // 512
        img_dim,                            // 784
        &device,
    )?;

    // Adam optimizer — adaptive per-parameter learning rates.
    let mut optimizer = MlpAdamOptimizer::new(&mlp, lr)?;

    println!(
        "Starting training for {} epochs with {}% label dropout...\n",
        epochs,
        (label_dropout * 100.0) as u32
    );

    // =========================================================================
    // Training loop — CFG-style noise prediction with label dropout
    // =========================================================================
    //
    // The training objective is the same MSE noise-prediction loss as standard
    // DDPM/conditional DDPM:
    //
    //   L = E_{t, x_0, c, ε} [ || ε − ε_θ(x_t, t, c̃) ||² ]
    //
    // where c̃ is the label *after* possible dropout (c̃ = c or c̃ = ∅).
    //
    // By randomly substituting c with ∅, we train the model to also handle
    // the unconditional case.  The *same* network serves both roles, which is
    // more parameter-efficient than training two separate models.
    // =========================================================================
    for epoch in 1..=epochs {
        // ---------------------------------------------------------------------
        // Step 1 — Random mini-batch: images + matching labels
        // ---------------------------------------------------------------------
        // WHY random sampling?  SGD needs i.i.d. samples.  Random indices also
        // ensure the batch covers a variety of digit classes uniformly.
        let index_tensor = Tensor::rand(
            0.0f32,
            total_samples as f32 - 1e-4f32,
            (batch_size,),
            &device,
        )?
        .to_dtype(DType::U32)?;

        // Copy indices to CPU Vec so we can index into the Rust Vec<u8> labels.
        let indices = index_tensor.to_vec1::<u32>()?;

        // Gather batch images: shape (batch_size, 784).
        let x0 = images.index_select(&index_tensor, 0)?;

        // Gather labels that correspond exactly to the selected image indices.
        let batch_labels: Vec<u8> = indices.iter().map(|&x| labels_raw[x as usize]).collect();

        // ---------------------------------------------------------------------
        // Step 2 — Build the (stochastically dropped) one-hot label tensor
        // ---------------------------------------------------------------------
        // Each label is independently dropped with probability `label_dropout`.
        // Dropped labels become all-zeros rows (the unconditional signal ∅).
        // This is the ONLY difference between CFG training and standard
        // conditional training.
        let label_one_hot = make_one_hot_cfg(&batch_labels, label_dropout, &device)?;

        // ---------------------------------------------------------------------
        // Step 3 — Sample random diffusion timesteps and corrupt images
        // ---------------------------------------------------------------------
        // Same forward-diffusion procedure as unconditional and conditional DDPM:
        //   x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
        //
        // The model's target is the ground-truth noise epsilon ~ N(0, I).
        let t_float = Tensor::rand(0.0f32, steps as f32 - 1e-4f32, (batch_size,), &device)?;
        let t = t_float.to_dtype(DType::U32)?;
        let noise = Tensor::randn(0.0f32, 1.0f32, (batch_size, img_dim), &device)?;
        let xt = scheduler.add_noise(&x0, &noise, &t)?;

        // ---------------------------------------------------------------------
        // Step 4 — Build conditioned model input
        // ---------------------------------------------------------------------
        // v = concat(x_t, time_embedding, label_one_hot_or_null)
        // shape: (batch_size, 784 + 16 + 10) = (128, 810)
        //
        // When label_one_hot is all-zeros (dropped), the model implicitly
        // performs unconditional denoising — the key to CFG.
        let time_emb = get_time_embedding(&t, time_emb_dim)?;
        let v = Tensor::cat(&[&xt, &time_emb, &label_one_hot], 1)?;

        // ---------------------------------------------------------------------
        // Step 5 — Forward pass: predict noise epsilon_hat
        // ---------------------------------------------------------------------
        // The MLP returns pred, a1 (post-ReLU hidden activations), and
        // z1 (pre-ReLU), which are needed for the manual backward pass.
        let (pred, intermediates) = DenoisingModel::forward(&mlp, &v)?;

        // ---------------------------------------------------------------------
        // Step 6 — MSE loss: L = (1/N) sum (epsilon_hat - epsilon)^2
        // ---------------------------------------------------------------------
        // The scalar `loss` is used only for logging.
        // The actual gradient flows through `mlp.backward` below.
        let diff = pred.sub(&noise)?;
        let loss = diff.sqr()?.mean_all()?.to_scalar::<f32>()?;

        // ---------------------------------------------------------------------
        // Step 7 — Backpropagation: compute dL/dW for all parameters
        // ---------------------------------------------------------------------
        // Manual chain-rule backward pass (Candle autograd not used here).
        let grads = DenoisingModel::backward(&mlp, &v, &intermediates, &pred, &noise)?;

        // ---------------------------------------------------------------------
        // Step 8 — Adam optimizer step: theta <- theta - lr * Adam(grad)
        // ---------------------------------------------------------------------
        optimizer.step(&mut mlp, &grads)?;

        let param_names = mlp.param_names();

        // ---------------------------------------------------------------------
        // Step 9 — Periodic logging: MSE loss + gradient norms
        // ---------------------------------------------------------------------
        // Gradient norms reveal vanishing (<< expected) or exploding (>> expected)
        // gradients without requiring a separate validation loop.
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
    // Classifier-Free Guided reverse diffusion sampling
    // =========================================================================
    //
    // Now that the model is trained with label dropout, we run the reverse
    // diffusion process using the CFG guidance formula at every step:
    //
    //   epsilon_guided = epsilon_cond + s * (epsilon_cond - epsilon_uncond)
    //
    // The guidance scale `s` controls how strongly we push the output toward
    // the requested class:
    //   s = 1.0  — equivalent to standard conditional sampling (no boost)
    //   s = 3.0  — moderate guidance (good quality + class accuracy trade-off)
    //   s = 7+   — high guidance (very class-faithful but lower diversity)
    // =========================================================================
    println!("\n=== Starting Classifier-Free Guided Reverse Sampling ===");

    // Choose the digit class to generate.  Change this to any value in {0..9}.
    let target_digit = 3u8;
    // guidance_scale s = 3.0: a moderate boost that sharpens class fidelity
    // while keeping images reasonably diverse.
    let guidance_scale = 3.0f64;
    println!(
        "Generating digit: {} with guidance scale: {}",
        target_digit, guidance_scale
    );

    let num_samples = 1; // generate one image

    // x_T ~ N(0, I): pure Gaussian noise starting point for reverse diffusion.
    let initial_noise = Tensor::randn(0.0f32, 1.0f32, (num_samples, img_dim), &device)?;

    // Build the one-hot conditioning vector for the target class.
    // drop_rate = 0.0: we NEVER drop the label during inference, we always
    // want the model to know which class to generate.
    let target_one_hot = make_one_hot_cfg(&[target_digit], 0.0, &device)?;

    // Run the CFG-guided reverse diffusion sampler.
    // The sampler computes both the conditional and unconditional noise
    // predictions at every step and blends them using the guidance scale.
    let generated = sample_ddpm_cfg(
        &mlp,
        &scheduler,
        initial_noise,
        img_dim,
        time_emb_dim,
        &target_one_hot,
        guidance_scale,
        &device,
    )?;

    // Flatten x_0 to a 1-D pixel vector and encode as PNG.
    let final_pixels = generated.flatten_all()?.to_vec1::<f32>()?;
    save_png("mnist_cfg_generated.png", &final_pixels)?;

    // Save a real example of the target digit for side-by-side comparison.
    // `position` finds the first index in the label list that matches.
    if let Some(pos) = labels_raw.iter().position(|&lbl| lbl == target_digit) {
        let original_sample = images
            .index_select(&Tensor::new(&[pos as u32], &device)?, 0)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        save_png("mnist_original_3.png", &original_sample)?;
    }

    Ok(())
}
