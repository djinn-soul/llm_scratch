use anyhow::{bail, Ok, Result};
use candle_core::{Device, Tensor};

use super::{get_time_embedding, BetaScheduler, DenoisingModel};

// =============================================================================
// DIFFUSION SAMPLING — DDPM, DDIM, and Classifier-Free Guidance
// =============================================================================
//
// This module implements inference-time (reverse) sampling for diffusion models.
// All sampling functions accept `&dyn DenoisingModel` so they work with any
// noise-prediction architecture (MLP, UNet, Transformer, etc.).
//
// ── MATHEMATICAL BACKGROUND ──────────────────────────────────────────────────
//
// THE FORWARD PROCESS (training time — not implemented here, but is the
// foundation everything derives from):
//
//   Given a clean image x_0, the forward process adds Gaussian noise over
//   T timesteps.  At each step t the forward transition is:
//
//       q(x_t | x_{t-1}) = N(x_t; sqrt(1 - beta_t) * x_{t-1},  beta_t * I)
//
//   where beta_t is a small noise variance that increases with t.
//
//   Define:
//       alpha_t     = 1 - beta_t         (per-step signal retention)
//       alpha_bar_t = prod_{s=1}^{t} alpha_s   (cumulative signal retention)
//
//   A key property: we can jump directly from x_0 to ANY x_t without
//   iterating, via the closed-form "reparameterization trick":
//
//       q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)
//
//   Equivalently, sampling x_t from x_0 is just:
//
//       x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
//       where epsilon ~ N(0, I)
//
//   This is what the training code uses: sample t, sample epsilon, compute x_t,
//   and train the model to predict epsilon given (x_t, t).
//
// THE REVERSE PROCESS (this module):
//
//   Generation runs the chain BACKWARD: x_T -> x_{T-1} -> ... -> x_0.
//   We want to sample from q(x_{t-1} | x_t), but this requires knowing
//   q(x_t) which is intractable. Instead, we:
//
//   1. Approximate q(x_{t-1} | x_t) with a learned model p_theta(x_{t-1} | x_t).
//   2. The model predicts epsilon_theta(x_t, t), the noise present in x_t.
//   3. We then use this prediction to compute the reverse step.
//
// TWO REVERSE STRATEGIES:
//
//   DDPM (Ho et al. 2020):
//     Uses the exact reverse posterior q(x_{t-1} | x_t, x_0) which is
//     Gaussian when conditioned on x_0. The model's epsilon prediction
//     gives us an estimate of x_0, yielding a posterior mean + variance.
//     Stochastic: adds sigma_t * z at each step.
//
//   DDIM (Song et al. 2020):
//     Defines a NON-MARKOVIAN forward process that shares the same marginals
//     q(x_t | x_0) but allows deterministic reverse steps. The update
//     "predicts x_0, then re-noises" to the next level without injecting
//     fresh randomness. Supports arbitrary stride jumps with no quality loss.
//
pub fn sample_ddpm(
    model: &dyn DenoisingModel,
    scheduler: &BetaScheduler,
    num_samples: usize,
    img_dim: usize,
    time_emb_dim: usize,
    device: &Device,
) -> Result<Tensor> {
    // Start from x_T ~ N(0, I). The reverse chain will denoise this into x_0.
    let xt = Tensor::randn(0.0f32, 1.0f32, (num_samples, img_dim), device)?;

    sample_ddpm_from_noise(model, scheduler, xt, img_dim, time_emb_dim, device)
}

pub fn sample_ddpm_from_noise(
    model: &dyn DenoisingModel,
    scheduler: &BetaScheduler,
    mut xt: Tensor,
    img_dim: usize,
    time_emb_dim: usize,
    device: &Device,
) -> Result<Tensor> {
    let num_samples = xt.dim(0)?;

    // Pre-extract schedule vectors as plain Rust Vecs.
    //
    // These are the key schedule quantities, all length T:
    //   betas[t]         = beta_t          — scheduled noise variance at step t
    //   alphas[t]        = alpha_t = 1-beta_t — signal retention at step t
    //   alphas_cumprod[t] = alpha_bar_t = product(alpha_1..alpha_t) — cumulative signal
    //   sigmas[t]        = sqrt(beta_t * (1-alpha_bar_{t-1}) / (1-alpha_bar_t))
    //                      — reverse posterior std dev
    //
    // Converting once avoids repeated Tensor slicing inside every denoising step.
    let betas = scheduler.betas.to_vec1::<f32>()?;
    let alphas = scheduler.alphas.to_vec1::<f32>()?;
    let alphas_cumprod = scheduler.alphas_cumprod.to_vec1::<f32>()?;
    let sigmas = scheduler.sigmas.to_vec1::<f32>()?;

    // Reverse diffusion loop: t = T-1 -> 0.
    //
    // WHY iterate in reverse?
    //   The forward process adds noise progressively: x_0 -> x_1 -> ... -> x_T.
    //   Generation must undo this: x_T -> x_{T-1} -> ... -> x_0.
    //   At each step t, the model looks at x_t and estimates what noise was
    //   added, then subtracts (a scaled version of) that estimate to produce
    //   x_{t-1}, which is slightly less noisy.
    for t_step in (0..scheduler.steps).rev() {
        xt = ddpm_reverse_step(
            model,
            &xt,
            t_step,
            num_samples,
            img_dim,
            time_emb_dim,
            betas.as_slice(),
            alphas.as_slice(),
            alphas_cumprod.as_slice(),
            sigmas.as_slice(),
            device,
        )?;
    }

    Ok(xt)
}

fn ddpm_reverse_step(
    model: &dyn DenoisingModel,
    xt: &Tensor,
    t_step: usize,
    num_samples: usize,
    img_dim: usize,
    time_emb_dim: usize,
    betas: &[f32],
    alphas: &[f32],
    alphas_cumprod: &[f32],
    sigmas: &[f32],
    device: &Device,
) -> Result<Tensor> {
    // Step R1: Build a batch-shaped timestep tensor.
    //
    // We need a Tensor, not a scalar, because get_time_embedding expects a
    // 1-D tensor of shape [num_samples]. For one generated image this is
    // [t_step]; for N images this is [t_step, t_step, ..., t_step].
    let t_vec = vec![t_step as u32; num_samples];
    let t_tensor = Tensor::new(t_vec.as_slice(), device)?;

    // Step R2: Compute sinusoidal time embedding for this exact reverse step.
    //
    // WHY the same embedding function as training?
    //
    // Consistency is critical. The model was trained with this exact timestep
    // representation, so inference must use the same representation.
    let time_emb = get_time_embedding(&t_tensor, time_emb_dim)?;

    // Step R3: Concatenate current x_t with the time embedding.
    //
    // This matches the exact model input format used during training:
    //
    // v = concat(x_t, time_embedding)
    let v = Tensor::cat(&[xt, &time_emb], 1)?;

    // Step R4: Ask the model to predict the noise epsilon_hat in x_t.
    //
    // We discard the intermediate activations because sampling is
    // inference only. There is no backward pass here.
    let (pred_noise, _intermediates) = model.forward(&v)?;

    // Step R5: Retrieve precomputed schedule coefficients for this timestep.
    //
    //   beta_t     — noise variance scheduled for this step
    //   alpha_t    = 1 - beta_t  — fraction of signal retained at this step
    //   alpha_bar_t = alpha_1 * alpha_2 * ... * alpha_t  — cumulative signal
    //   sigma_t    — std dev of the reverse posterior (see derivation below)
    let beta = betas[t_step];
    let alpha = alphas[t_step];
    let alpha_bar = alphas_cumprod[t_step];
    let sigma = sigmas[t_step];

    // Step R6: Compute the epsilon coefficient.
    //
    //   eps_coef = beta_t / sqrt(1 - alpha_bar_t)
    //
    // ── DERIVATION ──
    //
    // The true reverse posterior (when x_0 is known) is:
    //
    //   q(x_{t-1} | x_t, x_0) = N(x_{t-1}; mu_tilde_t, sigma_tilde_t^2 I)
    //
    // where the posterior mean is:
    //
    //   mu_tilde_t = [sqrt(alpha_bar_{t-1}) * beta_t / (1-alpha_bar_t)] * x_0
    //              + [sqrt(alpha_t) * (1-alpha_bar_{t-1}) / (1-alpha_bar_t)] * x_t
    //
    // But we don't know x_0. We only know x_t and the model's prediction
    // epsilon_theta. Since x_t = sqrt(alpha_bar_t)*x_0 + sqrt(1-alpha_bar_t)*epsilon,
    // we can solve for x_0:
    //
    //   x_0 = (x_t - sqrt(1-alpha_bar_t) * epsilon) / sqrt(alpha_bar_t)
    //
    // Substituting this into the posterior mean and simplifying yields the
    // "epsilon parameterization" of the mean:
    //
    //   mu_theta = (1/sqrt(alpha_t)) * (x_t - [beta_t/sqrt(1-alpha_bar_t)] * epsilon_theta)
    //                                         └─────────── eps_coef ────────────┘
    //
    // So eps_coef = beta_t / sqrt(1-alpha_bar_t) is the coefficient that
    // converts the model's noise prediction into the correct "noise to subtract"
    // at this specific timestep.
    let eps_coef = reverse_epsilon_coefficient(beta, alpha_bar, t_step)?;

    // Step R7: Compute the DDPM reverse posterior mean.
    //
    //   mu_theta(x_t, t) = (1/sqrt(alpha_t)) * (x_t - eps_coef * epsilon_theta)
    //
    // ── DERIVATION (continued from Step R6) ──
    //
    // Starting from:
    //   mu_theta = (1/sqrt(alpha_t)) * (x_t - [beta_t / sqrt(1-alpha_bar_t)] * eps_theta)
    //
    // This has two multiplicative factors:
    //
    //   INNER: (x_t - eps_coef * eps_theta)
    //     → Subtracts the scaled noise prediction from x_t. This removes the
    //       model's estimate of the noise that was added at step t. The result
    //       is a "partially denoised" signal, but still scaled by sqrt(alpha_t).
    //
    //   OUTER: 1/sqrt(alpha_t)
    //     → Recall q(x_t | x_{t-1}) = N(sqrt(alpha_t)*x_{t-1}, beta_t*I).
    //       The forward process multiplied the signal by sqrt(alpha_t), so
    //       dividing by sqrt(alpha_t) undoes that per-step signal scaling.
    //       Together, inner × outer yields the posterior mean at t-1.
    //
    // Numerical example (t=50, cosine schedule):
    //   If alpha_t ≈ 0.98, alpha_bar_t ≈ 0.3, beta_t ≈ 0.02, then:
    //   eps_coef = 0.02 / sqrt(0.7) ≈ 0.024
    //   1/sqrt(alpha_t) = 1/sqrt(0.98) ≈ 1.01
    //   The net effect: subtract ~2.4% of the predicted noise, then re-scale
    //   by ~1%. Each step makes a small correction.
    let mean = xt
        .sub(&pred_noise.affine(eps_coef as f64, 0.0)?)?
        .affine((1.0 / alpha.sqrt()) as f64, 0.0)?;

    // Step R8: Add stochastic noise for all steps except the final step.
    //
    // The full DDPM reverse step is:
    //
    //   x_{t-1} = mu_theta(x_t, t) + sigma_t * z,   z ~ N(0, I)
    //
    // ── WHY add sigma_t * z? ──
    //
    // The true reverse posterior q(x_{t-1} | x_t, x_0) is a Gaussian with:
    //   mean     = mu_tilde_t     (approximated by mu_theta above)
    //   variance = sigma_tilde_t^2 = beta_t * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)
    //
    // If we only output the mean, we'd get a DETERMINISTIC decoder that
    // always produces the same image from the same x_T. Adding sigma_t * z
    // samples from the full posterior, introducing the randomness needed for
    // diverse outputs. Without it, all samples converge to the mode (mean)
    // of the learned distribution — an over-smoothed average.
    //
    // ── WHY no noise at t = 0? ──
    //
    // At the final step we output x_0 directly. The posterior variance at
    // t=0 involves alpha_bar_{-1} which is undefined; conceptually, there's
    // no "noise level below zero" to sample from. Adding fresh noise would
    // just corrupt the final output.
    if t_step > 0 {
        // z ~ N(0, I): fresh independent noise, sampled EVERY step.
        // This z is independent of the z used in every other step.
        let z = Tensor::randn(0.0f32, 1.0f32, (num_samples, img_dim), device)?;

        // x_{t-1} = mu_theta + sigma_t * z
        Ok(mean.add(&z.affine(sigma as f64, 0.0)?)?)
    } else {
        // Final step: output deterministic mean as x_0.
        Ok(mean)
    }
}

// =============================================================================
// sample_ddpm_cond — class-conditioned DDPM reverse sampler
// =============================================================================
//
// This is the conditioned counterpart to `sample_ddpm_from_noise`.
// The only difference is that the model receives an extra class conditioning
// vector (one-hot label) concatenated to its input at every reverse step.
//
// Model input at each step:
//   v = concat(x_t, time_embedding, class_one_hot)
//       shape: (num_samples, img_dim + time_emb_dim + class_dim)
//
// WHY is the class vector the same at every step?
//   The target class does not change during sampling — we want to generate a
//   specific digit (e.g. "3") throughout the entire denoising chain.
//   Keeping the same conditioning vector guides the model to denoise toward
//   that class at every single reverse step.
//
// Arguments:
//   model            — any trained denoising model implementing DenoisingModel
//   scheduler        — beta/alpha schedule (same as training)
//   xt               — initial noise tensor x_T, shape (num_samples, img_dim)
//   img_dim          — flattened image size (784 for MNIST)
//   time_emb_dim     — sinusoidal time embedding dimension (must match training)
//   class_one_hot    — one-hot class vector, shape (num_samples, num_classes)
//   device           — CPU or CUDA device
pub fn sample_ddpm_cond(
    model: &dyn DenoisingModel,
    scheduler: &BetaScheduler,
    mut xt: Tensor,
    img_dim: usize,
    time_emb_dim: usize,
    class_one_hot: &Tensor,
    device: &Device,
) -> Result<Tensor> {
    let num_samples = xt.dim(0)?;
    let class_one_hot_broadcasted;
    let class_one_hot = if class_one_hot.dim(0)? == 1 && num_samples > 1 {
        class_one_hot_broadcasted = class_one_hot.broadcast_as((num_samples, class_one_hot.dim(1)?))?;
        &class_one_hot_broadcasted
    } else {
        class_one_hot
    };

    // Pre-extract all schedule coefficients into plain Vecs for fast indexed access.
    // WHY Vec and not Tensor indexing?
    //   Tensor slicing at runtime has overhead. Since the schedule is small
    //   (e.g. 100 elements) and fixed, a Vec<f32> lookup is cheapest.
    let betas = scheduler.betas.to_vec1::<f32>()?;
    let alphas = scheduler.alphas.to_vec1::<f32>()?;
    let alphas_cumprod = scheduler.alphas_cumprod.to_vec1::<f32>()?;
    let sigmas = scheduler.sigmas.to_vec1::<f32>()?;

    // Iterate reverse diffusion steps from T-1 down to 0.
    // WHY reverse? The forward process goes x_0 → x_T (adds noise).
    //              The reverse process goes x_T → x_0 (removes noise).
    for t_step in (0..scheduler.steps).rev() {
        // --- Step 1: Build timestep tensor -----------------------------------
        // Same t value repeated for every sample in the batch.
        let t_vec = vec![t_step as u32; num_samples];
        let t_tensor = Tensor::new(t_vec.as_slice(), device)?;

        // --- Step 2: Sinusoidal time embedding --------------------------------
        // Maps the integer timestep to a smooth continuous vector.
        // Must match the embedding used during training — consistency is critical.
        let time_emb = get_time_embedding(&t_tensor, time_emb_dim)?;

        // --- Step 3: Build the conditioned model input -----------------------
        // v = concat(x_t, time_embedding, class_one_hot)
        //
        // WHY concatenate the one-hot vector?
        //   The class label tells the model *which* digit to denoise toward.
        //   Concatenation is the simplest fusion mechanism for an MLP and is
        //   consistent with how the model was trained.
        let v = Tensor::cat(&[&xt, &time_emb, class_one_hot], 1)?;

        // --- Step 4: Predict noise -------------------------------------------
        // epsilon_hat = epsilon_theta(x_t, t, c)
        // The model's estimate of the added noise, conditioned on timestep t
        // and class label c (embedded as the one-hot vector).
        let (pred_noise, _intermediates) = model.forward(&v)?;

        // --- Step 5: Retrieve schedule coefficients for this step -----------
        let beta = betas[t_step];
        let alpha = alphas[t_step];
        let alpha_bar = alphas_cumprod[t_step];
        let sigma = sigmas[t_step];

        // --- Step 6: Epsilon scaling coefficient ----------------------------
        // eps_coef = beta_t / sqrt(1 - alpha_bar_t)
        // Re-scales the noise prediction to the correct amplitude for this step.
        let eps_coef = reverse_epsilon_coefficient(beta, alpha_bar, t_step)?;

        // --- Step 7: Reverse posterior mean ---------------------------------
        // mu_theta(x_t, t) = (1/sqrt(alpha_t)) * (x_t - eps_coef * epsilon_hat)
        let mean = xt
            .sub(&pred_noise.affine(eps_coef as f64, 0.0)?)?
            .affine((1.0 / alpha.sqrt()) as f64, 0.0)?;

        // --- Step 8: Stochastic sampling (all steps except the last) --------
        // Add sigma_t * z to preserve the Gaussian variance of the reverse posterior.
        // At t=0 we output the deterministic mean — no further noise is needed.
        if t_step > 0 {
            let z = Tensor::randn(0.0f32, 1.0f32, (num_samples, img_dim), device)?;
            // x_{t-1} = mean + sigma_t * z
            xt = mean.add(&z.affine(sigma as f64, 0.0)?)?;
        } else {
            // Final reverse step: x_0 = mean (deterministic)
            xt = mean;
        }
    }

    Ok(xt)
}

// =============================================================================
// sample_ddpm_cfg — Classifier-Free Guidance (CFG) reverse diffusion sampler
// =============================================================================
//
// ── WHAT IS CLASSIFIER-FREE GUIDANCE (CFG)? ──
//
// CFG (Ho & Salimans, 2022) is an inference-time technique that amplifies the
// model's class signal without requiring a separate classifier network.
//
// ── MATHEMATICAL DERIVATION ──
//
// The idea comes from classifier guidance (Dhariwal & Nichol, 2021), which
// modifies the score function using a trained classifier:
//
//   score_guided = score_uncond + s * grad_x(log p(c | x_t))
//
// The insight of CFG: we can AVOID training a separate classifier by noting
// that Bayes' rule gives us:
//
//   grad_x(log p(c | x_t)) = grad_x(log p(x_t | c)) - grad_x(log p(x_t))
//
// In the epsilon-prediction framework, the score is related to epsilon by:
//   score(x_t) = -epsilon(x_t, t) / sqrt(1 - alpha_bar_t)
//
// So the classifier gradient becomes:
//   grad_x(log p(c|x_t)) ∝ epsilon_uncond(x_t,t) - epsilon_cond(x_t,t,c)
//
// Substituting into the guided score and converting back to epsilon space:
//
//   epsilon_guided = epsilon_uncond + s * (epsilon_cond - epsilon_uncond)
//
// Expanding algebraically:
//   epsilon_guided = (1 - s) * epsilon_uncond + s * epsilon_cond
//
// KEY BOUNDARY VALUES:
//   s = 0.0 → epsilon_guided = epsilon_uncond  (purely unconditional)
//   s = 1.0 → epsilon_guided = epsilon_cond    (ordinary conditional)
//   s > 1.0 → EXTRAPOLATION beyond conditional (amplified class signal)
//
// ── WHY DOES s > 1 PRODUCE SHARPER CLASS FEATURES? ──
//
// The vector (epsilon_cond - epsilon_uncond) is the "class direction" in
// noise space — it's what makes the model's prediction different when it
// knows the class vs. when it doesn't. Scaling by s > 1 amplifies this
// direction, pushing the denoising trajectory further toward the class
// manifold. The cost: reduced diversity, since all samples are pulled
// harder toward the class prototype.
//
// ── WHY DOES THIS WORK WITHOUT A CLASSIFIER? ──
//
// During training, 15% of labels are randomly replaced with the null vector
// (all zeros). This means the SAME model learns both:
//   epsilon_theta(x_t, t, c)    — conditional prediction (label present)
//   epsilon_theta(x_t, t, null) — unconditional prediction (label dropped)
//
// At inference, we evaluate the model TWICE per step (or batch them) to get
// both predictions, then blend them using the formula above.
//
// Arguments:
//   model          — any trained CFG-aware model implementing DenoisingModel
//   scheduler      — pre-computed noise schedule (same as training)
//   xt             — initial noise tensor x_T ~ N(0,I), shape (N, img_dim)
//   img_dim        — flattened image size (784 for MNIST)
//   time_emb_dim   — sinusoidal time embedding size (must match training)
//   class_one_hot  — target class one-hot vector, shape (N, num_classes)
//   guidance_scale — guidance strength s (1.0 = no extra guidance, 3-7 typical)
//   device         — CPU or CUDA device
pub fn sample_ddpm_cfg(
    model: &dyn DenoisingModel,
    scheduler: &BetaScheduler,
    xt: Tensor,
    img_dim: usize,
    time_emb_dim: usize,
    class_one_hot: &Tensor,
    guidance_scale: f64,
    device: &Device,
) -> Result<Tensor> {
    // Preserve the original public API: callers that do not request a partial
    // trajectory always begin at the scheduler's final timestep.
    sample_ddpm_cfg_from_timestep(
        model,
        scheduler,
        xt,
        scheduler.steps - 1,
        img_dim,
        time_emb_dim,
        class_one_hot,
        guidance_scale,
        device,
    )
}

// Run classifier-free guidance from an explicit timestep without collecting
// intermediate frames. This is useful for reconstruction diagnostics where xt
// was created at a known t and must not be denoised as if it came from T - 1.
pub fn sample_ddpm_cfg_from_timestep(
    model: &dyn DenoisingModel,
    scheduler: &BetaScheduler,
    xt: Tensor,
    start_timestep: usize,
    img_dim: usize,
    time_emb_dim: usize,
    class_one_hot: &Tensor,
    guidance_scale: f64,
    device: &Device,
) -> Result<Tensor> {
    sample_ddpm_cfg_from_timestep_with_callback(
        model,
        scheduler,
        xt,
        start_timestep,
        img_dim,
        time_emb_dim,
        class_one_hot,
        guidance_scale,
        device,
        |_, _| Ok(()),
    )
}

// Backward-compatible full-chain callback wrapper. Keeping delegation here
// gives every CFG entrypoint one implementation of the reverse mathematics.
pub fn sample_ddpm_cfg_with_callback<F>(
    model: &dyn DenoisingModel,
    scheduler: &BetaScheduler,
    xt: Tensor,
    img_dim: usize,
    time_emb_dim: usize,
    class_one_hot: &Tensor,
    guidance_scale: f64,
    device: &Device,
    on_step: F,
) -> Result<Tensor>
where
    F: FnMut(usize, &Tensor) -> Result<()>,
{
    sample_ddpm_cfg_from_timestep_with_callback(
        model,
        scheduler,
        xt,
        scheduler.steps - 1,
        img_dim,
        time_emb_dim,
        class_one_hot,
        guidance_scale,
        device,
        on_step,
    )
}

// Core CFG reverse loop, respaced to an arbitrary number of steps.
//
// `start_timestep` is inclusive. When `num_inference_steps` equals
// start_timestep + 1 this walks every raw timestep, matching the original
// fixed-schedule DDPM loop exactly. For smaller counts it follows the Nichol
// & Dhariwal ("Improved DDPM") respacing trick: at each subsequence jump
// t_i -> t_{i+1}, treat the pair as if it were one adjacent step by deriving
// a synthetic beta/alpha from the ratio of their alpha_bars. This keeps the
// exact posterior-mean formula valid across skipped timesteps.
//
// The callback receives a forward-running frame index (0, 1, ...) rather than
// the decreasing diffusion timestep, which makes filenames naturally sortable.
pub fn sample_ddpm_cfg_strided_with_callback<F>(
    model: &dyn DenoisingModel,
    scheduler: &BetaScheduler,
    mut xt: Tensor,
    start_timestep: usize,
    num_inference_steps: usize,
    img_dim: usize,
    time_emb_dim: usize,
    class_one_hot: &Tensor,
    guidance_scale: f64,
    device: &Device,
    mut on_step: F,
) -> Result<Tensor>
where
    F: FnMut(usize, &Tensor) -> Result<()>,
{
    if start_timestep >= scheduler.steps {
        bail!(
            "start timestep {} is outside scheduler range 0..{}",
            start_timestep,
            scheduler.steps
        );
    }
    let num_samples = xt.dim(0)?;
    let class_one_hot_broadcasted;
    let class_one_hot = if class_one_hot.dim(0)? == 1 && num_samples > 1 {
        class_one_hot_broadcasted = class_one_hot.broadcast_as((num_samples, class_one_hot.dim(1)?))?;
        &class_one_hot_broadcasted
    } else {
        class_one_hot
    };

    // Pre-extract schedule coefficients into plain Vecs for O(1) indexed access.
    // Avoids repeated Tensor slicing inside the per-step loop.
    let alphas_cumprod = scheduler.alphas_cumprod.to_vec1::<f32>()?;

    // Build the "null" (unconditional) conditioning vector once.
    // Shape matches class_one_hot but every element is 0.0.
    //
    // WHY all-zeros for the null class?
    //   During training, dropped labels were represented as all-zeros rows.
    //   The model learned to associate all-zeros with "unconditional" denoising.
    //   Using the same convention at inference ensures it activates the correct
    //   unconditional behaviour.
    let null_one_hot = Tensor::zeros(class_one_hot.dims(), class_one_hot.dtype(), device)?;

    let timesteps = strided_timesteps(start_timestep, num_inference_steps);

    // --- Reverse diffusion loop over the (possibly strided) subsequence ----
    // WHY iterate in reverse? The forward process adds noise (x_0 → x_T).
    //                         The reverse process removes noise (x_T → x_0).
    for (frame_idx, &t_step) in timesteps.iter().enumerate() {
        // Step 1: Build the timestep tensor for this reverse step.
        // Same value repeated for all samples in the batch.
        let t_vec = vec![t_step as u32; num_samples];
        let t_tensor = Tensor::new(t_vec.as_slice(), device)?;

        // Step 2: Sinusoidal time embedding — must match the one used at training.
        let time_emb = get_time_embedding(&t_tensor, time_emb_dim)?;

        // Step 3: Build BOTH the conditional and unconditional model inputs.
        //
        // v_cond:  concat(x_t, time_emb, class_one_hot)  — the "guided" input
        // v_null:  concat(x_t, time_emb, null_one_hot)   — the "free" input
        //
        // WHY send x_t and time_emb to both?
        //   The only difference between the two predictions should be the label.
        //   Keeping x_t and time_emb identical ensures the class slot is the
        //   sole source of difference, making the guidance direction clean.
        let v_cond = Tensor::cat(&[&xt, &time_emb, class_one_hot], 1)?;
        let v_null = Tensor::cat(&[&xt, &time_emb, &null_one_hot], 1)?;

        // Step 4: One batched forward pass for both predictions.
        //
        // We need epsilon_cond and epsilon_uncond separately to compute the
        // guidance direction. Rather than evaluate the model twice, stack the
        // conditional and unconditional inputs along the batch dim and run a
        // single forward — forward is batch-agnostic, so this halves the
        // per-step model evaluations (the dominant sampling cost).
        //
        // Note: intermediate activations are discarded — we're at inference.
        let v_batched = Tensor::cat(&[&v_cond, &v_null], 0)?;
        let (pred_batched, _) = model.forward(&v_batched)?;
        let pred_cond = pred_batched.narrow(0, 0, num_samples)?;
        let pred_uncond = pred_batched.narrow(0, num_samples, num_samples)?;

        // Step 5: Compute the CFG-modified noise prediction.
        //
        // Standard CFG formula:
        //   epsilon_guided = epsilon_uncond + s * (epsilon_cond - epsilon_uncond)
        //
        // Expanded:
        //   epsilon_guided = s * epsilon_cond + (1 - s) * epsilon_uncond
        //
        // s=0 is unconditional, s=1 is ordinary conditional prediction, and
        // s>1 extrapolates toward stronger class conditioning.
        let pred_noise = combine_cfg_predictions(&pred_cond, &pred_uncond, guidance_scale)?;

        // Step 6: Derive respaced schedule coefficients for this jump.
        //
        // ── RESPACING DERIVATION (Nichol & Dhariwal, "Improved DDPM") ──
        //
        // The original DDPM uses one step per timestep: t -> t-1.
        // Respacing lets us skip timesteps: t_i -> t_{i+1} where t_{i+1}
        // may be many raw steps away. The trick is to compute a SYNTHETIC
        // beta/alpha that makes the posterior-mean formula valid for the jump.
        //
        // Key identity: for adjacent steps, alpha_bar_t = alpha_bar_{t-1} * alpha_t.
        // Rearranging:  alpha_t = alpha_bar_t / alpha_bar_{t-1}
        //               beta_t  = 1 - alpha_t = 1 - alpha_bar_t / alpha_bar_{t-1}
        //
        // For a JUMP from timestep t to timestep t_prev (possibly many steps away):
        //   synthetic_alpha = alpha_bar_t / alpha_bar_{t_prev}
        //   synthetic_beta  = 1 - synthetic_alpha
        //
        // This is equivalent to treating the entire jump as a SINGLE step with
        // its own effective beta. The posterior mean formula from Step R7 of
        // ddpm_reverse_step remains valid because it only depends on the ratio
        // of cumulative alpha products.
        //
        // When num_inference_steps == start_timestep + 1 (no skipping), these
        // reduce exactly to the original per-timestep beta/alpha values.
        //
        //   alpha_bar_prev = 1.0 when we've reached the end (x_0, pure signal)
        let alpha_bar = alphas_cumprod[t_step];
        let alpha_bar_prev = match timesteps.get(frame_idx + 1) {
            Some(&t_prev) => alphas_cumprod[t_prev],
            None => 1.0,
        };
        let beta = 1.0 - alpha_bar / alpha_bar_prev;
        let alpha = 1.0 - beta;
        // Posterior standard deviation for this (possibly skipped) jump:
        //
        //   sigma^2 = beta * (1 - alpha_bar_prev) / (1 - alpha_bar)
        //
        // ── DERIVATION ──
        // The exact reverse posterior variance is:
        //   sigma_tilde_t^2 = [beta_t * (1 - alpha_bar_{t-1})] / (1 - alpha_bar_t)
        //
        // Here beta/alpha_bar/alpha_bar_prev are the SYNTHETIC values for
        // this jump, so the formula applies directly to skipped steps too.
        let sigma = (beta * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)).sqrt();

        // Step 7: Recover x_0, constrain it to the normalized image domain,
        // then use the exact DDPM posterior mean. The algebraically equivalent
        // epsilon-only form can magnify small high-noise prediction errors far
        // outside [-1, 1], especially at the terminal cosine step.
        let mean =
            clipped_ddpm_posterior_mean(&xt, &pred_noise, beta, alpha, alpha_bar, alpha_bar_prev)?;

        // Step 9: Add stochasticity for all non-final steps.
        //
        // WHY add noise except on the last subsequence entry?
        //   The true reverse posterior is Gaussian with variance sigma_t^2.
        //   Adding sigma_t * z samples from this posterior correctly. On the
        //   last entry alpha_bar_prev = 1.0, so sigma is exactly 0 and no
        //   noise would be added regardless — the branch just skips the
        //   wasted randn call.
        if frame_idx + 1 < timesteps.len() {
            // z ~ N(0, I): independent noise for this reverse step.
            let z = Tensor::randn(0.0f32, 1.0f32, (num_samples, img_dim), device)?;
            // x_{t-1} = mean + sigma_t * z
            xt = mean.add(&z.affine(sigma as f64, 0.0)?)?;
        } else {
            // Final step: output the deterministic mean as x_0.
            xt = mean;
        }

        on_step(frame_idx, &xt)?;
    }
    Ok(xt)
}

pub fn sample_ddpm_cfg_from_timestep_with_callback<F>(
    model: &dyn DenoisingModel,
    scheduler: &BetaScheduler,
    xt: Tensor,
    start_timestep: usize,
    img_dim: usize,
    time_emb_dim: usize,
    class_one_hot: &Tensor,
    guidance_scale: f64,
    device: &Device,
    on_step: F,
) -> Result<Tensor>
where
    F: FnMut(usize, &Tensor) -> Result<()>,
{
    // Full-resolution DDPM: one reverse step per timestep, start_timestep -> 0.
    sample_ddpm_cfg_strided_with_callback(
        model,
        scheduler,
        xt,
        start_timestep,
        start_timestep + 1,
        img_dim,
        time_emb_dim,
        class_one_hot,
        guidance_scale,
        device,
        on_step,
    )
}

fn combine_cfg_predictions(
    conditional: &Tensor,
    unconditional: &Tensor,
    guidance_scale: f64,
) -> Result<Tensor> {
    // CFG blending formula:
    //
    //   eps_guided = eps_uncond + s * (eps_cond - eps_uncond)
    //
    // Algebraic expansion:
    //   eps_guided = (1-s)*eps_uncond + s*eps_cond
    //
    // Verification of boundary values (these are valuable invariants for testing):
    //   s=0:  eps_guided = eps_uncond                    (unconditional)
    //   s=1:  eps_guided = eps_cond                      (standard conditional)
    //   s=2:  eps_guided = 2*eps_cond - eps_uncond        (double guidance)
    //   s=-1: eps_guided = 2*eps_uncond - eps_cond        (anti-guidance)
    //
    // The code computes: unconditional + scale * (conditional - unconditional)
    // which is exactly the formula above.
    Ok(unconditional.add(
        &conditional
            .sub(unconditional)?
            .affine(guidance_scale, 0.0)?,
    )?)
}

fn clipped_ddpm_posterior_mean(
    xt: &Tensor,
    predicted_noise: &Tensor,
    beta: f32,
    alpha: f32,
    alpha_bar: f32,
    alpha_bar_prev: f32,
) -> Result<Tensor> {
    // ── STEP A: Recover the clean-image estimate x0_hat ──
    //
    // From the forward process reparameterization:
    //   x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
    //
    // Solving for x_0:
    //   x_0 = (x_t - sqrt(1 - alpha_bar_t) * epsilon) / sqrt(alpha_bar_t)
    //
    // Substituting the model's prediction epsilon_theta for epsilon:
    //   x0_hat = (x_t - sqrt(1 - alpha_bar_t) * eps_theta) / sqrt(alpha_bar_t)
    //
    // ── WHY CLAMP x0_hat to [-1, 1]? ──
    //
    // When alpha_bar_t is very small (high noise, late timesteps, especially
    // with a cosine schedule), dividing by sqrt(alpha_bar_t) amplifies any
    // error in the noise prediction ENORMOUSLY. For example:
    //   alpha_bar_t = 0.0001 → dividing by sqrt(0.0001) = 0.01 → 100x amplification
    //
    // A small prediction error of 0.01 becomes an x0_hat value of ±1.0 — and
    // larger errors push it to ±10 or ±100. Since training images live in
    // [-1, 1], values outside this range are nonsensical and will corrupt
    // the posterior mean calculation.
    //
    // Clamping is the MNIST-scale equivalent of "dynamic thresholding" used
    // in larger diffusion models (Imagen, etc.).
    let predicted_x0 = xt
        .sub(&predicted_noise.affine((1.0 - alpha_bar).sqrt() as f64, 0.0)?)?
        .affine((1.0 / alpha_bar.sqrt()) as f64, 0.0)?
        .clamp(-1.0f32, 1.0f32)?;

    // ── STEP B: Compute the exact reverse posterior mean ──
    //
    // The true reverse posterior q(x_{t-1} | x_t, x_0) has mean:
    //
    //   mu_tilde_t = c_x0 * x_0 + c_xt * x_t
    //
    // where:
    //   c_x0 = sqrt(alpha_bar_{t-1}) * beta_t / (1 - alpha_bar_t)
    //   c_xt = sqrt(alpha_t) * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)
    //
    // ── DERIVATION ──
    //
    // Starting from Bayes' rule on Gaussians:
    //   q(x_{t-1}|x_t,x_0) ∝ q(x_t|x_{t-1}) * q(x_{t-1}|x_0)
    //
    // Both are Gaussian, so the product is Gaussian. Completing the square
    // in the exponent gives:
    //
    //   mu_tilde = [sqrt(alpha_t)*(1-alpha_bar_{t-1})/(1-alpha_bar_t)] * x_t
    //            + [sqrt(alpha_bar_{t-1})*beta_t/(1-alpha_bar_t)] * x_0
    //
    // Note: c_x0 + c_xt ≠ 1 in general — this is NOT a convex combination.
    // The weights depend on the signal-to-noise ratios at t and t-1.
    //
    // We substitute x0_hat (the clamped prediction) for x_0.
    let denominator = 1.0 - alpha_bar;
    let x0_coefficient = beta * alpha_bar_prev.sqrt() / denominator;
    let xt_coefficient = (1.0 - alpha_bar_prev) * alpha.sqrt() / denominator;

    // mu = c_x0 * x0_hat + c_xt * x_t
    predicted_x0
        .affine(x0_coefficient as f64, 0.0)?
        .add(&xt.affine(xt_coefficient as f64, 0.0)?)
        .map_err(Into::into)
}

fn reverse_epsilon_coefficient(beta: f32, alpha_bar: f32, timestep: usize) -> Result<f32> {
    // Computes:  eps_coef = beta_t / sqrt(1 - alpha_bar_t)
    //
    // ── WHERE THIS COMES FROM ──
    //
    // The epsilon-parameterized posterior mean is:
    //   mu = (1/sqrt(alpha_t)) * (x_t - [beta_t / sqrt(1-alpha_bar_t)] * eps_theta)
    //
    // This coefficient (the bracketed part) converts the model's epsilon
    // prediction (which estimates the TOTAL noise in x_t relative to x_0)
    // into the right scale for a SINGLE reverse step.
    //
    // ── EDGE CASE: alpha_bar_t = 1.0 ──
    //
    // At alpha_bar_t = 1 (the original cosine schedule boundary at t=0),
    // 1 - alpha_bar_t = 0, so sqrt(1 - alpha_bar_t) = 0, producing 0/0.
    // We validate and bail rather than silently produce NaN that would
    // propagate through every subsequent step and into the final PNG.
    let coefficient = beta / (1.0 - alpha_bar).sqrt();
    if !coefficient.is_finite() {
        bail!(
            "non-finite reverse coefficient at timestep {}: beta={}, alpha_bar={}",
            timestep,
            beta,
            alpha_bar
        );
    }
    Ok(coefficient)
}

#[cfg(test)]
mod tests {
    use super::{
        clipped_ddpm_posterior_mean, combine_cfg_predictions, reverse_epsilon_coefficient,
    };
    use candle_core::{Device, Tensor};

    #[test]
    fn reverse_coefficient_rejects_invalid_schedule_boundary() {
        // This is the exact beta_0=0, alpha_bar_0=1 boundary that previously
        // evaluated as 0/sqrt(0).
        assert!(reverse_epsilon_coefficient(0.0, 1.0, 0).is_err());
    }

    #[test]
    fn cfg_scale_zero_is_unconditional_and_one_is_conditional() -> anyhow::Result<()> {
        let conditional = Tensor::new(&[2.0f32, 4.0], &Device::Cpu)?;
        let unconditional = Tensor::new(&[1.0f32, 3.0], &Device::Cpu)?;

        // Test semantic endpoints rather than one arbitrary scale; these catch
        // the common off-by-one CFG convention where s=0 means conditional.
        assert_eq!(
            combine_cfg_predictions(&conditional, &unconditional, 0.0)?.to_vec1::<f32>()?,
            vec![1.0, 3.0]
        );
        assert_eq!(
            combine_cfg_predictions(&conditional, &unconditional, 1.0)?.to_vec1::<f32>()?,
            vec![2.0, 4.0]
        );
        Ok(())
    }

    #[test]
    fn clipped_posterior_bounds_terminal_cosine_update() -> anyhow::Result<()> {
        let xt = Tensor::new(&[2.0f32, -3.0], &Device::Cpu)?;
        let predicted_noise = Tensor::new(&[0.0f32, 0.0], &Device::Cpu)?;
        // Terminal cosine coefficients deliberately stress alpha_bar close to
        // zero. The clipped-x0 path should remain bounded instead of amplifying
        // xt by roughly 1/sqrt(alpha_t).
        let mean = clipped_ddpm_posterior_mean(
            &xt,
            &predicted_noise,
            0.999,
            0.001,
            0.000_000_24,
            0.000_24,
        )?
        .to_vec1::<f32>()?;

        assert!(mean.iter().all(|value| value.abs() < 0.2));
        Ok(())
    }
}

// Build a descending timestep subsequence in [0, start_timestep] with
// `num_inference_steps` evenly spaced entries, always including both
// start_timestep and 0. Both DDIM (non-Markovian update) and respaced DDPM
// (Nichol & Dhariwal's "improved DDPM" respacing) can skip timesteps this
// way: evaluating the model at K << T timesteps costs K forward passes
// instead of T.
fn strided_timesteps(start_timestep: usize, num_inference_steps: usize) -> Vec<usize> {
    let total = start_timestep + 1;
    let num = num_inference_steps.clamp(1, total);
    if num == total {
        return (0..=start_timestep).rev().collect();
    }
    if num == 1 {
        // Single jump: evaluate at the noisiest step, then land on x_0.
        return vec![start_timestep];
    }
    let mut ts = Vec::with_capacity(num);
    for i in 0..num {
        let frac = i as f64 / (num - 1) as f64;
        let t = ((1.0 - frac) * start_timestep as f64).round() as usize;
        ts.push(t);
    }
    // Rounding can collide adjacent entries; keep the subsequence strictly
    // decreasing so every step advances.
    ts.dedup();
    ts
}

// =============================================================================
// sample_ddim_cfg_strided_with_call_back — DDIM reverse sampler with CFG
// =============================================================================
//
// WHAT IS DDIM?
//   DDIM (Denoising Diffusion Implicit Models, Song et al. 2020) reformulates
//   the reverse diffusion as a NON-MARKOVIAN process. Instead of sampling from
//   a Gaussian posterior at each step (like DDPM), DDIM uses a deterministic
//   update rule that "predicts x_0, then interpolates back" to the next step.
//
// HOW DOES THE DDIM UPDATE DIFFER FROM DDPM?
//   DDPM:  x_{t-1} = mu_theta(x_t, t) + sigma_t * z       (stochastic)
//   DDIM:  x_{t-1} = sqrt(alpha_bar_{t-1}) * x0_hat        (deterministic)
//                  + sqrt(1 - alpha_bar_{t-1}) * eps_theta   when eta=0
//
//   The key insight: DDIM first recovers a clean-image estimate (x0_hat),
//   then re-noises it to the level expected at the NEXT (lower) timestep.
//   This makes the trajectory deterministic and allows large stride jumps
//   without the variance accumulation that plagues strided DDPM.
//
// WHY IS DDIM BETTER FOR STRIDED (FEWER-STEP) SAMPLING?
//   DDPM's stochastic noise injection assumes adjacent timesteps. Skipping
//   many steps accumulates excess variance, degrading quality. DDIM's
//   deterministic path is exact regardless of stride — the model only needs
//   to predict epsilon at K chosen timesteps, and the x0-predict-then-re-noise
//   formula handles arbitrary gaps cleanly.
//
// Arguments:
//   model              — any CFG-aware model implementing DenoisingModel
//   scheduler          — pre-computed noise schedule (same as training)
//   xt                 — initial noise tensor x_T ~ N(0,I), shape (N, img_dim)
//   start_timestep     — inclusive starting timestep index (e.g. 99 for 100-step)
//   num_inference_steps — how many reverse steps to take (can be << total steps)
//   _img_dim           — (unused, kept for API symmetry with DDPM variant)
//   time_emb_dim       — sinusoidal time embedding size (must match training)
//   class_one_hot      — target class one-hot vector, shape (N, num_classes)
//   guidance_scale     — CFG strength s (1.0 = conditional only, >1 = amplified)
//   device             — CPU or CUDA device
//   on_step            — callback invoked after each reverse step
pub fn sample_ddim_cfg_strided_with_call_back<F>(
    model: &dyn DenoisingModel,
    scheduler: &BetaScheduler,
    mut xt: Tensor,
    start_timestep: usize,
    num_inference_steps: usize,
    _img_dim: usize,
    time_emb_dim: usize,
    class_one_hot: &Tensor,
    guidance_scale: f64,
    device: &Device,
    mut on_step: F,
) -> Result<Tensor>
where
    F: FnMut(usize, &Tensor) -> Result<()>,
{
    if start_timestep >= scheduler.steps {
        bail!(
            "start_timestep {} must be < scheduler.steps {}",
            start_timestep,
            scheduler.steps
        );
    }

    let num_samples = xt.dim(0)?;
    let class_one_hot_broadcasted;
    let class_one_hot = if class_one_hot.dim(0)? == 1 && num_samples > 1 {
        class_one_hot_broadcasted = class_one_hot.broadcast_as((num_samples, class_one_hot.dim(1)?))?;
        &class_one_hot_broadcasted
    } else {
        class_one_hot
    };

    // Pre-extract cumulative alpha products for O(1) lookup per step.
    // DDIM only needs alpha_bar (not beta/alpha/sigma individually) because
    // it works directly in the alpha_bar domain rather than per-step betas.
    let alphas_cumprod = scheduler.alphas_cumprod.to_vec1::<f32>()?;

    // Build the null (unconditional) one-hot for CFG — all zeros.
    // WHY all-zeros? During training, dropped labels used all-zeros rows.
    // The model learned to associate this with unconditional denoising.
    let null_one_hot = Tensor::zeros(class_one_hot.dims(), class_one_hot.dtype(), device)?;

    // Build the strided timestep subsequence. For num_inference_steps << T,
    // this picks evenly spaced timesteps from start_timestep down to 0.
    let timesteps = strided_timesteps(start_timestep, num_inference_steps);

    // --- Reverse diffusion loop over the (possibly strided) subsequence ----
    for (frame_idx, &t) in timesteps.iter().enumerate() {
        // Step 1: Build timestep tensor — same t repeated for each batch sample.
        let t_vec = vec![t as u32; num_samples];
        let t_tensor = Tensor::new(t_vec.as_slice(), device)?;

        // Step 2: Sinusoidal time embedding (must match training).
        let time_emb = get_time_embedding(&t_tensor, time_emb_dim)?;

        // Step 3: Construct conditional and unconditional model inputs.
        //   v_cond = concat(x_t, time_emb, class_one_hot)  — guided
        //   v_null = concat(x_t, time_emb, null_one_hot)   — unconditional
        let v_cond = Tensor::cat(&[&xt, &time_emb, class_one_hot], 1)?;
        let v_null = Tensor::cat(&[&xt, &time_emb, &null_one_hot], 1)?;

        // Step 4: Batched CFG forward pass.
        // Stack conditional + unconditional inputs along batch dim and run the
        // model once. This halves per-step evaluations (the dominant cost).
        // Split the result back into conditional and unconditional predictions.
        let v_batched = Tensor::cat(&[&v_cond, &v_null], 0)?;
        let (pred_batched, _) = model.forward(&v_batched)?;
        let pred_cond = pred_batched.narrow(0, 0, num_samples)?;
        let pred_uncond = pred_batched.narrow(0, num_samples, num_samples)?;

        // Step 5: Combine via CFG formula:
        //   eps_guided = eps_uncond + s * (eps_cond - eps_uncond)
        let pred_noise = combine_cfg_predictions(&pred_cond, &pred_uncond, guidance_scale)?;

        // Step 6: Retrieve alpha_bar values for the DDIM update.
        //
        // alpha_bar_t     = cumulative signal retention at current timestep t
        // alpha_bar_prev  = cumulative signal retention at the NEXT subsequence
        //                   entry (the "landing" timestep). For the final step
        //                   this is 1.0 (clean signal, no noise remaining).
        //
        // WHY is alpha_bar_prev from the subsequence, not t-1?
        //   In strided DDIM we jump from t to the next entry in the strided
        //   sequence, which may skip many raw timesteps. The DDIM formula
        //   handles this correctly because it only uses alpha_bar values (not
        //   per-step betas), making arbitrary jumps exact.
        let alpha_bar_t_val = alphas_cumprod[t] as f64;
        let alpha_bar_prev_val = match timesteps.get(frame_idx + 1) {
            Some(&t_prev) => alphas_cumprod[t_prev] as f64,
            None => 1.0,
        };

        // Step 7: Predict x_0 from the current noisy x_t and predicted noise.
        //
        //   x0_hat = (x_t - sqrt(1 - alpha_bar_t) * eps_theta) / sqrt(alpha_bar_t)
        //
        // WHY recover x_0 first?
        //   This is the key difference from DDPM. Instead of computing a
        //   posterior mean in x-space, DDIM first estimates what the clean
        //   image looks like, then re-noises it to the target level. This
        //   "predict x_0, then re-noise" approach is what makes DDIM
        //   deterministic and stride-agnostic.
        //
        // WHY clamp to [-1, 1]?
        //   The model was trained on images normalized to [-1, 1]. Without
        //   clamping, a weak noise prediction at high-noise timesteps can
        //   produce x0_hat far outside this range, causing the re-noising
        //   step to amplify the error. Clamping is the MNIST-scale equivalent
        //   of the "dynamic thresholding" used in larger systems.
        let pred_x0 = xt
            .sub(&pred_noise.affine((1.0 - alpha_bar_t_val).sqrt(), 0.0)?)?
            .affine(1.0 / alpha_bar_t_val.sqrt(), 0.0)?
            .clamp(-1.0, 1.0)?;

        // Step 8: Compute the "direction pointing to x_t" component.
        //
        //   direction = sqrt(1 - alpha_bar_{t-1}) * eps_theta
        //
        // WHY this term?
        //   The DDIM update reconstructs x_{t-1} as:
        //     x_{t-1} = sqrt(alpha_bar_{t-1}) * x0_hat + direction
        //
        //   The direction term re-injects exactly the right amount of
        //   "predicted noise structure" so that x_{t-1} is consistent with
        //   the noise level at timestep t-1. This is NOT random noise (like
        //   DDPM's z term) — it uses the model's own prediction, making the
        //   trajectory deterministic.
        let dir = pred_noise.affine((1.0 - alpha_bar_prev_val).sqrt(), 0.0)?;

        // Step 9: Assemble the DDIM update.
        //
        //   x_{t-1} = sqrt(alpha_bar_{t-1}) * x0_hat + sqrt(1 - alpha_bar_{t-1}) * eps_theta
        //
        // This is the full deterministic DDIM formula (eta=0).
        // At the final step, alpha_bar_prev = 1.0, so:
        //   - the x0_hat coefficient becomes 1.0 (we just output x0_hat)
        //   - the direction coefficient becomes 0.0 (no noise re-injection)
        //   → the output is exactly the clamped x0 prediction.
        xt = pred_x0.affine(alpha_bar_prev_val.sqrt(), 0.0)?.add(&dir)?;
        on_step(frame_idx, &xt)?;
    }

    Ok(xt)
}

// Full-resolution DDIM with callback: evaluates every single timestep from
// start_timestep down to 0 (no striding). Delegates to the strided variant
// with num_inference_steps == start_timestep + 1, which covers all timesteps.
//
// WHY a separate function instead of just calling the strided one directly?
//   API ergonomics — callers that want full-resolution don't need to compute
//   the step count themselves, and this wrapper makes the intent clear.
pub fn sample_ddim_cfg_from_timestep_with_call_back<F>(
    model: &dyn DenoisingModel,
    scheduler: &BetaScheduler,
    xt: Tensor,
    start_timestep: usize,
    img_dim: usize,
    time_emb_dim: usize,
    class_one_hot: &Tensor,
    guidance_scale: f64,
    device: &Device,
    on_step: F,
) -> Result<Tensor>
where
    F: FnMut(usize, &Tensor) -> Result<()>,
{
    // Full-resolution DDIM: one reverse step per timestep, start_timestep -> 0.
    sample_ddim_cfg_strided_with_call_back(
        model,
        scheduler,
        xt,
        start_timestep,
        start_timestep + 1,
        img_dim,
        time_emb_dim,
        class_one_hot,
        guidance_scale,
        device,
        on_step,
    )
}

// Convenience wrapper: full-resolution DDIM from a given timestep, no callback.
// Equivalent to the callback variant with a no-op closure.
pub fn sample_ddim_cfg_from_timestep(
    model: &dyn DenoisingModel,
    scheduler: &BetaScheduler,
    xt: Tensor,
    start_timestep: usize,
    img_dim: usize,
    time_emb_dim: usize,
    class_one_hot: &Tensor,
    guidance_scale: f64,
    device: &Device,
) -> Result<Tensor> {
    sample_ddim_cfg_from_timestep_with_call_back(
        model,
        scheduler,
        xt,
        start_timestep,
        img_dim,
        time_emb_dim,
        class_one_hot,
        guidance_scale,
        device,
        |_, _| Ok(()),
    )
}
