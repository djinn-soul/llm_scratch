use anyhow::{bail, Ok, Result};
use candle_core::{Device, Tensor};

use super::{get_time_embedding, BetaScheduler, DenoisingModel};

// DDPM REVERSE SAMPLING
//
// This module owns inference-time denoising. Keeping it separate from the
// training binary gives us a clean place to add other samplers later:
//
// - DDPM: stochastic reverse process, adds sigma_t * z at each t > 0.
// - DDIM: can use fewer steps and can be deterministic when eta = 0.
// - Class-conditioned sampling: same reverse loop shape, but the model input
//   must also receive class information or guidance.
//
// All sampling functions accept `&dyn DenoisingModel` so they work with any
// noise-prediction architecture (MLP, UNet, Transformer, etc.) without change.
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
    // During the reverse sampling loop we index these arrays by timestep.
    // Converting once avoids repeated Tensor slicing inside every denoising
    // step.
    let betas = scheduler.betas.to_vec1::<f32>()?;
    let alphas = scheduler.alphas.to_vec1::<f32>()?;
    let alphas_cumprod = scheduler.alphas_cumprod.to_vec1::<f32>()?;
    let sigmas = scheduler.sigmas.to_vec1::<f32>()?;

    // Reverse diffusion loop: t = T-1 -> 0.
    //
    // WHY iterate in reverse?
    //
    // Training corrupts x_0 -> x_T by adding noise.
    // Generation reconstructs x_T -> x_0 by removing predicted noise.
    // Each step removes a small amount of noise guided by the model.
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
    // beta      = beta_t: noise variance scheduled for this step
    // alpha     = alpha_t = 1 - beta_t: signal retention for this step
    // alpha_bar = cumulative signal retention up to this step
    // sigma     = standard deviation of the stochastic reverse term
    let beta = betas[t_step];
    let alpha = alphas[t_step];
    let alpha_bar = alphas_cumprod[t_step];
    let sigma = sigmas[t_step];

    // Step R6: Compute the epsilon coefficient.
    //
    // eps_coef = beta_t / sqrt(1 - alpha_bar_t)
    //
    // WHY this formula?
    //
    // It comes from rearranging the DDPM reverse posterior mean. Dividing by
    // sqrt(1 - alpha_bar_t) re-scales the model's full-noise prediction back
    // to the correct amplitude for this specific timestep.
    let eps_coef = reverse_epsilon_coefficient(beta, alpha_bar, t_step)?;

    // Step R7: Compute the DDPM reverse posterior mean.
    //
    // mean = (1 / sqrt(alpha_t)) *
    //        (x_t - beta_t / sqrt(1 - alpha_bar_t) * eps_theta(x_t, t))
    //
    // WHY subtract eps_coef * predicted_noise?
    //
    // The model predicts the noise currently inside x_t. Subtracting the
    // scaled prediction moves x_t one step toward being less corrupted.
    //
    // WHY multiply by 1 / sqrt(alpha_t)?
    //
    // The forward process also scaled the signal by sqrt(alpha_t). This factor
    // reverses that per-step signal scaling.
    let mean = xt
        .sub(&pred_noise.affine(eps_coef as f64, 0.0)?)?
        .affine((1.0 / alpha.sqrt()) as f64, 0.0)?;

    // Step R8: Add stochastic noise for all steps except the final step.
    //
    // WHY add noise when t > 0?
    //
    // The true reverse posterior is Gaussian, not a single point estimate.
    // Adding sigma_t * z restores the correct variance and keeps samples from
    // collapsing into an over-smoothed average.
    //
    // WHY no noise at t = 0?
    //
    // At the final step we output x_0. Adding fresh noise there would only
    // damage the final generated image.
    if t_step > 0 {
        // z ~ N(0, I): fresh independent noise for this reverse step.
        let z = Tensor::randn(0.0f32, 1.0f32, (num_samples, img_dim), device)?;

        // x_{t-1} = mean + sigma_t * z
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
// WHAT IS CFG?
//   CFG is an inference-time technique that amplifies the model's class signal
//   without requiring a separate classifier.  Because the model was trained
//   with stochastic label dropout, it has implicitly learned two distributions:
//
//     epsilon_theta(x_t, t, c)  — conditional noise prediction (given class c)
//     epsilon_theta(x_t, t, ∅)  — unconditional noise prediction (null label)
//
//   At each reverse step we compute BOTH predictions and blend them:
//
//     epsilon_guided = epsilon_cond + s * (epsilon_cond - epsilon_uncond)
//
//   where s ≥ 1 is the guidance_scale.
//
// WHY does this make the output more class-faithful?
//   (epsilon_cond - epsilon_uncond) is the "direction" in noise space that
//   points from unconditional toward the target class.  Scaling it by s and
//   adding it to epsilon_cond amplifies that direction, steering the denoising
//   trajectory more aggressively toward the requested class.
//
// WHY does a higher s reduce diversity?
//   Larger guidance pushes every sample further in the same class direction,
//   reducing the spread of the latent space around that class.  This trades
//   sample diversity for stronger class fidelity.
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

// Core CFG reverse loop.
//
// `start_timestep` is inclusive: t=6 performs seven updates ending at t=0.
// The callback receives a forward-running frame index (0, 1, ...) rather than
// the decreasing diffusion timestep, which makes filenames naturally sortable.
pub fn sample_ddpm_cfg_from_timestep_with_callback<F>(
    model: &dyn DenoisingModel,
    scheduler: &BetaScheduler,
    mut xt: Tensor,
    start_timestep: usize,
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

    // Pre-extract schedule coefficients into plain Vecs for O(1) indexed access.
    // Avoids repeated Tensor slicing inside the per-step loop.
    let betas = scheduler.betas.to_vec1::<f32>()?;
    let alphas = scheduler.alphas.to_vec1::<f32>()?;
    let alphas_cumprod = scheduler.alphas_cumprod.to_vec1::<f32>()?;
    let alphas_cumprod_prev = scheduler.alphas_cumprod_prev.to_vec1::<f32>()?;
    let sigmas = scheduler.sigmas.to_vec1::<f32>()?;

    // Build the "null" (unconditional) conditioning vector once.
    // Shape matches class_one_hot but every element is 0.0.
    //
    // WHY all-zeros for the null class?
    //   During training, dropped labels were represented as all-zeros rows.
    //   The model learned to associate all-zeros with "unconditional" denoising.
    //   Using the same convention at inference ensures it activates the correct
    //   unconditional behaviour.
    let null_one_hot = Tensor::zeros(class_one_hot.dims(), class_one_hot.dtype(), device)?;

    // --- Reverse diffusion loop: t = T-1 → 0 --------------------------------
    // WHY iterate in reverse? The forward process adds noise (x_0 → x_T).
    //                         The reverse process removes noise (x_T → x_0).
    for t_step in (0..=start_timestep).rev() {
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

        // Step 4: Two forward passes — one conditional, one unconditional.
        //
        // WHY two passes and not one?
        //   We need epsilon_cond and epsilon_uncond separately to compute the
        //   guidance direction.  There is no shortcut; we must evaluate the
        //   model twice per reverse step.
        //
        // Note: intermediate activations are discarded — we're at inference.
        let (pred_cond, _) = model.forward(&v_cond)?;
        let (pred_uncond, _) = model.forward(&v_null)?;

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

        // Step 6: Retrieve schedule coefficients for this timestep.
        //   beta      = beta_t: noise variance for this step
        //   alpha     = alpha_t = 1 - beta_t
        //   alpha_bar = cumulative product of alpha values up to t
        //   sigma     = sqrt(beta_tilde) — exact posterior standard deviation
        let beta = betas[t_step];
        let alpha = alphas[t_step];
        let alpha_bar = alphas_cumprod[t_step];
        let alpha_bar_prev = alphas_cumprod_prev[t_step];
        let sigma = sigmas[t_step];

        // Step 7: Recover x_0, constrain it to the normalized image domain,
        // then use the exact DDPM posterior mean. The algebraically equivalent
        // epsilon-only form can magnify small high-noise prediction errors far
        // outside [-1, 1], especially at the terminal cosine step.
        let mean =
            clipped_ddpm_posterior_mean(&xt, &pred_noise, beta, alpha, alpha_bar, alpha_bar_prev)?;

        // Step 9: Add stochasticity for all non-final steps.
        //
        // WHY add noise for t > 0?
        //   The true reverse posterior is Gaussian with variance sigma_t^2.
        //   Adding sigma_t * z samples from this posterior correctly.
        //
        // WHY no noise at t = 0?
        //   The final step outputs x_0 deterministically; noise would degrade it.
        if t_step > 0 {
            // z ~ N(0, I): independent noise for this reverse step.
            let z = Tensor::randn(0.0f32, 1.0f32, (num_samples, img_dim), device)?;
            // x_{t-1} = mean + sigma_t * z
            xt = mean.add(&z.affine(sigma as f64, 0.0)?)?;
        } else {
            // Final step: output the deterministic mean as x_0.
            xt = mean;
        }

        let frame_idx = start_timestep - t_step;
        on_step(frame_idx, &xt)?;
    }
    Ok(xt)
}

fn combine_cfg_predictions(
    conditional: &Tensor,
    unconditional: &Tensor,
    guidance_scale: f64,
) -> Result<Tensor> {
    // epsilon = epsilon_uncond + s * (epsilon_cond - epsilon_uncond)
    //
    // The endpoints are valuable invariants: s=0 must reproduce unconditional
    // prediction exactly, while s=1 must reproduce ordinary conditioning.
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
    // Recover the clean-image estimate implied by the epsilon prediction:
    //
    //   x0_hat = (xt - sqrt(1 - alpha_bar_t) * epsilon) / sqrt(alpha_bar_t)
    //
    // A weak predictor can make x0_hat enormous near the noisy end of a cosine
    // schedule, where alpha_bar is tiny. Dynamic thresholding is common in
    // larger systems; for normalized MNIST, clipping to the known [-1, 1]
    // training domain is the direct equivalent.
    let predicted_x0 = xt
        .sub(&predicted_noise.affine((1.0 - alpha_bar).sqrt() as f64, 0.0)?)?
        .affine((1.0 / alpha_bar.sqrt()) as f64, 0.0)?
        .clamp(-1.0f32, 1.0f32)?;
    // Exact q(x_{t-1} | xt, x0_hat) posterior-mean coefficients:
    //
    //   c_x0 = beta_t * sqrt(alpha_bar_{t-1}) / (1 - alpha_bar_t)
    //   c_xt = sqrt(alpha_t) * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)
    let denominator = 1.0 - alpha_bar;
    let x0_coefficient = beta * alpha_bar_prev.sqrt() / denominator;
    let xt_coefficient = (1.0 - alpha_bar_prev) * alpha.sqrt() / denominator;

    predicted_x0
        .affine(x0_coefficient as f64, 0.0)?
        .add(&xt.affine(xt_coefficient as f64, 0.0)?)
        .map_err(Into::into)
}

fn reverse_epsilon_coefficient(beta: f32, alpha_bar: f32, timestep: usize) -> Result<f32> {
    // Validate at the point of construction so callers cannot quietly propagate
    // NaN into every later reverse step and eventually encode it as a PNG.
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
