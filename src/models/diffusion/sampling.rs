use anyhow::{Ok, Result};
use candle_core::{Device, Tensor};

use super::{get_time_embedding, BetaScheduler, SimpleDenoisingMlp};

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
// The current function is intentionally MLP-specific because
// SimpleDenoisingMlp expects concat(x_t, time_embedding). A future UNet or
// class-conditioned model can get a sibling sampler or a shared denoiser trait.
pub fn sample_ddpm(
    mlp: &SimpleDenoisingMlp,
    scheduler: &BetaScheduler,
    num_samples: usize,
    img_dim: usize,
    time_emb_dim: usize,
    device: &Device,
) -> Result<Tensor> {
    // Start from x_T ~ N(0, I). The reverse chain will denoise this into x_0.
    let xt = Tensor::randn(0.0f32, 1.0f32, (num_samples, img_dim), device)?;

    sample_ddpm_from_noise(mlp, scheduler, xt, img_dim, time_emb_dim, device)
}

pub fn sample_ddpm_from_noise(
    mlp: &SimpleDenoisingMlp,
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
            mlp,
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
    mlp: &SimpleDenoisingMlp,
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
    // We discard the intermediate activations (_, _) because sampling is
    // inference only. There is no backward pass here.
    let (pred_noise, _, _) = mlp.forward(&v)?;

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
    let eps_coef = beta / (1.0 - alpha_bar).sqrt();

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
//   mlp              — trained class-conditioned denoising MLP
//   scheduler        — beta/alpha schedule (same as training)
//   xt               — initial noise tensor x_T, shape (num_samples, img_dim)
//   img_dim          — flattened image size (784 for MNIST)
//   time_emb_dim     — sinusoidal time embedding dimension (must match training)
//   class_one_hot    — one-hot class vector, shape (num_samples, num_classes)
//   device           — CPU or CUDA device
pub fn sample_ddpm_cond(
    mlp: &SimpleDenoisingMlp,
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
        let (pred_noise, _, _) = mlp.forward(&v)?;

        // --- Step 5: Retrieve schedule coefficients for this step -----------
        let beta      = betas[t_step];
        let alpha     = alphas[t_step];
        let alpha_bar = alphas_cumprod[t_step];
        let sigma     = sigmas[t_step];

        // --- Step 6: Epsilon scaling coefficient ----------------------------
        // eps_coef = beta_t / sqrt(1 - alpha_bar_t)
        // Re-scales the noise prediction to the correct amplitude for this step.
        let eps_coef = beta / (1.0 - alpha_bar).sqrt();

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


pub fn sample_ddpm_cfg(
    mlp: &SimpleDenoisingMlp,
    scheduler: &BetaScheduler,
    mut xt: Tensor,
    img_dim: usize,
    time_emb_dim: usize,
    class_one_hot: &Tensor,
    guidance_scale:f64,
    device: &Device,
) -> Result<Tensor> {
    let num_samples = xt.dim(0)?;

    let betas = scheduler.betas.to_vec1::<f32>()?;
    let alphas = scheduler.alphas.to_vec1::<f32>()?;
    let alphas_cumprod = scheduler.alphas_cumprod.to_vec1::<f32>()?;
    let sigmas = scheduler.sigmas.to_vec1::<f32>()?;
    
    let null_one_hot = Tensor::zeros(class_one_hot.dims(),class_one_hot.dtype(),device)?;

    for t_step in (0..scheduler.steps).rev() {
        let t_vec = vec![t_step as u32 ; num_samples];
        let t_tensor = Tensor::new(t_vec.as_slice(),device)?;
        let time_emb = get_time_embedding(&t_tensor, time_emb_dim)?;

        let v_cond = Tensor::cat(&[&xt,&time_emb,class_one_hot], 1)?;
        let v_null = Tensor::cat(&[&xt,&time_emb,&null_one_hot], 1)?;

        let (pred_cond,_,_) = mlp.forward(&v_cond)?;
        let (pred_uncond,_,_) = mlp.forward(&v_null)?;


        let pred_noise = pred_cond.add(
            &pred_cond.sub(&pred_uncond)?.affine(guidance_scale as f64,0.0)?
        )?;


        let beta = betas[t_step];
        let alpha = alphas[t_step];
        let alpha_bar = alphas_cumprod[t_step];
        let sigma = sigmas[t_step];

        let eps_coef = beta / (1.0 - alpha_bar).sqrt();

        let mean = xt
            .sub(&pred_noise.affine(eps_coef as f64, 0.0)?)?
            .affine((1.0 / alpha.sqrt()) as f64, 0.0)?;
        
        if t_step > 0 {
            let z = Tensor::randn(0.0f32, 1.0f32, (num_samples, img_dim), device)?;
            xt = mean.add(&z.affine(sigma as f64, 0.0)?)?;
        } else {
            xt = mean;
        }
    }
    
    Ok(xt)
}