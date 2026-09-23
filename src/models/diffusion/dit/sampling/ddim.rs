//! # Denoising Diffusion Implicit Models (DDIM) & Classifier-Free Guidance (CFG)
//!
//! ## 1. Deterministic Reverse Sampling: The Probability Flow ODE
//! Standard DDPM (Ho et al., 2020) defines reverse sampling via a stochastic Markov chain:
//!
//! $$p_\theta(x_{t-1} \mid x_t) = \mathcal{N}(x_{t-1};\, \mu_\theta(x_t, t),\, \sigma_t^2 \mathbf{I})$$
//!
//! DDIM (Song et al., 2020) generalizes this to non-Markovian inference distributions that share the
//! exact same marginals $q(x_t \mid x_0)$ as DDPM. By setting the stochastic variance $\sigma_t = 0$,
//! the reverse process becomes fully deterministic, tracing the unique trajectory of the **Probability Flow ODE**:
//!
//! 1. **Predict clean sample $\hat{x}_0$ from current noisy state $x_t$**:
//!    $$\hat{x}_0 = \frac{x_t - \sqrt{1 - \bar{\alpha}_t} \cdot \tilde{\epsilon}_\theta}{\sqrt{\bar{\alpha}_t}}$$
//!
//! 2. **Compute directional vector pointing to $x_{t_{\text{prev}}}$ along the ODE trajectory**:
//!    $$\text{dir}_{x_t} = \sqrt{1 - \bar{\alpha}_{t_{\text{prev}}}} \cdot \tilde{\epsilon}_\theta$$
//!
//! 3. **Propagate to preceding timestep without adding stochastic noise**:
//!    $$x_{t_{\text{prev}}} = \sqrt{\bar{\alpha}_{t_{\text{prev}}}} \cdot \hat{x}_0 + \text{dir}_{x_t}$$
//!
//! ## 2. Classifier-Free Guidance (CFG): "The Prompt Volume Knob"
//! Using Bayes' rule, the conditional score function decomposes into unconditioned score and guidance gradient:
//!
//! $$\nabla_x \log p(x \mid c) = \nabla_x \log p(x) + \nabla_x \log p(c \mid x)$$
//!
//! Ho & Salimans (2021) replaced the external classifier with implicit joint conditioning.
//! We evaluate the model twice at each step:
//! - Conditioned on prompt: $\epsilon_\theta(x_t, t, c)$
//! - Unconditioned null token: $\epsilon_\theta(x_t, t, \emptyset)$
//!
//! The guided noise vector extrapolates along the class-alignment direction:
//!
//! $$\tilde{\epsilon}_\theta = \epsilon_\theta(x_t, t, \emptyset) + s \cdot \Big(\epsilon_\theta(x_t, t, c) - \epsilon_\theta(x_t, t, \emptyset)\Big)$$
//!
//! - **$s = 0.0$**: Ignores class conditioning; generates unconditional prior images.
//! - **$s = 1.0$**: Standard class-conditioned generation.
//! - **$s = 1.5 - 2.5$**: Sweet spot; sharpens silhouettes, boosts contrast, and suppresses background artifacts.
//! - **$s \ge 4.0$**: Over-saturation; causes unnatural high-frequency clipping.
//!
//! ## 3. Latent Class Morphing: "The Garment Blender"
//! Because DiT processes class conditions as dense continuous vectors $\mathbf{e} \in \mathbb{R}^D$, we can
//! linearly interpolate (LERP) across the semantic embedding manifold:
//!
//! $$\mathbf{e}_{\text{blend}}(\lambda) = (1 - \lambda) \cdot \mathbf{e}_A + \lambda \cdot \mathbf{e}_B, \quad \lambda \in [0, 1]$$
//!
//! Stepping $\lambda$ from $0.0 \to 1.0$ produces a continuous morphing sequence across garment classes.

use burn::tensor::backend::Backend;
use burn::tensor::{Distribution, Tensor};

use super::scheduler::{get_time_step_embeddings, SimpleNoiseScheduler};
use crate::models::diffusion::dit::dit::DiffusionTransformer;

/// Samples images using continuous condition vectors with Classifier-Free Guidance (CFG).
///
/// Enables both standard discrete sampling and smooth continuous latent space morphing.
pub fn sample_ddim_continuous<B: Backend>(
    model: &DiffusionTransformer<B>,
    scheduler: &SimpleNoiseScheduler,
    mut x: Tensor<B, 4>,
    class_cond: Tensor<B, 2>,
    null_cond: Tensor<B, 2>,
    steps: usize,
    cfg_scale: f32,
    hidden_dim: usize,
    device: &B::Device,
) -> Tensor<B, 4> {
    // ------------------------------------------------------------------------
    // Step A: Discretize the diffusion trajectory into `steps` reverse intervals
    // ------------------------------------------------------------------------
    // For 1000 total steps and 50 sampling steps, stride = 20.
    // Generates a descending timeline: [980, 960, 940, ..., 20, 0].
    let total_steps = scheduler.alphas_cumprod.len();
    let step_stride = total_steps / steps;
    let time_indices: Vec<usize> = (0..steps)
        .rev()
        .map(|i| (i * step_stride).min(total_steps - 1))
        .collect();

    // ------------------------------------------------------------------------
    // Step B: Iterate backwards along the deterministic Probability Flow ODE
    // ------------------------------------------------------------------------
    for i in 0..time_indices.len() {
        let t_curr = time_indices[i];
        let t_prev = if i + 1 < time_indices.len() {
            time_indices[i + 1]
        } else {
            0 // Final destination is t = 0 (clean image x_0)
        };

        // B.1: Compute sinusoidal Fourier temporal embedding for current timestep t_curr
        let t_emb = get_time_step_embeddings::<B>(&[t_curr], hidden_dim, device);

        // B.2: Dual forward pass through DiT with the exact same noisy state x_t:
        // - Conditioned forward pass: evaluates class-conditioned noise prediction eps_cond
        let eps_cond = model.forward_with_cond_vec(x.clone(), t_emb.clone(), class_cond.clone());
        // - Unconditioned forward pass: evaluates unconditional dataset prior eps_uncond
        let eps_uncond = model.forward_with_cond_vec(x.clone(), t_emb, null_cond.clone());

        // B.3: Classifier-Free Guidance (CFG) extrapolation:
        // eps = eps_uncond + s * (eps_cond - eps_uncond)
        // Stepping s > 1.0 amplifies distinctive class silhouettes and suppresses off-class noise.
        let eps = eps_uncond.clone() + (eps_cond - eps_uncond) * cfg_scale;

        // B.4: Look up cumulative variance terms alpha_bar for t_curr and t_prev
        let alpha_bar_curr = scheduler.alphas_cumprod[t_curr];
        let alpha_bar_prev = scheduler.alphas_cumprod[t_prev];

        // B.5: Tweedie's Formula: Reconstruct predicted clean image x_0 from noisy state x_t
        // x_0 = (x_t - sqrt(1 - alpha_bar_t) * eps) / sqrt(alpha_bar_t)
        let pred_x0 =
            (x.clone() - eps.clone() * (1.0 - alpha_bar_curr).sqrt()) / alpha_bar_curr.sqrt();

        // B.6: Directional vector pointing to t_prev along the deterministic ODE flow
        let dir_xt = eps * (1.0 - alpha_bar_prev).sqrt();

        // B.7: Advance to preceding timestep (sigma = 0 for deterministic DDIM)
        // x_{t_prev} = sqrt(alpha_bar_prev) * pred_x0 + dir_xt
        x = pred_x0 * alpha_bar_prev.sqrt() + dir_xt;
    }

    x
}

/// Samples images for a discrete class category using DDIM and Classifier-Free Guidance.
pub fn sample_ddim_with_cfg<B: Backend>(
    model: &DiffusionTransformer<B>,
    scheduler: &SimpleNoiseScheduler,
    batch_size: usize,
    class_id: usize,
    steps: usize,
    cfg_scale: f32,
    hidden_dim: usize,
    device: &B::Device,
) -> Tensor<B, 4> {
    let x: Tensor<B, 4> = Tensor::random(
        [batch_size, 1, 28, 28],
        Distribution::Normal(0.0, 1.0),
        device,
    );
    let class_cond = model.get_class_embedding(class_id, device);
    // Class index 10 represents the unconditional null token learned during training
    let null_cond = model.get_class_embedding(10, device);

    sample_ddim_continuous(
        model, scheduler, x, class_cond, null_cond, steps, cfg_scale, hidden_dim, device,
    )
}
