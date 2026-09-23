//! # Diffusion Noise Schedulers & Timestep Embeddings
//!
//! ## 1. Mathematical Formulation of the Forward Diffusion Process
//! In Denoising Diffusion Probabilistic Models (DDPM), the forward process gradually corrupts
//! clean data $x_0 \sim q(x)$ into pure Gaussian noise over $T$ discrete steps:
//!
//! $$q(x_t \mid x_{t-1}) = \mathcal{N}(x_t;\, \sqrt{1 - \beta_t}\, x_{t-1},\, \beta_t \mathbf{I})$$
//!
//! By defining $\alpha_t = 1 - \beta_t$ and the cumulative product $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$,
//! we obtain the closed-form transition directly from $x_0$ to any arbitrary step $t$:
//!
//! $$q(x_t \mid x_0) = \mathcal{N}(x_t;\, \sqrt{\bar{\alpha}_t}\, x_0,\, (1 - \bar{\alpha}_t) \mathbf{I})$$
//! $$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})$$
//!
//! ## 2. Linear vs. Cosine Variance Schedules
//! - **Linear Schedule (Ho et al., 2020)**: Linearly interpolates $\beta_t \in [\beta_1, \beta_T]$.
//!   Drawback: $\bar{\alpha}_t$ drops precipitously in early-to-mid steps, destroying image structure too fast.
//! - **Cosine Schedule (Nichol & Dhariwal, 2021)**: Designed so $\bar{\alpha}_t$ decays smoothly like a cosine:
//!   $$f(t) = \cos^2\left(\frac{t/T + s}{1 + s} \cdot \frac{\pi}{2}\right)$$
//!   $$\bar{\alpha}_t = \frac{f(t)}{f(0)}$$
//!   The small offset $s = 0.008$ prevents singularities near $t = 0$ and ensures gentle decay.

use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

/// Forward noise schedule storing precomputed variance terms ($\bar{\alpha}_t, \sqrt{\bar{\alpha}_t}, \sqrt{1 - \bar{\alpha}_t}$).
///
/// Precomputing these values avoids redundant transcendental floating-point operations during training and DDIM sampling.
pub struct SimpleNoiseScheduler {
    /// Cumulative product of alphas: $\bar{\alpha}_t = \prod_{s=1}^t (1 - \beta_s)$
    pub alphas_cumprod: Vec<f32>,
    /// Signal scaling coefficient: $\sqrt{\bar{\alpha}_t}$
    pub sqrt_alphas_cumprod: Vec<f32>,
    /// Noise scaling coefficient: $\sqrt{1 - \bar{\alpha}_t}$
    pub sqrt_one_minus_alphas_cumprod: Vec<f32>,
}

impl SimpleNoiseScheduler {
    /// Constructs a cosine noise schedule (Nichol & Dhariwal, 2021).
    ///
    /// Preserves structural details for a larger fraction of diffusion steps compared to linear schedules.
    pub fn new_cosine(timesteps: usize) -> Self {
        // Small offset s = 0.008 prevents beta_t from being zero near t=0 and avoids singularities
        let s = 0.008f32;
        let mut alphas_cumprod = Vec::with_capacity(timesteps);

        // Normalization factor f(0): ensures alpha_bar(0) evaluates to 1.0 (unaltered clean data)
        let f0 = ((s / (1.0 + s)) * std::f32::consts::FRAC_PI_2)
            .cos()
            .powi(2);

        for t in 0..timesteps {
            // Normalized progression through the diffusion horizon: (t + 1) / T
            let progress = (t as f32 + 1.0) / (timesteps as f32);

            // Cosine decay function: f(t) = cos^2( ((progress + s) / (1 + s)) * pi / 2 )
            let ft = (((progress + s) / (1.0 + s)) * std::f32::consts::FRAC_PI_2)
                .cos()
                .powi(2);

            // Cumulative product alpha_bar_t = f(t) / f(0).
            // Clamped to [0.0001, 0.9999] to prevent division by zero or NaN square roots.
            let alpha_bar = (ft / f0).clamp(0.0001, 0.9999);
            alphas_cumprod.push(alpha_bar);
        }

        // Precompute square root terms used in closed-form q-sampling and reverse DDIM equations
        let sqrt_alphas_cumprod = alphas_cumprod.iter().map(|&a| a.sqrt()).collect();
        let sqrt_one_minus_alphas_cumprod =
            alphas_cumprod.iter().map(|&a| (1.0 - a).sqrt()).collect();

        Self {
            alphas_cumprod,
            sqrt_alphas_cumprod,
            sqrt_one_minus_alphas_cumprod,
        }
    }

    /// Constructs a linear noise schedule between $\beta_{\text{start}}$ and $\beta_{\text{end}}$.
    pub fn new_linear(timesteps: usize, beta_start: f32, beta_end: f32) -> Self {
        let mut alphas_cumprod = Vec::with_capacity(timesteps);
        let mut cumprod = 1.0f32;

        for t in 0..timesteps {
            // Linearly interpolate noise variance: beta_t = beta_start + (beta_end - beta_start) * (t / (T - 1))
            let beta = beta_start + (beta_end - beta_start) * (t as f32) / ((timesteps - 1) as f32);
            // Accumulate cumulative product: alpha_bar_t = product_{s=1}^t (1 - beta_s)
            cumprod *= 1.0 - beta;
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
}

/// Computes sinusoidal Fourier timestep embeddings (Vaswani et al., 2017).
///
/// Maps discrete or continuous diffusion timesteps $t \in [0, T)$ into continuous frequency representations:
///
/// $$\omega_k = \exp\left(-\frac{2k}{D} \ln(10000)\right) = \frac{1}{10000^{2k/D}}$$
/// $$\text{Embedding}(t) = [\sin(t \omega_0), \cos(t \omega_0), \sin(t \omega_1), \cos(t \omega_1), \dots]$$
///
/// This provides high-frequency geometric coordinates that enable transformer attention blocks
/// to discern subtle temporal differences between adjacent diffusion steps.
pub fn get_time_step_embeddings<B: Backend>(
    timesteps: &[usize],
    dim: usize,
    device: &B::Device,
) -> Tensor<B, 2> {
    let half_dim = dim / 2;
    let mut data = Vec::with_capacity(timesteps.len() * dim);
    for &t in timesteps {
        for i in 0..half_dim {
            // Geometric progression of frequency scales: omega_i = exp(-i * ln(10000) / half_dim)
            // Low i captures fine, high-frequency changes; high i captures broad, low-frequency shifts
            let freq = (-(i as f32) * (10000.0f32.ln()) / (half_dim as f32)).exp();
            let arg = t as f32 * freq;

            // Interleaved sine and cosine coordinates form orthogonal basis vectors
            data.push(arg.sin());
            data.push(arg.cos());
        }
    }
    Tensor::<B, 1>::from_floats(data.as_slice(), device).reshape([timesteps.len(), dim])
}
