//! # Forward Diffusion Noising & Optimization Steps
//!
//! ## 1. Forward Diffusion Process ($q$-sampling)
//! The forward Markovian chain adds Gaussian noise according to the variance schedule $\beta_t$:
//!
//! $$q(x_t \mid x_{t-1}) = \mathcal{N}(x_t;\, \sqrt{1 - \beta_t}\, x_{t-1},\, \beta_t \mathbf{I})$$
//!
//! Because linear combinations of independent Gaussians remain Gaussian, we reparameterize
//! the forward trajectory to jump directly from $x_0$ to any timestep $t$:
//!
//! $$x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})$$
//!
//! ## 2. Simplified Variational Objective ($L_{\text{simple}}$)
//! Ho et al. (2020) proved that optimizing a reweighted surrogate of the variational lower bound (ELBO)
//! dramatically enhances sample perceptual fidelity:
//!
//! $$L_{\text{simple}}(\theta) = \mathbb{E}_{t \sim [0, T),\, x_0,\, \epsilon}\! \left[ \big\| \epsilon - \epsilon_\theta(x_t,\, t,\, c) \big\|_2^2 \right]$$
//!
//! The network acts as a score estimator: $\nabla_{x_t} \log p(x_t) \propto -\frac{\epsilon_\theta(x_t, t)}{\sqrt{1 - \bar{\alpha}_t}}$.
//!
//! ## 3. Classifier-Free Conditioning Dropout ($p_{\text{uncond}} = 0.10$)
//! During training, class condition labels $c \in [0, 9]$ are randomly masked to the null token ($c_{\text{null}} = 10$)
//! with 10% probability. This enables the model to simultaneously learn conditional generation $p(x \mid c)$
//! and the unconditional prior $p(x)$ in a single unified neural network.

use burn::optim::{GradientsParams, Optimizer};
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::Tensor;

use super::super::dit::DiffusionTransformer;
use super::super::sampling::SimpleNoiseScheduler;

/// Directly perturbs clean data $x_0$ to noisy state $x_t$ using closed-form Gaussian noising.
///
/// $$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$$
pub fn q_sample<B: Backend>(
    x_0: Tensor<B, 4>,
    noise: Tensor<B, 4>,
    timesteps: &[usize],
    scheduler: &SimpleNoiseScheduler,
    device: &B::Device,
) -> Tensor<B, 4> {
    let b = timesteps.len();
    let mut sqrt_alpha_data = Vec::with_capacity(b);
    let mut sqrt_one_minus_alpha_data = Vec::with_capacity(b);

    // 1. Gather precomputed variance scaling coefficients for each sampled batch timestep
    for &t in timesteps {
        sqrt_alpha_data.push(scheduler.sqrt_alphas_cumprod[t]);
        sqrt_one_minus_alpha_data.push(scheduler.sqrt_one_minus_alphas_cumprod[t]);
    }

    // 2. Reshape to [B, 1, 1, 1] so coefficients broadcast over image channels, height, and width
    let sqrt_alpha =
        Tensor::<B, 1>::from_floats(sqrt_alpha_data.as_slice(), device).reshape([b, 1, 1, 1]);
    let sqrt_one_minus_alpha =
        Tensor::<B, 1>::from_floats(sqrt_one_minus_alpha_data.as_slice(), device)
            .reshape([b, 1, 1, 1]);

    // 3. Compute the closed-form forward diffusion perturbation:
    // x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
    x_0 * sqrt_alpha + noise * sqrt_one_minus_alpha
}

/// Computes forward noise prediction, MSE loss, and executes an AdamW gradient step.
pub fn train_steps<B: AutodiffBackend>(
    model: DiffusionTransformer<B>,
    x_t: Tensor<B, 4>,
    target_noise: Tensor<B, 4>,
    t_emb: Tensor<B, 2>,
    class_labels: Tensor<B, 1, burn::tensor::Int>,
    optimizer: &mut impl Optimizer<DiffusionTransformer<B>, B>,
    lr: f64,
) -> (DiffusionTransformer<B>, f32) {
    // 1. Forward pass: evaluate model to predict injected noise vector epsilon_theta(x_t, t, c)
    // Ho et al. (2020) demonstrated that predicting epsilon rather than clean x_0 yields superior FID.
    let pred_noise = model.forward(x_t, t_emb, class_labels);

    // 2. Compute Mean Squared Error (MSE) loss: || epsilon - epsilon_theta ||^2
    let diff = pred_noise - target_noise;
    let loss = diff.powf_scalar(2.0).mean();

    // Extract raw scalar f32 loss value for training logs and dashboard telemetry
    let loss_val = loss.clone().into_data().as_slice::<f32>().unwrap()[0];

    // 3. Autodiff reverse-mode gradient computation across all trainable model parameters
    let grads = GradientsParams::from_grads(loss.backward(), &model);

    // 4. Execute AdamW parameter update with weight decay and current scheduled learning rate
    let model = optimizer.step(lr, model, grads);
    (model, loss_val)
}
