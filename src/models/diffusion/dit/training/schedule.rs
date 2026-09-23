//! # Learning Rate Schedules: Warmup & Cosine Annealing
//!
//! ## Mathematical Formulation
//! Optimization of Vision Transformers is notoriously sensitive to early gradient shocks.
//! A two-phase learning rate schedule ensures training stability:
//!
//! ### Phase 1: Linear Warmup ($t < T_{\text{warmup}}$)
//! Prevents initial large gradient updates from destabilizing randomly initialized attention projections:
//!
//! $$\eta(t) = \eta_{\min} + (\eta_{\max} - \eta_{\min}) \cdot \frac{t}{T_{\text{warmup}}}$$
//!
//! ### Phase 2: Cosine Annealing ($T_{\text{warmup}} \le t \le T_{\text{total}}$)
//! Gradually lowers the learning rate along a half-period cosine curve (Loshchilov & Hutter, 2016),
//! smoothly decelerating optimization into a flat local minimum:
//!
//! $$p(t) = \frac{t - T_{\text{warmup}}}{T_{\text{total}} - T_{\text{warmup}}}$$
//! $$\eta(t) = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min}) \cdot \Big(1 + \cos(\pi \cdot p(t))\Big)$$

/// Calculates the learning rate using Linear Warmup followed by Cosine Annealing decay.
///
/// # Arguments
/// * `step` - Current optimization step index $t$
/// * `total_steps` - Total training budget $T_{\text{total}}$
/// * `warmup_steps` - Number of linear warmup steps $T_{\text{warmup}}$
/// * `lr_max` - Peak learning rate $\eta_{\max}$
/// * `lr_min` - Minimum learning rate floor $\eta_{\min}$
pub fn get_learning_rate(
    step: usize,
    total_steps: usize,
    warmup_steps: usize,
    lr_max: f64,
    lr_min: f64,
) -> f64 {
    if step < warmup_steps {
        lr_min + (lr_max - lr_min) * (step as f64 / warmup_steps.max(1) as f64)
    } else if step > total_steps {
        lr_min
    } else {
        let progress = (step - warmup_steps) as f64 / (total_steps - warmup_steps).max(1) as f64;
        let progress = progress.clamp(0.0, 1.0);
        lr_min + 0.5 * (lr_max - lr_min) * (1.0 + (std::f64::consts::PI * progress).cos())
    }
}
