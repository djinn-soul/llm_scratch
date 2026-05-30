// ════════════════════════════════════════════════════════════════════════════
// ADAM OPTIMIZER (REWRITTEN 2026)
// ════════════════════════════════════════════════════════════════════════════
//
// Original paper: Adam: A Method for Stochastic Optimization (Kingma & Ba, 2015)
//
// This is the CLEANED, STABILIZED, CORRECTED version.
// The code below fixes the unstable `beta1.powi(t)` bug present in the
// initial naive implementation.
//
// ─────────────────────────────────────────────────────────────────────────
// WHY Adam Beats Plain SGD
// ─────────────────────────────────────────────────────────────────────────
//
// Plain SGD has one fatal flaw:
//
//   It gives *every* weight the exact same learning rate.
//
// But in a neural network, some weights have:
//   • Large gradients  (common features) → needs SMALLER step
//   • Tiny gradients (rare features)    → needs LARGER step
//
// Adam solves this by tracking 2 internal states (buffers) per weight:
//
//   m  ← 1st moment (mean)           tracks the *direction* of gradients
//   v  ← 2nd moment (variance)       tracks the *magnitude* of gradients
//
// Then it performs bias correction and adaptive scaling:
//
//   m̂ = m / (1 - β₁^t)
//   v̂ = v / (1 - β₂^t)
//
//   Update = lr * m̂ / (√v̂ + ε)
//
// ─────────────────────────────────────────────────────────────────────────
// THE BUG FIX: How NOT to calculate powers of β
// ─────────────────────────────────────────────────────────────────────────
//
// Bad (incorrect, leads to explosive numbers):
//   m = β*m + (1-β)*g
//   v = β*v + (1-β)*g*g
//   m_hat = m / beta1.powi(t)        ← ✗ OVERFLOW!  (beta1 is ~0.9)
//   v_hat = v / beta2.powi(t)        ← ✗ OVERFLOW!  (beta2 is ~0.99)
//
// Why this happens:
//   When t=300, beta1^t = 0.9^300 ≈ 1e-14
//   Dividing by this tiny number causes massive explosions.
//
// Correct bias correction:
//   m = beta1*m + (1-beta1)*g
//   v = beta2*v + (1-beta2)*g*g
//   m_hat = m / (1 - beta1.powi(t))
//   v_hat = v / (1 - beta2.powi(t))
//
// The subtraction `(1 - beta^t)` gives a positive denominator that starts
// small and approaches 1. This removes the early-step bias from starting
// m and v at zero.
//
// ─────────────────────────────────────────────────────────────────────────
// TYPICAL HYPERPARAMETERS (stable defaults)
// ─────────────────────────────────────────────────────────────────────────
//
//   lr   = 3e-4   (much smaller than SGD's 1e-3 or 5e-3)
//   β₁   = 0.9    (controls momentum)
//   β₂   = 0.999  (controls adaptation)
//   ε    = 1e-8   (stabilization)
//
// ─────────────────────────────────────────────────────────────────────────
// CODE STRUCTURE
// ─────────────────────────────────────────────────────────────────────────
//
// 1. struct Adam with:
//    - lr, beta1, beta2, epsilon
//    - clipping: ClippingStrategy
//    - step_count (t)
//    - m: Vec<Vec<Vec<f32>>>  (momentum buffers)
//    - v: Vec<Vec<Vec<f32>>>  (variance buffers)
//
// 2. new() constructor
//    - receives learning rate and clipping policy up front
//    - leaves m/v buffers empty until the first update sees model shape
//
// 3. Optimizer::step() wrapper
//    - applies the configured clipping policy to gradients
//    - calls Adam::update() for the Adam-specific math
//
// 4. update() method with:
//    - t = step_count + 1
//    - Bias correction: (1 - beta^t)
//    - m update: β₁·m + (1-β₁)·g
//    - v update: β₂·v + (1-β₂)·g²
//    - Adaptive step: lr * m̂ / (√v̂ + ε)
//
// ─────────────────────────────────────────────────────────────────────────
// FINAL NOTE ON STABILITY
// ─────────────────────────────────────────────────────────────────────────
//
// This implementation intentionally:
//   • keeps optimizer state outside model weights
//   • adapts each weight independently
//   • uses correct bias correction
//   • lets NaN/Inf gradients propagate instead of hiding them
//     (bad gradients should be fixed at their source)
//
// The result:
//   Adam converges faster and more reliably than SGD for deep networks.
//
// =========================================================================

use crate::common::param::Param;

use super::{ClippingStrategy, Optimizer};

/// Adam optimizer.
///
/// Mental model:
///   - SGD asks: "what does the current gradient say?"
///   - Momentum asks: "what direction have gradients recently agreed on?"
///   - RMSProp asks: "which weights usually get large gradients?"
///   - Adam combines Momentum + RMSProp, then corrects the early-step bias.
pub struct Adam {
    /// Base learning rate before Adam's adaptive scaling.
    pub lr: f32,

    /// Decay rate for the first moment `m`.
    ///
    /// This is the momentum-like part of Adam. A common value is 0.9.
    pub beta1: f32,

    /// Decay rate for the second moment `v`.
    ///
    /// This is the RMSProp-like part of Adam. A common value is 0.999.
    pub beta2: f32,

    /// Small stabilizer used in `sqrt(v_hat) + epsilon`.
    pub epsilon: f32,

    /// Gradient clipping policy applied by `Optimizer::step()` before Adam math.
    pub clipping: ClippingStrategy,

    /// Training step number `t`, starting at 1 inside `update()`.
    ///
    /// Adam needs `t` for bias correction because `m` and `v` start at zero.
    step_count: u64,

    /// First moment buffers.
    ///
    /// Formula:
    ///   m_t = beta1 * m_(t-1) + (1 - beta1) * grad
    ///
    /// Shape: [param_index][row][col], matching every matrix in `params`.
    m: Vec<Vec<Vec<f32>>>,

    /// Second moment buffers.
    ///
    /// Formula:
    ///   v_t = beta2 * v_(t-1) + (1 - beta2) * grad^2
    ///
    /// Shape: [param_index][row][col], matching every matrix in `params`.
    v: Vec<Vec<Vec<f32>>>,
}

impl Adam {
    /// Create Adam with explicit gradient clipping policy.
    ///
    /// Pass `ClippingStrategy::None` for raw Adam, or a clipping strategy when
    /// training can produce unstable gradient spikes.
    pub fn new(lr: f32, clipping: ClippingStrategy) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            step_count: 0,
            clipping,
            m: Vec::new(),
            v: Vec::new(),
        }
    }
}

impl Optimizer for Adam {
    fn clipping(&self) -> &ClippingStrategy {
        &self.clipping
    }

    fn update(&mut self, params: &mut [&mut Param]) {
        // Called by `Optimizer::step()` after gradient clipping has already
        // been applied. This method only contains Adam's parameter-update math.
        self.step_count += 1;
        let t = self.step_count as f32;

        if self.m.is_empty() {
            // Lazy initialization:
            // Adam owns two history buffers per parameter matrix. We create
            // them on the first update because the optimizer constructor does
            // not know the model shape.
            for param in params.iter() {
                self.m.push(param.zeros_like_data());
                self.v.push(param.zeros_like_data());
            }
        }

        assert_eq!(
            self.m.len(),
            params.len(),
            "Adam parameter count changed after optimizer state was initialized"
        );

        // Bias-correction denominators:
        // Because m_0 and v_0 start at zero, the early moving averages are
        // biased too small. Dividing by `(1 - beta^t)` removes that bias.
        let bc1 = 1.0 - self.beta1.powf(t);
        let bc2 = 1.0 - self.beta2.powf(t);

        for (idx, param) in params.iter_mut().enumerate() {
            // `idx` selects which parameter matrix we are updating, and also
            // selects that matrix's matching `m` and `v` history buffers.
            for i in 0..param.data.len() {
                // `i` selects the row inside the current parameter matrix.
                for j in 0..param.data[i].len() {
                    // `j` selects one scalar weight and its gradient.
                    let g = param.grad[i][j];

                    // 1st moment: running average of gradient direction.
                    // Reads as: keep old direction, mix in current gradient.
                    self.m[idx][i][j] = self.beta1 * self.m[idx][i][j] + (1.0 - self.beta1) * g;

                    // 2nd moment: running average of squared gradient size.
                    // Large repeated gradients increase the denominator below.
                    self.v[idx][i][j] = self.beta2 * self.v[idx][i][j] + (1.0 - self.beta2) * g * g;

                    // Bias-corrected estimates:
                    //   m_hat = m_t / (1 - beta1^t)
                    //   v_hat = v_t / (1 - beta2^t)
                    let m_hat = self.m[idx][i][j] / bc1;
                    let v_hat = self.v[idx][i][j] / bc2;

                    // Adaptive direction:
                    //   update_dir = m_hat / (sqrt(v_hat) + epsilon)
                    //
                    // If a weight often has large gradients, sqrt(v_hat) is
                    // larger, so Adam automatically takes a smaller step for
                    // that weight.
                    let update_dir = m_hat / (v_hat.sqrt() + self.epsilon);
                    let update_amount = self.lr * update_dir;

                    // Final Adam update:
                    //   weight = weight - lr * update_dir
                    param.data[i][j] -= update_amount;
                }
            }
        }
    }
}
