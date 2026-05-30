// ════════════════════════════════════════════════════════════════════════════
// ADAMW OPTIMIZER
// ════════════════════════════════════════════════════════════════════════════
//
// AdamW = Adam + decoupled weight decay.
//
// References:
//   - Adam: A Method for Stochastic Optimization (Kingma & Ba, 2015)
//   - Decoupled Weight Decay Regularization (Loshchilov & Hutter, 2019)
//
// ─────────────────────────────────────────────────────────────────────────
// WHY AdamW EXISTS
// ─────────────────────────────────────────────────────────────────────────
//
// Adam already adapts the learning rate per weight:
//
//   m  ← running average of gradient direction
//   v  ← running average of squared gradient size
//
// But regularization is a separate idea:
//
//   weight decay nudges weights toward zero so they do not grow forever.
//
// In classic L2 regularization, weight decay is mixed into the gradient:
//
//   grad = grad + weight_decay * weight
//
// That is awkward for Adam because Adam rescales gradients using `v`.
// The decay strength then gets distorted by Adam's adaptive denominator.
//
// AdamW fixes this by decoupling the two effects:
//
//   1. Adam adaptive update learns from the gradient.
//   2. Weight decay directly shrinks the weight.
//
// ─────────────────────────────────────────────────────────────────────────
// UPDATE FORMULAS
// ─────────────────────────────────────────────────────────────────────────
//
// Adam part:
//   m_t     = beta1 * m_(t-1) + (1 - beta1) * grad
//   v_t     = beta2 * v_(t-1) + (1 - beta2) * grad^2
//   m_hat   = m_t / (1 - beta1^t)
//   v_hat   = v_t / (1 - beta2^t)
//   adam    = m_hat / (sqrt(v_hat) + epsilon)
//
// Decoupled decay part:
//   decay   = weight_decay * weight
//
// Final update used here:
//   weight = weight * (1 - lr * weight_decay)
//   weight = weight - lr * adam
//
// The important part is that `weight_decay` never enters `grad`, `m`, or `v`.
//
// ─────────────────────────────────────────────────────────────────────────
// TYPICAL HYPERPARAMETERS (stable defaults)
// ─────────────────────────────────────────────────────────────────────────
//
//   lr   = 3e-4   (much smaller than SGD's 1e-3 or 5e-3)
//   β₁   = 0.9    (controls momentum)
//   β₂   = 0.999  (controls adaptation)
//   ε    = 1e-8   (stabilization)
//   wd   = 0.01   (common AdamW weight decay starting point)
//
// ─────────────────────────────────────────────────────────────────────────
// CODE STRUCTURE
// ─────────────────────────────────────────────────────────────────────────
//
// 1. struct AdamW with:
//    - lr, beta1, beta2, epsilon, weight_decay
//    - step_count (t)
//    - m: Vec<Vec<Vec<f32>>>  (momentum buffers)
//    - v: Vec<Vec<Vec<f32>>>  (variance buffers)
//
// 2. new() constructor — lazy initialization
//    (buffers created on first step)
//
// 3. step() method with:
//    - t = step_count + 1
//    - Bias correction: (1 - beta^t)
//    - m update: β₁·m + (1-β₁)·g
//    - v update: β₂·v + (1-β₂)·g²
//    - Adaptive step: lr * m̂ / (√v̂ + ε)
//    - Decoupled decay on the old weight, then adaptive Adam update
//
// ─────────────────────────────────────────────────────────────────────────
// FINAL NOTE ON STABILITY
// ─────────────────────────────────────────────────────────────────────────
//
// This implementation intentionally:
//   • keeps optimizer state outside model weights
//   • adapts each weight independently
//   • keeps weight decay separate from gradient moments
//   • uses correct bias correction
//   • lets NaN/Inf gradients propagate instead of hiding them
//     (bad gradients should be fixed at their source)
//
// The result:
//   AdamW usually behaves like Adam, but with cleaner regularization.
//
// =========================================================================

use crate::common::param::Param;

use super::Optimizer;

/// AdamW optimizer.
///
/// Mental model:
///   - SGD asks: "what does the current gradient say?"
///   - Momentum asks: "what direction have gradients recently agreed on?"
///   - RMSProp asks: "which weights usually get large gradients?"
///   - Adam combines Momentum + RMSProp, then corrects the early-step bias.
///   - AdamW adds direct weight shrinkage outside the gradient path.
pub struct AdamW {
    /// Base learning rate before AdamW's adaptive scaling and decay.
    pub lr: f32,

    /// Decay rate for the first moment `m`.
    ///
    /// This is the momentum-like part of AdamW. A common value is 0.9.
    pub beta1: f32,

    /// Decay rate for the second moment `v`.
    ///
    /// This is the RMSProp-like part of AdamW. A common value is 0.999.
    pub beta2: f32,

    /// Small stabilizer used in `sqrt(v_hat) + epsilon`.
    pub epsilon: f32,

    /// Training step number `t`, starting at 1 inside `step()`.
    ///
    /// AdamW needs `t` for bias correction because `m` and `v` start at zero.
    step_count: u64,

    /// Weight decay multiplier (λ).
    ///
    /// In AdamW, this is applied directly to weights, separately from the
    /// gradient-based Adam update. It is not mixed into `grad`, `m`, or `v`.
    pub weight_decay: f32,

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

impl AdamW {
    pub fn new(lr: f32, weight_decay: f32) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            step_count: 0,
            weight_decay, // caller sets this — typical: 0.01

            m: Vec::new(),
            v: Vec::new(),
        }
    }
}

impl Optimizer for AdamW {
    fn step(&mut self, params: &mut [&mut Param]) {
        self.step_count += 1;
        let t = self.step_count as f32;

        if self.m.is_empty() {
            // Lazy initialization:
            // AdamW owns two history buffers per parameter matrix. We create
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
            "AdamW parameter count changed after optimizer state was initialized"
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
                    // larger, so AdamW automatically takes a smaller step for
                    // that weight.
                    let update_dir = m_hat / (v_hat.sqrt() + self.epsilon);
                    let update_amount = self.lr * update_dir;

                    // AdamW-only decoupled weight decay:
                    //   weight = weight * (1 - lr * weight_decay)
                    //
                    // This shrinkage is intentionally outside the gradient
                    // moments. `weight_decay` never changes `g`, `m`, or `v`.
                    param.data[i][j] *= 1.0 - self.lr * self.weight_decay;

                    // Final Adam-style adaptive update:
                    //   weight = weight - lr * update_dir
                    //
                    // Apply this after weight decay so the adaptive Adam
                    // update itself is not also shrunk by weight decay.
                    param.data[i][j] -= update_amount;
                }
            }
        }
    }
}
