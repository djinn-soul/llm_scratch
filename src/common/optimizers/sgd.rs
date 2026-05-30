use crate::common::param::Param;

use super::{ClippingStrategy, Optimizer};

/// Stochastic Gradient Descent (SGD) — the simplest optimizer.
///
/// Update rule (applied to every weight):
///   w = w - lr * grad
///
/// Where:
///   - `w` is one trainable scalar weight.
///   - `grad` is dLoss/dw, produced by backprop.
///   - `lr` is the learning rate, the knob that controls step size.
///
/// Intuition:
///   - `grad`  tells us which direction makes the loss INCREASE.
///   - We subtract it, so we move in the direction that makes the loss DECREASE.
///   - `lr` (learning rate) controls how big each step is.
///     Too large → overshoots, loss explodes.
///     Too small → learns correctly but very slowly.
///
/// Memory:
///   SGD has no history buffer. Every update depends only on the current
///   `param.grad` values from the latest backward pass.
///
/// Clipping:
///   The shared `Optimizer::step()` wrapper applies this optimizer's
///   `ClippingStrategy` before calling `SGD::update()`.
pub struct SGD {
    /// Learning rate — how big each weight update step is.
    ///
    /// Typical range: 0.001-0.1, depending on model and batch size.
    pub lr: f32,
    /// Gradient clipping policy applied before the SGD update.
    pub clipping: ClippingStrategy,
}

impl SGD {
    /// Create SGD with explicit gradient clipping policy.
    ///
    /// Pass `ClippingStrategy::None` for raw SGD, or a clipping strategy when
    /// training can produce unstable gradient spikes.
    pub fn new(lr: f32, clipping: ClippingStrategy) -> Self {
        Self { lr, clipping }
    }
}

impl Optimizer for SGD {
    fn clipping(&self) -> &ClippingStrategy {
        &self.clipping
    }

    fn update(&mut self, params: &mut [&mut Param]) {
        // Called by `Optimizer::step()` after gradient clipping has already
        // been applied. This method only contains SGD's update rule.
        // SGD has no memory between steps, so there are no optimizer-owned
        // buffers to initialize. It only needs the latest gradients.

        // Loop over every trainable parameter matrix in the model.
        for param in params {
            // Loop over every row (e.g. each neuron or embedding vector).
            for i in 0..param.data.len() {
                // Loop over every individual weight in that row.
                for j in 0..param.data[i].len() {
                    // Core SGD update:
                    //   w = w - lr * grad
                    //
                    // `grad[i][j]` was filled by the backward pass before
                    // the training loop called `optimizer.step(...)`.
                    param.data[i][j] -= self.lr * param.grad[i][j];
                }
            }
        }
    }
}
