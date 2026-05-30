use crate::common::param::Param;

use super::Optimizer;

/// Stochastic Gradient Descent (SGD) — the simplest optimizer.
///
/// Update rule (applied to every weight):
///   w = w - lr * grad
///
/// Intuition:
///   - `grad`  tells us which direction makes the loss INCREASE.
///   - We subtract it, so we move in the direction that makes the loss DECREASE.
///   - `lr` (learning rate) controls how big each step is.
///     Too large → overshoots, loss explodes.
///     Too small → learns correctly but very slowly.
pub struct SGD {
    /// Learning rate — how big each weight update step is. Typical range: 0.001–0.1
    pub lr: f32,
}

impl SGD {
    pub fn new(lr: f32) -> Self {
        Self { lr }
    }
}

impl Optimizer for SGD {
    fn step(&mut self, params: &mut [&mut Param]) {
        // SGD has no memory between steps, so there are no optimizer-owned
        // buffers to initialize. It only needs the latest gradients.

        // Loop over every trainable parameter matrix in the model
        for param in params {
            // Loop over every row (e.g. each neuron or embedding vector)
            for i in 0..param.data.len() {
                // Loop over every individual weight in that row
                for j in 0..param.data[i].len() {
                    // Core SGD update: w = w - lr * grad
                    // grad[i][j] was filled by the backward pass
                    param.data[i][j] -= self.lr * param.grad[i][j];
                }
            }
        }
    }
}
