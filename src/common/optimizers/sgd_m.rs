use crate::common::param::Param;

use super::Optimizer;

/// SGD with Momentum.
///
/// Plain SGD only looks at the current gradient. Momentum also remembers the
/// recent update direction in a velocity buffer:
///
///   velocity = momentum * velocity + lr * grad
///   weight   = weight - velocity
///
/// Read the velocity formula as:
///   - keep `momentum` percent of the previous direction.
///   - add the new gradient step for this training example/batch.
///
/// Intuition:
///   - repeated gradients in the same direction build speed.
///   - alternating gradients partially cancel out.
///   - this usually gives smoother movement through noisy loss landscapes.
///
/// Difference from Adam:
///   SGDM remembers direction only. It does not track squared gradients, so it
///   does not shrink steps separately for weights with large gradient history.
pub struct SGDM {
    /// Base learning rate used when adding the current gradient to velocity.
    pub lr: f32,

    /// How much of the previous velocity is kept each step.
    ///
    /// A common value is 0.9. Higher values remember longer history.
    pub momentum: f32,

    /// Per-parameter velocity buffers.
    ///
    /// Shape: [param_index][row][col], matching every matrix in `params`.
    /// `velocity[idx][i][j]` is the previous accumulated step for
    /// `params[idx].data[i][j]`.
    pub velocity: Vec<Vec<Vec<f32>>>,
}

impl SGDM {
    pub fn new(lr: f32, momentum: f32) -> Self {
        Self {
            lr,
            momentum,
            velocity: Vec::new(),
        }
    }
}

impl Optimizer for SGDM {
    fn step(&mut self, params: &mut [&mut Param]) {
        if self.velocity.is_empty() {
            // Lazy initialization:
            // create one zero-filled velocity buffer per parameter matrix.
            // This keeps the constructor independent from model shape.
            for param in params.iter() {
                self.velocity.push(param.zeros_like_data());
            }
        }

        for (idx, param) in params.iter_mut().enumerate() {
            // `idx` selects the matching velocity buffer for this parameter.
            for i in 0..param.data.len() {
                // `i` selects the row inside the current parameter matrix.
                for j in 0..param.data[i].len() {
                    // `j` selects one scalar weight and its gradient.
                    let g = param.grad[i][j];

                    // Momentum update:
                    // carry forward part of the previous velocity, then add
                    // the current gradient scaled by the learning rate.
                    //
                    // Formula:
                    //   velocity = momentum * velocity + lr * grad
                    self.velocity[idx][i][j] =
                        self.momentum * self.velocity[idx][i][j] + self.lr * g;

                    // Weight update:
                    //   weight = weight - velocity
                    //
                    // We subtract because gradients point uphill in loss.
                    param.data[i][j] -= self.velocity[idx][i][j];
                }
            }
        }
    }
}
