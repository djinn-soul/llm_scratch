use super::{Optimizer, Param};

/// SGD with Momentum.
///
/// Plain SGD only looks at the current gradient. Momentum also remembers the
/// recent update direction in a velocity buffer:
///
///   velocity = momentum * velocity + lr * grad
///   weight   = weight - velocity
///
/// Intuition:
///   - repeated gradients in the same direction build speed.
///   - alternating gradients partially cancel out.
///   - this usually gives smoother movement through noisy loss landscapes.
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
    /// The field name is kept as-is to avoid behavior/API churn in this pass.
    pub velociy: Vec<Vec<Vec<f32>>>,
}

impl SGDM {
    pub fn new(lr: f32, momentum: f32) -> Self {
        Self {
            lr,
            momentum,
            velociy: Vec::new(),
        }
    }
}

impl Optimizer for SGDM {
    fn step(&mut self, params: &mut [&mut Param]) {
        if self.velociy.is_empty() {
            // Lazy initialization:
            // create one zero-filled velocity buffer per parameter matrix.
            // This keeps the constructor independent from model shape.
            for param in params.iter() {
                self.velociy
                    .push(vec![vec![0.0; param.data[0].len()]; param.data.len()]);
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
                    self.velociy[idx][i][j] = self.momentum * self.velociy[idx][i][j] + self.lr * g;

                    // Weight update:
                    // move opposite the accumulated velocity.
                    param.data[i][j] -= self.velociy[idx][i][j];
                }
            }
        }
    }
}
