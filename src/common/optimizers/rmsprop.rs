use crate::common::param::Param;

use super::{ClippingStrategy, Optimizer};

/// RMSProp keeps a running average of squared gradients for each weight.
///
/// Update idea:
///   sq_avg = decay_rate * sq_avg + (1 - decay_rate) * grad^2
///   weight = weight - lr * grad / (sqrt(sq_avg) + epsilon)
///
/// Read `sq_avg` as "how large have this weight's gradients been recently?"
/// It is squared, so sign does not matter: -10 and +10 both mean "large".
///
/// Intuition:
///   - weights with consistently large gradients get smaller effective steps.
///   - weights with small gradients can still move because their denominator is small.
///   - this is adaptive learning rate scaling, not momentum.
///
/// Difference from SGDM:
///   RMSProp remembers gradient size, not direction. It scales the current
///   gradient but does not build a velocity direction.
///
/// Clipping:
///   The shared `Optimizer::step()` wrapper applies this optimizer's
///   `ClippingStrategy` before calling `RMSProp::update()`.
pub struct RMSProp {
    /// Base learning rate before the RMS scaling is applied.
    pub lr: f32,

    /// Decay rate for the squared-gradient average.
    ///
    /// Higher values remember older gradients for longer.
    pub decay_rate: f32,

    /// Small stabilizer so division never uses zero as the denominator.
    pub epsilon: f32,

    /// Gradient clipping policy applied before the RMSProp update.
    pub clipping: ClippingStrategy,

    /// Per-parameter running mean of squared gradients.
    ///
    /// Shape: [param_index][row][col], matching every matrix in `params`.
    /// `sq_avg[idx][i][j]` belongs to `params[idx].data[i][j]`.
    ///
    /// This is intentionally separate from `Param` because optimizer state
    /// belongs to the optimizer, not to the model weights.
    sq_avg: Vec<Vec<Vec<f32>>>,
}

impl RMSProp {
    /// Create RMSProp with explicit gradient clipping policy.
    ///
    /// Pass `ClippingStrategy::None` for raw RMSProp, or a clipping strategy
    /// when training can produce unstable gradient spikes.
    pub fn new(lr: f32, clipping: ClippingStrategy) -> Self {
        Self {
            lr,
            decay_rate: 0.9, // sensible default
            epsilon: 1e-8,   // sensible default
            clipping,
            sq_avg: Vec::new(),
        }
    }
}

impl Optimizer for RMSProp {
    fn clipping(&self) -> &ClippingStrategy {
        &self.clipping
    }

    fn update(&mut self, params: &mut [&mut Param]) {
        // Called by `Optimizer::step()` after gradient clipping has already
        // been applied. This method only contains RMSProp's update math.
        if self.sq_avg.is_empty() {
            // Lazy initialization:
            // At construction time we do not know how many parameter matrices
            // the model will pass in. On the first update, create one zero
            // buffer per parameter with the exact same 2D shape.
            for param in params.iter() {
                self.sq_avg.push(param.zeros_like_data());
            }
        }

        for (idx, param) in params.iter_mut().enumerate() {
            // `idx` selects the optimizer buffer for this parameter matrix.
            for i in 0..param.data.len() {
                // `i` selects the row inside the current parameter matrix.
                for j in 0..param.data[i].len() {
                    // `j` selects the individual weight and gradient.
                    let g = param.grad[i][j];

                    // Running squared-gradient average:
                    // New gradients contribute `(1 - decay_rate) * g^2`.
                    // Old history remains through `decay_rate * sq_avg`.
                    //
                    // Formula:
                    //   sq_avg = decay_rate * sq_avg + (1 - decay_rate) * g^2
                    self.sq_avg[idx][i][j] =
                        self.decay_rate * self.sq_avg[idx][i][j] + (1.0 - self.decay_rate) * g * g;

                    // Adaptive weight update:
                    // divide by root-mean-square gradient so frequently large
                    // gradients do not take uncontrolled large steps.
                    //
                    // Formula:
                    //   weight = weight - lr * g / (sqrt(sq_avg) + epsilon)
                    param.data[i][j] -= self.lr * g / (self.sq_avg[idx][i][j].sqrt() + self.epsilon)
                }
            }
        }
    }
}
