// https://www.gilesthomas.com/2026/02/llm-from-scratch-32b-interventions-gradient-clipping
/// Gradient clipping policy applied before an optimizer updates weights.
///
/// This belongs to the optimizer layer because clipping is training policy,
/// not part of a parameter's stored data/gradient shape.
///
/// Use `None` when you want raw gradients. Use clipping when gradients can
/// spike during training and make the optimizer take an unstable step.
///
/// Why this exists:
///   Backprop can sometimes produce very large gradients. If the optimizer uses
///   those raw numbers, one step can throw weights far away from a useful
///   region. Clipping keeps the gradient update bounded before Adam/SGD/etc.
///   sees it.
///
/// Important:
///   Clipping changes `Param.grad`, not `Param.data`. The actual weight update
///   still happens later inside the concrete optimizer's `update()` method.
///
/// How the gradient matrix gets clipped:
///   Each `Param` has a gradient matrix with the same shape as its data matrix:
///
///   ```text
///   grad = [
///     [g00, g01],
///     [g10, g11],
///   ]
///   ```
///
///   Clipping walks through those same row/column positions and rewrites the
///   gradient cells in place:
///
///   ```text
///   grad[row][col] = clipped_or_scaled_value
///   ```
///
///   After that, the optimizer reads the clipped gradients and updates weights:
///
///   ```text
///   data[row][col] -= optimizer_step_based_on(grad[row][col])
///   ```
///
/// Advanced Learning Notes:
///   - **GlobalNorm**: Highly recommended for LLM pretraining because it scales the entire
///     multidimensional gradient vector together. This **perfectly preserves the update direction**,
///     preventing distortion of optimization steps.
///   - **Adaptive Gradient Clipping (AGC)**: Clips gradients relative to the magnitude of the
///     corresponding weight parameters. Useful for extremely large models or architectures
///     without normalization layers. We don't need it because our GPT model has `LayerNorm`
///     after every attention and feed-forward block, keeping signals stable.
///   - **Gradient Scaling**: Scaled losses dynamically to prevent underflow (tiny gradients rounding to zero)
///     in mixed-precision (FP16) GPU pretraining.
// https://mbrenndoerfer.com/writing/gradient-clipping-deep-learning
use crate::common::param::Param;

pub enum ClippingStrategy {
    /// No clipping.
    ///
    /// Use this when gradients are already stable or when you want to inspect
    /// the raw optimizer behavior without safety limits.
    None,

    /// Clamp every individual gradient value to [-max_val, max_val].
    ///
    /// Example:
    ///   max_val = 1.0
    ///   [-3.0, 0.2, 5.0] -> [-1.0, 0.2, 1.0]
    ///
    /// This is simple and easy to reason about, but it ignores the total
    /// direction of the gradient vector.
    Value(f32),

    /// Clip each parameter matrix's L2 norm independently.
    ///
    /// For each `Param`, compute:
    ///   norm = sqrt(sum(g^2))
    ///
    /// If `norm > max_norm`, scale every gradient in that parameter by:
    ///   scale = max_norm / norm
    ///
    /// This keeps each weight matrix's gradient length bounded.
    Norm(f32),

    /// Clip the L2 norm of the entire model's gradients combined.
    ///
    /// This computes one norm across every gradient in every parameter:
    ///   total_norm = sqrt(sum(all_model_grads^2))
    ///
    /// If `total_norm > max_norm`, all gradients are scaled by the same factor.
    /// This preserves the global gradient direction better than value clipping.
    ///
    /// Common LLM training choice:
    ///   GlobalNorm(1.0)
    GlobalNorm(f32),
}

impl ClippingStrategy {
    pub fn apply(&self, params: &mut [&mut Param]) {
        match *self {
            // Leave gradients exactly as backprop produced them.
            ClippingStrategy::None => {}

            ClippingStrategy::Value(max_val) => {
                // Element-wise clipping:
                // each scalar gradient cell is capped independently.
                //
                // Matrix example with max_val = 1.0:
                //
                //   before grad = [
                //     [ 3.5, -0.2],
                //     [-7.0,  0.8],
                //   ]
                //
                //   after grad = [
                //     [ 1.0, -0.2],
                //     [-1.0,  0.8],
                //   ]
                //
                // Only values outside [-1.0, 1.0] change. This does not look
                // at the whole matrix direction; it just clamps each cell.
                for param in params.iter_mut() {
                    for row in &mut param.grad {
                        for g in row {
                            *g = g.clamp(-max_val, max_val);
                        }
                    }
                }
            }

            ClippingStrategy::Norm(max_norm) => {
                for param in params.iter_mut() {
                    // Compute this parameter matrix's L2 norm:
                    // sqrt(g_1^2 + g_2^2 + ... + g_n^2)
                    //
                    // Matrix example:
                    //
                    //   grad = [
                    //     [3.0, 4.0],
                    //   ]
                    //
                    //   norm = sqrt(3^2 + 4^2) = 5.0
                    //
                    // If max_norm = 1.0, scale = 1.0 / 5.0 = 0.2.
                    // Every cell in this parameter matrix gets multiplied by
                    // the same scale:
                    //
                    //   [3.0, 4.0] -> [0.6, 0.8]
                    let mut norm = 0.0;
                    for row in &param.grad {
                        for &g in row {
                            norm += g * g;
                        }
                    }
                    norm = norm.sqrt();

                    if norm > max_norm {
                        // Scale the whole parameter matrix so its norm becomes
                        // `max_norm` while keeping its gradient direction.
                        //
                        // This updates every cell:
                        //   grad[row][col] = grad[row][col] * scale
                        let scale = max_norm / (norm + 1e-6);
                        for row in &mut param.grad {
                            for g in row {
                                *g *= scale;
                            }
                        }
                    }
                }
            }

            ClippingStrategy::GlobalNorm(max_norm) => {
                // Compute one L2 norm for the whole model, across every
                // trainable parameter's gradient.
                //
                // Example with two parameter matrices:
                //
                //   param0.grad = [[3.0, 4.0]]
                //   param1.grad = [[0.0, 12.0]]
                //
                //   total_norm = sqrt(3^2 + 4^2 + 0^2 + 12^2) = 13.0
                //
                // If max_norm = 1.0, scale = 1.0 / 13.0.
                // Every gradient cell in every parameter gets the same scale.
                // That keeps all matrices moving in the same relative
                // direction, just with a smaller total step.
                let mut total_norm = 0.0;
                for param in params.iter() {
                    for row in &param.grad {
                        for &g in row {
                            total_norm += g * g;
                        }
                    }
                }
                total_norm = total_norm.sqrt();

                if total_norm > max_norm {
                    // Use one shared scale for every gradient. This preserves
                    // relative sizes between parameters better than clipping
                    // each parameter independently.
                    //
                    // This updates every gradient matrix:
                    //   param.grad[row][col] = param.grad[row][col] * scale
                    let scale = max_norm / (total_norm + 1e-6);
                    for param in params.iter_mut() {
                        for row in &mut param.grad {
                            for g in row {
                                *g *= scale;
                            }
                        }
                    }
                }
            }
        }
    }
}

/// The `Optimizer` trait defines the contract every optimizer must implement.
///
/// Any optimizer (SGD, Adam, AdamW, ...) implements `update()`.
/// The shared `step()` wrapper handles cross-optimizer policy first:
/// gradient clipping, then the optimizer-specific weight update.
///
/// This lets the training loop work with any optimizer without changing code:
///   `optimizer.step(&mut params);`
///
/// Training-loop order:
///   1. clear old gradients with `zero_grad()`
///   2. run forward pass
///   3. run backward pass so each `Param.grad` is filled
///   4. call `optimizer.step(&mut params)`
///   5. `step()` clips gradients, then calls `update()` to change `Param.data`
///
/// How matrix weights get updated:
///   Every trainable weight lives in a `Param`.
///
///   Param.data is the actual weight matrix used by the forward pass:
///     data[row][col] = current weight value
///
///   Param.grad is the matching gradient matrix filled by backward pass:
///     grad[row][col] = dLoss / d(data[row][col])
///
///   Optimizers loop over the same matrix coordinates:
///     for each param
///       for each row i
///         for each column j
///           read  grad[i][j]
///           write data[i][j]
///
///   SGD writes directly:
///     data[i][j] -= lr * grad[i][j]
///
///   Adam/AdamW/RMSProp/SGDM still update the same `data[i][j]`, but first use
///   optimizer-owned state buffers such as `m`, `v`, `sq_avg`, or `velocity`.
///   Those buffers have the same shape as `Param.data`, so
///   `state[param_index][i][j]` belongs to `params[param_index].data[i][j]`.
///
///   Key mental model:
///     backward decides the direction through `grad`;
///     clipping optionally limits that direction;
///     optimizer update decides how far each matrix element moves.
///
/// Design note:
///   The model owns weights and gradients through `Param`.
///   The optimizer owns training policy (`ClippingStrategy`), update strategy,
///   and any extra state, such as momentum buffers or squared-gradient averages.
pub trait Optimizer {
    /// Update all parameter weights.
    ///
    /// Every optimizer implements this by mutating `Param.data` in place.
    /// The matching `Param.grad` matrix is read as input; it is not cleared
    /// here. The training loop clears gradients before the next forward pass.
    fn update(&mut self, params: &mut [&mut Param]);

    /// Returns the clipping strategy configured for this optimizer.
    ///
    /// Concrete optimizers override this to return their stored field.
    /// The default is useful for simple custom optimizers that do not expose
    /// clipping yet.
    fn clipping(&self) -> &ClippingStrategy {
        &ClippingStrategy::None
    }
    /// Public training-loop entry point.
    ///
    /// Update all parameter weights using their stored gradients.
    /// Called once per training step, AFTER the backward pass fills `param.grad`.
    fn step(&mut self, params: &mut [&mut Param]) {
        // 1. Apply the optimizer's configured clipping policy.
        self.clipping().apply(params);
        // 2. Run the concrete optimizer's update rule.
        self.update(params);
    }
}
