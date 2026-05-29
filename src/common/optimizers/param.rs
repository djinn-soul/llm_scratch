// ════════════════════════════════════════════════════════════════════════════
// PARAM WRAPPER
// ════════════════════════════════════════════════════════════════════════════
// Holds a parameter and its gradient in a 2D vec: [1][d_model].
// This shape is chosen so it can be treated like a matrix for matmul:
//   - multiplication with [N][d_model]: broadcast works automatically
//   - transpose: [d_model][1]
//
// The wrapped [1] dimension is conceptual: it allows matmul broadcasting
// but isn't used in single-vector operations like LayerNorm.
pub struct Param {
    /// Trainable values used during the forward pass.
    pub data: Vec<Vec<f32>>,

    /// Gradient values produced by the backward pass.
    ///
    /// Must have the same shape as `data` so optimizers can update weights
    /// element-by-element.
    pub grad: Vec<Vec<f32>>,
}

impl Param {
    pub fn new(data: Vec<Vec<f32>>, grad: Vec<Vec<f32>>) -> Self {
        Self { data, grad }
    }

    pub fn zero_grad(&mut self) {
        // Gradients accumulate during backprop, so clear them before the next
        // forward/backward training step.
        for row in &mut self.grad {
            for g in row {
                *g = 0.0;
            }
        }
    }
}
