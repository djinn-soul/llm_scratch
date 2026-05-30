// ════════════════════════════════════════════════════════════════════════════
// PARAM WRAPPER
// ════════════════════════════════════════════════════════════════════════════
// Holds trainable values and their matching gradients.
//
// Most model weights are stored as 2D matrices. Vector-shaped parameters, such
// as LayerNorm gamma/beta, are stored as one-row matrices so every optimizer can
// use the same element-by-element update path.
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
    pub fn new(data: Vec<Vec<f32>>) -> Self {
        let grad = zero_like(&data);
        Self { data, grad }
    }

    pub fn with_grad(data: Vec<Vec<f32>>, grad: Vec<Vec<f32>>) -> Self {
        assert_same_shape(&data, &grad);
        Self { data, grad }
    }

    pub fn zeros_like_data(&self) -> Vec<Vec<f32>> {
        zero_like(&self.data)
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

fn zero_like(data: &[Vec<f32>]) -> Vec<Vec<f32>> {
    data.iter().map(|row| vec![0.0; row.len()]).collect()
}

fn assert_same_shape(data: &[Vec<f32>], grad: &[Vec<f32>]) {
    assert_eq!(data.len(), grad.len(), "Param data/grad row count mismatch");

    for (row_idx, (data_row, grad_row)) in data.iter().zip(grad.iter()).enumerate() {
        assert_eq!(
            data_row.len(),
            grad_row.len(),
            "Param data/grad column count mismatch at row {row_idx}"
        );
    }
}
