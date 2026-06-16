use anyhow::{Ok, Result};
use candle_core::{DType, Device, Tensor};

// MANUAL BACKPROPAGATION DENOISING MLP
//
// This is the small neural network used as the noise predictor.
//
// In diffusion training we create x_t by adding known random noise epsilon to
// clean data x_0. The model's job is:
//
// input:  noisy sample x_t + timestep embedding
// output: predicted epsilon noise
//
// During training, target is the exact noise tensor that was used by the
// scheduler. If the model predicts that noise well, reverse diffusion can
// subtract predicted noise step by step.
pub struct SimpleDenoisingMlp {
    pub w1: Tensor, // [hidden_dim, in_dim]
    pub b1: Tensor, // [hidden_dim]
    pub w2: Tensor, // [out_dim, hidden_dim]
    pub b2: Tensor, // [out_dim]
}

pub struct Gradients {
    pub dw1: Tensor,
    pub db1: Tensor,
    pub dw2: Tensor,
    pub db2: Tensor,
}

impl SimpleDenoisingMlp {
    pub fn new(in_dim: usize, hidden_dim: usize, out_dim: usize, device: &Device) -> Result<Self> {
        // Build trainable parameters for a 2-layer MLP.
        //
        // Dimension meaning in this diffusion model:
        //
        // in_dim     = noisy data width + timestep embedding width
        //            = concat(x_t, time_embedding) width
        //
        // hidden_dim = number of hidden neurons after the first layer
        //
        // out_dim    = predicted noise width
        //            = usually same width as x_0 / x_t / epsilon
        //
        // Forward equations:
        //
        // z1 = v @ w1^T + b1
        // a1 = leaky_relu(z1)
        // pred = a1 @ w2^T + b2
        //
        // Shape plan:
        //
        // v    [batch][in_dim]
        // w1   [hidden_dim][in_dim]
        // b1   [hidden_dim]
        // z1   [batch][hidden_dim]
        //
        // a1   [batch][hidden_dim]
        // w2   [out_dim][hidden_dim]
        // b2   [out_dim]
        // pred [batch][out_dim]
        //
        // He-style scaling:
        //
        // Each hidden neuron sums many input values. If in_dim is large and
        // random weights are too large, z1 can explode before training starts.
        // sqrt(2 / input_width) keeps starting activations in a useful range.
        let scale1 = (2.0f64 / in_dim as f64).sqrt();

        // w1 maps the input vector v into hidden neurons.
        //
        // Stored shape is [hidden_dim][in_dim] because each row is one hidden
        // neuron's weights. During forward we use w1.t() so matmul lines up:
        //
        // v [batch][in_dim] @ w1^T [in_dim][hidden_dim]
        // -> z1 [batch][hidden_dim]
        let w1 = (Tensor::randn(0.0f32, 1.0f32, (hidden_dim, in_dim), device)? * scale1)?;

        // Bias starts at zero because random w1 already breaks symmetry.
        // b1 has one value per hidden neuron and is broadcast across the batch.
        let b1 = Tensor::zeros(hidden_dim, DType::F32, device)?;

        // Layer 2 receives hidden activations, so its input width is
        // hidden_dim, not in_dim.
        let scale2 = (2.0f64 / hidden_dim as f64).sqrt();

        // w2 maps hidden activations into final noise prediction coordinates.
        //
        // a1 [batch][hidden_dim] @ w2^T [hidden_dim][out_dim]
        // -> pred [batch][out_dim]
        let w2 = (Tensor::randn(0.0f32, 1.0f32, (out_dim, hidden_dim), device)? * scale2)?;

        // b2 is one bias per predicted noise coordinate.
        let b2 = Tensor::zeros(out_dim, DType::F32, device)?;

        Ok(Self { w1, b1, w2, b2 })
    }

    pub fn forward(&self, v: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        // v is the full conditioning input:
        //
        // noisy sample x_t   [batch][data_dim]
        // timestep embedding [batch][time_emb_dim]
        //
        // concat gives:
        //
        // v [batch][in_dim]
        // in_dim = data_dim + time_emb_dim
        //
        // First affine layer:
        //
        // z1 = v @ w1^T + b1
        //
        // v [batch][in_dim] @ w1^T [in_dim][hidden_dim]
        // -> z1 [batch][hidden_dim]
        let z1 = v.matmul(&self.w1.t()?)?.broadcast_add(&self.b1)?;

        // LeakyReLU = max(0.01 * z1, z1). This keeps a small gradient for
        // negative values so hidden units do not become permanently silent.
        //
        // Example:
        //
        // z1 row = [-0.4, 1.2, 0.0]
        // a1 row = [-0.004, 1.2, 0.0]
        let a1 = z1.maximum(&z1.affine(0.01, 0.0)?)?;

        // Final affine layer:
        //
        // pred = a1 @ w2^T + b2
        //
        // If out_dim == data_dim, pred has the same shape as the real noise:
        //
        // target epsilon [batch][out_dim]
        // pred epsilon   [batch][out_dim]
        //
        // a1 [batch][hidden_dim] @ w2^T [hidden_dim][out_dim]
        // -> pred [batch][out_dim], same shape as the target noise.
        let pred = a1.matmul(&self.w2.t()?)?.broadcast_add(&self.b2)?;

        // Return more than pred because this implementation does manual
        // backpropagation:
        //
        // pred: compare against target noise
        // a1:   compute dw2 = delta2^T @ a1
        // z1:   build the LeakyReLU gradient mask for delta1
        Ok((pred, a1, z1))
    }

    pub fn backward(
        &self,
        v: &Tensor,
        a1: &Tensor,
        z1: &Tensor,
        pred: &Tensor,
        target: &Tensor,
    ) -> Result<Gradients> {
        // Exact MSE gradient:
        //
        // dL/dpred = (2.0 / (batch_size * out_dim)) * (pred - target)
        //
        // delta2 is the output-layer error after applying that MSE scale.
        let batch_size = pred.dim(0)?;
        let out_dim = pred.dim(1)?;
        let scale = 2.0 / (batch_size * out_dim) as f64;
        let delta2 = pred.sub(target)?.affine(scale, 0.0)?;

        // Layer 2 gradients from:
        //
        // pred = a1 @ w2^T + b2
        //
        // dw2 = delta2^T @ a1
        // db2 = sum rows of delta2
        //
        // dw2 shape is [out_dim][hidden_dim], matching w2.
        let dw2 = delta2.t()?.matmul(&a1)?;
        let db2 = delta2.sum(0)?;

        // LeakyReLU backward: positive z1 gets 1.0, negative z1 gets 0.01.
        //
        // ge(0) creates 1.0 for positive entries and 0.0 for negative entries.
        // affine(0.99, 0.01) converts that into:
        //
        // positive: 1.0 * 0.99 + 0.01 = 1.0
        // negative: 0.0 * 0.99 + 0.01 = 0.01
        let relu_grad = z1.ge(0.0f32)?.to_dtype(DType::F32)?.affine(0.99, 0.01)?;

        // Move output error back through w2, then apply activation gradient.
        //
        // delta1 = (delta2 @ w2) * relu_grad
        //
        // delta1 shape: [batch][hidden_dim]
        let delta1 = delta2.matmul(&self.w2)?.mul(&relu_grad)?;

        // Layer 1 gradients from:
        //
        // z1 = v @ w1^T + b1
        //
        // dw1 = delta1^T @ v
        // db1 = sum rows of delta1
        //
        // dw1 shape is [hidden_dim][in_dim], matching w1.
        let dw1 = delta1.t()?.matmul(&v)?;
        let db1 = delta1.sum(0)?;

        Ok(Gradients { dw1, db1, dw2, db2 })
    }

    pub fn update(&mut self, grads: &Gradients, lr: f64, _batch_size: usize) -> Result<()> {
        // Gradients are already averaged in backward, so SGD is:
        //
        // param = param - lr * grad.
        //
        // Candle tensor ops return new tensors instead of changing the old
        // tensor in place, so each updated parameter is assigned back to self.
        self.w1 = self.w1.sub(&grads.dw1.affine(lr, 0.0)?)?;
        self.b1 = self.b1.sub(&grads.db1.affine(lr, 0.0)?)?;
        self.w2 = self.w2.sub(&grads.dw2.affine(lr, 0.0)?)?;
        self.b2 = self.b2.sub(&grads.db2.affine(lr, 0.0)?)?;
        Ok(())
    }
}
