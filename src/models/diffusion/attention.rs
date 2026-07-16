use anyhow::Result;
use candle_core::{Device, Tensor};

pub struct SpatialSelfAttention {
    pub w_q: Tensor,
    pub w_k: Tensor,
    pub w_v: Tensor,
}

impl SpatialSelfAttention {
    pub fn new(channels: usize, device: &Device) -> Result<Self> {
        let scale = (1.0f64 / channels as f64).sqrt();
        let w_q = (Tensor::randn(0.0f32, 1.0f32, (channels, channels), device)? * scale)?;
        let w_k = (Tensor::randn(0.0f32, 1.0f32, (channels, channels), device)? * scale)?;
        let w_v = (Tensor::randn(0.0f32, 1.0f32, (channels, channels), device)? * scale)?;
        Ok(Self { w_q, w_k, w_v })
    }

    pub fn forward(&self, x: &Tensor) -> Result<(Tensor, Vec<Tensor>)> {
        let (b, c, h, w) = x.dims4()?;
        let n = h * w;

        // 1. Reshape to sequence format: [B, C, N]
        let x_seq = x.reshape((b, c, n))?.contiguous()?;

        // 2. Project Q, K, V
        let q = self.w_q.broadcast_matmul(&x_seq)?;
        let k = self.w_k.broadcast_matmul(&x_seq)?;
        let v = self.w_v.broadcast_matmul(&x_seq)?;

        // 3. Compute attention scores: S = (K^T @ Q) / sqrt(C)
        let scale = 1.0 / (c as f64).sqrt();
        let scores = k
            .transpose(1, 2)?
            .broadcast_matmul(&q)?
            .affine(scale, 0.0)?;

        // 4. Softmax over key dimension (dim 2)
        let max_scores = scores.max_keepdim(2)?;
        let scores_exp = scores
            .sub(&max_scores.broadcast_as(scores.shape())?)?
            .exp()?;
        let sum_exp = scores_exp.sum_keepdim(2)?;
        let attn_weights = scores_exp.div(&sum_exp.broadcast_as(scores_exp.shape())?)?;

        // 5. Output: O = V @ A^T
        let output_seq = v.broadcast_matmul(&attn_weights.transpose(1, 2)?)?;
        let output = output_seq.reshape((b, c, h, w))?;

        let cached = vec![x_seq, q, k, v, scores, attn_weights];
        Ok((output, cached))
    }

    pub fn backward(
        &self,
        intermediates: &[Tensor],
        delta_y: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        let b = delta_y.dim(0)?;
        let c = self.w_q.dim(0)?;
        let h = delta_y.dim(2)?;
        let w = delta_y.dim(3)?;
        let n = h * w;

        let x_seq = &intermediates[0];
        let q = &intermediates[1];
        let k = &intermediates[2];
        let v = &intermediates[3];
        let _scores = &intermediates[4];
        let attn_weights = &intermediates[5];

        let delta_y_seq = delta_y.reshape((b, c, n))?;

        // 1. Gradients w.r.t V and A^T
        let delta_v = delta_y_seq.broadcast_matmul(attn_weights)?;
        let delta_at = v.transpose(1, 2)?.broadcast_matmul(&delta_y_seq)?;
        let delta_a = delta_at.transpose(1, 2)?;

        // 2. Softmax backward: dS = A * (dA - sum(dA * A, keepdim=2))
        let sum_term = delta_a.mul(attn_weights)?.sum_keepdim(2)?;
        let delta_s = attn_weights.mul(&delta_a.sub(&sum_term.broadcast_as(delta_a.shape())?)?)?;

        // 3. Gradients w.r.t Q and K
        let scale = 1.0 / (c as f64).sqrt();
        let delta_q = k.broadcast_matmul(&delta_s)?.affine(scale, 0.0)?;
        let delta_k = q
            .broadcast_matmul(&delta_s.transpose(1, 2)?)?
            .affine(scale, 0.0)?;

        // 4. Gradients w.r.t projection weights w_q, w_k, w_v
        let d_wq = delta_q.broadcast_matmul(&x_seq.transpose(1, 2)?)?.sum(0)?;
        let d_wk = delta_k.broadcast_matmul(&x_seq.transpose(1, 2)?)?.sum(0)?;
        let d_wv = delta_v.broadcast_matmul(&x_seq.transpose(1, 2)?)?.sum(0)?;

        // 5. Gradient w.r.t input x_seq
        let delta_x_q = self.w_q.t()?.broadcast_matmul(&delta_q)?;
        let delta_x_k = self.w_k.t()?.broadcast_matmul(&delta_k)?;
        let delta_x_v = self.w_v.t()?.broadcast_matmul(&delta_v)?;

        let delta_x_seq = delta_x_q.add(&delta_x_k)?.add(&delta_x_v)?;
        let delta_x = delta_x_seq.reshape((b, c, h, w))?;

        Ok((delta_x, d_wq, d_wk, d_wv))
    }
}
