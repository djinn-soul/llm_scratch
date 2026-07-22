use anyhow::Result;
use candle_core::{Device, Tensor};
use candle_nn::VarMap;

use crate::common::varstore;

pub struct SpatialSelfAttention {
    pub w_q: Tensor,
    pub w_k: Tensor,
    pub w_v: Tensor,
}

impl SpatialSelfAttention {
    /// Build the attention block, registering its weights into `varmap`.
    ///
    /// This is a sub-module: it does not own a VarMap of its own, it registers
    /// into its parent's under `{prefix}w_q`, `{prefix}w_k`, `{prefix}w_v`. The
    /// prefix is what keeps names unique if a model ever holds more than one
    /// attention block, and it mirrors the usual `attn.w_q` checkpoint layout.
    ///
    /// The parent therefore stays the single owner of the parameter list, so
    /// its `params()` / `param_names()` ordering still covers these weights.
    pub fn new(channels: usize, varmap: &VarMap, prefix: &str, device: &Device) -> Result<Self> {
        let scale = (1.0f64 / channels as f64).sqrt();
        let w_q = varstore::register(
            varmap,
            &format!("{prefix}w_q"),
            (Tensor::randn(0.0f32, 1.0f32, (channels, channels), device)? * scale)?,
        )?;
        let w_k = varstore::register(
            varmap,
            &format!("{prefix}w_k"),
            (Tensor::randn(0.0f32, 1.0f32, (channels, channels), device)? * scale)?,
        )?;
        let w_v = varstore::register(
            varmap,
            &format!("{prefix}w_v"),
            (Tensor::randn(0.0f32, 1.0f32, (channels, channels), device)? * scale)?,
        )?;
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

        // 3. Compute attention scores: S = (Q^T @ K) / sqrt(C)
        let scale = 1.0 / (c as f64).sqrt();
        let scores = q
            .transpose(1, 2)?
            .broadcast_matmul(&k)?
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
        let delta_q = k
            .broadcast_matmul(&delta_s.transpose(1, 2)?)?
            .affine(scale, 0.0)?;
        let delta_k = q.broadcast_matmul(&delta_s)?.affine(scale, 0.0)?;

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

#[cfg(test)]
mod tests {
    use super::*;

    fn attention_loss(
        attention: &SpatialSelfAttention,
        input: &Tensor,
        delta: &Tensor,
    ) -> Result<f32> {
        let (output, _) = attention.forward(input)?;
        Ok(output.mul(delta)?.sum_all()?.to_scalar::<f32>()?)
    }

    #[test]
    fn attention_wq_backward_matches_finite_difference() -> Result<()> {
        let device = &Device::Cpu;
        let input = Tensor::new(&[[[[0.2f32, -0.4]], [[0.7, 0.1]]]], device)?;
        let delta = Tensor::new(&[[[[0.3f32, -0.2]], [[-0.5, 0.4]]]], device)?;
        let varmap = VarMap::new();
        let attention = SpatialSelfAttention::new(2, &varmap, "attn_", device)?;
        let (_, cached) = attention.forward(&input)?;
        let (_, analytic_wq, _, _) = attention.backward(&cached, &delta)?;
        let analytic = analytic_wq.flatten_all()?.to_vec1::<f32>()?[0];

        // Central finite difference approximates dL/dw with
        // (L(w + eps) - L(w - eps)) / (2 * eps). Comparing one representative
        // element is enough to catch the Q/K transpose error in this tiny case.
        let epsilon = 1e-3f32;
        let original = attention.w_q.flatten_all()?.to_vec1::<f32>()?;
        let mut plus_values = original.clone();
        let mut minus_values = original;
        plus_values[0] += epsilon;
        minus_values[0] -= epsilon;

        let plus = SpatialSelfAttention {
            w_q: Tensor::new(plus_values.as_slice(), device)?.reshape((2, 2))?,
            w_k: attention.w_k.clone(),
            w_v: attention.w_v.clone(),
        };
        let minus = SpatialSelfAttention {
            w_q: Tensor::new(minus_values.as_slice(), device)?.reshape((2, 2))?,
            w_k: attention.w_k.clone(),
            w_v: attention.w_v.clone(),
        };
        let numeric = (attention_loss(&plus, &input, &delta)?
            - attention_loss(&minus, &input, &delta)?)
            / (2.0 * epsilon);

        assert!(
            (analytic - numeric).abs() < 2e-3,
            "analytic={analytic}, numeric={numeric}"
        );
        Ok(())
    }
}
