use anyhow::Result;
use candle_core::{Device, Tensor};
use candle_nn::VarMap;

use crate::common::varstore;

pub struct SpatialSelfAttention {
    pub w_qkv: Tensor,
}

impl SpatialSelfAttention {
    pub fn new(channels: usize, varmap: &VarMap, prefix: &str, device: &Device) -> Result<Self> {
        let scale = (1.0f64 / channels as f64).sqrt();
        let w_qkv = varstore::register(
            varmap,
            &format!("{prefix}w_qkv"),
            (Tensor::randn(0.0f32, 1.0f32, (3 * channels, channels), device)? * scale)?,
        )?;
        Ok(Self { w_qkv })
    }

    pub fn forward(&self, x: &Tensor) -> Result<(Tensor, Vec<Tensor>)> {
        let (b, c, h, w) = x.dims4()?;
        let n = h * w;

        // 1. Reshape to sequence format: [B, C, N]
        let x_seq = x.reshape((b, c, n))?.contiguous()?;

        // 2. Project Q, K, V together
        let qkv = self.w_qkv.broadcast_matmul(&x_seq)?; // [B, 3C, N]
        let q = qkv.narrow(1, 0, c)?;
        let k = qkv.narrow(1, c, c)?;
        let v = qkv.narrow(1, 2 * c, c)?;

        // 3. Compute attention scores: S = (Q^T @ K) / sqrt(C)
        let scale = 1.0 / (c as f64).sqrt();
        let scores = q
            .transpose(1, 2)?
            .broadcast_matmul(&k)?
            .affine(scale, 0.0)?;

        // 4. Softmax over key dimension (dim 2)
        let attn_weights = candle_nn::ops::softmax(&scores, 2)?;

        // 5. Output: O = V @ A^T
        let output_seq = v.broadcast_matmul(&attn_weights.transpose(1, 2)?)?;
        let output = output_seq.reshape((b, c, h, w))?;

        let cached = vec![x_seq, q, k, v, attn_weights];
        Ok((output, cached))
    }

    pub fn backward(
        &self,
        intermediates: &[Tensor],
        delta_y: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let b = delta_y.dim(0)?;
        let c = self.w_qkv.dim(1)?;
        let h = delta_y.dim(2)?;
        let w = delta_y.dim(3)?;
        let n = h * w;

        let x_seq = &intermediates[0];
        let q = &intermediates[1];
        let k = &intermediates[2];
        let v = &intermediates[3];
        let attn_weights = &intermediates[4];

        let delta_y_seq = delta_y.reshape((b, c, n))?;

        // 1. Gradients w.r.t V and A
        //    delta_a = dY^T @ V, equivalent to (V^T @ dY)^T from the old code
        //    but avoids an extra transpose allocation.
        let delta_v = delta_y_seq.broadcast_matmul(attn_weights)?;
        let delta_a = delta_y_seq.transpose(1, 2)?.broadcast_matmul(v)?;

        // 2. Softmax backward: dS = A * (dA - sum(dA * A, keepdim=2))
        let sum_term = delta_a.mul(attn_weights)?.sum_keepdim(2)?;
        let delta_s = attn_weights.mul(&delta_a.broadcast_sub(&sum_term)?)?;

        // 3. Gradients w.r.t Q and K
        let scale = 1.0 / (c as f64).sqrt();
        let delta_q = k
            .broadcast_matmul(&delta_s.transpose(1, 2)?)?
            .affine(scale, 0.0)?;
        let delta_k = q.broadcast_matmul(&delta_s)?.affine(scale, 0.0)?;

        // 4. Gradients w.r.t fused projection weights w_qkv via 2D GEMM
        let delta_qkv = Tensor::cat(&[&delta_q, &delta_k, &delta_v], 1)?; // [B, 3C, N]
        let delta_qkv_2d = delta_qkv
            .transpose(0, 1)?
            .contiguous()?
            .reshape((3 * c, b * n))?;
        let x_seq_2d = x_seq
            .transpose(0, 1)?
            .contiguous()?
            .reshape((c, b * n))?;
        let d_wqkv = delta_qkv_2d.matmul(&x_seq_2d.t()?)?; // [3C, C]

        // 5. Gradient w.r.t input x_seq
        let delta_x_seq = self.w_qkv.t()?.broadcast_matmul(&delta_qkv)?; // [C, 3C] @ [B, 3C, N] -> [B, C, N]
        let delta_x = delta_x_seq.reshape((b, c, h, w))?;

        Ok((delta_x, d_wqkv))
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
    fn attention_wqkv_backward_matches_finite_difference() -> Result<()> {
        let device = &Device::Cpu;
        let input = Tensor::new(&[[[[0.2f32, -0.4]], [[0.7, 0.1]]]], device)?;
        let delta = Tensor::new(&[[[[0.3f32, -0.2]], [[-0.5, 0.4]]]], device)?;
        let varmap = VarMap::new();
        let attention = SpatialSelfAttention::new(2, &varmap, "attn_", device)?;
        let (_, cached) = attention.forward(&input)?;
        let (_, analytic_wqkv) = attention.backward(&cached, &delta)?;
        let analytic = analytic_wqkv.flatten_all()?.to_vec1::<f32>()?[0];

        // Central finite difference approximates dL/dw with
        // (L(w + eps) - L(w - eps)) / (2 * eps). Comparing one representative
        // element is enough to catch the Q/K transpose error in this tiny case.
        let epsilon = 1e-3f32;
        let original = attention.w_qkv.flatten_all()?.to_vec1::<f32>()?;
        let mut plus_values = original.clone();
        let mut minus_values = original;
        plus_values[0] += epsilon;
        minus_values[0] -= epsilon;

        let plus = SpatialSelfAttention {
            w_qkv: Tensor::new(plus_values.as_slice(), device)?.reshape((6, 2))?,
        };
        let minus = SpatialSelfAttention {
            w_qkv: Tensor::new(minus_values.as_slice(), device)?.reshape((6, 2))?,
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
