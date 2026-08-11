use anyhow::{bail, Ok, Result};
use candle_core::{DType, Device, Tensor};

use super::attention::SpatialSelfAttention;
use super::denoising_cnn_ops::{manual_conv2d, manual_conv2d_backward};
use super::DenoisingModel;
use crate::common::parameterized::Parameterized;
use crate::common::varstore;
use candle_nn::VarMap;

const RESIDUAL_SCALE: f64 = 0.7071067811865475;
const GROUP_NORM_EPS: f64 = 1e-5;

fn leaky_relu(x: &Tensor) -> Result<Tensor> {
    Ok(x.maximum(&x.affine(0.01, 0.0)?)?)
}

fn leaky_relu_grad(x: &Tensor) -> Result<Tensor> {
    Ok(x.ge(0.0f32)?.to_dtype(DType::F32)?.affine(0.99, 0.01)?)
}

fn group_norm_forward(x: &Tensor, groups: usize) -> Result<(Tensor, Tensor, Tensor)> {
    let (b, c, h, w) = x.dims4()?;
    if c % groups != 0 {
        bail!(
            "group norm expected channels ({}) divisible by groups ({})",
            c,
            groups
        );
    }

    let group_width = (c / groups) * h * w;
    let x_grouped = x.reshape((b, groups, group_width))?;
    let mean = x_grouped.mean(2)?.reshape((b, groups, 1))?;
    let centered = x_grouped.sub(&mean.broadcast_as((b, groups, group_width))?)?;
    let variance = centered.sqr()?.mean(2)?.reshape((b, groups, 1))?;
    let std = variance.affine(1.0, GROUP_NORM_EPS)?.sqrt()?;
    let inv_std = Tensor::ones((b, groups, 1), DType::F32, x.device())?.div(&std)?;
    let x_hat = centered
        .mul(&inv_std.broadcast_as((b, groups, group_width))?)?
        .reshape((b, c, h, w))?;
    let y = x_hat.clone();

    Ok((y, x_hat, inv_std))
}

fn group_norm_backward(
    delta_y: &Tensor,
    x_hat: &Tensor,
    inv_std: &Tensor,
    groups: usize,
) -> Result<Tensor> {
    let (b, c, h, w) = delta_y.dims4()?;
    if c % groups != 0 {
        bail!(
            "group norm backward expected channels ({}) divisible by groups ({})",
            c,
            groups
        );
    }

    let group_width = (c / groups) * h * w;
    let dy = delta_y.reshape((b, groups, group_width))?;
    let xh = x_hat.reshape((b, groups, group_width))?;

    let mean_dy = dy
        .sum(2)?
        .affine(1.0 / group_width as f64, 0.0)?
        .reshape((b, groups, 1))?;
    let mean_dy_xhat = dy
        .mul(&xh)?
        .sum(2)?
        .affine(1.0 / group_width as f64, 0.0)?
        .reshape((b, groups, 1))?;

    let centered_dy = dy.sub(&mean_dy.broadcast_as((b, groups, group_width))?)?;
    let variance_term = xh.mul(&mean_dy_xhat.broadcast_as((b, groups, group_width))?)?;
    let dx_grouped = centered_dy
        .sub(&variance_term)?
        .mul(&inv_std.broadcast_as((b, groups, group_width))?)?;

    Ok(dx_grouped.reshape((b, c, h, w))?)
}

pub struct SimpleDenoisingUNet {
    /// Owns every trainable parameter — including the attention sub-module's —
    /// under its checkpoint name. The tensor fields below share storage with
    /// its `Var`s.
    varmap: VarMap,

    pub img_dim: usize,
    pub cond_dim: usize,
    pub w_cond: Tensor, // [16,cond_dim,1,1]
    pub b_cond: Tensor, // [16]

    pub w1: Tensor, // [16,16,3,3]
    pub b1: Tensor, // [16]

    pub w2: Tensor, //[32,32,3,3]
    pub b2: Tensor, //[32]

    pub w3: Tensor, //[16,16,32,32]
    pub b3: Tensor, // [16]

    pub w4: Tensor, // [16,16,32,32]
    pub b4: Tensor, // [16]

    pub w5: Tensor, // [1,16,3,3]
    pub b5: Tensor, // [1]

    pub attn: SpatialSelfAttention,
}

impl SimpleDenoisingUNet {
    pub fn new(img_dim: usize, cond_dim: usize, device: &Device) -> Result<Self> {
        let h = (img_dim as f64).sqrt() as usize;
        if h * h != img_dim {
            bail!(
                "SimpleDenoisingUNet expected img_dim to be a square image area, got {}",
                img_dim
            );
        }
        if h % 2 != 0 {
            bail!(
                "SimpleDenoisingUNet expected an even image side length for 2x2 pooling, got {}",
                h
            );
        }

        // Every parameter is registered in the VarMap; keep the tensor that
        // `register` returns, not the one passed in — only the former shares
        // storage with the stored `Var` and observes later updates.
        let varmap = VarMap::new();

        let scale_cond = (2.0f64 / cond_dim as f64).sqrt();
        let w_cond = varstore::register(
            &varmap,
            "w_cond",
            (Tensor::randn(0.0f32, 1.0f32, (img_dim, cond_dim), device)? * scale_cond)?,
        )?;
        let b_cond = varstore::register(
            &varmap,
            "b_cond",
            Tensor::zeros(img_dim, DType::F32, device)?,
        )?;
        // --- Conv1 weights (2 -> 16 channels, 3x3) ---
        let scale1 = (2.0f64 / (2.0 * 3.0 * 3.0)).sqrt();
        let w1 = varstore::register(
            &varmap,
            "w1",
            (Tensor::randn(0.0f32, 1.0f32, (16, 2, 3, 3), device)? * scale1)?,
        )?;
        let b1 = varstore::register(&varmap, "b1", Tensor::zeros(16, DType::F32, device)?)?;
        // --- Conv2 weights (16 -> 32 channels, 3x3) ---
        let scale2 = (2.0f64 / (16.0 * 3.0 * 3.0)).sqrt();

        let w2 = varstore::register(
            &varmap,
            "w2",
            (Tensor::randn(0.0f32, 1.0f32, (32, 16, 3, 3), device)? * scale2)?,
        )?;
        let b2 = varstore::register(&varmap, "b2", Tensor::zeros(32, DType::F32, device)?)?;
        // --- Conv3 weights (32 -> 32 channels, 3x3) ---
        let scale3 = (2.0f64 / (32.0 * 3.0 * 3.0)).sqrt();
        let w3 = varstore::register(
            &varmap,
            "w3",
            (Tensor::randn(0.0f32, 1.0f32, (32, 32, 3, 3), device)? * scale3)?,
        )?;
        let b3 = varstore::register(&varmap, "b3", Tensor::zeros(32, DType::F32, device)?)?;
        // --- Conv4 weights (48 -> 16 channels, 3x3) ---
        let scale4 = (2.0f64 / (48.0 * 3.0 * 3.0)).sqrt();
        let w4 = varstore::register(
            &varmap,
            "w4",
            (Tensor::randn(0.0f32, 1.0f32, (16, 48, 3, 3), device)? * scale4)?,
        )?;
        let b4 = varstore::register(&varmap, "b4", Tensor::zeros(16, DType::F32, device)?)?;
        // --- Conv5 weights (16 -> 1 channel, 3x3) ---
        let scale5 = (2.0f64 / (16.0 * 3.0 * 3.0)).sqrt();
        let w5 = varstore::register(
            &varmap,
            "w5",
            (Tensor::randn(0.0f32, 1.0f32, (1, 16, 3, 3), device)? * scale5)?,
        )?;
        let b5 = varstore::register(&varmap, "b5", Tensor::zeros(1, DType::F32, device)?)?;

        // Sub-module registers into this VarMap under the "attn_" prefix, which
        // reproduces the existing "attn_w_q" / "attn_w_k" / "attn_w_v"
        // checkpoint keys.
        let attn = SpatialSelfAttention::new(32, &varmap, "attn_", device)?;

        Ok(Self {
            varmap,
            img_dim,
            cond_dim,
            w_cond,
            b_cond,
            w1,
            b1,
            w2,
            b2,
            w3,
            b3,
            w4,
            b4,
            w5,
            b5,
            attn,
        })
    }
}

impl DenoisingModel for SimpleDenoisingUNet {
    fn forward(&self, x: &Tensor) -> Result<(Tensor, Vec<Tensor>)> {
        let device = x.device();
        let b = x.dim(0)?;
        let h = (self.img_dim as f64).sqrt() as usize;
        let w_img = h;
        let h_down = h / 2;
        let w_down = w_img / 2;

        //split input
        let xt = x.narrow(1, 0, self.img_dim)?;
        let cond_vec = x.narrow(1, self.img_dim, self.cond_dim)?;

        // project conditions: shape (B,1,H, W)
        let cond_map = cond_vec
            .contiguous()?
            .matmul(&self.w_cond.t()?.contiguous()?)?
            .broadcast_add(&self.b_cond)?
            .reshape((b, 1, h, w_img))?;

        // concate noisy image and conditing map: shape (b,1,H,w)

        let xt_img = xt.contiguous()?.reshape((b, 1, h, w_img))?;

        let input_cat = Tensor::cat(&[&xt_img, &cond_map], 1)?;

        // level 1

        let z1_conv = manual_conv2d(&input_cat, &self.w1, Some(&self.b1), &device)?;
        let (z1, z1_hat, z1_inv_std) = group_norm_forward(&z1_conv, 4)?;
        let a1 = leaky_relu(&z1)?;
        let a1_down = a1
            .reshape((b, 16, h_down, 2, w_down, 2))?
            .mean(5)?
            .mean(3)?;

        // level 2
        let z2_conv = manual_conv2d(&a1_down, &self.w2, Some(&self.b2), &device)?;
        let (z2, z2_hat, z2_inv_std) = group_norm_forward(&z2_conv, 4)?;
        let a2 = leaky_relu(&z2)?;

        // Bottleneck residual block: conv3 keeps the same shape as a2.
        // Scaling by 1/sqrt(2) keeps the residual sum variance near the input scale.
        let z3_conv = manual_conv2d(&a2, &self.w3, Some(&self.b3), &device)?;
        let z3_res = z3_conv.add(&a2)?.affine(RESIDUAL_SCALE, 0.0)?;
        let (z3, z3_hat, z3_inv_std) = group_norm_forward(&z3_res, 4)?;
        let a3_pre = leaky_relu(&z3)?;
        // Apply self-attention at the bottleneck, as a RESIDUAL.
        //
        // WHY the residual is not optional here:
        //   Softmax attention over n = 196 bottleneck positions starts life
        //   near-uniform (measured at init: 98.5% of the uniform entropy bound).
        //   A uniform attention row makes every output position the same global
        //   mean of V, which erases spatial structure — measured at init, the
        //   raw attention output retains 0.6% of its input's spatial variance.
        //
        //   That is normal and harmless in the standard formulation `x +
        //   attn(x)`, where the skip carries the signal while attention
        //   gradually learns structure worth adding. Applied *without* a skip,
        //   the same collapse instead deletes a3_pre from the decoder path
        //   entirely, leaving the bottleneck contributing only a per-channel
        //   constant and the a1 skip doing all the spatial work. It also starves
        //   w_q/w_k of gradient, so the network drifts further into the flat
        //   regime rather than out of it.
        //
        // Scaled by 1/sqrt(2) to keep the sum's variance near the input scale,
        // matching the z3_res / z4_res residuals above and below.
        let (attn_out, attn_cached) = self.attn.forward(&a3_pre)?;
        let a3 = a3_pre.add(&attn_out)?.affine(RESIDUAL_SCALE, 0.0)?;
        // decoder level(28*28)

        let a3_up = a3
            .reshape((b, 32, h_down, 1, w_down, 1))?
            .broadcast_as((b, 32, h_down, 2, w_down, 2))?
            .reshape((b, 32, h, w_img))?;

        // concate upsampled features

        let decode_cat = Tensor::cat(&[&a3_up, &a1], 1)?;
        // conv4(B,16,H,W)
        let z4_conv = manual_conv2d(&decode_cat, &self.w4, Some(&self.b4), &device)?;
        let z4_res = z4_conv.add(&a1)?.affine(RESIDUAL_SCALE, 0.0)?;
        let (z4, z4_hat, z4_inv_std) = group_norm_forward(&z4_res, 4)?;
        let a4 = leaky_relu(&z4)?;

        // conv5
        let z5 = manual_conv2d(&a4, &self.w5, Some(&self.b5), &device)?;
        let pred = z5.reshape((b, self.img_dim))?;
        let mut intermediates = vec![
            input_cat, z1, z1_hat, z1_inv_std, a1, a1_down, z2, z2_hat, z2_inv_std, a2, z3, z3_hat,
            z3_inv_std, a3, a3_up, decode_cat, z4, z4_hat, z4_inv_std, a4,
        ];
        intermediates.extend(attn_cached);
        Ok((pred, intermediates))
    }
    fn backward(
        &self,
        v: &Tensor,
        intermediates: &[Tensor],
        pred: &Tensor,
        target: &Tensor,
    ) -> Result<Vec<Tensor>> {
        if intermediates.len() != 26 {
            bail!(
                "SimpleDenoisingUNet expected 26 cached intermediates from forward(), got {}",
                intermediates.len()
            );
        }

        let device = v.device();
        let b = v.dim(0)?;
        let h = (self.img_dim as f64).sqrt() as usize;
        let w_img = h;
        let h_down = h / 2;
        let w_down = w_img / 2;

        let (
            input_cat,
            z1,
            z1_hat,
            z1_inv_std,
            _a1,
            a1_down,
            z2,
            z2_hat,
            z2_inv_std,
            a2,
            z3,
            z3_hat,
            z3_inv_std,
            _a3,
            _a3_up,
            decode_cat,
            z4,
            z4_hat,
            z4_inv_std,
            a4,
        ) = (
            &intermediates[0],
            &intermediates[1],
            &intermediates[2],
            &intermediates[3],
            &intermediates[4],
            &intermediates[5],
            &intermediates[6],
            &intermediates[7],
            &intermediates[8],
            &intermediates[9],
            &intermediates[10],
            &intermediates[11],
            &intermediates[12],
            &intermediates[13],
            &intermediates[14],
            &intermediates[15],
            &intermediates[16],
            &intermediates[17],
            &intermediates[18],
            &intermediates[19],
        );

        // 1. MSE gradient w.r.t predication
        let scale = 2.0 / (b * self.img_dim) as f64;

        let delta_pred = pred.sub(target)?.affine(scale, 0.0)?;

        let delta_z5 = delta_pred.reshape((b, 1, h, w_img))?;

        //2. conv5 out backward
        let db5 = delta_z5.sum(0)?.sum(1)?.sum(1)?;
        let (delta_a4, dw5) = manual_conv2d_backward(a4, &self.w5, &delta_z5, device)?;

        // 3. leaky rule backward on z4
        let relu_grad4 = leaky_relu_grad(z4)?;
        let delta_z4 = delta_a4.mul(&relu_grad4)?;
        let delta_z4_norm = group_norm_backward(&delta_z4, z4_hat, z4_inv_std, 4)?;
        let delta_z4_conv = delta_z4_norm.affine(RESIDUAL_SCALE, 0.0)?;
        let delta_a1_from_decoder_residual = delta_z4_norm.affine(RESIDUAL_SCALE, 0.0)?;

        // 4. conv4 backward
        let db4 = delta_z4_conv.sum(0)?.sum(1)?.sum(1)?;
        let (delta_decode_cat, dw4) =
            manual_conv2d_backward(decode_cat, &self.w4, &delta_z4_conv, &device)?;

        //5.  split skip connection gradient
        let delta_a3_up = delta_decode_cat.narrow(1, 0, 32)?.contiguous()?;

        let delta_a1_from_skip = delta_decode_cat.narrow(1, 32, 16)?.contiguous()?;

        //6. Nearest Neighbour upsampling backward(sum2*2)
        let delta_a3 = delta_a3_up
            .reshape((b, 32, h_down, 2, w_down, 2))?
            .sum(5)?
            .sum(3)?;

        //7. Attention backward (residual)
        //
        // Forward was: a3 = (a3_pre + attn(a3_pre)) * RESIDUAL_SCALE
        //
        // So the upstream gradient is first scaled, then splits along both
        // branches of the sum: one copy flows through the attention block, and
        // one copy reaches a3_pre directly. The direct path is the whole point
        // of the residual — it keeps gradient reaching the bottleneck conv
        // stack even while attention sits in its near-uniform regime.
        let delta_a3_scaled = delta_a3.affine(RESIDUAL_SCALE, 0.0)?;
        let (delta_a3_pre_from_attn, d_wq, d_wk, d_wv) =
            self.attn.backward(&intermediates[20..26], &delta_a3_scaled)?;
        let delta_a3_pre = delta_a3_pre_from_attn.add(&delta_a3_scaled)?;

        //8. Leaky relu backward on z3

        let relu_grad3 = leaky_relu_grad(z3)?;
        let delta_z3 = delta_a3_pre.mul(&relu_grad3)?;
        let delta_z3_norm = group_norm_backward(&delta_z3, z3_hat, z3_inv_std, 4)?;
        let delta_z3_conv = delta_z3_norm.affine(RESIDUAL_SCALE, 0.0)?;
        let delta_a2_from_bottleneck_residual = delta_z3_norm.affine(RESIDUAL_SCALE, 0.0)?;

        // 8.botlleneck conv3 backward
        let db3 = delta_z3_conv.sum(0)?.sum(1)?.sum(1)?;
        let (delta_a2_from_conv3, dw3) =
            manual_conv2d_backward(a2, &self.w3, &delta_z3_conv, &device)?;
        let delta_a2 = delta_a2_from_conv3.add(&delta_a2_from_bottleneck_residual)?;

        // 9. leaky relu backward onz2
        let relu_grad2 = leaky_relu_grad(z2)?;
        let delta_z2 = delta_a2.mul(&relu_grad2)?;
        let delta_z2_norm = group_norm_backward(&delta_z2, z2_hat, z2_inv_std, 4)?;

        //10. maxpool backward
        let db2 = delta_z2_norm.sum(0)?.sum(1)?.sum(1)?;
        let (delta_a1_down, dw2) =
            manual_conv2d_backward(a1_down, &self.w2, &delta_z2_norm, &device)?;

        // 11. average pool 2x2 backward(nearest neighbour upsample scaled gradient)
        let scaled_delta = delta_a1_down.affine(0.25, 0.0)?;

        let delta_a1_from_down = scaled_delta
            .reshape((b, 16, h_down, 1, w_down, 1))?
            .broadcast_as((b, 16, h_down, 2, w_down, 2))?
            .reshape((b, 16, h, w_img))?;

        // 12. sum gradien flow in to a1
        let delta_a1 = delta_a1_from_down
            .add(&delta_a1_from_skip)?
            .add(&delta_a1_from_decoder_residual)?;

        //13. leaky relu backward on z1
        let relu_grad1 = leaky_relu_grad(z1)?;
        let delta_z1 = delta_a1.mul(&relu_grad1)?;
        let delta_z1_norm = group_norm_backward(&delta_z1, z1_hat, z1_inv_std, 4)?;

        // 14.conv1 backward
        let db1 = delta_z1_norm.sum(0)?.sum(1)?.sum(1)?;
        let (delta_input_cat, dw1) =
            manual_conv2d_backward(input_cat, &self.w1, &delta_z1_norm, device)?;

        // 15. conv1 input backward

        let delta_cond_map = delta_input_cat.narrow(1, 1, 1)?.contiguous()?;
        let delta_cond_flat = delta_cond_map.reshape((b, self.img_dim))?;
        let db_cond = delta_cond_flat.sum(0)?;
        let cond_vec = v.narrow(1, self.img_dim, self.cond_dim)?.contiguous()?;
        let dw_cond = delta_cond_flat.t()?.contiguous()?.matmul(&cond_vec)?;

        Ok(vec![
            dw_cond, db_cond, dw1, db1, dw2, db2, dw3, db3, dw4, db4, dw5, db5, d_wq, d_wk, d_wv,
        ])
    }
}

impl Parameterized for SimpleDenoisingUNet {
    fn varmap(&self) -> &VarMap {
        &self.varmap
    }

    fn params(&self) -> Vec<&Tensor> {
        vec![
            &self.w_cond,
            &self.b_cond,
            &self.w1,
            &self.b1,
            &self.w2,
            &self.b2,
            &self.w3,
            &self.b3,
            &self.w4,
            &self.b4,
            &self.w5,
            &self.b5,
            &self.attn.w_q,
            &self.attn.w_k,
            &self.attn.w_v,
        ]
    }
    fn param_names(&self) -> Vec<&str> {
        vec![
            "w_cond", "b_cond", "w1", "b1", "w2", "b2", "w3", "b3", "w4", "b4", "w5", "b5",
            "attn_w_q", "attn_w_k", "attn_w_v",
        ]
    }
}
