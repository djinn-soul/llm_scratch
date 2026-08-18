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

fn group_norm_forward(x: &Tensor, groups: usize) -> Result<(Tensor, Tensor)> {
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
    let mean = x_grouped.mean_keepdim(2)?;
    let centered = x_grouped.broadcast_sub(&mean)?;
    let variance = centered.sqr()?.mean_keepdim(2)?;
    let std = variance.affine(1.0, GROUP_NORM_EPS)?.sqrt()?;
    let inv_std = std.recip()?;
    let x_hat = centered.broadcast_mul(&inv_std)?.reshape((b, c, h, w))?;
    Ok((x_hat, inv_std))
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

    let mean_dy = dy.mean_keepdim(2)?;
    let mean_dy_xhat = dy.mul(&xh)?.mean_keepdim(2)?;

    let centered_dy = dy.broadcast_sub(&mean_dy)?;
    let variance_term = xh.broadcast_mul(&mean_dy_xhat)?;
    let dx_grouped = centered_dy.sub(&variance_term)?.broadcast_mul(&inv_std)?;

    Ok(dx_grouped.reshape((b, c, h, w))?)
}

/// Applies Adaptive Group Normalization (Forward)
///
/// Shapes:
// =============================================================================
// ADAPTIVE GROUP NORMALIZATION (AdaGN) HELPERS
// =============================================================================
//
// In standard GroupNorm, feature activations are normalized per-channel group.
// AdaGN extends this by dynamically scaling (gain γ) and shifting (bias β) the
// normalized features using conditioning information (timestep t + class label c):
//
//   AdaGN(x, c) = GroupNorm(x) ⊙ (1 + γ(c)) + β(c)
//
// ── WHY (1 + γ) INSTEAD OF γ? ──
// By initializing the projection layers to zeros (w_ada = 0, b_ada = 0), at step 0:
//   γ = 0, β = 0  ==>  y = GroupNorm(x) ⊙ (1 + 0) + 0 = GroupNorm(x)
// This "Identity Initialization" guarantees training starts smoothly as standard
// GroupNorm without exploding activations or destroying early gradients.

/// Applies Adaptive Group Normalization (Forward pass).
///
/// Modulates normalized activations x̂ with per-channel scale (γ) and shift (β).
///
/// Mathematical Formulation:
///   1. x̂ = GroupNorm(x) = (x - μ_g) / √(σ_g² + ε)
///   2. y = x̂ ⊙ (1 + γ) + β
///
/// Shapes:
///   - `x`:       `[B, C, H, W]`
///   - `gamma`:   `[B, C]` -> broadcasts across `[B, C, 1, 1]`
///   - `beta`:    `[B, C]` -> broadcasts across `[B, C, 1, 1]`
///   - `groups`:  number of channel groups G (e.g. 4 groups of 4 channels)
///
/// Returns:
///   - `y`:       `[B, C, H, W]` modulated output tensor
///   - `x_hat`:   `[B, C, H, W]` normalized activations (saved for backward)
///   - `inv_std`: `[B, G, 1]` group inverse standard deviations (saved for backward)
pub fn adagn_forward(
    x: &Tensor,
    gamma: &Tensor,
    beta: &Tensor,
    groups: usize,
) -> Result<(Tensor, Tensor, Tensor)> {
    let (b, c, _h, _w) = x.dims4()?;

    // Step 1: Standard group normalization: x̂ ~ N(0, 1) per group
    let (x_hat, inv_std) = group_norm_forward(x, groups)?;

    // Step 2: Reshape condition vectors (B, C) to (B, C, 1, 1) for spatial broadcast
    let gamma_b = gamma.reshape((b, c, 1, 1))?;
    let beta_b = beta.reshape((b, c, 1, 1))?;

    // Step 3: Compute modulation factor (1 + γ)
    let one_plus_gamma = gamma_b.affine(1.0, 1.0)?;

    // Step 4: Modulate activations: y = x̂ * (1 + γ) + β
    let y = x_hat.broadcast_mul(&one_plus_gamma)?.broadcast_add(&beta_b)?;

    Ok((y, x_hat, inv_std))
}

/// Backward pass for Adaptive Group Normalization.
///
/// Applies analytical Chain Rule calculus to compute gradients w.r.t.:
///   - Shift β:      ∂L/∂β = ∑_{H, W} ∂L/∂y
///   - Scale γ:      ∂L/∂γ = ∑_{H, W} (∂L/∂y ⊙ x̂)
///   - Normalized x̂: ∂L/∂x̂ = ∂L/∂y ⊙ (1 + γ)
///   - Raw input x:  ∂L/∂x = GroupNormBackward(∂L/∂x̂)
///
/// Arguments:
///   - `delta_y`:  `[B, C, H, W]` upstream gradient flowing into this layer
///   - `x_hat`:    `[B, C, H, W]` cached normalized activations from forward()
///   - `inv_std`:  `[B, G, 1]` cached inverse std from forward()
///   - `gamma`:    `[B, C]` scale parameter used during forward()
///   - `groups`:   channel groups G
///
/// Returns:
///   - `(delta_x, delta_gamma, delta_beta)`
pub fn adagn_backward(
    delta_y: &Tensor,
    x_hat: &Tensor,
    inv_std: &Tensor,
    gamma: &Tensor,
    groups: usize,
) -> Result<(Tensor, Tensor, Tensor)> {
    let (b, c, _, _) = delta_y.dims4()?;

    // 1. ∂L/∂β: sum gradient across spatial dimensions H (axis 2) and W (axis 3) -> [B, C]
    let delta_beta = delta_y.sum(3)?.sum(2)?;

    // 2. ∂L/∂γ: sum (delta_y * x_hat) across spatial dimensions -> [B, C]
    let delta_gamma = delta_y.mul(x_hat)?.sum(3)?.sum(2)?;

    // 3. ∂L/∂x̂: scale upstream gradient by (1 + γ) -> [B, C, H, W]
    let gamma_b = gamma.reshape((b, c, 1, 1))?;
    let one_plus_gamma = gamma_b.affine(1.0, 1.0)?;
    let delta_xhat = delta_y.broadcast_mul(&one_plus_gamma)?;

    // 4. ∂L/∂x: propagate gradient through the standard GroupNorm backprop equation
    let delta_x = group_norm_backward(&delta_xhat, x_hat, inv_std, groups)?;

    Ok((delta_x, delta_gamma, delta_beta))
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

        // Sub-module registers into this VarMap under the "attn_" prefix,
        // producing the "attn_w_qkv" checkpoint key.
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
        let (z1_hat, z1_inv_std) = group_norm_forward(&z1_conv, 4)?;
        let a1 = leaky_relu(&z1_hat)?;
        let a1_down = a1
            .reshape((b, 16, h_down, 2, w_down, 2))?
            .mean(5)?
            .mean(3)?;

        // level 2
        let z2_conv = manual_conv2d(&a1_down, &self.w2, Some(&self.b2), &device)?;
        let (z2_hat, z2_inv_std) = group_norm_forward(&z2_conv, 4)?;
        let a2 = leaky_relu(&z2_hat)?;

        // Bottleneck residual block: conv3 keeps the same shape as a2.
        // Scaling by 1/sqrt(2) keeps the residual sum variance near the input scale.
        let z3_conv = manual_conv2d(&a2, &self.w3, Some(&self.b3), &device)?;
        let z3_res = z3_conv.add(&a2)?.affine(RESIDUAL_SCALE, 0.0)?;
        let (z3_hat, z3_inv_std) = group_norm_forward(&z3_res, 4)?;
        let a3_pre = leaky_relu(&z3_hat)?;
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
        //   w_qkv of gradient, so the network drifts further into the flat
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
        let (z4_hat, z4_inv_std) = group_norm_forward(&z4_res, 4)?;
        let a4 = leaky_relu(&z4_hat)?;

        // conv5
        let z5 = manual_conv2d(&a4, &self.w5, Some(&self.b5), &device)?;
        let pred = z5.reshape((b, self.img_dim))?;
        let mut intermediates = vec![
            input_cat, z1_hat, z1_inv_std, a1, a1_down, z2_hat, z2_inv_std, a2, z3_hat, z3_inv_std,
            a3, a3_up, decode_cat, z4_hat, z4_inv_std, a4,
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
        if intermediates.len() != 21 {
            bail!(
                "SimpleDenoisingUNet expected 21 cached intermediates from forward(), got {}",
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
            z1_hat,
            z1_inv_std,
            _a1,
            a1_down,
            z2_hat,
            z2_inv_std,
            a2,
            z3_hat,
            z3_inv_std,
            _a3,
            _a3_up,
            decode_cat,
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
        );

        // 1. MSE gradient w.r.t predication
        let scale = 2.0 / (b * self.img_dim) as f64;

        let delta_pred = pred.sub(target)?.affine(scale, 0.0)?;

        let delta_z5 = delta_pred.reshape((b, 1, h, w_img))?;

        //2. conv5 out backward
        let db5 = delta_z5.sum(0)?.sum(1)?.sum(1)?;
        let (delta_a4, dw5) = manual_conv2d_backward(a4, &self.w5, &delta_z5, device)?;

        // 3. leaky rule backward on z4
        let relu_grad4 = leaky_relu_grad(z4_hat)?;
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
        let (delta_a3_pre_from_attn, d_wqkv) = self
            .attn
            .backward(&intermediates[16..21], &delta_a3_scaled)?;
        let delta_a3_pre = delta_a3_pre_from_attn.add(&delta_a3_scaled)?;

        //8. Leaky relu backward on z3

        let relu_grad3 = leaky_relu_grad(z3_hat)?;
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
        let relu_grad2 = leaky_relu_grad(z2_hat)?;
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
        let relu_grad1 = leaky_relu_grad(z1_hat)?;
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
            dw_cond, db_cond, dw1, db1, dw2, db2, dw3, db3, dw4, db4, dw5, db5, d_wqkv,
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
            &self.attn.w_qkv,
        ]
    }
    fn param_names(&self) -> Vec<&str> {
        vec![
            "w_cond",
            "b_cond",
            "w1",
            "b1",
            "w2",
            "b2",
            "w3",
            "b3",
            "w4",
            "b4",
            "w5",
            "b5",
            "attn_w_qkv",
        ]
    }
}

// =============================================================================
// ADAPTIVE GROUP NORMALIZATION U-NET (SimpleDenoisingUNetAdaGN)
// =============================================================================
//
// Modern Diffusion Architecture (ADM / Stable Diffusion standard).
//
// Key Differences from SimpleDenoisingUNet:
//   1. 1-Channel Input: Image enters directly into `w1: [16, 1, 3, 3]` without
//      needing an artificial 28x28 spatial conditioning broadcast map.
//   2. Deep Conditioning Injection: Conditioning c = [time_emb, class_one_hot]
//      is projected via Linear layers (w_ada1..4) into (γ, β) scales and shifts
//      that modulate every normalization layer throughout the entire depth.
//   3. Identity Start: All AdaGN projection weights and biases are initialized
//      to zeros, ensuring the network begins training identically to standard
//      GroupNorm.
//   4. Hand-written Backpropagation: Full analytical gradients across all 5 conv
//      layers, bottleneck self-attention, and the 4 AdaGN linear projections.
pub struct SimpleDenoisingUNetAdaGN {
    varmap: VarMap,
    pub img_dim: usize,
    pub cond_dim: usize,

    // Convolutions (Note: w1 now takes 1 channel instead of 2!)
    pub w1: Tensor, // [16, 1, 3, 3] Level 1 Encoder
    pub b1: Tensor, // [16]
    pub w2: Tensor, // [32, 16, 3, 3] Level 2 Encoder
    pub b2: Tensor, // [32]
    pub w3: Tensor, // [32, 32, 3, 3] Bottleneck Conv
    pub b3: Tensor, // [32]
    pub w4: Tensor, // [16, 48, 3, 3] Decoder Conv (concat 32-ch up + 16-ch skip)
    pub b4: Tensor, // [16]
    pub w5: Tensor, // [1, 16, 3, 3] Output projection to 1 image channel
    pub b5: Tensor, // [1]

    // AdaGN Linear Projections (cond_dim -> 2 * channels)
    pub w_ada1: Tensor, // [32, cond_dim] (outputs γ1[16] + β1[16])
    pub b_ada1: Tensor, // [32]
    pub w_ada2: Tensor, // [64, cond_dim] (outputs γ2[32] + β2[32])
    pub b_ada2: Tensor, // [64]
    pub w_ada3: Tensor, // [64, cond_dim] (outputs γ3[32] + β3[32])
    pub b_ada3: Tensor, // [64]
    pub w_ada4: Tensor, // [32, cond_dim] (outputs γ4[16] + β4[16])
    pub b_ada4: Tensor, // [32]

    // Bottleneck Spatial Self-Attention
    pub attn: SpatialSelfAttention,
}

impl SimpleDenoisingUNetAdaGN {
    pub fn new(img_dim: usize, cond_dim: usize, device: &Device) -> Result<Self> {
        let varmap = VarMap::new();

        // 1. Conv1: takes 1 input channel (1 -> 16)
        let scale1 = (2.0f64 / (1.0 * 3.0 * 3.0)).sqrt();
        let w1 = varstore::register(
            &varmap,
            "w1",
            (Tensor::randn(0.0f32, 1.0f32, (16, 1, 3, 3), device)? * scale1)?,
        )?;
        let b1 = varstore::register(&varmap, "b1", Tensor::zeros(16, DType::F32, device)?)?;

        // 2. Conv2: (16 -> 32)
        let scale2 = (2.0f64 / (16.0 * 3.0 * 3.0)).sqrt();
        let w2 = varstore::register(
            &varmap,
            "w2",
            (Tensor::randn(0.0f32, 1.0f32, (32, 16, 3, 3), device)? * scale2)?,
        )?;
        let b2 = varstore::register(&varmap, "b2", Tensor::zeros(32, DType::F32, device)?)?;

        // 3. Conv3: (32 -> 32)
        let scale3 = (2.0f64 / (32.0 * 3.0 * 3.0)).sqrt();
        let w3 = varstore::register(
            &varmap,
            "w3",
            (Tensor::randn(0.0f32, 1.0f32, (32, 32, 3, 3), device)? * scale3)?,
        )?;
        let b3 = varstore::register(&varmap, "b3", Tensor::zeros(32, DType::F32, device)?)?;

        // 4. Conv4: (48 -> 16)
        let scale4 = (2.0f64 / (48.0 * 3.0 * 3.0)).sqrt();
        let w4 = varstore::register(
            &varmap,
            "w4",
            (Tensor::randn(0.0f32, 1.0f32, (16, 48, 3, 3), device)? * scale4)?,
        )?;
        let b4 = varstore::register(&varmap, "b4", Tensor::zeros(16, DType::F32, device)?)?;

        // 5. Conv5: (16 -> 1)
        let scale5 = (2.0f64 / (16.0 * 3.0 * 3.0)).sqrt();
        let w5 = varstore::register(
            &varmap,
            "w5",
            (Tensor::randn(0.0f32, 1.0f32, (1, 16, 3, 3), device)? * scale5)?,
        )?;
        let b5 = varstore::register(&varmap, "b5", Tensor::zeros(1, DType::F32, device)?)?;

        // 6. AdaGN Projections: Zero-initialized for Identity start!
        let w_ada1 = varstore::register(
            &varmap,
            "w_ada1",
            Tensor::zeros((32, cond_dim), DType::F32, device)?,
        )?;
        let b_ada1 = varstore::register(&varmap, "b_ada1", Tensor::zeros(32, DType::F32, device)?)?;

        let w_ada2 = varstore::register(
            &varmap,
            "w_ada2",
            Tensor::zeros((64, cond_dim), DType::F32, device)?,
        )?;
        let b_ada2 = varstore::register(&varmap, "b_ada2", Tensor::zeros(64, DType::F32, device)?)?;

        let w_ada3 = varstore::register(
            &varmap,
            "w_ada3",
            Tensor::zeros((64, cond_dim), DType::F32, device)?,
        )?;
        let b_ada3 = varstore::register(&varmap, "b_ada3", Tensor::zeros(64, DType::F32, device)?)?;

        let w_ada4 = varstore::register(
            &varmap,
            "w_ada4",
            Tensor::zeros((32, cond_dim), DType::F32, device)?,
        )?;
        let b_ada4 = varstore::register(&varmap, "b_ada4", Tensor::zeros(32, DType::F32, device)?)?;

        let attn = SpatialSelfAttention::new(32, &varmap, "attn_", device)?;

        Ok(Self {
            varmap,
            img_dim,
            cond_dim,
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
            w_ada1,
            b_ada1,
            w_ada2,
            b_ada2,
            w_ada3,
            b_ada3,
            w_ada4,
            b_ada4,
            attn,
        })
    }
}

impl DenoisingModel for SimpleDenoisingUNetAdaGN {
    fn forward(&self, x: &Tensor) -> Result<(Tensor, Vec<Tensor>)> {
        let device = x.device();
        let b = x.dim(0)?;
        let h = (self.img_dim as f64).sqrt() as usize;
        let w_img = h;
        let h_down = h / 2;
        let w_down = w_img / 2;

        // 1. Separate input image and condition vector
        let xt = x.narrow(1, 0, self.img_dim)?;
        let cond_vec = x.narrow(1, self.img_dim, self.cond_dim)?;

        // 2. Compute AdaGN scale & shift for all layers
        let ada1 = cond_vec
            .matmul(&self.w_ada1.t()?)?
            .broadcast_add(&self.b_ada1)?;
        let gamma1 = ada1.narrow(1, 0, 16)?;
        let beta1 = ada1.narrow(1, 16, 16)?;

        let ada2 = cond_vec
            .matmul(&self.w_ada2.t()?)?
            .broadcast_add(&self.b_ada2)?;
        let gamma2 = ada2.narrow(1, 0, 32)?;
        let beta2 = ada2.narrow(1, 32, 32)?;

        let ada3 = cond_vec
            .matmul(&self.w_ada3.t()?)?
            .broadcast_add(&self.b_ada3)?;
        let gamma3 = ada3.narrow(1, 0, 32)?;
        let beta3 = ada3.narrow(1, 32, 32)?;

        let ada4 = cond_vec
            .matmul(&self.w_ada4.t()?)?
            .broadcast_add(&self.b_ada4)?;
        let gamma4 = ada4.narrow(1, 0, 16)?;
        let beta4 = ada4.narrow(1, 16, 16)?;

        // --- LEVEL 1 (28x28) ---
        let xt_img = xt.reshape((b, 1, h, w_img))?;
        let z1_conv = manual_conv2d(&xt_img, &self.w1, Some(&self.b1), &device)?;
        let (z1_norm, z1_hat, z1_inv_std) = adagn_forward(&z1_conv, &gamma1, &beta1, 4)?;
        let a1 = leaky_relu(&z1_norm)?;
        let a1_down = a1
            .reshape((b, 16, h_down, 2, w_down, 2))?
            .mean(5)?
            .mean(3)?;

        // --- LEVEL 2 (14x14) ---
        let z2_conv = manual_conv2d(&a1_down, &self.w2, Some(&self.b2), &device)?;
        let (z2_norm, z2_hat, z2_inv_std) = adagn_forward(&z2_conv, &gamma2, &beta2, 4)?;
        let a2 = leaky_relu(&z2_norm)?;

        // --- BOTTLENECK (14x14) ---
        let z3_conv = manual_conv2d(&a2, &self.w3, Some(&self.b3), &device)?;
        let z3_res = z3_conv.add(&a2)?.affine(RESIDUAL_SCALE, 0.0)?;
        let (z3_norm, z3_hat, z3_inv_std) = adagn_forward(&z3_res, &gamma3, &beta3, 4)?;
        let a3_pre = leaky_relu(&z3_norm)?;
        let (attn_out, attn_cached) = self.attn.forward(&a3_pre)?;
        let a3 = a3_pre.add(&attn_out)?.affine(RESIDUAL_SCALE, 0.0)?;

        // --- DECODER LEVEL (28x28) ---
        let a3_up = a3
            .reshape((b, 32, h_down, 1, w_down, 1))?
            .broadcast_as((b, 32, h_down, 2, w_down, 2))?
            .reshape((b, 32, h, w_img))?;
        let decode_cat = Tensor::cat(&[&a3_up, &a1], 1)?;
        let z4_conv = manual_conv2d(&decode_cat, &self.w4, Some(&self.b4), &device)?;
        let z4_res = z4_conv.add(&a1)?.affine(RESIDUAL_SCALE, 0.0)?;
        let (z4_norm, z4_hat, z4_inv_std) = adagn_forward(&z4_res, &gamma4, &beta4, 4)?;
        let a4 = leaky_relu(&z4_norm)?;

        // --- OUTPUT PROJECTION ---
        let z5 = manual_conv2d(&a4, &self.w5, Some(&self.b5), &device)?;
        let pred = z5.reshape((b, self.img_dim))?;

        let mut intermediates = vec![
            xt_img, cond_vec, gamma1, gamma2, gamma3, gamma4, z1_hat, z1_inv_std, a1, a1_down,
            z2_hat, z2_inv_std, a2, z3_hat, z3_inv_std, a3_pre, a3, a3_up, decode_cat, z4_hat,
            z4_inv_std, a4,
        ];
        intermediates.extend(attn_cached);

        Ok((pred, intermediates))
    }

    fn backward(
        &self,
        _v: &Tensor,
        intermediates: &[Tensor],
        pred: &Tensor,
        target: &Tensor,
    ) -> Result<Vec<Tensor>> {
        if intermediates.len() < 22 {
            bail!(
                "SimpleDenoisingUNetAdaGN expected at least 22 cached intermediates, got {}",
                intermediates.len()
            );
        }

        let device = pred.device();
        let b = pred.dim(0)?;
        let h = (self.img_dim as f64).sqrt() as usize;
        let w_img = h;
        let h_down = h / 2;
        let w_down = w_img / 2;

        let xt_img = &intermediates[0];
        let cond_vec = &intermediates[1];
        let gamma1 = &intermediates[2];
        let gamma2 = &intermediates[3];
        let gamma3 = &intermediates[4];
        let gamma4 = &intermediates[5];
        let z1_hat = &intermediates[6];
        let z1_inv_std = &intermediates[7];
        let a1 = &intermediates[8];
        let a1_down = &intermediates[9];
        let z2_hat = &intermediates[10];
        let z2_inv_std = &intermediates[11];
        let a2 = &intermediates[12];
        let z3_hat = &intermediates[13];
        let z3_inv_std = &intermediates[14];
        let a3_pre = &intermediates[15];
        let _a3 = &intermediates[16];
        let _a3_up = &intermediates[17];
        let decode_cat = &intermediates[18];
        let z4_hat = &intermediates[19];
        let z4_inv_std = &intermediates[20];
        let a4 = &intermediates[21];

        // 1. MSE gradient w.r.t prediction
        let scale = 2.0 / (b * self.img_dim) as f64;
        let delta_pred = pred.sub(target)?.affine(scale, 0.0)?;
        let delta_z5 = delta_pred.reshape((b, 1, h, w_img))?;

        // 2. Conv5 backward
        let db5 = delta_z5.sum(0)?.sum(1)?.sum(1)?;
        let (delta_a4, dw5) = manual_conv2d_backward(a4, &self.w5, &delta_z5, device)?;

        // 3. LeakyReLU + AdaGN on Level 4
        let relu_grad4 = leaky_relu_grad(a4)?;
        let delta_z4_norm = delta_a4.mul(&relu_grad4)?;
        let (delta_z4_res, delta_gamma4, delta_beta4) =
            adagn_backward(&delta_z4_norm, z4_hat, z4_inv_std, gamma4, 4)?;

        // 4. Conv4 backward
        let delta_z4_conv = delta_z4_res.affine(RESIDUAL_SCALE, 0.0)?;
        let delta_a1_from_decoder_residual = delta_z4_res.affine(RESIDUAL_SCALE, 0.0)?;
        let db4 = delta_z4_conv.sum(0)?.sum(1)?.sum(1)?;
        let (delta_decode_cat, dw4) =
            manual_conv2d_backward(decode_cat, &self.w4, &delta_z4_conv, device)?;

        let delta_a3_up = delta_decode_cat.narrow(1, 0, 32)?.contiguous()?;
        let delta_a1_from_skip = delta_decode_cat.narrow(1, 32, 16)?.contiguous()?;

        // 5. Nearest Neighbour upsampling backward (sum 2x2 blocks)
        let delta_a3 = delta_a3_up
            .reshape((b, 32, h_down, 2, w_down, 2))?
            .sum(5)?
            .sum(3)?;

        // 6. Attention backward (residual)
        let delta_a3_scaled = delta_a3.affine(RESIDUAL_SCALE, 0.0)?;
        let (delta_a3_pre_from_attn, d_wqkv) = self
            .attn
            .backward(&intermediates[22..], &delta_a3_scaled)?;
        let delta_a3_pre_grad = delta_a3_pre_from_attn.add(&delta_a3_scaled)?;

        // 7. LeakyReLU + AdaGN on Level 3
        let relu_grad3 = leaky_relu_grad(a3_pre)?;
        let delta_z3_norm = delta_a3_pre_grad.mul(&relu_grad3)?;
        let (delta_z3_res, delta_gamma3, delta_beta3) =
            adagn_backward(&delta_z3_norm, z3_hat, z3_inv_std, gamma3, 4)?;

        // 8. Conv3 backward
        let delta_z3_conv = delta_z3_res.affine(RESIDUAL_SCALE, 0.0)?;
        let delta_a2_from_bottleneck_residual = delta_z3_res.affine(RESIDUAL_SCALE, 0.0)?;
        let db3 = delta_z3_conv.sum(0)?.sum(1)?.sum(1)?;
        let (delta_a2_from_conv3, dw3) =
            manual_conv2d_backward(a2, &self.w3, &delta_z3_conv, device)?;
        let delta_a2 = delta_a2_from_conv3.add(&delta_a2_from_bottleneck_residual)?;

        // 9. LeakyReLU + AdaGN on Level 2
        let relu_grad2 = leaky_relu_grad(a2)?;
        let delta_z2 = delta_a2.mul(&relu_grad2)?;
        let (delta_z2_conv, delta_gamma2, delta_beta2) =
            adagn_backward(&delta_z2, z2_hat, z2_inv_std, gamma2, 4)?;

        // 10. Conv2 backward
        let db2 = delta_z2_conv.sum(0)?.sum(1)?.sum(1)?;
        let (delta_a1_down, dw2) =
            manual_conv2d_backward(a1_down, &self.w2, &delta_z2_conv, device)?;

        // 11. Average pool 2x2 backward
        let scaled_delta = delta_a1_down.affine(0.25, 0.0)?;
        let delta_a1_from_down = scaled_delta
            .reshape((b, 16, h_down, 1, w_down, 1))?
            .broadcast_as((b, 16, h_down, 2, w_down, 2))?
            .reshape((b, 16, h, w_img))?;

        let delta_a1 = delta_a1_from_down
            .add(&delta_a1_from_skip)?
            .add(&delta_a1_from_decoder_residual)?;

        // 12. LeakyReLU + AdaGN on Level 1
        let relu_grad1 = leaky_relu_grad(a1)?;
        let delta_z1 = delta_a1.mul(&relu_grad1)?;
        let (delta_z1_conv, delta_gamma1, delta_beta1) =
            adagn_backward(&delta_z1, z1_hat, z1_inv_std, gamma1, 4)?;

        // 13. Conv1 backward (takes 1-channel xt_img!)
        let db1 = delta_z1_conv.sum(0)?.sum(1)?.sum(1)?;
        let (_delta_xt_img, dw1) =
            manual_conv2d_backward(xt_img, &self.w1, &delta_z1_conv, device)?;

        // 14. AdaGN Projections backward: [γ, β] linear layers
        let delta_ada1 = Tensor::cat(&[&delta_gamma1, &delta_beta1], 1)?;
        let dw_ada1 = delta_ada1.t()?.contiguous()?.matmul(cond_vec)?;
        let db_ada1 = delta_ada1.sum(0)?;

        let delta_ada2 = Tensor::cat(&[&delta_gamma2, &delta_beta2], 1)?;
        let dw_ada2 = delta_ada2.t()?.contiguous()?.matmul(cond_vec)?;
        let db_ada2 = delta_ada2.sum(0)?;

        let delta_ada3 = Tensor::cat(&[&delta_gamma3, &delta_beta3], 1)?;
        let dw_ada3 = delta_ada3.t()?.contiguous()?.matmul(cond_vec)?;
        let db_ada3 = delta_ada3.sum(0)?;

        let delta_ada4 = Tensor::cat(&[&delta_gamma4, &delta_beta4], 1)?;
        let dw_ada4 = delta_ada4.t()?.contiguous()?.matmul(cond_vec)?;
        let db_ada4 = delta_ada4.sum(0)?;

        Ok(vec![
            dw1, db1, dw2, db2, dw3, db3, dw4, db4, dw5, db5, dw_ada1, db_ada1, dw_ada2, db_ada2,
            dw_ada3, db_ada3, dw_ada4, db_ada4, d_wqkv,
        ])
    }
}

impl Parameterized for SimpleDenoisingUNetAdaGN {
    fn varmap(&self) -> &VarMap {
        &self.varmap
    }

    fn params(&self) -> Vec<&Tensor> {
        vec![
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
            &self.w_ada1,
            &self.b_ada1,
            &self.w_ada2,
            &self.b_ada2,
            &self.w_ada3,
            &self.b_ada3,
            &self.w_ada4,
            &self.b_ada4,
            &self.attn.w_qkv,
        ]
    }

    fn param_names(&self) -> Vec<&str> {
        vec![
            "w1",
            "b1",
            "w2",
            "b2",
            "w3",
            "b3",
            "w4",
            "b4",
            "w5",
            "b5",
            "w_ada1",
            "b_ada1",
            "w_ada2",
            "b_ada2",
            "w_ada3",
            "b_ada3",
            "w_ada4",
            "b_ada4",
            "attn_w_qkv",
        ]
    }
}
