// =============================================================================
// denoising_cnn_5layers.rs — SimpleDenoisingCNN5Layers: deeper CNN denoiser
// =============================================================================
//
// WHY a 5-layer CNN with 5×5 kernels?
//   `SimpleDenoisingCNN` (2 layers, 3×3 kernels) is fast to train but has a
//   small receptive field: after 2 conv layers with 3×3 kernels each pixel can
//   only "see" a 5×5 region of the input.  For MNIST this is often enough, but
//   a deeper model with wider kernels captures longer-range correlations:
//
//   Receptive field growth with 5×5 kernels:
//     After Conv1: 5×5  (single kernel)
//     After Conv2: 9×9  (5 + 2*(5-1)/2*2)  — roughly a quarter of a 28×28 image
//     After Conv3: 13×13
//     After Conv4: 17×17
//     After Conv5: 21×21 — covers the entire digit stroke area
//
//   Wider kernels (5×5) also process more neighbours per position, giving the
//   model richer spatial context at each layer without needing as many layers
//   as a 3×3 stack to reach the same receptive field size.
//
// WHY the encoder-decoder (hourglass) channel structure?
//   Conv1: 2  → 64  (expand)    — extract rich low-level features
//   Conv2: 64 → 128 (expand)    — learn more abstract patterns
//   Conv3: 128→ 128 (bottleneck)— process at peak representation width
//   Conv4: 128→  64 (contract)  — collapse redundant features
//   Conv5: 64 →   1 (output)    — project to single noise-prediction channel
//
//   This hour-glass shape mirrors the encoder-decoder structure of U-Net,
//   which is the standard architecture for diffusion model denoisers.
//   The expand-then-contract pattern allows the model to first build up
//   rich intermediate representations and then distil them into a clean
//   noise prediction.
//
// Architecture:
//   Input: v = concat(x_t, time_emb, class_one_hot) → shape (B, 784 + 26)
//   1. Split v into x_t (784 dims) and cond_vec (26 dims).
//   2. Conditioning projection: Linear(26 → 784) + reshape → (B, 1, 28, 28)
//   3. Channel cat: cat([xt_img, cond_map], dim=1) → (B, 2, 28, 28)
//   4. Conv1 (2  → 64,  5×5) + Leaky-ReLU(α=0.01)
//   5. Conv2 (64 → 128, 5×5) + Leaky-ReLU(α=0.01)
//   6. Conv3 (128→ 128, 5×5) + Leaky-ReLU(α=0.01)
//   7. Conv4 (128→  64, 5×5) + Leaky-ReLU(α=0.01)
//   8. Conv5 (64 →   1, 5×5) → reshape to (B, 784)
//
// All conv layers use SAME zero-padding (pad = 2 for 5×5 kernels) so the
// spatial size remains 28×28 throughout the network.
//
// All convolutions and gradient computations use the shared primitives from
// `denoising_cnn_ops`, which parallelise over kernel positions with rayon.
//
// This model implements the `DenoisingModel` trait, making it a drop-in
// replacement for `SimpleDenoisingCNN` in any training binary.
// =============================================================================

use anyhow::{bail, Result};
use candle_core::{DType, Device, Tensor};

use super::denoising_cnn_ops::{manual_conv2d, manual_conv2d_backward};
use super::DenoisingModel;
use crate::common::parameterized::Parameterized;
use crate::common::varstore;
use candle_nn::VarMap;

// =============================================================================
// SimpleDenoisingCNN5Layers — struct definition
// =============================================================================
//
// All parameters are public so training code can read them directly.  Writes go
// through `set_param()` by name.  Fields are ordered to match `params()` /
// `param_names()` / the gradient vector returned by `backward()`.
//
// Tensor shape annotations use [C_out, C_in, kH, kW] for conv kernels:
//
//   w_cond: (img_dim=784, cond_dim=26) — conditioning projection
//   b_cond: (784,)
//   w1:     (64,   2,   5, 5)          — Conv1: 2→64 channels
//   b1:     (64,)
//   w2:     (128,  64,  5, 5)          — Conv2: 64→128 channels
//   b2:     (128,)
//   w3:     (128,  128, 5, 5)          — Conv3: 128→128 channels (bottleneck)
//   b3:     (128,)
//   w4:     (64,   128, 5, 5)          — Conv4: 128→64 channels
//   b4:     (64,)
//   w5:     (1,    64,  5, 5)          — Conv5: 64→1 channel (noise prediction)
//   b5:     (1,)
pub struct SimpleDenoisingCNN5Layers {
    /// Owns every trainable parameter under its checkpoint name; the tensor
    /// fields below share storage with its `Var`s.
    varmap: VarMap,

    pub img_dim: usize,  // flattened image size (784 for MNIST)
    pub cond_dim: usize, // conditioning vector size (time_emb_dim + class_dim = 26)
    pub w_cond: Tensor,  // [img_dim, cond_dim]
    pub b_cond: Tensor,  // [img_dim]
    pub w1: Tensor,      // [64,  2,   5, 5]
    pub b1: Tensor,      // [64]
    pub w2: Tensor,      // [128, 64,  5, 5]
    pub b2: Tensor,      // [128]
    pub w3: Tensor,      // [128, 128, 5, 5]
    pub b3: Tensor,      // [128]
    pub w4: Tensor,      // [64,  128, 5, 5]
    pub b4: Tensor,      // [64]
    pub w5: Tensor,      // [1,   64,  5, 5]
    pub b5: Tensor,      // [1]
}

impl SimpleDenoisingCNN5Layers {
    // =========================================================================
    // new — initialise all parameters with He (Kaiming) initialisation
    // =========================================================================
    //
    // He initialisation: std = sqrt(2 / fan_in)
    //   This keeps activation variance stable across layers with ReLU-family
    //   nonlinearities.  For conv layers: fan_in = C_in * kH * kW.
    //
    // NOTE: The conv channel counts in the struct comments (16/32) reflect
    // earlier versions of the model.  The actual initialised shapes use the
    // wider channel widths (64/128) specified below, which were found to give
    // better convergence on MNIST with 5×5 kernels.
    pub fn new(img_dim: usize, cond_dim: usize, device: &Device) -> Result<Self> {
        // Every parameter is registered in the VarMap; keep the tensor that
        // `register` returns, not the one passed in — only the former shares
        // storage with the stored `Var` and observes later updates.
        let varmap = VarMap::new();

        // --- Conditioning projection ------------------------------------------
        // fan_in = cond_dim (26 input features per output neuron).
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

        // --- Conv1 weights (2 in-channels → 64 out-channels, 5×5 kernel) ----
        // fan_in = C_in * kH * kW = 2 * 5 * 5 = 50
        // scale1 = sqrt(2 / 50) ≈ 0.2000
        let scale1 = (2.0f64 / (2.0 * 5.0 * 5.0)).sqrt();
        let w1 = varstore::register(
            &varmap,
            "w1",
            (Tensor::randn(0.0f32, 1.0f32, (64, 2, 5, 5), device)? * scale1)?,
        )?;
        let b1 = varstore::register(&varmap, "b1", Tensor::zeros(64, DType::F32, device)?)?;

        // --- Conv2 weights (64 → 128 channels, 5×5 kernel) ------------------
        // fan_in = 64 * 5 * 5 = 1600
        // scale2 = sqrt(2 / 1600) ≈ 0.03536
        let scale2 = (2.0f64 / (64.0 * 5.0 * 5.0)).sqrt();
        let w2 = varstore::register(
            &varmap,
            "w2",
            (Tensor::randn(0.0f32, 1.0f32, (128, 64, 5, 5), device)? * scale2)?,
        )?;
        let b2 = varstore::register(&varmap, "b2", Tensor::zeros(128, DType::F32, device)?)?;

        // --- Conv3 weights (128 → 128 channels, 5×5 kernel) -----------------
        // fan_in = 128 * 5 * 5 = 3200
        // scale3 = sqrt(2 / 3200) ≈ 0.02500
        let scale3 = (2.0f64 / (128.0 * 5.0 * 5.0)).sqrt();
        let w3 = varstore::register(
            &varmap,
            "w3",
            (Tensor::randn(0.0f32, 1.0f32, (128, 128, 5, 5), device)? * scale3)?,
        )?;
        let b3 = varstore::register(&varmap, "b3", Tensor::zeros(128, DType::F32, device)?)?;

        // --- Conv4 weights (128 → 64 channels, 5×5 kernel) ------------------
        // fan_in = 128 * 5 * 5 = 3200  (same as Conv3; contracting path)
        // scale4 = sqrt(2 / 3200) ≈ 0.02500
        let scale4 = (2.0f64 / (128.0 * 5.0 * 5.0)).sqrt();
        let w4 = varstore::register(
            &varmap,
            "w4",
            (Tensor::randn(0.0f32, 1.0f32, (64, 128, 5, 5), device)? * scale4)?,
        )?;
        let b4 = varstore::register(&varmap, "b4", Tensor::zeros(64, DType::F32, device)?)?;

        // --- Conv5 weights (64 → 1 channel, 5×5 kernel) ---------------------
        // fan_in = 64 * 5 * 5 = 1600
        // scale5 = sqrt(2 / 1600) ≈ 0.03536
        let scale5 = (2.0f64 / (64.0 * 5.0 * 5.0)).sqrt();
        let w5 = varstore::register(
            &varmap,
            "w5",
            (Tensor::randn(0.0f32, 1.0f32, (1, 64, 5, 5), device)? * scale5)?,
        )?;
        let b5 = varstore::register(&varmap, "b5", Tensor::zeros(1, DType::F32, device)?)?;

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
        })
    }
}

// =============================================================================
// DenoisingModel implementation
// =============================================================================
impl DenoisingModel for SimpleDenoisingCNN5Layers {
    // =========================================================================
    // forward — 5-layer CNN noise prediction
    // =========================================================================
    //
    // Input `x`: flat concatenated tensor (B, img_dim + cond_dim).
    //   x[:, 0:784]   = x_t   (noisy image, flat)
    //   x[:, 784:810] = cond  (time_emb 16-dim + class_one_hot 10-dim)
    //
    // Data flow:
    //   1. Split x → xt (784), cond_vec (26).
    //   2. cond_vec @ w_cond^T + b_cond → (B, 784) → reshape → (B, 1, 28, 28)
    //      WHY project to 784? We need the conditioning signal to have the same
    //      spatial resolution as the noisy image for channel-concatenation.
    //   3. cat([xt_img, cond_map], dim=1) → (B, 2, 28, 28)
    //      Two-channel input: channel 0 = noisy image, channel 1 = context.
    //   4-8. 5 × [Conv(5×5) + Leaky-ReLU(0.01)], except last (no activation).
    //      No activation after Conv5: noise predictions can be any real value.
    //
    // Returns:
    //   pred          — predicted noise ε̂, shape (B, 784)
    //   intermediates — 9 cached tensors needed by backward():
    //                   [input_cat, z1, a1, z2, a2, z3, a3, z4, a4]
    //
    // WHY cache pre- and post-activation tensors separately?
    //   The backward pass needs z (pre-activation) to compute the Leaky-ReLU
    //   derivative and a (post-activation) as the input to the next conv's
    //   weight gradient.  Caching both avoids re-running the forward pass.
    fn forward(&self, x: &Tensor) -> Result<(Tensor, Vec<Tensor>)> {
        let device = x.device();
        let b = x.dim(0)?;

        // Compute spatial size from img_dim (assumes square images).
        // MNIST: sqrt(784) = 28, so h = w_img = 28.
        let h = (self.img_dim as f64).sqrt() as usize;
        let w_img = h;

        // --- Step 1: Split input -------------------------------------------
        let xt = x.narrow(1, 0, self.img_dim)?;
        let cond_vec = x.narrow(1, self.img_dim, self.cond_dim)?;

        // --- Step 2: Conditioning projection → spatial map ------------------
        // .contiguous() before matmul ensures a dense memory layout, which
        // is required by Candle's matmul op on non-contiguous slices.
        let cond_map = cond_vec
            .contiguous()?
            .matmul(&self.w_cond.t()?.contiguous()?)?
            .broadcast_add(&self.b_cond)?
            .reshape((b, 1, h, w_img))?;

        // --- Step 3: 2-channel input ----------------------------------------
        let xt_img = xt.reshape((b, 1, h, w_img))?;
        let input_cat = Tensor::cat(&[&xt_img, &cond_map], 1)?; // (B, 2, H, W)

        // --- Step 4: Conv1 + Leaky-ReLU  (2→64 channels, 5×5) --------------
        // z1 = pre-activation (B, 64, 28, 28)
        // a1 = Leaky-ReLU(z1): max(z, 0.01*z)
        let z1 = manual_conv2d(&input_cat, &self.w1, Some(&self.b1), &device)?;
        let a1 = z1.maximum(&z1.affine(0.01, 0.0)?)?;

        // --- Step 5: Conv2 + Leaky-ReLU  (64→128 channels, 5×5) ------------
        let z2 = manual_conv2d(&a1, &self.w2, Some(&self.b2), &device)?;
        let a2 = z2.maximum(&z2.affine(0.01, 0.0)?)?;

        // --- Step 6: Conv3 + Leaky-ReLU  (128→128 channels, 5×5) -----------
        // Bottleneck: same channel count in/out; processes at max capacity.
        let z3 = manual_conv2d(&a2, &self.w3, Some(&self.b3), &device)?;
        let a3 = z3.maximum(&z3.affine(0.01, 0.0)?)?;

        // --- Step 7: Conv4 + Leaky-ReLU  (128→64 channels, 5×5) ------------
        // Contracting: halve the channel count, collapsing redundant features.
        let z4 = manual_conv2d(&a3, &self.w4, Some(&self.b4), &device)?;
        let a4 = z4.maximum(&z4.affine(0.01, 0.0)?)?;

        // --- Step 8: Conv5 → output  (64→1 channel, 5×5) -------------------
        // No activation: noise predictions are unbounded real values.
        // z5 shape: (B, 1, 28, 28) → flattened to (B, 784).
        let z5 = manual_conv2d(&a4, &self.w5, Some(&self.b5), &device)?;
        let pred = z5.reshape((b, self.img_dim))?;

        // Cache 9 tensors required by backward().
        // Order is fixed: [input_cat, z1, a1, z2, a2, z3, a3, z4, a4]
        let intermediates = vec![input_cat, z1, a1, z2, a2, z3, a3, z4, a4];
        Ok((pred, intermediates))
    }

    // =========================================================================
    // backward — 5-layer chain-rule gradient computation
    // =========================================================================
    //
    // Computes the gradient of the MSE loss:
    //   L = (2 / (B * img_dim)) * ||pred - target||²
    //
    // with respect to all 12 parameters via back-propagation through the
    // 5-layer encoder-decoder:
    //
    //   Loss → pred → z5 → Conv5 → a4 → LReLU4 → Conv4 → a3 → LReLU3
    //               → Conv3 → a2 → LReLU2 → Conv2 → a1 → LReLU1 → Conv1
    //               → input_cat → cond_map → w_cond, b_cond
    //
    // For each conv layer the backward pass calls manual_conv2d_backward which
    // returns (delta_input, dw) simultaneously.
    // For each Leaky-ReLU the gradient is: f'(z) = 1 if z≥0, 0.01 if z<0.
    //   Implemented as: (z >= 0) * 0.99 + 0.01  (see WHY in denoising_cnn.rs).
    //
    // Returns: [dw_cond, db_cond, dw1, db1, dw2, db2, dw3, db3, dw4, db4, dw5, db5]
    //   (same order as params() / param_names())
    fn backward(
        &self,
        v: &Tensor,
        intermediates: &[Tensor],
        pred: &Tensor,
        target: &Tensor,
    ) -> Result<Vec<Tensor>> {
        // Sanity check: forward() caches exactly 9 tensors.
        if intermediates.len() != 9 {
            bail!(
                "SimpleDenoisingCNN5Layers expected 9 cached intermediates from forward(), got {}",
                intermediates.len()
            );
        }

        let device = v.device();
        let b = v.dim(0)?;
        let h = (self.img_dim as f64).sqrt() as usize;
        let w_img = h;

        // Unpack cached intermediates in the same order as forward().
        let input_cat = &intermediates[0]; // (B, 2,   28, 28) — 2-channel input
        let z1 = &intermediates[1]; // (B, 64,  28, 28) — Conv1 pre-act
        let a1 = &intermediates[2]; // (B, 64,  28, 28) — Conv1 post-act
        let z2 = &intermediates[3]; // (B, 128, 28, 28) — Conv2 pre-act
        let a2 = &intermediates[4]; // (B, 128, 28, 28) — Conv2 post-act
        let z3 = &intermediates[5]; // (B, 128, 28, 28) — Conv3 pre-act
        let a3 = &intermediates[6]; // (B, 128, 28, 28) — Conv3 post-act
        let z4 = &intermediates[7]; // (B, 64,  28, 28) — Conv4 pre-act
        let a4 = &intermediates[8]; // (B, 64,  28, 28) — Conv4 post-act

        // --- MSE gradient ∂L/∂pred -----------------------------------------
        // scale = 2 / (B * img_dim) — the derivative of (1/N)||pred-target||²
        // multiplied by 2 from the squared-difference expansion.
        let scale = 2.0 / (b * self.img_dim) as f64;
        let delta_pred = pred.sub(target)?.affine(scale, 0.0)?;

        // Reshape to spatial: (B, 784) → (B, 1, 28, 28) for Conv5 backward.
        let delta_z5 = delta_pred.reshape((b, 1, h, w_img))?;

        // --- Conv5 backward: (64→1 channels) --------------------------------
        // db5: sum over B, H, W — bias gradient sums out the spatial axes.
        let db5 = delta_z5.sum(0)?.sum(1)?.sum(1)?;
        // dw5: weight gradient (1, 64, 5, 5); delta_a4: backprop into a4.
        let (delta_a4, dw5) = manual_conv2d_backward(a4, &self.w5, &delta_z5, &device)?;

        // --- Leaky-ReLU4 backward -------------------------------------------
        // f'(z4) = 0.99 * (z4>=0) + 0.01 correctly encodes:
        //   f'(z) = 1.00 for z≥0 and f'(z) = 0.01 for z<0.
        let relu_grad4 = z4.ge(0.0f32)?.to_dtype(DType::F32)?.affine(0.99, 0.01)?;
        let delta_z4 = delta_a4.mul(&relu_grad4)?;

        // --- Conv4 backward: (128→64 channels) ------------------------------
        let db4 = delta_z4.sum(0)?.sum(1)?.sum(1)?;
        let (delta_a3, dw4) = manual_conv2d_backward(a3, &self.w4, &delta_z4, &device)?;

        // --- Leaky-ReLU3 backward -------------------------------------------
        let relu_grad3 = z3.ge(0.0f32)?.to_dtype(DType::F32)?.affine(0.99, 0.01)?;
        let delta_z3 = delta_a3.mul(&relu_grad3)?;

        // --- Conv3 backward: (128→128 channels, bottleneck) -----------------
        let db3 = delta_z3.sum(0)?.sum(1)?.sum(1)?;
        let (delta_a2, dw3) = manual_conv2d_backward(a2, &self.w3, &delta_z3, &device)?;

        // --- Leaky-ReLU2 backward -------------------------------------------
        let relu_grad2 = z2.ge(0.0f32)?.to_dtype(DType::F32)?.affine(0.99, 0.01)?;
        let delta_z2 = delta_a2.mul(&relu_grad2)?;

        // --- Conv2 backward: (64→128 channels) ------------------------------
        let db2 = delta_z2.sum(0)?.sum(1)?.sum(1)?;
        let (delta_a1, dw2) = manual_conv2d_backward(a1, &self.w2, &delta_z2, &device)?;

        // --- Leaky-ReLU1 backward -------------------------------------------
        let relu_grad1 = z1.ge(0.0f32)?.to_dtype(DType::F32)?.affine(0.99, 0.01)?;
        let delta_z1 = delta_a1.mul(&relu_grad1)?;

        // --- Conv1 backward: (2→64 channels) --------------------------------
        // delta_input_cat: gradient w.r.t. the 2-channel (image + cond) input.
        let db1 = delta_z1.sum(0)?.sum(1)?.sum(1)?;
        let (delta_input_cat, dw1) =
            manual_conv2d_backward(input_cat, &self.w1, &delta_z1, &device)?;

        // --- Conditioning projection backward --------------------------------
        // input_cat has 2 channels: [0]=xt_img, [1]=cond_map.
        // We only need the gradient w.r.t. channel 1 (the conditioning map).
        // WHY not channel 0? The noisy image xt is a *data input*, not a
        // model parameter.  Gradients w.r.t. model parameters only flow through
        // the conditioning projection (w_cond, b_cond).
        let delta_cond_map = delta_input_cat.narrow(1, 1, 1)?;
        let delta_cond_flat = delta_cond_map.reshape((b, self.img_dim))?;

        // db_cond: sum over batch → (img_dim,)
        let db_cond = delta_cond_flat.sum(0)?;

        // dw_cond = delta_cond_flat^T @ cond_vec → (img_dim, cond_dim)
        // Outer-product gradient: W has shape (img_dim, cond_dim), so
        //   ∂L/∂W = (∂L/∂output)^T × input = (B, img_dim)^T × (B, cond_dim)
        let cond_vec = v.narrow(1, self.img_dim, self.cond_dim)?.contiguous()?;
        let dw_cond = delta_cond_flat.t()?.contiguous()?.matmul(&cond_vec)?;

        // Return all 12 gradients in params() / param_names() order.
        Ok(vec![
            dw_cond, db_cond, dw1, db1, dw2, db2, dw3, db3, dw4, db4, dw5, db5,
        ])
    }
}

// =============================================================================
// Parameterized implementation — weight access for optimizers/EMA/checkpoints
// =============================================================================
impl Parameterized for SimpleDenoisingCNN5Layers {
    fn varmap(&self) -> &VarMap {
        &self.varmap
    }

    // =========================================================================
    // params — immutable parameter references (Adam initialisation)
    // =========================================================================
    // Order must be consistent with backward() output and param_names().
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
        ]
    }

    // =========================================================================
    // param_names — human-readable labels for gradient-norm logging
    // =========================================================================
    // Zipped with the gradient vector in training binaries to print per-layer
    // gradient norms.  Same order as params() and backward().
    fn param_names(&self) -> Vec<&str> {
        vec![
            "w_cond", "b_cond", "w1", "b1", "w2", "b2", "w3", "b3", "w4", "b4", "w5", "b5",
        ]
    }
}
