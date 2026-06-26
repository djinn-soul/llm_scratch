// =============================================================================
// denoising_cnn.rs — SimpleDenoisingCNN: a 2-layer CNN noise predictor
// =============================================================================
//
// This module implements a small convolutional denoising network that serves
// as a drop-in replacement for `SimpleDenoisingMlp` in the DDPM pipeline.
//
// WHY a CNN for denoising?
//   Diffusion models need to predict the noise ε added to an image x_0.
//   An MLP operates on the flattened 784-pixel vector, treating every pixel
//   independently — it has no built-in notion of proximity.  A CNN instead
//   processes the image as a 2-D grid with 3×3 sliding kernels, so it can
//   detect and remove structured noise patterns (edges, strokes) that span
//   adjacent pixels.  This spatial inductive bias is especially useful for
//   natural images where nearby pixels are strongly correlated.
//
// Architecture (forward path):
//
//   Input: v = concat(x_t, time_emb, class_one_hot)
//          shape: (B, img_dim + cond_dim) = (B, 784 + 26)
//
//   1. Split v into x_t (784 dims) and cond_vec (26 dims).
//
//   2. Conditioning projection:
//      cond_map = cond_vec @ w_cond^T + b_cond   → shape (B, 784)
//      reshape  → (B, 1, 28, 28)
//      This broadcasts the conditioning signal (time step + class) to every
//      spatial location of the image grid.
//
//   3. Channel concatenation:
//      xt_img   = x_t reshaped → (B, 1, 28, 28)
//      input_cat = cat([xt_img, cond_map], dim=1) → (B, 2, 28, 28)
//      The CNN receives two aligned spatial maps: noisy image and context.
//
//   4. Conv1 (first convolutional block):
//      z1 = manual_conv2d(input_cat, w1, b1)  [2 → 16 channels, 3×3 kernel]
//      a1 = Leaky-ReLU(z1, alpha=0.01)
//      WHY 16 channels? Enough feature maps to learn edge detectors and
//      stroke patterns without being over-parameterised for 28×28 images.
//
//   5. Conv2 (output block):
//      z2 = manual_conv2d(a1, w2, b2)         [16 → 1 channel, 3×3 kernel]
//      pred = z2 reshaped → (B, 784)
//      WHY 1 output channel? The noise ε has the same single-channel shape as
//      the input image; we predict one scalar per pixel.
//
// Backward path (manual chain rule):
//   - Conv2 backward: gradient w.r.t. a1, w2, b2
//   - Leaky-ReLU backward: gradient through the activation
//   - Conv1 backward: gradient w.r.t. input_cat, w1, b1
//   - Conditioning projection backward: gradient w.r.t. w_cond, b_cond
//   Gradients are returned as [dw_cond, db_cond, dw1, db1, dw2, db2],
//   matching the order in `params()`.
//
// Convolution implementation:
//   Both conv layers are implemented with a manual sliding-window loop
//   (`manual_conv2d`) that uses `Tensor::narrow` and `Tensor::matmul` rather
//   than a built-in conv op.  This approach works on any Candle backend
//   (CPU / CUDA / Metal) and avoids a dependency on Candle's candle-nn layer
//   wrappers.  The backward pass uses a matching `manual_conv2d_backward`
//   that computes gradients via shifted-matmul, equivalent to the standard
//   transposed-convolution identity.
// =============================================================================

use anyhow::{bail, Result};

use super::DenoisingModel;
use candle_core::{DType, Device, Tensor};

// =============================================================================
// SimpleDenoisingCNN — struct definition
// =============================================================================
//
// All six parameter tensors are public so the Adam optimizer can update them
// via `params_mut()` without unsafe pointer manipulation.
//
// Field layout (dimension comments show [out, in, kH, kW] for conv weights):
//
//   w_cond  — conditioning projection: (img_dim, cond_dim)
//             Maps the flat 26-dim (time_emb + class) vector to a spatial map.
//
//   b_cond  — conditioning projection bias: (img_dim,)
//
//   w1      — Conv1 kernel: (16, 2, 3, 3)
//             16 output feature maps from 2 input channels (image + cond_map).
//
//   b1      — Conv1 bias: (16,)
//
//   w2      — Conv2 kernel: (1, 16, 3, 3)
//             Reduces 16 feature maps back to 1 (noise prediction channel).
//
//   b2      — Conv2 bias: (1,)
pub struct SimpleDenoisingCNN {
    pub img_dim:  usize,   // flattened image size (784 for MNIST)
    pub cond_dim: usize,   // conditioning vector size (time_emb_dim + class_dim)
    pub w_cond:   Tensor,  // [img_dim, cond_dim]
    pub b_cond:   Tensor,  // [img_dim]
    pub w1:       Tensor,  // [16, 2, 3, 3]
    pub b1:       Tensor,  // [16]
    pub w2:       Tensor,  // [1, 16, 3, 3]
    pub b2:       Tensor,  // [1]
}

impl SimpleDenoisingCNN {
    // =========================================================================
    // new — initialise all parameters with He (Kaiming) uniform scaling
    // =========================================================================
    //
    // WHY He initialisation?
    //   He initialisation sets the standard deviation of random weights to
    //   sqrt(2 / fan_in), where fan_in is the number of inputs per neuron.
    //   This keeps the variance of activations roughly constant across layers
    //   even with ReLU/Leaky-ReLU nonlinearities, which helps gradients
    //   propagate through deep networks without vanishing or exploding.
    //
    //   For a conv layer with C_in input channels and K×K kernels:
    //     fan_in = C_in * K * K
    //     scale  = sqrt(2 / fan_in)
    //
    // The conditioning projection uses a similar scale with fan_in = cond_dim.
    pub fn new(img_dim: usize, cond_dim: usize, device: &Device) -> Result<Self> {
        // --- Conditioning projection weights ---------------------------------
        // fan_in = cond_dim (number of input features per output neuron).
        let scale_cond = (2.0f64 / cond_dim as f64).sqrt();
        let w_cond = (Tensor::randn(0.0f32, 1.0f32, (img_dim, cond_dim), device)? * scale_cond)?;
        let b_cond = Tensor::zeros(img_dim, DType::F32, device)?;

        // --- Conv1 weights ---------------------------------------------------
        // fan_in = C_in * kH * kW = 2 * 3 * 3 = 18
        // scale1 = sqrt(2 / 18) ≈ 0.333
        let scale1 = (2.0f64 / (2.0 * 3.0 * 3.0)).sqrt();
        let w1 = (Tensor::randn(0.0f32, 1.0f32, (16, 2, 3, 3), device)? * scale1)?;
        let b1 = Tensor::zeros(16, DType::F32, device)?;

        // --- Conv2 weights ---------------------------------------------------
        // fan_in = C_in * kH * kW = 16 * 3 * 3 = 144
        // scale2 = sqrt(2 / 144) ≈ 0.118
        let scale2 = (2.0f64 / (16.0 * 3.0 * 3.0)).sqrt();
        let w2 = (Tensor::randn(0.0f32, 1.0f32, (1, 16, 3, 3), device)? * scale2)?;
        let b2 = Tensor::zeros(1, DType::F32, device)?;

        Ok(Self {
            img_dim,
            cond_dim,
            w_cond,
            b_cond,
            w1,
            b1,
            w2,
            b2,
        })
    }
}

// =============================================================================
// manual_conv2d — zero-padded 2-D convolution implemented via matmul
// =============================================================================
//
// WHY implement conv from scratch?
//   Candle's built-in conv2d requires the `candle-nn` crate and ties the model
//   to that API.  Implementing manually keeps the model self-contained and
//   makes the backward pass straightforward to pair with (same index math).
//
// Algorithm:
//   For each kernel position (dy, dx) in {0,1,2} × {0,1,2}:
//     1. Slice the zero-padded input at position (dy, dx):
//        x_slice = x_padded[:, :, dy:dy+H, dx:dx+W]  → (B, C_in, H, W)
//     2. Reshape to (C_in, B*H*W) and multiply by the weight slice
//        w_slice = w[:, :, dy, dx]  → (C_out, C_in)
//        out_slice = w_slice @ x_slice_flat  → (C_out, B*H*W)
//     3. Reshape back to (B, C_out, H, W) and accumulate into y.
//
// WHY loop over (dy, dx) and use matmul instead of direct conv?
//   Candle tensors support matmul efficiently on both CPU and GPU.  Decomposing
//   the 3×3 kernel into 9 independent (C_out × C_in) @ (C_in × B*H*W) matmuls
//   leverages BLAS/cuBLAS for the heavy computation while avoiding a custom
//   CUDA kernel.
//
// Padding:
//   Zero-pad by 1 pixel on all four sides so output spatial size = input size.
//   This "SAME" padding preserves the 28×28 grid throughout the network.
fn manual_conv2d(x: &Tensor, w: &Tensor, bias: Option<&Tensor>, device: &Device) -> Result<Tensor> {
    let (b, c_in, h, w_img) = x.dims4()?;
    let (c_out, _, _, _) = w.dims4()?;

    // --- Zero-pad: add one row of zeros at top and bottom (height axis) ------
    let zero_row = Tensor::zeros((b, c_in, 1, w_img), DType::F32, device)?;
    let x_padded_y = Tensor::cat(&[&zero_row, x, &zero_row], 2)?;

    // --- Zero-pad: add one column of zeros on left and right (width axis) ----
    let zero_col = Tensor::zeros((b, c_in, h + 2, 1), DType::F32, device)?;
    let x_padded = Tensor::cat(&[&zero_col, &x_padded_y, &zero_col], 3)?;

    // Accumulator for the convolution output.
    let mut y = Tensor::zeros((b, c_out, h, w_img), DType::F32, device)?;

    // --- Slide the 3×3 kernel over all (dy, dx) positions -------------------
    // WHY iterate rather than a single call?
    //   Candle has no fused sliding-window gather op.  Each (dy, dx) position
    //   selects a spatial slice of the padded input and performs one matmul.
    //   9 matmuls total for a 3×3 kernel, each of shape (C_out × B*H*W).
    for dy in 0..3 {
        for dx in 0..3 {
            // Extract the input patch corresponding to kernel position (dy, dx).
            // `narrow(dim, start, len)` selects `len` elements along `dim`.
            let x_slice = x_padded.narrow(2, dy, h)?.narrow(3, dx, w_img)?;

            // Reshape the spatial patch to (C_in, B*H*W) for matrix multiply.
            let x_flat = x_slice.reshape((b, c_in, h * w_img))?;
            let x_perm = x_flat.permute((1, 0, 2))?.reshape((c_in, b * h * w_img))?;

            // Select the kernel weights for this (dy, dx) position.
            // w_slice shape: (C_out, C_in)
            let w_slice = w
                .narrow(2, dy, 1)?
                .narrow(3, dx, 1)?
                .reshape((c_out, c_in))?;

            // out_slice = w_slice @ x_perm  → (C_out, B*H*W)
            let out_slice = w_slice.matmul(&x_perm)?;

            // Reshape back to (B, C_out, H, W) and accumulate.
            let out_reshaped = out_slice
                .reshape((c_out, b, h, w_img))?
                .permute((1, 0, 2, 3))?;
            y = y.add(&out_reshaped)?;
        }
    }

    // Add bias by broadcasting (1, C_out, 1, 1) across (B, C_out, H, W).
    if let Some(bi) = bias {
        y = y.broadcast_add(&bi.reshape((1, c_out, 1, 1))?)?;
    }
    Ok(y)
}

// =============================================================================
// shift_and_pad — spatial shift helper used in manual_conv2d_backward
// =============================================================================
//
// Shifts a 4-D tensor by (sy, sx) pixels along the (H, W) axes and fills the
// vacated border with zeros.  This is the "roll with zero-fill" operation.
//
// WHY is this needed for the backward pass?
//   The gradient with respect to the input of a convolution is itself a
//   convolution with the *transposed* (flipped) kernel.  In the sliding-window
//   decomposition above, "transposing" amounts to reversing the (dy, dx) offsets
//   and shifting the gradient tensor by (-sy, -sx).  `shift_and_pad` performs
//   that shift for each kernel position in `manual_conv2d_backward`.
//
// Arguments:
//   t        — input tensor of shape (B, C, H, W)
//   sy, sx   — shift amounts in {-1, 0, 1} along H and W respectively
//              Positive values shift content *down/right* (new zeros at top/left).
//              Negative values shift content *up/left* (new zeros at bottom/right).
//   device   — target device for zero tensors
fn shift_and_pad(t: &Tensor, sy: i32, sx: i32, device: &Device) -> Result<Tensor> {
    let (b, c, h, w) = t.dims4()?;
    let mut out = t.clone();

    // --- Vertical shift (along the height dimension) -------------------------
    if sy == 1 {
        // Shift content down by 1: prepend a row of zeros, drop the last row.
        let zero = Tensor::zeros((b, c, 1, w), DType::F32, device)?;
        let sliced = t.narrow(2, 0, h - 1)?; // keep all but last row
        out = Tensor::cat(&[&zero, &sliced], 2)?;
    } else if sy == -1 {
        // Shift content up by 1: drop the first row, append a row of zeros.
        let zero = Tensor::zeros((b, c, 1, w), DType::F32, device)?;
        let sliced = t.narrow(2, 1, h - 1)?; // keep all but first row
        out = Tensor::cat(&[&sliced, &zero], 2)?;
    }
    // sy == 0: no vertical shift.

    // --- Horizontal shift (along the width dimension) ------------------------
    if sx == 1 {
        // Shift content right by 1: prepend a column of zeros, drop the last.
        let zero = Tensor::zeros((b, c, h, 1), DType::F32, device)?;
        let sliced = out.narrow(3, 0, w - 1)?;
        out = Tensor::cat(&[&zero, &sliced], 3)?;
    } else if sx == -1 {
        // Shift content left by 1: drop the first column, append zeros.
        let zero = Tensor::zeros((b, c, h, 1), DType::F32, device)?;
        let sliced = out.narrow(3, 1, w - 1)?;
        out = Tensor::cat(&[&sliced, &zero], 3)?;
    }
    // sx == 0: no horizontal shift.

    Ok(out)
}

// =============================================================================
// manual_conv2d_backward — gradient computation for manual_conv2d
// =============================================================================
//
// Given:
//   x       — the forward input, shape (B, C_in, H, W)
//   w       — the kernel weights, shape (C_out, C_in, 3, 3)
//   delta_y — gradient of the loss w.r.t. the conv output, shape (B, C_out, H, W)
//
// Computes and returns:
//   delta_x — gradient w.r.t. the input x,   shape (B, C_in, H, W)
//   dw      — gradient w.r.t. the weights w,  shape (C_out, C_in, 3, 3)
//
// Algorithm (per kernel position (dy, dx)):
//
//   Weight gradient (dw at position [dy, dx]):
//     dw_slice = delta_y_flat @ x_shift^T
//     where x_shift = shift_and_pad(x, sy=1-dy, sx=1-dx)
//     This is the "correlation" between the input patch and output gradient.
//
//   Input gradient (delta_x at position [dy, dx]):
//     dx_perm = w_slice^T @ delta_y_flat
//     dx_slice = reshape dx_perm, then shift back by (-sy, -sx)
//     Summed over all 9 kernel positions = full-convolution with flipped kernel.
//
// WHY shift_and_pad instead of explicit padding?
//   In the forward pass we zero-padded and then narrowed.  For the backward
//   pass, propagating through `narrow` requires shifting the gradient so that
//   it aligns with the correct position in the input gradient accumulator.
//   shift_and_pad provides exactly this cyclic-with-zero-fill operation.
fn manual_conv2d_backward(
    x: &Tensor,
    w: &Tensor,
    delta_y: &Tensor,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let (b, c_in, h, w_img) = x.dims4()?;
    let (c_out, _, _, _) = w.dims4()?;

    // Accumulator for the input gradient.
    let mut delta_x = Tensor::zeros((b, c_in, h, w_img), DType::F32, device)?;

    // Collect weight gradient slices for each (dy, dx) position.
    let mut dw_dy_list = Vec::with_capacity(3);

    for dy in 0..3 {
        let mut dw_dx_list = Vec::with_capacity(3);

        // sy, sx are the shift offsets that reverse the forward narrowing.
        // In the forward pass, position (dy, dx) reads from (dy..dy+H, dx..dx+W)
        // of the padded input.  The corresponding backward shift is (1-dy, 1-dx).
        let sy = 1 - (dy as i32);

        for dx in 0..3 {
            let sx = 1 - (dx as i32);

            // --- Weight gradient at kernel position (dy, dx) -----------------
            // Shift x so that the correct input patch aligns with delta_y.
            let x_slice = shift_and_pad(x, sy, sx, device)?;
            let x_flat = x_slice.reshape((b, c_in, h * w_img))?;
            // x_perm shape: (C_in, B*H*W)
            let x_perm = x_flat.permute((1, 0, 2))?.reshape((c_in, b * h * w_img))?;

            // delta_y_flat shape: (C_out, B*H*W)
            let delta_out_slice = delta_y
                .reshape((b, c_out, h * w_img))?
                .permute((1, 0, 2))?
                .reshape((c_out, b * h * w_img))?;

            // dw_slice = delta_y_flat @ x_perm^T  → (C_out, C_in)
            // Reshape to (C_out, C_in, 1, 1) so we can cat along kH and kW dims.
            let dw_slice = delta_out_slice
                .matmul(&x_perm.t()?)?
                .reshape((c_out, c_in, 1, 1))?;
            dw_dx_list.push(dw_slice);

            // --- Input gradient at kernel position (dy, dx) ------------------
            // dx_perm = w_slice^T @ delta_y_flat  → (C_in, B*H*W)
            let w_slice = w
                .narrow(2, dy, 1)?
                .narrow(3, dx, 1)?
                .reshape((c_out, c_in))?;
            let dx_perm = w_slice.t()?.matmul(&delta_out_slice)?;

            // Reshape to (B, C_in, H, W).
            let dx_slice = dx_perm
                .reshape((c_in, b, h * w_img))?
                .permute((1, 0, 2))?
                .reshape((b, c_in, h, w_img))?;

            // Reverse the shift to align with the unpadded input coordinates.
            let dx_shifted = shift_and_pad(&dx_slice, -sy, -sx, device)?;
            delta_x = delta_x.add(&dx_shifted)?;
        }

        // Concatenate the three dx weight slices along the kW dimension.
        let dw_dy = Tensor::cat(
            &[&dw_dx_list[0], &dw_dx_list[1], &dw_dx_list[2]],
            3,
        )?;
        dw_dy_list.push(dw_dy);
    }

    // Concatenate the three kH rows of weight gradients → (C_out, C_in, 3, 3).
    let dw = Tensor::cat(&dw_dy_list, 2)?;

    Ok((delta_x, dw))
}

// =============================================================================
// DenoisingModel implementation for SimpleDenoisingCNN
// =============================================================================
//
// Implementing the `DenoisingModel` trait allows `SimpleDenoisingCNN` to work
// with the shared `sample_ddpm_cfg` sampler and `MlpAdamOptimizer` without any
// code changes — the trait provides a stable interface across architectures.
impl DenoisingModel for SimpleDenoisingCNN {
    // =========================================================================
    // forward — CNN noise prediction pass
    // =========================================================================
    //
    // Input `x`: concatenated tensor of shape (B, img_dim + cond_dim).
    //   x[:, 0:img_dim]           = x_t (noisy image, flat)
    //   x[:, img_dim:img_dim+cond_dim] = conditioning (time_emb + class_one_hot)
    //
    // Steps:
    //   1. Split v → x_t (flat), cond_vec (flat).
    //   2. Project cond_vec → cond_map (spatial).
    //   3. Reshape x_t → spatial grid, cat with cond_map → 2-channel input.
    //   4. Conv1 (2→16 channels) + Leaky-ReLU.
    //   5. Conv2 (16→1 channel) + reshape to flat output.
    //
    // Returns:
    //   pred          — predicted noise ε̂, shape (B, img_dim)
    //   intermediates — [input_cat, z1, a1] cached for the backward pass
    //
    // WHY cache input_cat, z1, a1?
    //   The backward pass needs the layer inputs to compute weight gradients.
    //   Rather than recomputing them, we cache exactly what backward() requires.
    fn forward(&self, x: &Tensor) -> Result<(Tensor, Vec<Tensor>)> {
        let device = x.device();
        let b = x.dim(0)?;

        // Compute the spatial height/width from img_dim (assumes square images).
        // For MNIST: sqrt(784) = 28, so h = w = 28.
        let h = (self.img_dim as f64).sqrt() as usize;
        let w_img = h;

        // --- Step 1: Split input into image and conditioning parts -----------
        // x_t: the noisy image, shape (B, 784).
        let xt = x.narrow(1, 0, self.img_dim)?;
        // cond_vec: time embedding + class one-hot, shape (B, 26).
        let cond_vec = x.narrow(1, self.img_dim, self.cond_dim)?;

        // --- Step 2: Project conditioning vector → spatial conditioning map --
        // cond_vec @ w_cond^T → (B, img_dim=784) + bias → reshape (B, 1, 28, 28)
        //
        // WHY project to img_dim (784) and then reshape?
        //   We want the conditioning map to have the same spatial resolution as
        //   the noisy image so we can concatenate them channel-wise.  A linear
        //   projection from 26 → 784 followed by reshape is the simplest way to
        //   produce a (B, 1, 28, 28) conditioning grid.
        let cond_map = cond_vec
            .matmul(&self.w_cond.t()?)?
            .broadcast_add(&self.b_cond)?
            .reshape((b, 1, h, w_img))?;

        // --- Step 3: Reshape x_t to spatial + channel-concatenate ------------
        // xt_img: (B, 1, 28, 28)
        // input_cat: (B, 2, 28, 28) — channel 0: image, channel 1: conditioning
        let xt_img = xt.reshape((b, 1, h, w_img))?;
        let input_cat = Tensor::cat(&[&xt_img, &cond_map], 1)?;

        // --- Step 4: Conv1 + Leaky-ReLU --------------------------------------
        // z1: pre-activation, shape (B, 16, 28, 28)
        // a1: post-activation using max(z, 0.01*z) = Leaky-ReLU with alpha=0.01
        //
        // WHY Leaky-ReLU instead of ReLU?
        //   Standard ReLU zeroes out all negative values, which can cause "dead
        //   neurons" — units that permanently output 0 because their gradient
        //   also becomes 0.  Leaky-ReLU passes a small fraction (0.01×) of the
        //   negative input, keeping gradient flow alive even for negative units.
        let z1 = manual_conv2d(&input_cat, &self.w1, Some(&self.b1), &device)?;
        let a1 = z1.maximum(&z1.affine(0.01, 0.0)?)?;

        // --- Step 5: Conv2 → noise prediction --------------------------------
        // z2: shape (B, 1, 28, 28)
        // pred: flattened noise prediction, shape (B, 784)
        //
        // No activation after Conv2: the noise prediction ε̂ can be any real
        // value (positive or negative), matching the Gaussian noise target.
        let z2 = manual_conv2d(&a1, &self.w2, Some(&self.b2), &device)?;
        let pred = z2.reshape((b, self.img_dim))?;

        // Cache [input_cat, z1, a1] for backward — order must match backward().
        let intermediates = vec![input_cat, z1, a1];
        Ok((pred, intermediates))
    }

    // =========================================================================
    // backward — manual chain-rule gradient computation
    // =========================================================================
    //
    // Computes gradients of the MSE loss:
    //   L = (1 / (2 * B * img_dim)) * ||pred - target||²
    //
    // with respect to all 6 parameters: w_cond, b_cond, w1, b1, w2, b2.
    //
    // Chain rule path (back-to-front):
    //
    //   1. Loss → pred:
    //      delta_pred = (pred - target) * (2 / (B * img_dim)) ← MSE gradient
    //
    //   2. pred → z2 (reshape):
    //      delta_z2 = delta_pred reshaped → (B, 1, 28, 28)
    //
    //   3. z2 → w2, b2, a1 (Conv2 backward):
    //      db2      = sum(delta_z2, over B, H, W)
    //      dw2, delta_a1 = manual_conv2d_backward(a1, w2, delta_z2)
    //
    //   4. a1 → z1 (Leaky-ReLU backward):
    //      relu_grad = (z1 >= 0) * 0.99 + 0.01   [Leaky-ReLU derivative]
    //      delta_z1  = delta_a1 * relu_grad
    //
    //   5. z1 → w1, b1, input_cat (Conv1 backward):
    //      db1             = sum(delta_z1, over B, H, W)
    //      dw1, delta_input_cat = manual_conv2d_backward(input_cat, w1, delta_z1)
    //
    //   6. input_cat → cond_map (select channel 1):
    //      delta_cond_map = delta_input_cat[:, 1:2, :, :]   (the cond channel)
    //
    //   7. cond_map → w_cond, b_cond (Linear projection backward):
    //      delta_cond_flat = delta_cond_map reshaped → (B, img_dim)
    //      db_cond = sum(delta_cond_flat, over B)
    //      dw_cond = delta_cond_flat^T @ cond_vec
    //
    // Returns: [dw_cond, db_cond, dw1, db1, dw2, db2]
    //          (same order as params() / param_names())
    fn backward(
        &self,
        v: &Tensor,
        intermediates: &[Tensor],
        pred: &Tensor,
        target: &Tensor,
    ) -> Result<Vec<Tensor>> {
        // Sanity check: we expect exactly the 3 tensors cached by forward().
        if intermediates.len() != 3 {
            bail!(
                "SimpleDenoisingCNN expected 3 cached intermediates from forward(), got {}",
                intermediates.len()
            );
        }

        let device = v.device();
        let b = v.dim(0)?;
        let h = (self.img_dim as f64).sqrt() as usize;
        let w_img = h;

        // Unpack cached intermediates (must match the order in forward()).
        let input_cat = &intermediates[0]; // (B, 2, 28, 28)
        let z1 = &intermediates[1];        // (B, 16, 28, 28) pre-activation
        let a1 = &intermediates[2];        // (B, 16, 28, 28) post-activation

        // --- Step 1: MSE gradient ∂L/∂pred -----------------------------------
        // For L = (1/(2*N)) * ||pred - target||^2, the gradient is:
        //   ∂L/∂pred = (pred - target) / N    where N = B * img_dim
        //
        // The 2 in the denominator of the forward pass cancels with the 2 in
        // the squared difference's derivative.  Here we use scale = 2/N to
        // fold in the factor from the MSE expansion.
        let scale = 2.0 / (b * self.img_dim) as f64;
        let delta_pred = pred.sub(target)?.affine(scale, 0.0)?;

        // --- Step 2: Reshape gradient back to spatial shape ------------------
        let delta_z2 = delta_pred.reshape((b, 1, h, w_img))?;

        // --- Step 3: Conv2 backward ------------------------------------------
        // db2: sum over all spatial positions and batch items.
        // The bias gradient is the sum of the output gradient over B, H, W.
        let db2 = delta_z2.sum(0)?.sum(1)?.sum(1)?;

        // dw2:       weight gradient, shape (1, 16, 3, 3)
        // delta_a1:  gradient flowing back to the post-activation of Conv1
        let (delta_a1, dw2) = manual_conv2d_backward(a1, &self.w2, &delta_z2, &device)?;

        // --- Step 4: Leaky-ReLU backward -------------------------------------
        // Leaky-ReLU derivative:
        //   f'(z) = 1.0   if z >= 0   (passes gradient through)
        //           0.01  if z <  0   (leaks a small fraction)
        //
        // Implemented as: (z >= 0) * 0.99 + 0.01
        //   When z >= 0: 1 * 0.99 + 0.01 = 1.00 ✓
        //   When z <  0: 0 * 0.99 + 0.01 = 0.01 ✓
        //
        // WHY 0.99 and 0.01? They add up to 1.0 for z >= 0 and give exactly
        // alpha=0.01 for z < 0, matching the forward pass `affine(0.01, 0.0)`.
        let relu_grad = z1.ge(0.0f32)?.to_dtype(DType::F32)?.affine(0.99, 0.01)?;
        let delta_z1 = delta_a1.mul(&relu_grad)?;

        // --- Step 5: Conv1 backward ------------------------------------------
        // db1: sum over B, H, W for each of the 16 output channels.
        let db1 = delta_z1.sum(0)?.sum(1)?.sum(1)?;

        // dw1:            weight gradient, shape (16, 2, 3, 3)
        // delta_input_cat: gradient flowing into the 2-channel input
        let (delta_input_cat, dw1) =
            manual_conv2d_backward(input_cat, &self.w1, &delta_z1, &device)?;

        // --- Step 6: Select gradient for the conditioning channel only -------
        // input_cat has 2 channels: [channel 0 = xt_img, channel 1 = cond_map].
        // We only need the gradient w.r.t. the conditioning map (channel 1).
        // The gradient w.r.t. xt_img (channel 0) is not used — we don't need to
        // propagate back into the input image, only into the model parameters.
        let delta_cond_map = delta_input_cat.narrow(1, 1, 1)?;

        // --- Step 7: Linear projection backward ------------------------------
        // delta_cond_flat: (B, img_dim=784)
        let delta_cond_flat = delta_cond_map.reshape((b, self.img_dim))?;

        // db_cond: sum over the batch dimension → (img_dim,)
        let db_cond = delta_cond_flat.sum(0)?;

        // dw_cond = delta_cond_flat^T @ cond_vec → (img_dim, cond_dim)
        // This is the outer-product gradient for a linear layer:
        //   W has shape (img_dim, cond_dim), so its gradient is
        //   (B, img_dim)^T × (B, cond_dim) = (img_dim, cond_dim).
        let cond_vec = v.narrow(1, self.img_dim, self.cond_dim)?;
        let dw_cond = delta_cond_flat.t()?.matmul(&cond_vec)?;

        // Return all gradients in the same order as params() / param_names().
        Ok(vec![dw_cond, db_cond, dw1, db1, dw2, db2])
    }

    // =========================================================================
    // params — ordered immutable references to all trainable tensors
    // =========================================================================
    //
    // The optimizer calls this once to initialise per-parameter Adam state
    // (moment vectors m and v with the same shape as each parameter).
    // Order must match the gradient output of backward() and param_names().
    fn params(&self) -> Vec<&Tensor> {
        vec![
            &self.w_cond,
            &self.b_cond,
            &self.w1,
            &self.b1,
            &self.w2,
            &self.b2,
        ]
    }

    // =========================================================================
    // param_names — human-readable labels for gradient-norm logging
    // =========================================================================
    //
    // The training binary zips these names with the gradient tensors produced
    // by backward() to print per-layer gradient norms.  Same order as params().
    fn param_names(&self) -> Vec<&str> {
        vec!["w_cond", "b_cond", "w1", "b1", "w2", "b2"]
    }

    // =========================================================================
    // params_mut — ordered mutable references for in-place parameter updates
    // =========================================================================
    //
    // The Adam optimizer calls this to apply weight updates:
    //   for (p, g) in params_mut().zip(grads) { *p = *p - lr * adam(g) }
    // Must be in the same order as params() and backward().
    fn params_mut(&mut self) -> Vec<&mut Tensor> {
        vec![
            &mut self.w_cond,
            &mut self.b_cond,
            &mut self.w1,
            &mut self.b1,
            &mut self.w2,
            &mut self.b2,
        ]
    }
}
