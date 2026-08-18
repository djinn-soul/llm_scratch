// =============================================================================
// denoising_cnn_ops.rs — Shared convolution primitives for CNN denoisers
// =============================================================================
//
// WHY a separate ops module?
//   Both `SimpleDenoisingCNN` (2-layer, 3×3 kernels) and
//   `SimpleDenoisingCNN5Layers` (5-layer, 5×5 kernels) need the same
//   forward-convolution and backward-gradient routines.  Extracting them
//   into a single shared module avoids duplication and ensures both models
//   benefit from any future optimisation (e.g. switching to cuDNN) in one place.
//
// What lives here:
//   1. `manual_conv2d`          — zero-padded 2-D convolution, SAME padding.
//   2. `shift_and_pad`          — spatial shift with zero-fill (backward helper).
//   3. `manual_conv2d_backward` — gradient w.r.t. weights and input.
//
// Algorithm — Im2Col:
//   Both forward and backward use the im2col transformation to express
//   convolution as a single matrix multiply. The kernel is flattened to
//   w_col (C_out, C_in * kH * kW) and the input patches are stacked into
//   x_col (B, C_in * kH * kW, H * W). The forward pass is then just
//   w_col @ x_col, yielding (B, C_out, H * W) in one matmul regardless of
//   kernel size.
//
// Kernel-size agnostic:
//   All functions derive padding from the kernel dimensions (kh, kw) rather
//   than hardcoding values, making them work correctly for both 3×3 and 5×5
//   (and any other odd-sized) kernels with "SAME" zero-padding.
// =============================================================================

use anyhow::Result;
use candle_core::{Device, Tensor};

// =============================================================================
// manual_conv2d — zero-padded 2-D convolution via im2col + matmul
// =============================================================================
//
// WHY implement conv from scratch instead of using candle-nn?
//   `candle-nn` provides Conv2d layers, but using it ties the backward pass
//   to Candle's autograd engine.  Since we implement backward manually (to
//   match the exact intermediate tensors cached in forward), keeping the
//   forward pass as explicit matmuls makes the index math transparent and
//   the forward ↔ backward pairing unambiguous.
//
// Algorithm: im2col transformation + single matmul
//
//   1. Zero-pad the input with SAME padding:
//        pad_h = (kh - 1) / 2,  pad_w = (kw - 1) / 2
//
//   2. Extract all kH × kW overlapping patches from the padded input,
//      reshape each to (B, C_in, H*W), and concatenate along the channel
//      dimension to form:
//        x_col  shape (B, C_in * kH * kW, H * W)
//
//   3. Flatten the kernel:
//        w_col  shape (C_out, C_in * kH * kW)
//
//   4. Single batched matmul:
//        y_flat = w_col @ x_col  shape (B, C_out, H * W)
//
//   5. Reshape y_flat to (B, C_out, H, W) and optionally add bias.
//
// Zero-padding ("SAME" mode):
//   pad_h = (kh - 1) / 2   (1 for 3×3, 2 for 5×5)
//   pad_w = (kw - 1) / 2
//   Adds pad_h zero rows top/bottom and pad_w zero columns left/right so that
//   the output spatial size equals the input spatial size (H_out = H_in).
//
// Arguments:
//   x     — input feature map, shape (B, C_in, H, W)
//   w     — kernel weights,    shape (C_out, C_in, kH, kW)
//   bias  — optional bias,     shape (C_out,)
//   device — tensor device
pub fn manual_conv2d(
    x: &Tensor,
    w: &Tensor,
    bias: Option<&Tensor>,
    _device: &Device,
) -> Result<Tensor> {
    let (b, c_in, h, w_img) = x.dims4()?;
    let (c_out, _, kh, kw) = w.dims4()?;

    // Compute SAME-padding sizes from kernel dimensions.
    let pad_h = (kh - 1) / 2;
    let pad_w = (kw - 1) / 2;

    // --- Zero-pad height axis -----------------------------------------------
    // WHY pad conditionally? For kh=1 (no spatial extent), padding is zero
    // and we skip the Tensor::cat to avoid an unnecessary allocation.
    let x_padded = if pad_h > 0 || pad_w > 0 {
        x.pad_with_zeros(2, pad_h, pad_h)?
            .pad_with_zeros(3, pad_w, pad_w)?
    } else {
        x.clone()
    };


    // --- Im2Col Transformation ------------------------------------------------
    // Extract all kernel patches and concatenate them along the channel dimension.
    let mut slices = Vec::with_capacity(kh * kw);
    for dy in 0..kh {
        for dx in 0..kw {
            let x_slice = x_padded.narrow(2, dy, h)?.narrow(3, dx, w_img)?;
            slices.push(x_slice.reshape((b, c_in, h * w_img))?);
        }
    }
    // x_col shape: (B, C_in * kH * kW, H * W)
    let x_col = Tensor::cat(&slices, 1)?;

    // w_col shape: (C_out, C_in * kH * kW)
    let w_col = w.reshape((c_out, c_in * kh * kw))?;

    // y_flat shape: (B, C_out, H * W)
    let y_flat = w_col.broadcast_matmul(&x_col.contiguous()?)?;

    // Reshape back to (B, C_out, H, W)
    let mut y = y_flat.reshape((b, c_out, h, w_img))?;

    // Add bias, broadcast (1, C_out, 1, 1) → (B, C_out, H, W).
    if let Some(bi) = bias {
        y = y.broadcast_add(&bi.reshape((1, c_out, 1, 1))?)?;
    }
    Ok(y)
}

// =============================================================================
// shift_and_pad — spatial shift with zero border fill
// =============================================================================
//
// WHY is this needed in the backward pass?
//   The gradient of a convolution with respect to its input is itself a
//   convolution with the *transposed* (spatially flipped) kernel.  In the
//   sliding-window decomposition, "transposing" amounts to reversing the
//   (dy, dx) offsets: a forward position (dy, dx) corresponds to a backward
//   shift of (pad_h - dy, pad_w - dx).  `shift_and_pad` implements that
//   shift with zero-fill (roll with zero boundary), so the gradient tensor
//   aligns correctly with the unpadded input gradient accumulator.
//
// General form (supports any shift magnitude, not just ±1):
//   sy > 0: shift content down  by sy rows  → sy zero rows prepended at top
//   sy < 0: shift content up    by |sy| rows → |sy| zero rows appended at bottom
//   sx > 0: shift content right by sx cols  → sx zero cols prepended at left
//   sx < 0: shift content left  by |sx| cols → |sx| zero cols appended at right
//
// This is the generalised version of the original ±1-only implementation,
// required to support arbitrary kernel sizes (e.g. pad_h=2 for 5×5 kernels
// leads to shifts in {-2, -1, 0, 1, 2}).
//
// Arguments:
//   t      — input tensor, shape (B, C, H, W)
//   sy, sx — shift amounts along H and W respectively
//   device — device for zero tensors
pub fn shift_and_pad(t: &Tensor, sy: i32, sx: i32, device: &Device) -> Result<Tensor> {
    if sy == 0 && sx == 0 {
        return Ok(t.clone());
    }
    let (b, c, h, w) = t.dims4()?;
    let mut out = t.clone();

    // --- Vertical shift (height axis = dim 2) --------------------------------
    if sy > 0 {
        // Shift down by sy: prepend sy zero rows, drop last sy rows.
        let zero = Tensor::zeros((b, c, sy as usize, w), t.dtype(), device)?;
        let sliced = t.narrow(2, 0, h - sy as usize)?; // keep rows [0, H-sy)
        out = Tensor::cat(&[&zero, &sliced], 2)?;
    } else if sy < 0 {
        // Shift up by |sy|: drop first |sy| rows, append |sy| zero rows.
        let abs_sy = sy.unsigned_abs() as usize;
        let zero = Tensor::zeros((b, c, abs_sy, w), t.dtype(), device)?;
        let sliced = t.narrow(2, abs_sy, h - abs_sy)?; // keep rows [|sy|, H)
        out = Tensor::cat(&[&sliced, &zero], 2)?;
    }
    // sy == 0: no vertical shift (no allocation).

    // --- Horizontal shift (width axis = dim 3) -------------------------------
    if sx > 0 {
        // Shift right by sx: prepend sx zero columns, drop last sx columns.
        let zero = Tensor::zeros((b, c, h, sx as usize), t.dtype(), device)?;
        let sliced = out.narrow(3, 0, w - sx as usize)?;
        out = Tensor::cat(&[&zero, &sliced], 3)?;
    } else if sx < 0 {
        // Shift left by |sx|: drop first |sx| columns, append |sx| zero columns.
        let abs_sx = sx.unsigned_abs() as usize;
        let zero = Tensor::zeros((b, c, h, abs_sx), t.dtype(), device)?;
        let sliced = out.narrow(3, abs_sx, w - abs_sx)?;
        out = Tensor::cat(&[&sliced, &zero], 3)?;
    }
    // sx == 0: no horizontal shift.

    Ok(out)
}

// =============================================================================
// manual_conv2d_backward — gradient computation paired with manual_conv2d
// =============================================================================
//
// Given the cached forward inputs and the upstream gradient (from the next
// layer), computes:
//
//   delta_x — gradient of the loss w.r.t. the conv input x
//              shape: (B, C_in, H, W)
//
//   dw      — gradient of the loss w.r.t. the conv weights w
//              shape: (C_out, C_in, kH, kW)
//
// Algorithm — im2col backward:
//
//   1. Build x_col (B, C_in * kH * kW, H * W) from shifted input patches,
//      mirroring the forward im2col transformation.
//
//   2. Weight gradient:
//        dw = sum_B(delta_y_flat @ x_col^T)  →  reshape to (C_out, C_in, kH, kW)
//      This is the outer product of upstream gradient and input columns,
//      summed over the batch dimension.
//
//   3. Input gradient (col2im):
//        delta_x_col = w_col^T @ delta_y_flat  →  (B, C_in * kH * kW, H * W)
//      Then fold back into (B, C_in, H, W) by extracting each offset's
//      slice and applying the inverse shift via shift_and_pad(-sy, -sx).
//
// Arguments:
//   x       — the forward input, shape (B, C_in, H, W)
//   w       — the kernel weights used in forward, shape (C_out, C_in, kH, kW)
//   delta_y — upstream gradient (from the next layer), shape (B, C_out, H, W)
//   device  — tensor device
pub fn manual_conv2d_backward(
    x: &Tensor,
    w: &Tensor,
    delta_y: &Tensor,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let (b, c_in, h, w_img) = x.dims4()?;
    let (c_out, _, kh, kw) = w.dims4()?;

    // Derive the same SAME-padding as in the forward pass.
    // WHY must this match? shift_and_pad uses (pad_h - dy, pad_w - dx) as
    // the backward shift, which is the arithmetic inverse of the forward
    // narrowing offset.  Using a different pad here would produce incorrect
    // gradient alignment.
    let pad_h = (kh - 1) as i32 / 2;
    let pad_w = (kw - 1) as i32 / 2;

    // --- Im2Col Backward Transformation ---------------------------------------
    // Flatten delta_y: shape (B, C_out, H * W)
    let delta_y_flat = delta_y.reshape((b, c_out, h * w_img))?;

    // 1. Build x_col using zero-padded input slicing (matching forward im2col)
    let x_padded = if pad_h > 0 || pad_w > 0 {
        x.pad_with_zeros(2, pad_h as usize, pad_h as usize)?
            .pad_with_zeros(3, pad_w as usize, pad_w as usize)?
    } else {
        x.clone()
    };

    let mut slices = Vec::with_capacity(kh * kw);
    for dy in 0..kh {
        for dx in 0..kw {
            let x_slice = x_padded.narrow(2, dy, h)?.narrow(3, dx, w_img)?;
            slices.push(x_slice.reshape((b, c_in, h * w_img))?);
        }
    }
    // x_col shape: (B, C_in * kH * kW, H * W)
    let x_col = Tensor::cat(&slices, 1)?;

    // 2. Weight gradient: dw = sum_B(delta_y_flat @ x_col^T)
    // Reshape to 2D (C_out, B * H * W) and (C_in * kH * kW, B * H * W) to compute
    // dw via a single high-throughput 2D GEMM without allocating a 3D batch tensor.
    let delta_y_2d = delta_y_flat
        .transpose(0, 1)?
        .contiguous()?
        .reshape((c_out, b * h * w_img))?;
    let x_col_2d = x_col
        .transpose(0, 1)?
        .contiguous()?
        .reshape((c_in * kh * kw, b * h * w_img))?;
    let dw = delta_y_2d
        .matmul(&x_col_2d.t()?)?
        .reshape((c_out, c_in, kh, kw))?;

    // 3. Input gradient: delta_x_col = w_col^T @ delta_y_flat
    // w_col shape: (C_out, C_in * kH * kW)
    let w_col = w.reshape((c_out, c_in * kh * kw))?;
    // w_col^T @ delta_y_flat -> (B, C_in * kH * kW, H * W)
    let delta_x_col = w_col.t()?.broadcast_matmul(&delta_y_flat.contiguous()?)?;

    // 4. Fold delta_x_col back into delta_x using inverse shifts (Col2Im)
    let mut delta_x = Tensor::zeros((b, c_in, h, w_img), x.dtype(), device)?;
    for dy in 0..kh {
        for dx in 0..kw {
            let idx = dy * kw + dx;
            let sy = pad_h - (dy as i32);
            let sx = pad_w - (dx as i32);

            // Extract the slice corresponding to this offset
            let dx_slice_flat = delta_x_col.narrow(1, idx * c_in, c_in)?; // (B, C_in, H * W)
            let dx_slice = dx_slice_flat.reshape((b, c_in, h, w_img))?;

            // Reverse the shift
            let dx_shifted = shift_and_pad(&dx_slice, -sy, -sx, device)?;
            delta_x = delta_x.add(&dx_shifted)?;
        }
    }

    Ok((delta_x, dw))
}
