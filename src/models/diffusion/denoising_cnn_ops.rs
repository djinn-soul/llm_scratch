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
// Parallelism:
//   All three functions use `rayon` to parallelise the per-kernel-position
//   (dy, dx) work.  For a 5×5 kernel this is 25 independent matmuls per call;
//   rayon schedules these across all available CPU cores.  On GPU, Candle's
//   built-in matmul already runs in parallel, so rayon adds CPU-level
//   parallelism on top.
//
// Kernel-size agnostic:
//   All functions derive padding from the kernel dimensions (kh, kw) rather
//   than hardcoding values, making them work correctly for both 3×3 and 5×5
//   (and any other odd-sized) kernels with "SAME" zero-padding.
// =============================================================================

use anyhow::Result;
use candle_core::{DType, Device, Tensor};

// =============================================================================
// manual_conv2d — zero-padded 2-D convolution implemented via matmul loops
// =============================================================================
//
// WHY implement conv from scratch instead of using candle-nn?
//   `candle-nn` provides Conv2d layers, but using it ties the backward pass
//   to Candle's autograd engine.  Since we implement backward manually (to
//   match the exact intermediate tensors cached in forward), keeping the
//   forward pass as explicit matmuls makes the index math transparent and
//   the forward ↔ backward pairing unambiguous.
//
// Algorithm: sliding-window decomposition into matmuls
//   For each kernel position (dy, dx) in {0..kh} × {0..kw}:
//
//     1. Slice the zero-padded input at this kernel offset:
//          x_slice = x_padded[:, :, dy : dy+H, dx : dx+W]
//          shape   = (B, C_in, H, W)
//
//     2. Reshape to a 2-D matrix for matmul:
//          x_perm  = x_slice.permute(C_in, B, H, W) → reshape(C_in, B*H*W)
//
//     3. Extract the weight sub-matrix for this position:
//          w_slice = w[:, :, dy, dx]  shape (C_out, C_in)
//
//     4. Multiply: out_slice = w_slice @ x_perm  shape (C_out, B*H*W)
//
//     5. Reshape back and accumulate into y: (B, C_out, H, W)
//
// Parallelism (rayon):
//   Each (dy, dx) pair is independent — it produces one additive contribution
//   to the output.  `rayon::flat_map + into_par_iter` schedules all kh×kw
//   matmuls in parallel across CPU threads, then accumulates sequentially.
//   For 3×3 kernels this is 9-way parallelism; for 5×5 kernels, 25-way.
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
    device: &Device,
) -> Result<Tensor> {
    let (b, c_in, h, w_img) = x.dims4()?;
    let (c_out, _, kh, kw) = w.dims4()?;

    // Compute SAME-padding sizes from kernel dimensions.
    let pad_h = (kh - 1) / 2;
    let pad_w = (kw - 1) / 2;

    // --- Zero-pad height axis -----------------------------------------------
    // WHY pad conditionally? For kh=1 (no spatial extent), padding is zero
    // and we skip the Tensor::cat to avoid an unnecessary allocation.
    let x_padded_y = if pad_h > 0 {
        let zero_row = Tensor::zeros((b, c_in, pad_h, w_img), DType::F32, device)?;
        Tensor::cat(&[&zero_row, x, &zero_row], 2)?
    } else {
        x.clone()
    };

    // --- Zero-pad width axis ------------------------------------------------
    let x_padded = if pad_w > 0 {
        let zero_col = Tensor::zeros((b, c_in, h + 2 * pad_h, pad_w), DType::F32, device)?;
        Tensor::cat(&[&zero_col, &x_padded_y, &zero_col], 3)?
    } else {
        x_padded_y
    };

    // --- Sequential per-position matmuls, folded straight into the accumulator --
    //
    // WHY sequential instead of rayon here?
    //   Collecting all kh*kw position outputs into a Vec before summing (the
    //   previous approach) keeps every one of them alive simultaneously — a
    //   kh*kw-fold peak memory multiplier on top of the accumulator itself.
    //   Folding each position straight into `y` as it's computed caps the
    //   extra live memory at one position's worth, regardless of kernel size.
    //   Candle's matmul is already internally parallel (BLAS/cuBLAS), so the
    //   rayon layer here was CPU-thread parallelism over tiny per-position
    //   work, not the dominant cost — trading it away buys back memory
    //   headroom that matters far more on memory-constrained boxes.
    let mut y = Tensor::zeros((b, c_out, h, w_img), DType::F32, device)?;
    for dy in 0..kh {
        for dx in 0..kw {
            // Extract the spatial slice at kernel offset (dy, dx).
            let x_slice = x_padded.narrow(2, dy, h)?.narrow(3, dx, w_img)?;

            // Reshape to (C_in, B*H*W) for the matmul.
            let x_flat = x_slice.reshape((b, c_in, h * w_img))?;
            let x_perm = x_flat.permute((1, 0, 2))?.reshape((c_in, b * h * w_img))?;

            // w_slice: (C_out, C_in) — weights for this kernel position.
            let w_slice = w
                .narrow(2, dy, 1)?
                .narrow(3, dx, 1)?
                .reshape((c_out, c_in))?;

            // out_slice = w_slice @ x_perm  → (C_out, B*H*W)
            // .contiguous() ensures the matmul operates on a dense layout,
            // avoiding potential incorrect results from non-contiguous strides.
            let out_slice = w_slice.contiguous()?.matmul(&x_perm.contiguous()?)?;

            // Reshape (C_out, B*H*W) → (B, C_out, H, W) and fold in immediately.
            let out_reshaped = out_slice
                .reshape((c_out, b, h, w_img))?
                .permute((1, 0, 2, 3))?;
            y = y.add(&out_reshaped)?;
        }
    }

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
    let (b, c, h, w) = t.dims4()?;
    let mut out = t.clone();

    // --- Vertical shift (height axis = dim 2) --------------------------------
    if sy > 0 {
        // Shift down by sy: prepend sy zero rows, drop last sy rows.
        let zero = Tensor::zeros((b, c, sy as usize, w), DType::F32, device)?;
        let sliced = t.narrow(2, 0, h - sy as usize)?; // keep rows [0, H-sy)
        out = Tensor::cat(&[&zero, &sliced], 2)?;
    } else if sy < 0 {
        // Shift up by |sy|: drop first |sy| rows, append |sy| zero rows.
        let abs_sy = sy.unsigned_abs() as usize;
        let zero = Tensor::zeros((b, c, abs_sy, w), DType::F32, device)?;
        let sliced = t.narrow(2, abs_sy, h - abs_sy)?; // keep rows [|sy|, H)
        out = Tensor::cat(&[&sliced, &zero], 2)?;
    }
    // sy == 0: no vertical shift (no allocation).

    // --- Horizontal shift (width axis = dim 3) -------------------------------
    if sx > 0 {
        // Shift right by sx: prepend sx zero columns, drop last sx columns.
        let zero = Tensor::zeros((b, c, h, sx as usize), DType::F32, device)?;
        let sliced = out.narrow(3, 0, w - sx as usize)?;
        out = Tensor::cat(&[&zero, &sliced], 3)?;
    } else if sx < 0 {
        // Shift left by |sx|: drop first |sx| columns, append |sx| zero columns.
        let abs_sx = sx.unsigned_abs() as usize;
        let zero = Tensor::zeros((b, c, h, abs_sx), DType::F32, device)?;
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
// Theory (chain rule through the sliding-window matmul):
//
//   For each kernel position (dy, dx):
//
//   Weight gradient:
//     At forward time, position (dy, dx) computed:
//       out_slice = w_slice @ x_perm,  where x_perm = shift_and_pad(x, sy, sx)
//     By the chain rule:
//       dw_slice = delta_y_flat @ x_perm^T
//     This is the outer product of upstream gradient and the input patch.
//
//   Input gradient:
//     The input contributes to the output at every kernel position, so:
//       dx_perm  = w_slice^T @ delta_y_flat
//       dx_slice = reshape dx_perm, then shift back by (-sy, -sx)
//     Summing over all (dy, dx) positions gives the full input gradient —
//     equivalent to full convolution with the spatially-flipped kernel.
//
// Parallelism:
//   Same rayon parallel strategy as manual_conv2d: kh×kw independent tasks.
//   Results are stored in a flat Vec indexed by (dy * kw + dx) and then
//   assembled into (C_out, C_in, kH, kW) via nested Tensor::cat.
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

    // --- Sequential per-position computation, folded straight into delta_x ---
    //
    // WHY sequential instead of rayon here?
    //   Same reasoning as manual_conv2d's forward: collecting all kh*kw
    //   (dx_shifted, dw_slice) pairs before summing kept every dx_shifted
    //   alive at once — each one full (B, C_in, H, W), the dominant memory
    //   cost of this function. Folding delta_x immediately caps that to one
    //   position's worth. dw_slice stays cheap ((C_out, C_in, 1, 1)) and is
    //   collected into a small Vec for the final cat — that part is unchanged.
    let mut delta_x = Tensor::zeros((b, c_in, h, w_img), DType::F32, device)?;
    let mut dw_dy_list = Vec::with_capacity(kh);
    for dy in 0..kh {
        let mut dw_dx_list = Vec::with_capacity(kw);
        for dx in 0..kw {
            // sy, sx: the backward shift that undoes the forward narrowing.
            //   Forward at (dy, dx) read x_padded[dy..dy+H, dx..dx+W].
            //   Backward shift = (pad_h - dy, pad_w - dx).
            let sy = pad_h - (dy as i32);
            let sx = pad_w - (dx as i32);

            // x_slice: the same input patch that forward saw at (dy, dx).
            let x_slice = shift_and_pad(x, sy, sx, device)?;
            let x_flat = x_slice.reshape((b, c_in, h * w_img))?;
            // x_perm shape: (C_in, B*H*W)
            let x_perm = x_flat.permute((1, 0, 2))?.reshape((c_in, b * h * w_img))?;

            // Flatten delta_y for this position: shape (C_out, B*H*W).
            // We use the same delta_y for all positions (it's the output
            // gradient which doesn't vary with (dy, dx)).
            let delta_out_slice = delta_y
                .reshape((b, c_out, h * w_img))?
                .permute((1, 0, 2))?
                .reshape((c_out, b * h * w_img))?;

            // --- Weight gradient for this (dy, dx) -----------------------
            // dw_slice = delta_y_flat @ x_perm^T  → (C_out, C_in)
            // Reshaped to (C_out, C_in, 1, 1) for assembly into (C_out, C_in, kH, kW).
            let dw_slice = delta_out_slice
                .contiguous()?
                .matmul(&x_perm.t()?.contiguous()?)?
                .reshape((c_out, c_in, 1, 1))?;
            dw_dx_list.push(dw_slice);

            // --- Input gradient for this (dy, dx) ------------------------
            // w_slice: (C_out, C_in) — same kernel slice as in forward.
            let w_slice = w
                .narrow(2, dy, 1)?
                .narrow(3, dx, 1)?
                .reshape((c_out, c_in))?;

            // dx_perm = w_slice^T @ delta_y_flat  → (C_in, B*H*W)
            let dx_perm = w_slice
                .t()?
                .contiguous()?
                .matmul(&delta_out_slice.contiguous()?)?;

            // Reshape to (B, C_in, H, W).
            let dx_slice = dx_perm
                .reshape((c_in, b, h * w_img))?
                .permute((1, 0, 2))?
                .reshape((b, c_in, h, w_img))?;

            // Reverse the forward shift to align with the unpadded input.
            // WHY -sy, -sx? We shifted x forward by (sy, sx) to extract
            // the patch; the inverse shift puts the gradient back in the
            // correct coordinate frame.
            let dx_shifted = shift_and_pad(&dx_slice, -sy, -sx, device)?;
            delta_x = delta_x.add(&dx_shifted)?;
        }
        // Concatenate across the kW dimension for this row.
        let dw_dx_refs: Vec<&Tensor> = dw_dx_list.iter().collect();
        let dw_dy = Tensor::cat(&dw_dx_refs, 3)?;
        dw_dy_list.push(dw_dy);
    }

    // Concatenate all kH rows along dim 2 → final shape (C_out, C_in, kH, kW).
    let dw_dy_refs: Vec<&Tensor> = dw_dy_list.iter().collect();
    let dw = Tensor::cat(&dw_dy_refs, 2)?;

    Ok((delta_x, dw))
}
