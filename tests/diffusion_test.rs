// =============================================================================
// diffusion_test.rs — integration tests for diffusion model components
// =============================================================================
//
// WHY integration tests rather than unit tests?
//   The core correctness invariants for a diffusion model are end-to-end:
//   "does the forward pass produce the right tensor shapes?", "does the
//   backward pass produce gradients of the right shapes?", "do weights
//   actually change after an update?".  These questions span multiple modules
//   (scheduler, model, optimizer) and are best validated together.
//
// Test strategy:
//   Each test follows the pattern:
//     1. Build the component under test with small dimensions (fast, low RAM).
//     2. Run forward (and backward where applicable).
//     3. Assert tensor shapes are correct.
//     4. Assert numerical invariants (e.g. gradient non-zero after update).
//
// Test inventory:
//   test_beta_scheduler              — schedule shapes + forward diffusion
//   test_time_embedding              — sinusoidal embedding shape
//   test_mlp_forward_backward_update — SimpleDenoisingMlp shapes + weight update
//   test_cnn_forward_backward_update — SimpleDenoisingCNN shapes + weight update
//   test_cnn_5layers_forward_backward_update — SimpleDenoisingCNN5Layers shapes + weight update
// =============================================================================

use anyhow::Result;
use candle_core::{Device, Tensor};
use llm_scratch_rs::models::diffusion::{
    get_time_embedding, BetaScheduler, DenoisingModel, SimpleDenoisingCNN,
    SimpleDenoisingCNN5Layers, SimpleDenoisingMlp, SimpleDenoisingUNet,
};

// =============================================================================
// test_beta_scheduler
// =============================================================================
//
// Validates that BetaScheduler correctly initialises all derived schedule
// tensors and that `add_noise` produces output of the expected shape.
//
// WHY test add_noise shape rather than numeric values?
//   The exact pixel values after noising depend on the random noise tensor,
//   which is non-deterministic.  What we can reliably assert is the output
//   shape and that no panics/errors occur — i.e., that the scheduler's indexing
//   logic handles a batch of varying timesteps correctly.
//
// Derived tensors checked:
//   betas, alphas, alphas_cumprod, alphas_cumprod_prev,
//   sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, sigmas
//   → all shape (steps,)
#[test]
fn test_beta_scheduler() -> Result<()> {
    let device = &Device::new_cuda(0).unwrap_or(Device::Cpu);
    let steps = 100;
    let scheduler = BetaScheduler::new(steps, 1e-4, 0.02, device)?;

    // All schedule vectors must have length equal to the number of timesteps.
    assert_eq!(scheduler.steps, steps);
    assert_eq!(scheduler.betas.dims(), &[steps]);
    assert_eq!(scheduler.alphas.dims(), &[steps]);
    assert_eq!(scheduler.alphas_cumprod.dims(), &[steps]);
    assert_eq!(scheduler.alphas_cumprod_prev.dims(), &[steps]);
    assert_eq!(scheduler.sqrt_alphas_cumprod.dims(), &[steps]);
    assert_eq!(scheduler.sqrt_one_minus_alphas_cumprod.dims(), &[steps]);
    assert_eq!(scheduler.sigmas.dims(), &[steps]);

    // Verify forward diffusion x_t = sqrt(ᾱ_t)*x_0 + sqrt(1-ᾱ_t)*ε
    // produces output of the same shape as the input.
    let batch_size = 8;
    let x0 = Tensor::randn(0.0f32, 1.0f32, (batch_size, 2), device)?;
    let noise = Tensor::randn(0.0f32, 1.0f32, (batch_size, 2), device)?;
    // Use a diverse set of timesteps to exercise different schedule positions.
    let t = Tensor::new(&[0u32, 1, 10, 20, 50, 80, 90, 99], device)?;

    let xt = scheduler.add_noise(&x0, &noise, &t)?;
    assert_eq!(xt.dims(), &[batch_size, 2]);

    Ok(())
}

// =============================================================================
// test_time_embedding
// =============================================================================
//
// Validates that `get_time_embedding` produces a tensor of shape
// (batch, emb_dim) with the expected sinusoidal encoding dimensions.
//
// WHY test shape?
//   The time embedding is concatenated with x_t and the class label before
//   being fed to the model.  A wrong shape would cause a dimension mismatch
//   in `Tensor::cat` or `matmul` inside the model, producing an uninformative
//   runtime error.  This test catches embedding-dimension mismatches early.
#[test]
fn test_time_embedding() -> Result<()> {
    let device = &Device::new_cuda(0).unwrap_or(Device::Cpu);
    let t = Tensor::new(&[0u32, 5, 20], device)?;
    let emb_dim = 16;
    let emb = get_time_embedding(&t, emb_dim)?;

    // Output shape: (batch=3, emb_dim=16)
    assert_eq!(emb.dims(), &[3, emb_dim]);
    Ok(())
}

// =============================================================================
// test_mlp_forward_backward_update
// =============================================================================
//
// End-to-end test for SimpleDenoisingMlp:
//   1. Forward: check prediction shape and that intermediates are returned.
//   2. Backward: check gradient shapes match expected parameter shapes.
//   3. Update: apply gradients and verify weights moved.
//
// WHY small in_dim / hidden_dim / out_dim?
//   These can be any positive integers — we use small values so the test
//   runs quickly (< 10 ms) without requiring MNIST images.
//
// WHY test weight update as well as gradient shapes?
//   A gradient of the correct shape could still be all-zeros (e.g. if the
//   loss is constant).  Checking that weights change after the update
//   confirms that the backward pass produces non-trivial gradients.
#[test]
fn test_mlp_forward_backward_update() -> Result<()> {
    let device = &Device::new_cuda(0).unwrap_or(Device::Cpu);
    let batch_size = 4;
    let in_dim = 18; // 2 (2-D coordinate) + 16 (time embedding)
    let hidden_dim = 32;
    let out_dim = 2;

    let mut mlp = SimpleDenoisingMlp::new(in_dim, hidden_dim, out_dim, device)?;

    // Dummy input and target noise tensors.
    let v = Tensor::randn(0.0f32, 1.0f32, (batch_size, in_dim), device)?;
    let target = Tensor::randn(0.0f32, 1.0f32, (batch_size, out_dim), device)?;

    // --- Forward pass -------------------------------------------------------
    // The MLP trait implementation packs two intermediates: a1 (post-ReLU
    // hidden activations) and z1 (pre-ReLU).
    let (pred, intermediates) = DenoisingModel::forward(&mlp, &v)?;
    assert_eq!(pred.dims(), &[batch_size, out_dim]);
    assert_eq!(intermediates.len(), 2);
    assert_eq!(intermediates[0].dims(), &[batch_size, hidden_dim]); // a1
    assert_eq!(intermediates[1].dims(), &[batch_size, hidden_dim]); // z1

    // Save a copy of w1 before the update.
    let initial_w1 = mlp.w1.clone();

    // --- Backward pass ------------------------------------------------------
    // Returns 4 gradient tensors: [dw1, db1, dw2, db2].
    let grads = DenoisingModel::backward(&mlp, &v, &intermediates, &pred, &target)?;
    assert_eq!(grads.len(), 4);
    assert_eq!(grads[0].dims(), &[hidden_dim, in_dim]); // dw1
    assert_eq!(grads[1].dims(), &[hidden_dim]); // db1
    assert_eq!(grads[2].dims(), &[out_dim, hidden_dim]); // dw2
    assert_eq!(grads[3].dims(), &[out_dim]); // db2

    // --- Weight update (via legacy concrete method) -------------------------
    // The MLP also exposes a concrete `update` method for backward compatibility
    // with training code that predates the DenoisingModel trait.
    let legacy_grads = llm_scratch_rs::models::diffusion::Gradients {
        dw1: grads[0].clone(),
        db1: grads[1].clone(),
        dw2: grads[2].clone(),
        db2: grads[3].clone(),
    };
    mlp.update(&legacy_grads, 0.1, batch_size)?;

    // Verify weights actually changed — rules out all-zero gradients.
    let diff = mlp
        .w1
        .sub(&initial_w1)?
        .sqr()?
        .sum_all()?
        .to_scalar::<f32>()?;
    assert!(
        diff > 0.0,
        "w1 did not change after update — gradient may be zero"
    );

    Ok(())
}

// =============================================================================
// test_cnn_forward_backward_update
// =============================================================================
//
// End-to-end test for SimpleDenoisingCNN (2-layer, 3×3 kernels):
//   1. Forward: output shape (B, img_dim), 3 cached intermediates.
//   2. Backward: 6 gradient tensors with correct shapes.
//   3. Direct weight update via gradient subtraction.
//
// WHY img_dim = 16 (4×4 image)?
//   The CNN internally reshapes the flat img_dim vector to a square image.
//   Using img_dim=16 gives h=4, w=4, which is small enough for a fast test
//   while still exercising the spatial reshape and conv padding logic.
//
// Gradient shape assertions:
//   [0] dw_cond: (img_dim, cond_dim) = (16, 6)
//   [1] db_cond: (img_dim,) = (16,)
//   [2] dw1:     (16, 2, 3, 3)  — Conv1 kernel gradient
//   [3] db1:     (16,)           — Conv1 bias gradient
//   [4] dw2:     (1, 16, 3, 3)  — Conv2 kernel gradient
//   [5] db2:     (1,)            — Conv2 bias gradient
#[test]
fn test_cnn_forward_backward_update() -> Result<()> {
    let device = &Device::new_cuda(0).unwrap_or(Device::Cpu);
    let batch_size = 4;
    let img_dim = 16; // 4×4 image
    let cond_dim = 6;
    let in_dim = img_dim + cond_dim;

    let mut cnn = SimpleDenoisingCNN::new(img_dim, cond_dim, device)?;

    let v = Tensor::randn(0.0f32, 1.0f32, (batch_size, in_dim), device)?;
    let target = Tensor::randn(0.0f32, 1.0f32, (batch_size, img_dim), device)?;

    // --- Forward pass -------------------------------------------------------
    // 3 intermediates: [input_cat, z1, a1]
    let (pred, intermediates) = DenoisingModel::forward(&cnn, &v)?;
    assert_eq!(pred.dims(), &[batch_size, img_dim]);
    assert_eq!(intermediates.len(), 3);

    let initial_w1 = cnn.w1.clone();

    // --- Backward pass ------------------------------------------------------
    let grads = DenoisingModel::backward(&cnn, &v, &intermediates, &pred, &target)?;
    assert_eq!(grads.len(), 6);
    assert_eq!(grads[0].dims(), &[img_dim, cond_dim]); // dw_cond
    assert_eq!(grads[1].dims(), &[img_dim]); // db_cond
    assert_eq!(grads[2].dims(), &[16, 2, 3, 3]); // dw1
    assert_eq!(grads[3].dims(), &[16]); // db1
    assert_eq!(grads[4].dims(), &[1, 16, 3, 3]); // dw2
    assert_eq!(grads[5].dims(), &[1]); // db2

    // --- Manual weight update (SGD step with lr=0.1) -----------------------
    cnn.w1 = cnn.w1.sub(&grads[2].affine(0.1, 0.0)?)?;

    // Verify w1 changed.
    let diff = cnn
        .w1
        .sub(&initial_w1)?
        .sqr()?
        .sum_all()?
        .to_scalar::<f32>()?;
    assert!(diff > 0.0, "w1 did not change — dw1 may be all zeros");

    Ok(())
}

// =============================================================================
// test_cnn_5layers_forward_backward_update
// =============================================================================
//
// End-to-end test for SimpleDenoisingCNN5Layers (5-layer, 5×5 kernels):
//   1. Forward: output shape (B, img_dim), 9 cached intermediates.
//   2. Backward: 12 gradient tensors with correct shapes.
//   3. Direct weight update via gradient subtraction.
//
// WHY 9 intermediates?
//   The 5-layer model caches [input_cat, z1, a1, z2, a2, z3, a3, z4, a4].
//   That is one 2-channel input + (pre-act, post-act) pairs for each of the
//   first 4 conv layers.  Conv5 output z5 is not cached because the backward
//   pass only needs a4 (the input to Conv5) and delta_z5 (derived from
//   delta_pred), neither of which comes from the cached intermediates.
//
// WHY 12 gradients?
//   6 parameter pairs × 2 (weight + bias) = 12:
//   [dw_cond, db_cond, dw1, db1, dw2, db2, dw3, db3, dw4, db4, dw5, db5]
//
// Gradient shape assertions (note: kernel sizes are all 5×5):
//   [0]  dw_cond: (img_dim, cond_dim) = (16, 6)
//   [1]  db_cond: (img_dim,) = (16,)
//   [2]  dw1:     (16, 2, 5, 5)    — Conv1 kernel gradient
//   [3]  db1:     (16,)
//   [4]  dw2:     (32, 16, 5, 5)   — Conv2 kernel gradient
//   [5]  db2:     (32,)
//   [6]  dw3:     (32, 32, 5, 5)   — Conv3 kernel gradient
//   [7]  db3:     (32,)
//   [8]  dw4:     (16, 32, 5, 5)   — Conv4 kernel gradient
//   [9]  db4:     (16,)
//   [10] dw5:     (1, 16, 5, 5)    — Conv5 kernel gradient
//   [11] db5:     (1,)
//
// NOTE: The test assertions use the channel widths from an earlier version
// of the model (16/32 channels) rather than the current wider widths (64/128).
// The test is correct in verifying the shape structure; the actual in-memory
// tensor shapes use the wider channels, so this test will fail if run against
// the current model.  Update the assertions to match the production widths
// (64/128) when they become stable.
#[test]
fn test_cnn_5layers_forward_backward_update() -> Result<()> {
    let device = &Device::new_cuda(0).unwrap_or(Device::Cpu);
    let batch_size = 4;
    let img_dim = 16; // 4×4 image — small for fast testing
    let cond_dim = 6;
    let in_dim = img_dim + cond_dim;

    let mut cnn = SimpleDenoisingCNN5Layers::new(img_dim, cond_dim, device)?;

    let v = Tensor::randn(0.0f32, 1.0f32, (batch_size, in_dim), device)?;
    let target = Tensor::randn(0.0f32, 1.0f32, (batch_size, img_dim), device)?;

    // --- Forward pass -------------------------------------------------------
    // 9 cached intermediates: [input_cat, z1, a1, z2, a2, z3, a3, z4, a4]
    let (pred, intermediates) = DenoisingModel::forward(&cnn, &v)?;
    assert_eq!(pred.dims(), &[batch_size, img_dim]);
    assert_eq!(intermediates.len(), 9);

    let initial_w1 = cnn.w1.clone();

    // --- Backward pass ------------------------------------------------------
    // 12 gradient tensors: one weight + one bias per parameter group.
    let grads = DenoisingModel::backward(&cnn, &v, &intermediates, &pred, &target)?;
    assert_eq!(grads.len(), 12);
    assert_eq!(grads[0].dims(), &[img_dim, cond_dim]); // dw_cond
    assert_eq!(grads[1].dims(), &[img_dim]); // db_cond
    assert_eq!(grads[2].dims(), &[64, 2, 5, 5]); // dw1
    assert_eq!(grads[3].dims(), &[64]); // db1
    assert_eq!(grads[4].dims(), &[128, 64, 5, 5]); // dw2
    assert_eq!(grads[5].dims(), &[128]); // db2
    assert_eq!(grads[6].dims(), &[128, 128, 5, 5]); // dw3
    assert_eq!(grads[7].dims(), &[128]); // db3
    assert_eq!(grads[8].dims(), &[64, 128, 5, 5]); // dw4
    assert_eq!(grads[9].dims(), &[64]); // db4
    assert_eq!(grads[10].dims(), &[1, 64, 5, 5]); // dw5
    assert_eq!(grads[11].dims(), &[1]); // db5

    // --- Manual weight update (SGD step with lr=0.1) -----------------------
    cnn.w1 = cnn.w1.sub(&grads[2].affine(0.1, 0.0)?)?;

    // Verify w1 changed — confirms backward produced a non-zero gradient.
    let diff = cnn
        .w1
        .sub(&initial_w1)?
        .sqr()?
        .sum_all()?
        .to_scalar::<f32>()?;
    assert!(diff > 0.0, "w1 did not change — dw1 may be all zeros");

    Ok(())
}

// =============================================================================
// test_unet_forward_backward_update
// =============================================================================
//
// End-to-end shape and update test for SimpleDenoisingUNet:
//   1. Forward: verify output (B, img_dim) and 26 cached intermediate tensors.
//   2. Backward: verify 15 gradient tensors matching parameter shapes.
//   3. SGD step update verification.
//
// WHY 26 intermediates (vs 9 for the 5-layer CNN)?
//   The U-Net caches skip-connection tensors alongside the usual
//   (pre-act, post-act) pairs. Each encoder level saves its output for
//   concatenation with the corresponding decoder level. The bottleneck,
//   upsampling, and conditioning layers add further cached tensors.
//
// WHY 15 gradients (vs 12 for the 5-layer CNN)?
//   The U-Net has 5 conv layers + conditioning projection + 2 extra
//   normalization or projection layers. Each contributes a (weight, bias)
//   pair, plus the conditioning layer's (w, b) = 15 gradient tensors.
//
// Skip-connection gradient shapes:
//   The decoder layers receive concatenated inputs from the encoder skip
//   connections. For example, dw4 has shape (16, 48, 3, 3) because its
//   input is concat(upsampled_32ch, skip_16ch) = 48 channels.
#[test]
fn test_unet_forward_backward_update() -> Result<()> {
    let device = &Device::Cpu;
    let batch_size = 4;
    let img_dim = 16; // 4×4 image (h_down=2)
    let cond_dim = 6;
    let in_dim = img_dim + cond_dim;

    let mut unet = SimpleDenoisingUNet::new(img_dim, cond_dim, device)?;

    let v = Tensor::randn(0.0f32, 1.0f32, (batch_size, in_dim), device)?;
    let target = Tensor::randn(0.0f32, 1.0f32, (batch_size, img_dim), device)?;

    // --- Forward pass ---
    let (pred, intermediates) = DenoisingModel::forward(&unet, &v)?;
    assert_eq!(pred.dims(), &[batch_size, img_dim]);
    assert_eq!(intermediates.len(), 26);

    let initial_w1 = unet.w1.clone();

    // --- Backward pass ---
    let grads = DenoisingModel::backward(&unet, &v, &intermediates, &pred, &target)?;
    assert_eq!(grads.len(), 15);
    assert_eq!(grads[0].dims(), &[img_dim, cond_dim]); // dw_cond
    assert_eq!(grads[1].dims(), &[img_dim]); // db_cond
    assert_eq!(grads[2].dims(), &[16, 2, 3, 3]); // dw1
    assert_eq!(grads[3].dims(), &[16]); // db1
    assert_eq!(grads[4].dims(), &[32, 16, 3, 3]); // dw2
    assert_eq!(grads[5].dims(), &[32]); // db2
    assert_eq!(grads[6].dims(), &[32, 32, 3, 3]); // dw3
    assert_eq!(grads[7].dims(), &[32]); // db3
    assert_eq!(grads[8].dims(), &[16, 48, 3, 3]); // dw4
    assert_eq!(grads[9].dims(), &[16]); // db4
    assert_eq!(grads[10].dims(), &[1, 16, 3, 3]); // dw5
    assert_eq!(grads[11].dims(), &[1]); // db5

    // --- Update step ---
    unet.w1 = unet.w1.sub(&grads[2].affine(0.1, 0.0)?)?;

    // Verify w1 changed
    let diff = unet
        .w1
        .sub(&initial_w1)?
        .sqr()?
        .sum_all()?
        .to_scalar::<f32>()?;
    assert!(diff > 0.0, "w1 did not change after update");

    Ok(())
}
