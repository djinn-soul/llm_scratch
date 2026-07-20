// =============================================================================
// sampling_tests.rs — sampler-level tests for DDPM and DDIM reverse loops
// =============================================================================
//
// Test strategy:
//   All tests use a ZeroNoiseModel that always predicts epsilon = 0. This
//   isolates the sampler's schedule arithmetic and reverse-loop logic from
//   learned model weights. If a test fails here, the bug is in the sampler
//   math (schedule coefficients, loop bounds, respacing) — never in the
//   U-Net or training.
//
// Test inventory:
//   DDPM:
//     - cfg_sampling_respects_requested_start_timestep — callback count
//     - cosine_cfg_sampling_stays_finite_through_timestep_zero — NaN guard
//   DDIM:
//     - ddim_cfg_sampling_respects_requested_start_timestep — callback count
//     - ddim_cfg_sampling_stays_finite_through_timestep_zero — NaN guard
//     - ddim_zero_noise_collapses_to_clamped_x0_at_final_step — x0 recovery
//     - ddim_strided_sampling_uses_requested_step_count — stride accuracy
//     - ddim_strided_sampling_stays_finite — strided NaN guard
//     - ddim_full_resolution_wrapper_matches_full_strided_sequence — API parity
// =============================================================================

use anyhow::Result;
use candle_core::{DType, Device, Tensor};

use super::sampling::{
    sample_ddim_cfg_from_timestep_with_call_back, sample_ddim_cfg_strided_with_call_back,
    sample_ddpm_cfg, sample_ddpm_cfg_from_timestep_with_callback,
    sample_ddpm_cfg_strided_with_callback,
};
use super::{BetaScheduler, DenoisingModel};

// A zero predictor: always predicts epsilon = 0 for every pixel.
// WHY? This isolates sampler algebra from learned model behavior. If a
// test fails here, the schedule/reverse loop is responsible — not U-Net weights.
// With epsilon=0, the reverse update reduces to a known closed-form that we
// can verify analytically (e.g. x0_hat = xt / sqrt(alpha_bar_t)).
struct ZeroNoiseModel {
    image_size: usize,
}

// Verify that starting CFG sampling at an explicit timestep produces exactly
// the right number of callback invocations (start_timestep + 1 steps).
#[test]
fn cfg_sampling_respects_requested_start_timestep() -> Result<()> {
    let device = &Device::Cpu;
    let image_size = 16;
    let scheduler = BetaScheduler::new_cosine(8, device)?;
    let model = ZeroNoiseModel { image_size };
    let initial_noise = Tensor::zeros((1, image_size), DType::F32, device)?;
    let class = Tensor::zeros((1, 2), DType::F32, device)?;
    let mut callback_count = 0;

    // Starting at t=6 is inclusive, so callbacks must observe seven reverse
    // states: t=6,5,4,3,2,1,0 mapped to frame indices 0 through 6.
    sample_ddpm_cfg_from_timestep_with_callback(
        &model,
        &scheduler,
        initial_noise,
        6,
        image_size,
        2,
        &class,
        1.0,
        device,
        |frame_index, _| {
            assert_eq!(frame_index, callback_count);
            callback_count += 1;
            Ok(())
        },
    )?;
    assert_eq!(callback_count, 7);
    Ok(())
}

impl DenoisingModel for ZeroNoiseModel {
    fn forward(&self, input: &Tensor) -> Result<(Tensor, Vec<Tensor>)> {
        // Predict epsilon=0 for every pixel. No intermediates are required
        // because inference tests never call backward.
        Ok((
            Tensor::zeros((input.dim(0)?, self.image_size), DType::F32, input.device())?,
            Vec::new(),
        ))
    }

    fn backward(
        &self,
        _input: &Tensor,
        _intermediates: &[Tensor],
        _prediction: &Tensor,
        _target: &Tensor,
    ) -> Result<Vec<Tensor>> {
        Ok(Vec::new())
    }

    fn params(&self) -> Vec<&Tensor> {
        Vec::new()
    }

    fn params_mut(&mut self) -> Vec<&mut Tensor> {
        Vec::new()
    }

    fn param_names(&self) -> Vec<&str> {
        Vec::new()
    }
}

// Regression test for the cosine schedule boundary at t=0.
// The former cosine schedule had beta_0=0, alpha_bar_0=1, producing 0/sqrt(0)
// in the last reverse step. This test runs the FULL reverse chain to confirm
// no NaN/Inf values leak into the final output.
#[test]
fn cosine_cfg_sampling_stays_finite_through_timestep_zero() -> Result<()> {
    let device = &Device::Cpu;
    let image_size = 16;
    let scheduler = BetaScheduler::new_cosine(8, device)?;
    let model = ZeroNoiseModel { image_size };
    let initial_noise = Tensor::zeros((1, image_size), DType::F32, device)?;
    let class = Tensor::zeros((1, 2), DType::F32, device)?;

    // The former cosine boundary used beta_0=0 and alpha_bar_0=1, producing
    // 0/sqrt(0) in the last reverse update. Traversing the complete chain is
    // the regression test; checking only schedule construction is insufficient.
    let generated = sample_ddpm_cfg(
        &model,
        &scheduler,
        initial_noise,
        image_size,
        2,
        &class,
        3.0,
        device,
    )?;
    let values = generated.flatten_all()?.to_vec1::<f32>()?;
    assert!(values.iter().all(|value| value.is_finite()));
    Ok(())
}

// DDIM callback-count test, analogous to the DDPM one above.
//
// HISTORY: DDIM had no coverage at all before this. A prior version silently
// used `(1.0 - alpha_bar_t.sqrt())` instead of `(1.0 - alpha_bar_t).sqrt()`
// in the x0 reconstruction, and the reverse loop skipped denoising at
// t=start_timestep entirely. Both were wrong but produced finite,
// plausible-looking tensors, so nothing caught them without a dedicated test.
#[test]
fn ddim_cfg_sampling_respects_requested_start_timestep() -> Result<()> {
    let device = &Device::Cpu;
    let image_size = 16;
    let scheduler = BetaScheduler::new_cosine(8, device)?;
    let model = ZeroNoiseModel { image_size };
    let initial_noise = Tensor::zeros((1, image_size), DType::F32, device)?;
    let class = Tensor::zeros((1, 2), DType::F32, device)?;
    let mut callback_count = 0;

    // Starting at t=6 is inclusive, so callbacks must observe seven reverse
    // states: t=6,5,4,3,2,1,0 mapped to frame indices 0 through 6.
    sample_ddim_cfg_from_timestep_with_call_back(
        &model,
        &scheduler,
        initial_noise,
        6,
        image_size,
        2,
        &class,
        1.0,
        device,
        |frame_index, _| {
            assert_eq!(frame_index, callback_count);
            callback_count += 1;
            Ok(())
        },
    )?;
    assert_eq!(callback_count, 7);
    Ok(())
}

// DDIM finiteness check across the full reverse chain.
// Ensures the deterministic DDIM path (eta=0) doesn't produce NaN or Inf
// at any point, especially the cosine schedule boundary near t=0.
#[test]
fn ddim_cfg_sampling_stays_finite_through_timestep_zero() -> Result<()> {
    let device = &Device::Cpu;
    let image_size = 16;
    let scheduler = BetaScheduler::new_cosine(8, device)?;
    let model = ZeroNoiseModel { image_size };
    let initial_noise = Tensor::ones((1, image_size), DType::F32, device)?;
    let class = Tensor::zeros((1, 2), DType::F32, device)?;

    let generated = sample_ddim_cfg_from_timestep_with_call_back(
        &model,
        &scheduler,
        initial_noise,
        7,
        image_size,
        2,
        &class,
        3.0,
        device,
        |_, _| Ok(()),
    )?;
    let values = generated.flatten_all()?.to_vec1::<f32>()?;
    assert!(values.iter().all(|value| value.is_finite()));
    Ok(())
}

// With epsilon=0, x0_hat = xt / sqrt(alpha_bar_t), clamped to [-1, 1]. At the
// final step (t=0), alpha_bar_prev=1.0 and the direction term vanishes, so
// the deterministic DDIM output must collapse exactly onto that clamped x0
// estimate rather than drifting from an off-by-one in the reverse loop.
#[test]
fn ddim_zero_noise_collapses_to_clamped_x0_at_final_step() -> Result<()> {
    let device = &Device::Cpu;
    let image_size = 4;
    let scheduler = BetaScheduler::new_cosine(8, device)?;
    let model = ZeroNoiseModel { image_size };
    let initial_noise = Tensor::zeros((1, image_size), DType::F32, device)?;
    let class = Tensor::zeros((1, 2), DType::F32, device)?;

    let generated = sample_ddim_cfg_from_timestep_with_call_back(
        &model,
        &scheduler,
        initial_noise,
        0,
        image_size,
        2,
        &class,
        1.0,
        device,
        |_, _| Ok(()),
    )?;
    let values = generated.flatten_all()?.to_vec1::<f32>()?;
    // xt=0 and epsilon=0 imply x0_hat=0, so the single t=0 step should leave
    // the tensor at exactly zero.
    assert!(values.iter().all(|value| value.abs() < 1e-6));
    Ok(())
}

// Strided DDIM step count test.
// Verifies that with num_inference_steps=20 and a 100-step schedule, the model
// is evaluated exactly 20 times (not 100), and callback frame indices run from
// 0 to 19. This is the core guarantee of strided/respaced sampling.
#[test]
fn ddim_strided_sampling_uses_requested_step_count() -> Result<()> {
    let device = &Device::Cpu;
    let image_size = 16;
    let scheduler = BetaScheduler::new_cosine(100, device)?;
    let model = ZeroNoiseModel { image_size };
    let initial_noise = Tensor::zeros((1, image_size), DType::F32, device)?;
    let class = Tensor::zeros((1, 2), DType::F32, device)?;
    let mut callback_count = 0;

    sample_ddim_cfg_strided_with_call_back(
        &model,
        &scheduler,
        initial_noise,
        99,
        20,
        image_size,
        2,
        &class,
        1.0,
        device,
        |frame_index, _| {
            assert_eq!(frame_index, callback_count);
            callback_count += 1;
            Ok(())
        },
    )?;
    assert_eq!(callback_count, 20);
    Ok(())
}

// Strided DDIM finiteness check.
// Strides skip timesteps, which changes the alpha_bar ratio used in the
// update. This ensures the respaced coefficients don't produce NaN/Inf
// even with aggressive striding (10 steps over a 100-step schedule).
#[test]
fn ddim_strided_sampling_stays_finite() -> Result<()> {
    let device = &Device::Cpu;
    let image_size = 16;
    let scheduler = BetaScheduler::new_cosine(100, device)?;
    let model = ZeroNoiseModel { image_size };
    let initial_noise = Tensor::ones((1, image_size), DType::F32, device)?;
    let class = Tensor::zeros((1, 2), DType::F32, device)?;

    let generated = sample_ddim_cfg_strided_with_call_back(
        &model,
        &scheduler,
        initial_noise,
        99,
        10,
        image_size,
        2,
        &class,
        3.0,
        device,
        |_, _| Ok(()),
    )?;
    let values = generated.flatten_all()?.to_vec1::<f32>()?;
    assert!(values.iter().all(|value| value.is_finite()));
    Ok(())
}

// API equivalence regression test.
// The full-resolution DDIM wrapper delegates to the strided variant with
// num_inference_steps = start_timestep + 1. This test verifies that both
// paths produce identical outputs, guarding against regressions in the
// delegation refactor.
#[test]
fn ddim_full_resolution_wrapper_matches_full_strided_sequence() -> Result<()> {
    let device = &Device::Cpu;
    let image_size = 8;
    let scheduler = BetaScheduler::new_cosine(8, device)?;
    let model = ZeroNoiseModel { image_size };
    let class = Tensor::zeros((1, 2), DType::F32, device)?;

    let via_wrapper = sample_ddim_cfg_from_timestep_with_call_back(
        &model,
        &scheduler,
        Tensor::ones((1, image_size), DType::F32, device)?,
        7,
        image_size,
        2,
        &class,
        1.5,
        device,
        |_, _| Ok(()),
    )?
    .flatten_all()?
    .to_vec1::<f32>()?;

    let via_strided = sample_ddim_cfg_strided_with_call_back(
        &model,
        &scheduler,
        Tensor::ones((1, image_size), DType::F32, device)?,
        7,
        8,
        image_size,
        2,
        &class,
        1.5,
        device,
        |_, _| Ok(()),
    )?
    .flatten_all()?
    .to_vec1::<f32>()?;

    for (a, b) in via_wrapper.iter().zip(via_strided.iter()) {
        assert!((a - b).abs() < 1e-6);
    }
    Ok(())
}

// Respaced DDPM had no test coverage either: the respacing math (synthetic
// beta/alpha derived from the alpha_bar ratio across a skipped jump) is easy
// to get subtly wrong (e.g. dividing the wrong way, or forgetting sigma
// collapses to exactly 0 at the final entry).
#[test]
fn ddpm_strided_sampling_uses_requested_step_count() -> Result<()> {
    let device = &Device::Cpu;
    let image_size = 16;
    let scheduler = BetaScheduler::new_cosine(100, device)?;
    let model = ZeroNoiseModel { image_size };
    let initial_noise = Tensor::zeros((1, image_size), DType::F32, device)?;
    let class = Tensor::zeros((1, 2), DType::F32, device)?;
    let mut callback_count = 0;

    sample_ddpm_cfg_strided_with_callback(
        &model,
        &scheduler,
        initial_noise,
        99,
        20,
        image_size,
        2,
        &class,
        1.0,
        device,
        |frame_index, _| {
            assert_eq!(frame_index, callback_count);
            callback_count += 1;
            Ok(())
        },
    )?;
    assert_eq!(callback_count, 20);
    Ok(())
}

#[test]
fn ddpm_strided_sampling_stays_finite() -> Result<()> {
    let device = &Device::Cpu;
    let image_size = 16;
    let scheduler = BetaScheduler::new_cosine(100, device)?;
    let model = ZeroNoiseModel { image_size };
    let initial_noise = Tensor::ones((1, image_size), DType::F32, device)?;
    let class = Tensor::zeros((1, 2), DType::F32, device)?;

    let generated = sample_ddpm_cfg_strided_with_callback(
        &model,
        &scheduler,
        initial_noise,
        99,
        10,
        image_size,
        2,
        &class,
        3.0,
        device,
        |_, _| Ok(()),
    )?;
    let values = generated.flatten_all()?.to_vec1::<f32>()?;
    assert!(values.iter().all(|value| value.is_finite()));
    Ok(())
}

// A single-step request (num_inference_steps=1) has exactly one subsequence
// entry, so alpha_bar_prev=1.0 and sigma=0 with no randn call — fully
// deterministic. With epsilon=0 the posterior mean's x0 term is xt/sqrt(abar)
// clamped, and its xt term coefficient must vanish (alpha_bar_prev=1 makes
// the (1-alpha_bar_prev) factor zero), so zero input must map to exactly zero.
#[test]
fn ddpm_strided_single_step_zero_noise_collapses_to_zero() -> Result<()> {
    let device = &Device::Cpu;
    let image_size = 4;
    let scheduler = BetaScheduler::new_cosine(8, device)?;
    let model = ZeroNoiseModel { image_size };
    let initial_noise = Tensor::zeros((1, image_size), DType::F32, device)?;
    let class = Tensor::zeros((1, 2), DType::F32, device)?;
    let mut callback_count = 0;

    let generated = sample_ddpm_cfg_strided_with_callback(
        &model,
        &scheduler,
        initial_noise,
        7,
        1,
        image_size,
        2,
        &class,
        1.0,
        device,
        |_, _| {
            callback_count += 1;
            Ok(())
        },
    )?;
    assert_eq!(callback_count, 1);
    let values = generated.flatten_all()?.to_vec1::<f32>()?;
    assert!(values.iter().all(|value| value.abs() < 1e-6));
    Ok(())
}
