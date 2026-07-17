use anyhow::Result;
use candle_core::{DType, Device, Tensor};

use super::sampling::{sample_ddpm_cfg, sample_ddpm_cfg_from_timestep_with_callback};
use super::{BetaScheduler, DenoisingModel};

// A zero predictor isolates sampler algebra from learned model behavior. If a
// test fails here, the schedule/reverse loop is responsible—not U-Net weights.
struct ZeroNoiseModel {
    image_size: usize,
}

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
