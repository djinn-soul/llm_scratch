// Finite-difference validation of SimpleDenoisingUNet's hand-written backward().
//
// The existing UNet test asserts gradient *shapes* only, so an error in the
// backward math produces correctly-shaped, silently wrong gradients — training
// still runs, it just learns the wrong thing. This test checks gradient
// *values* against a central difference of the actual loss.
//
// The parameters below are chosen to cover the bottleneck attention residual
// specifically:
//   - w_cond / w2 / w3 sit UPSTREAM of the residual, so their gradients flow
//     through the combined (attention + skip) path.
//   - attn_w_q / attn_w_v sit inside the attention branch.
// An error in either half of the residual split shows up here.

use anyhow::Result;
use candle_core::{Device, Tensor};
use llm_scratch_rs::common::parameterized::Parameterized;
use llm_scratch_rs::models::diffusion::{DenoisingModel, SimpleDenoisingUNet};

const IMG_DIM: usize = 16; // 4x4 image -> h_down = 2, attention length n = 4
const COND_DIM: usize = 6;
const BATCH: usize = 4;

/// The exact loss that `backward()` differentiates:
///   L = sum((pred - target)^2) / (batch * img_dim)
fn loss_of(unet: &SimpleDenoisingUNet, v: &Tensor, target: &Tensor) -> Result<f64> {
    let (pred, _) = DenoisingModel::forward(unet, v)?;
    let (b, d) = pred.dims2()?;
    let sq = pred.sub(target)?.sqr()?.sum_all()?.to_scalar::<f32>()? as f64;
    Ok(sq / (b * d) as f64)
}

/// Central difference dL/dparam[idx], restoring the parameter afterwards.
fn numeric_grad(
    unet: &SimpleDenoisingUNet,
    name: &str,
    idx: usize,
    v: &Tensor,
    target: &Tensor,
    device: &Device,
    eps: f32,
) -> Result<f64> {
    let original = unet.get(name)?;
    let dims = original.dims().to_vec();
    let mut values = original.flatten_all()?.to_vec1::<f32>()?;
    let saved = values[idx];

    // `Tensor::from_vec` allocates fresh storage, which `set_param` requires —
    // candle rejects writing a variable from a tensor derived from its own value.
    values[idx] = saved + eps;
    unet.set_param(
        name,
        &Tensor::from_vec(values.clone(), dims.clone(), device)?,
    )?;
    let plus = loss_of(unet, v, target)?;

    values[idx] = saved - eps;
    unet.set_param(
        name,
        &Tensor::from_vec(values.clone(), dims.clone(), device)?,
    )?;
    let minus = loss_of(unet, v, target)?;

    values[idx] = saved;
    unet.set_param(name, &Tensor::from_vec(values, dims, device)?)?;

    Ok((plus - minus) / (2.0 * eps as f64))
}

#[test]
fn unet_backward_matches_finite_difference_through_attention_residual() -> Result<()> {
    let device = &Device::Cpu;
    let unet = SimpleDenoisingUNet::new(IMG_DIM, COND_DIM, device)?;

    let v = Tensor::randn(0.0f32, 1.0f32, (BATCH, IMG_DIM + COND_DIM), device)?;
    let target = Tensor::randn(0.0f32, 1.0f32, (BATCH, IMG_DIM), device)?;

    let (pred, intermediates) = DenoisingModel::forward(&unet, &v)?;
    let grads = DenoisingModel::backward(&unet, &v, &intermediates, &pred, &target)?;

    let names = unet.param_names();

    // (parameter, flat index) pairs spanning both sides of the residual split.
    let checks = [
        ("w_cond", 0usize),
        ("w2", 5),
        ("w3", 11),
        ("attn_w_q", 3),
        ("attn_w_v", 7),
    ];

    let eps = 2e-3f32;
    for (name, idx) in checks {
        let grad_idx = names
            .iter()
            .position(|n| *n == name)
            .unwrap_or_else(|| panic!("unknown parameter {name}"));
        let analytic = grads[grad_idx].flatten_all()?.to_vec1::<f32>()?[idx] as f64;
        let numeric = numeric_grad(&unet, name, idx, &v, &target, device, eps)?;

        // Relative comparison with an absolute floor: gradients here span
        // several orders of magnitude, and f32 finite differences carry noise
        // that a pure relative bound would flag on the near-zero entries.
        let tolerance = 5e-2 * analytic.abs().max(numeric.abs()) + 2e-4;
        assert!(
            (analytic - numeric).abs() <= tolerance,
            "{name}[{idx}]: analytic={analytic:.8}, numeric={numeric:.8}, tol={tolerance:.8}"
        );
    }

    Ok(())
}
