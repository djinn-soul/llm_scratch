use anyhow::Result;
use candle_core::{Device, Tensor};
use llm_scratch_rs::models::diffusion::{
    get_time_embedding, BetaScheduler, DenoisingModel, SimpleDenoisingMlp,
};

#[test]
fn test_beta_scheduler() -> Result<()> {
    let device = &Device::Cpu;
    let steps = 100;
    let scheduler = BetaScheduler::new(steps, 1e-4, 0.02, device)?;

    assert_eq!(scheduler.steps, steps);
    assert_eq!(scheduler.betas.dims(), &[steps]);
    assert_eq!(scheduler.alphas.dims(), &[steps]);
    assert_eq!(scheduler.alphas_cumprod.dims(), &[steps]);
    assert_eq!(scheduler.alphas_cumprod_prev.dims(), &[steps]);
    assert_eq!(scheduler.sqrt_alphas_cumprod.dims(), &[steps]);
    assert_eq!(scheduler.sqrt_one_minus_alphas_cumprod.dims(), &[steps]);
    assert_eq!(scheduler.sigmas.dims(), &[steps]);

    // Check that forward noising works
    let batch_size = 8;
    let x0 = Tensor::randn(0.0f32, 1.0f32, (batch_size, 2), device)?;
    let noise = Tensor::randn(0.0f32, 1.0f32, (batch_size, 2), device)?;
    let t = Tensor::new(&[0u32, 1, 10, 20, 50, 80, 90, 99], device)?;

    let xt = scheduler.add_noise(&x0, &noise, &t)?;
    assert_eq!(xt.dims(), &[batch_size, 2]);

    Ok(())
}

#[test]
fn test_time_embedding() -> Result<()> {
    let device = &Device::Cpu;
    let t = Tensor::new(&[0u32, 5, 20], device)?;
    let emb_dim = 16;
    let emb = get_time_embedding(&t, emb_dim)?;

    assert_eq!(emb.dims(), &[3, emb_dim]);
    Ok(())
}

#[test]
fn test_mlp_forward_backward_update() -> Result<()> {
    let device = &Device::Cpu;
    let batch_size = 4;
    let in_dim = 18; // 2 (coords) + 16 (time embedding)
    let hidden_dim = 32;
    let out_dim = 2;

    let mut mlp = SimpleDenoisingMlp::new(in_dim, hidden_dim, out_dim, device)?;

    // Dummy inputs and targets
    let v = Tensor::randn(0.0f32, 1.0f32, (batch_size, in_dim), device)?;
    let target = Tensor::randn(0.0f32, 1.0f32, (batch_size, out_dim), device)?;

    // Forward pass (through the trait).
    let (pred, intermediates) = DenoisingModel::forward(&mlp, &v)?;
    assert_eq!(pred.dims(), &[batch_size, out_dim]);
    // The MLP implementation packs two intermediates: a1 and z1.
    assert_eq!(intermediates.len(), 2);
    assert_eq!(intermediates[0].dims(), &[batch_size, hidden_dim]); // a1
    assert_eq!(intermediates[1].dims(), &[batch_size, hidden_dim]); // z1

    // Save initial weights
    let initial_w1 = mlp.w1.clone();

    // Backward pass (through the trait).
    let grads = DenoisingModel::backward(&mlp, &v, &intermediates, &pred, &target)?;
    // The MLP returns 4 gradient tensors: dw1, db1, dw2, db2.
    assert_eq!(grads.len(), 4);
    assert_eq!(grads[0].dims(), &[hidden_dim, in_dim]); // dw1
    assert_eq!(grads[1].dims(), &[hidden_dim]); // db1
    assert_eq!(grads[2].dims(), &[out_dim, hidden_dim]); // dw2
    assert_eq!(grads[3].dims(), &[out_dim]); // db2

    // Update weights via the legacy concrete method.
    let legacy_grads = llm_scratch_rs::models::diffusion::Gradients {
        dw1: grads[0].clone(),
        db1: grads[1].clone(),
        dw2: grads[2].clone(),
        db2: grads[3].clone(),
    };
    mlp.update(&legacy_grads, 0.1, batch_size)?;

    // Verify weights actually changed
    let diff = mlp
        .w1
        .sub(&initial_w1)?
        .sqr()?
        .sum_all()?
        .to_scalar::<f32>()?;
    assert!(diff > 0.0);

    Ok(())
}
