// classifiedr free guidance

use core::f32;

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use llm_scratch_rs::models::diffusion::sampling::sample_ddpm_cfg;
use llm_scratch_rs::models::diffusion::{
    get_time_embedding, BetaScheduler, MlpAdamOptimizer, SimpleDenoisingMlp,
};
use llm_scratch_rs::utils::mnist_utils::{acquire_mnist, save_png};
use rand::RngExt;

fn make_one_hot_cfg(labels: &[u8], drop_rate: f32, device: &Device) -> Result<Tensor> {
    let n = labels.len();
    let num_classes = 10;
        let mut rng = rand::rng();

    let mut hot = vec![0.0f32; n * num_classes];
    for (i, &label) in labels.iter().enumerate() {
        if rng.random::<f32>() > drop_rate {
            let idx = (i * num_classes) + label as usize;
            hot[idx] = 1.0f32;
        }
    }
        Ok(Tensor::from_vec(hot, (n, num_classes), device)?)

}

pub fn main() -> Result<()> {
    let device = Device::Cpu;

    println!("==Classifier -Free Guidance (CFG) Sampler==");

    let (images, labels_raw) = acquire_mnist(&device)?;

    let total_samples = images.dim(0)?;
    println!("Total samples: {}", total_samples);
    // Hyperparameters
    let steps = 100;
    let time_emb_dim = 16;
    let class_dim = 10;
    let img_dim = 784;
    let hidden_dim = 512;
    let batch_size = 128;
    let epochs = 12000; // Increased to 12,000 for better convergence
    let lr = 0.001;
    let label_dropout = 0.15; // 15% probability to drop the class label during training

    let scheduler = BetaScheduler::new(steps, 0.0001, 0.02, &device)?;
    let mut mlp = SimpleDenoisingMlp::new(
        img_dim + time_emb_dim + class_dim,
        hidden_dim,
        img_dim,
        &device,
    )?;
    let mut optimizer = MlpAdamOptimizer::new(&mlp, lr)?;
    println!(
        "Starting training for {} epochs with 15% label dropout...\n",
        epochs
    );

    for epoch in 1..=epochs {
       let index_tensor = Tensor::rand(
            0.0f32,
            total_samples as f32 - 1e-4f32,
            (batch_size,),
            &device,
        )?
        .to_dtype(DType::U32)?;
        let indices = index_tensor.to_vec1::<u32>()?;
        let x0 = images.index_select(&index_tensor, 0)?;

        let batch_labels:Vec<u8> = indices.iter().map(|&x| labels_raw[x as usize]).collect();

        let label_one_hot = make_one_hot_cfg(&batch_labels, label_dropout, &device)?;

        let t_float = Tensor::rand(0.0f32,steps as f32 - 1e-4f32, (batch_size,), &device)?;
        let t = t_float.to_dtype(DType::U32)?;
        let noise = Tensor::randn(0.0f32, 1.0f32, (batch_size, img_dim), &device)?;
        let xt = scheduler.add_noise(&x0, &noise, &t)?;

        let time_emb = get_time_embedding(&t, time_emb_dim)?;
        
        let v = Tensor::cat(&[&xt, &time_emb, &label_one_hot], 1)?;

        let (pred,a1,z1) = mlp.forward(&v)?;

        let diff = pred.sub(&noise)?;
        let loss = diff.sqr()?.mean_all()?.to_scalar::<f32>()?;

        let grads = mlp.backward(&v, &a1, &z1, &pred, &noise)?;


        optimizer.step(&mut mlp, &grads)?;
        if epoch % 100 == 0 || epoch == 1 {
            let dw1_norm = grads.dw1.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
            let dw2_norm = grads.dw2.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
            println!(
                "Epoch {:5}/{} - MSE Loss: {:.6} | dw1 norm: {:.4}, dw2 norm: {:.4}",
                epoch, epochs, loss, dw1_norm, dw2_norm
            );
        }

    }
    println!("\n=== Starting Classifier-Free Guided Reverse Sampling ===");
// Generate digit class 3
    let target_digit = 3u8;
    let guidance_scale = 3.0f64; // Guidance scale s = 3.0
    println!("Generating digit: {} with guidance scale: {}", target_digit, guidance_scale);


    let num_samples = 1;
    let initial_noise =  Tensor::randn(0.0f32, 1.0f32, (num_samples, img_dim), &device)?;
        let target_one_hot = make_one_hot_cfg(&[target_digit], 0.0, &device)?;
        // / Run guided reverse sampling
    let generated = sample_ddpm_cfg(
        &mlp,
        &scheduler,
        initial_noise,
        img_dim,
        time_emb_dim,
        &target_one_hot,
        guidance_scale,
        &device,
    )?;
 
     // Save final generated image
    let final_pixels = generated.flatten_all()?.to_vec1::<f32>()?;
    save_png("mnist_cfg_generated.png", &final_pixels)?;
    Ok(())
}
