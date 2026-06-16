use anyhow::{bail, Result};
use candle_core::{DType, Device, Tensor};
use flate2::read::GzDecoder;
use llm_scratch_rs::models::diffusion::{
    get_time_embedding, BetaScheduler, MlpAdamOptimizer, SimpleDenoisingMlp,
};
use std::fs::{create_dir_all, File};
use std::io::{BufReader, Read};
use std::path::Path;
fn acquire_mnit(device: &Device) -> Result<Tensor> {
    let dest_path = "mnist/MNIST/raw/train-images-idx3-ubyte";
    let dest_dir = Path::new("mnist/MNIST/raw");
    if !Path::new(dest_path).exists() {
        println!("MNIST dataset not found locally. Preparing programmatic download...");
        create_dir_all(dest_dir)?;
        // Direct raw download link to fgnt's MNIST mirror on GitHub
        let url = "https://raw.githubusercontent.com/fgnt/mnist/master/train-images-idx3-ubyte.gz";
        println!("Downloading from: {}", url);
        let response = reqwest::blocking::get(url)?;
        if !response.status().is_success() {
            bail!(
                "Failed to download dataset. HTTP Status: {}",
                response.status()
            );
        }
        println!("Decompressing GZIP archive to {}...", dest_path);
        let mut gz_decoder = GzDecoder::new(response);
        let mut out_file = File::create(dest_path)?;
        std::io::copy(&mut gz_decoder, &mut out_file)?;
        println!("Download and extraction complete!");
    }
    load_mnist_images(dest_path, device)
}

/// Custom binary loader for MNIST IDX3-ubyte image file
fn load_mnist_images(path: &str, device: &Device) -> Result<Tensor> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    // 1. Read magic number (4 bytes)
    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic)?;
    let magic_num = u32::from_be_bytes(magic);
    if magic_num != 2051 {
        bail!("Invalid magic number for MNIST images: {}", magic_num);
    }

    let mut meta = [0u8; 12];
    reader.read_exact(&mut meta)?;

    let num_images = u32::from_be_bytes([meta[0], meta[1], meta[2], meta[3]]) as usize;
    let rows = u32::from_be_bytes([meta[4], meta[5], meta[6], meta[7]]) as usize;
    let cols = u32::from_be_bytes([meta[8], meta[9], meta[10], meta[11]]) as usize;
    println!("Loading {} images of size {}x{}...", num_images, rows, cols);

    // 3. Read raw pixel buffer
    let mut buffer = vec![0u8; num_images * rows * cols];
    reader.read_exact(&mut buffer)?;
    let tensor = Tensor::from_vec(buffer, (num_images, 1, rows, cols), device)?
        .to_dtype(DType::F32)?
        .affine(1.0 / 127.5, -1.0)?;
    Ok(tensor)
}

fn main() -> Result<()> {
    let device = Device::Cpu;
    println!("==DDPM MNIST Model Training");

    let dataset = acquire_mnit(&device)?;
    let total_samples = dataset.dim(0)?;
    println!("Total samples: {}", total_samples);

    // Hyperparameters
    // Hyperparameters
    let steps = 100;
    let time_emb_dim = 16;
    let img_dim = 784;
    let hidden_dim = 512;
    let batch_size = 128;
    let epochs = 20000;
    let lr = 0.001;

    let scheduler = BetaScheduler::new(steps, 0.0001, 0.02, &device)?;

    let mut mlp = SimpleDenoisingMlp::new(img_dim + time_emb_dim, hidden_dim, img_dim, &device)?;
    let mut optimizer = MlpAdamOptimizer::new(&mlp, lr)?;

    println!("Scheduler: Linear, {} steps", steps);
    println!(
        "MLP: input_dim={} (784+16), hidden_dim={}, output_dim=784",
        img_dim + time_emb_dim,
        hidden_dim
    );
    println!("Starting training for {} epochs...\n", epochs);

    let betas_f32 = scheduler.betas.to_vec1::<f32>()?;
    let alphas_f32 = scheduler.alphas.to_vec1::<f32>()?;
    let alphas_cumprod_f32 = scheduler.alphas_cumprod.to_vec1::<f32>()?;
    let sigmas_f32 = scheduler.sigmas.to_vec1::<f32>()?;

    for epoch in 1..=epochs {
        // random batch of dataset

        // 1. Randomly sample batch_size indices from the MNIST dataset entirely in Candle
        let index_tensor = Tensor::rand(
            0.0f32,
            total_samples as f32 - 1e-4f32,
            (batch_size,),
            &device,
        )?
        .to_dtype(DType::U32)?;
        let x0 = dataset
            .index_select(&index_tensor, 0)?
            .reshape((batch_size, img_dim))?;

        // random time stpes t in range [0,steps - 1e-4)
        let t_float = Tensor::rand(0.0f32, steps as f32 - 1e-4f32, (batch_size,), &device)?;

        let t = t_float.to_dtype(DType::U32)?;

        // Target Gaussian noise of shape [batch_size, img_dim]
        let noise = Tensor::randn(0.0f32, 1.0f32, (batch_size, img_dim), &device)?;

        // x_t = sqrt(alpha_bar_t) * x_0
        //     + sqrt(1 - alpha_bar_t) * noise

        let xt = scheduler.add_noise(&x0, &noise, &t)?;

        // time embeddings
        let time_emb = get_time_embedding(&t, time_emb_dim)?;

        // concat xt and time embedding
        let v = Tensor::cat(&[&xt, &time_emb], 1)?;

        let (pred, a1, z1) = mlp.forward(&v)?;

        // let mse loss compute
        let diff = pred.sub(&noise)?;

        let loss = diff.sqr()?.mean_all()?.to_scalar::<f32>()?;

        // backpropagation

        let grads = mlp.backward(&v, &a1, &z1, &pred, &noise)?;

        // update weights
        // update weights using Adam optimizer
        optimizer.step(&mut mlp, &grads)?;

        if epoch % 100 == 0 || epoch == 1 {
            let dw1_norm = grads.dw1.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
            let dw2_norm = grads.dw2.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
            println!(
                "Epoch {:4}/{} - MSE Loss: {:.6} | dw1 norm: {:.4}, dw2 norm: {:.4}",
                epoch, epochs, loss, dw1_norm, dw2_norm
            );
        }
    }

    println!("\n=== Starting Reverse Diffusion Sampling (Generation) ===");

    // Save one original image from the dataset as a reference
    let original_sample = dataset
        .index_select(&Tensor::new(&[0u32], &device)?, 0)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    save_png("mnist_original.png", &original_sample)?;

    // Sample 1 image starting from pure Gaussian noise
    let num_samples = 1;
    let mut xt = Tensor::randn(0.0f32, 1.0f32, (num_samples, img_dim), &device)?;

    // Save the starting pure random noise
    save_png("mnist_noisy.png", &xt.flatten_all()?.to_vec1::<f32>()?)?;

    // Denoising loop
    for t_step in (0..steps).rev() {
        let t_vec = vec![t_step as u32; num_samples];
        let t_tensor = Tensor::new(t_vec.as_slice(), &device)?;
        let time_emb = get_time_embedding(&t_tensor, time_emb_dim)?;
        let v = Tensor::cat(&[&xt, &time_emb], 1)?;
        // Predict noise
        let (pred_noise, _, _) = mlp.forward(&v)?;
        // Reverse process formula math
        let beta = betas_f32[t_step];
        let alpha = alphas_f32[t_step];
        let alpha_bar = alphas_cumprod_f32[t_step];
        let sigma = sigmas_f32[t_step];
        let eps_coef = beta / (1.0 - alpha_bar).sqrt();
        let mean = xt
            .sub(&pred_noise.affine(eps_coef as f64, 0.0)?)?
            .affine((1.0 / alpha.sqrt()) as f64, 0.0)?;
        if t_step > 0 {
            let z = Tensor::randn(0.0f32, 1.0f32, (num_samples, img_dim), &device)?;
            xt = mean.add(&z.affine(sigma as f64, 0.0)?)?;
        } else {
            xt = mean;
        }
    }
    let final_pixels = xt.flatten_all()?.to_vec1::<f32>()?;

    // Save the final generated image as a PNG file
    save_png("mnist_generated.png", &final_pixels)?;

    Ok(())
}

/// Save 28x28 grayscale image as a PNG file using the png crate
fn save_png(path: &str, image_flat: &[f32]) -> Result<()> {
    use std::io::BufWriter;
    let file = File::create(path)?;
    let ref mut w = BufWriter::new(file);

    let mut encoder = png::Encoder::new(w, 28, 28);
    encoder.set_color(png::ColorType::Grayscale);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header()?;

    let mut data = vec![0u8; 784];
    for (i, &val) in image_flat.iter().enumerate() {
        let norm = ((val + 1.0) / 2.0).clamp(0.0, 1.0);
        data[i] = (norm * 255.0).round() as u8;
    }

    writer.write_image_data(&data)?;
    println!("Saved generated image as PNG to: {}", path);
    Ok(())
}
