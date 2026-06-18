use anyhow::{bail, Result};
use candle_core::{DType, Device, Tensor};
use flate2::read::GzDecoder;
use llm_scratch_rs::models::diffusion::{
    get_time_embedding, BetaScheduler, MlpAdamOptimizer, SimpleDenoisingMlp,
};
use std::fs::{create_dir_all, File};
use std::io::{BufReader, Read};
use std::path::Path;

/// Download and load MNIST labels (IDX1 format)
fn load_mnist_labels(path: &str) -> Result<Vec<u8>> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic)?;
    let magic_num = u32::from_be_bytes(magic);
    if magic_num != 2049 {
        bail!("Invalid magic number for MNIST labels: {}", magic_num);
    }
    let mut meta = [0u8; 4];
    reader.read_exact(&mut meta)?;
    let num_items = u32::from_be_bytes(meta) as usize;
    let mut buffer = vec![0u8; num_items];
    reader.read_exact(&mut buffer)?;
    Ok(buffer)
}

/// Download and load MNIST images
fn load_mnist_images(path: &str, device: &Device) -> Result<Tensor> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
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
    let mut buffer = vec![0u8; num_images * rows * cols];
    reader.read_exact(&mut buffer)?;
    let tensor = Tensor::from_vec(buffer, (num_images, rows * cols), device)?
        .to_dtype(DType::F32)?
        .affine(1.0 / 127.5, -1.0)?;
    Ok(tensor)
}
/// Helper that downloads both binary files if missing and returns them
fn acquire_mnist(device: &Device) -> Result<(Tensor, Vec<u8>)> {
    let img_path = "mnist/MNIST/raw/train-images-idx3-ubyte";
    let lbl_path = "mnist/MNIST/raw/train-labels-idx1-ubyte";
    let dest_dir = Path::new("mnist/MNIST/raw");
    if !Path::new(img_path).exists() {
        create_dir_all(dest_dir)?;
        let url = "https://raw.githubusercontent.com/fgnt/mnist/master/train-images-idx3-ubyte.gz";
        println!("Downloading images from: {}", url);
        let response = reqwest::blocking::get(url)?;
        let mut gz_decoder = GzDecoder::new(response);
        let mut out_file = File::create(img_path)?;
        std::io::copy(&mut gz_decoder, &mut out_file)?;
    }
    if !Path::new(lbl_path).exists() {
        create_dir_all(dest_dir)?;
        let url = "https://raw.githubusercontent.com/fgnt/mnist/master/train-labels-idx1-ubyte.gz";
        println!("Downloading labels from: {}", url);
        let response = reqwest::blocking::get(url)?;
        let mut gz_decoder = GzDecoder::new(response);
        let mut out_file = File::create(lbl_path)?;
        std::io::copy(&mut gz_decoder, &mut out_file)?;
    }
    let images = load_mnist_images(img_path, device)?;
    let labels = load_mnist_labels(lbl_path)?;
    Ok((images, labels))
}

/// Helper to convert a list of labels (e.g. [3, 7]) into a one-hot Tensor of shape [N, 10]
fn make_one_hot(labels: &[u8], device: &Device) -> Result<Tensor> {
    let mut one_hot_vec = Vec::with_capacity(labels.len() * 10);
    for &label in labels {
        let mut row = vec![0.0f32; 10];
        if (label as usize) < 10 {
            row[label as usize] = 1.0;
        }
        one_hot_vec.extend_from_slice(&row);
    }
    Ok(Tensor::from_vec(one_hot_vec, (labels.len(), 10), device)?)
}

fn main() -> Result<()> {
    let device = Device::Cpu;
    println!("==Class-Conditioned DDPM MNIST Model Training");

    let (images, labels_raw) = acquire_mnist(&device)?;
    let total_samples = images.dim(0)?;
    println!("Total samples: {}", total_samples);

    // Hyperparameters
    let steps = 100;
    let time_emb_dim = 16;
    let class_dim = 10; // 10 digit classes (0-9)
    let img_dim = 784;
    let hidden_dim = 512;
    let batch_size = 128;
    let epochs = 8000;
    let lr = 0.001;

    let scheduler = BetaScheduler::new(steps, 0.0001, 0.02, &device)?;

    // We instantiate the model with input size 784 + 16 + 10 = 810
    let mut mlp = SimpleDenoisingMlp::new(
        img_dim + time_emb_dim + class_dim,
        hidden_dim,
        img_dim,
        &device,
    )?;
    let mut optimizer = MlpAdamOptimizer::new(&mlp, lr)?;

    println!("Starting training for {} epochs...\n", epochs);

    let betas_f32 = scheduler.betas.to_vec1::<f32>()?;
    let alphas_f32 = scheduler.alphas.to_vec1::<f32>()?;
    let alphas_cumprod_f32 = scheduler.alphas_cumprod.to_vec1::<f32>()?;
    let sigmas_f32 = scheduler.sigmas.to_vec1::<f32>()?;

    for epoch in 1..=epochs {
        // 1. Sample random batch indices
        let index_tensor = Tensor::rand(
            0.0f32,
            total_samples as f32 - 1e-4f32,
            (batch_size,),
            &device,
        )?
        .to_dtype(DType::U32)?;
        let indices = index_tensor.to_vec1::<u32>()?;

        // 2. Load batch images and matching labels
        let x0 = images.index_select(&index_tensor, 0)?;

        let batch_labels: Vec<u8> = indices
            .iter()
            .map(|&idx| labels_raw[idx as usize])
            .collect();
        let label_one_hot = make_one_hot(&batch_labels, &device)?;

        // 3. Diffusion noising
        let t_float = Tensor::rand(0.0f32, steps as f32 - 1e-4f32, (batch_size,), &device)?;
        let t = t_float.to_dtype(DType::U32)?;
        let noise = Tensor::randn(0.0f32, 1.0f32, (batch_size, img_dim), &device)?;
        let xt = scheduler.add_noise(&x0, &noise, &t)?;

        // 4. Concatenate conditional input: [xt, time_emb, label_one_hot]
        let time_emb = get_time_embedding(&t, time_emb_dim)?;
        let v = Tensor::cat(&[&xt, &time_emb, &label_one_hot], 1)?;

        // 5. Forward, backward, and update
        let (pred, a1, z1) = mlp.forward(&v)?;
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

    println!("\n=== Starting Class-Conditioned Sampling ===");

    // Generate class conditional digit class (Let's generate the digit '3')
    let target_digit = 3u8;
    println!("Generating digit: {}", target_digit);
    let num_samples = 1;
    let mut xt = Tensor::randn(0.0f32, 1.0f32, (num_samples, img_dim), &device)?;

    // Create one-hot code for the target digit
    let target_label_one_hot = make_one_hot(&[target_digit], &device)?;
    for t_step in (0..steps).rev() {
        let t_vec = vec![t_step as u32; num_samples];
        let t_tensor = Tensor::new(t_vec.as_slice(), &device)?;
        let time_emb = get_time_embedding(&t_tensor, time_emb_dim)?;

        // Concatenate [xt, time_emb, target_label_one_hot]
        let v = Tensor::cat(&[&xt, &time_emb, &target_label_one_hot], 1)?;
        let (pred_noise, _, _) = mlp.forward(&v)?;

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
    save_png("mnist_cond_generated.png", &final_pixels)?;
    Ok(())
}
fn save_png(path: &str, image_flat: &[f32]) -> Result<()> {
    let file = File::create(path)?;
    let ref mut w = std::io::BufWriter::new(file);
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
    println!("Saved conditional generated image as PNG to: {}", path);
    Ok(())
}
