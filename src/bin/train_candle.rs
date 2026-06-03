use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::{loss::cross_entropy, optim::Optimizer, AdamW, ParamsAdamW, VarBuilder, VarMap};
use llm_scratch_rs::common::sampling::sample_next_token;
use llm_scratch_rs::{
    common::{dataloader::DataLoader, lr_scheduler::LRScheduler},
    models::gpt2_candle::{Gpt2Config, Gpt2Model},
    tokenizers::bpe::{download_file_if_not_present, BytePair},
};

fn calc_loss_loader(
    model: &Gpt2Model,
    loader: &DataLoader,
    device: &Device,
    num_batches: usize,
) -> Result<f32> {
    let mut total_loss = 0.0;
    let mut count = 0;
    for (inputs, targets) in loader.iter() {
        // 1. Cast Vec<usize> tokens to Vec<u32> (Candle requires u32 for integer tensors)
        let inputs_u32: Vec<u32> = inputs.iter().map(|&x| x as u32).collect();
        let targets_u32: Vec<u32> = targets.iter().map(|&x| x as u32).collect();

        // 2. Load slices as Tensors on CPU/GPU
        let inputs_t = Tensor::new(inputs_u32.as_slice(), device)?;
        let targets_t = Tensor::new(targets_u32.as_slice(), device)?;

        // 3. Compute logits and loss
        let logits = model.forward(&inputs_t)?;
        let loss = cross_entropy(&logits, &targets_t)?;

        total_loss += loss.to_scalar::<f32>()?;
        count += 1;
        if count >= num_batches {
            break;
        }
    }
    Ok(if count == 0 {
        0.0
    } else {
        total_loss / count as f32
    })
}

pub fn main() -> Result<()> {
    // Device
    let device = Device::Cpu;
    let mut varmap = VarMap::new();

    // Create a VarBuilder that registers all weights inside varmap
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    // Config
    let cfg = Gpt2Config::gpt2_mini(); // Miniature GPT-2 configuration
    let model = Gpt2Model::load(vs, &cfg)?;
    let max_lr = 3e-4;

    let params = ParamsAdamW {
        lr: max_lr,
        weight_decay: 0.01,
        ..Default::default()
    };
    // Extract all trainable tensors from the VarMap
    let mut optimizer = AdamW::new(varmap.all_vars(), params)?;
    // 1. Download and read the dataset
    let url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt";
    let _ = download_file_if_not_present(url, "./the-verdict.txt");
    let text = std::fs::read_to_string("./the-verdict.txt").expect("Failed to read file");

    // 2. Train BPE tokenizer matching the miniature vocab size
    let vocab_size = 1000;
    let mut tokenizer = BytePair::new();
    tokenizer.train(&text, vocab_size, None);

    // 3. Extract the target text slice
    let start_phrase = "It was not till three years later";
    let start_pos = text.find(start_phrase).expect("Target phrase not found");
    let test = &text[start_pos..start_pos + 1500];

    // 4. Tokenize and split sequential train/validation tokens (80% / 20%)
    let all_tokens = tokenizer.encode(test.to_string(), None).unwrap();
    let split_idx = (all_tokens.len() as f32 * 0.8) as usize;
    let train_tokens = all_tokens[..split_idx].to_vec();
    let val_tokens = all_tokens[split_idx..].to_vec();

    // 5. Initialize the generic DataLoaders (context length = 8)
    let max_seq_len = 8;
    let stride = 1;
    let train_loader = DataLoader::new(train_tokens, max_seq_len, stride);
    let val_loader = DataLoader::new(val_tokens, max_seq_len, stride);

    // 6. Set up the learning rate scheduler (Cosine Warmup over 80 epochs)
    let epochs = 100;
    let steps_per_epoch = train_loader.len();
    let total_steps = epochs * steps_per_epoch;
    let warmup_steps = (total_steps as f32 * 0.1) as usize; // 10% warmup
    let scheduler = LRScheduler::CosineWarmup {
        max_lr: max_lr as f32,
        min_lr: 1e-5,
        warmup_steps: warmup_steps,
        total_steps: total_steps,
    };

    // 7. Evaluate and print initial loss before training
    let initial_train_loss = calc_loss_loader(&model, &train_loader, &device, 10)?;
    let initial_val_loss = calc_loss_loader(&model, &val_loader, &device, 10)?;
    println!(
        "Initial Loss before training -> Train: {:.6}, Val: {:.6}",
        initial_train_loss, initial_val_loss
    );
    // 8. Start the training loop
    let mut global_step = 0;
    let start_time = std::time::Instant::now();
    println!("\nStarting training loop using Candle autograd...");
    for epoch in 1..=epochs {
        let mut steps = 0;
        let spinner = ["|", "/", "-", "\\"];

        for (inputs, targets) in train_loader.iter() {
            // Update learning rate dynamically
            let lr = scheduler.get_lr(global_step);
            optimizer.set_learning_rate(lr as f64);

            // Load batch onto Device
            let inputs_u32: Vec<u32> = inputs.iter().map(|&x| x as u32).collect();
            let targets_u32: Vec<u32> = targets.iter().map(|&x| x as u32).collect();
            let inputs_t = Tensor::new(inputs_u32.as_slice(), &device)?;
            let targets_t = Tensor::new(targets_u32.as_slice(), &device)?;

            // Forward pass & Loss computation
            let logits = model.forward(&inputs_t)?;
            let loss = cross_entropy(&logits, &targets_t)?;

            // Backpropagate gradients and update weights
            optimizer.backward_step(&loss)?;

            global_step += 1;
            steps += 1;
            print!(
                "\r  Training Epoch {:2}/{}... {} ",
                epoch,
                epochs,
                spinner[steps % spinner.len()]
            );
            std::io::Write::flush(&mut std::io::stdout())?;
        }

        print!("\r                                         \r");
        if epoch == 1 || epoch % 10 == 0 || epoch == epochs {
            let train_loss = calc_loss_loader(&model, &train_loader, &device, 10)?;
            let val_loss = calc_loss_loader(&model, &val_loader, &device, 10)?;
            println!(
                "  Epoch {:2}/{}: Train Loss = {:.6} | Val Loss = {:.6}",
                epoch, epochs, train_loss, val_loss
            );
        }
    }
    println!("\nTraining completed in {:.2?}!", start_time.elapsed());
    // 9. Save trained Candle weights to disk
    println!("\nSaving trained Candle weights...");
    varmap.save("candle_model.safetensors")?;
    println!("  ✅ Weights successfully saved to 'candle_model.safetensors'!");
    // 10. Load weights back and run Autoregressive Generation!
    println!("\nReloading weights from 'candle_model.safetensors' to test prediction...");
    varmap.load("candle_model.safetensors")?;

    let prompt_text = "if she had not dragged him down ";
    let prompt_tokens = tokenizer.encode(prompt_text.to_string(), None).unwrap();
    println!("  Prompt: \"{}\" -> {:?}", prompt_text, prompt_tokens);
    if prompt_tokens.len() > cfg.n_positions {
        let start_idx = prompt_tokens.len() - cfg.n_positions;
        println!(
            "  Context window: using last {} prompt tokens -> {:?}",
            cfg.n_positions,
            &prompt_tokens[start_idx..]
        );
    }

    let mut input_tokens = prompt_tokens.clone();
    let mut generated_tokens = Vec::new();
    let mut generated_text = String::new();
    print!("  Decoded result: {}", prompt_text);
    std::io::Write::flush(&mut std::io::stdout())?;

    for _ in 0..12 {
        let seq_len = input_tokens.len();
        let start_idx = if seq_len > cfg.n_positions {
            seq_len - cfg.n_positions
        } else {
            0
        };
        let context_tokens = &input_tokens[start_idx..];
        let inputs_u32: Vec<u32> = context_tokens.iter().map(|&x| x as u32).collect();
        let inputs_t = Tensor::new(inputs_u32.as_slice(), &device)?;
        let logits = model.forward(&inputs_t)?;

        // Get logits of the last token
        let seq_len = logits.dim(0)?;
        let last_logits = logits.narrow(0, seq_len - 1, 1)?.squeeze(0)?;
        let logits_vec = last_logits.to_vec1::<f32>()?;

        // Greedy decoding (temperature = 0.0)
        let next_token = sample_next_token(&logits_vec, 0.0, None, None);
        input_tokens.push(next_token);
        generated_tokens.push(next_token);

        let token_text = tokenizer
            .decode(vec![next_token])
            .map_err(anyhow::Error::msg)?;
        generated_text.push_str(&token_text);
        print!("{}", token_text);
        std::io::Write::flush(&mut std::io::stdout())?;
    }
    println!();
    println!("  Generated token ids: {:?}", generated_tokens);
    println!("  Generated suffix (debug): {:?}", generated_text);
    println!(
        "  Full decoded result (debug): {:?}",
        tokenizer.decode(input_tokens).map_err(anyhow::Error::msg)?
    );

    Ok(())
}
