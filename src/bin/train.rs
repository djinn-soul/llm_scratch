use std::fs;
use std::io::{self, Write};
use std::time::Instant;

use llm_scratch_rs::{
    common::{
        dataloader::DataLoader,
        loss::{cross_entropy_backward, cross_entropy_loss},
        optimizers::{AdamW, ClippingStrategy, Optimizer},
        serilization::SaveableModel,
    },
    models::gpt::GPT,
    tokenizers::bpe::{download_file_if_not_present, BytePair},
};

/// Computes the average cross-entropy loss over a given number of batches from a DataLoader.
pub fn calc_loss_loader(gpt: &mut GPT, data_loader: &DataLoader, num_batches: usize) -> f32 {
    let mut total_loss = 0.0;
    let mut count = 0;

    for (input_slice, target_slice) in data_loader.iter() {
        let logits = gpt.forward(&input_slice);
        let loss = cross_entropy_loss(&logits, &target_slice);
        total_loss += loss;
        count += 1;

        if count >= num_batches {
            break;
        }
    }

    if count == 0 {
        0.0
    } else {
        total_loss / count as f32
    }
}

fn main() {
    let url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt";
    let _ = download_file_if_not_present(url, "./the-verdict.txt");

    let text = fs::read_to_string("./the-verdict.txt").expect("Failed to read file");
    println!("BytePair initialized successfully!");
    let vocab_size = 1000;
    let mut tokenizer = BytePair::new();
    println!("Training BPE tokenizer...");
    tokenizer.train(&text, vocab_size, None);
    println!("Tokenizer trained! Vocab size: {}\n", tokenizer.vocab.len());

    // 3. Define target training corpus to memorize (sliding window from the book!)
    let start_phrase = "It was not till three years later";
    let start_pos = text
        .find(start_phrase)
        .expect("Target phrase not found in book");

    // Slice 1,500 characters containing the sentence: "if she had not dragged him down..."
    let test = &text[start_pos..start_pos + 1500];
    println!("\n[1/5] Target text to learn:\n  \"{}\"", test);

    // Tokenize our corpus
    let all_tokens = tokenizer.encode(test.to_string(), None).unwrap();
    println!(
        "Encoded corpus ({} tokens): {:?}",
        all_tokens.len(),
        all_tokens
    );
    // 90% training / 10% validation split sequentially
    let split_idx = (all_tokens.len() as f32 * 0.9) as usize;
    let train_tokens = all_tokens[..split_idx].to_vec();
    let val_tokens = all_tokens[split_idx..].to_vec();
    println!(
        "Split corpus into {} training tokens and {} validation tokens.",
        train_tokens.len(),
        val_tokens.len()
    );

    // 4. Initialize modular DataLoader (max_length = 8, stride = 1)
    let max_seq_len = 8; // Context window size
    let stride = 1; // Step 1 token at a time
    let train_loader = DataLoader::new(train_tokens, max_seq_len, stride);
    let val_loader = DataLoader::new(val_tokens, max_seq_len, stride);
    println!(
        "DataLoader initialized! Total samples: {}",
        train_loader.len()
    );

    // 5. Initialize a miniature GPT model
    let d_model = 16;
    let max_seq_len = 8; // Context window size
    let num_heads = 2;
    let d_ff = 32;
    let num_blocks = 2;
    // AdamW hyperparameters:
    //   lr=3e-4  — the "Karpathy constant", safe default for LLM training with Adam
    //   wd=0.01  — standard weight decay used in GPT-2, BERT, and LLaMA training
    let learning_rate = 3e-4;
    let weight_decay = 0.01;

    let mut gpt = GPT::new(
        vocab_size,
        d_model,
        max_seq_len,
        num_heads,
        d_ff,
        num_blocks,
    );
    // Initialize AdamW — Adam with decoupled weight decay
    // Weight decay shrinks weights each step: w *= (1 - lr * wd)
    // This prevents overfitting without corrupting Adam's moment buffers.
    let mut optimizer = AdamW::new(
        learning_rate,
        weight_decay,
        ClippingStrategy::Norm(1.0),
    );

    println!(
        "Initialized mini-GPT and AdamW optimizer (lr = {:.0e}, wd = {}).\n",
        learning_rate, weight_decay
    );
    // Let's print initial loss before training using the first batch
    // Replace lines 119-126 in train.rs:
    let initial_train_loss = calc_loss_loader(&mut gpt, &train_loader, 10);
    let initial_val_loss = calc_loss_loader(&mut gpt, &val_loader, 10);
    println!(
        "[2/5] Initial cross-entropy loss -> Train: {:.6}, Val: {:.6}",
        initial_train_loss, initial_val_loss
    );

    // 7. Training Loop using our manual backpropagation and DataLoader
    let start_time = Instant::now();
    println!("\n[3/5] Starting manual backpropagation training loop...");
    let epochs = 80;
    for epoch in 1..=epochs {
        let mut steps = 0;
        let spinner = ["|", "/", "-", "\\"];

        // Iterate over the dataset using the generic DataLoader iterator
        for (input_slice, target_slice) in train_loader.iter() {
            // A. Zero out gradients from previous step
            {
                let mut params = gpt.parameters();
                for param in &mut params {
                    param.zero_grad();
                }
            }
            // B. Forward Pass: compute raw logits
            let logits = gpt.forward(&input_slice);
            steps += 1;

            // Render interactive spinner
            print!(
                "\r  Training Epoch {:2}/{}... {} ",
                epoch,
                epochs,
                spinner[steps % spinner.len()]
            );
            io::stdout().flush().unwrap();

            // D. Backward Pass: compute and route gradients back through the entire network
            let d_logits = cross_entropy_backward(&logits, &target_slice);
            gpt.backward(&d_logits);
            // E. Optimizer Step: update all parameter weights.
            {
                let mut params = gpt.parameters();
                optimizer.step(&mut params);
            }
        }
        // Clear the spinner line cleanly
        print!("\r                                         \r");
        if epoch == 1 || epoch % 10 == 0 || epoch == epochs {
            let train_loss = calc_loss_loader(&mut gpt, &train_loader, 10);
            let val_loss = calc_loss_loader(&mut gpt, &val_loader, 10);
            println!(
                "  Epoch {:2}/{}: Train Loss = {:.6} | Val Loss = {:.6}",
                epoch, epochs, train_loss, val_loss
            );
        } else {
            io::stdout().flush().unwrap();
        }
    }
    // 8. Save Model Weights (Auto-detecting JSON and Binary based on file extension!)
    println!("\n[4/5] Saving model weights to disk...");
    gpt.save_weights("model_weights.json")
        .expect("Failed to save JSON weights");
    gpt.save_weights("model_weights.bin")
        .expect("Failed to save binary weights");
    println!("  ✅ Weights successfully saved to 'model_weights.json' and 'model_weights.bin'!");

    // 9. Load weights back and run Autoregressive Generation!
    println!("\n[5/5] Testing generation with the reloaded model...");

    // Prompt the model with a starting sequence
    let prompt_text = "if she had not dragged him down ";
    let prompt_tokens = tokenizer.encode(prompt_text.to_string(), None).unwrap();
    println!("  Prompt: \"{}\" -> {:?}", prompt_text, prompt_tokens);

    // Let's reload the weights from the binary file to verify loading works perfectly!
    println!("  Loading weights back from 'model_weights.bin' before generation...");
    gpt.load_weights("model_weights.bin")
        .expect("Failed to load binary weights");

    // Generate 12 new tokens to complete the sequence autoregressively
    let generated_ids = gpt.generate(&prompt_tokens, 12);
    println!("  Generated Token IDs: {:?}", generated_ids);

    let decoded_result = tokenizer.decode(generated_ids).unwrap();
    println!("\nFinal decoded sequence:\n  \"{}\"", decoded_result.trim());

    let elapsed = start_time.elapsed();
    println!(
        "\n🎉 Training and saving complete in {:.2?}! The model successfully memorized and reproduced the sequence.",
        elapsed
    );
}
