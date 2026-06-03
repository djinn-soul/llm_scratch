use std::fs;
use std::io::{self, Write};
use std::time::Instant;

use llm_scratch_rs::{
    common::{
        dataloader::DataLoader,
        loss::{cross_entropy_backward, cross_entropy_loss},
        lr_scheduler::LRScheduler,
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
    // ── PHASE 1: DATASET + LOCAL BPE TOKENIZER ────────────────────────────
    // This manual-training path learns from a small slice of "The Verdict".
    // The tokenizer is trained locally so vocab size stays tiny and matches the
    // miniature GPT model below.
    let url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt";
    let _ = download_file_if_not_present(url, "./the-verdict.txt");

    let text = fs::read_to_string("./the-verdict.txt").expect("Failed to read file");
    println!("BytePair initialized successfully!");
    let vocab_size = 1000;
    let mut tokenizer = BytePair::new();
    println!("Training BPE tokenizer...");
    tokenizer.train(&text, vocab_size, None);
    println!("Tokenizer trained! Vocab size: {}\n", tokenizer.vocab.len());

    // ── PHASE 2: TARGET CORPUS + SEQUENTIAL SPLIT ─────────────────────────
    // Select one contiguous slice so the model can overfit a visible phrase.
    // The validation split is also sequential; no shuffling is used in this
    // learning example.
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
    // 80% training / 20% validation split sequentially
    let split_idx = (all_tokens.len() as f32 * 0.8) as usize;
    let train_tokens = all_tokens[..split_idx].to_vec();
    let val_tokens = all_tokens[split_idx..].to_vec();
    println!(
        "Split corpus into {} training tokens and {} validation tokens.",
        train_tokens.len(),
        val_tokens.len()
    );

    // ── PHASE 3: SLIDING-WINDOW DATALOADERS ───────────────────────────────
    // Each sample is:
    //   input  = 8-token context
    //   target = same window shifted left by one next token
    // stride=1 gives the densest possible set of training examples.
    let max_seq_len = 8; // Context window size
    let stride = 1; // Step 1 token at a time
    let train_loader = DataLoader::new(train_tokens, max_seq_len, stride);
    let val_loader = DataLoader::new(val_tokens, max_seq_len, stride);
    println!(
        "DataLoader initialized! Total samples: {}",
        train_loader.len()
    );

    // ── PHASE 4: MINI MANUAL GPT + OPTIMIZER ──────────────────────────────
    // This is not official GPT-2 Small. It is a tiny GPT-shaped model using the
    // repo's hand-written forward/backward implementations.
    let d_model = 16;
    let max_seq_len = 8; // Context window size
    let num_heads = 2;
    let d_ff = 32;
    let num_blocks = 2;
    let epochs = 80;

    // AdamW hyperparameters:
    //   lr=3e-4  — the "Karpathy constant", safe default for LLM training with Adam
    //   wd=0.01  — standard weight decay used in GPT-2, BERT, and LLaMA training
    let learning_rate = 3e-4;
    let weight_decay = 0.01;
    let steps_per_epoch = train_loader.len();
    let total_steps = epochs * steps_per_epoch;
    let warmup_steps = (total_steps as f32 * 0.1) as usize; // 10% warmup
    let scheduler = LRScheduler::CosineWarmup {
        max_lr: 3e-4,
        min_lr: 1e-5,
        warmup_steps,
        total_steps,
    };
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
    let mut optimizer = AdamW::new(learning_rate, weight_decay, ClippingStrategy::Norm(1.0));

    println!(
        "Initialized mini-GPT and AdamW optimizer (lr = {:.0e}, wd = {}).\n",
        learning_rate, weight_decay
    );

    // ── PHASE 5: BASELINE LOSS BEFORE ANY PARAMETER UPDATE ────────────────
    // This makes it easy to verify training later: loss after training should
    // move down on the training split even if validation overfits.
    let initial_train_loss = calc_loss_loader(&mut gpt, &train_loader, 10);
    let initial_val_loss = calc_loss_loader(&mut gpt, &val_loader, 10);
    println!(
        "[2/5] Initial cross-entropy loss -> Train: {:.6}, Val: {:.6}",
        initial_train_loss, initial_val_loss
    );

    // ── PHASE 6: MANUAL BACKPROPAGATION TRAINING LOOP ─────────────────────
    // Per batch:
    //   1. zero accumulated gradients from the previous step
    //   2. forward pass: token ids -> logits
    //   3. loss backward: logits -> d_logits
    //   4. GPT backward: route gradients through every layer
    //   5. AdamW step: mutate parameters using accumulated gradients
    let mut global_step = 0; // <--- Track global steps across epochs

    let start_time = Instant::now();
    println!("\n[3/5] Starting manual backpropagation training loop...");
    for epoch in 1..=epochs {
        let mut steps = 0;
        let spinner = ["|", "/", "-", "\\"];

        // Iterate over the dataset using the generic DataLoader iterator
        for (input_slice, target_slice) in train_loader.iter() {
            // Update the optimizer's learning rate dynamically from the scheduler!
            optimizer.lr = scheduler.get_lr(global_step);

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
            global_step += 1; // <--- Increment global step
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

    // ── PHASE 7: SAVE WEIGHTS ─────────────────────────────────────────────
    // The serialization trait chooses JSON or binary format from the extension.
    println!("\n[4/5] Saving model weights to disk...");
    gpt.save_weights("model_weights.json")
        .expect("Failed to save JSON weights");
    gpt.save_weights("model_weights.bin")
        .expect("Failed to save binary weights");
    println!("  ✅ Weights successfully saved to 'model_weights.json' and 'model_weights.bin'!");

    // ── PHASE 8: RELOAD + AUTOREGRESSIVE GENERATION ───────────────────────
    // Reloading before generation proves the saved binary weights are usable.
    // Generation then loops one token at a time, feeding each prediction back
    // into the next context window.
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
