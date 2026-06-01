use std::fs;
use std::sync::Arc;
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

const BENCHMARK_EPOCHS: usize = 100;

// Calculates average loss over a fixed number of batches.
//
// This is evaluation-only: we run the model forward, compare logits to targets,
// and do not call backward or optimizer.step().
fn calc_loss_loader(gpt: &mut GPT, data_loader: &DataLoader, num_batches: usize) -> f32 {
    let mut total_loss = 0.0;
    let mut count = 0;

    for (input_slice, target_slice) in data_loader.iter() {
        // 1. Ask the current model to predict the next token at every position.
        let logits = gpt.forward(&input_slice);

        // 2. Convert those raw scores into one scalar loss for this batch.
        let loss = cross_entropy_loss(&logits, &target_slice);

        // 3. Accumulate batch losses so we can report the mean loss.
        total_loss += loss;
        count += 1;

        // 4. Stop early so evaluation stays cheap during benchmarking.
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

// Calculates next-token accuracy (%) over the validation dataset.
//
// Loss tells us "how confident and correct" the model is. Accuracy gives the
// simpler view: how often the highest-scoring token equals the real next token.
fn calc_accuracy(gpt: &mut GPT, data_loader: &DataLoader) -> f32 {
    let mut correct = 0;
    let mut total = 0;

    for (input_slice, target_slice) in data_loader.iter() {
        // 1. Run a forward pass to get one score vector per context position.
        let logits = gpt.forward(&input_slice);

        // 2. Loop over each token position in the context window.
        for (pos, &target_token) in target_slice.iter().enumerate() {
            let position_logits = &logits[pos];
            let mut pred_token = 0;
            let mut max_score = f32::NEG_INFINITY;

            // 3. Pick the token with the largest logit as the model prediction.
            for (token_id, &score) in position_logits.iter().enumerate() {
                if score > max_score {
                    max_score = score;
                    pred_token = token_id;
                }
            }

            // 4. Count whether the prediction matches the expected next token.
            if pred_token == target_token {
                correct += 1;
            }
            total += 1;
        }
    }

    if total == 0 {
        0.0
    } else {
        (correct as f32 / total as f32) * 100.0
    }
}

// Helper to run a complete pretraining pipeline under one scheduler strategy.
//
// Each scheduler gets its own fresh model and optimizer, so the comparison is
// about the learning-rate schedule rather than leftover weights from a prior run.
fn run_benchmark(
    scheduler_name: &str,
    scheduler: LRScheduler,
    train_loader: &DataLoader,
    val_loader: &DataLoader,
    vocab_size: usize,
) -> (f32, f32, f32, f32) {
    println!("🧪 Running Experiment: {}...", scheduler_name);

    // Step 1: Start a timer so the final table can compare speed too.
    let start_time = Instant::now();

    // Step 2: Re-initialize a fresh, identical GPT model for this run.
    // Keeping architecture constants here makes every scheduler train the same
    // tiny GPT and keeps the benchmark quick enough for experimentation.
    let d_model = 16;
    let max_seq_len = 8;
    let num_heads = 2;
    let d_ff = 32;
    let num_blocks = 2;
    let mut gpt = GPT::new(
        vocab_size,
        d_model,
        max_seq_len,
        num_heads,
        d_ff,
        num_blocks,
    );

    // Step 3: Re-initialize AdamW so optimizer state starts from zero.
    // The scheduler will overwrite `optimizer.lr` before each update step.
    let mut optimizer = AdamW::new(3e-4, 0.01, ClippingStrategy::Norm(1.0));

    // Step 4: Track total update count across epochs.
    // Schedulers usually depend on global step, not epoch-local batch index.
    let mut global_step = 0;

    for _epoch in 1..=BENCHMARK_EPOCHS {
        for (input_slice, target_slice) in train_loader.iter() {
            // Step 5: Ask the selected scheduler what LR to use now.
            optimizer.lr = scheduler.get_lr(global_step);

            // Step 6: Clear old gradients before the next backward pass.
            // Gradients accumulate in Param.grad, so leaving old values would
            // mix the previous batch's signal into the current batch.
            {
                let mut params = gpt.parameters();
                for param in &mut params {
                    param.zero_grad();
                }
            }

            // Step 7: Forward pass turns token IDs into prediction logits.
            let logits = gpt.forward(&input_slice);

            // Step 8: Convert the loss into dL/dlogits, the first backward signal.
            let d_logits = cross_entropy_backward(&logits, &target_slice);

            // Step 9: Backpropagate dL/dlogits through GPT into parameter grads.
            gpt.backward(&d_logits);

            // Step 10: Let AdamW read each Param.grad and update Param.data.
            {
                let mut params = gpt.parameters();
                optimizer.step(&mut params);
            }

            // Step 11: Move the scheduler clock forward for the next batch.
            global_step += 1;
        }
    }

    // Step 12: Stop the timer after training finishes.
    let elapsed = start_time.elapsed().as_secs_f32();

    // Step 13: Calculate final evaluation metrics using the trained model.
    let final_train_loss = calc_loss_loader(&mut gpt, train_loader, 15);
    let final_val_loss = calc_loss_loader(&mut gpt, val_loader, 15);
    let final_val_acc = calc_accuracy(&mut gpt, val_loader);

    println!(
        "   ✅ Finished in {:.1}s | Train Loss: {:.4} | Val Loss: {:.4} | Val Acc: {:.1}%\n",
        elapsed, final_train_loss, final_val_loss, final_val_acc
    );

    (final_train_loss, final_val_loss, final_val_acc, elapsed)
}

fn main() {
    // Step 1: Download the tiny text corpus if it is not already on disk.
    let url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt";
    let _ = download_file_if_not_present(url, "./the-verdict.txt");

    // Step 2: Read the corpus into memory for tokenizer training and slicing.
    let text = fs::read_to_string("./the-verdict.txt").expect("Failed to read file");

    // Step 3: Train a small BPE tokenizer directly on this corpus.
    let vocab_size = 1000;
    let mut tokenizer = BytePair::new();
    tokenizer.train(&text, vocab_size, None);

    // Step 4: Select a short, stable excerpt so every benchmark uses the same data.
    let start_phrase = "It was not till three years later";
    let start_pos = text.find(start_phrase).expect("Target phrase not found");
    let test = &text[start_pos..start_pos + 1500];

    // Step 5: Encode text into token IDs, then split them into train/validation.
    let all_tokens = tokenizer.encode(test.to_string(), None).unwrap();
    let split_idx = (all_tokens.len() as f32 * 0.8) as usize; // 80/20 split.
    let train_tokens = all_tokens[..split_idx].to_vec();
    let val_tokens = all_tokens[split_idx..].to_vec();

    // Step 6: Build dataloaders with context length 8 and batch size 1.
    // Arc lets each benchmark thread share the immutable loader cheaply.
    let train_loader = Arc::new(DataLoader::new(train_tokens, 8, 1));
    let val_loader = Arc::new(DataLoader::new(val_tokens, 8, 1));

    println!("===============================================================");
    println!("         STARTING SYSTEMATIC LEARNING RATE BENCHMARKS");
    println!("===============================================================\n");

    // Step 7: Estimate how many updates the cosine scheduler should span.
    let total_steps = BENCHMARK_EPOCHS * train_loader.len();

    // Step 8: Define swappable scheduler configurations.
    // Every experiment below will train the same model on the same tokens, but
    // the LR value used at each optimizer step will come from a different rule.
    let experiments = vec![
        ("Constant LR (Baseline)", LRScheduler::Constant { lr: 3e-4 }),
        (
            "Step Decay (Halves every 15 epochs)",
            LRScheduler::StepDecay {
                initial_lr: 5e-4,
                decay_rate: 0.5,
                step_size: 15 * train_loader.len(),
            },
        ),
        (
            "Exponential Decay (0.999 per step)",
            LRScheduler::ExponentialDecay {
                initial_lr: 5e-4,
                decay_rate: 0.999,
            },
        ),
        (
            "Cosine Warmup (10% warmup)",
            LRScheduler::CosineWarmup {
                max_lr: 5e-4,
                min_lr: 1e-5,
                warmup_steps: (total_steps as f32 * 0.1) as usize,
                total_steps,
            },
        ),
    ];

    // Step 9: Spawn one thread per scheduler so the benchmark runs in parallel.
    let mut handles = Vec::new();

    for (name, scheduler) in experiments {
        // Clone cheap references into the thread; the token data itself is shared.
        let name = name.to_string();
        let train_loader_clone = Arc::clone(&train_loader);
        let val_loader_clone = Arc::clone(&val_loader);

        // Move the scheduler and cloned loader handles into the worker thread.
        let handle = std::thread::spawn(move || {
            let (train_loss, val_loss, val_acc, elapsed) = run_benchmark(
                &name,
                scheduler,
                &train_loader_clone,
                &val_loader_clone,
                vocab_size,
            );
            (name, train_loss, val_loss, val_acc, elapsed)
        });
        handles.push(handle);
    }

    // Step 10: Wait for every worker to finish and collect its metrics.
    let mut results = Vec::new();
    for handle in handles {
        let res = handle.join().unwrap();
        results.push(res);
    }

    // Step 11: Print the comparison table after all experiments are complete.
    println!("\n==========================================================================");
    println!("                            FINAL RESULTS TABLE                           ");
    println!("==========================================================================");
    println!(
        "{:<32} | {:<12} | {:<12} | {:<12} | {:<10}",
        "Scheduler Strategy", "Train Loss", "Val Loss", "Val Acc (%)", "Time (s)"
    );
    println!("--------------------------------------------------------------------------");
    for (name, train_loss, val_loss, val_acc, elapsed) in results {
        println!(
            "{:<32} | {:<12.4} | {:<12.4} | {:<12.1}% | {:.1}s",
            name, train_loss, val_loss, val_acc, elapsed
        );
    }
    println!("==========================================================================");
}
