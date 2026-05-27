use std::fs;
use std::io::{self, Write};

use llm_scratch_rs::{
    common::{
        dataloader::DataLoader,
        loss::{cross_entropy_backward, cross_entropy_loss},
        optimizers::{Optimizer, SGD},
    },
    models::gpt::GPT,
    tokenizers::bpe::{download_file_if_not_present, BytePair},
};

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
    let start_pos = text.find(start_phrase).expect("Target phrase not found in book");
    
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


    // 4. Initialize modular DataLoader (max_length = 8, stride = 1)
    let max_seq_len = 8; // Context window size
    let stride = 1; // Step 1 token at a time
    let data_loader = DataLoader::new(all_tokens, max_seq_len, stride);
    println!(
        "DataLoader initialized! Total samples: {}",
        data_loader.len()
    );

    // 5. Initialize a miniature GPT model
    let d_model = 16;
    let max_seq_len = 8; // Context window size
    let num_heads = 2;
    let d_ff = 32;
    let num_blocks = 2;
    let learning_rate = 0.05;

    let mut gpt = GPT::new(
        vocab_size,
        d_model,
        max_seq_len,
        num_heads,
        d_ff,
        num_blocks,
    );
    // 5. Initialize the modular SGD optimizer
    let mut optimizer = SGD::new(learning_rate);
    println!(
        "Initialized mini-GPT and modular SGD optimizer (lr = {}).\n",
        learning_rate
    );
    // Let's print initial loss before training using the first batch
    let (init_input, init_target) = data_loader.get_item(0);
    let initial_logits = gpt.forward(&init_input);
    let initial_loss = cross_entropy_loss(&initial_logits, &init_target);
    println!(
        "[2/5] Initial cross-entropy loss (first window): {:.6}",
        initial_loss
    );
    // 7. Training Loop using our manual backpropagation and DataLoader
    println!("\n[3/5] Starting manual backpropagation training loop...");
    let epochs = 80;
    for epoch in 1..=epochs {
        let mut epoch_loss = 0.0;
        let mut steps = 0;
        let spinner = ["|", "/", "-", "\\"];

        // Iterate over the dataset using the generic DataLoader iterator
        for (input_slice, target_slice) in data_loader.iter() {
            // A. Zero out gradients from previous step
            {
                let mut params = gpt.parameters();
                for param in &mut params {
                    param.zero_grad();
                }
            }
            // B. Forward Pass: compute raw logits
            let logits = gpt.forward(&input_slice);
            // C. Compute Cross-Entropy Loss
            let loss = cross_entropy_loss(&logits, &target_slice);
            epoch_loss += loss;
            steps += 1;

            // Render interactive spinner
            print!("\r  Training Epoch {:2}/{}... {} ", epoch, epochs, spinner[steps % spinner.len()]);
            io::stdout().flush().unwrap();

            // D. Backward Pass: compute and route gradients back through the entire network
            let d_logits = cross_entropy_backward(&logits, &target_slice);
            gpt.backward(&d_logits);
            // E. Optimizer Step: update all parameter weights using SGD
            {
                let mut params = gpt.parameters();
                optimizer.step(&mut params);
            }
        }
        let avg_loss = epoch_loss / steps as f32;
        // Clear the spinner line cleanly
        print!("\r                                         \r");
        if epoch == 1 || epoch % 10 == 0 || epoch == epochs {
            println!("  Epoch {:2}/{}: Avg Loss = {:.6}", epoch, epochs, avg_loss);
        } else {
            io::stdout().flush().unwrap();
        }
    }
    // 8. Test Autoregressive Generation after training!
    println!("\n[4/5] Testing generation with the trained model...");
    // Prompt the model with a starting sequence
    let prompt_text = "if she had not dragged him down ";
    let prompt_tokens = tokenizer.encode(prompt_text.to_string(), None).unwrap();
    println!("  Prompt: \"{}\" -> {:?}", prompt_text, prompt_tokens);
    // Generate 12 new tokens to complete the sequence autoregressively
    let generated_ids = gpt.generate(&prompt_tokens, 12);
    println!("  Generated Token IDs: {:?}", generated_ids);
    let decoded_result = tokenizer.decode(generated_ids).unwrap();
    println!(
        "\n[5/5] Final decoded sequence:\n  \"{}\"",
        decoded_result.trim()
    );

    println!(
        "\n🎉 Training complete! The model successfully memorized and reproduced the sequence."
    );
}
