use llm_scratch_rs::{
    common::serilization::SaveableModel,
    models::gpt::GPT,
    tokenizers::bpe::{download_file_if_not_present, BytePair},
};
use std::fs;
use std::path::Path;

pub fn main() {
    let url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt";
    let vocab_size = 1000; // Matches train.rs configuration
    let _ = download_file_if_not_present(url, "./the-verdict.txt");

    // Read the file
    let text = fs::read_to_string("./the-verdict.txt").expect("Failed to read file");

    let mut tokenizer = BytePair::new();
    println!("Training BPE tokenizer (vocab size: {})...", vocab_size);
    tokenizer.train(&text, vocab_size, None);

    // Initialize the exact GPT architecture matching train.rs
    let mut gpt = GPT::new(
        vocab_size, // 1000
        16,         // d_model
        8,          // max_seq_len
        2,          // num_heads
        32,         // d_ff
        2,          // num_blocks
    );

    // Dynamic weight loading check
    let weights_path = "./model_weights.bin";
    if Path::new(weights_path).exists() {
        println!(
            "\n💾 Found pre-trained weights at '{}'! Loading...",
            weights_path
        );
        gpt.load_weights(weights_path)
            .expect("Failed to load weights");
        println!("  ✅ Pre-trained weights successfully loaded!");
    } else {
        println!("\n⚠️ No pre-trained weights found. Running on untrained/random weights.");
        println!("  (Run `cargo run --bin train` first to produce optimized weights!)");
    }

    // Set a prompt from the training split
    let prompt_text = "if she had not dragged him down ";
    let prompt_tokens = tokenizer.encode(prompt_text.to_string(), None).unwrap();

    println!("\n==========================================================================");
    println!("                           SAMPLING SHOWCASE");
    println!("==========================================================================");
    println!("Prompt: \"{}\"\n", prompt_text);

    // Strategy 1: Greedy Decoding
    println!("🧪 [1/3] Running Greedy Decoding (T = 0.0)...");
    let greedy_tokens = gpt.generate_sample(&prompt_tokens, 15, 0.0, None, None);
    let greedy_output = tokenizer.decode(greedy_tokens).unwrap();
    println!("  ↳ Output: \"{}\"\n", greedy_output.trim());

    // Strategy 2: High Randomness (No constraints)
    println!("🧪 [2/3] Running Highly Random Decoding (T = 1.3, No Top-K/Top-P)...");
    let random_tokens = gpt.generate_sample(&prompt_tokens, 15, 1.3, None, None);
    let random_output = tokenizer.decode(random_tokens).unwrap();
    println!("  ↳ Output: \"{}\"\n", random_output.trim());

    // Strategy 3: Nucleus & Top-K Sampling (Creative but focused)
    println!("🧪 [3/3] Running Balanced Decoding (T = 0.8, Top-K = 10, Top-P = 0.85)...");
    let balanced_tokens = gpt.generate_sample(&prompt_tokens, 15, 0.8, Some(10), Some(0.85));
    let balanced_output = tokenizer.decode(balanced_tokens).unwrap();
    println!("  ↳ Output: \"{}\"\n", balanced_output.trim());
    println!("==========================================================================");
}
