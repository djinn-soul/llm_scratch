use llm_scratch_rs::{
    common::serilization::SaveableModel,
    models::gpt::GPT,
    tokenizers::bpe::{download_file_if_not_present, BytePair},
};
use std::fs;
use std::path::Path;

pub fn main() {
    // ── PHASE 1: MATCH THE TRAINING SETUP ─────────────────────────────────
    // This demo must rebuild the same tokenizer/model shape used by
    // `src/bin/train.rs`, otherwise saved token ids or weights would not line
    // up. The important shared constants are vocab_size=1000, d_model=16,
    // context window=8, heads=2, d_ff=32, and blocks=2.
    let url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt";
    let vocab_size = 1000; // Matches train.rs configuration
    let _ = download_file_if_not_present(url, "./the-verdict.txt");

    // Train the same local BPE tokenizer used by the manual training binary.
    // Token ids are only meaningful relative to the tokenizer vocabulary that
    // produced them.
    let text = fs::read_to_string("./the-verdict.txt").expect("Failed to read file");

    let mut tokenizer = BytePair::new();
    println!("Training BPE tokenizer (vocab size: {})...", vocab_size);
    tokenizer.train(&text, vocab_size, None);

    let mut gpt = GPT::new(
        vocab_size, 16, // d_model
        8,  // max_seq_len
        2,  // num_heads
        32, // d_ff
        2,  // num_blocks
    );

    // ── PHASE 2: LOAD TRAINED WEIGHTS IF AVAILABLE ────────────────────────
    // `model_weights.bin` is produced by `cargo run --bin train`. If it is not
    // present, generation still runs, but it samples from random model weights
    // and the outputs are useful only as shape/control-flow smoke tests.
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

    // ── PHASE 3: TOKENIZE A PROMPT FROM THE TRAINING SLICE ────────────────
    // The prompt is chosen from the phrase the tiny model was trained to
    // memorize, so differences between decoding strategies are easier to see.
    let prompt_text = "if she had not dragged him down ";
    let prompt_tokens = tokenizer.encode(prompt_text.to_string(), None).unwrap();

    println!("\n==========================================================================");
    println!("                           SAMPLING SHOWCASE");
    println!("==========================================================================");
    println!("Prompt: \"{}\"\n", prompt_text);

    // ── PHASE 4: COMPARE DECODING STRATEGIES ──────────────────────────────
    // All three calls use the same model and prompt. Only the sampling policy
    // changes, so the printed outputs show how decoding affects generation.

    // Strategy 1: Greedy decoding
    // temperature=0.0 bypasses randomness and always takes the highest-logit
    // next token. This is stable, but often repetitive.
    println!("🧪 [1/3] Running Greedy Decoding (T = 0.0)...");
    let greedy_tokens = gpt.generate_sample(&prompt_tokens, 15, 0.0, None, None);
    let greedy_output = tokenizer.decode(greedy_tokens).unwrap();
    println!("  ↳ Output: \"{}\"\n", greedy_output.trim());

    // Strategy 2: high-temperature random decoding
    // temperature=1.3 flattens the probability distribution. With no top-k or
    // top-p filter, even weak candidates can be sampled.
    println!("🧪 [2/3] Running Highly Random Decoding (T = 1.3, No Top-K/Top-P)...");
    let random_tokens = gpt.generate_sample(&prompt_tokens, 15, 1.3, None, None);
    let random_output = tokenizer.decode(random_tokens).unwrap();
    println!("  ↳ Output: \"{}\"\n", random_output.trim());

    // Strategy 3: constrained stochastic decoding
    // temperature adds variety, top-k keeps only the 10 strongest logits, and
    // top-p keeps the smallest token set whose probability mass reaches 0.85.
    println!("🧪 [3/3] Running Balanced Decoding (T = 0.8, Top-K = 10, Top-P = 0.85)...");
    let balanced_tokens = gpt.generate_sample(&prompt_tokens, 15, 0.8, Some(10), Some(0.85));
    let balanced_output = tokenizer.decode(balanced_tokens).unwrap();
    println!("  ↳ Output: \"{}\"\n", balanced_output.trim());
    println!("==========================================================================");
}
