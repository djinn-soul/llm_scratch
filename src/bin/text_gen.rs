use llm_scratch_rs::{
    models::gpt::GPT,
    tokenizers::bpe::{download_file_if_not_present, BytePair},
};
use std::fs;
pub fn main() {
    let url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt";
    let vocab_size = 500;
    let _ = download_file_if_not_present(url, "./the-verdict.txt");

    // read the file
    let text = fs::read_to_string("./the-verdict.txt").expect("Failed to read file");
    println!("BytePair initialized successfully!");

    let mut tokenizer = BytePair::new();
    println!("Training tokenizer...");
    tokenizer.train(&text, vocab_size, None);
    let mut gpt = GPT::new(
        vocab_size, // 100
        16,         // d_model
        32,         // max_seq_len
        4,          // num_heads
        64,         // d_ff
        2,          // num_blocks
    );
    // 4. Encode a text prompt into numbers
    let prompt_text = "The tired man sat on the bench and \n";
    println!("\nPrompt: '{}'", prompt_text);
    let prompt_tokens = tokenizer.encode(prompt_text.to_string(), None).unwrap();
    println!("Encoded tokens: {:?}", prompt_tokens);

    // 5. Generate new numbers using the model
    println!("Generating new tokens...");
    let generated_tokens = gpt.generate(&prompt_tokens, 15); // Generate 15 new tokens
    println!("Generated sequence: {:?}", generated_tokens);
    // 6. Decode the numbers back into text!
    let output_text = tokenizer.decode(generated_tokens).unwrap();
    println!("\n=== FINAL GPT OUTPUT ===");
    println!("{}", output_text);
    println!("========================")
}
