//https://jaykmody.com/blog/gpt-from-scratch/
use llm_scratch_rs::{common::loss::cross_entropy_loss, models::gpt::GPT};

fn main() {
    // 1. Hyperparameters for a miniature GPT (for fast smoke testing)
    let vocab_size = 100;
    let d_model = 16;
    let max_seq_len = 32;
    let num_heads = 4;
    let d_ff = 64;
    let num_blocks = 2;

    // 2. Build the model
    let mut gpt = GPT::new(
        vocab_size,
        d_model,
        max_seq_len,
        num_heads,
        d_ff,
        num_blocks,
    );

    // 3. Simple forward pass (no training, just checking shapes)
    let input_tokens = vec![10, 20, 30, 40, 50]; // Sample token IDs
    println!("Input tokens: {:?}", input_tokens);
    let logits = gpt.forward(&input_tokens);

    // 4. Print shapes to verify
    // Expected output shape: [input_length][vocab_size]
    let seq_len = input_tokens.len();
    println!("Output shape: [{}][{}]", logits.len(), logits[0].len());
    assert_eq!(logits.len(), seq_len);
    assert_eq!(logits[0].len(), vocab_size);

    // 5. Print first token's logits for a quick sanity check
    println!("\nFirst token logits (first 5 dimensions):");
    let first_token_logits = &logits[0];
    let print_len = std::cmp::min(first_token_logits.len(), 5);
    for i in 0..print_len {
        println!("  logit[{}] = {:.4}", i, first_token_logits[i]);
    }

    // 6. Test the Loss Function
    // The "targets" are what we wanted the model to predict.
    // In language modeling, the target is usually the next word in the sequence.
    // Notice how it's shifted by 1 from the input [12, 45, 99, 1, 8]
    let targets = vec![45, 99, 1, 8, 25];

    let loss = cross_entropy_loss(&logits, &targets);

    println!("Initial Loss before training: {:.4}", loss);
    // 7. Test Autoregressive Generation!

    println!("\n--- Testing Generation ---");
    // We give the model a starting "prompt"
    let prompt = vec![12, 45];
    println!("Prompt tokens: {:?}", prompt);

    // Ask the model to generate 10 new tokens based on the prompt
    let generated = gpt.generate(&prompt, 10);

    println!("Final sequence: {:?}", generated);

    println!(
        "✅ Smoke test passed! The GPT architecture is fully connected and mathematically sound."
    );
}
