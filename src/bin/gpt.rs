//https://jaykmody.com/blog/gpt-from-scratch/
use llm_scratch_rs::{
    common::{loss::cross_entropy_loss, sampling::sample_next_token},
    models::gpt::GPT,
};

fn main() {
    // ── PHASE 1: BUILD A TINY GPT FOR SHAPE TESTING ───────────────────────
    // This binary does not train. It only proves that the manual GPT forward
    // path, loss helper, and autoregressive loop agree on tensor shapes.
    let vocab_size = 100;
    let d_model = 16;
    let max_seq_len = 32;
    let num_heads = 4;
    let d_ff = 64;
    let num_blocks = 2;

    // Model shape:
    //   token ids -> [seq_len, d_model] hidden states
    //   hidden states -> [seq_len, vocab_size] logits
    let mut gpt = GPT::new(
        vocab_size,
        d_model,
        max_seq_len,
        num_heads,
        d_ff,
        num_blocks,
    );

    // ── PHASE 2: FORWARD PASS SMOKE TEST ──────────────────────────────────
    // Input is a short sequence of fake token ids. The model should return one
    // vocabulary-sized logit row for each input token.
    let input_tokens = vec![10, 20, 30, 40, 50]; // Sample token IDs
    println!("Input tokens: {:?}", input_tokens);
    let logits = gpt.forward(&input_tokens);

    // Expected output shape: [input_length][vocab_size].
    let seq_len = input_tokens.len();
    println!("Output shape: [{}][{}]", logits.len(), logits[0].len());
    assert_eq!(logits.len(), seq_len);
    assert_eq!(logits[0].len(), vocab_size);

    // Print a few logits so the demo shows real numeric output, not only shape.
    println!("\nFirst token logits (first 5 dimensions):");
    let first_token_logits = &logits[0];
    let print_len = std::cmp::min(first_token_logits.len(), 5);
    for i in 0..print_len {
        println!("  logit[{}] = {:.4}", i, first_token_logits[i]);
    }

    // ── PHASE 3: LOSS HELPER SMOKE TEST ───────────────────────────────────
    // Targets are next-token ids. In real language-model training they are the
    // input sequence shifted left by one position.
    let targets = vec![45, 99, 1, 8, 25];

    let loss = cross_entropy_loss(&logits, &targets);

    println!("Initial Loss before training: {:.4}", loss);

    // ── PHASE 4: AUTOREGRESSIVE GENERATION SMOKE TEST ─────────────────────
    // Every loop:
    //   1. score the whole current sequence
    //   2. take the last row of logits
    //   3. greedily choose the next token
    //   4. append it and repeat
    println!("\n--- Testing Generation (Streaming token IDs) ---");
    let prompt = vec![12, 45];
    print!("Prompt: {:?}", prompt);
    std::io::Write::flush(&mut std::io::stdout()).unwrap();

    let mut current_tokens = prompt.clone();
    for _ in 0..10 {
        let logits = gpt.forward(&current_tokens);
        let last_logits = logits.last().unwrap();
        // temperature=0.0 means deterministic argmax decoding.
        let next_token = sample_next_token(last_logits, 0.0, None, None);
        current_tokens.push(next_token);

        print!(" -> {}", next_token);
        std::io::Write::flush(&mut std::io::stdout()).unwrap();
    }
    println!("\nFinal sequence: {:?}", current_tokens);
}
