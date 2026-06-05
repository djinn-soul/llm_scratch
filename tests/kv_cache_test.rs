use llm_scratch_rs::models::gpt::GPT;

#[test]
fn test_kv_cache_equivalence() {
    let vocab_size = 100;
    let d_model = 16;
    let max_seq_len = 32;
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

    let tokens = vec![5usize, 12, 18, 4, 45, 90, 3];

    // 1. Without KV cache: run the whole sequence as one matrix pass.
    //
    // Internally each attention head sees:
    //   Q,K,V  [7][d_k or d_v]
    //   scores [7][7]
    //   logits [7][vocab_size]
    //
    // The last logits row is the reference answer for "after seeing all 7
    // tokens, what should the model predict next?"
    gpt.set_use_cache(false);
    let full_logits = gpt.forward(&tokens);
    let expected_last_row = full_logits.last().unwrap();

    // 2. With KV cache: split the same logical sequence across several forward
    // calls. The math should match the full pass even though the work is split.
    gpt.set_use_cache(true);
    gpt.clear_cache();

    // Pass the first part of the prompt. This fills every attention head's
    // cache with K/V rows for tokens 0..3.
    let first_part = &tokens[0..4];
    let _ = gpt.forward(first_part);

    // Pass the subsequent tokens one by one. Each call builds Q/K/V for only
    // the new token, then appends its K/V to the saved tables:
    //
    //   call for token 4: Q [1][d_k], cached K/V become [5][...]
    //   call for token 5: Q [1][d_k], cached K/V become [6][...]
    //   call for token 6: Q [1][d_k], cached K/V become [7][...]
    //
    // The final logits row should equal the full-pass row because both paths
    // represent the same visible token history.
    let mut cached_logits = Vec::new();
    for i in 4..tokens.len() {
        cached_logits = gpt.forward(&tokens[i..=i]);
    }
    let actual_last_row = cached_logits.last().unwrap();

    // 3. Compare logits row element-by-element. Tiny floating-point differences
    // are okay, but a large difference means the cached matrix path is not
    // equivalent to the full matrix path.
    assert_eq!(expected_last_row.len(), actual_last_row.len());
    for i in 0..expected_last_row.len() {
        let diff = (expected_last_row[i] - actual_last_row[i]).abs();
        assert!(
            diff < 1e-4,
            "Logit mismatch at index {}: expected {}, got {} (diff: {})",
            i,
            expected_last_row[i],
            actual_last_row[i],
            diff
        );
    }
}
