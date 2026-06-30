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

#[test]
fn test_candle_kv_cache_equivalence() {
    use candle_core::{DType, Device, Tensor};
    use candle_nn::{VarBuilder, VarMap};
    use llm_scratch_rs::models::gpt2_candle::{Gpt2Config, Gpt2Model};

    let device = Device::new_cuda(0).unwrap_or(Device::Cpu);
    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let cfg = Gpt2Config::gpt2_mini();
    let model = Gpt2Model::load(vs, &cfg).unwrap();

    // Context / prompt tokens (cast to u32)
    let tokens = vec![5u32, 12, 18, 4, 45, 90, 3];

    // 1. Without KV cache: run the whole sequence as one pass.
    //
    // This builds Q/K/V for every token in one matrix call:
    //   tokens [0,1,2,3,4,5,6]
    //   K/V    [k0,k1,k2,k3,k4,k5,k6]
    //
    // The last logits row is the reference prediction for token position 6.
    model.set_use_cache(false);
    let tokens_t = Tensor::new(tokens.as_slice(), &device).unwrap();
    let full_logits = model.forward(&tokens_t).unwrap();

    // The last logits row (shape [seq_len, vocab_size]) is the reference prediction.
    let seq_len = full_logits.dim(0).unwrap();
    let expected_last_row = full_logits
        .narrow(0, seq_len - 1, 1)
        .unwrap()
        .squeeze(0)
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    // 2. With KV cache: run the first 4 tokens, then feed the rest one by one.
    //
    // First call:
    //   input [0,1,2,3]
    //   cache becomes K/V rows [k0,k1,k2,k3] and [v0,v1,v2,v3]
    //
    // Next calls:
    //   input [4] -> append k4/v4, score q4 against k0..k4
    //   input [5] -> append k5/v5, score q5 against k0..k5
    //   input [6] -> append k6/v6, score q6 against k0..k6
    //
    // If the cache update, position offset, and causal mask are correct, the
    // final logits for position 6 match the full-pass logits above.
    model.set_use_cache(true);
    model.clear_cache();

    // First part: tokens 0..4
    let first_part = &tokens[0..4];
    let first_part_t = Tensor::new(first_part, &device).unwrap();
    let _ = model.forward(&first_part_t).unwrap();

    // Subsequent tokens one-by-one: tokens 4, 5, 6
    let mut cached_logits = Vec::new();
    for &token in &tokens[4..] {
        let single_token_t = Tensor::new(&[token], &device).unwrap();
        let logits = model.forward(&single_token_t).unwrap();
        cached_logits = logits
            .narrow(0, logits.dim(0).unwrap() - 1, 1)
            .unwrap()
            .squeeze(0)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
    }

    // Compare logits
    assert_eq!(expected_last_row.len(), cached_logits.len());
    for i in 0..expected_last_row.len() {
        let diff = (expected_last_row[i] - cached_logits[i]).abs();
        assert!(
            diff < 1e-4,
            "Candle Logit mismatch at index {}: expected {}, got {} (diff: {})",
            i,
            expected_last_row[i],
            cached_logits[i],
            diff
        );
    }
}
