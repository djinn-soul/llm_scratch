use anyhow::Result;
use candle_core::{DType, Tensor};
use candle_nn::VarBuilder;
use hf_hub::api::sync::Api;
use llm_scratch_rs::{
    common::sampling::sample_next_token,
    models::gpt2_candle::{Gpt2Config, Gpt2Model},
};
use tokenizers::Tokenizer;

fn main() -> Result<()> {
    // ── PHASE 1: DOWNLOAD PRETRAINED GPT-2 ASSETS ─────────────────────────
    // `model.safetensors` contains neural-network weights.
    // `tokenizer.json` contains GPT-2's production BPE tokenizer.
    println!("Initializing API & downloading GPT-2 weights...");
    let api = Api::new()?;
    let repo = api.model("openai-community/gpt2".to_string());
    let model_file = repo.get("model.safetensors")?;
    let tokenizer_file = repo.get("tokenizer.json")?;

    // ── PHASE 2: LOAD TOKENIZER + MODEL WEIGHTS ───────────────────────────
    // This path is inference-only: no VarMap or optimizer is needed because
    // all tensors come from the checkpoint.
    println!("Loading BPE tokenizer...");
    let tokenizer = Tokenizer::from_file(tokenizer_file).map_err(anyhow::Error::msg)?;

    println!("Mapping weights into memory...");
    let device = candle_core::Device::Cpu;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_file], DType::F32, &device)? };

    println!("Initializing model...");
    let cfg = Gpt2Config::gpt2_small();
    let model = Gpt2Model::load(vb, &cfg)?;

    // ── PHASE 3: ENCODE THE PROMPT ────────────────────────────────────────
    // The tokenizer converts visible text into GPT-2 token ids. Those ids are
    // what the embedding table in the model expects.
    let prompt = "Alan Turing was a mathematician who";
    println!("\nPrompt: {}", prompt);
    let encoding = tokenizer.encode(prompt, true).map_err(anyhow::Error::msg)?;
    let mut input_tokens = encoding.get_ids().to_vec();
    print!("{}", prompt);
    std::io::Write::flush(&mut std::io::stdout())?;

    // ── PHASE 4: AUTOREGRESSIVE GENERATION LOOP ───────────────────────────
    // Each iteration:
    //   1. crop to GPT-2's 1024-token context window if needed
    //   2. run the model and keep only the last-position logits
    //   3. sample one token
    //   4. append and print that token, then repeat
    for _ in 0..30 {
        let seq_len = input_tokens.len();
        let start_idx = if seq_len > cfg.n_positions {
            seq_len - cfg.n_positions
        } else {
            0
        };
        let context_tokens = &input_tokens[start_idx..];
        let tokens_tensor = Tensor::new(context_tokens, &device)?;
        let logits = model.forward(&tokens_tensor)?;

        // The model returns one vocab distribution per input position. The last
        // row is the prediction for the next token after the full context.
        let seq_len = logits.dim(0)?;
        let last_logits = logits.narrow(0, seq_len - 1, 1)?.squeeze(0)?;
        let logits_vec = last_logits.to_vec1::<f32>()?;

        // Sample using temperature=0.8, top_k=50, top_p=0.95. This is less
        // deterministic than greedy decoding but avoids extremely unlikely ids.
        let next_token = sample_next_token(&logits_vec, 0.8, Some(50), Some(0.95)) as u32;
        input_tokens.push(next_token);

        let token_text = tokenizer
            .decode(&[next_token], true)
            .map_err(anyhow::Error::msg)?;
        print!("{}", token_text);
        std::io::Write::flush(&mut std::io::stdout())?;
    }
    println!();
    Ok(())
}
