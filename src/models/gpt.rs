// ════════════════════════════════════════════════════════════════════════════
// GPT — TOKEN EMBEDDINGS + TRANSFORMER DECODER STACK + LANGUAGE MODEL HEAD
// ════════════════════════════════════════════════════════════════════════════
// Autoregressive decoder-only language model.
//
// Algorithm:
//   1. EMBED    — map token IDs to vectors and add positional embeddings
//   2. DECODE   — pass vectors through stacked causal transformer blocks
//   3. NORM     — normalize the final residual stream
//   4. PROJECT  — convert hidden vectors into vocabulary logits
//   5. PREDICT  — use the last position's logits for next-token prediction
//   6. APPEND   — add the selected token and repeat for generation
//
// Key idea: GPT predicts the next token using only previous/current tokens.
// Causal attention prevents the model from reading future positions.
//
// Shape flow:
//   tokens -> embeddings -> decoder blocks -> logits
//   [seq_len] -> [seq_len][d_model] -> [seq_len][vocab_size]
//
// This file wires the architecture only. Useful text requires training.
// https://jaykmody.com/blog/gpt-from-scratch/
// https://sebastianraschka.com/llms-from-scratch/



// kv cacge 
// https://touchdown-labs.com/blog/kv-cache-memory-hierarchy-inference.html
// ════════════════════════════════════════════════════════════════════════════

use crate::common::param::Param;
use crate::common::sampling::sample_next_token;
use crate::common::serilization::SaveableModel;
use crate::common::util::mat_transpose;
use crate::common::util::matmul;
use crate::layers::embedding::{embed_sequence, PositionalEmbedding, TokenEmbedding};
use crate::layers::layer_norm::LayerNorm;
use crate::models::transformer::Transformer;

// token_emb:    token id -> dense vector lookup
// position_emb: position id -> dense vector lookup
// blocks:       stacked causal transformer decoder blocks
// norm:         final layer normalization before vocabulary projection
// lm_head:      derived projection token_emb.weight^T, not a separate Param
//
// Weight tying:
//   The language-model head is always computed from token_emb.weight^T.
//   Gradients from the output projection are transposed back into token_emb.
pub struct GPT {
    pub token_emb: TokenEmbedding,
    pub position_emb: PositionalEmbedding,
    pub blocks: Vec<Transformer>,
    pub norm: LayerNorm,
    // Inference switch for token-by-token generation.
    //
    // false: every forward() call recomputes attention over all input tokens.
    // true:  the prompt pass stores K/V inside every attention head, and later
    //        calls can pass only the newest token while reusing past K/V.
    pub use_kv_cache: bool,

    // ── Activations saved during forward(), used by backward() ──
    // BACKWARD: these caches preserve the forward pass context so gradients
    // can travel back through the same model path.
    cache_embed: Vec<Vec<f32>>,
    cache_blocks: Vec<Vec<Vec<f32>>>,
    cache_norm: Vec<Vec<f32>>,
    // BACKWARD: token IDs from the latest forward pass are needed to scatter
    // bottom-level gradients back into token and position embedding tables.
    cache_tokens: Vec<usize>,
}

impl GPT {
    pub fn new(
        vocab_size: usize,
        d_model: usize,
        max_seq_len: usize,
        num_heads: usize,
        d_ff: usize,
        num_blocks: usize,
    ) -> Self {
        // 1. Token Embeddings: Maps vocabulary IDs to dense vectors of size `d_model`.
        let token_emb = TokenEmbedding::new(vocab_size, d_model);

        // 2. Positional Embeddings: Gives the model a sense of order/sequence position.
        let position_emb = PositionalEmbedding::new(max_seq_len, d_model);

        // 3. Transformer Decoder Blocks: Stacks of self-attention and feed-forward layers.
        let mut blocks = Vec::with_capacity(num_blocks);
        for _ in 0..num_blocks {
            blocks.push(Transformer::new(d_model, num_heads, d_ff));
        }

        // 4. Final layer normalization: stabilizes the last residual stream.
        let norm = LayerNorm::new(d_model);

        Self {
            token_emb,
            position_emb,
            blocks,
            norm,
            use_kv_cache: false,
            cache_blocks: Vec::new(),
            cache_embed: Vec::new(),
            cache_norm: Vec::new(),
            cache_tokens: Vec::new(),
        }
    }

    pub fn set_use_cache(&mut self, use_cache: bool) {
        self.use_kv_cache = use_cache;
        for block in &mut self.blocks {
            block.set_use_cache(use_cache);
        }
    }
    pub fn clear_cache(&mut self) {
        for block in &mut self.blocks {
            block.clear_cache();
        }
    }
    pub fn forward(&mut self, tokens: &[usize]) -> Vec<Vec<f32>> {
        // Clear caches from previous forward pass
        self.cache_embed.clear();
        self.cache_blocks.clear();
        self.cache_norm.clear();
        // BACKWARD: keep token IDs so the embedding layer knows which rows
        // receive gradients during the final scatter step.
        self.cache_tokens = tokens.to_vec();

        // Retrieve position offset from the first block's cached keys if active.
        //
        // When generation feeds only one new token, that token is local row 0
        // in this call, but it is not absolute position 0 in the sequence. If
        // four tokens are already cached, the next token must use position 4:
        //
        //   cached K rows: [token0, token1, token2, token3]
        //   incoming ids: [next_token]
        //   pos_offset:   4
        //   embed row:    token_emb[next_token] + pos_emb[4]
        //
        // We read the first head's cached K length because every block/head is
        // advanced together during generation.
        let pos_offset = if self.use_kv_cache {
            self.blocks[0].mha.heads[0]
                .cache_kv
                .as_ref()
                .map(|(k, _)| k.len())
                .unwrap_or(0)
        } else {
            0
        };

        // ── STEP 1: EMBED TOKEN IDS ─────────────────────────────────────────
        // Combine token identity with absolute position information.
        //
        // Without KV cache:
        //   tokens [seq_len] -> x [seq_len][d_model], positions start at 0.
        //
        // With KV cache after the prompt:
        //   tokens [1] -> x [1][d_model], position starts at cached length.
        //
        // That keeps the new row's positional embedding aligned with the full
        // sequence even though we did not resend the old token ids.
        let mut x = embed_sequence(tokens, &self.token_emb, &self.position_emb, pos_offset);
        self.cache_embed = x.clone();

        // ── STEP 2: RUN THE DECODER STACK ───────────────────────────────────
        // Each block adds causal context and per-token nonlinear processing.
        for block in &mut self.blocks {
            self.cache_blocks.push(x.clone());
            x = block.forward(&x);
        }

        // ── STEP 3: FINAL NORMALIZATION ─────────────────────────────────────
        // GPT-style models normalize before the final vocabulary projection.
        let x = self.norm.forward(&x);
        // BACKWARD: final hidden states are needed for tied lm_head gradient.
        self.cache_norm = x.clone();

        // ── STEP 4: PROJECT TO VOCABULARY LOGITS ────────────────────────────
        // One row of logits per input token position.
        // Weight tying: do not store a separate `lm_head`; derive it from the
        // token embedding table every time so the two cannot drift apart.
        let lm_head = self.token_emb.transposed_weight();
        matmul(&x, &lm_head)
    }

    pub fn backward(&mut self, d_logits: &Vec<Vec<f32>>) {
        // ── STEP 1: LANGUAGE-MODEL HEAD BACKWARD ──────────────────────────
        // Forward:   logits    = cache_norm @ token_emb.weight^T
        // Backward:  d_norm    = d_logits  @ token_emb.weight
        //            d_lm_head = cache_norm^T @ d_logits
        //
        // Because `lm_head` is tied, its gradient is not stored in a separate
        // Param. It is transposed back into token_emb.weight.grad.
        let norm_t = mat_transpose(&self.cache_norm);
        let d_lm_head = matmul(&norm_t, d_logits);
        self.token_emb.add_transposed_grad(&d_lm_head);

        // d_hidden_final_from_logits has shape [seq_len][d_model]
        let lm_head = self.token_emb.transposed_weight();
        let lm_head_t = mat_transpose(&lm_head);
        let d_norm = matmul(d_logits, &lm_head_t);

        // ── STEP 2: LayerNorm backward ──────────────────────────────────────
        // d_norm flows through the final LayerNorm into the decoder stack.
        let mut d_x = self.norm.backward(&d_norm);

        // ── STEP 3: Transformer blocks backward (reversed) ──────────────────
        // Backprop runs from the last decoder block down to the first.
        for block in self.blocks.iter_mut().rev() {
            d_x = block.backward(&d_x);
        }

        // ── STEP 4: Embedding scatter backward ──────────────────────────────
        // Route bottom-level gradients into token rows and position rows.
        let seq_len = self.cache_tokens.len();
        self.token_emb.backward(&self.cache_tokens, &d_x);
        self.position_emb.backward(seq_len, &d_x);
    }

    // Gready old sampling method
    // pub fn generate(&mut self, context: &[usize], max_new_tokens: usize) -> Vec<usize> {
    //     let mut tokens = context.to_vec();
    //     let max_seq_len = self.position_emb.max_seq_len;

    //     for _ in 0..max_new_tokens {
    //         // ── STEP 1: CROP TO THE MODEL'S CONTEXT WINDOW ─────────────────
    //         // Positional embeddings only exist for max_seq_len positions.
    //         let start_idx = if tokens.len() > max_seq_len {
    //             tokens.len() - max_seq_len
    //         } else {
    //             0
    //         };

    //         let cropped_tokens = &tokens[start_idx..];

    //         // ── STEP 2: SCORE THE CURRENT CONTEXT ──────────────────────────
    //         let logits = self.forward(&cropped_tokens);

    //         // ── STEP 3: USE THE LAST POSITION FOR NEXT-TOKEN PREDICTION ────
    //         let last_logits = logits.last().unwrap();

    //         // ── STEP 4: GREEDY DECODE ──────────────────────────────────────
    //         // Pick the highest-scoring vocabulary ID. Sampling comes later.
    //         let mut best_id = 0;
    //         let mut highest_score = f32::NEG_INFINITY;

    //         for (id, score) in last_logits.iter().enumerate() {
    //             if score > &highest_score {
    //                 highest_score = *score;
    //                 best_id = id;
    //             }
    //         }
    //         tokens.push(best_id);
    //     }
    //     tokens
    // }
    /// Autoregressive text generation using advanced sampling strategies (Temperature, Top-K, Top-P)
    pub fn generate_sample(
        &mut self,
        context: &[usize],
        max_new_tokens: usize,
        temperature: f32,
        top_k: Option<usize>,
        top_p: Option<f32>,
    ) -> Vec<usize> {
        let mut tokens = context.to_vec();
        let max_seq_len = self.position_emb.max_seq_len;
        // 1. Turn on and clean the cache.
        //
        // Generation needs persistent K/V state across forward() calls. Start
        // from an empty cache so this prompt does not mix with any previous
        // prompt or earlier generation run.
        self.set_use_cache(true);
        self.clear_cache();

        // Crop prompt context to max_seq_len if it is too long. Positional
        // embeddings only exist for positions 0..max_seq_len-1, so an
        // overlong prompt must be reduced before the first cached pass.
        let start_idx = if tokens.len() > max_seq_len {
            tokens.len() - max_seq_len
        } else {
            0
        };
        let initial_context = &tokens[start_idx..];

        // 2. First pass: process the full prompt context at once.
        //
        // This call fills every attention head's cache:
        //   prompt tokens [prompt_len]
        //   embeddings    [prompt_len][d_model]
        //   cached K/V    [prompt_len][d_k or d_v]
        //   logits        [prompt_len][vocab_size]
        //
        // We use only the last logits row for the first generated token, but
        // the earlier rows are still useful because they created the cache.
        let logits = self.forward(initial_context);
        let last_logits = logits.last().unwrap().clone();
        let mut next_token = sample_next_token(&last_logits, temperature, top_k, top_p);
        tokens.push(next_token);

        for _ in 0..max_new_tokens {
            // ── STEP 1: CROP TO THE MODEL'S CONTEXT WINDOW ─────────────────
            let start_idx = if tokens.len() > max_seq_len {
                tokens.len() - max_seq_len
            } else {
                0
            };

            let last_logits = if start_idx > 0 {
                // If the context window shifted, cached K/V rows no longer
                // match the cropped token window. Clear and rebuild from the
                // current window so positions and visible history stay aligned.
                self.clear_cache();
                let cropped_tokens = &tokens[start_idx..];
                let logits = self.forward(cropped_tokens);
                logits.last().unwrap().clone()
            } else {
                // Normal cached step: feed only the newly generated token.
                //
                // The old non-cached loop would resend the whole growing
                // sequence:
                //   [prompt..., token_a, token_b, ...]
                //
                // With KV cache, old tokens already have saved K/V rows, so
                // this call sends only:
                //   [next_token]
                //
                // Attention then scores this one query against cached old keys
                // plus the new key, producing one logits row for the next
                // sampling decision.
                let logits = self.forward(&[next_token]);
                logits.last().unwrap().clone()
            };

            // ── STEP 2: ADVANCED DECODING AND SAMPLING ─────────────────────
            next_token = sample_next_token(&last_logits, temperature, top_k, top_p);
            tokens.push(next_token);
        }
        // 4. Reset cache states when done
        self.set_use_cache(false);
        self.clear_cache();
        tokens
    }
    /// Standard autoregressive text generation using Greedy Decoding (backwards compatible)
    pub fn generate(&mut self, context: &[usize], max_new_tokens: usize) -> Vec<usize> {
        self.generate_sample(context, max_new_tokens, 0.0, None, None)
    }
}

impl SaveableModel for GPT {
    fn parameters(&mut self) -> Vec<&mut Param> {
        let mut params = Vec::new();
        params.extend(self.token_emb.parameters());
        params.extend(self.position_emb.parameters());
        for block in &mut self.blocks {
            params.extend(block.parameters());
        }
        params.extend(self.norm.parameters());
        params
    }
}

// ════════════════════════════════════════════════════════════════════════════
// HOW GPT WORKS — FULL WALKTHROUGH WITH EXAMPLE
// ════════════════════════════════════════════════════════════════════════════
//
// INPUT TOKENS: [12, 45, 7]
//
// ── PHASE 1: TOKEN + POSITION EMBEDDINGS ───────────────────────────────────
// GPT does not read raw text here. It receives token IDs from a tokenizer.
// Each token ID becomes a learned vector, and each position gets its own vector.
//
//   tokens = [12, 45, 7]
//   positions = [0, 1, 2]
//
//   token_emb[12] + position_emb[0] → x[0]
//   token_emb[45] + position_emb[1] → x[1]
//   token_emb[7]  + position_emb[2] → x[2]
//
// Shape:
//   [seq_len] -> [seq_len][d_model]
//
// Example with seq_len = 3 and d_model = 4:
//
//   x = [
//     [0.12, 0.20, 0.05, 0.31],  ← token 12 at position 0
//     [0.44, 0.18, 0.27, 0.09],  ← token 45 at position 1
//     [0.07, 0.52, 0.13, 0.36],  ← token 7  at position 2
//   ]
//
// ── PHASE 2: STACKED TRANSFORMER DECODER BLOCKS ────────────────────────────
// The embedded sequence is passed through each decoder block.
// Every block keeps the same shape, so blocks can be stacked:
//
//   block1 input  [3][4] -> block1 output [3][4]
//   block2 input  [3][4] -> block2 output [3][4]
//
// Inside each block:
//   1. LayerNorm prepares token vectors
//   2. Causal self-attention lets tokens read earlier/current tokens
//   3. Residual add keeps the original stream
//   4. Feed-forward transforms each token vector
//   5. Another residual add produces the block output
//
// Causal mask:
//   token 0 can read token 0
//   token 1 can read token 0 and token 1
//   token 2 can read token 0, token 1, and token 2
//
// Token 0 cannot read token 1 or token 2. That prevents future-token leakage.
//
// ── PHASE 3: FINAL LAYER NORMALIZATION ─────────────────────────────────────
// After all decoder blocks, GPT normalizes the final hidden states.
//
//   hidden = LayerNorm(block_output)
//
// Shape stays:
//   [seq_len][d_model] -> [seq_len][d_model]
//
// Example:
//
//   hidden = [
//     [-0.80,  0.10,  1.20, -0.50],
//     [ 0.30, -1.10,  0.70,  0.10],
//     [ 1.00, -0.40, -0.20,  0.60],
//   ]
//
// ── PHASE 4: LANGUAGE MODEL HEAD ───────────────────────────────────────────
// The language-model head projects each hidden vector to vocabulary scores.
// These scores are called logits.
//
//   lm_head = token_emb.weight^T
//   logits  = hidden @ lm_head
//
// Shape:
//   hidden  [seq_len][d_model]
//   lm_head [d_model][vocab_size]  derived from token_emb
//   logits  [seq_len][vocab_size]
//
// If vocab_size = 6, logits might look like:
//
//   logits = [
//     [ 0.1,  0.4, -0.2,  0.0,  0.8, -0.5],  ← scores after token 12
//     [-0.3,  0.2,  0.9,  0.1, -0.4,  0.3],  ← scores after token 45
//     [ 0.5, -0.1,  0.2,  1.3,  0.0, -0.2],  ← scores after token 7
//   ]
//
// Each row predicts the next-token distribution for that position.
//
// ── PHASE 5: NEXT-TOKEN PREDICTION ─────────────────────────────────────────
// For generation, GPT only uses the last row of logits.
//
//   last_logits = logits[2]
//               = [0.5, -0.1, 0.2, 1.3, 0.0, -0.2]
//
// Greedy decoding chooses the highest score:
//
//   best_id = 3   because logit[3] = 1.3
//
// The generated sequence becomes:
//
//   [12, 45, 7, 3]
//
// ── PHASE 6: AUTOREGRESSIVE LOOP ───────────────────────────────────────────
// Generation repeats the same process:
//
//   1. Crop tokens to the max context window if needed
//   2. Run forward pass
//   3. Take the last logits row
//   4. Pick the next token
//   5. Append it to the sequence
//
// Example:
//
//   start: [12, 45]
//   step1: [12, 45, 7]
//   step2: [12, 45, 7, 3]
//   step3: [12, 45, 7, 3, 18]
//
// This implementation uses greedy decoding. Later versions can add sampling,
// temperature, top-k, or top-p decoding.
//
// ── SUMMARY: DATA STRUCTURES ───────────────────────────────────────────────
//
//   token_emb     TokenEmbedding       token id -> vector
//   position_emb  PositionalEmbedding  position id -> vector
//   blocks        Vec<Transformer>     stacked causal decoder blocks
//   norm          LayerNorm            final normalization
//   lm_head       token_emb^T          derived projection, not stored separately
//
//   tokens        Vec<usize>           [seq_len]
//   x             Vec<Vec<f32>>        [seq_len][d_model]
//   logits        Vec<Vec<f32>>        [seq_len][vocab_size]
//
// GPT is "autoregressive" because each new token is appended and becomes part
// of the input context for predicting the next token.
// ════════════════════════════════════════════════════════════════════════════
