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
// ════════════════════════════════════════════════════════════════════════════

use crate::common::optimizers::Param;
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
// lm_head:      hidden vector -> vocabulary logits projection
//
// Weight tying: lm_head reuses token_emb as a transposed matrix.
pub struct GPT {
    pub token_emb: TokenEmbedding,
    pub position_emb: PositionalEmbedding,
    pub blocks: Vec<Transformer>,
    pub norm: LayerNorm,
    pub lm_head: Param,
    // TODO(backward): decide whether lm_head stays tied to token_emb weights
    // during updates, then route lm_head gradients into token embedding grads.

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

        // 5. Language-model head: ties output projection to token embeddings. (weight tying)
        let lm_head_data = token_emb.transposed_weight();

        Self {
            token_emb,
            position_emb,
            blocks,
            norm,
            lm_head: Param::new(lm_head_data, vec![vec![0.0; vocab_size]; d_model]),
            cache_blocks: Vec::new(),
            cache_embed: Vec::new(),
            cache_norm: Vec::new(),
            cache_tokens: Vec::new(),
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

        // ── STEP 1: EMBED TOKEN IDS ─────────────────────────────────────────
        // Combine token identity with absolute position information.
        let mut x = embed_sequence(tokens, &self.token_emb, &self.position_emb);
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
        // BACKWARD: final hidden states are needed for lm_head_grad.
        self.cache_norm = x.clone();

        // ── STEP 4: PROJECT TO VOCABULARY LOGITS ────────────────────────────
        // One row of logits per input token position.
        matmul(&x, &self.lm_head.data)
    }

    pub fn backward(&mut self, d_logits: &Vec<Vec<f32>>) {
        // ── STEP 1: LANGUAGE-MODEL HEAD BACKWARD ──────────────────────────
        // Forward:   logits    = cache_norm @ lm_head
        // Backward:  d_norm    = d_logits  @ lm_head^T   (flows left)
        //            d_lm_head = cache_norm^T @ d_logits  (weight gradient)
        let norm_t = mat_transpose(&self.cache_norm);
        self.lm_head.grad = matmul(&norm_t, d_logits);

        // d_hidden_final_from_logits has shape [seq_len][d_model]
        let lm_head_t = mat_transpose(&self.lm_head.data);
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

    pub fn generate(&mut self, context: &[usize], max_new_tokens: usize) -> Vec<usize> {
        let mut tokens = context.to_vec();
        let max_seq_len = self.position_emb.max_seq_len;

        for _ in 0..max_new_tokens {
            // ── STEP 1: CROP TO THE MODEL'S CONTEXT WINDOW ─────────────────
            // Positional embeddings only exist for max_seq_len positions.
            let start_idx = if tokens.len() > max_seq_len {
                tokens.len() - max_seq_len
            } else {
                0
            };

            let cropped_tokens = &tokens[start_idx..];

            // ── STEP 2: SCORE THE CURRENT CONTEXT ──────────────────────────
            let logits = self.forward(&cropped_tokens);

            // ── STEP 3: USE THE LAST POSITION FOR NEXT-TOKEN PREDICTION ────
            let last_logits = logits.last().unwrap();

            // ── STEP 4: GREEDY DECODE ──────────────────────────────────────
            // Pick the highest-scoring vocabulary ID. Sampling comes later.
            let mut best_id = 0;
            let mut highest_score = f32::NEG_INFINITY;

            for (id, score) in last_logits.iter().enumerate() {
                if score > &highest_score {
                    highest_score = *score;
                    best_id = id;
                }
            }
            tokens.push(best_id);
        }
        tokens
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
        params.push(&mut self.lm_head);
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
//   logits = hidden @ lm_head
//
// Shape:
//   hidden  [seq_len][d_model]
//   lm_head [d_model][vocab_size]
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
//   lm_head       Vec<Vec<f32>>        hidden vector -> vocab logits
//
//   tokens        Vec<usize>           [seq_len]
//   x             Vec<Vec<f32>>        [seq_len][d_model]
//   logits        Vec<Vec<f32>>        [seq_len][vocab_size]
//
// GPT is "autoregressive" because each new token is appended and becomes part
// of the input context for predicting the next token.
// ════════════════════════════════════════════════════════════════════════════
