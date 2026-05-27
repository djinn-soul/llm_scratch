// ════════════════════════════════════════════════════════════════════════════
// TRANSFORMER DECODER BLOCK — PRE-NORM GPT-STYLE BLOCK
// ════════════════════════════════════════════════════════════════════════════
// Reusable decoder block used inside GPT.
//
// Algorithm:
//   1. NORM       — normalize the residual stream before attention
//   2. ATTEND     — causal multi-head attention mixes earlier token context
//   3. RESIDUAL   — add the attention update back to the original stream
//   4. NORM       — normalize again before the feed-forward network
//   5. MLP        — transform each token vector independently
//   6. RESIDUAL   — add the feed-forward update back to the stream
//
// Key idea: attention mixes information ACROSS tokens, while feed-forward mixes
// information WITHIN each token vector. Residual connections keep the original
// signal flowing so deeper stacks remain trainable.
//
// Layout:
//   x -> LayerNorm -> CausalAttention -> ResidualAdd
//     -> LayerNorm -> FeedForward     -> ResidualAdd
// ════════════════════════════════════════════════════════════════════════════

use crate::attention::multi_head_attention::MultiHeadAttention;
use crate::common::util::{add_mat, random_matrix};
use crate::layers::feed_forward::FeedForward;
use crate::layers::layer_norm::LayerNorm;

// layer_norm:  first pre-norm before self-attention
// mha:         multi-head causal self-attention
// layer_norm2: second pre-norm before feed-forward
// ff:          position-wise MLP applied to each token vector
//
// Shape stays stable so blocks can be stacked:
//   [seq_len][d_model] -> [seq_len][d_model]
pub struct Transformer {
    pub layer_norm: LayerNorm,
    pub mha: MultiHeadAttention,
    pub layer_norm2: LayerNorm,
    pub ff: FeedForward,
    // BACKWARD: residual adds have derivative 1 on both branches, while
    // LayerNorm, attention, and feed-forward keep their own forward caches.
    // This block only has to route and merge gradients in reverse order.
}

impl Transformer {
    pub fn new(d_model: usize, num_heads: usize, d_ff: usize) -> Self {
        // Feed-forward expands d_model -> d_ff, then projects back d_ff -> d_model.
        let w1 = random_matrix(d_model, d_ff);
        let w2 = random_matrix(d_ff, d_model);
        Self {
            layer_norm: LayerNorm::new(d_model),
            mha: MultiHeadAttention::new(d_model, num_heads),
            layer_norm2: LayerNorm::new(d_model),
            ff: FeedForward::new(w1, w2, d_model, d_ff),
        }
    }

    pub fn forward(&mut self, x: &[Vec<f32>]) -> Vec<Vec<f32>> {
        // ── STEP 1: NORMALIZE BEFORE ATTENTION ──────────────────────────────
        // Pre-norm stabilizes deep transformer stacks because each sublayer
        // receives normalized activations.
        let norm1 = self.layer_norm.forward(x);

        // ── STEP 2: CAUSAL MULTI-HEAD SELF-ATTENTION ───────────────────────
        // Each token gathers context from earlier tokens through multiple heads.
        let attention = self.mha.forward(&norm1);

        // ── STEP 3: FIRST RESIDUAL CONNECTION ──────────────────────────────
        // Keep the original token signal and add the attention update.
        let h = add_mat(x, &attention);

        // ── STEP 4: NORMALIZE BEFORE FEED-FORWARD ──────────────────────────
        // The feed-forward network sees a normalized residual stream.
        let norm2 = self.layer_norm2.forward(&h);

        // ── STEP 5: POSITION-WISE FEED-FORWARD ─────────────────────────────
        // Applied to each token independently; no cross-token mixing here.
        let ff = self.ff.forward(&norm2);

        // ── STEP 6: SECOND RESIDUAL CONNECTION ─────────────────────────────
        // Output shape remains [seq_len][d_model], so blocks can be stacked.
        add_mat(&h, &ff)
    }

    pub fn backward(&mut self, d_output: &[Vec<f32>]) -> Vec<Vec<f32>> {
        // STEP 1: reverse the second residual add.
        // output = h + ff, so d_output flows into both the direct residual
        // branch and the feed-forward branch.
        let d_ff = d_output.to_vec();
        let d_norm2 = self.ff.backward(&d_ff);

        // STEP 2: move the feed-forward branch through the second LayerNorm.
        let d_h_from_norm2 = self.layer_norm2.backward(&d_norm2);

        // STEP 3: merge gradients for the intermediate residual stream `h`.
        // d_h = d_output from the residual branch + d_h_from_norm2 from MLP.
        let d_h = add_mat(&d_output.to_vec(), &d_h_from_norm2);

        // STEP 4: reverse the first residual add and attention branch.
        // h = x + attention, so d_h flows into attention and the direct x path.
        let d_attention = d_h.clone();
        let d_norm1 = self.mha.backward(&d_attention);

        // STEP 5: move the attention branch through the first LayerNorm.
        let d_x_norm1 = self.layer_norm.backward(&d_norm1);

        // STEP 6: merge gradients for the block input `x`.
        // d_x = d_h from the residual branch + d_x_norm1 from attention.
        add_mat(&d_h, &d_x_norm1)
    }
}

// ════════════════════════════════════════════════════════════════════════════
// HOW A TRANSFORMER DECODER BLOCK WORKS — FULL WALKTHROUGH WITH EXAMPLE
// ════════════════════════════════════════════════════════════════════════════
//
// INPUT:
//   x = sequence of token vectors after token + positional embeddings.
//
// Example shape:
//   seq_len = 3, d_model = 4
//
//   x = [
//     [0.10, 0.20, 0.30, 0.40],  ← token 0
//     [0.50, 0.60, 0.70, 0.80],  ← token 1
//     [0.90, 1.00, 1.10, 1.20],  ← token 2
//   ]
//
// ── PHASE 1: PRE-NORM BEFORE ATTENTION ─────────────────────────────────────
// Normalize each token vector independently:
//
//   norm1 = LayerNorm(x)
//
// Why? LayerNorm keeps each token vector in a stable range before attention.
// In pre-norm transformers, normalization happens before each sublayer instead
// of after the residual add.
//
// Shape stays the same:
//   [seq_len][d_model] -> [seq_len][d_model]
//
// Example normalized matrix (illustrative values):
//
//   norm1 = [
//     [-1.34, -0.45,  0.45,  1.34],
//     [-1.34, -0.45,  0.45,  1.34],
//     [-1.34, -0.45,  0.45,  1.34],
//   ]
//
// Each row is normalized independently. The rows look identical here because
// the example rows have the same spacing pattern.
//
// ── PHASE 2: MULTI-HEAD CAUSAL SELF-ATTENTION ──────────────────────────────
// Attention lets each token pull information from other allowed tokens.
// In GPT-style causal attention:
//
//   token 0 can read token 0
//   token 1 can read token 0 and token 1
//   token 2 can read token 0, token 1, and token 2
//
// It must not read future tokens because next-token prediction would leak the
// answer during training.
//
// Multi-head attention runs several attention heads in parallel:
//
//   head 0 might track local syntax
//   head 1 might track repeated words
//   head 2 might track subject/object relationships
//
// In this learning implementation the weights are random, so the math is wired
// correctly but the heads have not learned useful behavior yet.
//
// Result:
//   attention = MultiHeadAttention(norm1)
//   attention shape = [seq_len][d_model]
//
// Example attention update matrix:
//
//   attention = [
//     [ 0.03, -0.02,  0.01,  0.04],  ← token 0 update from token 0 only
//     [ 0.02, -0.01,  0.05,  0.03],  ← token 1 update from tokens 0..1
//     [-0.01,  0.04,  0.02,  0.06],  ← token 2 update from tokens 0..2
//   ]
//
// Matrix shape:
//   norm1     [3][4]
//   attention [3][4]
//
// ── PHASE 3: FIRST RESIDUAL ADD ────────────────────────────────────────────
// Add the attention update back to the original stream:
//
//   h = x + attention
//
// The residual path means the block does not have to recreate the original
// token vector from scratch. It only learns an update to add on top.
//
// Example:
//   x[1]         = [0.50, 0.60, 0.70, 0.80]
//   attention[1] = [0.02, -0.01, 0.05, 0.03]
//   h[1]         = [0.52, 0.59, 0.75, 0.83]
//
// Full residual matrix:
//
//   h = x + attention
//
//   h = [
//     [0.13, 0.18, 0.31, 0.44],
//     [0.52, 0.59, 0.75, 0.83],
//     [0.89, 1.04, 1.12, 1.26],
//   ]
//
// ── PHASE 4: PRE-NORM BEFORE FEED-FORWARD ──────────────────────────────────
// Normalize the residual stream again:
//
//   norm2 = LayerNorm(h)
//
// This prepares each token vector for the feed-forward network.
//
// Example:
//
//   norm2 = [
//     [-1.12, -0.72,  0.32,  1.52],
//     [-1.23, -0.79,  0.50,  1.52],
//     [-1.31, -0.38,  0.12,  1.56],
//   ]
//
// ── PHASE 5: POSITION-WISE FEED-FORWARD NETWORK ────────────────────────────
// Feed-forward is an MLP applied to each token independently:
//
//   ff = activation(norm2 @ W1) @ W2
//
// Shape flow:
//   [seq_len][d_model]
//       @ W1 [d_model][d_ff]
//   -> [seq_len][d_ff]
//       @ W2 [d_ff][d_model]
//   -> [seq_len][d_model]
//
// Tiny example with d_ff = 8:
//
//   W1 shape = [4][8]
//   W2 shape = [8][4]
//
//   norm2 @ W1:
//
//   hidden = [
//     [0.10, 0.00, 0.42, 0.18, 0.00, 0.31, 0.07, 0.00],
//     [0.14, 0.00, 0.38, 0.21, 0.00, 0.27, 0.11, 0.00],
//     [0.08, 0.02, 0.46, 0.16, 0.00, 0.35, 0.05, 0.00],
//   ]
//
// ReLU/activation keeps positive values and zeros out negative values.
//
//   hidden @ W2:
//
//   ff = [
//     [ 0.01,  0.03, -0.02,  0.04],
//     [ 0.00,  0.02, -0.01,  0.05],
//     [ 0.02,  0.01, -0.03,  0.03],
//   ]
//
// Attention mixed information between tokens. Feed-forward now transforms each
// token's own feature vector after that context has been added.
//
// ── PHASE 6: SECOND RESIDUAL ADD ───────────────────────────────────────────
// Add the feed-forward update back to the residual stream:
//
//   output = h + ff
//
// The output has the same shape as the input:
//
//   [seq_len][d_model]
//
// Full final matrix:
//
//   output = h + ff
//
//   output = [
//     [0.14, 0.21, 0.29, 0.48],
//     [0.52, 0.61, 0.74, 0.88],
//     [0.91, 1.05, 1.09, 1.29],
//   ]
//
// That is why transformer blocks can be stacked:
//
//   block1 output -> block2 input -> block3 input -> ...
//
// ── SUMMARY: DATA FLOW ─────────────────────────────────────────────────────
//
//   x
//   ├─ LayerNorm ─ MultiHeadAttention ─┐
//   └────────────────────────────────── +  = h
//                                      │
//   h
//   ├─ LayerNorm ─ FeedForward ────────┐
//   └────────────────────────────────── +  = output
//
// ── DATA STRUCTURES ────────────────────────────────────────────────────────
//
//   layer_norm   LayerNorm            normalizes before attention
//   mha          MultiHeadAttention   context mixing across tokens
//   layer_norm2  LayerNorm            normalizes before feed-forward
//   ff           FeedForward          per-token nonlinear projection
//
//   x            Vec<Vec<f32>>        [seq_len][d_model]
//   attention    Vec<Vec<f32>>        [seq_len][d_model]
//   h            Vec<Vec<f32>>        [seq_len][d_model]
//   ff           Vec<Vec<f32>>        [seq_len][d_model]
//   output       Vec<Vec<f32>>        [seq_len][d_model]
// ════════════════════════════════════════════════════════════════════════════
