// ════════════════════════════════════════════════════════════════════════════
// POSITION-WISE FEED-FORWARD NETWORK (FFN)
// ════════════════════════════════════════════════════════════════════════════
// After attention blends token context, each token gets its own private
// "thinking step" — a small two-layer MLP applied independently.
// No token talks to another here; it's purely per-token computation.
//
// Algorithm:
//   1. EXPAND  — project each token from d_model → d_ff (wider hidden layer)
//   2. RELU    — zero out negative values (non-linearity lets it learn curves)
//   3. SHRINK  — project back from d_ff → d_model (restore original width)
//
// Why the expansion? A wider hidden layer gives the network more capacity to
// compute complex per-token features before squashing back. Standard ratio
// in GPT-style models: d_ff = 4 × d_model (e.g. 64 → 256 → 64).
//
// w_1 and w_2 are learned during training. Random here — meaningless output
// until back-propagation is added.
// https://sebastianraschka.com/blog/2023/self-attention-from-scratch.html
// ════════════════════════════════════════════════════════════════════════════
use crate::common::optimizers::Param;
use crate::common::util::{mat_transpose, matmul};
// w_1: expansion weights  [d_model][d_ff]  — widens each token vector
// w_2: shrink weights     [d_ff][d_model]  — restores original width
// d_model: input/output width (same — FFN is shape-preserving)
// d_ff:    hidden width (typically 4 × d_model)
pub struct FeedForward {
    pub w_1: Param, // [d_model][d_ff] expands per token
    pub w_2: Param, // [d_ff][d_model] shrinks per token
    pub d_model: usize,
    pub d_ff: usize,
    // ── Caches saved during forward() ──
    cache_x: Vec<Vec<f32>>,         // Input: [seq_len][d_model]
    cache_hidden: Vec<Vec<f32>>,    // Pre-ReLU: [seq_len][d_ff]
    cache_activated: Vec<Vec<f32>>, // Post-ReLU: [seq_len]
}

impl FeedForward {
    // Build FFN layer. Caller supplies pre-built weight matrices (use
    // random_matrix from attention.rs during development; learned weights
    // from training in production).
    pub fn new(w_1: Vec<Vec<f32>>, w_2: Vec<Vec<f32>>, d_model: usize, d_ff: usize) -> Self {
        Self {
            w_1: Param::new(w_1, vec![vec![0.0; d_ff]; d_model]),
            w_2: Param::new(w_2, vec![vec![0.0; d_model]; d_ff]),
            d_model,
            d_ff,
            cache_x: Vec::new(),
            cache_hidden: Vec::new(),
            cache_activated: Vec::new(),
        }
    }

    // Forward pass: x = [seq_len][d_model] → output [seq_len][d_model]
    // Shape is preserved — FFN can be stacked or composed with attention freely.
    pub fn forward(&mut self, x: &[Vec<f32>]) -> Vec<Vec<f32>> {
        self.cache_x = x.to_vec();

        // ── STEP 1: EXPAND ──────────────────────────────────────────────────
        // Each token row: [d_model] @ w_1[d_model][d_ff] → [d_ff]
        // Applied to every row at once via matmul → [seq_len][d_ff]
        // No mixing between tokens — every row is independent.
        let hidden = matmul(&x.to_vec(), &self.w_1.data);
        self.cache_hidden = hidden.clone();
        // ── STEP 2: RELU ────────────────────────────────────────────────────
        // ReLU(x) = max(0, x) — element-wise, no weights involved.
        // Negative values become 0; positive values pass through unchanged.
        // Without this the whole FFN would be linear (just two matrix mults),
        // which collapses to a single matrix and loses expressive power.
        let activated: Vec<Vec<f32>> = hidden
            .iter()
            .map(|row| row.iter().map(|v| v.max(0.0)).collect())
            .collect();
        self.cache_activated = activated.clone();
        // ── STEP 3: SHRINK ──────────────────────────────────────────────────
        // Each activated row: [d_ff] @ w_2[d_ff][d_model] → [d_model]
        // Applied to every row via matmul → back to [seq_len][d_model]
        matmul(&activated, &self.w_2.data)
    }

    // Backward pass: d_out = [seq_len][d_model] -> d_x = [seq_len][d_model]
    //
    // This walks the forward algorithm in reverse:
    //   forward  STEP 1 EXPAND -> STEP 2 RELU -> STEP 3 SHRINK
    //   backward STEP 3 SHRINK -> STEP 2 RELU -> STEP 1 EXPAND
    //
    // The forward pass cached both the pre-ReLU hidden values and the
    // post-ReLU activated values. Backward needs both: activated values train
    // w_2, while pre-ReLU hidden values decide where ReLU passes gradient.
    pub fn backward(&mut self, d_out: &[Vec<f32>]) -> Vec<Vec<f32>> {
        // ── BACKWARD: REVERSE FORWARD STEP 3 (SHRINK) ──────────────────────
        // Forward STEP 3 did:
        //   output = activated @ W_2
        // Meaning:
        //   each token's hidden vector [d_ff] was projected back to d_model so
        //   the FFN output can plug into the residual transformer stack.
        //
        // Since output depended on both activated and W_2:
        //   d_w2        = activated^T @ d_out
        //   d_activated = d_out @ W_2^T
        //
        // Shapes:
        //   activated^T [d_ff][seq_len]
        //   d_out       [seq_len][d_model]
        //   d_w2        [d_ff][d_model]
        let activated_t = mat_transpose(&self.cache_activated);
        let batch_d_w2 = matmul(&activated_t, &d_out.to_vec());
        let w_2_t = mat_transpose(&self.w_2.data);
        let d_activated = matmul(&d_out.to_vec(), &w_2_t);
        for i in 0..self.d_ff {
            for j in 0..self.d_model {
                self.w_2.grad[i][j] += batch_d_w2[i][j];
            }
        }

        // ── BACKWARD: REVERSE FORWARD STEP 2 (RELU) ────────────────────────
        // Forward STEP 2 did:
        //   activated = max(0, hidden)
        // Meaning:
        //   positive hidden values passed through unchanged, while negative
        //   hidden values were clipped to 0.
        //
        // ReLU backward uses the cached pre-ReLU hidden value:
        //   if hidden > 0: gradient passes through
        //   else:          gradient becomes 0
        let mut d_hidden = vec![vec![0.0; self.d_ff]; d_out.len()];
        for i in 0..d_out.len() {
            for j in 0..self.d_ff {
                if self.cache_hidden[i][j] > 0.0 {
                    d_hidden[i][j] = d_activated[i][j];
                }
            }
        }

        // ── BACKWARD: REVERSE FORWARD STEP 1 (EXPAND) ──────────────────────
        // Forward STEP 1 did:
        //   hidden = X @ W_1
        // Meaning:
        //   each token row was widened from d_model to d_ff so the FFN had more
        //   capacity before shrinking back.
        //
        // Since hidden depended on both X and W_1:
        //   d_w1 = X^T @ d_hidden
        //   d_x  = d_hidden @ W_1^T
        //
        // d_x is returned to the previous layer; d_w1 accumulates for the
        // optimizer step.
        let w_1_t = mat_transpose(&self.w_1.data);
        let d_x = matmul(&d_hidden, &w_1_t);
        let x_t = mat_transpose(&self.cache_x);
        let batch_d_w1 = matmul(&x_t, &d_hidden);
        for i in 0..self.d_model {
            for j in 0..self.d_ff {
                self.w_1.grad[i][j] += batch_d_w1[i][j];
            }
        }
        d_x
    }
    pub fn parameters(&mut self) -> Vec<&mut Param> {
        vec![&mut self.w_1, &mut self.w_2]
    }
}

// ════════════════════════════════════════════════════════════════════════════
// HOW THE FFN WORKS — FULL WALKTHROUGH WITH EXAMPLE
// ════════════════════════════════════════════════════════════════════════════
//
// SETUP: d_model = 2, d_ff = 4  (4 = 2 × d_model; 4× in practice)
//
// INPUT: 2 tokens, each a d_model = 2 vector
//   x = [[1.0, -1.0],    ← token 0
//        [0.5,  0.5]]    ← token 1
//
// Pretend weights (small numbers for readable arithmetic):
//   w_1 = [[1, 0, -1, 0],    shape [2][4]
//           [0, 1,  0, -1]]
//
//   w_2 = [[1, 0],    shape [4][2]
//           [0, 1],
//           [1, 0],
//           [0, 1]]
//
// ── STEP 1: EXPAND — x @ w_1 ─────────────────────────────────────────────
//
// token 0: [1, -1] @ w_1
//   col 0: 1*1 + (-1)*0 =  1
//   col 1: 1*0 + (-1)*1 = -1
//   col 2: 1*(-1) + (-1)*0 = -1
//   col 3: 1*0 + (-1)*(-1) =  1
//   → hidden[0] = [1, -1, -1, 1]
//
// token 1: [0.5, 0.5] @ w_1
//   col 0: 0.5*1 + 0.5*0  = 0.5
//   col 1: 0.5*0 + 0.5*1  = 0.5
//   col 2: 0.5*(-1) + 0.5*0 = -0.5
//   col 3: 0.5*0 + 0.5*(-1) = -0.5
//   → hidden[1] = [0.5, 0.5, -0.5, -0.5]
//
//   hidden = [[1,   -1,   -1,   1  ],
//             [0.5,  0.5, -0.5, -0.5]]
//
// ── STEP 2: RELU — max(0, x) element-wise ────────────────────────────────
//
//   activated = [[1,   0,   0,   1  ],   ← negatives zeroed out
//                [0.5, 0.5, 0,   0  ]]
//
// ── STEP 3: SHRINK — activated @ w_2 ─────────────────────────────────────
//
// token 0: [1, 0, 0, 1] @ w_2
//   col 0: 1*1 + 0*0 + 0*1 + 1*0 = 1
//   col 1: 1*0 + 0*1 + 0*0 + 1*1 = 1
//   → output[0] = [1, 1]
//
// token 1: [0.5, 0.5, 0, 0] @ w_2
//   col 0: 0.5*1 + 0.5*0 + 0 + 0 = 0.5
//   col 1: 0.5*0 + 0.5*1 + 0 + 0 = 0.5
//   → output[1] = [0.5, 0.5]
//
//   output = [[1,   1  ],    ← shape [2][2] = [seq_len][d_model] ✓
//             [0.5, 0.5]]
//
// ── WHY EACH STEP EXISTS ─────────────────────────────────────────────────
//   EXPAND   more hidden units → more capacity to represent complex functions
//   RELU     non-linearity — without it, two matrix mults collapse to one
//   SHRINK   restore d_model width so FFN output plugs back into the stack
//
// ── WHERE FFN SITS IN THE TRANSFORMER BLOCK ──────────────────────────────
//   x → [MHA] → add x → LayerNorm → [FFN] → add → LayerNorm → output
//                ↑ residual                    ↑ residual
//
// Attention handles "which tokens matter to each other."
// FFN handles "what to compute given what I've learned so far."
// Together they cover both inter-token and per-token processing.
//
// ── DATA STRUCTURES ──────────────────────────────────────────────────────
//   w_1        Vec<Vec<f32>>   [d_model][d_ff]   expansion weights
//   w_2        Vec<Vec<f32>>   [d_ff][d_model]   shrink weights
//   x          &[Vec<f32>]     [seq_len][d_model] input (borrowed slice)
//   hidden     Vec<Vec<f32>>   [seq_len][d_ff]   after expansion
//   activated  Vec<Vec<f32>>   [seq_len][d_ff]   after ReLU
//   output     Vec<Vec<f32>>   [seq_len][d_model] final, shape = input
// ════════════════════════════════════════════════════════════════════════════
