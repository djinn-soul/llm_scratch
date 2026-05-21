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
use crate::common::util::matmul;

// w_1: expansion weights  [d_model][d_ff]  — widens each token vector
// w_2: shrink weights     [d_ff][d_model]  — restores original width
// d_model: input/output width (same — FFN is shape-preserving)
// d_ff:    hidden width (typically 4 × d_model)
pub struct FeedForward {
    pub w_1: Vec<Vec<f32>>, // [d_model][d_ff] expands per token
    pub w_2: Vec<Vec<f32>>, // [d_ff][d_model] shrinks per token
    // TODO(backward): store forward caches and gradients for w_1/w_2 so the
    // training loop can update both FFN matrices.
    pub d_model: usize,
    pub d_ff: usize,
}

impl FeedForward {
    // Build FFN layer. Caller supplies pre-built weight matrices (use
    // random_matrix from attention.rs during development; learned weights
    // from training in production).
    pub fn new(w_1: Vec<Vec<f32>>, w_2: Vec<Vec<f32>>, d_model: usize, d_ff: usize) -> Self {
        Self {
            w_1,
            w_2,
            d_model,
            d_ff,
        }
    }

    // Forward pass: x = [seq_len][d_model] → output [seq_len][d_model]
    // Shape is preserved — FFN can be stacked or composed with attention freely.
    pub fn forward(&self, x: &[Vec<f32>]) -> Vec<Vec<f32>> {
        // ── STEP 1: EXPAND ──────────────────────────────────────────────────
        // Each token row: [d_model] @ w_1[d_model][d_ff] → [d_ff]
        // Applied to every row at once via matmul → [seq_len][d_ff]
        // No mixing between tokens — every row is independent.
        let hidden = matmul(&x.to_vec(), &self.w_1);

        // ── STEP 2: RELU ────────────────────────────────────────────────────
        // ReLU(x) = max(0, x) — element-wise, no weights involved.
        // Negative values become 0; positive values pass through unchanged.
        // Without this the whole FFN would be linear (just two matrix mults),
        // which collapses to a single matrix and loses expressive power.
        let activated: Vec<Vec<f32>> = hidden
            .iter()
            .map(|row| row.iter().map(|v| v.max(0.0)).collect())
            .collect();

        // ── STEP 3: SHRINK ──────────────────────────────────────────────────
        // Each activated row: [d_ff] @ w_2[d_ff][d_model] → [d_model]
        // Applied to every row via matmul → back to [seq_len][d_model]
        matmul(&activated, &self.w_2)
    }

    // TODO(backward): implement reverse pass:
    // d_out -> d_w2 + d_relu -> d_hidden -> d_w1 + d_x.
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
