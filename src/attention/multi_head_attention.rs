// ════════════════════════════════════════════════════════════════════════════
// MULTI-HEAD ATTENTION
// ════════════════════════════════════════════════════════════════════════════
// Run several independent self-attention "heads" in parallel, then merge them.
// Each head sees the SAME input but has its own w_q/w_k/w_v, so each learns a
// different relationship (subject-verb, pronoun-referent, adjacency, ...).
//
// Algorithm:
//   1. SPLIT   — each head gets d_k = d_model / num_heads (smaller subspace)
//   2. RUN     — every head does full scaled-dot-product attention on x
//   3. CONCAT  — glue head outputs side by side → back to width d_model
//   4. MIX     — concat @ w_o blends the heads into one coherent output
//
// Cost ≈ same as single-head: 8 heads of width 8 ≈ 1 head of width 64.
// It's a SPLIT of the work, not extra work.
//
// w_o is the only new weight multi-head adds on top of the per-head weights.
// Like all weights here it's random — meaningful only after training.
// https://sebastianraschka.com/blog/2023/self-attention-from-scratch.html
// https://machinelearningmastery.com/the-attention-mechanism-from-scratch/
// ════════════════════════════════════════════════════════════════════════════
use crate::attention::self_attention::SelfAttention;
use crate::common::util::{matmul, random_matrix};
// heads:     num_heads independent SelfAttention layers, each width d_model/num_heads
// w_o:       output projection, shape [num_heads * d_v][d_model] — mixes heads
// num_heads: how many parallel heads (d_model must divide evenly by this)
// d_model:   model width — input and output stay this wide
pub struct MultiHeadAttention {
    pub heads: Vec<SelfAttention>,
    pub w_o: Vec<Vec<f32>>, // [num_heads * d_v][d_model]
    // TODO(backward): cache concatenated head outputs and add d_w_o so the
    // output projection can be trained.
    pub num_heads: usize,
    pub d_model: usize,
}

impl MultiHeadAttention {
    // Build multi-head layer. Each head gets its own random weights; w_o random too.
    pub fn new(d_model: usize, num_heads: usize) -> Self {
        // d_model must split evenly across heads, else concat won't line up
        assert!(
            d_model % num_heads == 0,
            "d_model must be divisible by num_heads"
        );
        let d_k = d_model / num_heads;
        let d_v = d_k; // For simplicity, we set d_v = d_k

        // num_heads separate SelfAttention — each new() call draws fresh random
        // weights, so no two heads start identical (lets them specialize)
        let heads: Vec<SelfAttention> = (0..num_heads)
            .map(|_| SelfAttention::new(d_model, d_k, d_v))
            .collect();

        // w_o maps concatenated heads [num_heads*d_v] back to [d_model]
        let w_o = random_matrix(num_heads * d_v, d_model);

        Self {
            heads,
            w_o: w_o,
            num_heads,
            d_model,
        }
    }

    // Forward pass: x = [seq_len][d_model] → output [seq_len][d_model]
    // Output width matches input width, so multi-head layers can be stacked.
    pub fn forward(&self, x: &[Vec<f32>]) -> Vec<Vec<f32>> {
        // ── PART 1: RUN EACH HEAD ───────────────────────────────────────────
        // Every head runs full attention on the same x, independently.
        //   head_outputs[h] = [seq_len][d_v]
        let head_outputs: Vec<Vec<Vec<f32>>> =
            self.heads.iter().map(|head| head.forward(x)).collect();

        // ── PART 2: CONCAT ──────────────────────────────────────────────────
        // Glue head outputs side by side, per row:
        //   row i = head0[i] ++ head1[i] ++ ... ++ headN[i]
        //   shape goes [seq_len][d_v] * num_heads → [seq_len][num_heads*d_v]
        let seq_len: usize = x.len();
        let mut concatenated: Vec<Vec<f32>> = Vec::new();
        for i in 0..seq_len {
            let mut row: Vec<f32> = Vec::new();
            for h in head_outputs.iter() {
                row.extend(&h[i]);
            }
            concatenated.push(row);
        }

        // ── PART 3: MIX ─────────────────────────────────────────────────────
        // concat @ w_o → [seq_len][d_model]. w_o lets the heads "talk":
        // it learns how to weight and blend the independent views into one.
        matmul(&concatenated, &self.w_o)
    }

    pub fn backward(&mut self, _d_out: &[Vec<f32>]) -> Vec<Vec<f32>> {
        todo!("MultiHeadAttention::backward must split head grads, sum d_x, and compute d_w_o")
    }
}

// ════════════════════════════════════════════════════════════════════════════
// HOW MULTI-HEAD ATTENTION WORKS — FULL WALKTHROUGH WITH EXAMPLE
// ════════════════════════════════════════════════════════════════════════════
//
// SETUP: d_model = 4, num_heads = 2  →  d_k = d_v = 4 / 2 = 2
//
// INPUT: 3 tokens, each width d_model = 4
//   x = [[1, 0, 1, 0],     ← token 0
//        [0, 1, 0, 1],     ← token 1
//        [1, 1, 0, 0]]     ← token 2
//
// ── PART 1: RUN EACH HEAD ────────────────────────────────────────────────────
// Two heads, each is a full SelfAttention with its OWN random w_q/w_k/w_v.
// Each head projects x into width d_k = 2, does scaled-dot-product attention,
// and returns [seq_len][d_v] = [3][2].
//
//   head 0 sees x → its weights → out0 = [[a0, a1],
//                                         [a2, a3],
//                                         [a4, a5]]
//
//   head 1 sees x → DIFFERENT weights → out1 = [[b0, b1],
//                                               [b2, b3],
//                                               [b4, b5]]
//
// Same x both times. Different weights → different attention pattern → out0 ≠ out1.
//
// ── PART 2: CONCAT ───────────────────────────────────────────────────────────
// Glue the two head outputs side by side, row by row:
//   row 0 = out0[0] ++ out1[0] = [a0, a1, b0, b1]
//   row 1 = out0[1] ++ out1[1] = [a2, a3, b2, b3]
//   row 2 = out0[2] ++ out1[2] = [a4, a5, b4, b5]
//
//   concatenated = [[a0, a1, b0, b1],
//                   [a2, a3, b2, b3],
//                   [a4, a5, b4, b5]]
//   shape [3][2] + [3][2] → [3][4]   (back to width d_model)
//
// At this point the heads are just stacked — they don't know about each other.
//
// ── PART 3: MIX ──────────────────────────────────────────────────────────────
// w_o has shape [num_heads*d_v][d_model] = [4][4].
//   output = concatenated @ w_o   →   [3][4]
//
// w_o is a learned matrix: it decides how the 4 concatenated numbers (2 from
// head 0, 2 from head 1) combine into the final 4-wide output per token.
// This is the step where head 0's view and head 1's view actually merge.
//
//   output = [[...4 values...],    ← token 0, context-aware, all heads mixed
//             [...4 values...],    ← token 1
//             [...4 values...]]    ← token 2
//
// ── SINGLE-HEAD vs MULTI-HEAD ────────────────────────────────────────────────
//
//                    Single head (d_model=4)      Multi-head (d_model=4, heads=2)
//                    -------------------------    -------------------------------
//   attention blocks 1, width d_k = 4             2, each width d_k = 2
//   what it learns   one relationship pattern     one pattern PER head (2 total)
//   extra weight     none                         w_o [4][4] to mix heads
//   output shape     [seq][d_model]               [seq][d_model]  (same!)
//   compute cost     1 × (d_k=4)                  2 × (d_k=2)  ≈ same total
//
// ── WHY EACH PART EXISTS ─────────────────────────────────────────────────────
//   SPLIT   small subspaces force each head to specialize, not duplicate
//   RUN     each head independently attends — different w_q/w_k/w_v = different view
//   CONCAT  reassemble per-head views back to full d_model width
//   MIX     w_o blends the independent views — without it heads never interact
//
// ── DATA STRUCTURES ──────────────────────────────────────────────────────────
//   heads          Vec<SelfAttention>   num_heads layers, each width d_model/num_heads
//   w_o            Vec<Vec<f32>>        [num_heads*d_v][d_model]  output mix matrix
//   head_outputs   Vec<Vec<Vec<f32>>>   [num_heads][seq_len][d_v] per-head results
//   concatenated   Vec<Vec<f32>>        [seq_len][num_heads*d_v]  glued heads
//   output         Vec<Vec<f32>>        [seq_len][d_model]        final, heads mixed
// ════════════════════════════════════════════════════════════════════════════
