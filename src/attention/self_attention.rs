// ════════════════════════════════════════════════════════════════════════════
// SCALED DOT-PRODUCT SELF-ATTENTION
// ════════════════════════════════════════════════════════════════════════════
// Each token "looks at" every other token and builds a new representation as a
// weighted blend of all tokens. The weight = how relevant token j is to token i.
//
// Algorithm:
//   1. PROJECT — multiply input X by three learned matrices to get Q, K, V
//   2. SCORE   — Q @ K^T gives raw relevance scores (every query vs every key)
//   3. SCALE   — divide scores by sqrt(d_k) so softmax doesn't saturate
//   4. WEIGHT  — softmax each row → attention weights (each row sums to 1)
//   5. BLEND   — attention_weights @ V → output (weighted sum of value vectors)
//
// Key difference from a plain lookup: the output for token i depends on EVERY
// token in the sequence, not just token i. That's how context flows.
//
// W_Q, W_K, W_V are learned during training. Here they're random — forward pass
// is math-correct but produces meaningless output until back-propagation is added.
// https://sebastianraschka.com/blog/2023/self-attention-from-scratch.html
// https://machinelearningmastery.com/the-attention-mechanism-from-scratch/
// ════════════════════════════════════════════════════════════════════════════

use crate::common::activation::softmax;
use crate::common::util::random_matrix;
// w_q / w_k / w_v: learned projection matrices, shape [d_model][d_k or d_v]
// d_model: width of each input token vector (e.g. 64)
// d_k:     width of query/key vectors — controls score-space dimension
// d_v:     width of value vectors — controls output dimension
pub struct SelfAttention {
    pub w_q: Vec<Vec<f32>>,
    pub w_k: Vec<Vec<f32>>,
    pub w_v: Vec<Vec<f32>>,
    pub d_model: usize, // dimension of the model
    pub d_k: usize,     // dimension of the key
    pub d_v: usize,     // dimension of the value
}

impl SelfAttention {
    // Build attention layer. Weight matrices random-initialized once here and
    // reused for every forward pass (they only change during training).
    pub fn new(d_model: usize, d_k: usize, d_v: usize) -> SelfAttention {
        SelfAttention {
            w_q: random_matrix(d_model, d_k),
            w_k: random_matrix(d_model, d_k),
            w_v: random_matrix(d_model, d_v),
            d_model,
            d_k,
            d_v,
        }
    }

    // Forward pass: x = [seq_len][d_model] → output [seq_len][d_v]
    // Same weights applied to every token; output blends the whole sequence.
    pub fn forward(&self, x: &[Vec<f32>]) -> Vec<Vec<f32>> {
        // ── STEP 1: PROJECT ─────────────────────────────────────────────────
        // Multiply input by each learned matrix to get query/key/value views.
        //   q[i] = "what is token i looking for?"
        //   k[i] = "what does token i offer?"
        //   v[i] = "what does token i actually carry?"
        let q = matmul(&x.to_vec(), &self.w_q);
        let k = matmul(&x.to_vec(), &self.w_k);
        let v = matmul(&x.to_vec(), &self.w_v);

        // ── STEP 2: SCORE ───────────────────────────────────────────────────
        // Attention = softmax(Q @ K^T / sqrt(d_k)) @ V
        // Transpose K so columns become keys, then Q @ K^T gives a
        // [seq_len][seq_len] grid: scores[i][j] = how much query i matches key j.
        let k_t = transpose(&k);
        let scores: Vec<Vec<f32>> = matmul(&q, &k_t);

        // ── STEP 3: SCALE ───────────────────────────────────────────────────
        // Divide by sqrt(d_k). Without this, large d_k makes scores huge,
        // softmax collapses to near one-hot, and gradients vanish.
        let dk_sqrt = (self.d_k as f32).sqrt();
        let mut scaled_scores: Vec<Vec<f32>> = scores
            .iter()
            .map(|row| row.iter().map(|scr| scr / dk_sqrt).collect())
            .collect();
        // ── CAUSAL MASK ─────────────────────────────────────────────────────
        // GPT predicts the next token — so a token must NOT see the future.
        // For each query row i, every key column j > i is "the future":
        // set it to -inf now, before softmax. Since exp(-inf) = 0, softmax
        // turns those into 0 weight. End result: token i only attends to 0..=i.
        //
        //         key0 key1 key2 key3
        //   query0  ok  -inf -inf -inf
        //   query1  ok   ok  -inf -inf
        //   query2  ok   ok   ok  -inf
        //   query3  ok   ok   ok   ok
        //
        // i = query row, j = key column. Inner loop starts at i+1 = first future key.
        for i in 0..scaled_scores.len() {
            for j in (i + 1)..scaled_scores[0].len() {
                scaled_scores[i][j] = -f32::INFINITY;
            }
        }
        // ── STEP 4: WEIGHT ──────────────────────────────────────────────────
        // Softmax each row independently → attention weights. Every row now
        // sums to 1: it's a probability distribution over all tokens.
        let attention_weights: Vec<Vec<f32>> = scaled_scores
            .iter()
            .map(|row| softmax(row))
            .collect::<Vec<Vec<f32>>>();

        // ── STEP 5: BLEND ───────────────────────────────────────────────────
        // weighted sum of values = attention_weights @ V
        // output[i] = Σ_j attention_weights[i][j] * v[j]
        matmul(&attention_weights, &v)
    }
}

// Matrix multiply: a[m][n] @ b[n][p] = result[m][p]
// Triple nested loop — O(m*n*p). Slow but clear.
//   result[i][j] = Σ_k a[i][k] * b[k][j]
pub fn matmul(a: &Vec<Vec<f32>>, b: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    let mut result: Vec<Vec<f32>> = Vec::new();
    for i in 0..a.len() {
        let mut row: Vec<f32> = Vec::new();
        for j in 0..b[0].len() {
            let mut sum = 0.0;
            // walk shared inner dimension, multiply-accumulate
            for k in 0..a[0].len() {
                sum += a[i][k] * b[k][j];
            }
            row.push(sum);
        }
        result.push(row);
    }

    result
}

// Transpose: flip rows and columns. [rows][cols] → [cols][rows]
// transposed[i][j] = matrix[j][i]. Needed to compute Q @ K^T.
pub fn transpose(matrix: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    let rows = matrix.len();
    let cols = matrix[0].len();
    let mut transposed: Vec<Vec<f32>> = Vec::new();
    for i in 0..cols {
        let mut row: Vec<f32> = Vec::new();
        for j in 0..rows {
            row.push(matrix[j][i])
        }
        transposed.push(row);
    }
    transposed
}

// ════════════════════════════════════════════════════════════════════════════
// HOW SELF-ATTENTION WORKS — FULL WALKTHROUGH WITH EXAMPLE
// ════════════════════════════════════════════════════════════════════════════
//
// INPUT: 3 tokens, d_model = 2 (tiny for readability)
//   x = [[1.0, 0.0],     ← token 0
//        [0.0, 1.0],     ← token 1
//        [1.0, 1.0]]     ← token 2
//
// Weight matrices (d_model=2 → d_k=2). Pretend new() produced:
//   w_q = [[1,0],[0,1]]   w_k = [[1,0],[0,1]]   w_v = [[1,0],[0,1]]
//   (identity matrices chosen so the example math stays simple)
//
// ── STEP 1: PROJECT ──────────────────────────────────────────────────────────
// q = x @ w_q,  k = x @ w_k,  v = x @ w_v
// With identity weights, q = k = v = x:
//   q = [[1,0],[0,1],[1,1]]
//   k = [[1,0],[0,1],[1,1]]
//   v = [[1,0],[0,1],[1,1]]
//
// ── STEP 2: SCORE — Q @ K^T ──────────────────────────────────────────────────
// k_t = transpose(k) = [[1,0,1],
//                       [0,1,1]]
//
// scores = q @ k_t   →  scores[i][j] = q[i] · k[j]
//   scores[0] = [ 1·1+0·0, 1·0+0·1, 1·1+0·1 ] = [1, 0, 1]
//   scores[1] = [ 0·1+1·0, 0·0+1·1, 0·1+1·1 ] = [0, 1, 1]
//   scores[2] = [ 1·1+1·0, 1·0+1·1, 1·1+1·1 ] = [1, 1, 2]
//   scores = [[1,0,1],
//             [0,1,1],
//             [1,1,2]]
// Read row 2: "token 2's query matches key 0 by 1, key 1 by 1, key 2 by 2."
//
// ── STEP 3: SCALE — divide by sqrt(d_k) ──────────────────────────────────────
// d_k = 2 → sqrt(2) ≈ 1.414
//   scaled = [[0.71, 0.00, 0.71],
//             [0.00, 0.71, 0.71],
//             [0.71, 0.71, 1.41]]
//
// ── STEP 4: WEIGHT — softmax each row ────────────────────────────────────────
// softmax row 0: e^0.71, e^0, e^0.71 = 2.03, 1.00, 2.03  sum = 5.06
//   → [0.40, 0.20, 0.40]
// softmax row 1: by symmetry → [0.20, 0.40, 0.40]
// softmax row 2: e^0.71, e^0.71, e^1.41 = 2.03, 2.03, 4.10  sum = 8.16
//   → [0.25, 0.25, 0.50]
//
//   attention_weights = [[0.40, 0.20, 0.40],
//                        [0.20, 0.40, 0.40],
//                        [0.25, 0.25, 0.50]]
// Every row sums to 1 — it's a probability distribution over the 3 tokens.
//
// ── STEP 5: BLEND — attention_weights @ V ────────────────────────────────────
// output[i] = Σ_j weights[i][j] * v[j]
//   output[0] = 0.40*[1,0] + 0.20*[0,1] + 0.40*[1,1] = [0.80, 0.60]
//   output[1] = 0.20*[1,0] + 0.40*[0,1] + 0.40*[1,1] = [0.60, 0.80]
//   output[2] = 0.25*[1,0] + 0.25*[0,1] + 0.50*[1,1] = [0.75, 0.75]
//
//   output = [[0.80, 0.60],
//             [0.60, 0.80],
//             [0.75, 0.75]]
//
// Each output token is now a context-aware mix of the whole sequence.
// Same shape as input ([seq_len][d_model] when d_v = d_model) so attention
// layers can be stacked.
//
// ── WHY EACH STEP EXISTS ─────────────────────────────────────────────────────
//   PROJECT  separate "looking for" (Q), "offering" (K), "carrying" (V) roles
//   SCORE    dot product measures similarity between query and key
//   SCALE    keeps softmax in a sane range so gradients don't vanish
//   WEIGHT   softmax turns raw scores into a normalised blend ratio
//   BLEND    output = weighted average of values = context flows between tokens
//
// ── DATA STRUCTURES ──────────────────────────────────────────────────────────
//   w_q / w_k / w_v   Vec<Vec<f32>>   [d_model][d_k|d_v]  learned weights
//   q / k / v         Vec<Vec<f32>>   [seq_len][d_k|d_v]  per-token projections
//   scores            Vec<Vec<f32>>   [seq_len][seq_len]  query·key grid
//   attention_weights Vec<Vec<f32>>   [seq_len][seq_len]  softmaxed, rows sum 1
//   output            Vec<Vec<f32>>   [seq_len][d_v]      context-aware vectors
// ════════════════════════════════════════════════════════════════════════════
