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

// KV-CACHE
// During generation, the prompt is processed once and every attention head saves
// the prompt's K and V rows. Later steps pass only the new token into forward().
// That new token still builds its own Q/K/V, but the new K/V rows are appended
// to the saved past K/V matrix before scoring:
//
//   prompt pass:    Q,K,V all have shape [prompt_len][d_k or d_v]
//   next-token pass Q has shape [1][d_k]
//                   K,V become [past_len + 1][d_k or d_v]
//   scores = Q @ K^T becomes [1][past_len + 1]
//
// So the expensive old token projections are reused, while the new token can
// still attend over the full visible history.
//
// Tiny row-level example with d_k = d_v = 2:
//
//   After the prompt "A B C D", the cache holds old keys and values:
//
//     past_k = [
//       k0 = [0.10, 0.20],   // token A
//       k1 = [0.30, 0.40],   // token B
//       k2 = [0.50, 0.60],   // token C
//       k3 = [0.70, 0.80],   // token D
//     ]
//
//     past_v = [
//       v0 = [1.00, 1.10],
//       v1 = [1.20, 1.30],
//       v2 = [1.40, 1.50],
//       v3 = [1.60, 1.70],
//     ]
//
//   Now generation sends only the new token "E":
//
//     x_new = [[...d_model values for E...]]
//     q_new = [[q4a, q4b]]     // shape [1][2]
//     k_new = [[k4a, k4b]]     // shape [1][2]
//     v_new = [[v4a, v4b]]     // shape [1][2]
//
//   The cache append builds full K/V for the visible history:
//
//     full_k = [k0, k1, k2, k3, k4]    // shape [5][2]
//     full_v = [v0, v1, v2, v3, v4]    // shape [5][2]
//
//   Then scoring uses the one new query against all visible keys:
//
//     full_k^T = [
//       [k0a, k1a, k2a, k3a, k4a],
//       [k0b, k1b, k2b, k3b, k4b],
//     ]                                // shape [2][5]
//
//     scores = q_new @ full_k^T
//            = [[
//                q4 dot k0,
//                q4 dot k1,
//                q4 dot k2,
//                q4 dot k3,
//                q4 dot k4,
//              ]]                      // shape [1][5]
//
//   After softmax, weights also have shape [1][5]:
//
//     attention_weights = [[w0, w1, w2, w3, w4]]
//
//   The final blend reads every cached value row:
//
//     output =
//       w0*v0 + w1*v1 + w2*v2 + w3*v3 + w4*v4
//     output shape = [1][2]
//
// That is the whole cache idea: compute only q4/k4/v4 now, but still use
// k0..k4 and v0..v4 as if the full sequence had been sent again.
// https://medium.com/@joaolages/kv-caching-explained-276520203249
// https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/

// ════════════════════════════════════════════════════════════════════════════

use crate::common::activation::softmax;
use crate::common::param::Param;
use crate::common::util::{mat_transpose, matmul, random_matrix};

// w_q / w_k / w_v: learned projection matrices, shape [d_model][d_k or d_v]
// d_model: width of each input token vector (e.g. 64)
// d_k:     width of query/key vectors — controls score-space dimension
// d_v:     width of value vectors — controls output dimension
pub struct SelfAttention {
    pub w_q: Param,
    pub w_k: Param,
    pub w_v: Param,
    pub d_model: usize, // dimension of the model
    pub d_k: usize,     // dimension of the key
    pub d_v: usize,     // dimension of the value

    // Forward caches needed by backward(): X, Q, K, V, and masked softmax
    // weights from the most recent forward() call.
    cache_x: Vec<Vec<f32>>,
    cache_q: Vec<Vec<f32>>,
    cache_k: Vec<Vec<f32>>,
    cache_v: Vec<Vec<f32>>,
    cache_attention_weights: Vec<Vec<f32>>,
    // --- KV CACHE FOR INFERENCE ---
    // cache_kv stores the full historical key/value tables for this attention
    // head. It is inference-only state: training/backward still uses the normal
    // per-forward caches above.
    pub use_cache: bool,
    pub cache_kv: Option<(Vec<Vec<f32>>, Vec<Vec<f32>>)>, // (cached_K, cached_V)
}

impl SelfAttention {
    // Build attention layer. Weight matrices random-initialized once here and
    // reused for every forward pass (they only change during training).
    pub fn new(d_model: usize, d_k: usize, d_v: usize) -> SelfAttention {
        SelfAttention {
            w_q: Param::new(random_matrix(d_model, d_k)),
            w_k: Param::new(random_matrix(d_model, d_k)),
            w_v: Param::new(random_matrix(d_model, d_v)),
            d_model,
            d_k,
            d_v,
            cache_x: Vec::new(),
            cache_q: Vec::new(),
            cache_k: Vec::new(),
            cache_v: Vec::new(),
            cache_attention_weights: Vec::new(),
            use_cache: false,
            cache_kv: None,
        }
    }

    // Forward pass: x = [seq_len][d_model] → output [seq_len][d_v]
    // Same weights applied to every token; output blends the whole sequence.
    pub fn forward(&mut self, x: &[Vec<f32>]) -> Vec<Vec<f32>> {
        self.cache_x = x.to_vec();
        // ── STEP 1: PROJECT ─────────────────────────────────────────────────
        // Multiply input by each learned matrix to get query/key/value views.
        //   q[i] = "what is token i looking for?"
        //   k[i] = "what does token i offer?"
        //   v[i] = "what does token i actually carry?"
        let q = matmul(&x.to_vec(), &self.w_q.data);
        let mut k = matmul(&x.to_vec(), &self.w_k.data);
        let mut v = matmul(&x.to_vec(), &self.w_v.data);

        // ── KV-CACHE APPEND ────────────────────────────────────────────────
        // Non-cached attention receives the whole sequence every time:
        //   x [seq_len][d_model]
        //   q,k,v [seq_len][d_k or d_v]
        //
        // Cached generation is different after the prompt pass:
        //   x      [1][d_model]       only the newest token row
        //   q      [1][d_k]           only the newest query is needed
        //   k,v    [1][d_k or d_v]    newest key/value rows
        //   past_k [past_len][d_k]    saved from earlier forward calls
        //   past_v [past_len][d_v]
        //
        // We append the new k/v rows to the saved tables so the current query
        // can score against every previous token plus itself.
        //
        // Matrix picture when past_len = 4 and current_len = 1:
        //
        //   before append:
        //     past_k = [k0, k1, k2, k3]      [4][d_k]
        //     k      = [k4]                  [1][d_k]
        //
        //   after append:
        //     k      = [k0, k1, k2, k3, k4]  [5][d_k]
        //
        // Same for V:
        //     v      = [v0, v1, v2, v3, v4]  [5][d_v]
        //
        // Q is NOT appended here:
        //     q      = [q4]                  [1][d_k]
        //
        // Reason: old tokens do not need new outputs during generation. We
        // only need the newest token's output row, so only the newest query is
        // required. Old keys/values are enough to let it read history.
        let mut pos_offset = 0;
        if self.use_cache {
            if let Some((ref past_k, ref past_v)) = self.cache_kv {
                pos_offset = past_k.len();

                // full_k shape: [past_len + current_len][d_k]
                // current_len is usually 1 during token-by-token generation.
                let mut full_k = past_k.clone();
                full_k.extend(k);
                k = full_k;

                // full_v shape: [past_len + current_len][d_v]
                // Values must grow with keys because attention_weights @ V
                // needs one value row for every key column in the score grid.
                let mut full_v = past_v.clone();
                full_v.extend(v);
                v = full_v;
            }
            // Store the merged table for the next generation step. After this:
            //   cache_kv.0.len() == cache_kv.1.len() == total visible tokens.
            self.cache_kv = Some((k.clone(), v.clone()));
        }

        self.cache_q = q.clone();
        self.cache_k = k.clone();
        self.cache_v = v.clone();
        // ── STEP 2: SCORE ───────────────────────────────────────────────────
        // Attention = softmax(Q @ K^T / sqrt(d_k)) @ V
        // Transpose K so columns become keys, then Q @ K^T gives a score grid.
        //
        // Without cache:
        //   Q [seq_len][d_k] @ K^T [d_k][seq_len] -> scores [seq_len][seq_len]
        //
        // With cache during generation:
        //   Q [1][d_k] @ K^T [d_k][past_len + 1]
        //     -> scores [1][past_len + 1]
        //
        // scores[i][j] = how much current query row i matches visible key row j.
        let k_t = mat_transpose(&k);
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
        // i = query row, j = key column.
        //
        // With cache, i is local to the current call, but j is in the full
        // cached timeline. Example after 4 cached tokens and 1 new token:
        //   local i = 0, query_pos = 4
        //   keys visible at j = 0..4
        //   keys future at j > 4
        //
        // The offset keeps the causal mask aligned to absolute sequence
        // positions instead of mistakenly treating the new token as position 0.
        for i in 0..scaled_scores.len() {
            let query_pos = pos_offset + i;

            // A key is in the future if its sequence position j is ahead of
            // this query's absolute position.
            for j in (query_pos + 1)..scaled_scores[0].len() {
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

        self.cache_attention_weights = attention_weights.clone();

        // ── STEP 5: BLEND ───────────────────────────────────────────────────
        // weighted sum of values = attention_weights @ V
        // output[i] = sum_j attention_weights[i][j] * v[j]
        //
        // Cached generation shape:
        //   attention_weights [1][past_len + 1]
        //   V                 [past_len + 1][d_v]
        //   output            [1][d_v]
        //
        // This single output row is the new token's context-aware vector, built
        // from all cached previous values plus the current value.
        matmul(&attention_weights, &v)
    }

    // Backward pass: d_out = [seq_len][d_v] -> d_x = [seq_len][d_model]
    //
    // This walks the forward algorithm in reverse:
    //   forward  STEP 1 PROJECT -> STEP 2 SCORE -> STEP 3 SCALE
    //            -> CAUSAL MASK -> STEP 4 WEIGHT -> STEP 5 BLEND
    //   backward STEP 5 BLEND -> STEP 4 WEIGHT -> MASK/STEP 3 SCALE
    //            -> STEP 2 SCORE -> STEP 1 PROJECT
    //
    // Every cached value below came from the most recent forward() call. That
    // matters because backprop needs the exact Q/K/V and attention weights that
    // produced the current output, not freshly recomputed or random values.
    pub fn backward(&mut self, d_out: &[Vec<f32>]) -> Vec<Vec<f32>> {
        // ── BACKWARD: REVERSE FORWARD STEP 5 (BLEND) ───────────────────────
        // Forward STEP 5 did:
        //   output = attention_weights @ V
        // Meaning:
        //   each output token was built by mixing value vectors V using the
        //   attention probabilities from STEP 4.
        //
        // d_out is the gradient arriving from the layer above. Since output
        // depended on TWO inputs (attention_weights and V), the gradient splits:
        //   d_attention_weights = d_out @ V^T
        //   d_v                 = attention_weights^T @ d_out
        //
        // Shapes:
        //   d_out              [seq_len][d_v]
        //   V^T                [d_v][seq_len]
        //   d_attention_w      [seq_len][seq_len]
        //   d_v                [seq_len][d_v]
        let v_t = mat_transpose(&self.cache_v);
        let d_attention_w = matmul(&d_out.to_vec(), &v_t);
        let a_t = mat_transpose(&self.cache_attention_weights);
        let d_v = matmul(&a_t, &d_out.to_vec());

        let seq_len = d_out.len();
        let mut d_scaled = vec![vec![0.0; seq_len]; seq_len];

        let dk_sqrt = (self.d_k as f32).sqrt();

        // ── BACKWARD: REVERSE FORWARD STEP 4 (WEIGHT / SOFTMAX) ────────────
        // Forward STEP 4 did:
        //   attention_weights[i] = softmax(scaled_scores[i])
        // Meaning:
        //   raw score row i became a probability row over all visible keys.
        //   The probabilities in one row are linked because they all share the
        //   same softmax denominator.
        //
        // Softmax is row-wise: each query row has its own probability
        // distribution over keys. Changing one score in a row affects every
        // probability in that same row, so the derivative is not just element-
        // by-element multiplication.
        //
        // Optimized row formula:
        //   d_scaled[i][j] =
        //       P[i][j] * (dP[i][j] - sum_k(dP[i][k] * P[i][k]))
        //
        // where:
        //   P  = cached attention_weights
        //   dP = d_attention_w
        //
        // Same idea as the full Jacobian:
        //   ∂P_j/∂s_k = P_j * (δ_jk - P_k)
        for i in 0..seq_len {
            let mut sum_dp = 0.0;
            for k in 0..seq_len {
                sum_dp += d_attention_w[i][k] * self.cache_attention_weights[i][k];
            }
            for j in 0..seq_len {
                d_scaled[i][j] =
                    self.cache_attention_weights[i][j] * (d_attention_w[i][j] - sum_dp);
            }
        }

        // ── BACKWARD: REVERSE CAUSAL MASK + FORWARD STEP 3 (SCALE) ─────────
        // Forward masking did:
        //   future-token scores were replaced with -inf before softmax
        //
        // Forward STEP 3 did:
        //   scaled_scores = scores / sqrt(d_k)
        //   scaled_scores[i][j] = -inf for future tokens j > i
        // Meaning:
        //   allowed score magnitudes were reduced before softmax, while future
        //   positions were removed from the computation entirely.
        //
        // Future positions were blocked before softmax. Because they were not
        // allowed to influence output, their gradients must stay exactly 0.
        //
        // For allowed positions, reverse the scale operation:
        //   d_scores = d_scaled_scores / sqrt(d_k)
        for i in 0..seq_len {
            for j in 0..seq_len {
                if j > i {
                    d_scaled[i][j] = 0.0;
                } else {
                    d_scaled[i][j] /= dk_sqrt;
                }
            }
        }

        // ── BACKWARD: REVERSE FORWARD STEP 2 (SCORE) ───────────────────────
        // Forward STEP 2 did:
        //   scores = Q @ K^T
        // Meaning:
        //   each score[i][j] measured how strongly query token i matched key
        //   token j. Since the score used both Q and K, its gradient must flow
        //   back into both projection outputs.
        //
        // Matrix multiply backward:
        //   d_q = d_scores @ K
        //   d_k = d_scores^T @ Q
        //
        // Shapes:
        //   d_scores [seq_len][seq_len]
        //   K        [seq_len][d_k]
        //   Q        [seq_len][d_k]
        //   d_q/d_k  [seq_len][d_k]
        let d_q = matmul(&d_scaled, &self.cache_k);
        let d_scaled_t = mat_transpose(&d_scaled);
        let d_k_grad = matmul(&d_scaled_t, &self.cache_q);

        // ── BACKWARD: REVERSE FORWARD STEP 1 (PROJECT) ─────────────────────
        // Forward STEP 1 did three independent projections from the same input:
        //   Q = X @ W_Q
        //   K = X @ W_K
        //   V = X @ W_V
        // Meaning:
        //   X was copied into three branches, then each branch used a different
        //   learned matrix to create a different view of the same tokens.
        //
        // Weight gradients use the cached input:
        //   d_w_q = X^T @ d_q
        //   d_w_k = X^T @ d_k
        //   d_w_v = X^T @ d_v
        //
        // We add into d_w_* instead of assigning so multiple backward calls can
        // accumulate gradients before the optimizer step.
        let x_t = mat_transpose(&self.cache_x);
        let batch_d_wq = matmul(&x_t, &d_q);
        let batch_d_wk = matmul(&x_t, &d_k_grad);
        let batch_q_wv = matmul(&x_t, &d_v);

        for i in 0..self.d_model {
            for j in 0..self.d_k {
                self.w_q.grad[i][j] += batch_d_wq[i][j];
                self.w_k.grad[i][j] += batch_d_wk[i][j];
            }
            for j in 0..self.d_v {
                self.w_v.grad[i][j] += batch_q_wv[i][j];
            }
        }

        // ── RETURN GRADIENT TO THE PREVIOUS LAYER ──────────────────────────
        // Forward connection this reverses:
        //   the same X fed STEP 1's Q, K, and V projection branches.
        //
        // X fed all three projection branches. By the chain rule, when one
        // value branches into several paths, its total gradient is the sum of
        // the gradients returning from each path:
        //   d_x = d_x_from_q + d_x_from_k + d_x_from_v
        //
        // Each branch reverses its projection:
        //   d_x_from_q = d_q @ W_Q^T
        //   d_x_from_k = d_k @ W_K^T
        //   d_x_from_v = d_v @ W_V^T
        let w_q_t = mat_transpose(&self.w_q.data);
        let w_k_t = mat_transpose(&self.w_k.data);
        let w_v_t = mat_transpose(&self.w_v.data);

        let d_xq = matmul(&d_q, &w_q_t);
        let d_xk = matmul(&d_k_grad, &w_k_t);
        let d_xv = matmul(&d_v, &w_v_t);

        let mut d_x = vec![vec![0.0; self.d_model]; seq_len];

        for i in 0..seq_len {
            for j in 0..self.d_model {
                d_x[i][j] = d_xq[i][j] + d_xk[i][j] + d_xv[i][j];
            }
        }

        d_x
    }
    pub fn parameters(&mut self) -> Vec<&mut Param> {
        vec![&mut self.w_q, &mut self.w_k, &mut self.w_v]
    }
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
