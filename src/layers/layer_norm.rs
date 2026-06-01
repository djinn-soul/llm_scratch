// ════════════════════════════════════════════════════════════════════════════
// LAYER NORMALIZATION
// ════════════════════════════════════════════════════════════════════════════
// After attention or FFN, token values can be wildly scaled (some near 500,
// some near 0.001). LayerNorm re-centers and re-scales each token row so the
// next layer always sees well-behaved inputs — mean ≈ 0, std ≈ 1.
//
// Algorithm (applied independently to each token row):
//   1. MEAN     — average of all d_model values in the row
//   2. VARIANCE — average squared distance from the mean
//   3. STDDEV   — sqrt(variance + eps), eps avoids division by zero
//   4. NORMALIZE— shift by mean, scale by stddev → mean=0, std=1
//   5. RESCALE  — multiply by learned γ (gamma), add learned β (beta)
//
// Why per-row (not per-batch)?
//   BatchNorm normalizes across all tokens/samples — it needs a full batch and
//   behaves differently at inference with batch=1. LayerNorm normalizes each
//   token independently, so it works identically at train and inference time.
//   That's why every modern transformer uses LayerNorm.
//
// γ and β are learned during training. Both start at identity (γ=1, β=0)
// so the initial output is pure normalization. The model adjusts them to
// stretch/shift the normalized values as needed for the task.
//
// https://docs.pytorch.org/docs/2.12/generated/torch.nn.LayerNorm.html
// https://stats.stackexchange.com/questions/474440/why-do-transformers-use-layer-norm-instead-of-batch-norm
// ════════════════════════════════════════════════════════════════════════════

// gamma: per-feature scale  [d_model] — learned, init 1.0
// beta:  per-feature shift  [d_model] — learned, init 0.0
// eps:   stability constant            — fixed ~1e-5, never trained
use crate::common::param::Param;

pub struct LayerNorm {
    pub gamma: Param,           // Wrapped row parameter: [1][d_model] — scale, init 1.0
    pub beta: Param,            // Wrapped row parameter: [1][d_model] — shift, init 0.0
    pub eps: f32,               // ~1e-5, prevents div-by-zero
    cache_x_hat: Vec<Vec<f32>>, // normalized values per row, before gamma/beta
    cache_std: Vec<f32>,        // std_dev per row
    pub d_gamma: Vec<f32>,
    pub d_beta: Vec<f32>,
}

impl LayerNorm {
    // Build LayerNorm for a given model width. γ=1, β=0 → identity at start.
    pub fn new(d_model: usize) -> Self {
        Self {
            gamma: Param::new(vec![vec![1.0; d_model]]),
            beta: Param::new(vec![vec![0.0; d_model]]),
            eps: 1e-5,
            cache_x_hat: Vec::new(),
            cache_std: Vec::new(),
            d_gamma: vec![0.0; d_model],
            d_beta: vec![0.0; d_model],
        }
    }

    // Normalize a single token row. Private — callers always use forward().
    // formula: y[i] = gamma[i] * (x[i] - mean) / sqrt(variance + eps) + beta[i]
    fn norm_row(&mut self, row: &[f32]) -> Vec<f32> {
        let n = row.len() as f32;

        // ── STEP 1: MEAN ────────────────────────────────────────────────────
        // Average value across all d_model features for this one token.
        let mean = row.iter().sum::<f32>() / n;

        // ── STEP 2: VARIANCE ────────────────────────────────────────────────
        // Average squared deviation from the mean. Measures how spread out
        // the values are. High variance → wide spread; low → tight cluster.
        let variance = row.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;

        // ── STEP 3: STDDEV ──────────────────────────────────────────────────
        // Square root of variance = standard deviation. eps added before sqrt
        // to prevent a crash if all values in the row are identical (var = 0).
        let std_dev = (variance + self.eps).sqrt();

        // ── STEP 4: NORMALIZE + RESCALE ─────────────────────────────────────
        // (x - mean) / std_dev  →  zero-mean, unit-variance
        let x_hat: Vec<f32> = row.iter().map(|v| (v - mean) / std_dev).collect();
        // x_hat * gamma + beta    →  learned scale and shift per feature
        // All three iterators walk in lock-step: x[i], gamma[i], beta[i].
        let y: Vec<f32> = x_hat
            .iter()
            .zip(self.gamma.data[0].iter())
            .zip(self.beta.data[0].iter())
            .map(|((xh, g), b)| g * xh + b)
            .collect();

        self.cache_x_hat.push(x_hat);
        self.cache_std.push(std_dev);

        y
    }

    // Forward pass: x = [seq_len][d_model] → output [seq_len][d_model]
    // Each token row normalized independently. Shape is always preserved.
    pub fn forward(&mut self, x: &[Vec<f32>]) -> Vec<Vec<f32>> {
        self.cache_x_hat.clear();
        self.cache_std.clear();
        x.iter().map(|row| self.norm_row(row)).collect()
    }

    //
    // Forward:
    // mean     = sum(x) / n
    // variance = sum((x - mean)^2) / n
    // x_hat    = (x - mean) / std
    // output   = gamma * x_hat + beta
    //
    // Backward goal:
    // Given d_out = dL/d_output
    // compute:
    // 1. d_gamma
    // 2. d_beta
    // 3. d_x (gradient flowing to previous layer)
    // Final dL/dx formula, per token row:
    // dL/dx = (d_xhat - mean(d_xhat) - x_hat * mean(d_xhat * x_hat)) / std

    // Backward pass: d_out = [seq_len][d_model] -> d_x = [seq_len][d_model]
    //
    // This walks the forward algorithm in reverse:
    //   forward  STEP 1 MEAN -> STEP 2 VARIANCE -> STEP 3 STDDEV
    //            -> STEP 4 NORMALIZE -> STEP 5 RESCALE
    //   backward STEP 5 RESCALE -> STEP 4 NORMALIZE
    //            -> STEP 3/2/1 row-stat effects
    //
    // LayerNorm has a compact closed-form backward formula. Instead of
    // separately storing mean and variance, it uses cached x_hat and std_dev
    // from forward() to account for all row-level dependencies.
    //
    // Final per-token formula:
    //   d_x = (d_xhat - mean(d_xhat) - x_hat * mean(d_xhat * x_hat)) / std
    pub fn backward(&mut self, d_out: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        let n = self.gamma.data[0].len() as f32;
        let mut d_x: Vec<Vec<f32>> = Vec::with_capacity(d_out.len());

        // These gradients belong to this backward call. Reset before summing
        // contributions from every token row in the sequence.
        for j in 0..self.gamma.data[0].len() {
            self.d_gamma[j] = 0.0;
            self.d_beta[j] = 0.0;
        }

        for i in 0..d_out.len() {
            let x_hat = &self.cache_x_hat[i];
            let std_dev = self.cache_std[i];

            // ── BACKWARD: REVERSE FORWARD STEP 5 (RESCALE) ────────────────
            // Forward STEP 5 did:
            //   output = gamma * x_hat + beta
            // Meaning:
            //   normalized values were stretched by learned gamma and shifted
            //   by learned beta, independently for each feature.
            //
            // Since output depended on gamma, beta, and x_hat:
            //   d_gamma += d_out * x_hat
            //   d_beta  += d_out
            //   d_xhat   = d_out * gamma
            for j in 0..self.gamma.data[0].len() {
                // dL/d_gamma
                self.d_gamma[j] += d_out[i][j] * x_hat[j];
                // dL/d_beta
                self.d_beta[j] += d_out[i][j];
            }
            let mut d_xhat = vec![0.0; self.gamma.data[0].len()];
            for j in 0..self.gamma.data[0].len() {
                d_xhat[j] = d_out[i][j] * self.gamma.data[0][j];
            }

            // ── BACKWARD: REVERSE FORWARD STEP 4 (NORMALIZE) ──────────────
            // Forward STEP 4 did:
            //   x_hat = (x - mean) / std
            // Meaning:
            //   every feature in this token row was centered by the same mean
            //   and scaled by the same std_dev. That couples all features in
            //   the row, so backward needs row-level sums, not independent
            //   element-wise gradients.
            //
            // Helper sums:
            //   sum_dxhat       = sum_j d_xhat[j]
            //   sum_dxhat_xhat  = sum_j d_xhat[j] * x_hat[j]
            //
            // These compactly represent the mean and variance correction terms.
            let sum_dxhat: f32 = d_xhat.iter().sum();
            let sum_dxhat_xhat: f32 = d_xhat
                .iter()
                .zip(x_hat.iter())
                .map(|(dxh, xh)| dxh * xh)
                .sum();

            // ── BACKWARD: REVERSE FORWARD STEPS 3/2/1 (STDDEV/STATS) ──────
            // Forward STEP 1 computed the row mean.
            // Forward STEP 2 computed row variance from that mean.
            // Forward STEP 3 computed std_dev = sqrt(variance + eps).
            //
            // Meaning:
            //   changing one input feature changes the row mean and variance,
            //   which then changes every normalized feature in that row.
            //
            // inv_std is the reverse scale for the forward division by std_dev.
            let inv_std = 1.0 / (std_dev);

            // Final LayerNorm backward formula:
            //   dx = inv_std * (
            //       d_xhat
            //       - mean(d_xhat)
            //       - x_hat * mean(d_xhat * x_hat)
            //   )
            //
            // Terms:
            //   d_xhat                         direct gradient through normalize
            //   - mean(d_xhat)                 correction for forward mean
            //   - x_hat * mean(d_xhat*x_hat)   correction for forward variance
            let mut dx_inside = vec![0.0; self.gamma.data[0].len()];
            for j in 0..self.gamma.data[0].len() {
                dx_inside[j] =
                    inv_std * (d_xhat[j] - (sum_dxhat / n) - (x_hat[j] * sum_dxhat_xhat / n));
            }
            d_x.push(dx_inside);
        }
        d_x
    }
    pub fn parameters(&mut self) -> Vec<&mut Param> {
        vec![&mut self.gamma, &mut self.beta]
    }
}

// ════════════════════════════════════════════════════════════════════════════
// HOW LAYER NORM WORKS — FULL WALKTHROUGH WITH EXAMPLE
// ════════════════════════════════════════════════════════════════════════════
//
// INPUT: 1 token, d_model = 4  (wildly different values to show the effect)
//   row = [2.0, 100.0, -50.0, 3.0]
//
// ── STEP 1: MEAN ─────────────────────────────────────────────────────────────
//   mean = (2 + 100 + (-50) + 3) / 4 = 55 / 4 = 13.75
//
// ── STEP 2: VARIANCE ─────────────────────────────────────────────────────────
//   (2 - 13.75)²   = (-11.75)² = 138.06
//   (100 - 13.75)² = (86.25)²  = 7439.06
//   (-50 - 13.75)² = (-63.75)² = 4064.06
//   (3 - 13.75)²   = (-10.75)² = 115.56
//   variance = (138.06 + 7439.06 + 4064.06 + 115.56) / 4 = 2939.19
//
// ── STEP 3: STDDEV ───────────────────────────────────────────────────────────
//   std_dev = sqrt(2939.19 + 1e-5) ≈ 54.22
//
// ── STEP 4: NORMALIZE + RESCALE (γ=1, β=0 initially) ────────────────────────
//   y_0 = 1 × (2   - 13.75) / 54.22 + 0 = -0.217
//   y_1 = 1 × (100 - 13.75) / 54.22 + 0 =  1.591
//   y_2 = 1 × (-50 - 13.75) / 54.22 + 0 = -1.176
//   y_3 = 1 × (3   - 13.75) / 54.22 + 0 = -0.198
//
//   output = [-0.217, 1.591, -1.176, -0.198]
//
// Verify: mean ≈ 0, std ≈ 1 ✓  (same token — just rescaled, not mixed)
//
// ── WHERE LAYERNORM SITS IN THE TRANSFORMER BLOCK ────────────────────────────
//
//   x ──┬──► MHA(x) ──► + ──► LayerNorm ──┬──► FFN ──► + ──► LayerNorm ──► out
//       │              ▲                   │            ▲
//       └──────────────┘ residual          └────────────┘ residual
//
// Two LayerNorm instances per block: one after attention, one after FFN.
// The residual (skip connection) adds the original x back before normalizing,
// which keeps gradient flow healthy through many stacked layers.
//
// ── WHY EACH STEP EXISTS ─────────────────────────────────────────────────────
//   MEAN       re-centers the distribution to 0
//   VARIANCE   measures the current spread of values
//   STDDEV     converts variance to the same unit as x (square root)
//   NORMALIZE  forces mean=0, std=1 → stable input to next layer
//   RESCALE    γ/β give the model freedom to undo normalization if needed
//
// ── DATA STRUCTURES ──────────────────────────────────────────────────────────
//   gamma      Vec<f32>        [d_model]           learned scale per feature
//   beta       Vec<f32>        [d_model]           learned shift per feature
//   eps        f32             scalar              div-by-zero guard
//   x          &[Vec<f32>]     [seq_len][d_model]  input (borrowed slice)
//   mean       f32             scalar per row      average of row values
//   variance   f32             scalar per row      avg squared deviation
//   std_dev    f32             scalar per row      sqrt(variance + eps)
//   output     Vec<Vec<f32>>   [seq_len][d_model]  normalized, same shape
// ════════════════════════════════════════════════════════════════════════════
