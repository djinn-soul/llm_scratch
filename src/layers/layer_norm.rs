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
pub struct LayerNorm {
    pub gamma: Vec<f32>, // [d_model] — scale, init 1.0
    pub beta: Vec<f32>,  // [d_model] — shift, init 0.0
    pub eps: f32,        // ~1e-5, prevents div-by-zero
}

impl LayerNorm {
    // Build LayerNorm for a given model width. γ=1, β=0 → identity at start.
    pub fn new(d_model: usize) -> Self {
        Self {
            gamma: vec![1.0; d_model],
            beta: vec![0.0; d_model],
            eps: 1e-5,
        }
    }

    // Normalize a single token row. Private — callers always use forward().
    // formula: y[i] = gamma[i] * (x[i] - mean) / sqrt(variance + eps) + beta[i]
    fn norm_row(&self, row: &[f32]) -> Vec<f32> {
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
        // × gamma + beta        →  learned scale and shift per feature
        // All three iterators walk in lock-step: x[i], gamma[i], beta[i].
        row.iter()
            .zip(self.gamma.iter())
            .zip(self.beta.iter())
            .map(|((v, g), b)| g * (v - mean) / std_dev + b)
            .collect()
    }

    // Forward pass: x = [seq_len][d_model] → output [seq_len][d_model]
    // Each token row normalized independently. Shape is always preserved.
    pub fn forward(&self, x: &[Vec<f32>]) -> Vec<Vec<f32>> {
        x.iter().map(|row| self.norm_row(row)).collect()
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
