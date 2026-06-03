use crate::common::activation::softmax;

/// Sample the next token ID from logits using temperature scaling and top-k/top-p filtering.
///
/// # Visual Step-by-Step Matrix Walkthrough (Vocabulary Size = 5: [A, B, C, D, E])
///
/// ```text
/// ── STEP 0: RAW LOGITS INPUT (from GPT model head) ──────────────────────────────────
/// Let's say: logits = [2.0,  1.0,  5.0,  0.5,  3.0]
///
/// ── STEP 1: TEMPERATURE SCALING (with T = 0.5) ──────────────────────────────────────
/// Each raw logit is divided by T (dividing by 0.5 multiplies all values by 2):
///   scaled_logits = logits / 0.5
///                 = [4.0,  2.0,  10.0,  1.0,  6.0]
///
/// ── STEP 2: TOP-K FILTERING (with K = 3) ────────────────────────────────────────────
/// 1. We clone and sort scores descending: [10.0 (C), 6.0 (E), 4.0 (A), 2.0 (B), 1.0 (D)]
/// 2. Threshold is the K-th (3rd) value: threshold = 4.0 (A)
/// 3. Any score strictly less than 4.0 is eliminated by set to -infinity (-inf):
///   scaled_logits = [4.0, -inf,  10.0, -inf,  6.0]   <-- (B and D are masked out)
///
/// ── STEP 3: TOP-P (NUCLEUS) FILTERING (with P = 0.99) ──────────────────────────────
/// 1. Run a temporary Softmax on current scaled_logits:
///      exps  = [e^4,  e^-inf, e^10,    e^-inf, e^6]  = [54.6, 0.0, 22026.5, 0.0, 403.4]
///      probs = exps / sum(exps)                     = [0.0024, 0.0, 0.9796, 0.0, 0.0179]
///                                                    (A: 0.24%, C: 97.96%, E: 1.79%)
///
/// 2. Sort probabilities descending with original indices:
///      [(Index 2 (C), 97.96%), (Index 4 (E), 1.79%), (Index 0 (A), 0.24%), ...]
///
/// 3. Accumulate sum up to P = 0.99:
///      - Token C: cumulative = 97.96% (Keep C, cumulative < 99.0%, continue)
///      - Token E: cumulative = 97.96% + 1.79% = 99.75% (Keep E, cumulative >= 99.0%, break!)
///      - Token A: Excluded!
///
/// 4. Mask out any token not in the kept nucleus:
///   scaled_logits = [-inf, -inf,  10.0, -inf,  6.0]  <-- (Token A is now masked out)
///
/// ── STEP 4: FINAL SOFTMAX ──────────────────────────────────────────────────────────
/// Run Softmax on the final scaled_logits:
///   final_probs = [0.0,   0.0,    0.982,  0.0,   0.018]  (C is 98.2% likely, E is 1.8% likely)
///
/// ── STEP 5: ROULETTE-WHEEL WEIGHTED INDEX SELECTION ─────────────────────────────────
/// 1. Draw a random decimal R in [0.0, 1.0). Let's say: R = 0.99
/// 2. Scan the tokens and add probabilities:
///      - Index 0: cumulative = 0.0.                 (0.99 > 0.0, keep going)
///      - Index 1: cumulative = 0.0.                 (0.99 > 0.0, keep going)
///      - Index 2 (C): cumulative = 0.0 + 0.982 = 0.982. (0.99 > 0.982, keep going)
///      - Index 3: cumulative = 0.982.               (0.99 > 0.982, keep going)
///      - Index 4 (E): cumulative = 0.982 + 0.018 = 1.0. (0.99 < 1.00, TRIGGER!)
/// 3. Return Selected Index: 4 (Token E).
/// ```
pub fn sample_next_token(
    logits: &[f32],       // raw output from the Transformer head (no softmax yet)
    temperature: f32,     // controls randomness (0 = deterministic, >0 = stochastic)
    top_k: Option<usize>, // keep only the top K most likely tokens
    top_p: Option<f32>,   // nucleus sampling: keep tokens whose cumulative probability mass ≥ p
) -> usize {
    // 1. Handle Greedy decoding directly if temperature is near 0
    if temperature <= 1e-6 {
        return logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap_or(0);
    }

    // 2. Apply Temperature scaling
    let mut scaled_logits: Vec<f32> = logits.iter().map(|&x| x / temperature).collect();

    // 3. Apply Top-K Filtering
    if let Some(k) = top_k {
        let k = std::cmp::max(1, k);
        if k < scaled_logits.len() {
            // Find the k-th largest value by sorting a copy descending
            let mut sorted = scaled_logits.clone();
            sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            let threshold = sorted[k - 1];

            // Mask out anything smaller than the k-th largest value
            for val in &mut scaled_logits {
                if *val < threshold {
                    *val = f32::NEG_INFINITY;
                }
            }
        }
    }

    // 4. Apply Top-P (Nucleus) Filtering
    if let Some(p) = top_p {
        if p > 0.0 && p < 1.0 {
            // Softmax current logits (Top-K mask already applied if active)
            let probs = softmax(&scaled_logits);

            // Pair probabilities with original indices and sort descending
            let mut indexed_probs: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
            indexed_probs
                .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            // Accumulate probabilities up to P
            let mut cumulative_sum = 0.0;
            let mut cut_off_idx = indexed_probs.len();
            for (i, &(_, prob)) in indexed_probs.iter().enumerate() {
                cumulative_sum += prob;
                if cumulative_sum >= p {
                    cut_off_idx = i + 1; // Include the threshold token
                    break;
                }
            }

            // Ensure we keep at least the top 1 token to prevent completely empty selections
            cut_off_idx = std::cmp::max(1, cut_off_idx);

            // Create a lookup of indices to keep
            let mut keep = vec![false; scaled_logits.len()];
            for &(idx, _) in &indexed_probs[0..cut_off_idx] {
                keep[idx] = true;
            }

            // Mask out any index not in the top-P cumulative set
            for (idx, &is_kept) in keep.iter().enumerate() {
                if !is_kept {
                    scaled_logits[idx] = f32::NEG_INFINITY;
                }
            }
        }
    }

    // 5. Softmax to get final probability distribution over remaining tokens
    let final_probs = softmax(&scaled_logits);

    // 6. Roulette-Wheel Weighted Index Sampling
    let r: f32 = rand::random();
    let mut cumulative = 0.0;
    for (idx, &prob) in final_probs.iter().enumerate() {
        cumulative += prob;
        if r < cumulative {
            return idx;
        }
    }

    // Fallback: return highest-probability token if rounding errors prevent trigger
    final_probs
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx)
        .unwrap_or(0)
}
