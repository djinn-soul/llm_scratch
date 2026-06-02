// Softmax: turn a vector of raw scores into probabilities that sum to 1.
//   softmax(x_i) = e^(x_i - max(x)) / Σ_j e^(x_j - max(x))
// Denominator (sum) is the same for every element, so compute it once.
// Note: Uses max-subtraction to prevent numerical overflow on large values.
pub fn softmax(x: &[f32]) -> Vec<f32> {
    if x.is_empty() {
        return Vec::new();
    }

    // Find the maximum value in the slice to perform the shift
    let max_val = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    // Compute the exponentials shifted by the maximum value
    let exps: Vec<f32> = x
        .iter()
        .map(|&val| {
            if val == f32::NEG_INFINITY {
                0.0
            } else {
                (val - max_val).exp()
            }
        })
        .collect();

    // Sum the exponentials
    let sum: f32 = exps.iter().sum();
    if sum == 0.0 {
        // Fallback: return uniform distribution if all exps are 0
        let len = x.len() as f32;
        return vec![1.0 / len; x.len()];
    }

    // Divide each exp by the sum
    exps.iter().map(|&val| val / sum).collect()
}
