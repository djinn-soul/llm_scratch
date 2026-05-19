// Softmax: turn a vector of raw scores into probabilities that sum to 1.
//   softmax(x_i) = e^(x_i) / Σ_j e^(x_j)
// Denominator (sum) is the same for every element, so compute it once.
// Note: no max-subtraction here — fine for small values, can overflow on big.
pub fn softmax(x: &Vec<f32>) -> Vec<f32> {
    let mut result: Vec<f32> = Vec::new();
    let sum: f32 = x.iter().map(|v| v.exp()).sum();
    for i in 0..x.len() {
        result.push(x[i].exp() / sum);
    }
    result
}
