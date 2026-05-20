use crate::common::activation::softmax;

/// Calculates the Cross-Entropy Loss for a batch/sequence of predictions.
///
/// `logits`: The raw output from the GPT model. Shape: [seq_len][vocab_size]
/// `targets`: The actual next-token IDs we wanted the model to predict. Shape: [seq_len]
///
/// Loss = -∑ y(i) * log(p(i))  where y(i) is 1 if the target token is i else 0 and p(i) is the probability of token i. For all i.
/// Returns a single f32 scalar representing the average loss across the sequence.
pub fn cross_entropy_loss(logits: &Vec<Vec<f32>>, targets: &Vec<usize>) -> f32 {
    assert_eq!(
        logits.len(),
        targets.len(),
        "Must have one target per logit row!"
    );

    let seq_len = logits.len();
    let mut total_loss = 0.0;

    for i in 0..seq_len {
        // convert raw logits into probabilities (The Activation Step)
        let probs = softmax(&logits[i]);
        // What probability did the model assign to the CORRECT word?
        let target_id = targets[i];
        let correct_prob = probs[target_id];
        // Negative log likelyhood
        let loss = -(correct_prob + 1e-8).ln(); // 1e-8 is just to prevent log(0) errors

        total_loss += loss;
    }

    total_loss / seq_len as f32
}

/// Calculates the backward gradient for
/// Softmax + Cross Entropy loss.
///
/// Formula:
/// dL/dlogits = probs - target
///
/// where:
/// - probs  = softmax(logits)
/// - target = one-hot encoded correct token
///
/// Shapes:
/// logits  -> [seq_len][vocab_size]
/// targets -> [seq_len]
///
/// Returns:
/// d_logits -> gradient for each logit
/// Shape: [seq_len][vocab_size]

pub fn cross_entropy_backward(logits: &Vec<Vec<f32>>, targets: &Vec<usize>) -> Vec<Vec<f32>> {
    let seq_len = logits.len();
    let mut d_logits: Vec<Vec<f32>> = Vec::with_capacity(seq_len);
    for i in 0..seq_len {
        let probs = softmax(&logits[i]);
        d_logits.push(probs);

        // Subtract 1.0 from correct token position
        //
        // This efficiently performs:
        // probs - one_hot_target
        //
        // Example:
        // probs  = [0.66, 0.24, 0.10]
        // target = [0,    1,    0]
        //
        // result = [0.66, -0.76, 0.10]
        let target_id = targets[i];
        d_logits[i][target_id] -= 1.0;

        // Normalize gradients by sequence length
        for j in 0..d_logits[i].len() {
            d_logits[i][j] = d_logits[i][j] / seq_len as f32;
        }
    }
    d_logits
}
