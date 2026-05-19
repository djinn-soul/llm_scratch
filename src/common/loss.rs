use crate::common::activation::softmax;

/// Calculates the Cross-Entropy Loss for a batch/sequence of predictions.
///
/// `logits`: The raw output from the GPT model. Shape: [seq_len][vocab_size]
/// `targets`: The actual next-token IDs we wanted the model to predict. Shape: [seq_len]
///
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
