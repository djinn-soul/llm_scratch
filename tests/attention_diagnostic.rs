// Diagnostic: is the bottleneck self-attention preserving spatial structure,
// or collapsing toward a global average?
//
// Attention output is O[:, i] = sum_j A[i, j] * V[:, j]. If the softmax rows A
// are near-uniform, every output position becomes the same global mean of V and
// all spatial detail at the bottleneck is destroyed. Because the UNet applies
// attention WITHOUT a residual (a3 = attn(a3_pre)), that collapse is not merely
// a no-op — it erases a3_pre from the decoder path entirely.
//
// Reported per run:
//   - mean softmax entropy vs the uniform maximum log(n)
//   - spatial variance of the attention input vs its output
//
// Entropy at ~100% of max, or an output/input spatial-variance ratio near zero,
// both indicate collapse.

use anyhow::Result;
use candle_core::{Device, Tensor};
use llm_scratch_rs::models::diffusion::{DenoisingModel, SimpleDenoisingUNet};

const IMG_DIM: usize = 784;
const COND_DIM: usize = 26;

/// Mean per-row Shannon entropy of the attention matrix, and the uniform bound.
fn softmax_entropy(attn_weights: &Tensor) -> Result<(f64, f64)> {
    let (b, n, _) = attn_weights.dims3()?;
    let values = attn_weights.flatten_all()?.to_vec1::<f32>()?;

    let mut total = 0.0f64;
    for row in 0..(b * n) {
        let offset = row * n;
        let mut entropy = 0.0f64;
        for j in 0..n {
            let p = values[offset + j] as f64;
            if p > 1e-12 {
                entropy -= p * p.ln();
            }
        }
        total += entropy;
    }

    Ok((total / (b * n) as f64, (n as f64).ln()))
}

/// Mean variance across spatial positions, averaged over batch and channel.
///
/// A tensor that is constant in space (the signature of an averaging collapse)
/// scores ~0 here regardless of its absolute magnitude.
fn mean_spatial_variance(seq: &Tensor) -> Result<f64> {
    let (b, c, n) = seq.dims3()?;
    let values = seq.flatten_all()?.to_vec1::<f32>()?;

    let mut total = 0.0f64;
    for row in 0..(b * c) {
        let offset = row * n;
        let slice = &values[offset..offset + n];
        let mean = slice.iter().map(|v| *v as f64).sum::<f64>() / n as f64;
        let var = slice
            .iter()
            .map(|v| {
                let d = *v as f64 - mean;
                d * d
            })
            .sum::<f64>()
            / n as f64;
        total += var;
    }

    Ok(total / (b * c) as f64)
}

#[test]
fn bottleneck_attention_preserves_spatial_structure() -> Result<()> {
    let device = Device::Cpu;
    let model = SimpleDenoisingUNet::new(IMG_DIM, COND_DIM, &device)?;

    // Batch of 2 keeps this cheap; the statistics below are per-row averages so
    // they do not depend on batch size.
    let input = Tensor::randn(0.0f32, 1.0f32, (2, IMG_DIM + COND_DIM), &device)?;
    let (_pred, intermediates) = model.forward(&input)?;

    // Layout from SimpleDenoisingUNet::forward: 20 UNet intermediates, then the
    // attention cache [x_seq, q, k, v, scores, attn_weights].
    let a3 = &intermediates[13];
    let x_seq = &intermediates[20];
    let scores = &intermediates[24];
    let attn_weights = &intermediates[25];

    let (b, c, h, w) = a3.dims4()?;
    let a3_seq = a3.reshape((b, c, h * w))?;

    let (entropy, max_entropy) = softmax_entropy(attn_weights)?;
    let input_var = mean_spatial_variance(x_seq)?;
    let output_var = mean_spatial_variance(&a3_seq)?;

    let score_values = scores.flatten_all()?.to_vec1::<f32>()?;
    let score_mean = score_values.iter().map(|v| *v as f64).sum::<f64>() / score_values.len() as f64;
    let score_std = (score_values
        .iter()
        .map(|v| {
            let d = *v as f64 - score_mean;
            d * d
        })
        .sum::<f64>()
        / score_values.len() as f64)
        .sqrt();

    println!("--- bottleneck attention diagnostic (untrained init) ---");
    println!("sequence length n            : {}", h * w);
    println!("score std                    : {score_std:.4}");
    println!(
        "softmax entropy              : {entropy:.4} of max {max_entropy:.4}  ({:.1}% of uniform)",
        100.0 * entropy / max_entropy
    );
    println!("spatial variance  in (x_seq) : {input_var:.6}");
    println!("spatial variance out (a3)    : {output_var:.6}");
    let retained = output_var / input_var;
    println!("output / input variance      : {retained:.4}");

    // With the residual in place this sits at ~0.50: attention itself adds
    // almost no spatial variance at init, so a3 ≈ a3_pre * RESIDUAL_SCALE and
    // the ratio is RESIDUAL_SCALE^2. Without the residual it measured 0.006 —
    // the bottleneck's spatial content deleted before it reaches the decoder.
    //
    // The bound is deliberately far below 0.5 and far above 0.006, so it tracks
    // the structural question (does a skip path exist?) rather than the exact
    // scaling constant.
    assert!(
        retained > 0.3,
        "bottleneck attention is destroying spatial structure: retained {retained:.4} of input \
         variance (expected ~0.5 with the residual). A near-zero value means attention collapsed \
         to a global average and no skip path carries a3_pre to the decoder."
    );

    Ok(())
}
