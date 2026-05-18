// Smoke test: instantiate one TransformerBlock and run a forward pass.
// Checks that shapes are preserved: [seq_len][d_model] → [seq_len][d_model].
use llm_scratch_rs::transformer::Transformer;

fn main() {
    let d_model = 64;
    let num_heads = 8;
    let d_ff = 256; // 4 × d_model (standard GPT ratio)
    let seq_len = 5;

    // Build one transformer block with random weights
    let block = Transformer::new(d_model, num_heads, d_ff);

    // Fake input: seq_len tokens, each a d_model-wide zero vector
    // let x: Vec<Vec<f32>> = vec![vec![1_f32; d_model]; seq_len];
    let x = llm_scratch_rs::util::random_matrix(seq_len, d_model);

    let output = block.forward(&x);

    // Verify shape is preserved
    assert_eq!(output.len(), seq_len, "seq_len must be preserved");
    assert_eq!(output[0].len(), d_model, "d_model must be preserved");

    println!("✅ TransformerBlock forward pass OK");
    println!("   input  shape: [{seq_len}][{d_model}]");
    println!("   output shape: [{}][{}]", output.len(), output[0].len());
    println!("   output: {:?}", output);
}
