use llm_scratch_rs::embedding::{embed_sequence, PositionalEmbedding, TokenEmbedding};
use llm_scratch_rs::multi_head_attention::MultiHeadAttention;

fn main() {
    let d_model = 64;
    let tok = TokenEmbedding::new(1000, d_model);
    let pos = PositionalEmbedding::new(512, d_model);
    let x = embed_sequence(&[45, 12, 300, 7], &tok, &pos);

    let mha = MultiHeadAttention::new(d_model, 8);
    let out = mha.forward(&x);

    println!("In:  [{}][{}]", x.len(), x[0].len());
    println!("Out: [{}][{}]", out.len(), out[0].len()); // [4][64]
}
