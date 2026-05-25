use llm_scratch_rs::attention::self_attention::SelfAttention;
use llm_scratch_rs::layers::embedding::{embed_sequence, PositionalEmbedding, TokenEmbedding};

fn main() {
    let d_model = 64;
    let tok = TokenEmbedding::new(1000, d_model);
    let pos = PositionalEmbedding::new(512, d_model);
    let x = embed_sequence(&[45, 12, 300, 7], &tok, &pos);

    let mut attn = SelfAttention::new(d_model, d_model, d_model);
    let out = attn.forward(&x);

    println!("Out shape: [{}][{}]", out.len(), out[0].len());
}
