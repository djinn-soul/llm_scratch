use llm_scratch_rs::common::util::random_matrix;
use llm_scratch_rs::layers::embedding::{embed_sequence, PositionalEmbedding, TokenEmbedding};
use llm_scratch_rs::layers::feed_forward::FeedForward;

fn main() {
    let d_model = 64;
    let d_ff = 128;

    let tok = TokenEmbedding::new(1000, d_model);
    let pos = PositionalEmbedding::new(500, d_model);
    let x = embed_sequence(&[1, 2, 3], &tok, &pos, 0);
    let mut ff = FeedForward::new(
        random_matrix(d_model, d_ff),
        random_matrix(d_ff, d_model),
        d_model,
        d_ff,
    );
    let out = ff.forward(&x);

    println!("In:  [{}][{}]", x.len(), x[0].len()); // [4][64]
    println!("Out: [{}][{}]", out.len(), out[0].len()); // [4][64] — shape preserved
}
