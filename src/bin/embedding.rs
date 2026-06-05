use llm_scratch_rs::layers::embedding::{embed_sequence, PositionalEmbedding, TokenEmbedding};

fn main() {
    let vocab_size = 1000;
    let d_model = 64;
    let max_seq_len = 512;

    let tok = TokenEmbedding::new(vocab_size, d_model);
    let pos = PositionalEmbedding::new(max_seq_len, d_model);

    let ids = vec![45usize, 12, 300, 7];
    let embedded = embed_sequence(&ids, &tok, &pos, 0);

    println!("Sequence length: {}", embedded.len()); // 4
    println!("Embedding dim:   {}", embedded[0].len()); // 64
    println!("First vector:    {:?}", &embedded[0][..4]); // first 4 values
}
