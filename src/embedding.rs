// https://dubeyrahul.github.io/posts/llm-from-scratch/token-embeddings.html
use rand::RngExt;
pub struct TokenEmbedding {
    weight: Vec<Vec<f32>>, //[vocab_size][d_model]
    pub vocab_size: usize,
    pub d_model: usize, // dimension of token embedding
}

pub struct PositionalEmbedding {
    weight: Vec<Vec<f32>>, // [max_seq_len][d_model]
    pub max_seq_len: usize,
    pub d_model: usize, // dimension of positional embedding
}

impl TokenEmbedding {
    pub fn new(vocab_size: usize, d_model: usize) -> Self {
        let mut rng = rand::rng();
        let weight = (0..vocab_size)
            .map(|_| (0..d_model).map(|_| rng.random_range(-1.0..1.0)).collect())
            .collect();
        Self {
            weight,
            vocab_size,
            d_model,
        }
    }
    pub fn forward(&self, ids: usize) -> Vec<f32> {
        self.weight[ids].clone()
    }
}

impl PositionalEmbedding {
    pub fn new(max_seq_len: usize, d_model: usize) -> Self {
        let mut rng = rand::rng();
        let weight = (0..max_seq_len)
            .map(|_| (0..d_model).map(|_| rng.random_range(-1.0..1.0)).collect())
            .collect();
        Self {
            weight,
            max_seq_len,
            d_model,
        }
    }
    pub fn forward(&self, ids: usize) -> Vec<f32> {
        self.weight[ids].clone()
    }
}

pub fn add(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

pub fn embed_sequence(
    ids: &[usize],
    token_embedding: &TokenEmbedding,
    positional_embedding: &PositionalEmbedding,
) -> Vec<Vec<f32>> {
    ids.iter()
        .enumerate()
        .map(|(position, &id)| {
            add(
                &token_embedding.forward(id),
                &positional_embedding.forward(position),
            )
        })
        .collect()
}
