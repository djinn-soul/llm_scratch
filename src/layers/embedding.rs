// https://dubeyrahul.github.io/posts/llm-from-scratch/token-embeddings.html
// https://www.yadavsaurabh.com/building-a-transformer-llm-with-code-evolution-of-positional-encoding/
use crate::common::util::add;
use rand::RngExt;

pub struct TokenEmbedding {
    weight: Vec<Vec<f32>>, //[vocab_size][d_model]
    // TODO(backward): add weight gradients so repeated token IDs accumulate
    // d_token_embedding during GPT training.
    pub vocab_size: usize,
    pub d_model: usize, // dimension of token embedding
}

pub struct PositionalEmbedding {
    weight: Vec<Vec<f32>>, // [max_seq_len][d_model]
    // TODO(backward): add positional weight gradients so each position row
    // receives the gradient from its matching sequence index.
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
    pub fn transposed_weight(&self) -> Vec<Vec<f32>> {
        let mut transposed = vec![vec![0.0; self.vocab_size]; self.d_model];

        for token_id in 0..self.vocab_size {
            for dim in 0..self.d_model {
                transposed[dim][token_id] = self.weight[token_id][dim];
            }
        }
        transposed
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
