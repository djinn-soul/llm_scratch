// https://dubeyrahul.github.io/posts/llm-from-scratch/token-embeddings.html
// https://www.yadavsaurabh.com/building-a-transformer-llm-with-code-evolution-of-positional-encoding/
use crate::common::util::add;
use rand::RngExt;

pub struct TokenEmbedding {
    weight: Vec<Vec<f32>>, //[vocab_size][d_model]
    // BACKWARD: gradient table mirrors `weight`.
    // Each output-row gradient is scattered back into its token row; repeated
    // token IDs accumulate into the same `d_weight[token_id]` row.
    pub d_weight: Vec<Vec<f32>>, //[vocab_size][d_model]

    pub vocab_size: usize,
    pub d_model: usize, // dimension of token embedding
}

pub struct PositionalEmbedding {
    weight: Vec<Vec<f32>>, // [max_seq_len][d_model]
    // BACKWARD: gradient table mirrors `weight`.
    // Each sequence position accumulates into the matching absolute-position
    // row: position 0 -> d_weight[0], position 1 -> d_weight[1], etc.
    pub d_weight: Vec<Vec<f32>>, // [max_seq_len][d_model]

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
            // Start with no accumulated gradients; backward adds into rows.
            d_weight: vec![vec![0.0; d_model]; vocab_size],

            vocab_size,
            d_model,
        }
    }
    pub fn forward(&self, ids: usize) -> Vec<f32> {
        self.weight[ids].clone()
    }

    pub fn backward(&mut self, ids: &[usize], d_out: &[Vec<f32>]) {
        // ── BACKWARD: TOKEN EMBEDDING SCATTER ──────────────────────────────
        // Forward copies weight[token_id] into the sequence output.
        // Backward sends each output-row gradient back into that same token row.
        // If a token appears more than once, each occurrence adds to the row.
        for i in 0..ids.len() {
            let token_id = ids[i];
            for j in 0..self.d_model {
                self.d_weight[token_id][j] += d_out[i][j];
            }
        }
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
            // Start with no accumulated gradients; backward adds into rows.
            d_weight: vec![vec![0.0; d_model]; max_seq_len],
            max_seq_len,
            d_model,
        }
    }
    pub fn forward(&self, ids: usize) -> Vec<f32> {
        self.weight[ids].clone()
    }

    pub fn backward(&mut self, seq_len: usize, d_out: &[Vec<f32>]) {
        // ── BACKWARD: POSITION EMBEDDING SCATTER ───────────────────────────
        // Forward copies weight[position] into the sequence output.
        // Backward sends each position's gradient back into its matching row:
        // d_out[0] -> position 0, d_out[1] -> position 1, and so on.
        for i in 0..seq_len {
            for j in 0..self.d_model {
                self.d_weight[i][j] += d_out[i][j];
            }
        }
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
