// https://sebastianraschka.com/blog/2023/self-attention-from-scratch.html
// https://machinelearningmastery.com/the-attention-mechanism-from-scratch/
use crate::attention::{matmul, random_matrix, SelfAttention};

pub struct MultiHeadAttention {
    pub heads: Vec<SelfAttention>,
    pub w_o: Vec<Vec<f32>>, // [num_heads * d_v][d_model]
    pub num_heads: usize,
    pub d_model: usize,
}

impl MultiHeadAttention {
    pub fn new(d_model: usize, num_heads: usize) -> Self {
        assert!(
            d_model % num_heads == 0,
            "d_model must be divisible by num_heads"
        );
        let d_k = d_model / num_heads;
        let d_v = d_k; // For simplicity, we set d_v = d_k
        let heads: Vec<SelfAttention> = (0..num_heads)
            .map(|_| SelfAttention::new(d_model, d_k, d_v))
            .collect();
        let w_o = random_matrix(num_heads * d_v, d_model);

        Self {
            heads,
            w_o: w_o,
            num_heads,
            d_model,
        }
    }

    pub fn forward(&self, x: &[Vec<f32>]) -> Vec<Vec<f32>> {
        // part 1: compute attention for each head
        let head_outputs: Vec<Vec<Vec<f32>>> =
            self.heads.iter().map(|head| head.forward(x)).collect();

        // part 2: concatenate head outputs
        // row-wise concatenation
        // row : head1[0] + head2[0] + ... + headN[0]
        let seq_len: usize = x.len();
        let mut concatenated: Vec<Vec<f32>> = Vec::new();
        for i in 0..seq_len {
            let mut row: Vec<f32> = Vec::new();
            for h in head_outputs.iter() {
                row.extend(&h[i]);
            }
            concatenated.push(row);
        }
        matmul(&concatenated, &self.w_o)
    }
}
