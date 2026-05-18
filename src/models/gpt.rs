use crate::attention::self_attention::matmul;
use crate::common::util::random_matrix;
use crate::layers::embedding::{embed_sequence, PositionalEmbedding, TokenEmbedding};
use crate::layers::layer_norm::LayerNorm;
use crate::models::transformer::Transformer;

pub struct GPT {
    pub token_emb: TokenEmbedding,
    pub position_emb: PositionalEmbedding,
    pub blocks: Vec<Transformer>,
    pub norm: LayerNorm,
    pub lm_head: Vec<Vec<f32>>,
}

impl GPT {
    pub fn forward(&self, tokens: &[usize]) -> Vec<Vec<f32>> {
        
        // token id with positional encoding and token embedding 
        let mut x = embed_sequence(tokens, &self.token_emb, &self.position_emb);

        // add to transformer decoder blocks
        for block in &self.blocks {
            x = block.forward(&x);
        }
        // final normalization
        let x = self.norm.forward(&x);

//output layer 
        matmul(&x, &self.lm_head)
    }
}
