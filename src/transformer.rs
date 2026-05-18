use crate::feed_forward::FeedForward;
use crate::layer_norm::LayerNorm;
use crate::multi_head_attention::MultiHeadAttention;
use crate::util::add_mat;
use crate::util::random_matrix;

pub struct Transformer {
    pub layer_norm: LayerNorm,
    pub mha: MultiHeadAttention,
    pub layer_norm2: LayerNorm,
    pub ff: FeedForward,
}

impl Transformer {
    pub fn new(d_model: usize, num_heads: usize, d_ff: usize) -> Self {
        let w1 = random_matrix(d_model, d_ff);
        let w2 = random_matrix(d_ff, d_model);
        Self {
            layer_norm: LayerNorm::new(d_model),
            mha: MultiHeadAttention::new(d_model, num_heads),
            layer_norm2: LayerNorm::new(d_model),
            ff: FeedForward::new(w1, w2, d_model, d_ff),
        }
    }

    pub fn forward(&self, x: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let norml = self.layer_norm.forward(x); //normalize first
        let attention = self.mha.forward(&norml); // self.attention
        let h = add_mat(x, &attention); // x + attention
        let norm2 = self.layer_norm2.forward(&h); // from learning residuals from original output
        let ff = self.ff.forward(&norm2);
        add_mat(&h, &ff) //h + ff
    }
}
