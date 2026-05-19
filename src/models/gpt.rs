use crate::attention::self_attention::matmul;
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
    pub fn new(
        vocab_size: usize,
        d_model: usize,
        max_seq_len: usize,
        num_heads: usize,
        d_ff: usize,
        num_blocks: usize,
    ) -> Self {
        // 1. Token Embeddings: Maps vocabulary IDs to dense vectors of size `d_model`.

        let token_emb = TokenEmbedding::new(vocab_size, d_model);
        // 2. Positional Embeddings: Gives the model a sense of order/sequence position.
        let position_emb = PositionalEmbedding::new(max_seq_len, d_model);
        // 3. Transformer Decoder Blocks: Stacks of self-attention and feed-forward layers.
        let mut blocks = Vec::with_capacity(num_blocks);
        for _ in 0..num_blocks {
            blocks.push(Transformer::new(d_model, num_heads, d_ff));
        }

        // 4. final layer normalization
        let norm = LayerNorm::new(d_model);
        // lm head

        let lm_head = token_emb.transposed_weight();

        Self {
            token_emb,
            position_emb,
            blocks,
            norm,
            lm_head,
        }
    }
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

    pub fn generate(&self, context: &[usize], max_new_tokens: usize) -> Vec<usize> {
        let mut tokens = context.to_vec();
        let max_seq_len = self.position_emb.max_seq_len;
        for _ in 0..max_new_tokens {
            // If we have more tokens than chairs...
            let start_idx = if tokens.len() > max_seq_len {
                tokens.len() - max_seq_len // Shift the starting point forward
            } else {
                0
            };

            // Only look at the newest `max_seq_len` tokens!
            let cropped_tokens = &tokens[start_idx..];

            // get predicatios
            let logits = self.forward(&cropped_tokens);

            // get last token predicatios
            let last_logits = logits.last().unwrap();

            // 3. Greedy Decoding: Find the index (vocab ID) with the highest score
            let mut best_id = 0;
            let mut highest_score = f32::NEG_INFINITY;

            for (id, score) in last_logits.iter().enumerate() {
                if score > &highest_score {
                    highest_score = *score;
                    best_id = id;
                }
            }
            tokens.push(best_id);
        }
        tokens
    }
}
