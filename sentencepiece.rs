use crate::tokenizer::Tokenizer;

pub struct SentencePieceTokenizer;

impl SentencePieceTokenizer {
    pub fn new() -> Self {
        Self
    }

    pub fn train(&mut self, _text: &str, _allowed_special: Option<Vec<String>>) {
        unimplemented!("SentencePiece training is intentionally left as boilerplate");
    }
}

impl Tokenizer for SentencePieceTokenizer {
    fn encode(&mut self, _text: &str) -> Result<Vec<usize>, String> {
        unimplemented!("SentencePiece encode is intentionally left as boilerplate");
    }

    fn decode(&self, _ids: &[usize]) -> Result<String, String> {
        unimplemented!("SentencePiece decode is intentionally left as boilerplate");
    }
}
