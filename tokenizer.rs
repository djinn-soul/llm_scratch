pub trait Tokenizer {
    fn encode(&mut self, text: &str) -> Result<Vec<usize>, String>;
    fn decode(&self, ids: &[usize]) -> Result<String, String>;
}
