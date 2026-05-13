use llm_scratch_rs::sentence_piece::{SeedMethod, SentencePieceTokenizer};
use llm_scratch_rs::tokenizer::Tokenizer;

fn main() {
    let text =
        std::fs::read_to_string("./the-verdict.txt").expect("Expect file but failed to read");
    println!("Read {} characters from the file.", text.len());

    let mut tokenizer = SentencePieceTokenizer::new();
    tokenizer.train(&text, 500, SeedMethod::Bpe, None);

    let ids = tokenizer.encode("the quick brown a fox").unwrap();
    println!("ids: {:?}", ids);
    println!("decoded: {:?}", tokenizer.decode(&ids).unwrap());

    let ids = tokenizer.encode("the quick αβγ fox").unwrap();
    println!("ids: {:?}", ids);
    println!("decoded: {:?}", tokenizer.decode(&ids).unwrap());

    tokenizer.save("./sp_model.json").unwrap();
    let mut sp2 = SentencePieceTokenizer::new();
    sp2.load("./sp_model.json").unwrap();
    let mut tokenizer = SentencePieceTokenizer::new();
    tokenizer.train(&text, 500, SeedMethod::Substring, None);
    let ids = sp2.encode("the quick brown fox").unwrap();
    println!("loaded encode: {:?}", ids);
    println!("decoded: {:?}", sp2.decode(&ids).unwrap());
}
