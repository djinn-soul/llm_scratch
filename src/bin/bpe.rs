use llm_scratch_rs::tokenizers::bpe::{download_file_if_not_present, BytePair};
use std::fs;

fn main() {
    let mut bpe = BytePair::new();
    let allowed_special = Some(vec!["<|endoftext|>".to_string()]);

    let url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt";
    let _ = download_file_if_not_present(url, "./the-verdict.txt");

    let text = fs::read_to_string("./the-verdict.txt").expect("Failed to read file");
    println!("BytePair initialized successfully!");

    bpe.train(&text, 1000, allowed_special);
    println!("BytePair trained successfully!");

    bpe.save_vocab_and_merges("./vocab.json", "./bpe_merges.json")
        .unwrap();
    println!("Vocab: {}", bpe.vocab.len());
    println!("merges: {}", bpe.bpe_merges.len());

    let input_text = "Jack embraced beauty through art and life.<|endoftext|> ";

    let tokens = bpe.encode(input_text.to_string(), None).unwrap();
    println!("{:?}", tokens);

    let tokens_with_special = bpe
        .encode(
            input_text.to_string(),
            Some(vec!["<|endoftext|>".to_string()]),
        )
        .unwrap();
    println!("{:?}", tokens_with_special);

    println!("Number of characters: {}", input_text.chars().count());
    println!("Number of token IDs: {}", tokens_with_special.len());

    for i in &tokens_with_special {
        println!(
            "{}",
            format!("id: {}--> {}", i, bpe.decode(vec![*i]).unwrap())
        );
    }
    println!(
        "Decoded: {}",
        bpe.decode(tokens_with_special.clone()).unwrap()
    );

    let mut bpe1 = BytePair::new();
    println!("Loading vocab and merges...");
    bpe1.load_vocab_and_merges("./vocab.json", "./bpe_merges.json")
        .unwrap();
    println!("Vocab: {}", bpe1.vocab.len());
    println!("merges: {}", bpe1.bpe_merges.len());

    println!(
        "Decoded Loaded: {}",
        bpe1.decode(tokens_with_special.clone()).unwrap()
    );

    let gpt2_files = [
        (
            "https://openaipublic.blob.core.windows.net/gpt-2/models/124M/vocab.bpe",
            "./vocab.bpe",
        ),
        (
            "https://openaipublic.blob.core.windows.net/gpt-2/models/124M/encoder.json",
            "./encoder.json",
        ),
    ];
    for (url, dest) in &gpt2_files {
        download_file_if_not_present(url, dest);
    }

    let mut bpe_gpt2 = BytePair::new();
    bpe_gpt2
        .load_vocab_and_merges_from_llm("./encoder.json", "./vocab.bpe")
        .unwrap();
    println!("GPT-2 vocab size: {}", bpe_gpt2.vocab.len());
    println!("GPT-2 bpe_ranks size: {}", bpe_gpt2.bpe_ranks.len());
    let input_text = "This is some text";
    println!(
        "{:?}",
        bpe_gpt2.encode(input_text.to_string(), None).unwrap()
    );
}
