mod tokenizer;
use crate::tokenizer::Tokenizer;
use std::{collections::HashMap, iter};

pub struct SentencePieceTokenizer {
    vocab: HashMap<String, f64>,         // token-> log_prob
    token_to_id: HashMap<String, usize>, // token-> id
    id_to_token: Vec<String>,            // id-> token
}

impl SentencePieceTokenizer {
    const MAX_SUBSTR_LEN: usize = 16;
    const SEED_VOCAB_MULTIPLIER: usize = 16;
    const SPECIAL_TOKEN_SPACE_REPLACE: &str = "▁";
    pub fn new() -> Self {
        Self {
            vocab: HashMap::new(),
            token_to_id: HashMap::new(),
            id_to_token: Vec::new(),
        }
    }

    fn build_seed_vocab(
        &mut self,
        text: &str,
        max_sub_str_len: usize,
        seed_vocab_size: usize,
    ) -> HashMap<String, f64> {
        // _marked word boundary: hello world -> _hello_world

        let marker = Self::SPECIAL_TOKEN_SPACE_REPLACE;

        let marked_text = text
            .split_whitespace()
            .map(|w| format!("{marker}{w}"))
            .collect::<Vec<_>>()
            .join("");

        // substring count upto max substing length
        let chars: Vec<char> = marked_text.chars().collect();
        let mut counts: HashMap<String, usize> = HashMap::new();
        for i in 0..chars.len() {
            for j in 1..max_sub_str_len.min(chars.len() - i) {
                let substr: String = chars[i..i + j].iter().collect();
                *counts.entry(substr).or_insert(0) += 1;
            }
        }

        //single characters alwasys in vocab...
        let single_chr: Vec<(String, usize)> = counts
            .iter()
            .filter(|(k, _)| k.chars().count() == 1)
            .map(|(k, v)| (k.clone(), *v))
            .collect();
        // multiple character and sort with top frequecny
        let mut mult_substr: Vec<(String, usize)> = counts
            .iter()
            .filter(|(k, _)| k.chars().count() > 1)
            .map(|(k, v)| (k.clone(), *v))
            .collect();
        // sort by frequency in descending order
        mult_substr.sort_by(|a, b| b.1.cmp(&a.1));

        //init seed vocab with all single characters
        let mut seed: HashMap<String, f64> = HashMap::new();
        for (t, c) in single_chr {
            seed.insert(t, c as f64);
        }
        // takeing remaining size of vocab from mult_substr sorted by frequency
        let remaining_size = seed_vocab_size - seed.len();
        for (t, c) in mult_substr.iter().take(remaining_size) {
            seed.insert(t.clone(), *c as f64);
        }
        let log_total_tokens: f64 = seed.values().sum();

        seed.into_iter()
            .map(|(k, v)| (k, (v / log_total_tokens).ln()))
            .collect()
    }

    fn viterbi(&self, text: &str) -> Vec<String> {
        let chars: Vec<char> = text.chars().collect();

        let n = chars.len();

        let mut dp = vec![f64::NEG_INFINITY; n + 1];
        let mut bck = vec![0usize; n + 1];
        dp[0] = 0.0;
        for i in 1..=n {
            for j in 0..i {
                let substr: String = chars[j..i].iter().collect();
                if let Some(&lp) = self.vocab.get(&substr) {
                    let score = dp[j] + lp;
                    if score > dp[i] {
                        dp[i] = score;
                        bck[i] = j;
                    }
                }
            }
        }
        let mut tokens = Vec::new();
        let mut i = n;
        //backtrack to recover the tokens and best segmentation
        while i > 0 {
            let j = bck[i];
            tokens.push(chars[j..i].iter().collect());
            i = j;
        }
        tokens.reverse();
        tokens
    }

    fn em_step(&self, sentesnce: &[&str]) -> HashMap<String, f64> {
        let mut counts: HashMap<String, f64> = HashMap::new();
        for sent in sentesnce {
            // tokenizing using viterbi for all sentences to probale tokens
            let tokens = self.viterbi(sent);
            for token in tokens {
                *counts.entry(token).or_insert(0.0) += 1.0;
            }
        }
        let total_tokens: f64 = counts.values().sum();

        counts
            .into_iter()
            .map(|(k, v)| (k, (v / total_tokens).ln()))
            .collect()
    }
    pub fn train(&mut self, text: &str, vocab_size: usize, _allowed_special: Option<Vec<String>>) {
        self.vocab = self.build_seed_vocab(
            text,
            Self::MAX_SUBSTR_LEN,
            vocab_size * Self::SEED_VOCAB_MULTIPLIER,
        );
        // self.save_vocab("./seed_vocab.txt").unwrap();
        // println!("Saved vocab to seed_vocab.txt");
        println!("Seed vocab size: {}", self.vocab.len());
        let marker = Self::SPECIAL_TOKEN_SPACE_REPLACE;

        let marked: Vec<String> = text
            .split_whitespace()
            .map(|w| format!("{marker}{w}"))
            .collect();
        let refs: Vec<&str> = marked.iter().map(|s| s.as_str()).collect();
        while self.vocab.len() > vocab_size {
            // E + M Step : recompute probs from viterbi counts
            self.vocab = self.em_step(&refs);
            println!("After EM: {} tokens", self.vocab.len());
        }
    }
    // fn save_vocab(&self, path: &str) -> Result<(), String> {
    //     let mut tokens: Vec<(&String, &f64)> = self.vocab.iter().collect();
    //     tokens.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap()); // highest log-prob first

    //     let lines: Vec<String> = tokens
    //         .iter()
    //         .map(|(tok, lp)| format!("{}\t{:.6}", tok, lp))
    //         .collect();

    //     std::fs::write(path, lines.join("\n")).map_err(|e| e.to_string())
    // }
}

impl Tokenizer for SentencePieceTokenizer {
    fn encode(&mut self, _text: &str) -> Result<Vec<usize>, String> {
        unimplemented!("SentencePiece encode is intentionally left as boilerplate");
    }

    fn decode(&self, _ids: &[usize]) -> Result<String, String> {
        unimplemented!("SentencePiece decode is intentionally left as boilerplate");
    }
}

fn main() {
    let text =
        std::fs::read_to_string("./the-verdict.txt").expect("Expect file but failed to read");

    println!("Read {} characters from the file.", text.len());

    let mut tokenizer = SentencePieceTokenizer::new();
    tokenizer.train(&text, 500, None);
}
