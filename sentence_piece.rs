mod tokenizer;
use crate::tokenizer::Tokenizer;
use std::{collections::HashMap, fmt::format};

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

        let mut result: HashMap<String, f64> = seed
            .into_iter()
            .map(|(k, v)| (k, (v / log_total_tokens).ln()))
            .collect();
        // unknown token handling
        result.insert("<unk>".to_string(), -1e10);

        result
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
        // after DP loop, before backtrack — replace unknown chars with <unk>
        if dp[n] == f64::NEG_INFINITY {
            return chars
                .iter()
                .map(|c| {
                    let s = c.to_string();
                    if self.vocab.contains_key(&s) {
                        s
                    } else {
                        "<unk>".to_string()
                    }
                })
                .collect();
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
        // smoothing: keep all single chars even if viterbi never picked them
        for tok in self.vocab.keys() {
            if tok.chars().count() == 1 || tok == "<unk>" {
                counts.entry(tok.clone()).or_insert(0.5);
            }
        }
        let total_tokens: f64 = counts.values().sum();

        counts
            .into_iter()
            .map(|(k, v)| (k, (v / total_tokens).ln()))
            .collect()
    }

    fn prune(&self, target_size: usize) -> HashMap<String, f64> {
        let single_chars: Vec<(String, f64)> = self
            .vocab
            .iter()
            .filter(|(k, _)| k.chars().count() == 1)
            .map(|(k, v)| (k.clone(), *v))
            .collect();

        let mut multi_char: Vec<(String, f64)> = self
            .vocab
            .iter()
            .filter(|(k, _)| k.chars().count() > 1)
            .map(|(k, v)| (k.clone(), *v))
            .collect();
        //
        multi_char.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let mut result: HashMap<String, f64> = single_chars.iter().cloned().collect();
        let unk_lp = self.vocab.get("<unk>").copied();

        if let Some(lp) = unk_lp {
            result.insert("<unk>".to_string(), lp);
        }
        let remaining_size = target_size.saturating_sub(result.len());

        for (token, log_pb) in multi_char.iter().take(remaining_size) {
            result.insert(token.clone(), *log_pb);
        }

        result
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

            // keep top 80 % of vocab
            let target_size = (self.vocab.len() * 8 / 10).max(vocab_size);

            self.vocab = self.prune(target_size);
            println!("After Prune: {} tokens", self.vocab.len());
            // freeze vocab → assign stable ids
            let mut tokens: Vec<String> = self.vocab.keys().cloned().collect();
            tokens.sort(); // deterministic ids
            self.id_to_token = tokens.clone();
            self.token_to_id = tokens
                .iter()
                .enumerate()
                .map(|(i, t)| (t.clone(), i))
                .collect();
        }
    }

    fn save(&self, path: &str) -> Result<(), String> {
        let data = serde_json::json!({
            "vocab":self.vocab,
            "id_to_token":self.id_to_token,
        });
        std::fs::write(path, data.to_string()).map_err(|e| e.to_string())
    }

    fn load(&mut self, path: &str) -> Result<(), String> {
        let content = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
        let data: serde_json::Value = serde_json::from_str(&content).map_err(|e| e.to_string())?;

        self.vocab = serde_json::from_value(data["vocab"].clone()).map_err(|e| e.to_string())?;

        self.id_to_token =
            serde_json::from_value(data["id_to_token"].clone()).map_err(|e| e.to_string())?;

        self.token_to_id = self
            .id_to_token
            .iter()
            .enumerate()
            .map(|(i, t)| (t.clone(), i))
            .collect();

        Ok(())
    }
}

impl Tokenizer for SentencePieceTokenizer {
    fn encode(&mut self, text: &str) -> Result<Vec<usize>, String> {
        let marker = Self::SPECIAL_TOKEN_SPACE_REPLACE;
        let mut ids = Vec::new();
        for word in text.split_whitespace() {
            let marked = format!("{marker}{word}");
            let tokens = self.viterbi(&marked);
            for t in tokens {
                let id = self
                    .token_to_id
                    .get(&t)
                    .ok_or_else(|| format!("Unknown token: {}", t))?;
                ids.push(*id);
            }
        }
        Ok(ids)
    }

    fn decode(&self, ids: &[usize]) -> Result<String, String> {
        let marker = Self::SPECIAL_TOKEN_SPACE_REPLACE;
        let mut s = String::new();
        for &id in ids {
            let tok = self
                .id_to_token
                .get(id)
                .ok_or_else(|| format!("unknown id: {id}"))?;
            s.push_str(tok);
        }
        Ok(s.replace(marker, " ").trim_start().to_string())
    }
}

fn main() {
    let text =
        std::fs::read_to_string("./the-verdict.txt").expect("Expect file but failed to read");

    println!("Read {} characters from the file.", text.len());

    let mut tokenizer = SentencePieceTokenizer::new();
    tokenizer.train(&text, 500, None);
    let ids = tokenizer.encode("the quick brown a fox").unwrap();
    println!("ids: {:?}", ids);
    println!("decoded: {:?}", tokenizer.decode(&ids).unwrap());
    let ids = tokenizer.encode("the quick αβγ fox").unwrap();
    println!("ids: {:?}", ids);
    println!("decoded: {:?}", tokenizer.decode(&ids).unwrap());
    tokenizer.save("./sp_model.json").unwrap();
    let mut sp2 = SentencePieceTokenizer::new();
    sp2.load("./sp_model.json").unwrap();
    let ids = sp2.encode("the quick brown fox").unwrap();
    println!("loaded encode: {:?}", ids);
    println!("decoded: {:?}", sp2.decode(&ids).unwrap());
}
