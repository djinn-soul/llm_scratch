// ════════════════════════════════════════════════════════════════════════════
// SENTENCEPIECE — UNIGRAM LANGUAGE MODEL TOKENIZER
// ════════════════════════════════════════════════════════════════════════════
// Top-down subword tokenizer (opposite of BPE which is bottom-up).
//
// Algorithm:
//   1. SEED — build large initial vocabulary (~8000) of substring frequencies
//   2. EM   — expectation-maximization loop:
//             E-step: Viterbi decode every word using current log-probs
//             M-step: re-estimate log-probs from token usage counts
//   3. PRUNE — drop lowest log-prob tokens (never drop single chars or <unk>)
//   4. Repeat 2+3 until vocab shrinks to target size
//
// Key difference from BPE: no merge rules — each token has independent prob.
// Viterbi finds max-probability segmentation using DP.
// https://everdark.github.io/k9/notebooks/ml/natural_language_understanding/subword_units/subword_units.nb.html
// https://guillaume-be.github.io/2020-05-30/sentence_piece
// https://huggingface.co/learn/llm-course/en/chapter6/7
// https://mbrenndoerfer.com/writing/unigram-language-model-tokenization
// https://mbrenndoerfer.com/writing/sentencepiece-subword-tokenization-bpe-unigram
// https://en.wikipedia.org/wiki/Viterbi_algorithm
// ════════════════════════════════════════════════════════════════════════════

use crate::tokenizer::Tokenizer;
use std::collections::HashMap;

pub struct SentencePieceTokenizer {
    pub vocab: HashMap<String, f64>,
    pub token_to_id: HashMap<String, usize>,
    pub id_to_token: Vec<String>,
}

pub enum SeedMethod {
    Bpe,
    Substring,
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

    // ── PHASE 1: SEED VOCABULARY ────────────────────────────────────────────
    // Build initial large vocab from raw text. Steps:
    //   1. Prepend ▁ to each word: "hello world" → "▁hello▁world"
    //   2. Count every substring up to MAX_SUBSTR_LEN chars
    //   3. Keep ALL single chars (required — never pruned later)
    //   4. Fill remaining slots with top-frequency multi-char substrings
    //   5. Convert raw counts to log-probabilities: log(count / total)
    //   6. Add <unk> with very low log-prob (only used as fallback)
    fn build_seed_vocab(
        &mut self,
        text: &str,
        max_sub_str_len: usize,
        seed_vocab_size: usize,
    ) -> HashMap<String, f64> {
        let marker = Self::SPECIAL_TOKEN_SPACE_REPLACE;

        let marked_text = text
            .split_whitespace()
            .map(|w| format!("{marker}{w}"))
            .collect::<Vec<_>>()
            .join("");

        let chars: Vec<char> = marked_text.chars().collect();
        let mut counts: HashMap<String, usize> = HashMap::new();
        for i in 0..chars.len() {
            for j in 1..max_sub_str_len.min(chars.len() - i) {
                let substr: String = chars[i..i + j].iter().collect();
                *counts.entry(substr).or_insert(0) += 1;
            }
        }

        let single_chr: Vec<(String, usize)> = counts
            .iter()
            .filter(|(k, _)| k.chars().count() == 1)
            .map(|(k, v)| (k.clone(), *v))
            .collect();

        let mut mult_substr: Vec<(String, usize)> = counts
            .iter()
            .filter(|(k, _)| k.chars().count() > 1)
            .map(|(k, v)| (k.clone(), *v))
            .collect();
        mult_substr.sort_by(|a, b| b.1.cmp(&a.1));

        let mut seed: HashMap<String, f64> = HashMap::new();
        for (t, c) in single_chr {
            seed.insert(t, c as f64);
        }
        let remaining_size = seed_vocab_size - seed.len();
        for (t, c) in mult_substr.iter().take(remaining_size) {
            seed.insert(t.clone(), *c as f64);
        }

        let log_total_tokens: f64 = seed.values().sum();
        let mut result: HashMap<String, f64> = seed
            .into_iter()
            .map(|(k, v)| (k, (v / log_total_tokens).ln()))
            .collect();

        result.insert("<unk>".to_string(), -1e10);
        result
    }

    // ── ALTERNATIVE SEED: BPE-BASED ─────────────────────────────────────────
    // Use greedy pair-merging (BPE) instead of substring frequency for seeding.
    fn build_bpe_seed_vocab(text: &str, seed_size: usize) -> HashMap<String, usize> {
        let marker = Self::SPECIAL_TOKEN_SPACE_REPLACE;

        let mut words: Vec<Vec<String>> = text
            .split_whitespace()
            .map(|w| {
                let mut toks: Vec<String> = w.chars().map(|c| c.to_string()).collect();
                toks.insert(0, marker.to_string());
                toks
            })
            .collect();

        let mut token_counts: HashMap<String, usize> = HashMap::new();
        for word in &words {
            for token in word {
                *token_counts.entry(token.clone()).or_insert(0) += 1;
            }
        }

        for _ in 0..seed_size {
            let mut pair_counts: HashMap<(String, String), usize> = HashMap::new();
            for word in &words {
                if word.len() < 2 {
                    continue;
                }
                for i in 0..word.len() - 1 {
                    let pair = (word[i].clone(), word[i + 1].clone());
                    *pair_counts.entry(pair).or_insert(0) += 1;
                }
            }

            let max_pair = pair_counts.iter().max_by_key(|&(_, c)| c);
            let Some(((a, b), &cnt)) = max_pair else {
                break;
            };
            if cnt < 2 {
                break;
            }

            let (a, b) = (a.clone(), b.clone());
            let merge = format!("{a}{b}");
            token_counts.insert(merge.clone(), cnt);

            for word in words.iter_mut() {
                let mut new_word: Vec<String> = Vec::new();
                let mut i = 0;
                while i < word.len() - 1 {
                    if i + 1 < word.len() && word[i] == a && word[i + 1] == b {
                        new_word.push(merge.clone());
                        i += 2;
                    } else {
                        new_word.push(word[i].clone());
                        i += 1;
                    }
                }
                if i < word.len() {
                    new_word.push(word[i].clone());
                }
                *word = new_word;
            }
        }

        token_counts
    }

    // ── VITERBI DECODE ──────────────────────────────────────────────────────
    // Find best (highest log-prob) segmentation of text using current vocab.
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
        while i > 0 {
            let j = bck[i];
            tokens.push(chars[j..i].iter().collect());
            i = j;
        }
        tokens.reverse();
        tokens
    }

    // ── EM STEP ─────────────────────────────────────────────────────────────
    // One iteration of Expectation-Maximization.
    fn em_step(&self, sentesnce: &[&str]) -> HashMap<String, f64> {
        let mut counts: HashMap<String, f64> = HashMap::new();
        for sent in sentesnce {
            let tokens = self.viterbi(sent);
            for token in tokens {
                *counts.entry(token).or_insert(0.0) += 1.0;
            }
        }

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

    // ── PRUNE ───────────────────────────────────────────────────────────────
    // Shrink vocab to target_size by keeping highest log-prob tokens.
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

    // ── TRAIN — FULL PIPELINE ───────────────────────────────────────────────
    pub fn train(
        &mut self,
        text: &str,
        vocab_size: usize,
        seed_method: SeedMethod,
        _allowed_special: Option<Vec<String>>,
    ) {
        let seed_size = vocab_size * Self::SEED_VOCAB_MULTIPLIER;

        self.vocab = match seed_method {
            SeedMethod::Substring => self.build_seed_vocab(text, Self::MAX_SUBSTR_LEN, seed_size),
            SeedMethod::Bpe => {
                let counts = Self::build_bpe_seed_vocab(text, seed_size);
                let total: usize = counts.values().sum();
                let mut v: HashMap<String, f64> = counts
                    .iter()
                    .map(|(k, c)| (k.clone(), (*c as f64 / total as f64).ln()))
                    .collect();
                v.insert("<unk>".to_string(), -1e10);
                v
            }
        };
        println!("Seed vocab size: {}", self.vocab.len());

        let marker = Self::SPECIAL_TOKEN_SPACE_REPLACE;
        let marked: Vec<String> = text
            .split_whitespace()
            .map(|w| format!("{marker}{w}"))
            .collect();
        let refs: Vec<&str> = marked.iter().map(|s| s.as_str()).collect();

        while self.vocab.len() > vocab_size {
            self.vocab = self.em_step(&refs);
            println!("After EM: {} tokens", self.vocab.len());

            let target_size = (self.vocab.len() * 8 / 10).max(vocab_size);
            self.vocab = self.prune(target_size);
            println!("After Prune: {} tokens", self.vocab.len());

            let mut tokens: Vec<String> = self.vocab.keys().cloned().collect();
            tokens.sort();
            self.id_to_token = tokens.clone();
            self.token_to_id = tokens
                .iter()
                .enumerate()
                .map(|(i, t)| (t.clone(), i))
                .collect();
        }
    }

    pub fn save(&self, path: &str) -> Result<(), String> {
        let data = serde_json::json!({
            "vocab": self.vocab,
            "id_to_token": self.id_to_token,
        });
        std::fs::write(path, data.to_string()).map_err(|e| e.to_string())
    }

    pub fn load(&mut self, path: &str) -> Result<(), String> {
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
