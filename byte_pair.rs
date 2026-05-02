// CORRECTED BPE IMPLEMENTATION
use std::collections::HashMap;
struct BytePair {
    vocab: HashMap<String, usize>,
    inverse_vocab: HashMap<usize, String>,
    bpe_merges: HashMap<String, usize>,
    bpe_ranks: HashMap<String, usize>,
}

impl BytePair {
    fn new() -> Self {
        Self {
            vocab: HashMap::new(),
            inverse_vocab: HashMap::new(),
            bpe_merges: HashMap::new(),
            bpe_ranks: HashMap::new(),
        }
    }

    fn find_frequent_pair(&self, tokens: &Vec<usize>, mode: &str) -> Option<(usize, usize)> {
        let mut pair_count: HashMap<(usize, usize), usize> = HashMap::new();
        for t in tokens.windows(2) {
            *pair_count.entry((t[0], t[1])).or_insert(0) += 1;
        }
        if pair_count.is_empty() {
            return None;
        }
        match mode {
            "most" => pair_count
                .into_iter()
                .max_by_key(|(_, v)| *v)
                .map(|(k, _)| k),
            "least" => pair_count
                .into_iter()
                .min_by_key(|(_, v)| *v)
                .map(|(k, _)| k),
            _ => panic!("Invalid mode {}", mode),
        }
    }
    fn replace_pair(
        &mut self,
        tokens: &mut Vec<usize>,
        pair_id: &(usize, usize),
        new_pair_id: usize,
    ) -> Vec<usize> {
        let mut replace = Vec::new();
        let mut i = 0;
        let n = tokens.len();
        while i < n {
            if i + 1 < n && tokens[i] == pair_id.0 && tokens[i + 1] == pair_id.1 {
                replace.push(new_pair_id);
                i += 2;
            } else {
                replace.push(tokens[i]);
                i += 1
            }
        }
        replace
    }

    fn train(&mut self, text: &str, vocab_size: usize, allowed_special: Option<Vec<String>>) {
        let mut processed_text: Vec<String> = Vec::new();

        println!("Training BPE on text: {}", text);
        println!("Vocabulary size: {}", vocab_size);

        if let Some(ref special_tokens) = allowed_special {
            println!("Allowed special tokens: {:?}", special_tokens);
        }

        for (i, item) in text.chars().enumerate() {
            if item == ' ' && i != 0 {
                processed_text.push("Ġ".to_string());
            } else if item != ' ' {
                processed_text.push(item.to_string());
            }
        }

        let process_text = processed_text.join("");
        // println!("processed_text: {:?}", process_text);

        // Correctly generate the first 256 Unicode scalar values
        let mut unique_chars: Vec<char> = (0..256u32).filter_map(std::char::from_u32).collect();

        // println!("unique_chars count: {:?}", unique_chars.len());

        let seen: HashMap<String, usize> = unique_chars
            .iter()
            .cloned()
            .enumerate()
            .map(|(i, c)| (c.to_string(), i))
            .collect();
        // println!("seen: {:?}", seen);

        let extra_char: Vec<char> = process_text
            .chars()
            .filter(|c| !seen.contains_key(&c.to_string()))
            .collect();
        // println!("extra_char: {:?}", extra_char);

        unique_chars.extend(extra_char);
        if !unique_chars.contains(&'Ġ') {
            unique_chars.push('Ġ');
        }

        for (i, c) in unique_chars.iter().enumerate() {
            self.vocab.insert(c.to_string(), i);
            self.inverse_vocab.insert(i, c.to_string());
        }

        if let Some(tokens) = allowed_special {
            for token in tokens {
                if !self.vocab.contains_key(&token) {
                    let idx = self.vocab.len();
                    self.vocab.insert(token.to_string(), idx);
                    self.inverse_vocab.insert(idx, token.to_string());
                }
            }
        }

        println!("self.vocab count: {:?}", self.vocab.len());

        let mut token_ids: Vec<usize> = process_text
            .chars()
            .map(|i| {
                *self
                    .vocab
                    .get(&i.to_string())
                    .expect("Token not found in vocab")
            })
            .collect();
        println!("token_ids: {:?}", token_ids);
        // let pair = self.find_frequent_pair(&token_ids, "most");
        // println!("Most frequent pair: {:?}", pair);
        for new_id in self.vocab.len()..vocab_size {
            // let pair = self.find_frequent_pair(&token_ids, "most");
            // if pair.is_none() {
            //     break;
            // }
            if let Some(pair) = self.find_frequent_pair(&token_ids, "most") {
                token_ids = self.replace_pair(&mut token_ids, &pair, new_id)
            } else {
                break;
            }
        }
        println!("final token_ids: {:?}", token_ids);
    }
}

fn main() {
    let mut bpe = BytePair::new();
    bpe.train(
        "hello world , i come from hell",
        300,
        Some(vec!["<|PAD|>".to_string(), "<|UNK|>".to_string()]),
    );
    println!("BytePair initialized successfully!");
}
