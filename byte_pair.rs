// CORRECTED BPE IMPLEMENTATION
use std::collections::HashMap;
use std::fs;
struct BytePair {
    vocab: HashMap<usize, String>,
    inverse_vocab: HashMap<String, usize>,
    bpe_merges: HashMap<(usize, usize), usize>,
    bpe_ranks: HashMap<(usize, usize), usize>,
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
            self.vocab.insert(i, c.to_string());
            self.inverse_vocab.insert(c.to_string(), i);
        }

        if let Some(tokens) = allowed_special {
            for token in tokens {
                if !self.inverse_vocab.contains_key(&token) {
                    let idx = self.vocab.len();
                    self.vocab.insert(idx, token.to_string());
                    self.inverse_vocab.insert(token.to_string(), idx);
                }
            }
        }

        println!("self.vocab count: {:?}", self.vocab.len());

        let mut token_ids: Vec<usize> = process_text
            .chars()
            .map(|i| {
                *self
                    .inverse_vocab
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
                token_ids = self.replace_pair(&mut token_ids, &pair, new_id);
                self.bpe_merges.insert(pair, new_id);
            } else {
                break;
            }
        }
        println!("final token_ids: {:?}", token_ids);

        let mut sorted_merges: Vec<_> = self.bpe_merges.iter().collect();
        sorted_merges.sort_by_key(|(_, &new_id)| new_id);
        for (pair, new_id) in sorted_merges {
            let merged_token = format!(
                "{}{}",
                self.vocab.get(&pair.0).unwrap(),
                self.vocab.get(&pair.1).unwrap()
            );
            self.vocab.insert(*new_id, merged_token.clone());
            self.inverse_vocab.insert(merged_token, *new_id);
        }
        println!("self.vocab count: {:?}", self.vocab.len());
        println!("self.inverse_vocab count: {:?}", self.inverse_vocab.len());
        println!("self.bpe_merges count: {:?}", self.bpe_merges.len());
        println!("self.bpe_ranks: {:?}", self.bpe_ranks);
        println!("self.bpe_merges: {:?}", self.bpe_merges);
    }

    // Load pre-trained vocab and BPE merges from GPT-2 files.
    // vocab_path: path to encoder.json (maps token string -> id)
    // bpe_merges_path: path to vocab.bpe (each line is a merge pair, e.g. "h e")
    fn load_vocab_and_merges_from_llm(
        &mut self,
        vocab_path: &str,
        bpe_merges_path: &str,
    ) -> Result<(), String> {
        //load vocab from file

        let contents = fs::read_to_string(vocab_path).map_err(|e| e.to_string())?;

        let vocab_data_map: HashMap<String, usize> =
            serde_json::from_str(&contents).map_err(|e| e.to_string())?;
        for (token, id) in vocab_data_map {
            self.vocab.insert(id, token.clone());
            self.inverse_vocab.insert(token, id);
        }

        if !self.inverse_vocab.contains_key("\n") {
            let fallback_id = ["<|endoftext|>", "Ġ", ""]
                .iter()
                .find_map(|t| self.inverse_vocab.get(*t).copied());
            let newline_id = fallback_id
                .ok_or_else(|| "No suitable token found in vocabulary to map '\\n'.".to_string())?;
            self.inverse_vocab.insert("\n".to_string(), newline_id);
            self.vocab.insert(newline_id, "\n".to_string());
        }

        // load merge pairs from file
        let contents = fs::read_to_string(bpe_merges_path).map_err(|e| e.to_string())?;

        self.bpe_ranks = HashMap::new();
        let mut lines = contents.lines();
        if let Some(first) = lines.next() {
            if !first.starts_with('#') {
                println!("No header found: {}", first);
            }
        }
        let mut rank: usize = 0;

        for line in lines {
            let parts: Vec<&str> = line.split(' ').collect();
            if parts.len() != 2 {
                println!("Invalid line: {}", line);
                continue;
            }
            let token_a = parts[0];
            let token_b = parts[1];
            if let (Some(&id_a), Some(&id_b)) = (
                self.inverse_vocab.get(token_a),
                self.inverse_vocab.get(token_b),
            ) {
                self.bpe_ranks.insert((id_a, id_b), rank);
                rank += 1;
            } else {
                println!("Skipping merge with unknown token: {}, {}", token_a, token_b);
            }
        }

        Ok(())
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
