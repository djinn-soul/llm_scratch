// CORRECTED BPE IMPLEMENTATION
// https://sebastianraschka.com/blog/2025/bpe-from-scratch.html
use regex_lite::Regex;
use serde::Deserialize;
use std::collections::HashMap;
use std::collections::HashSet;
use std::fs;

// vocab: id -> token string (e.g. 65 -> "A")
// inverse_vocab: token string -> id (e.g. "A" -> 65)
// bpe_merges: (id_a, id_b) -> merged_id  (used during custom training)
// bpe_ranks: (str_a, str_b) -> rank      (used when loading GPT-2 merges)
pub struct BytePair {
    pub vocab: HashMap<usize, String>,
    pub inverse_vocab: HashMap<String, usize>,
    pub bpe_merges: HashMap<(usize, usize), usize>,
    pub bpe_ranks: HashMap<(String, String), usize>,
}

// Serializable struct for saving/loading bpe_merges to JSON
#[derive(serde::Serialize, Deserialize)]
pub struct MergeEntry {
    pub pair: [usize; 2],
    pub new_id: usize,
}

impl BytePair {
    pub fn new() -> Self {
        Self {
            vocab: HashMap::new(),
            inverse_vocab: HashMap::new(),
            bpe_merges: HashMap::new(),
            bpe_ranks: HashMap::new(),
        }
    }

    // Count adjacent token pairs and return the most/least frequent one
    pub fn find_freq_pair(&self, tokens: &Vec<usize>, mode: &str) -> Option<(usize, usize)> {
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

    // Replace every occurrence of pair_id in tokens with new_pair_id
    pub fn replace_pair(
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

    pub fn train(&mut self, text: &str, vocab_size: usize, allowed_special: Option<Vec<String>>) {
        // println!("Training BPE on text: {}", text);
        println!("Vocabulary size: {}", vocab_size);

        if let Some(ref special_tokens) = allowed_special {
            println!("Allowed special tokens: {:?}", special_tokens);
        }

        // Replace spaces with Ġ (GPT-2 style), skip leading space
        let mut processed_text: Vec<String> = Vec::new();
        for (i, item) in text.chars().enumerate() {
            if item == ' ' && i != 0 {
                processed_text.push("Ġ".to_string());
            } else if item != ' ' {
                processed_text.push(item.to_string());
            }
        }
        let process_text = processed_text.join("");

        // Start with first 256 Unicode scalar values as base vocab
        let mut unique_chars: Vec<char> = (0..256u32).filter_map(std::char::from_u32).collect();

        // Find any chars in text not already in base vocab
        let seen: HashMap<String, usize> = unique_chars
            .iter()
            .cloned()
            .enumerate()
            .map(|(i, c)| (c.to_string(), i))
            .collect();

        let extra_char: HashSet<char> = process_text
            .chars()
            .filter(|c| !seen.contains_key(&c.to_string()))
            .collect();
        println!("extra_char: {:?}", extra_char);

        // Add extra chars and ensure Ġ is in vocab
        unique_chars.extend(extra_char);
        if !unique_chars.contains(&'Ġ') {
            unique_chars.push('Ġ');
        }

        // Populate vocab and inverse_vocab with base characters
        for (i, c) in unique_chars.iter().enumerate() {
            self.vocab.insert(i, c.to_string());
            self.inverse_vocab.insert(c.to_string(), i);
        }

        // Add special tokens to vocab if provided
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

        // Convert processed text to token id sequence
        let mut token_ids: Vec<usize> = process_text
            .chars()
            .map(|i| {
                *self
                    .inverse_vocab
                    .get(&i.to_string())
                    .expect("Token not found in vocab")
            })
            .collect();

        // BPE training loop: repeatedly merge most frequent pair until vocab_size reached
        for new_id in self.vocab.len()..vocab_size {
            if let Some(pair) = self.find_freq_pair(&token_ids, "most") {
                token_ids = self.replace_pair(&mut token_ids, &pair, new_id);
                self.bpe_merges.insert(pair, new_id);
            } else {
                break;
            }
        }

        // Add merged tokens to vocab in order of their new ids
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
    }

    // Load pre-trained vocab and BPE merges from GPT-2 files.
    // vocab_path: path to encoder.json (maps token string -> id)
    // bpe_merges_path: path to vocab.bpe (each line is a merge pair, e.g. "h e")
    pub fn load_vocab_and_merges_from_llm(
        &mut self,
        vocab_path: &str,
        bpe_merges_path: &str,
    ) -> Result<(), String> {
        // Load vocab JSON: {"token_str": id, ...}
        let contents = fs::read_to_string(vocab_path).map_err(|e| e.to_string())?;
        let vocab_data_map: HashMap<String, usize> =
            serde_json::from_str(&contents).map_err(|e| e.to_string())?;
        for (token, id) in vocab_data_map {
            self.vocab.insert(id, token.clone());
            self.inverse_vocab.insert(token, id);
        }

        // GPT-2 vocab doesn't have "\n" — map it to a fallback token id
        if !self.inverse_vocab.contains_key("\n") {
            let fallback_id = ["<|endoftext|>", "Ġ", ""]
                .iter()
                .find_map(|t| self.inverse_vocab.get(*t).copied());
            let newline_id = fallback_id
                .ok_or_else(|| "No suitable token found in vocabulary to map '\\n'.".to_string())?;
            self.inverse_vocab.insert("\n".to_string(), newline_id);
            self.vocab.insert(newline_id, "\n".to_string());
        }

        // Load BPE merge file: each line is "token_a token_b", rank = line order
        let contents = fs::read_to_string(bpe_merges_path).map_err(|e| e.to_string())?;
        self.bpe_ranks = HashMap::new();
        let mut lines = contents.lines();

        // Skip header line (starts with #)
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
            // Only store merge if both tokens exist in vocab
            if self.inverse_vocab.contains_key(token_a) && self.inverse_vocab.contains_key(token_b)
            {
                self.bpe_ranks
                    .insert((token_a.to_string(), token_b.to_string()), rank);
                rank += 1;
            } else {
                println!(
                    "Skipping merge with unknown token: {}, {}",
                    token_a, token_b
                );
            }
        }

        Ok(())
    }

    // Escapes regex metacharacters so special tokens can be used as literals in regex
    // e.g. "<|endoftext|>" -> "<\|endoftext\|>"
    pub fn escape_regex(s: &str) -> String {
        s.chars()
            .flat_map(|c| {
                if r"\.+*?()|[]{}^$".contains(c) {
                    vec!['\\', c]
                } else {
                    vec![c]
                }
            })
            .collect()
    }

    pub fn encode(
        &mut self,
        text: String,
        allowed_special: Option<Vec<String>>,
    ) -> Result<Vec<usize>, String> {
        let mut tokens_ids: Vec<usize> = Vec::new();

        if let Some(ref allowed_special) = allowed_special {
            // Sort longest first to avoid partial matches (e.g. <|end|> before <|endoftext|>)
            let mut sorted_special = allowed_special.clone();
            sorted_special.sort_by_key(|t| std::cmp::Reverse(t.len()));

            // Build regex pattern from escaped special tokens joined by |
            let escaped: Vec<String> = sorted_special
                .iter()
                .map(|t| Self::escape_regex(t))
                .collect();
            let special_pattern = escaped.join("|");
            let re = Regex::new(&special_pattern).expect("Invalid regex pattern");

            let mut last_index = 0;
            for mat_chg in re.find_iter(&text) {
                // Encode plain text before this special token
                let prefix = &text[last_index..mat_chg.start()];
                if !prefix.is_empty() {
                    tokens_ids.extend(self.encode(prefix.to_string(), None)?);
                }

                // Look up special token id — error if not registered in vocab
                let special_token = mat_chg.as_str();
                let id = self.inverse_vocab.get(special_token).ok_or_else(|| {
                    format!("Special token '{}' not found in vocabulary", special_token)
                })?;
                tokens_ids.push(*id);
                last_index = mat_chg.end();
            }

            // Check remaining text for disallowed special tokens (not in allowed_special)
            let remaining_text = &text[last_index..];
            let mut disallowed_tokens = Vec::new();
            for tok in self.inverse_vocab.keys() {
                if tok.starts_with("<|")
                    && tok.ends_with("|>")
                    && remaining_text.contains(tok.as_str())
                    && !allowed_special.contains(tok)
                {
                    disallowed_tokens.push(tok.clone());
                }
            }
            if !disallowed_tokens.is_empty() {
                return Err(format!(
                    "Disallowed special tokens encountered in text:{:?}",
                    disallowed_tokens
                ));
            }

            // Encode any remaining plain text after the last special token
            if !remaining_text.is_empty() {
                tokens_ids.extend(self.encode(remaining_text.to_string(), None)?);
            }
            return Ok(tokens_ids);
        }

        // Split on newlines, prepend Ġ to words after a space (GPT-2 style)
        let mut tokens: Vec<String> = Vec::new();
        for (id, tok) in text.split("\n").enumerate() {
            if id > 0 {
                tokens.push("\n".to_string());
            }
            let words = tok.split(" ");
            for (id_x, word) in words.enumerate() {
                if id_x == 0 && id > 0 {
                    tokens.push(format!("Ġ{}", &word));
                } else if id_x == 0 {
                    tokens.push(word.to_string());
                } else {
                    tokens.push(format!("Ġ{}", &word));
                }
            }
        }

        // Map each token to its vocab id; unknown tokens go through BPE merging
        for tok in tokens {
            if self.inverse_vocab.contains_key(&tok) {
                tokens_ids.push(*self.inverse_vocab.get(&tok).unwrap());
            } else {
                tokens_ids.extend(match self.tokenize_with_bpe(&tok) {
                    Ok(tokens) => tokens,
                    Err(_) => panic!("Failed to tokenize: {}", tok),
                });
            }
        }
        Ok(tokens_ids)
    }

    pub fn tokenize_with_bpe(&self, text: &str) -> Result<Vec<usize>, String> {
        // ── STEP 1: Split text into characters, look up each char's ID ──────────
        // Example: token = "Ġlow"
        //   chars  →  ['Ġ', 'l', 'o', 'w']
        //   IDs    →  [256, 108, 111, 119]   (whatever ids were assigned during training)
        // If any character is missing from vocab, we error immediately.
        let mut tokens: Vec<usize> = text
            .chars()
            .map(|c| {
                self.inverse_vocab
                    .get(&c.to_string())
                    .copied()
                    .ok_or_else(|| format!("unknown token: '{c}'"))
            })
            .collect::<Result<Vec<usize>, _>>()?;

        // ── DECISION: which merge strategy to use? ───────────────────────────────
        // bpe_ranks is only populated when loading GPT-2 vocab (load_vocab_and_merges_from_llm).
        // bpe_merges is only populated during custom training (train).
        // So: empty bpe_ranks → we trained our own model → use bpe_merges path.
        if self.bpe_ranks.is_empty() {
            // ── CUSTOM TRAINING PATH ─────────────────────────────────────────────
            // bpe_merges stores: (id_a, id_b) → merged_id
            // Example after training: (108, 111) → 260  meaning "l"+"o" → "lo"
            //
            // We scan left-to-right, merge any known pair on the spot, then repeat
            // until no more pairs can be merged.
            //
            // Round 1 on [256, 108, 111, 119]  ("Ġ","l","o","w"):
            //   i=0: pair (256,108) → not in bpe_merges → keep 256, i=1
            //   i=1: pair (108,111) → in bpe_merges! → push 260 ("lo"), i=3, can_merge=true
            //   i=3: last token 119 → carry over
            //   tokens = [256, 260, 119]  ("Ġ","lo","w")
            //
            // Round 2 on [256, 260, 119]  ("Ġ","lo","w"):
            //   i=0: pair (256,260) → in bpe_merges! → push 300 ("Ġlo"), i=2, can_merge=true
            //   i=2: last token 119 → carry over
            //   tokens = [300, 119]  ("Ġlo","w")
            //
            // Round 3 on [300, 119]  ("Ġlo","w"):
            //   i=0: pair (300,119) → in bpe_merges! → push 350 ("Ġlow"), i=2
            //   tokens = [350]  ("Ġlow")
            //
            // can_merge stays false in next round → loop exits → return [350]
            let mut can_merge: bool = true;
            while can_merge && tokens.len() > 1 {
                can_merge = false;
                let mut new_tokens: Vec<usize> = Vec::new();
                let mut i: usize = 0;
                while i < tokens.len() - 1 {
                    let pair: (usize, usize) = (tokens[i], tokens[i + 1]);
                    if self.bpe_merges.contains_key(&pair) {
                        // Pair found in merge table → replace both with merged id
                        let merge_token: usize = self.bpe_merges.get(&pair).unwrap().clone();
                        new_tokens.push(merge_token);
                        i += 2; // skip both tokens (they became one)
                        can_merge = true; // signal: run another pass
                    } else {
                        // No merge for this pair → keep left token as-is
                        new_tokens.push(tokens[i]);
                        i += 1;
                    }
                }
                // The inner loop stops at len-1; if last token wasn't consumed by merge, add it
                if i < tokens.len() {
                    new_tokens.push(tokens[i])
                }
                tokens = new_tokens;
            }
            return Ok(tokens);
        }

        // ── GPT-2 PATH ───────────────────────────────────────────────────────────
        // bpe_ranks stores: ("l", "o") → rank  where lower rank = higher priority merge
        // Unlike custom path (id pairs), GPT-2 works on string symbols.
        //
        // STEP 2: convert ids back to string symbols so we can do rank-based merging
        // Example: [256, 108, 111, 119] → ["Ġ", "l", "o", "w"]
        let mut symbols: Vec<String> = tokens
            .iter()
            .map(|&tok| {
                self.vocab
                    .get(&tok)
                    .cloned()
                    .ok_or_else(|| format!("id {tok} not in vocab"))
            })
            .collect::<Result<_, _>>()?;

        loop {
            // STEP 3: collect every unique adjacent pair from current symbols
            // Example: ["Ġ","l","o","w"] → pairs = {("Ġ","l"), ("l","o"), ("o","w")}
            let pairs: HashSet<(String, String)> = symbols
                .windows(2)
                .map(|w| (w[0].clone(), w[1].clone()))
                .collect();

            if pairs.is_empty() {
                break; // single symbol left, nothing to merge
            }

            // STEP 4: find pair with lowest rank (= was merged earliest in GPT-2 training)
            // Example: bpe_ranks has ("l","o")→5, ("o","w")→99, ("Ġ","l")→200
            //   → bigram = ("l","o") with rank 5  (lowest = highest priority)
            let mut min_rank = usize::MAX;
            let mut bigram: Option<(String, String)> = None;
            for p in &pairs {
                if let Some(&rank) = self.bpe_ranks.get(p) {
                    if rank < min_rank {
                        min_rank = rank;
                        bigram = Some(p.clone());
                    }
                }
            }

            // If none of our pairs appear in bpe_ranks, no more merges possible
            let Some((p1, p2)) = bigram else {
                break;
            };

            // STEP 5: merge ALL occurrences of the winning pair in one pass
            // Example: bigram=("l","o"), symbols=["Ġ","l","o","w"]
            //   i=0: "Ġ" ≠ "l" → keep "Ġ", i=1
            //   i=1: "l"=="l" and "o"=="o" → push "lo", i=3
            //   i=3: "w" → keep "w", i=4
            //   symbols = ["Ġ","lo","w"]
            // Next loop iteration finds ("Ġ","lo") or ("lo","w") as next best pair, etc.
            let mut new_symbols: Vec<String> = Vec::new();
            let mut i: usize = 0;
            while i < symbols.len() {
                if i + 1 < symbols.len() && symbols[i] == p1 && symbols[i + 1] == p2 {
                    new_symbols.push(format!("{}{}", p1, p2)); // concatenate into merged symbol
                    i += 2;
                } else {
                    new_symbols.push(symbols[i].clone());
                    i += 1;
                }
            }
            symbols = new_symbols; // replace symbols with merged version, repeat loop
        }

        // STEP 6: convert final merged symbols back to token ids
        // Example: ["Ġlow"] → [350]
        let merged_ids: Vec<usize> = symbols
            .iter()
            .map(|tok| {
                self.inverse_vocab
                    .get(tok)
                    .copied()
                    .ok_or_else(|| format!("unknown token: '{tok}'"))
            })
            .collect::<Result<_, _>>()?;

        return Ok(merged_ids);
    }

    pub fn decode(&self, token_ids: Vec<usize>) -> Result<String, String> {
        let mut decoded = String::new();
        for id in &token_ids {
            let token = self
                .vocab
                .get(id)
                .ok_or_else(|| format!("unknown id: {id}"))?;
            if token == "\n" {
                if !decoded.ends_with(" ") && decoded.len() > 0 {
                    decoded.push_str(" ");
                }
            } else {
                decoded.push_str(token);
            }
        }
        let clean = decoded.replace('Ġ', " ");
        Ok(clean)
    }

    // Save vocab and bpe_merges to JSON files for later loading
    pub fn save_vocab_and_merges(
        &self,
        vocab_path: &str,
        bpe_merges_path: &str,
    ) -> Result<(), String> {
        fs::write(
            vocab_path,
            serde_json::to_string(&self.vocab).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;

        let merges_list: Vec<MergeEntry> = self
            .bpe_merges
            .iter()
            .map(|(&(id1, id2), &new_id)| MergeEntry {
                pair: [id1, id2],
                new_id,
            })
            .collect();

        fs::write(
            bpe_merges_path,
            serde_json::to_string(&merges_list).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        Ok(())
    }

    // Load previously saved vocab and bpe_merges from JSON files
    pub fn load_vocab_and_merges(
        &mut self,
        vocab_path: &str,
        bpe_merges_path: &str,
    ) -> Result<(), String> {
        let vocab_file_data = fs::read_to_string(vocab_path).map_err(|e| e.to_string())?;
        let bpe_merges_file_data =
            fs::read_to_string(bpe_merges_path).map_err(|e| e.to_string())?;

        let load_vocab_data: HashMap<usize, String> =
            serde_json::from_str(&vocab_file_data).unwrap();
        let load_bpe_merges_data: Vec<MergeEntry> =
            serde_json::from_str(&bpe_merges_file_data).unwrap();

        for (id, token) in load_vocab_data {
            self.vocab.insert(id, token.clone());
            self.inverse_vocab.insert(token, id);
        }

        for merge_entry in load_bpe_merges_data {
            self.bpe_merges.insert(
                (merge_entry.pair[0], merge_entry.pair[1]),
                merge_entry.new_id,
            );
        }

        Ok(())
    }
}

// Download file from url only if it doesn't already exist at dest_path
pub fn download_file_if_not_present(url: &str, dest_path: &str) {
    if fs::metadata(dest_path).is_ok() {
        println!("File already exists: {}", dest_path);
        return;
    }
    println!("Downloading from {}", url);
    let response = match reqwest::blocking::get(url) {
        Ok(response) => response,
        Err(e) => {
            println!("Failed to download from url: {}", e);
            return;
        }
    };
    println!("Download status: {}", response.status());
    let bytes = match response.bytes() {
        Ok(bytes) => bytes,
        Err(e) => {
            println!("Failed to read response bytes: {}", e);
            return;
        }
    };
    match fs::write(dest_path, bytes) {
        Ok(_) => println!("Successfully wrote to file"),
        Err(e) => println!("Failed to write to file: {}", e),
    };
}

// ════════════════════════════════════════════════════════════════════════════
// HOW BYTE PAIR ENCODING (BPE) WORKS — FULL WALKTHROUGH WITH EXAMPLE
// ════════════════════════════════════════════════════════════════════════════
//
// INPUT TEXT: "low lower newest"
//
// ── PHASE 1: PRE-PROCESSING ──────────────────────────────────────────────────
// Spaces are replaced with special character Ġ (GPT-2 style).
// Leading space is dropped. Each word boundary becomes visible.
//
//   "low lower newest"
//    → "lowĠlowerĠnewest"
//
// Why Ġ? So the tokenizer knows "low" at start of text vs "Ġlow" after a space
// are different surface forms, just like GPT-2 does.
//
// ── PHASE 2: BASE VOCABULARY ─────────────────────────────────────────────────
// Start with every Unicode character 0..255 as individual tokens.
// Each char gets an ID:
//
//   ID 101 → 'e'
//   ID 108 → 'l'
//   ID 111 → 'o'
//   ID 119 → 'w'
//   ID 110 → 'n'
//   ID 115 → 's'
//   ID 116 → 't'
//   ID 114 → 'r'
//   ID 256 → 'Ġ'   (added because not in 0..255)
//
// Also add special tokens like "<|endoftext|>" → ID 257
//
// ── PHASE 3: INITIAL TOKEN ID SEQUENCE ──────────────────────────────────────
// Convert the processed text character by character to IDs:
//
//   "lowĠlowerĠnewest"
//    → [108, 111, 119, 256, 108, 111, 119, 101, 114, 256, 110, 101, 119, 101, 115, 116]
//    →  l    o    w   Ġ    l    o    w    e    r   Ġ    n    e    w    e    s    t
//
// ── PHASE 4: BPE TRAINING LOOP ───────────────────────────────────────────────
// Repeatedly find the MOST FREQUENT adjacent pair and merge it into a new token.
// Each merge creates one new ID and shrinks the sequence.
//
// ITERATION 1:
//   Count all pairs:
//     (108,111)="lo" appears 2 times  ← most frequent
//     (111,119)="ow" appears 2 times
//     (119,256)="wĠ" appears 1 time
//     ... etc
//   Winner: (108,111) → new ID 258, new token "lo"
//   Replace ALL occurrences:
//     [108,111,119,256, 108,111,119,101,114, 256,110,101,119,101,115,116]
//      → [258,119,256, 258,119,101,114, 256,110,101,119,101,115,116]
//         lo  w  Ġ    lo  w  e  r      Ġ  n  e  w  e  s  t
//   Record: bpe_merges[(108,111)] = 258
//
// ITERATION 2:
//   Count pairs in new sequence:
//     (258,119)="low" appears 2 times  ← most frequent
//   Winner: (258,119) → new ID 259, new token "low"
//   Replace:
//     [258,119,256, 258,119,101,114, 256,110,101,119,101,115,116]
//      → [259,256, 259,101,114, 256,110,101,119,101,115,116]
//         low Ġ    low e   r    Ġ   n   e   w   e   s   t
//   Record: bpe_merges[(258,119)] = 259
//
// ITERATION 3:
//   (259,101)="lowe" appears 1 time, (101,114)="er" appears 1 time, ...
//   Winner: say (101,114)="er" → new ID 260
//   ... and so on until vocab reaches target size (e.g. 1000)
//
// ── PHASE 5: ENCODING A NEW TEXT ─────────────────────────────────────────────
// Given: "low newest"  (never seen exactly during training)
//
// Step A — split into words with Ġ prefix:
//   ["low", "Ġnewest"]
//
// Step B — for each word, check if whole word is in vocab:
//   "low"     → ID 259  (yes! learned during training)
//   "Ġnewest" → not in vocab → call tokenize_with_bpe("Ġnewest")
//
// Step C — tokenize_with_bpe("Ġnewest"):
//   chars → ['Ġ','n','e','w','e','s','t'] → IDs [256,110,101,119,101,115,116]
//   Round 1 scan pairs:
//     (256,110)="Ġn" → not in bpe_merges → keep 256
//     (110,101)="ne" → not in bpe_merges → keep 110
//     (101,119)="ew" → not in bpe_merges → keep 101
//     (119,101)="we" → not in bpe_merges → keep 119
//     (101,115)="es" → in bpe_merges! (if trained) → push merged id
//     ... continues until no more merges possible
//   Result: e.g. [256, 110, 101, 119, 260, 116]  ("Ġ","n","e","w","es","t")
//
// Step D — final token ids: [259, 256, 110, 101, 119, 260, 116]
//
// ── PHASE 6: DECODING ────────────────────────────────────────────────────────
// Convert IDs back to strings:
//   259 → "low"
//   256 → "Ġ"  → becomes " " (space before next word)
//   110 → "n"
//   101 → "e"
//   119 → "w"
//   260 → "es"
//   116 → "t"
//   Result: "low newest"  ✓
//
// ── SUMMARY: DATA STRUCTURES ─────────────────────────────────────────────────
//
//   vocab          HashMap<usize, String>         258 → "lo"
//   inverse_vocab  HashMap<String, usize>         "lo" → 258
//   bpe_merges     HashMap<(usize,usize), usize>  (108,111) → 258   ← custom training
//   bpe_ranks      HashMap<(String,String), usize> ("l","o") → 5    ← GPT-2 loading
//
//   bpe_merges is used during ENCODING (tokenize_with_bpe, custom path)
//   bpe_ranks  is used during ENCODING (tokenize_with_bpe, GPT-2 path)
//   Only one is populated at a time depending on how the model was loaded.
// ════════════════════════════════════════════════════════════════════════════
