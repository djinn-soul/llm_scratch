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
// ════════════════════════════════════════════════════════════════════════════

mod tokenizer;
use crate::tokenizer::Tokenizer;
use std::{collections::HashMap, fmt::format};

pub struct SentencePieceTokenizer {
    vocab: HashMap<String, f64>,         // token -> log_prob (negative floats)
    token_to_id: HashMap<String, usize>, // token -> integer id (frozen after training)
    id_to_token: Vec<String>,            // id -> token (index = id)
}

impl SentencePieceTokenizer {
    // Max length of substring to consider during seed vocab building
    const MAX_SUBSTR_LEN: usize = 16;
    // Seed vocab is 16x final target — gives EM room to converge before pruning
    const SEED_VOCAB_MULTIPLIER: usize = 16;
    // U+2581 LOWER ONE EIGHTH BLOCK — marks word boundary (standard SentencePiece convention)
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

        // Mark word boundaries: "hello world" → "▁hello▁world"
        let marked_text = text
            .split_whitespace()
            .map(|w| format!("{marker}{w}"))
            .collect::<Vec<_>>()
            .join("");

        // Count every substring up to max length — outer = start pos, inner = length
        let chars: Vec<char> = marked_text.chars().collect();
        let mut counts: HashMap<String, usize> = HashMap::new();
        for i in 0..chars.len() {
            for j in 1..max_sub_str_len.min(chars.len() - i) {
                let substr: String = chars[i..i + j].iter().collect();
                *counts.entry(substr).or_insert(0) += 1;
            }
        }

        // Single chars must always be in vocab — guarantees full text coverage
        let single_chr: Vec<(String, usize)> = counts
            .iter()
            .filter(|(k, _)| k.chars().count() == 1)
            .map(|(k, v)| (k.clone(), *v))
            .collect();

        // Multi-char substrings — sort by frequency descending, take top N
        let mut mult_substr: Vec<(String, usize)> = counts
            .iter()
            .filter(|(k, _)| k.chars().count() > 1)
            .map(|(k, v)| (k.clone(), *v))
            .collect();
        mult_substr.sort_by(|a, b| b.1.cmp(&a.1));

        // Fill seed: all single chars + top-N multi-char by frequency
        let mut seed: HashMap<String, f64> = HashMap::new();
        for (t, c) in single_chr {
            seed.insert(t, c as f64);
        }
        let remaining_size = seed_vocab_size - seed.len();
        for (t, c) in mult_substr.iter().take(remaining_size) {
            seed.insert(t.clone(), *c as f64);
        }

        // Convert raw counts to log-probabilities: log(count / total)
        let log_total_tokens: f64 = seed.values().sum();
        let mut result: HashMap<String, f64> = seed
            .into_iter()
            .map(|(k, v)| (k, (v / log_total_tokens).ln()))
            .collect();

        // <unk> at -1e10 ensures Viterbi never picks it for valid paths
        // Only used in fallback when no segmentation exists
        result.insert("<unk>".to_string(), -1e10);

        result
    }

    // ── VITERBI DECODE ──────────────────────────────────────────────────────
    // Find best (highest log-prob) segmentation of text using current vocab.
    //
    // DP recurrence:
    //   dp[i] = max log-prob to segment chars[0..i]
    //   dp[i] = max over j<i of (dp[j] + log_prob(chars[j..i]))   if substr in vocab
    //   bck[i] = best j that achieved dp[i]  (for backtracking)
    //
    // Backtracking: i=n, push chars[bck[i]..i], jump i=bck[i], repeat until 0.
    //
    // TODO: subword regularization — n-best sampling, track top-K paths in DP,
    //       sample weighted by exp(alpha * score) for training robustness
    fn viterbi(&self, text: &str) -> Vec<String> {
        let chars: Vec<char> = text.chars().collect();
        let n = chars.len();

        // dp[i] = best log-prob ending at position i; back[i] = predecessor index
        let mut dp = vec![f64::NEG_INFINITY; n + 1];
        let mut bck = vec![0usize; n + 1];
        dp[0] = 0.0; // empty prefix has prob 1, log = 0

        // Forward pass — for each end position, try every possible start
        for i in 1..=n {
            for j in 0..i {
                let substr: String = chars[j..i].iter().collect();
                if let Some(&lp) = self.vocab.get(&substr) {
                    let score = dp[j] + lp; // log-prob addition = prob multiplication
                    if score > dp[i] {
                        dp[i] = score;
                        bck[i] = j;
                    }
                }
            }
        }

        // Fallback: if no valid path exists (some char not in vocab),
        // emit each char individually, replacing unknowns with <unk>
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

        // Backward pass — walk bck[] from n to 0, emit tokens in reverse
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
    //
    // E-step: Viterbi-decode every sentence with current probs → count token usage
    // M-step: new log_prob[t] = log(count[t] / total_count)
    //
    // Tokens with count = 0 naturally drop out of vocab (Viterbi never picked them).
    //
    // Smoothing: force single chars + <unk> to have at least count 0.5,
    // preventing them from being dropped — they're essential for fallback paths.
    fn em_step(&self, sentesnce: &[&str]) -> HashMap<String, f64> {
        // E-step: count expected token usage from best Viterbi paths
        let mut counts: HashMap<String, f64> = HashMap::new();
        for sent in sentesnce {
            let tokens = self.viterbi(sent);
            for token in tokens {
                *counts.entry(token).or_insert(0.0) += 1.0;
            }
        }

        // Smoothing: keep single chars + <unk> alive even when unused
        // Without this, EM drops them → next iteration has no fallback paths
        for tok in self.vocab.keys() {
            if tok.chars().count() == 1 || tok == "<unk>" {
                counts.entry(tok.clone()).or_insert(0.5);
            }
        }

        // M-step: normalize counts to probabilities, take log
        let total_tokens: f64 = counts.values().sum();
        counts
            .into_iter()
            .map(|(k, v)| (k, (v / total_tokens).ln()))
            .collect()
    }

    // ── PRUNE ───────────────────────────────────────────────────────────────
    // Shrink vocab to target_size by keeping highest log-prob tokens.
    //
    // Invariants:
    //   - All single chars always kept (full character coverage)
    //   - <unk> always kept (fallback for unknown inputs)
    //   - Remaining slots filled with multi-char tokens by log-prob descending
    fn prune(&self, target_size: usize) -> HashMap<String, f64> {
        // Single chars — always preserved
        let single_chars: Vec<(String, f64)> = self
            .vocab
            .iter()
            .filter(|(k, _)| k.chars().count() == 1)
            .map(|(k, v)| (k.clone(), *v))
            .collect();

        // Multi-char tokens — sorted by log-prob descending (highest first)
        let mut multi_char: Vec<(String, f64)> = self
            .vocab
            .iter()
            .filter(|(k, _)| k.chars().count() > 1)
            .map(|(k, v)| (k.clone(), *v))
            .collect();
        multi_char.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Build result: single chars + <unk> first (count toward budget)
        let mut result: HashMap<String, f64> = single_chars.iter().cloned().collect();
        let unk_lp = self.vocab.get("<unk>").copied();
        if let Some(lp) = unk_lp {
            result.insert("<unk>".to_string(), lp);
        }

        // Fill remaining slots with top multi-char tokens
        let remaining_size = target_size.saturating_sub(result.len());
        for (token, log_pb) in multi_char.iter().take(remaining_size) {
            result.insert(token.clone(), *log_pb);
        }

        result
    }

    // ── TRAIN — FULL PIPELINE ───────────────────────────────────────────────
    // Builds tokenizer from raw text:
    //   1. Seed vocab at vocab_size * 16 tokens
    //   2. Pre-mark all words: ["hello", "world"] → ["▁hello", "▁world"]
    //   3. Loop EM + prune (shrink by 20% per iter) until vocab_size reached
    //   4. Freeze vocab → assign stable integer ids (sorted alphabetically)
    pub fn train(&mut self, text: &str, vocab_size: usize, _allowed_special: Option<Vec<String>>) {
        // Step 1: large seed vocab
        self.vocab = self.build_seed_vocab(
            text,
            Self::MAX_SUBSTR_LEN,
            vocab_size * Self::SEED_VOCAB_MULTIPLIER,
        );
        println!("Seed vocab size: {}", self.vocab.len());

        // Step 2: prepare corpus as list of ▁-prefixed words for per-word Viterbi
        let marker = Self::SPECIAL_TOKEN_SPACE_REPLACE;
        let marked: Vec<String> = text
            .split_whitespace()
            .map(|w| format!("{marker}{w}"))
            .collect();
        let refs: Vec<&str> = marked.iter().map(|s| s.as_str()).collect();

        // Step 3: EM + prune loop — gradual shrink to target
        while self.vocab.len() > vocab_size {
            // E-step + M-step: recompute log-probs from Viterbi token counts
            self.vocab = self.em_step(&refs);
            println!("After EM: {} tokens", self.vocab.len());

            // Prune: shrink by 20% each iter (or jump to target if close)
            let target_size = (self.vocab.len() * 8 / 10).max(vocab_size);
            self.vocab = self.prune(target_size);
            println!("After Prune: {} tokens", self.vocab.len());

            // Step 4: freeze vocab → stable ids (sort makes deterministic)
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

    // Persist trained tokenizer as JSON (vocab log-probs + id ordering)
    // token_to_id is reconstructed on load from id_to_token (no need to save both)
    fn save(&self, path: &str) -> Result<(), String> {
        let data = serde_json::json!({
            "vocab": self.vocab,
            "id_to_token": self.id_to_token,
        });
        std::fs::write(path, data.to_string()).map_err(|e| e.to_string())
    }

    // Restore tokenizer state from saved JSON file
    fn load(&mut self, path: &str) -> Result<(), String> {
        let content = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
        let data: serde_json::Value = serde_json::from_str(&content).map_err(|e| e.to_string())?;

        self.vocab = serde_json::from_value(data["vocab"].clone()).map_err(|e| e.to_string())?;
        self.id_to_token =
            serde_json::from_value(data["id_to_token"].clone()).map_err(|e| e.to_string())?;

        // Rebuild reverse map from saved id order
        self.token_to_id = self
            .id_to_token
            .iter()
            .enumerate()
            .map(|(i, t)| (t.clone(), i))
            .collect();

        Ok(())
    }
}

// ── TOKENIZER TRAIT IMPL ────────────────────────────────────────────────────
// encode/decode are exposed through the Tokenizer trait so SentencePiece
// is interchangeable with other tokenizers (like BPE).
impl Tokenizer for SentencePieceTokenizer {
    // Encode text → token ids:
    //   1. Split on whitespace
    //   2. Prefix each word with ▁
    //   3. Viterbi-decode each marked word into tokens
    //   4. Map tokens to integer ids
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

    // Decode token ids → text:
    //   1. Map ids back to token strings
    //   2. Concatenate all tokens
    //   3. Replace ▁ with space, trim leading space
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
    // Load training corpus (reuses the-verdict.txt from BPE)
    let text =
        std::fs::read_to_string("./the-verdict.txt").expect("Expect file but failed to read");
    println!("Read {} characters from the file.", text.len());

    // Train: seed → EM/prune loop → final vocab of 500 tokens
    let mut tokenizer = SentencePieceTokenizer::new();
    tokenizer.train(&text, 500, None);

    // Round-trip test 1: in-vocab text
    let ids = tokenizer.encode("the quick brown a fox").unwrap();
    println!("ids: {:?}", ids);
    println!("decoded: {:?}", tokenizer.decode(&ids).unwrap());

    // Round-trip test 2: unknown chars (Greek) → become <unk>
    let ids = tokenizer.encode("the quick αβγ fox").unwrap();
    println!("ids: {:?}", ids);
    println!("decoded: {:?}", tokenizer.decode(&ids).unwrap());

    // Save/load test: train once, save, load into fresh tokenizer, re-encode
    tokenizer.save("./sp_model.json").unwrap();
    let mut sp2 = SentencePieceTokenizer::new();
    sp2.load("./sp_model.json").unwrap();
    let ids = sp2.encode("the quick brown fox").unwrap();
    println!("loaded encode: {:?}", ids);
    println!("decoded: {:?}", sp2.decode(&ids).unwrap());
}

// ════════════════════════════════════════════════════════════════════════════
// HOW UNIGRAM LM SENTENCEPIECE WORKS — FULL WALKTHROUGH WITH EXAMPLE
// ════════════════════════════════════════════════════════════════════════════
//
// INPUT TEXT (small example): "low low lower newest widest"
//
// ── PHASE 1: SEED VOCABULARY ─────────────────────────────────────────────────
// Step A — mark word boundaries (replace spaces with ▁):
//   "low low lower newest widest"
//   → "▁low▁low▁lower▁newest▁widest"
//
// Step B — count every substring up to MAX_SUBSTR_LEN (16):
//   chars = ['▁','l','o','w','▁','l','o','w','▁','l','o','w','e','r','▁','n','e','w','e','s','t','▁','w','i','d','e','s','t']
//   For each (i, len) pair, count substring chars[i..i+len].
//   counts = {
//     "▁": 5, "l": 3, "o": 3, "w": 4, "e": 4, "r": 1, "n": 1, "s": 2, "t": 2,
//     "i": 1, "d": 1, "▁l": 3, "lo": 3, "ow": 3, "▁lo": 3, "low": 3, "▁low": 3,
//     "owe": 1, "wer": 1, "ower": 1, "▁lower": 1, "▁n": 1, "ne": 1, "ew": 2,
//     "▁ne": 1, "▁new": 1, "newe": 1, "wes": 1, "est": 2, "▁newest": 1,
//     "▁w": 1, "wi": 1, "id": 1, "de": 1, "wid": 1, "ide": 1, "des": 1,
//     "▁widest": 1, ...
//   }
//
// Step C — separate single chars (always keep) from multi-char:
//   single_chars: ▁, l, o, w, e, r, n, s, t, i, d  (11 tokens)
//   multi_chars: sorted by count desc — "▁": 5, "low": 3, "▁low": 3, "lo": 3, ...
//
// Step D — fill seed vocab (single chars + top multi-char by count):
//   If seed_vocab_size = 8000 and we have 11 single chars,
//   take top 7989 multi-char substrings.
//
// Step E — counts → log-probs:
//   total = sum of all counts (e.g. 200)
//   log_prob["▁"] = log(5/200) = log(0.025) ≈ -3.69
//   log_prob["low"] = log(3/200) = log(0.015) ≈ -4.20
//   log_prob["t"] = log(2/200) = log(0.01) ≈ -4.61
//
// Step F — add <unk> with log_prob = -1e10 (only used as fallback)
//
// ── PHASE 2: VITERBI DECODING ────────────────────────────────────────────────
// Find best segmentation of "▁lower" using seed log-probs.
//
//   chars = ['▁','l','o','w','e','r']  (n = 6)
//   dp[0..6] = [0.0, -∞, -∞, -∞, -∞, -∞, -∞]   bck = [0,0,0,0,0,0,0]
//
// Forward pass:
//   i=1: j=0, substr="▁", lp=-3.69
//        score = dp[0] + lp = -3.69 > dp[1]=-∞ → dp[1]=-3.69, bck[1]=0
//
//   i=2: j=0, substr="▁l", lp=-4.50 (from seed)
//        score = -4.50 > -∞ → dp[2]=-4.50, bck[2]=0
//        j=1, substr="l", lp=-5.20
//        score = -3.69 + -5.20 = -8.89 < -4.50 → no change
//        Result: dp[2]=-4.50, bck[2]=0  (means "▁l" picked as single token)
//
//   i=3: j=0, "▁lo", lp=-4.20 → score=-4.20  → dp[3]=-4.20, bck[3]=0
//        j=1, "lo", lp=-5.10 → score=-3.69+-5.10=-8.79 (worse)
//        j=2, "o", lp=-5.20 → score=-4.50+-5.20=-9.70 (worse)
//        Result: dp[3]=-4.20, bck[3]=0
//
//   i=4: j=0, "▁low", lp=-4.05 → dp[4]=-4.05, bck[4]=0
//        (everything else worse)
//
//   i=5: j=0, "▁lowe", lp=NOT IN VOCAB → skip
//        j=1, "lowe", lp=NOT IN VOCAB → skip
//        j=2, "owe", lp=-7.5 → score=-4.50+-7.5=-12 (best so far)
//        j=3, "we", lp=-6.8 → score=-4.20+-6.8=-11
//        j=4, "e", lp=-4.0 → score=-4.05+-4.0=-8.05  ← best
//        Result: dp[5]=-8.05, bck[5]=4   (means "▁low" + "e")
//
//   i=6: j=0..4 all worse than:
//        j=4, "er", lp=-5.5 → score=-4.05+-5.5=-9.55  ← best
//        j=5, "r", lp=-6.0 → score=-8.05+-6.0=-14.05
//        Result: dp[6]=-9.55, bck[6]=4   (means "▁low" + "er")
//
// Backward pass:
//   i=6: j=bck[6]=4, push chars[4..6]="er", i=4
//   i=4: j=bck[4]=0, push chars[0..4]="▁low", i=0
//   Reverse: ["▁low", "er"]
//
// "▁lower" → ["▁low", "er"]   (uses 2 tokens instead of 6 individual chars)
//
// ── PHASE 3: EM ITERATION ────────────────────────────────────────────────────
// E-step: Viterbi-decode every word, count token usage
//   "▁low"     → ["▁low"]              count("▁low") += 1
//   "▁low"     → ["▁low"]              count("▁low") += 1  → total 2
//   "▁lower"   → ["▁low", "er"]        count("▁low") += 1, count("er") += 1
//   "▁newest"  → ["▁n", "ew", "est"]   etc.
//   "▁widest"  → ["▁w", "id", "est"]
//
//   counts = {"▁low": 3, "er": 1, "▁n": 1, "ew": 1, "est": 2, "▁w": 1, "id": 1}
//
//   Smoothing: ensure all single chars + <unk> have count ≥ 0.5
//   (so they survive when Viterbi never picks them)
//
// M-step: normalize → new log_probs
//   total = 11 (say)
//   log_prob["▁low"] = log(3/11) ≈ -1.30   (high prob — picked often)
//   log_prob["est"] = log(2/11) ≈ -1.70
//   log_prob["o"]   = log(0.5/11) ≈ -3.09  (smoothed, low prob)
//
// Notice: tokens not picked by Viterbi (with no smoothing) vanish.
//   "low" was in seed vocab but not used → drops out.
//   "lo" was in seed vocab but not used → drops out.
//   This is how Unigram LM naturally shrinks vocab during EM.
//
// ── PHASE 4: PRUNE ───────────────────────────────────────────────────────────
// After EM: vocab might be 1500 tokens. Target = 500. Need to drop 1000.
//
// Prune logic:
//   1. Identify single chars → always keep (say 50 chars)
//   2. Keep <unk> → always keep (1 more)
//   3. Multi-char tokens sorted by log-prob desc
//   4. Take top (target_size - 51) = 449 multi-char tokens
//
//   target_size per iter = max(current * 0.8, final_target)
//     1500 → 1200 → 960 → 768 → 614 → 500
//
//   Gradual shrink allows EM to re-segment with each smaller vocab.
//
// ── PHASE 5: TRAIN LOOP ──────────────────────────────────────────────────────
// while vocab.len() > 500:
//   run em_step  → recompute probs (some tokens drop naturally)
//   compute target = max(vocab.len() * 0.8, 500)
//   run prune    → shrink to target
//   freeze       → assign stable integer ids (sorted alphabetically)
//
// Output: vocab = exactly 500 tokens, each with log-prob and integer id.
//
// ── PHASE 6: ENCODE ──────────────────────────────────────────────────────────
// Input: "the quick brown fox"
//
// Step 1: split on whitespace → ["the", "quick", "brown", "fox"]
// Step 2: prefix each with ▁ → ["▁the", "▁quick", "▁brown", "▁fox"]
// Step 3: Viterbi each separately → tokens
//   "▁the"   → ["▁the"]
//   "▁quick" → ["▁qu", "ick"]
//   "▁brown" → ["▁br", "own"]
//   "▁fox"   → ["▁f", "ox"]
// Step 4: map tokens to ids → [457, 218, 174, 277, 168, 322, 213]
//
// ── PHASE 7: DECODE ──────────────────────────────────────────────────────────
// Input ids: [457, 218, 174, 277, 168, 322, 213]
//
// Step 1: map ids back to tokens → ["▁the", "▁qu", "ick", "▁br", "own", "▁f", "ox"]
// Step 2: concatenate → "▁the▁quick▁brown▁fox"
// Step 3: replace ▁ with space → " the quick brown fox"
// Step 4: trim leading space → "the quick brown fox"
//
// ── KEY DIFFERENCES FROM BPE ─────────────────────────────────────────────────
//
//                  BPE                          Unigram LM (SentencePiece)
//                  --------------------------   --------------------------------
//   Direction      Bottom-up (merge pairs)      Top-down (prune tokens)
//   Pre-tokenize   Yes (split on spaces)        No (raw Unicode + ▁ marker)
//   Vocab grows    From chars upward            From large seed downward
//   Decoding       Greedy left-to-right merge   Viterbi DP (optimal path)
//   Probabilities  None (rule-based merging)    Yes (log-probs, EM-estimated)
//   Word boundary  Ġ prefix (GPT-2 convention)  ▁ prefix (SentencePiece convention)
//
// ── DATA STRUCTURES ──────────────────────────────────────────────────────────
//
//   vocab          HashMap<String, f64>      "▁low" → -1.30 (log-prob)
//   token_to_id    HashMap<String, usize>    "▁low" → 42
//   id_to_token    Vec<String>               id_to_token[42] = "▁low"
//
//   <unk>          Always at id 0 (or wherever sort puts it) with log-prob -1e10
//                  Used as fallback when input contains chars not in training data
// ════════════════════════════════════════════════════════════════════════════
