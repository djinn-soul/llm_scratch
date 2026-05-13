# llm_scratch_rs

Building an LLM from scratch in Rust. Each piece implemented by hand — no `torch`, no `transformers`, no `tiktoken`. Follows the structure of Sebastian Raschka's *LLMs from Scratch* book, but rewritten in idiomatic Rust.

## Index

- [Status](#status)
- [Project Layout](#project-layout)
- [Run](#run)
- [Quick Start (API)](#quick-start-api)
- [Tokenizer Trait](#tokenizer-trait)
- [BPE](#bpe--byte-pair-encoding)
- [SentencePiece (Unigram LM)](#sentencepiece--unigram-language-model)
- [BPE vs Unigram](#bpe-vs-unigram-lm)
- [Data Loaders](#data-loaders)
- [Roadmap](#roadmap)
- [Learning Progression](#learning-progression)
- [Dependencies](#dependencies)
- [License](#license)

## Status

Maps to chapters of *LLMs from Scratch* (Sebastian Raschka):

| Topic | Book Ch. | Status |
|-------|----------|--------|
| Tokenizer trait | infra | ✅ Done |
| Byte Pair Encoding (BPE) | Ch 2 | ✅ Done |
| SentencePiece (Unigram LM) | Ch 2 extension | ✅ Done |
| Data loaders | Ch 2 | ⬜ Next |
| Self-attention | Ch 3 | ⬜ |
| Multi-head attention | Ch 3 | ⬜ |
| Transformer block | Ch 4 | ⬜ |
| Full GPT model | Ch 4 | ⬜ |
| Training loop | Ch 5 | ⬜ |
| Sampling / generation | Ch 5 | ⬜ |
| Fine-tuning | Ch 6+7 | ⬜ |

**Requires:** Rust 1.70+ (edition 2021).

## Project Layout

```
.
├── tokenizer.rs            # shared Tokenizer trait (encode/decode)
├── bpe.rs                  # BPE tokenizer (GPT-2 style)
├── sentence_piece.rs       # SentencePiece (Unigram LM)
├── the-verdict.txt         # training corpus (Edith Wharton short story)
├── vocab.json              # saved BPE vocab
├── bpe_merges.json         # saved BPE merge rules
├── encoder.json            # GPT-2 official vocab (downloaded)
├── vocab.bpe               # GPT-2 official merges (downloaded)
├── sp_model.json           # saved SentencePiece model
├── seed_vocab.txt          # debug — dumped seed vocab
├── Cargo.toml
├── PLANS.md                # rough roadmap / scratchpad
└── CONTINUITY.md           # session notes between work sessions
```

**Corpus:** `the-verdict.txt` is an Edith Wharton short story, sourced from [rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch02/01_main-chapter-code/the-verdict.txt) — auto-downloaded on first run.

Each tokenizer is its own `[[bin]]` target — run independently.

## Run

```bash
# Train + run BPE
cargo run --bin byte_pair

# Train + run SentencePiece (Unigram)
cargo run --bin sentence_piece
```

Both will:
1. Download/read training corpus (`the-verdict.txt`)
2. Train a tokenizer from scratch
3. Encode + decode a test string
4. Save the trained model to JSON

## Quick Start (API)

```rust
use crate::tokenizer::Tokenizer;

// SentencePiece example
let text = std::fs::read_to_string("./the-verdict.txt")?;
let mut tok = SentencePieceTokenizer::new();
tok.train(&text, 500, SeedMethod::Bpe, None);  // vocab_size=500

let ids = tok.encode("hello world")?;
let back = tok.decode(&ids)?;
assert_eq!(back, "hello world");

tok.save("./sp_model.json")?;

// Later: load without retraining
let mut tok2 = SentencePieceTokenizer::new();
tok2.load("./sp_model.json")?;
let ids = tok2.encode("the quick brown fox")?;
```

BPE has same `Tokenizer` trait — swap struct, same calls.

## Tokenizer Trait

```rust
pub trait Tokenizer {
    fn encode(&mut self, text: &str) -> Result<Vec<usize>, String>;
    fn decode(&self, ids: &[usize]) -> Result<String, String>;
}
```

Both `BytePair` and `SentencePieceTokenizer` implement this — interchangeable at the call site.

---

## BPE — Byte Pair Encoding

**Reference:** [rasbt's BPE blog post](https://sebastianraschka.com/blog/2025/bpe-from-scratch.html)

**Algorithm (bottom-up):**
1. Start with every Unicode char 0..255 as base vocab
2. Pre-tokenize: replace spaces with `Ġ` (GPT-2 convention)
3. Find most-frequent adjacent token pair → merge into new token
4. Record merge rule `(id_a, id_b) → new_id`
5. Repeat until vocab reaches target size

**Encoding** uses learned merge rules — for each word, apply merges left-to-right until no more pairs match.

**Two modes supported:**
- **Custom training:** uses `bpe_merges: HashMap<(usize, usize), usize>` — id pairs
- **GPT-2 loading:** uses `bpe_ranks: HashMap<(String, String), usize>` — string pairs with merge priority

Save/load both vocab and merge rules to JSON. Can also load GPT-2's official `encoder.json` + `vocab.bpe`.

---

## SentencePiece — Unigram Language Model

**References:**
- [Unigram Language Model Tokenization (interactive)](https://mbrenndoerfer.com/writing/unigram-language-model-tokenization)
- [SentencePiece overview](https://mbrenndoerfer.com/writing/sentencepiece-subword-tokenization-bpe-unigram)
- [HuggingFace tokenizer summary](https://huggingface.co/learn/llm-course/en/chapter6/7)

**Algorithm (top-down):** opposite of BPE — start with a huge vocabulary and prune down.

1. **Seed** — build a large initial vocab (~8000 tokens):
   - Option A: substring frequency (count every substring up to 16 chars)
   - Option B: BPE-based seeding (run greedy merges, take learned tokens)
2. **EM loop** — alternate Expectation + Maximization:
   - **E-step:** Viterbi-decode every word using current log-probs → count token usage
   - **M-step:** new `log_prob[t] = log(count[t] / total_count)`
3. **Prune** — drop lowest-prob tokens until vocab shrinks by 20%
4. **Repeat** EM + prune until target size (e.g. 500)

### Viterbi (DP segmentation)

For each word, find segmentation with maximum log-probability sum:

```
dp[i]  = best log-prob to segment chars[0..i]
bck[i] = predecessor index for backtracking

dp[i] = max over j<i of (dp[j] + log_prob(chars[j..i]))
```

Backtrack from `n` to `0` using `bck[]` to recover the token sequence.

### Word Boundary Marker

Uses `▁` (U+2581) prepended to each word — no pre-tokenization needed. Lets the tokenizer learn space-aware tokens like `▁the`, `▁and`.

### Special Tokens

| Token | Purpose |
|-------|---------|
| `▁` | Word boundary marker |
| `<unk>` | Fallback for unseen characters (log-prob = -1e10) |

### Seed Method Choice

```rust
pub enum SeedMethod {
    Substring,  // count substrings — fast, noisy
    Bpe,        // BPE merges — slower, cleaner subwords
}

tokenizer.train(&text, 500, SeedMethod::Bpe, None);
```

---

## BPE vs Unigram LM

| | BPE | Unigram LM |
|---|---|---|
| Direction | Bottom-up (merge pairs) | Top-down (prune tokens) |
| Pre-tokenize | Yes (split on spaces) | No (raw Unicode + ▁) |
| Word boundary | `Ġ` (GPT-2) | `▁` (SentencePiece) |
| Decoding | Greedy left-to-right merge | Viterbi DP (optimal) |
| Probabilities | None (rule-based) | Yes (log-probs, EM-estimated) |
| Train speed | Fast | Slow (EM iterations) |
| Encode speed | Fast (greedy) | Slow (Viterbi) |
| Quality | Good | Slightly better |
| Used by | GPT-2/3/4, LLaMA | T5, ALBERT, XLNet, mBERT |

---

## Data Loaders

Bridge between tokenizer and model. Convert tokenized stream → batches of `(input, target)` pairs for training.

For GPT-style next-token prediction:
- `input  = ids[i..i+ctx]`
- `target = ids[i+1..i+ctx+1]`  (shifted by 1)

Components to build:
- `GptDataset` — sliding windows over flat token stream
- `GptDataLoader` — batched iterator, optional shuffle
- Train/val split

Knobs: `context_len`, `stride`, `batch_size`, `shuffle`, `drop_last`.

## Roadmap

- [ ] Data loaders (next)
- [ ] Pick tensor library (`ndarray` / `candle` / hand-roll)
- [ ] Self-attention (Ch 3)
- [ ] Multi-head attention (Ch 3)
- [ ] Positional encoding
- [ ] Transformer block (attention + FFN + LayerNorm + residual)
- [ ] Full GPT-2 architecture
- [ ] Training loop (loss + backprop + optimizer)
- [ ] Generation: greedy, top-k, top-p, temperature
- [ ] Subword regularization in SentencePiece (n-best path sampling)

## Learning Progression

Project follows *LLMs from Scratch*. Full progression after tokenizers:

1. **Tokenizer** ✓ (BPE + Unigram done)
2. **Embedding layer** — token ids → dense vectors
3. **Positional encoding** — inject position info
4. **Self-attention** — core transformer primitive
5. **Multi-head attention** — parallel attention heads
6. **Transformer block** — attention + feedforward + norm + residual
7. **Full GPT model** — stack of transformer blocks + output head
8. **Training loop** — loss + backprop + optimizer
9. **Generation/sampling** — greedy, top-k, top-p, temperature

### Recommended next: Self-attention

Skip embeddings — they're trivial (just `vocab_size × dim` lookup table). Attention has the most educational value with smallest code.

**Why attention next:**
- Core primitive everything else builds on
- Self-contained — just matrix math
- Without it, embeddings/positions are meaningless
- Single biggest "aha moment" in deep learning

### Rust tensor library options

No `numpy` / `torch` in std. Pick one:

| Option | Pros | Cons |
|--------|------|------|
| `ndarray` | numpy-like API, pure Rust | No autograd, no GPU |
| `candle` | Has autograd + GPU (HuggingFace) | Bigger dep, more abstraction |
| Hand-roll | Most educational, see every op | Slow, no GPU, lots of code |

Recommendation: `ndarray` for learning attention/transformer math, switch to `candle` if you want real training (autograd does backprop for you).

## Dependencies

- `serde` + `serde_json` — model save/load
- `regex-lite` — minimal regex (BPE special token matching)
- `reqwest` — download training corpus + GPT-2 vocab files

## License

MIT — educational reference code, not production-ready. Credit to Sebastian Raschka's *LLMs from Scratch* (Manning, 2024) for the structure and algorithms.
