# llm_scratch_rs

Building an LLM from scratch in Rust for learning. The repo uses
Sebastian Raschka's [`LLMs-from-scratch`](https://github.com/rasbt/LLMs-from-scratch)
as the main build-order reference and the
[Language AI Handbook](https://mbrenndoerfer.com/books/language-ai-handbook) as a
concept reference.

The code is organized by concepts, not book chapters. The local layout should
remain understandable without needing to remember chapter numbers.

## Index

- [Status](#status)
- [Concept Layout](#concept-layout)
- [Run](#run)
- [Quick Start API](#quick-start-api)
- [Tokenizer Trait](#tokenizer-trait)
- [BPE](#bpe--byte-pair-encoding)
- [SentencePiece](#sentencepiece--unigram-language-model)
- [BPE vs Unigram](#bpe-vs-unigram)
- [Current Model Stack](#current-model-stack)
- [Roadmap](#roadmap)
- [Dependencies](#dependencies)

## Status

Maps to chapters of _LLMs from Scratch_ (Sebastian Raschka):

| Topic                         | Book Ch.       | Status      |
| ----------------------------- | -------------- | ----------- |
| Tokenizer trait               | infra          | Done        |
| Byte Pair Encoding (BPE)      | Ch 2           | Done        |
| SentencePiece (Unigram LM)    | Ch 2 extension | Done        |
| Data loaders                  | Ch 2           | Next        |
| Token + positional embeddings | Ch 2 / Ch 4    | Done        |
| Self-attention                | Ch 3           | Done        |
| Multi-head attention          | Ch 3           | Done        |
| Feed-forward network          | Ch 4           | Done        |
| LayerNorm                     | Ch 4           | Done        |
| Transformer block             | Ch 4           | Done        |
| Full GPT model                | Ch 4           | Next        |
| Training loop                 | Ch 5           | Not started |
| Sampling / generation         | Ch 5           | Not started |
| Fine-tuning                   | Ch 6+7         | Not started |

**Requires:** Rust 1.70+ (edition 2021).

## Concept Layout

```text
src/
  tokenizers/
    tokenizer.rs          shared Tokenizer trait
    bpe.rs                Byte Pair Encoding tokenizer
    sentence_piece.rs     SentencePiece-style unigram tokenizer

  attention/
    self_attention.rs     causal scaled dot-product self-attention
    multi_head_attention.rs

  layers/
    embedding.rs          token + positional embeddings
    feed_forward.rs       position-wise feed-forward network
    layer_norm.rs         layer normalization

  models/
    transformer.rs        transformer block

  common/
    util.rs               shared vector/matrix helpers

  bin/
    *.rs                  runnable learning demos
```

## Run

```bash
cargo check
cargo run --bin bpe
cargo run --bin sentence_piece
cargo run --bin embedding
cargo run --bin attention
cargo run --bin multi_head_attention
cargo run --bin feed_forward
cargo run --bin layer_norm
cargo run --bin transformer
```

## Quick Start API

```rust
use llm_scratch_rs::tokenizers::sentence_piece::{SeedMethod, SentencePieceTokenizer};
use llm_scratch_rs::tokenizers::tokenizer::Tokenizer;

let text = std::fs::read_to_string("./the-verdict.txt")?;
let mut tok = SentencePieceTokenizer::new();
tok.train(&text, 500, SeedMethod::Bpe, None);

let ids = tok.encode("hello world")?;
let back = tok.decode(&ids)?;
assert_eq!(back, "hello world");

tok.save("./sp_model.json")?;
```

BPE uses the same `Tokenizer` trait, so callers can swap tokenizer
implementations without changing the encode/decode interface.

## Tokenizer Trait

```rust
pub trait Tokenizer {
    fn encode(&mut self, text: &str) -> Result<Vec<usize>, String>;
    fn decode(&self, ids: &[usize]) -> Result<String, String>;
}
```

## BPE - Byte Pair Encoding

Reference: [Raschka's BPE from scratch article](https://sebastianraschka.com/blog/2025/bpe-from-scratch.html)

Algorithm:

1. Start with base vocabulary entries.
2. Pre-tokenize text and preserve space information.
3. Count adjacent token pairs.
4. Merge the most frequent pair into a new token.
5. Repeat until the target vocabulary size is reached.

The implementation supports custom-trained merge rules and loading GPT-2-style
vocabulary/merge files.

## SentencePiece - Unigram Language Model

References:

- [Unigram Language Model Tokenization](https://mbrenndoerfer.com/writing/unigram-language-model-tokenization)
- [SentencePiece subword tokenization](https://mbrenndoerfer.com/writing/sentencepiece-subword-tokenization-bpe-unigram)

Algorithm:

1. Build a large seed vocabulary.
2. Use Viterbi dynamic programming to segment words by maximum probability.
3. Re-estimate token probabilities from segment counts.
4. Prune lower-value tokens.
5. Repeat until the vocabulary reaches the target size.

The implementation uses `▁` as the word-boundary marker and `<unk>` for unknown
pieces.

## BPE vs Unigram

|                | BPE                      | Unigram                        |
| -------------- | ------------------------ | ------------------------------ |
| Direction      | Bottom-up merge pairs    | Top-down prune pieces          |
| Main operation | frequent-pair merge      | probabilistic segmentation     |
| Encoding       | greedy merge application | Viterbi search                 |
| Probabilities  | no token probabilities   | token log-probabilities        |
| Common usage   | GPT-style tokenizers     | SentencePiece-style tokenizers |

## Current Model Stack

Learning order in this repo:

1. Tokenization: `src/tokenizers/*`
2. Embeddings: `src/layers/embedding.rs`
3. Self-attention: `src/attention/self_attention.rs`
4. Multi-head attention: `src/attention/multi_head_attention.rs`
5. Feed-forward + LayerNorm: `src/layers/feed_forward.rs`, `src/layers/layer_norm.rs`
6. Transformer block: `src/models/transformer.rs`
7. GPT model wrapper: planned `src/models/gpt.rs`
8. Training loop
9. Generation / sampling

The current transformer demo checks that one block preserves shape:

```text
[seq_len][d_model] -> [seq_len][d_model]
```

## Roadmap

- [x] Tokenizers
- [x] Embeddings
- [x] Causal self-attention
- [x] Multi-head attention
- [x] Feed-forward network
- [x] LayerNorm
- [x] Transformer block
- [ ] GPT model wrapper
- [ ] Data loader for next-token prediction
- [ ] Loss and optimizer
- [ ] Tiny training loop
- [ ] Generation and sampling

## TODO

- [ ] 1. Rebuild your mini GPT in Candle
- [ ]2. Add Candle autograd training
- [ ] 3. Add Candle generation with top-k/top-p
- [ ] 4. Add KV cache
- [ ] 5. Load GPT-2 or small HF model with Candle
- [ ]6. Learn safetensors
- [ ] 7. Learn quantization
- [ ]8. Learn LoRA / QLoRA
- [ ] 9. Build Axum inference API
- [ ]10. Try Burn for training abstraction
- [ ] 11. Learn PyTorch/HF for industry workflows

## Next Step

Add:

```text
src/models/gpt.rs
src/bin/gpt.rs
```

The first GPT smoke test should prove:

```text
token ids -> embeddings -> transformer block(s) -> logits [seq_len][vocab_size]
```

## Dependencies

- `serde` + `serde_json`: save/load tokenizer artifacts
- `regex-lite`: lightweight tokenizer pattern matching
- `reqwest`: corpus and tokenizer asset downloads
- `rand`: simple random initialization for learning demos
