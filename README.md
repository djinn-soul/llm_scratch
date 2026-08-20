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
- [BPE](#bpe---byte-pair-encoding)
- [SentencePiece](#sentencepiece---unigram-language-model)
- [BPE vs Unigram](#bpe-vs-unigram)
- [Current Model Stack](#current-model-stack)
- [Diffusion Samplers & Resampling CLI](#diffusion-samplers--resampling-cli)
- [Roadmap](#roadmap)
- [TODO](#todo)
- [Next Step](#next-step)
- [Dependencies](#dependencies)

## Status

Maps to chapters of _LLMs from Scratch_ (Sebastian Raschka):

| Topic                         | Book Ch.       | Status      |
| ----------------------------- | -------------- | ----------- |
| Tokenizer trait               | infra          | Done        |
| Byte Pair Encoding (BPE)      | Ch 2           | Done        |
| SentencePiece (Unigram LM)    | Ch 2 extension | Done        |
| Data loaders                  | Ch 2           | Done        |
| Token + positional embeddings | Ch 2 / Ch 4    | Done        |
| Self-attention                | Ch 3           | Done        |
| Multi-head attention          | Ch 3           | Done        |
| Feed-forward network          | Ch 4           | Done        |
| LayerNorm                     | Ch 4           | Done        |
| Transformer block             | Ch 4           | Done        |
| Full GPT model                | Ch 4           | Done        |
| Training loop                 | Ch 5           | Done        |
| Sampling / generation         | Ch 5           | Done        |
| Fine-tuning                   | Ch 6+7         | Next        |

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

## Diffusion Samplers & Resampling CLI

The codebase supports plug-and-play reverse diffusion inference on trained U-Net checkpoints (`ema_epoch_25000.safetensors`) with zero retraining:

| Sampler | Order | Default Steps | Pass / NFE | Key Mechanism | Best For |
| :--- | :---: | :---: | :---: | :--- | :--- |
| **DPM-Solver++ (2M)** | **2nd-Order ODE** | **8–10 steps** | **$1\times$ NFE** | **Exact exponential integration** in log-SNR ($\lambda$) space + multistep $\hat{x}_0$ polynomial extrapolation | **Fastest inference & sharpest details** |
| **DDIM** | 1st-Order ODE | 20–50 steps | $1\times$ NFE | Deterministic Euler jumps with $\hat{x}_0$ recovery | Balanced baseline |
| **DDPM** | 1st-Order SDE | 100 steps | $1\times$ NFE | Stochastic reverse Markov transitions with per-step noise | Training validation & ground-truth |

### Plug-and-Play Resampling CLI

Run the modular resampler with customized samplers, steps, or side-by-side benchmark comparison:

```bash
# 1. Default ultra-fast DPM-Solver++ (8 steps)
cargo run --release --bin resample_diffusion_unet

# 2. Explicit DPM-Solver++ (10 steps, Guidance = 3.0, Class = 7)
cargo run --release --bin resample_diffusion_unet -- --sampler dpm --steps 10 --guidance 3.0 --class 7

# 3. Standard DDIM (20 steps)
cargo run --release --bin resample_diffusion_unet -- --sampler ddim --steps 20

# 4. Stochastic DDPM (100 steps)
cargo run --release --bin resample_diffusion_unet -- --sampler ddpm --steps 100

# 5. Comparative Benchmark (Runs DDPM, DDIM, and DPM-Solver++ and generates a 3-row comparison grid)
cargo run --release --bin resample_diffusion_unet -- --sampler all
```

### Diffusion Architectures: Standard vs. AdaGN U-Net

The repository implements two high-performance U-Net architectures:

1. **`SimpleDenoisingUNet` (Spatial Concatenation):**
   * Projects conditioning vector $c = [t_{\text{emb}}, \text{class}]$ into a $28\times 28$ spatial feature map.
   * Concatenates with the input image (`[B, 2, 28, 28]`).
   * Conv1 takes 2 channels: `w1: [16, 2, 3, 3]`.
2. **`SimpleDenoisingUNetAdaGN` (Adaptive Group Normalization):**
   * **Direct 1-Channel Input:** `w1: [16, 1, 3, 3]` takes raw grayscale image directly without spatial broadcasting.
   * **Per-Layer Adaptive Modulation:** Injects conditioning dynamically into every GroupNorm layer via learned scale $\gamma(c)$ and shift $\beta(c)$:
     $$\text{AdaGN}(x, c) = \text{GroupNorm}(x) \odot (1 + \gamma(c)) + \beta(c)$$
   * **Zero-Initialized Identity Start:** Projection weights initialize at $0$ so the network starts training at pure standard GroupNorm without gradient explosion.
   * **Analytical Backpropagation:** Hand-written analytical gradients verified against central-difference numerical differentiation (`tests/unet_gradient_check.rs`).

## Roadmap

- [x] Tokenizers
- [x] Embeddings
- [x] Causal self-attention
- [x] Multi-head attention
- [x] Feed-forward network
- [x] LayerNorm
- [x] Transformer block
- [x] GPT model wrapper
- [x] Data loader for next-token prediction
- [x] Loss and optimizer
- [x] Tiny training loop
- [x] Generation and sampling

## TODO

### 1. Candle & PyTorch/HF Foundations (Modern Frameworks)
- [ ] 1. Rebuild your mini GPT in Candle
- [ ] 2. Add Candle autograd training
- [ ] 3. Add Candle generation with top-k/top-p
- [ ] 4. Add KV cache
- [ ] 5. Load GPT-2 or small HF model with Candle
- [ ] 6. Learn safetensors
- [ ] 7. Learn quantization
- [ ] 8. Try Burn for training abstraction
- [ ] 9. Build Axum inference API
- [ ] 10. Learn PyTorch/HF for industry workflows

### 2. Model Architectures & Core Implementations
- [x] 11. Attention is all you need (Transformers)
    - [x] 11.1 Implement causal scaled dot-product self-attention
    - [x] 11.2 Implement Multi-Head Attention (MHA)
    - [x] 11.3 Implement position-wise feed-forward networks (FFN)
    - [x] 11.4 Implement Layer Normalization (LayerNorm)
    - [x] 11.5 Assemble Transformer Block decoder with residual connections
    - [x] 11.6 Wire into full GPT model architecture with weight tying
- [ ] 12. BERT (Bidirectional Encoder Representation from Transformers)
    - [ ] 12.1 Implement Masked Language Modeling (MLM) task
    - [ ] 12.2 Implement Next Sentence Prediction (NSP) task
    - [ ] 12.3 Implement bidirectional attention masking (no causal mask)
    - [ ] 12.4 Load pretrained BERT weights/tokenize with WordPiece
    - [ ] 12.5 Fine-tune BERT on text classification tasks
- [ ] 13. LLaMA & DeepSeek Architectures
    - [ ] 13.1 Implement Rotary Position Embeddings (RoPE)
    - [ ] 13.2 Implement SwiGLU activation function
    - [ ] 13.3 Implement RMSNorm (Root Mean Square Normalization)
    - [ ] 13.4 Implement Grouped-Query Attention (GQA) and Multi-Query Attention (MQA)
    - [ ] 13.5 Load pretrained LLaMA-style model weights (e.g., TinyLLaMA) in Rust
    - [ ] 13.6 Implement Low-Rank KV Compression (DeepSeek Multi-Head Latent Attention / MLA)
- [ ] 14. MoE (Mixture of Experts)
    - [ ] 14.1 Implement router/gating network (top-k routing)
    - [ ] 14.2 Implement multiple feed-forward expert layers
    - [ ] 14.3 Add auxiliary load balancing loss to prevent routing collapse
    - [ ] 14.4 Verify routing path and expert utilization
    - [ ] 14.5 Implement auxiliary-loss-free load balancing (DeepSeek-style routing)
- [ ] 14b. Mamba & State Space Models (SSMs)
    - [ ] 14b.1 Implement selective scan algorithm
    - [ ] 14b.2 Implement Mamba block architecture in Candle
    - [ ] 14b.3 Compare generation efficiency against standard Transformers

### 3. Fine-Tuning, PEFT, & Alignment
- [ ] 15. Fine-tuning for classification
    - [ ] 15.1 Different categories of fine-tuning
    - [ ] 15.2 Preparing the dataset
    - [ ] 15.3 Creating data loaders
    - [ ] 15.4 Initializing a model with pretrained weights
    - [ ] 15.5 Adding a classification head
    - [ ] 15.6 Calculating the classification loss and accuracy
    - [ ] 15.7 Fine-tuning the model on supervised data
    - [ ] 15.8 Using the LLM as a spam classifier
- [ ] 16. Fine-tuning to follow instructions
    - [ ] 16.1 Introduction to instruction fine-tuning
    - [ ] 16.2 Preparing a dataset for supervised instruction fine-tuning
    - [ ] 16.3 Organizing data into training batches
        - [ ] Why replacement by -100
    - [ ] 16.4 Creating data loaders for an instruction dataset
    - [ ] 16.5 Loading a pretrained LLM
    - [ ] 16.6 Fine-tuning the LLM on instruction data
    - [ ] 16.7 Extracting and saving responses
    - [ ] 16.8 Evaluating the fine-tuned LLM
    - [ ] 16.9 Conclusions
    - [ ] 16.10 Summary
- [ ] 17. LoRA (Low rank adaption)
    - [ ] 17.1 Implement low-rank matrices A and B
    - [ ] 17.2 Add scaling factor alpha / r
    - [ ] 17.3 Create a LoRA linear layer wrapper/helper
    - [ ] 17.4 Freeze pretrained base weights
    - [ ] 17.5 Integrate LoRA weights into forward pass and autograd
- [ ] 18. PEFT (Parameter Efficient Fine Tuning)
    - [ ] 18.1 Explore alternative PEFT methods (Prefix Tuning, Prompt Tuning)
    - [ ] 18.2 Implement unified PEFT adapter manager
    - [ ] 18.3 Verify parameter count efficiency (trainable vs. frozen)
- [ ] 19. RLHF (Reinforcement Learning from Human Feedback)
    - [ ] 19.1 Train/Load a Reward Model (RM) from preference data
    - [ ] 19.2 Implement PPO (Proximal Policy Optimization) reinforcement learning loop
    - [ ] 19.3 Implement KL divergence penalty against reference SFT model
    - [ ] 19.4 Compare generation quality before and after RLHF alignment
- [ ] 20. DPO & Direct Preference Variants
    - [ ] 20.1 Prepare a preference dataset (chosen vs. rejected responses)
    - [ ] 20.2 Implement the DPO loss function from scratch
    - [ ] 20.3 Fine-tune the policy model relative to a reference model
    - [ ] 20.4 Implement Kahneman-Tversky Optimization (KTO) for binary preference signals
    - [ ] 20.5 Implement Odds Ratio Preference Optimization (ORPO) without reference models

### 4. Reasoning & System 2 Thinking
- [ ] 21. Reasoning Models (Chain of Thought & RL/Search)
    - [ ] 21.1 Implement training dataset parsing with thinking/reasoning tags
    - [ ] 21.2 Implement search-based decoding (e.g., MCTS or Beam Search over reasoning steps)
    - [ ] 21.3 Implement Process-supervised Reward Model (PRM)/value network scoring
    - [ ] 21.4 Implement Group Relative Policy Optimization (GRPO) or similar RL loop for reasoning stability

### 5. Efficiency & Optimization
- [ ] 22. FlashAttention & Memory-Efficient Decoding
    - [ ] 22.1 Implement online softmax tiling for forward attention pass
    - [ ] 22.2 Avoid materializing the full N x N matrix in memory
    - [ ] 22.3 Compare memory usage and speed against standard attention
    - [ ] 22.4 Implement PagedAttention (vLLM-style paging) to optimize decoding memory
    - [ ] 22.5 Implement INT4 / FP4 KV-cache Quantization to reduce inference footprint
- [ ] 23. Speculative Decoding & Constrained Generation
    - [ ] 23.1 Implement draft model and target model generation loops
    - [ ] 23.2 Implement speculative draft verification logic
    - [ ] 23.3 Benchmark generation speedup ratios
    - [ ] 23.4 Implement Structured Generation / Constrained decoding (regular expression and JSON schema logit masking)
- [ ] 24. Custom Quantization & Context Window Expansion
    - [ ] 24.1 Implement Round-to-Nearest (RTN) weight quantization
    - [ ] 24.2 Create quantized linear layer forward implementations (e.g., 4-bit / 8-bit)
    - [ ] 24.3 Compare accuracy loss and memory usage profiles
    - [ ] 24.4 Implement Context Window Expansion (YaRN, NTK-aware RoPE Scaling)

### 6. Vision, Generative & Other Domains
- [ ] 25. VIT (Vision Transformers) & Multimodal Models
    - [ ] 25.1 Implement patch projection (convert 2D images to 1D patch embeddings)
    - [ ] 25.2 Add CLS class token and positional embeddings
    - [ ] 25.3 Implement ViT encoder blocks with self-attention
    - [ ] 25.4 Create classification head for vision tasks
    - [ ] 25.5 Train and evaluate on toy dataset (e.g., MNIST/CIFAR-10)
    - [ ] 25.6 Implement Multimodal projection layer (linking patch embeddings to text decoder input space)
    - [ ] 25.7 Implement a basic Vision-Language Model (VLM) causal forward pass
- [ ] 26. VAE (Variational Auto Encoder)
    - [ ] 26.1 Implement Encoder network (outputting mean and log variance)
    - [ ] 26.2 Implement Reparameterization trick (sampling epsilon from N(0, I))
    - [ ] 26.3 Implement Decoder network (reconstruct input from latent space)
    - [ ] 26.4 Implement loss function (Reconstruction Loss + KL Divergence)
    - [ ] 26.5 Train and generate synthetic images
- [ ] 27. GANs (Generative Adversarial Networks)
    - [ ] 27.1 Implement Generator network
    - [ ] 27.2 Implement Discriminator network
    - [ ] 27.3 Implement adversarial minimax training loop
    - [ ] 27.4 Implement training stabilizers (e.g., Wasserstein GAN, Gradient Penalty)
    - [ ] 27.5 Generate images and plot loss/discriminator accuracy
- [/] 28. Diffusion Models (Stable Diffusion)
    - [x] 28.1 Implement forward diffusion process (noise scheduler)
    - [x] 28.2 Implement reverse diffusion denoising process
    - [x] 28.3 Implement MLP denoiser (baseline)
    - [x] 28.4 Implement CNN denoiser (2-layer, 3×3 kernels)
    - [x] 28.5 Implement CNN denoiser (5-layer, 5×5 kernels)
    - [x] 28.6 Train DDPM on MNIST (28×28 image generation)
    - [x] 28.7 Integrate classifier-free guidance (CFG)
    - [x] 28.8 Implement DDIM sampler (deterministic reverse diffusion, fewer steps)
    - [x] 28.9 Implement cosine noise schedule (Nichol & Dhariwal, 2021)
    - [x] 28.10 Implement U-Net architecture (encoder-decoder + skip connections)
    - [x] 28.11 Add attention layers in U-Net bottleneck
    - [x] 28.11b Implement DPM-Solver++ (2M) 2nd-order exponential ODE sampler (8–10 steps)
    - [x] 28.11c Implement Adaptive Group Normalization (AdaGN) U-Net architecture (per-layer modulation & direct 1-ch input)
    - [x] 28.12 Implement DiT (Diffusion Transformer) denoiser
    - [ ] 28.13 Implement latent diffusion (VAE encoder → diffuse in latent space → decode)
- [ ] 28b. RAG & Vector Databases
    - [ ] 28b.1 Implement similarity search functions (Cosine, Dot Product)
    - [ ] 28b.2 Implement a basic HNSW (Hierarchical Navigable Small World) index builder
    - [ ] 28b.3 Implement Dense Passage Retrieval (DPR) bi-encoder pipeline



### 7. Testing & Validation
- [ ] 29. Unit test coverage for core math and data flow
    - [ ] 29.1 Test stable softmax and cross-entropy against known values
    - [ ] 29.2 Test DataLoader input/target window alignment
    - [ ] 29.3 Test tokenizer encode/decode round-trips
    - [ ] 29.4 Test sampling behavior for greedy, top-k, and top-p
- [ ] 30. Model and training smoke tests
    - [ ] 30.1 Test GPT forward output shape `[seq_len][vocab_size]`
    - [ ] 30.2 Test backward pass creates gradients for trainable weights
    - [ ] 30.3 Test one optimizer step updates parameters
    - [ ] 30.4 Test JSON and binary weight save/load round-trips
- [ ] 31. Candle/GPT-2 integration validation
    - [ ] 31.1 Test GPT-2 config and tokenizer loading
    - [ ] 31.2 Test safetensors tensor-name lookup
    - [ ] 31.3 Test one GPT-2 forward pass with loaded weights
    - [ ] 31.4 Test generation smoke output from `gpt2_candle`

### 8. Agentic Systems & Tool Use
- [ ] 32. Tool Calling & ReAct Agent
    - [ ] 32.1 Implement function signature formatting and system prompt builder
    - [ ] 32.2 Implement JSON/XML tool execution result injection
    - [ ] 32.3 Build a standalone ReAct (Reason-Action-Observation) agent execution loop

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
