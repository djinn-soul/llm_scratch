# Attention Mechanisms — Complete Reference

> Reference doc. Not part of the scratch build. Full map of the attention
> landscape as of 2026, plus a deep dive on DeepSeek-V4 internals.

**Contents**
- [Part 1 — The 5 axes of attention](#part-1--the-5-axes-of-attention)
- [Part 2 — What top models use (2026)](#part-2--what-top-models-use-2026)
- [Part 3 — Where this repo sits](#part-3--where-this-repo-sits)
- [Part 4 — Deep dive: DeepSeek-V4 hybrid attention](#part-4--deep-dive-deepseek-v4-hybrid-attention)
- [Sources](#sources)

Attention variants split along **five independent axes**. A real model picks
one option from each axis. They are not mutually exclusive — e.g. Llama 3 =
GQA + causal + RoPE + Flash kernel.

---

## Part 1 — The 5 axes of attention

### Axis 1 — Scope (which tokens can a token see?)

| Type | Each token attends to | Complexity | Notes |
|------|----------------------|------------|-------|
| **Dense / Full** | every other token | O(L²) | richest, original Transformer, the wall |
| **Causal / Masked** | only past + self | O(L²) | required for autoregressive GPT decoders |
| **Sliding Window (SWA)** | fixed local window | O(L·w) | linear in length, local detail only |
| **Local + Global hybrid** | local window + few global tokens | O(L·w) | Longformer, BigBird |
| **Block-sparse** | sliding + global + random blocks | O(L·√L) | BigBird |
| **Sparse (top-k)** | k most relevant, selected | O(L·k) | DeepSeek DSA — needs a selector network |

### Axis 2 — KV sharing (how many key/value heads?)

| Type | KV heads | KV cache size | Quality | Used in |
|------|----------|---------------|---------|---------|
| **MHA** (Multi-Head) | N (one per query head) | largest | best | GPT-2, original Transformer — **this repo** |
| **MQA** (Multi-Query) | 1 shared | smallest | some loss | PaLM, Falcon |
| **GQA** (Grouped-Query) | G groups share K,V | medium | near-MHA | **de facto standard** — Llama 3, Mistral, Qwen |
| **MLA** (Multi-head Latent) | low-rank latent compression | small | near-MHA | DeepSeek-V2, V3 |
| **Hybrid CSA + HCA** | distance-tiered compression | tiny | near-MHA | DeepSeek-V4 |

### Axis 3 — Position encoding (how does attention know token order?)

| Type | How | Extrapolates? | Used in |
|------|-----|---------------|---------|
| **Absolute (learned)** | add position vector to input | poorly | GPT-2 — **this repo** (`PositionalEmbedding`) |
| **Sinusoidal** | fixed sin/cos pattern added to input | somewhat | original Transformer |
| **RoPE** (Rotary) | rotate Q,K by angle proportional to position | well | Llama, Mistral, Qwen, Gemma |
| **ALiBi** | add linear distance penalty to scores | very well | BLOOM, MPT |
| **NoPE** | no explicit position — causal mask alone | surprisingly OK | some research models |

Gemma 3 trick: different RoPE base freq per layer — 10K local, 1M global.

### Axis 4 — Compute kernel (same math, different execution)

| Type | Idea | Benefit | Used in |
|------|------|---------|---------|
| **Naive** | plain matmul + softmax loops | clear, slow | **this repo** |
| **Flash Attention** | fuse softmax+matmul, exploit SRAM/HBM hierarchy | big speed + memory win, exact | Llama 3, Mistral, Claude — default everywhere |
| **Paged Attention** | OS-style memory paging for KV cache | high-throughput serving | vLLM |
| **Linear Attention** | kernel trick — avoid all-pairs comparison | O(L) not O(L²) | approximation, not exact |

### Axis 5 — Source of Q vs K,V

| Type | Query from | Key/Value from | Used in |
|------|-----------|----------------|---------|
| **Self-attention** | sequence X | same sequence X | GPT, BERT — **this repo** |
| **Cross-attention** | decoder sequence | encoder sequence | T5, encoder-decoder, image captioning |

---

## Part 2 — What top models use (2026)

| Model | KV sharing | Scope | Position | Kernel |
|-------|-----------|-------|----------|--------|
| GPT-2 | MHA | causal | absolute learned | naive (era) |
| Llama 3 | GQA | causal | RoPE | Flash |
| Mistral / Mixtral | GQA | sliding window | RoPE | Flash |
| Gemma 3 | GQA | local+global mix | RoPE (per-layer base) | Flash |
| DeepSeek-V3 | MLA | dense | RoPE | Flash |
| DeepSeek-V3.2 | MLA | sparse (DSA top-k) | RoPE | Flash |
| DeepSeek-V4 | Hybrid CSA+HCA | sparse + sliding window | RoPE | Flash + FP4 |
| GLM-5 | — | DeepSeek Sparse Attention | RoPE | Flash |
| Qwen 3.5 | Gated DeltaNet + MoE (not classic attention) | — | — | — |

Trend: **GQA + RoPE + Flash** = the safe modern default. DeepSeek line pushes
sparsity/compression hardest. Qwen 3.5 experiments with leaving classic
attention behind entirely (DeltaNet — linear-attention family).

---

## Part 3 — Where this repo sits

| Axis | This repo | Modern default | Gap |
|------|-----------|----------------|-----|
| Scope | dense → **causal next** | causal (+ sparse at scale) | add causal mask |
| KV sharing | MHA | GQA | fine for learning, skip GQA |
| Position | absolute learned | RoPE | fine for learning, RoPE optional later |
| Kernel | naive | Flash | naive is the point — clarity over speed |
| Q vs K,V | self-attention | self-attention | already correct |

**Next concrete step:** add the **causal mask** to `multi_head_attention.rs` —
it's the one item from this whole map that a GPT-from-scratch genuinely needs.
Everything else (GQA, RoPE, Flash, sparse) is scale/efficiency, not core math.

---

## Part 4 — Deep dive: DeepSeek-V4 hybrid attention

> Context for where attention research is in 2026, after the
> MHA → MQA → GQA → MLA line. Published April 27, 2026.

### 4.1 — Evolution of attention, the full line

| Gen | Mechanism | KV strategy | Complexity | Used in |
|-----|-----------|-------------|------------|---------|
| 1 | **MHA** (Multi-Head) | N full K,V — one per query head | O(L²) | GPT-2, original Transformer |
| 2 | **MQA** (Multi-Query) | 1 shared K,V for all query heads | O(L²) | PaLM, Falcon |
| 3 | **GQA** (Grouped-Query) | G groups, each group shares K,V | O(L²) | Llama 3, Mistral, Qwen |
| 4 | **MLA** (Multi-head Latent) | compress K,V to low-rank latent | O(L²) | DeepSeek-V2, V3 |
| 5 | **DSA** (DeepSeek Sparse) | top-k token selection via indexer | O(L·k) | DeepSeek-V3.2 |
| 6 | **Hybrid CSA + HCA** | distance-tiered compression + sparsity | O(L·k) | **DeepSeek-V4** |

Each generation chases the same bottleneck: **decoding is memory-bandwidth-bound**,
not compute-bound. KV cache size is the wall.

### 4.2 — The two mechanisms

V4 interleaves **two** attention types across transformer layers. Not one scheme — two, alternating.

| | **CSA** (Compressed Sparse Attention) | **HCA** (Heavily Compressed Attention) |
|---|---|---|
| Descendant of | V3.2's DSA | new in V4 |
| Compression stride | every **4** tokens → 1 entry | every **128** tokens → 1 entry |
| Compressor | learned token-level compressor | learned, aggressive consolidation |
| After compression | DSA — top-k sparse over compressed entries | dense attention over compressed entries |
| Selection | Lightning Indexer picks ~128 top-k blocks | none — dense over the few entries left |
| Role | near/medium context, keeps detail | far context, keeps only gist |

Both also run a **sliding-window branch** for the most recent `n_win = 128` tokens
(exact, uncompressed — local detail never lost).

### 4.3 — Compressed Sparse Attention (CSA) pipeline

| Step | What happens |
|------|--------------|
| 1. Compress | learned compressor, stride 4 — every 4 KV tokens → 1 compressed entry |
| 2. Index | Lightning Indexer scores query against all compressed KV blocks |
| 3. Select | top-k (~128) compressed entries chosen per query |
| 4. Attend | expensive softmax + matmul runs ONLY on selected entries |
| 5. Local | parallel sliding-window branch covers recent 128 tokens, exact |

**Lightning Indexer** — the critical piece:

| Property | Detail |
|----------|--------|
| Type | lightweight scoring network |
| Precision | FP4 (very cheap) |
| Job | score every compressed KV block for a given query |
| Output | top-k block selection for sparse attention |
| Origin | evolved from V3.2 DSA indexer (was FP8, single-kernel batched matmul) |

### 4.4 — Heavily Compressed Attention (HCA) pipeline

| Step | What happens |
|------|--------------|
| 1. Compress | learned compressor, stride 128 — every 128 KV tokens → 1 entry |
| 2. Attend | dense attention over the (now very few) compressed entries |
| 3. Local | same sliding-window branch, recent 128 tokens, exact |

No indexer, no top-k. After 128:1 compression there are so few entries that
dense attention over them is already cheap.

### 4.5 — Layer interleaving, two model variants

| Variant | First 2 layers | Remaining layers |
|---------|----------------|------------------|
| **V4-Flash** | pure sliding-window attention | alternate CSA / HCA |
| **V4-Pro** | HCA | alternate CSA / HCA |

Early layers handle local structure; deeper layers alternate the two
compression regimes so the model mixes fine and coarse context views.

### 4.6 — The core idea: distance-tiered compression

```
recent tokens     → sliding window, EXACT, uncompressed   (need full detail)
medium distance   → CSA, 4:1 compress + sparse top-k       (some detail)
far distance      → HCA, 128:1 compress + dense            (just the gist)
```

Plus **saliency-based filtering**: before compressing, a saliency function
scores how much each token contributes to its compressed entry. High-saliency
tokens get more weight — compression is weighted, not blind averaging.

### 4.7 — Efficiency gains

| Metric | DeepSeek-V4-Pro vs V3.2 (1M-token context) |
|--------|--------------------------------------------|
| Per-token inference FLOPs | **27%** of V3.2 (≈73% reduction) |
| KV cache memory | **10%** of V3.2 (≈90% reduction) |
| Attention complexity | O(L²) → O(L·k), k ≪ L |
| Context length | 1,000,000 tokens, usable by agents |

Also ships with FP4 quantization + QAT (quantization-aware training) stability work —
orthogonal to attention but part of the same efficiency push.

### 4.8 — Relevance to this project

| Concept | In `llm_scratch_rs`? | Why / why not |
|---------|----------------------|---------------|
| MHA | ✅ `multi_head_attention.rs` | core mechanism, must understand |
| Causal mask | ⬜ next step | required for GPT decoder — can't skip |
| GQA / MLA | ❌ | inference optimization, not learning the math |
| CSA / HCA / DSA | ❌ | production-scale memory tricks, irrelevant to scratch build |
| Lightning Indexer | ❌ | same — FP4 inference engineering |

**Takeaway for the build:** stay on plain **MHA + causal mask**. Everything from
GQA onward is solving a problem (KV cache at 1M-token context) that a learning
project never hits. Know it exists; don't implement it.

---

## Sources

- [Efficient Attention Mechanisms for LLMs: A Survey (arXiv 2507.19595)](https://arxiv.org/abs/2507.19595)
- [The Big LLM Architecture Comparison — Sebastian Raschka](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison)
- [A Visual Guide to Attention Variants in Modern LLMs — Sebastian Raschka](https://magazine.sebastianraschka.com/p/visual-attention-variants)
- [A Technical Tour of the DeepSeek Models from V3 to V3.2 — Sebastian Raschka](https://magazine.sebastianraschka.com/p/technical-deepseek)
- [Taxonomy of Attention Mechanisms — ML Digest](https://ml-digest.com/taxonomy-of-attention-mechanisms/)
- [13+ Attention Mechanisms You Should Know — Turing Post](https://www.turingpost.com/p/attention-types)
- [Attention Mechanisms Complete Guide — Learn Code Camp](https://learncodecamp.net/attention-mechanisms-complete-guide/)
- [Sparse Attention Mechanisms Overview — apxml](https://apxml.com/courses/foundations-transformers-architecture/chapter-6-advanced-architectural-variants-analysis/sparse-attention-mechanisms)
- [Sliding Window Attention in Transformers — Emergent Mind](https://www.emergentmind.com/topics/sliding-window-attention-swa)
- [DeepSeek-V4 Technical Documentation (model card PDF)](https://fe-static.deepseek.com/chat/transparency/deepseek-V4-model-card-EN.pdf)
- [DeepSeek-V4: a million-token context that agents can actually use — HuggingFace](https://huggingface.co/blog/deepseekv4)
- [DeepSeek AI Releases DeepSeek-V4: CSA and HCA — MarkTechPost](https://www.marktechpost.com/2026/04/24/deepseek-ai-releases-deepseek-v4-compressed-sparse-attention-and-heavily-compressed-attention-enable-one-million-token-contexts/)
- [DeepSeek-V4: The Interesting Part Is the Attention Architecture — The Salt](https://thesalt.substack.com/p/deepseek-v4-the-interesting-part)
- [DeepSeek Sparse Attention Mechanism (DSA) — Emergent Mind](https://www.emergentmind.com/topics/deepseek-sparse-attention-dsa)
- [Best AI Models May 2026 Leaderboard — Build Fast with AI](https://www.buildfastwithai.com/blogs/best-ai-models-may-2026-leaderboard)
