use anyhow::Result;
use candle_core::{DType, Module, Tensor};
use candle_nn::{linear, Linear, VarBuilder};

// ════════════════════════════════════════════════════════════════
// GPT-2 CONFIGURATION
// These match the exact values in openai-community/gpt2's config.json
// ════════════════════════════════════════════════════════════════
#[derive(Debug, Clone)]
struct Gpt2Config {
    vocab_size: usize,  // 50257 unique BPE tokens
    n_embd: usize,      // 768  — hidden/embedding dimension
    n_head: usize,      // 12   — number of attention heads
    n_layer: usize,     // 12   — number of transformer blocks
    n_positions: usize, // 1024 — context window length
    n_inner: usize,     // 3072 — feed-forward hidden size (4 × n_embd)
}

impl Gpt2Config {
    fn gpt2_small() -> Self {
        Self {
            vocab_size: 50257,
            n_embd: 768,
            n_head: 12,
            n_layer: 12,
            n_positions: 1024,
            n_inner: 3072,
        }
    }

    // d_head = 768 / 12 = 64 values per attention head
    fn head_dim(&self) -> usize {
        self.n_embd / self.n_head
    }
}

// ════════════════════════════════════════════════════════════════
// CAUSAL SELF-ATTENTION
//
// Shape flow:
//   input  [seq_len, n_embd]
//   qkv    [seq_len, 3 * n_embd]  — one fused linear projection
//   q,k,v  [n_head, seq_len, head_dim]  each after reshape + transpose
//   scores [n_head, seq_len, seq_len]   = Q @ K^T / sqrt(head_dim)
//   output [seq_len, n_embd]            after masking, softmax, V-weighted sum
// ════════════════════════════════════════════════════════════════
struct Attention {
    c_attn: Linear, // fused QKV projection: n_embd → 3 * n_embd
    c_proj: Linear, // output projection: n_embd → n_embd
    n_head: usize,
    n_embd: usize,
    head_dim: usize,
}

impl Attention {
    fn load(vb: VarBuilder, cfg: &Gpt2Config) -> Result<Self> {
        let c_attn = linear(cfg.n_embd, 3 * cfg.n_embd, vb.pp("c_attn"))?;
        let c_proj = linear(cfg.n_embd, cfg.n_embd, vb.pp("c_proj"))?;
        Ok(Self {
            c_attn,
            c_proj,
            n_head: cfg.n_head,
            n_embd: cfg.n_embd,
            head_dim: cfg.head_dim(),
        })
    }
}

struct Mlp {
    c_fc: Linear,   // expansion:  n_embd → 4 × n_embd
    c_proj: Linear, // projection: 4 × n_embd → n_embd
}
impl Mlp {
    fn load(vb: VarBuilder, cfg: &Gpt2Config) -> Result<Self> {
        let c_fc = linear(cfg.n_embd, cfg.n_inner, vb.pp("c_fc"))?;
        let c_proj = linear(cfg.n_inner, cfg.n_embd, vb.pp("c_proj"))?;
        Ok(Self { c_fc, c_proj })
    }
}

// ════════════════════════════════════════════════════════════════
// TRANSFORMER BLOCK (DECODER)
//
// GPT-2 uses Pre-LayerNorm (normalize BEFORE attention/mlp).
// This is slightly different from the original "Attention is All You Need"
// paper which used Post-LayerNorm (normalize AFTER).
//
// Formula per block:
//   x = x + Attention(LayerNorm1(x))   ← residual + self-attention
//   x = x + Mlp(LayerNorm2(x))         ← residual + feed-forward
//
// Shape flow (unchanged throughout):
//   input  [seq_len, n_embd]
//   output [seq_len, n_embd]   ← shape is preserved by every block
// ════════════════════════════════════════════════════════════════
struct TransformerBlock {
    ln_1: candle_nn::LayerNorm, // LayerNorm before attention
    attn: Attention,            // Causal Self-Attention
    ln_2: candle_nn::LayerNorm, // LayerNorm before MLP
    mlp: Mlp,                   // Feed-Forward Network
}

impl TransformerBlock {
    fn load(vb: VarBuilder, cfg: &Gpt2Config) -> Result<Self> {
        let ln_1 = candle_nn::layer_norm(cfg.n_embd, 1e-5, vb.pp("ln_1"))?;
        let attn = Attention::load(vb.pp("attn"), cfg)?;
        let ln_2 = candle_nn::layer_norm(cfg.n_embd, 1e-5, vb.pp("ln_2"))?;
        let mlp = Mlp::load(vb.pp("mlp"), cfg)?;
        Ok(Self {
            ln_1,
            attn,
            ln_2,
            mlp,
        })
    }
}


impl Module for Attention {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let (seq_len, _n_embd) = xs.dims2()?;

        // 1. Compute fused QKV projection → [seq_len, 3 * n_embd]
        let qkv = self.c_attn.forward(xs)?;

        // 2. Split into Q, K, V → each [seq_len, n_embd]
        let q = qkv.narrow(1, 0, self.n_embd)?;
        let k = qkv.narrow(1, self.n_embd, self.n_embd)?;
        let v = qkv.narrow(1, 2 * self.n_embd, self.n_embd)?;

        // 3. Reshape to [n_head, seq_len, head_dim]
        // NOTE: Candle's reshape takes a single tuple — not separate arguments!
        let q = q
            .reshape((seq_len, self.n_head, self.head_dim))?
            .transpose(0, 1)?;
        let k = k
            .reshape((seq_len, self.n_head, self.head_dim))?
            .transpose(0, 1)?;
        let v = v
            .reshape((seq_len, self.n_head, self.head_dim))?
            .transpose(0, 1)?;

        // 4. Scaled dot-product scores = Q @ K^T / sqrt(head_dim)
        let scale = (self.head_dim as f64).sqrt();
        let scores = q.matmul(&k.transpose(1, 2)?)?.affine(1.0 / scale, 0.0)?;

        // 5. Build causal mask (lower-triangular matrix of ones)
        // NOTE: Candle 0.10 uses Tensor::tril2(n, dtype, device) — NOT Tensor::tril(...)
        let mask = Tensor::tril2(seq_len, DType::F32, xs.device())?.broadcast_as((
            self.n_head,
            seq_len,
            seq_len,
        ))?;

        // 6. Mask future positions: where mask=0, replace score with -inf
        let neg_inf = Tensor::full(f32::NEG_INFINITY, scores.shape(), xs.device())?;
        let scores = mask.where_cond(&scores, &neg_inf)?;

        // 7. Softmax over last dimension → attention weights [n_head, seq_len, seq_len]
        let attn_weights = candle_nn::ops::softmax(&scores, 2)?;

        // 8. Weighted sum of values → [n_head, seq_len, head_dim]
        let context = attn_weights.matmul(&v)?;

        // 9. Merge heads back: [n_head, seq_len, head_dim] → [seq_len, n_embd]
        let context = context.transpose(0, 1)?.reshape((seq_len, self.n_embd))?;

        // 10. Final output projection → [seq_len, n_embd]
        self.c_proj.forward(&context)
    }
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let hidden = self.c_fc.forward(xs)?;
        // GELU activation
        let hidden = hidden.gelu_erf()?;
        // Final output projection
        self.c_proj.forward(&hidden)
    }
}

impl Module for TransformerBlock {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let residual = xs;
        let normalized = self.ln_1.forward(xs)?;
        let attended = self.attn.forward(&normalized)?;
        let x = residual.add(&attended)?;

        let residual = &x;
        let normalized = self.ln_2.forward(&x)?;
        let mlp_out = self.mlp.forward(&normalized)?;
        let x = residual.add(&mlp_out)?;
        Ok(x)
    }
}

fn main() -> Result<()> {
    Ok(())
}
