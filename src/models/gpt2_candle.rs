use anyhow::Result;
use candle_core::{DType, Module, Tensor};
use candle_nn::{Linear, VarBuilder};

// ════════════════════════════════════════════════════════════════
// GPT-2 CONFIGURATION
// ════════════════════════════════════════════════════════════════
#[derive(Debug, Clone)]
pub struct Gpt2Config {
    pub vocab_size: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_layer: usize,
    pub n_positions: usize,
    pub n_inner: usize,
}

impl Gpt2Config {
    pub fn gpt2_small() -> Self {
        Self {
            vocab_size: 50257,
            n_embd: 768,
            n_head: 12,
            n_layer: 12,
            n_positions: 1024,
            n_inner: 3072,
        }
    }

    pub fn gpt2_mini() -> Self {
        Self {
            vocab_size: 1000,
            n_embd: 16,
            n_head: 2,
            n_layer: 2,
            n_positions: 8,
            n_inner: 32,
        }
    }

    pub fn head_dim(&self) -> usize {
        self.n_embd / self.n_head
    }
}

// Helper to transpose Conv1D weights from GPT-2 checkpoints
fn load_conv1d(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Linear> {
    let init_w = candle_nn::Init::Randn {
        mean: 0.0,
        stdev: 0.02,
    };
    let w = vb
        .get_with_hints((in_dim, out_dim), "weight", init_w)?
        .t()?;
    let init_b = candle_nn::Init::Const(0.0);
    let b = vb.get_with_hints(out_dim, "bias", init_b)?;
    Ok(Linear::new(w, Some(b)))
}

// ════════════════════════════════════════════════════════════════
// CAUSAL SELF-ATTENTION
// ════════════════════════════════════════════════════════════════
pub struct Attention {
    c_attn: Linear,
    c_proj: Linear,
    n_head: usize,
    n_embd: usize,
    head_dim: usize,
}

impl Attention {
    pub fn load(vb: VarBuilder, cfg: &Gpt2Config) -> Result<Self> {
        Ok(Self {
            c_attn: load_conv1d(cfg.n_embd, 3 * cfg.n_embd, vb.pp("c_attn"))?,
            c_proj: load_conv1d(cfg.n_embd, cfg.n_embd, vb.pp("c_proj"))?,
            n_head: cfg.n_head,
            n_embd: cfg.n_embd,
            head_dim: cfg.head_dim(),
        })
    }
}

impl Module for Attention {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let (seq_len, _n_embd) = xs.dims2()?;
        let qkv = self.c_attn.forward(xs)?;

        let q = qkv.narrow(1, 0, self.n_embd)?;
        let k = qkv.narrow(1, self.n_embd, self.n_embd)?;
        let v = qkv.narrow(1, 2 * self.n_embd, self.n_embd)?;

        let q = q
            .reshape((seq_len, self.n_head, self.head_dim))?
            .transpose(0, 1)?;
        let k = k
            .reshape((seq_len, self.n_head, self.head_dim))?
            .transpose(0, 1)?;
        let v = v
            .reshape((seq_len, self.n_head, self.head_dim))?
            .transpose(0, 1)?;

        let scale = (self.head_dim as f64).sqrt();
        let scores = q.matmul(&k.transpose(1, 2)?)?.affine(1.0 / scale, 0.0)?;

        let mask = Tensor::tril2(seq_len, DType::U8, xs.device())?.broadcast_as((
            self.n_head,
            seq_len,
            seq_len,
        ))?;

        let neg_inf = Tensor::full(f32::NEG_INFINITY, scores.shape(), xs.device())?;
        let scores = mask.where_cond(&scores, &neg_inf)?;

        let attn_weights = candle_nn::ops::softmax(&scores, 2)?;
        let context = attn_weights.matmul(&v)?;

        let context = context.transpose(0, 1)?.reshape((seq_len, self.n_embd))?;
        self.c_proj.forward(&context)
    }
}

// ════════════════════════════════════════════════════════════════
// FEED-FORWARD MLP
// ════════════════════════════════════════════════════════════════
pub struct Mlp {
    c_fc: Linear,
    c_proj: Linear,
}

impl Mlp {
    pub fn load(vb: VarBuilder, cfg: &Gpt2Config) -> Result<Self> {
        Ok(Self {
            c_fc: load_conv1d(cfg.n_embd, cfg.n_inner, vb.pp("c_fc"))?,
            c_proj: load_conv1d(cfg.n_inner, cfg.n_embd, vb.pp("c_proj"))?,
        })
    }
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let hidden = self.c_fc.forward(xs)?.gelu_erf()?;
        self.c_proj.forward(&hidden)
    }
}

// ════════════════════════════════════════════════════════════════
// TRANSFORMER DECODER BLOCK
// ════════════════════════════════════════════════════════════════
pub struct TransformerBlock {
    ln_1: candle_nn::LayerNorm,
    attn: Attention,
    ln_2: candle_nn::LayerNorm,
    mlp: Mlp,
}

impl TransformerBlock {
    pub fn load(vb: VarBuilder, cfg: &Gpt2Config) -> Result<Self> {
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

impl Module for TransformerBlock {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let residual = xs;
        let normalized = self.ln_1.forward(xs)?;
        let attended = self.attn.forward(&normalized)?;
        let x = residual.add(&attended)?;

        let residual = &x;
        let normalized = self.ln_2.forward(&x)?;
        let mlp_out = self.mlp.forward(&normalized)?;
        residual.add(&mlp_out)
    }
}

// ════════════════════════════════════════════════════════════════
// FULL GPT-2 MODEL
// ════════════════════════════════════════════════════════════════
pub struct Gpt2Model {
    wte: candle_nn::Embedding,
    wpe: candle_nn::Embedding,
    h: Vec<TransformerBlock>,
    ln_f: candle_nn::LayerNorm,
    cfg: Gpt2Config,
}

impl Gpt2Model {
    pub fn load(vb: VarBuilder, cfg: &Gpt2Config) -> Result<Self> {
        let init_w = candle_nn::Init::Randn {
            mean: 0.0,
            stdev: 0.02,
        };
        let wte_weights =
            vb.pp("wte")
                .get_with_hints((cfg.vocab_size, cfg.n_embd), "weight", init_w.clone())?;
        let wte = candle_nn::Embedding::new(wte_weights, cfg.n_embd);

        let wpe_weights =
            vb.pp("wpe")
                .get_with_hints((cfg.n_positions, cfg.n_embd), "weight", init_w)?;
        let wpe = candle_nn::Embedding::new(wpe_weights, cfg.n_embd);

        let mut h = Vec::with_capacity(cfg.n_layer);
        let vb_h = vb.pp("h");
        for i in 0..cfg.n_layer {
            h.push(TransformerBlock::load(vb_h.pp(i), cfg)?);
        }

        let ln_f = candle_nn::layer_norm(cfg.n_embd, 1e-5, vb.pp("ln_f"))?;

        Ok(Self {
            wte,
            wpe,
            h,
            ln_f,
            cfg: cfg.clone(),
        })
    }

    pub fn forward(&self, tokens: &Tensor) -> Result<Tensor> {
        let seq_len = tokens.dim(0)?;
        if seq_len > self.cfg.n_positions {
            return Err(anyhow::anyhow!(
                "Sequence length {} exceeds maximum positions {}",
                seq_len,
                self.cfg.n_positions
            ));
        }

        let positions = Tensor::arange(0u32, seq_len as u32, tokens.device())?;
        let token_emb = self.wte.forward(tokens)?;
        let pos_emb = self.wpe.forward(&positions)?;
        let mut x = token_emb.add(&pos_emb)?;

        for block in &self.h {
            x = block.forward(&x)?;
        }

        let x = self.ln_f.forward(&x)?;
        let wte_weights = self.wte.embeddings();
        let logits = x.matmul(&wte_weights.t()?)?;
        Ok(logits)
    }
}
