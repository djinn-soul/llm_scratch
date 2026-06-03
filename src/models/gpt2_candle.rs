use anyhow::Result;
use candle_core::{DType, Module, Tensor};
use candle_nn::{Linear, VarBuilder};

// Candle version of the GPT-2 decoder stack.
//
// This file mirrors the manual GPT model in `src/models/gpt.rs`, but delegates
// tensor math, autograd, and optimizer-compatible parameter storage to Candle.
// The important learning path is still the same:
//   1. token ids -> token embeddings + position embeddings
//   2. repeated decoder blocks: pre-norm attention, residual, pre-norm MLP
//   3. final layer norm
//   4. tied LM head: hidden states @ token_embedding_weight^T -> vocab logits
//
// Shape convention in this file:
//   seq_len = number of tokens in the current context window
//   n_embd  = model width / hidden size
//   n_head  = number of attention heads
//   head_dim = n_embd / n_head

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
    // Real GPT-2 Small dimensions. Use this when loading the public GPT-2
    // checkpoint from Hugging Face because checkpoint tensor shapes must match.
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

    // Tiny learning configuration. Same architecture pattern as GPT-2, but
    // with small dimensions so CPU training runs in a reasonable time.
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

// Helper for GPT-2-style Conv1D weights.
//
// OpenAI GPT-2 stores many projection matrices under names such as `c_attn`
// and `c_proj`. In practice they behave like linear layers, but checkpoint
// orientation is opposite of Candle's `Linear::new` expectation. We load the
// tensor with GPT-2's shape and transpose it once so forward calls can use the
// normal Candle linear layer API.
//
// When training from scratch through `VarMap`, `get_with_hints` also uses the
// initializer here to create the same parameter if it does not already exist.
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
        // Input shape: [seq_len, n_embd].
        // Each token row contains the current residual-stream vector.
        let (seq_len, _n_embd) = xs.dims2()?;

        // STEP 1: PROJECT TO Q, K, V IN ONE LINEAR LAYER
        // GPT-2 packs the three projections into one matrix:
        //   [seq_len, n_embd] -> [seq_len, 3 * n_embd]
        // The result is then split into query, key, and value chunks.
        let qkv = self.c_attn.forward(xs)?;

        let q = qkv.narrow(1, 0, self.n_embd)?;
        let k = qkv.narrow(1, self.n_embd, self.n_embd)?;
        let v = qkv.narrow(1, 2 * self.n_embd, self.n_embd)?;

        // STEP 2: SPLIT MODEL WIDTH INTO ATTENTION HEADS
        // Before: [seq_len, n_embd]
        // After reshape: [seq_len, n_head, head_dim]
        // After transpose: [n_head, seq_len, head_dim]
        // This lets every head run its own attention calculation.
        let q = q
            .reshape((seq_len, self.n_head, self.head_dim))?
            .transpose(0, 1)?;
        let k = k
            .reshape((seq_len, self.n_head, self.head_dim))?
            .transpose(0, 1)?;
        let v = v
            .reshape((seq_len, self.n_head, self.head_dim))?
            .transpose(0, 1)?;

        // STEP 3: SCORE TOKENS AGAINST TOKENS
        // q @ k^T gives one score for "how much token i should read token j".
        // Dividing by sqrt(head_dim) keeps the softmax from becoming too sharp
        // when vectors get wider.
        let scale = (self.head_dim as f64).sqrt();
        let scores = q.matmul(&k.transpose(1, 2)?)?.affine(1.0 / scale, 0.0)?;

        // STEP 4: APPLY CAUSAL MASK
        // Decoder-only language models cannot look at future tokens. The lower
        // triangular mask keeps current and past positions and replaces future
        // scores with -inf so their softmax probability becomes zero.
        let mask = Tensor::tril2(seq_len, DType::U8, xs.device())?.broadcast_as((
            self.n_head,
            seq_len,
            seq_len,
        ))?;

        let neg_inf = Tensor::full(f32::NEG_INFINITY, scores.shape(), xs.device())?;
        let scores = mask.where_cond(&scores, &neg_inf)?;

        // STEP 5: TURN SCORES INTO WEIGHTS, THEN BLEND VALUE VECTORS
        // softmax runs across the source-token axis. The final matmul produces
        // one context vector per head and per destination token.
        let attn_weights = candle_nn::ops::softmax(&scores, 2)?;
        let context = attn_weights.matmul(&v)?;

        // STEP 6: MERGE HEADS BACK TO MODEL WIDTH AND APPLY OUTPUT PROJECTION
        // [n_head, seq_len, head_dim] -> [seq_len, n_embd]
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
        // GPT-2 feed-forward block:
        //   1. expand width:  [seq_len, n_embd] -> [seq_len, n_inner]
        //   2. apply GELU non-linearity
        //   3. project back:  [seq_len, n_inner] -> [seq_len, n_embd]
        // It is position-wise: tokens do not talk to each other here; attention
        // already handled token mixing.
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
        // GPT-2 uses pre-norm decoder blocks:
        //
        //   x -> ln_1 -> attention -> +x
        //      -> ln_2 -> MLP       -> +residual
        //
        // Residual connections keep the original stream available while each
        // sub-layer adds a learned correction.
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

        // STEP 1: BUILD TOKEN POSITIONS
        // `tokens` contains ids like [15496, 995]. Position ids are [0, 1, ...].
        // Token embeddings tell the model *what* each token is; position
        // embeddings tell it *where* each token sits in the context window.
        let positions = Tensor::arange(0u32, seq_len as u32, tokens.device())?;
        let token_emb = self.wte.forward(tokens)?;
        let pos_emb = self.wpe.forward(&positions)?;
        let mut x = token_emb.add(&pos_emb)?;

        // STEP 2: RUN THE DECODER STACK
        // Each block updates the residual stream with masked self-attention and
        // a position-wise MLP.
        for block in &self.h {
            x = block.forward(&x)?;
        }

        // STEP 3: FINAL NORMALIZATION
        // GPT-2 applies one last LayerNorm before converting hidden vectors into
        // vocabulary logits.
        let x = self.ln_f.forward(&x)?;

        // STEP 4: TIED LANGUAGE-MODEL HEAD
        // Reuse the token embedding matrix as the output classifier:
        //   [seq_len, n_embd] @ [n_embd, vocab_size] -> [seq_len, vocab_size]
        // This is called weight tying. It reduces parameters and matches GPT-2.
        let wte_weights = self.wte.embeddings();
        let logits = x.matmul(&wte_weights.t()?)?;
        Ok(logits)
    }
}
