use anyhow::Result;
use candle_core::{Module, Tensor};
use candle_nn::kv_cache::ConcatKvCache;
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
//
// Tiny matrix view of the whole forward path:
//   token ids      [15496, 995]
//   token_emb      [2, n_embd]   one learned row per token id
//   pos_emb        [2, n_embd]   one learned row per position 0,1
//   x              [2, n_embd]   token_emb + pos_emb
//   decoder blocks [2, n_embd]   same shape, richer values
//   logits         [2, vocab]    one vocabulary score row per position

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
// Matrix orientation example:
//   checkpoint weight behaves like:  input [seq_len, in_dim] @ W [in_dim, out_dim]
//   Candle Linear stores weight for its own layout, so we transpose at load.
//
// For `c_attn`, `out_dim = 3 * n_embd` because one projection produces Q, K,
// and V side by side.
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
    // In normal training/full-context inference this stays false and every
    // forward call receives the full sequence. During generation this becomes
    // true after the prompt pass, so later calls can send only the newest token.
    pub use_cache: std::cell::Cell<bool>,
    // Candle stores cached keys/values per attention block.
    // Shape after head splitting:
    //   K/V before cache append: [n_head, current_seq_len, head_dim]
    //   K/V after cache append:  [n_head, past_len + current_seq_len, head_dim]
    //
    // The RefCell is needed because `Module::forward` takes `&self`, but cache
    // append mutates the stored K/V history during inference.
    pub kv_cache: std::cell::RefCell<ConcatKvCache>,
}

impl Attention {
    pub fn load(vb: VarBuilder, cfg: &Gpt2Config) -> Result<Self> {
        Ok(Self {
            c_attn: load_conv1d(cfg.n_embd, 3 * cfg.n_embd, vb.pp("c_attn"))?,
            c_proj: load_conv1d(cfg.n_embd, cfg.n_embd, vb.pp("c_proj"))?,
            n_head: cfg.n_head,
            n_embd: cfg.n_embd,
            head_dim: cfg.head_dim(),
            use_cache: std::cell::Cell::new(false),
            kv_cache: std::cell::RefCell::new(ConcatKvCache::new(1)),
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
        //
        // If n_embd = 4, one token row becomes 12 numbers:
        //   qkv row = [q0 q1 q2 q3 | k0 k1 k2 k3 | v0 v1 v2 v3]
        //
        // For seq_len = 2:
        //   qkv = [
        //     [q0..., k0..., v0...],   // token position 0
        //     [q1..., k1..., v1...],   // token position 1
        //   ]
        let qkv = self.c_attn.forward(xs)?;

        let q = qkv.narrow(1, 0, self.n_embd)?;
        let k = qkv.narrow(1, self.n_embd, self.n_embd)?;
        let v = qkv.narrow(1, 2 * self.n_embd, self.n_embd)?;

        // STEP 2: SPLIT MODEL WIDTH INTO ATTENTION HEADS
        // Before: [seq_len, n_embd]
        // After reshape: [seq_len, n_head, head_dim]
        // After transpose: [n_head, seq_len, head_dim]
        // This lets every head run its own attention calculation.
        //
        // Example when n_embd = 4, n_head = 2, head_dim = 2:
        //   q token row before split: [a, b, c, d]
        //   head 0 receives: [a, b]
        //   head 1 receives: [c, d]
        //
        // After transpose the head dimension comes first:
        //   q[0] = all token rows for head 0
        //   q[1] = all token rows for head 1
        let q = q
            .reshape((seq_len, self.n_head, self.head_dim))?
            .transpose(0, 1)?;
        let mut k = k
            .reshape((seq_len, self.n_head, self.head_dim))?
            .transpose(0, 1)?;
        let mut v = v
            .reshape((seq_len, self.n_head, self.head_dim))?
            .transpose(0, 1)?;

        // STEP 2B: UPDATE THE KV CACHE MATRIX
        //
        // Full pass, no cache:
        //   q [n_head, seq_len, head_dim]
        //   k [n_head, seq_len, head_dim]
        //   v [n_head, seq_len, head_dim]
        //
        // Cached generation after the prompt:
        //   q [n_head, 1, head_dim]       only the newest token asks a question
        //   k [n_head, 1, head_dim]       newest key row
        //   v [n_head, 1, head_dim]       newest value row
        //   cached K/V already contain previous token rows.
        //
        // `append(&k, &v)` grows the K/V matrices along the sequence axis:
        //
        //   before append, past_len = 4:
        //     cached_k = [k0, k1, k2, k3]          shape [n_head, 4, head_dim]
        //     new_k    = [k4]                      shape [n_head, 1, head_dim]
        //
        //   after append:
        //     full_k   = [k0, k1, k2, k3, k4]      shape [n_head, 5, head_dim]
        //     full_v   = [v0, v1, v2, v3, v4]      shape [n_head, 5, head_dim]
        //
        // Concrete toy numbers for one head with head_dim = 2:
        //
        //   cached_k = [
        //     [1.0, 0.0],   // k0
        //     [0.0, 1.0],   // k1
        //     [1.0, 1.0],   // k2
        //     [2.0, 0.0],   // k3
        //   ]
        //
        //   new_k = [
        //     [0.5, 1.5],   // k4
        //   ]
        //
        //   full_k after append = [
        //     [1.0, 0.0],   // k0 from cache
        //     [0.0, 1.0],   // k1 from cache
        //     [1.0, 1.0],   // k2 from cache
        //     [2.0, 0.0],   // k3 from cache
        //     [0.5, 1.5],   // k4 just appended
        //   ]
        //
        // Same update happens to values:
        //
        //   cached_v = [[10.0, 0.0], [0.0, 10.0], [5.0, 5.0], [8.0, 2.0]]
        //   new_v    = [[2.0, 8.0]]
        //   full_v   = [[10.0, 0.0], [0.0, 10.0], [5.0, 5.0],
        //               [8.0, 2.0], [2.0, 8.0]]
        //
        // Q is not appended. Old tokens already produced their output rows, so
        // the current step only needs q4 scoring against all visible K rows.
        //
        // `pos_offset` is the count of cached rows before this call. It lets
        // the causal mask treat local query row 0 as absolute position 4, not
        // as the beginning of a brand-new sequence.
        let mut pos_offset = 0;
        if self.use_cache.get() {
            let (full_k, full_v) = self.kv_cache.borrow_mut().append(&k, &v)?;
            pos_offset = full_k.dim(1)? - k.dim(1)?;
            k = full_k;
            v = full_v;
        }

        // STEP 3: SCORE TOKENS AGAINST TOKENS
        // q @ k^T gives one score for "how much token i should read token j".
        // Dividing by sqrt(head_dim) keeps the softmax from becoming too sharp
        // when vectors get wider.
        //
        // With cache during one-token generation:
        //   q                 [n_head, 1, head_dim]
        //   k.transpose(1, 2) [n_head, head_dim, past_len + 1]
        //   scores            [n_head, 1, past_len + 1]
        //
        // Example with past_len = 4:
        //   scores[head, 0, :] = q4 dot [k0, k1, k2, k3, k4]
        //
        // Continuing the toy numbers above:
        //
        //   q4 = [[1.0, 2.0]]
        //
        //   full_k^T = [
        //     [1.0, 0.0, 1.0, 2.0, 0.5],
        //     [0.0, 1.0, 1.0, 0.0, 1.5],
        //   ]
        //
        //   scores = q4 @ full_k^T
        //          = [[1.0, 2.0, 3.0, 2.0, 3.5]]
        //
        // The score row has five columns because token 4 can read four cached
        // tokens plus itself.
        //
        // That single row is the whole point of KV cache: recompute the newest
        // query, but reuse all old key/value rows instead of rebuilding them.
        let scale = (self.head_dim as f64).sqrt();
        let scores = q.matmul(&k.transpose(1, 2)?)?.affine(1.0 / scale, 0.0)?;

        // STEP 4: APPLY CAUSAL MASK
        // Decoder-only language models cannot look at future tokens. The lower
        // triangular mask keeps current and past positions and replaces future
        // scores with -inf so their softmax probability becomes zero.
        //
        // In a cached call, query indices are absolute positions:
        //   pos_offset = 4, seq_len = 1 -> q_indices = [4]
        //   total_seq_len = 5           -> k_indices = [0, 1, 2, 3, 4]
        //
        // Matrix view before broadcasting over heads:
        //
        //   q_indices = [[4, 4, 4, 4, 4]]
        //   k_indices = [[0, 1, 2, 3, 4]]
        //
        //   q_indices >= k_indices
        //             = [[T, T, T, T, T]]
        //
        // The comparison `q_indices >= k_indices` means:
        //   key 0..4 are visible to query 4
        //   key >4 would be future and therefore masked
        //
        // This coordinate mask also works when the prompt call has seq_len > 1,
        // because each query row receives its real absolute position.
        let total_seq_len = pos_offset + seq_len;
        let q_indices = Tensor::arange(
            pos_offset as u32,
            (pos_offset + seq_len) as u32,
            xs.device(),
        )?
        .reshape((seq_len, 1))?
        .broadcast_as((self.n_head, seq_len, total_seq_len))?;

        let k_indices = Tensor::arange(0u32, total_seq_len as u32, xs.device())?
            .reshape((1, total_seq_len))?
            .broadcast_as((self.n_head, seq_len, total_seq_len))?;

        let mask = q_indices.ge(&k_indices)?;
        let neg_inf = Tensor::full(f32::NEG_INFINITY, scores.shape(), xs.device())?;
        let scores = mask.where_cond(&scores, &neg_inf)?;

        // STEP 5: TURN SCORES INTO WEIGHTS, THEN BLEND VALUE VECTORS
        // softmax runs across the source-token axis. The final matmul produces
        // one context vector per head and per destination token.
        //
        // Cached one-token shape:
        //   attn_weights [n_head, 1, past_len + 1]
        //   v            [n_head, past_len + 1, head_dim]
        //   context      [n_head, 1, head_dim]
        //
        // So the newest token output is a weighted blend of cached value rows:
        //   context4 = w0*v0 + w1*v1 + w2*v2 + w3*v3 + w4*v4
        //
        // With the toy full_v above, suppose softmax produces:
        //   weights = [[0.05, 0.10, 0.25, 0.10, 0.50]]
        //
        // Then the newest context row is:
        //   context4 =
        //     0.05 * [10.0, 0.0]  +
        //     0.10 * [0.0, 10.0]  +
        //     0.25 * [5.0, 5.0]   +
        //     0.10 * [8.0, 2.0]   +
        //     0.50 * [2.0, 8.0]
        //
        //   context4 = [3.55, 6.45]
        let attn_weights = candle_nn::ops::softmax(&scores, 2)?;
        let context = attn_weights.matmul(&v)?;

        // STEP 6: MERGE HEADS BACK TO MODEL WIDTH AND APPLY OUTPUT PROJECTION
        // [n_head, seq_len, head_dim] -> [seq_len, n_embd]
        //
        // This reverses the earlier head split. With two heads:
        //   head0 context for token4 = [h0a, h0b]
        //   head1 context for token4 = [h1a, h1b]
        //
        // merged token4 row = [h0a, h0b, h1a, h1b]
        //
        // `c_proj` then mixes those head outputs so information found by
        // different heads can interact before returning to the residual stream.
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
        //
        // Matrix view for one token row:
        //   x          [n_embd]
        //   c_fc(x)    [n_inner]   wider feature space
        //   GELU       [n_inner]   gates small/negative features smoothly
        //   c_proj     [n_embd]    returns to residual-stream width
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
        //
        // Toy row update:
        //   residual token row = [1.0, 2.0, 3.0, 4.0]
        //   attention output   = [0.1, 0.0, 0.3, -0.2]
        //   after attention    = [1.1, 2.0, 3.3, 3.8]
        //
        // The MLP then adds another same-width correction to that updated row.
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
    /// enable / disable KV cache for inference
    pub fn set_use_cache(&self, use_cache: bool) {
        for block in &self.h {
            block.attn.use_cache.set(use_cache);
        }
    }
    pub fn clear_cache(&self) {
        for block in &self.h {
            block.attn.kv_cache.borrow_mut().reset();
        }
    }
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
        let pos_offset = if self.h[0].attn.use_cache.get() {
            self.h[0].attn.kv_cache.borrow().current_seq_len()
        } else {
            0
        };
        if pos_offset + seq_len > self.cfg.n_positions {
            return Err(anyhow::anyhow!(
                "Sequence length {} + pos_offset {} exceeds maximum positions {}",
                seq_len,
                pos_offset,
                self.cfg.n_positions
            ));
        }

        // STEP 1: BUILD TOKEN POSITIONS
        // `tokens` contains ids like [15496, 995]. Position ids are [0, 1, ...].
        // Token embeddings tell the model *what* each token is; position
        // embeddings tell it *where* each token sits in the context window.
        //
        // For a two-token prompt:
        //   tokens    = [15496, 995]
        //   positions = [0, 1]
        //
        // The embedding lookup returns two matrices with identical shape:
        //   token_emb = [
        //     row_for_token_15496,
        //     row_for_token_995,
        //   ]
        //   pos_emb = [
        //     row_for_position_0,
        //     row_for_position_1,
        //   ]
        //
        // The residual stream starts as:
        //   x[0] = token_emb[0] + pos_emb[0]
        //   x[1] = token_emb[1] + pos_emb[1]
        //
        // With KV cache, this forward call may contain only the newest token:
        //   cached prompt length = 4
        //   tokens              = [token4]
        //   positions           = [4], not [0]
        //
        // This must match the attention cache. The key/value cache already has
        // rows for positions 0..3, so the new token must use the absolute
        // positional embedding row 4 before its K/V rows are appended.
        let positions = Tensor::arange(
            pos_offset as u32,
            (pos_offset + seq_len) as u32,
            tokens.device(),
        )?;
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
        //
        // For the last token row:
        //   hidden_last [n_embd] dot embedding_row[token_id]
        //     -> one logit for that token id
        //
        // Doing this against every vocabulary row gives `logits_last`, and the
        // generation code samples the next token from that final row.
        let wte_weights = self.wte.embeddings();
        let logits = x.matmul(&wte_weights.t()?)?;
        Ok(logits)
    }
}
