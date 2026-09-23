use burn::module::Module;
use burn::nn::attention::{MhaInput, MultiHeadAttention, MultiHeadAttentionConfig};
use burn::nn::{Gelu, LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

/// Diffusion Transformer Block with Adaptive Layer Normalization (adaLN-Zero).
///
/// In standard Vision Transformers (ViT), Layer Normalization is static: learned gain (γ)
/// and bias (β) are shared across all samples regardless of context.
///
/// In **DiT** (Peebles & Xie, 2022), diffusion models require conditioning on:
/// 1. Current diffusion timestep `t` (how noisy the image is)
/// 2. Class label `c` (what digit/object to generate)
///
/// Rather than using expensive Cross-Attention mechanisms, DiT introduces **adaLN-Zero**:
/// A single Linear layer predicts 6 conditioning parameters per block directly from the condition vector `c`:
/// - **Attention Modulation**:
///   - `γ1`: Scale factor for attention pre-normalization
///   - `β1`: Shift factor for attention pre-normalization
///   - `α1`: Dimension-wise gate factor for the attention residual branch
/// - **Feed-Forward (MLP) Modulation**:
///   - `γ2`: Scale factor for MLP pre-normalization
///   - `β2`: Shift factor for MLP pre-normalization
///   - `α2`: Dimension-wise gate factor for the MLP residual branch
///
/// ### Mathematical Intuition Behind `(1 + γ)` and `α`:
/// - **Identity Initialization**: We multiply LayerNorm output by `(1 + γ1)` instead of `γ1`.
///   When `γ1 = 0`, `(1 + 0) = 1`, which preserves standard LayerNorm.
/// - **Zero-Gating (α)**: In the adaLN-Zero paper, the projection weights for `α` are initialized to 0.
///   When `α = 0`, `x_out = x + 0 * SubLayer(x) = x`.
///   This makes every transformer block act as an **identity function** at initialization,
///   allowing very deep transformers to train stably without gradient explosions!
///
/// ### Forward Computational Graph:
/// ```text
///                         ┌───────────────────────────────────┐
///                         │   Conditioning Vector `cond`      │
///                         │             [B, D]                │
///                         └─────────────────┬─────────────────┘
///                                           │  Linear (adaLN) [D -> 6D]
///                                           ▼
///                          [gamma1, beta1, alpha1, gamma2, beta2, alpha2]
///                                           │
///     x [B, N, D] ──► LayerNorm ──► * (1 + gamma1) + beta1 ──► Self-Attention ──► * alpha1 ──(+)──► x_mid
///        │                                                                                     ▲
///        └────────────────────────────────────── Residual ─────────────────────────────────────┘
///
///     x_mid [B,N,D] ─► LayerNorm ──► * (1 + gamma2) + beta2 ──────► MLP ──────────► * alpha2 ──(+)──► Output
///        │                                                                                     ▲
///        └────────────────────────────────────── Residual ─────────────────────────────────────┘
/// ```
#[derive(Module, Debug)]
pub struct DiTBlock<B: Backend> {
    /// Pre-attention Layer Normalization
    norm1: LayerNorm<B>,
    /// Multi-Head Self-Attention module
    attn: MultiHeadAttention<B>,
    /// Pre-MLP Layer Normalization
    norm2: LayerNorm<B>,
    /// MLP linear projection 1: expands hidden_dim -> mlp_hidden (e.g. 128 -> 512)
    mlp_fc1: Linear<B>,
    /// MLP linear projection 2: projects mlp_hidden back -> hidden_dim (e.g. 512 -> 128)
    mlp_fc2: Linear<B>,
    /// Gaussian Error Linear Unit (GELU) activation function
    mlp_act: Gelu,
    /// Adaptive LayerNorm linear projection: maps [B, cond_dim] -> [B, 6 * hidden_dim]
    ada_ln: Linear<B>,
}

impl<B: Backend> DiTBlock<B> {
    /// Creates and initializes a new `DiTBlock`.
    ///
    /// # Arguments
    /// * `hidden_dim` - Embedding size `D` of the transformer (e.g. 128 or 256)
    /// * `num_heads` - Number of attention heads (e.g. 4 or 8)
    /// * `mlp_ratio` - Expansion ratio for the inner MLP layer (typically 4.0)
    /// * `cond_dim` - Dimension of the conditioning vector (timestep + class embeddings)
    /// * `device` - Compute device (CPU/GPU)
    pub fn new(
        hidden_dim: usize,
        num_heads: usize,
        mlp_ratio: f64,
        cond_dim: usize,
        device: &B::Device,
    ) -> Self {
        let norm1 = LayerNormConfig::new(hidden_dim).init(device);
        let attn = MultiHeadAttentionConfig::new(hidden_dim, num_heads).init(device);
        let norm2 = LayerNormConfig::new(hidden_dim).init(device);

        // Calculate MLP hidden dimension: hidden_dim * mlp_ratio (e.g., 128 * 4.0 = 512)
        let mlp_hidden = (hidden_dim as f64 * mlp_ratio) as usize;
        let fc1 = LinearConfig::new(hidden_dim, mlp_hidden).init(device);
        let fc2 = LinearConfig::new(mlp_hidden, hidden_dim).init(device);
        let act = Gelu::new();

        // Projects conditioning vector [B, cond_dim] to 6 separate vectors of size hidden_dim:
        // [gamma1, beta1, alpha1, gamma2, beta2, alpha2] -> total size = 6 * hidden_dim
        let ada_ln = LinearConfig::new(cond_dim, 6 * hidden_dim).init(device);

        Self {
            norm1,
            attn,
            norm2,
            mlp_fc1: fc1,
            mlp_fc2: fc2,
            mlp_act: act,
            ada_ln,
        }
    }

    /// Forward pass through the modulated DiT Block.
    ///
    /// ### Step-by-Step Execution:
    /// 1. **Modulation Parameter Slicing**:
    ///    - Linearly project `cond` `[B, D]` -> `[B, 1, 6D]`
    ///    - Slice into 6 individual chunks `[B, 1, D]`: `γ1, β1, α1, γ2, β2, α2`.
    /// 2. **Modulated Multi-Head Self-Attention**:
    ///    - `x_norm1 = LayerNorm(x)` `[B, N, D]`
    ///    - `x_mod1 = x_norm1 * (1 + γ1) + β1` (broadcast along tokens)
    ///    - `attn_out = SelfAttention(x_mod1)`
    ///    - `x = x + α1 * attn_out`
    /// 3. **Modulated Feed-Forward Network (MLP)**:
    ///    - `x_norm2 = LayerNorm(x)` `[B, N, D]`
    ///    - `x_mod2 = x_norm2 * (1 + γ2) + β2`
    ///    - `mlp_out = FC2(GELU(FC1(x_mod2)))`
    ///    - `x = x + α2 * mlp_out`
    ///
    /// # Arguments
    /// * `x` - Token sequence tensor of shape `[Batch, Num_Patches, hidden_dim]`
    /// * `cond` - Conditioning vector of shape `[Batch, cond_dim]`
    ///
    /// # Returns
    /// Modulated output sequence tensor of shape `[Batch, Num_Patches, hidden_dim]`
    pub fn forward(&self, x: Tensor<B, 3>, cond: Tensor<B, 2>) -> Tensor<B, 3> {
        let [b, _n, d] = x.dims();

        // --------------------------------------------------------------------
        // Step 1: Regress 6 Modulation Parameters from Condition Vector
        // --------------------------------------------------------------------
        // Project cond [B, D] -> [B, 6 * D] and unsqueeze to [B, 1, 6 * D] for sequence broadcasting
        let mod_params = self.ada_ln.forward(cond).unsqueeze_dim(1);

        // Slice the 6 parameters along dimension 2 (channels):
        let gamma1 = mod_params.clone().slice([0..b, 0..1, 0..d]); // Attention scale: [B, 1, D]
        let beta1 = mod_params.clone().slice([0..b, 0..1, d..d * 2]); // Attention shift: [B, 1, D]
        let alpha1 = mod_params.clone().slice([0..b, 0..1, d * 2..d * 3]); // Attention gate:  [B, 1, D]
        let gamma2 = mod_params.clone().slice([0..b, 0..1, d * 3..d * 4]); // MLP scale:        [B, 1, D]
        let beta2 = mod_params.clone().slice([0..b, 0..1, d * 4..d * 5]); // MLP shift:        [B, 1, D]
        let alpha2 = mod_params.clone().slice([0..b, 0..1, d * 5..d * 6]); // MLP gate:         [B, 1, D]

        // --------------------------------------------------------------------
        // Step 2: Modulated Multi-Head Self-Attention Sub-Layer
        // --------------------------------------------------------------------
        // 2a. Pre-LayerNorm: [B, N, D] -> [B, N, D]
        let x_norm1 = self.norm1.forward(x.clone());

        // 2b. Adaptive Modulation: scale by (1 + gamma1) and shift by beta1
        let x_mod1 = x_norm1 * (gamma1 + 1.0) + beta1;

        // 2c. Self-Attention: tokens attend to each other
        let mha_in = MhaInput::self_attn(x_mod1);
        let attn_out = self.attn.forward(mha_in).context;

        // 2d. Residual connection gated by alpha1: x = x + alpha1 * attn_out
        let x = x + attn_out * alpha1;

        // --------------------------------------------------------------------
        // Step 3: Modulated Feed-Forward (MLP) Sub-Layer
        // --------------------------------------------------------------------
        // 3a. Pre-LayerNorm: [B, N, D] -> [B, N, D]
        let x_norm2 = self.norm2.forward(x.clone());

        // 3b. Adaptive Modulation: scale by (1 + gamma2) and shift by beta2
        let x_mod2 = x_norm2 * (gamma2 + 1.0) + beta2;

        // 3c. 2-layer MLP with GELU non-linearity: [B, N, D] -> [B, N, 4D] -> [B, N, D]
        let mlp_out = self
            .mlp_fc2
            .forward(self.mlp_act.forward(self.mlp_fc1.forward(x_mod2)));

        // 3d. Residual connection gated by alpha2: x = x + alpha2 * mlp_out
        x + mlp_out * alpha2
    }
}
