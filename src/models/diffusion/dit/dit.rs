use burn::module::{Module, Param};
use burn::nn::{
    Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig, Relu,
};
use burn::tensor::backend::Backend;
use burn::tensor::{Distribution, Tensor};

use super::config::DiTConfig;
use super::dit_block::DiTBlock;
use super::patch_embed::{unpatchify, PatchEmbed};

/// Complete Diffusion Transformer (DiT) architecture.
///
/// Implements the class-conditional Vision Transformer backbone for diffusion models
/// based on the seminal paper *"Scalable Diffusion Models with Transformers"* (Peebles & Xie, 2022).
///
/// ### Core High-Level Concept:
/// Traditional diffusion models (like DDPM and Stable Diffusion 1.x/2.x) use a U-Net with convolutional
/// downsampling and upsampling blocks. DiT replaces the entire U-Net backbone with a standard
/// Vision Transformer (ViT) operating on a flattened sequence of image patches, modulated by
/// diffusion time $t$ and class label $c$ at every single block.
///
/// ### Full Architecture Pipeline & Shape Evolution:
/// ```text
/// 1. Input Image:           x_t               Shape: [B, C, H, W]              (e.g., [64, 1, 28, 28])
///                                │
/// 2. Patchify + PosEmbed:   PatchEmbed(x_t)   Shape: [B, N, D]                 (e.g., [64, 49, 128])
///                                │            where N = (H/p)*(W/p) = 7*7 = 49
///                                ▼
/// 3. Conditioning:          t_emb [B, D] + class_labels [B]
///                                │
///                           MLP(t_emb) + ClassEmbed(c)
///                                │
///                           cond [B, D]       (Unified condition vector)
///                                │
///                                ▼
/// 4. Stack of DiT Blocks:   x = DiTBlock_k(x, cond) for k in 0..depth
///                                │            Shape remains: [B, 49, 128]
///                                ▼
/// 5. Final Modulation:      adaLN(cond) -> (gamma, beta)
///                           LayerNorm(x) * (1 + gamma) + beta
///                                │
/// 6. Linear Projection:     Linear(D -> p * p * C)
///                           Shape: [B, N, p*p*C]                              (e.g., [64, 49, 16])
///                                │
/// 7. Unpatchify:            Reconstruct spatial 2D grid
///                           Output Shape: [B, C, H, W]                        (e.g., [64, 1, 28, 28])
/// ```
#[derive(Module, Debug)]
pub struct DiffusionTransformer<B: Backend> {
    /// Convolutional patch embedding layer that converts raw image pixels into tokens
    patch_embed: PatchEmbed<B>,
    /// Learnable 1D spatial position embeddings `[1, Num_Patches, hidden_dim]`
    /// Wrapped in `Param` so Burn tracks it as a trainable parameter during optimization
    pos_embed: Param<Tensor<B, 3>>,
    /// Timestep MLP layer 1: projects raw sinusoidal embeddings
    t_embed_fc1: Linear<B>,
    /// Timestep MLP layer 2: produces final timestep condition representation
    t_embed_fc2: Linear<B>,
    /// Class label embedding table: maps discrete class IDs [0..num_classes) -> continuous vectors of size `hidden_dim`
    class_embed: Embedding<B>,

    /// Stack of transformer blocks, each equipped with adaLN-Zero adaptive normalization
    dit_blocks: Vec<DiTBlock<B>>,
    /// Final LayerNorm applied before projecting back to pixel space
    final_norm: LayerNorm<B>,
    /// Final adaptive linear layer predicting scale ($\gamma$) and shift ($\beta$) from the condition vector
    final_adaln: Linear<B>,
    /// Final linear projection mapping hidden dimension $D$ to flat patch pixels $p^2 \cdot C$
    final_proj: Linear<B>,
    /// Stored hyperparameter configuration
    config: DiTConfig,
}

impl<B: Backend> DiffusionTransformer<B> {
    /// Instantiates a new `DiffusionTransformer` module initialized on the specified device.
    ///
    /// # Parameter Calculations:
    /// - For $28 \times 28$ image and patch size $p = 4$:
    ///   - Grid dimensions: $G_h = 28 / 4 = 7$, $G_w = 28 / 4 = 7$
    ///   - Number of tokens $N = 7 \times 7 = 49$
    ///   - Flat patch dimension $= 4 \times 4 \times 1 = 16$ pixels
    ///
    /// # Arguments
    /// * `config` - Hyperparameter configuration ([DiTConfig])
    /// * `device` - Compute device (CPU via `NdArray`, GPU via `Wgpu` or `Rocm`)
    pub fn new(config: DiTConfig, device: &B::Device) -> Self {
        // Compute total number of patches: (H/p) * (W/p)
        let num_patches = (config.img_size / config.patch_size).pow(2);
        // Compute total values per patch: p * p * C (e.g. 4 * 4 * 1 = 16 for MNIST)
        let patch_dim = config.patch_size * config.patch_size * config.in_channels;

        // The conditioning dimension matches the model's hidden dimension
        let cond_dim = config.hidden_dim;

        // Initialize convolutional patch projection: [B, C, H, W] -> [B, N, D]
        let patch_embed = PatchEmbed::new(
            config.in_channels,
            config.hidden_dim,
            config.patch_size,
            device,
        );

        // Learnable 1D spatial position embeddings initialized from N(0, 0.02^2)
        // Shape: [1, num_patches, hidden_dim] — broadcastable across any batch size B
        let pos_embed = Param::from_tensor(Tensor::random(
            [1, num_patches, config.hidden_dim],
            Distribution::Normal(0.0, 0.02),
            device,
        ));

        // 2-layer MLP for timestep embedding projection: Linear -> ReLU -> Linear
        // Allows non-linear transformation of sinusoidal frequencies into the latent condition space
        let t_embed_fc1 = LinearConfig::new(config.hidden_dim, config.hidden_dim).init(device);
        let t_embed_fc2 = LinearConfig::new(config.hidden_dim, config.hidden_dim).init(device);

        // Class embedding table: look up table of shape [num_classes, hidden_dim]
        let class_embed = EmbeddingConfig::new(config.num_classes, config.hidden_dim).init(device);

        // Build the sequential stack of transformer blocks (depth L)
        let mut blocks = Vec::new();
        for _ in 0..config.depth {
            blocks.push(DiTBlock::new(
                config.hidden_dim,
                config.num_heads,
                config.mlp_ratio,
                cond_dim,
                device,
            ));
        }

        // Final output layers:
        // 1. LayerNorm across hidden_dim
        let final_norm = LayerNormConfig::new(config.hidden_dim).init(device);
        // 2. Linear layer mapping cond [B, D] -> 2 * D (gamma and beta scale/shift parameters)
        let final_adaln = LinearConfig::new(config.hidden_dim, 2 * config.hidden_dim).init(device);
        // 3. Linear projection mapping hidden_dim -> patch_dim (e.g. 128 -> 16)
        let final_proj = LinearConfig::new(config.hidden_dim, patch_dim).init(device);

        Self {
            patch_embed,
            pos_embed,
            t_embed_fc1,
            t_embed_fc2,
            class_embed,
            dit_blocks: blocks,
            final_norm,
            final_adaln,
            final_proj,
            config,
        }
    }

    /// Retrieves the continuous embedding vector for a given discrete class label.
    ///
    /// # Parameter Mechanics:
    /// Queries the learned embedding lookup table of shape `[num_classes, hidden_dim]`.
    /// Returns a continuous vector `[1, hidden_dim]` that can be manipulated via vector arithmetic
    /// (e.g. linear blending $\mathbf{e} = (1-\lambda)\mathbf{e}_A + \lambda\mathbf{e}_B$ for Latent Class Morphing).
    pub fn get_class_embedding(&self, class_id: usize, device: &B::Device) -> Tensor<B, 2> {
        let label_tensor: Tensor<B, 1, burn::tensor::Int> =
            Tensor::from_ints([class_id as i32], device);
        self.class_embed
            .forward(label_tensor.unsqueeze_dim(1))
            .reshape([1, self.config.hidden_dim])
    }

    /// Executes the forward pass of DiT to predict the noise added to the image.
    ///
    /// # Detailed Step-by-Step Flow:
    /// 1. **Conditioning Vector Construction**:
    ///    - `t_cond = FC2(ReLU(FC1(t_emb)))` `[B, D]`
    ///    - `c_cond = Embedding(c)` `[B, D]`
    ///    - `cond = t_cond + c_cond` `[B, D]`
    /// 2. **Patch Embedding & Positional Addition**:
    ///    - Image `x_t` `[B, C, H, W]` is patchified to tokens `[B, N, D]`
    ///    - Add learned positional embeddings `pos_embed` `[1, N, D]` (broadcast over batch `B`)
    /// 3. **Transformer Processing**:
    ///    - Tokens pass sequentially through all `depth` DiTBlock layers, each modulated by `cond`.
    /// 4. **Final adaLN & Pixel Reconstruction**:
    ///    - `cond` -> `[gamma, beta]` via `final_adaln`
    ///    - `x = LayerNorm(x) * (1 + gamma) + beta`
    ///    - Linearly project tokens to raw patch pixels `[B, N, p^2 * C]`
    ///    - Call `unpatchify` to permute and reshape patches back into the 4D image grid `[B, C, H, W]`.
    ///
    /// Accepts raw continuous vectors rather than integer IDs, directly supporting CFG extrapolation
    /// and continuous latent morphing between multiple classes.
    pub fn forward_with_cond_vec(
        &self,
        x_t: Tensor<B, 4>,
        t_emb: Tensor<B, 2>,
        c_cond: Tensor<B, 2>,
    ) -> Tensor<B, 4> {
        let [b, _c, _h, _w] = x_t.dims();
        let d = self.config.hidden_dim;

        // Step 1: Project sinusoidal Fourier timestep embeddings [B, D] through a 2-layer MLP.
        // Linear -> ReLU -> Linear projects rigid mathematical frequencies into the semantic latent space.
        let t_cond = self
            .t_embed_fc2
            .forward(Relu::new().forward(self.t_embed_fc1.forward(t_emb)));

        // Step 1b: Merge time and class signals via element-wise addition [B, D].
        // Both time and class condition the diffusion flow equally as a single joint anchor.
        let cond = t_cond + c_cond;

        // Step 2: Linearly project 2D image patches into 1D token sequences [B, N, D] via PatchEmbed.
        // Self-attention is permutation-invariant (order-blind), so we add learned 1D spatial
        // position embeddings pos_embed [1, N, D] so tokens retain spatial grid coordinates.
        let mut x = self.patch_embed.forward(x_t) + self.pos_embed.val();

        // Step 3: Pass tokens sequentially through the stack of L DiTBlocks.
        // Each block internally modulates attention and MLP pathways using adaLN-Zero driven by cond [B, D].
        for block in &self.dit_blocks {
            x = block.forward(x, cond.clone());
        }

        // Step 4: Regress final adaptive LayerNorm modulation parameters [gamma, beta] from cond [B, D].
        // Projects cond [B, D] -> [B, 1, 2D], then slices into multiplicative scale and additive shift.
        let final_params = self.final_adaln.forward(cond).unsqueeze_dim(1);
        let gamma = final_params.clone().slice([0..b, 0..1, 0..d]); // [B, 1, D]
        let beta = final_params.slice([0..b, 0..1, d..2 * d]); // [B, 1, D]

        // Step 4b: Apply adaptive LayerNorm: normalize token vectors across hidden_dim, scale by (1 + gamma)
        // to preserve standard LayerNorm identity at gamma=0, and shift by beta.
        x = self.final_norm.forward(x) * (gamma + 1.0) + beta;

        // Step 5: Linearly project transformer tokens [B, N, D] to raw patch pixels [B, N, p^2 * C].
        // For MNIST with patch_size=4 and C=1, each token expands into 16 raw scalar pixel values.
        let x_patches = self.final_proj.forward(x);

        // Step 5b: Invert patchification: reshape and permute tokens back into standard 2D spatial grid [B, C, H, W].
        // The output matches the exact shape of input x_t and represents predicted noise epsilon_hat.
        unpatchify(
            x_patches,
            self.config.in_channels,
            self.config.img_size,
            self.config.img_size,
            self.config.patch_size,
        )
    }

    /// Executes the standard forward pass of DiT by mapping discrete class IDs to embeddings.
    pub fn forward(
        &self,
        x_t: Tensor<B, 4>,
        t_emb: Tensor<B, 2>,
        class_labels: Tensor<B, 1, burn::tensor::Int>,
    ) -> Tensor<B, 4> {
        let [b, _c, _h, _w] = x_t.dims();
        let d = self.config.hidden_dim;

        // Look up class label embeddings: [B] -> [B, 1, D] -> [B, D]
        let c_cond = self
            .class_embed
            .forward(class_labels.unsqueeze_dim(1))
            .reshape([b, d]);

        self.forward_with_cond_vec(x_t, t_emb, c_cond)
    }
}
