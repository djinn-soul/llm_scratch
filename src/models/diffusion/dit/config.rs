use burn::config::Config;

/// Configuration hyperparameters for the Diffusion Transformer (DiT).
///
/// DiT replaces standard U-Net backbones in diffusion models with a Vision Transformer (ViT).
/// Key architecture introduced in Peebles & Xie (2022): "Scalable Diffusion Models with Transformers".
#[derive(Config, Debug)]
pub struct DiTConfig {
    /// Spatial resolution of input images (e.g., 28 for MNIST, 32 for CIFAR-10, 256 for ImageNet).
    /// Assumes square images: Height = Width = `img_size`.
    #[config(default = 28)]
    pub img_size: usize,

    /// Number of input channels (e.g., 1 for grayscale MNIST, 3 for RGB).
    #[config(default = 1)]
    pub in_channels: usize,

    /// Size of each square image patch (e.g., 4x4 or 2x2).
    /// The image is decomposed into `(img_size / patch_size)^2` sequence tokens.
    /// For 28x28 with patch_size=4, this yields (28/4)^2 = 7x7 = 49 tokens.
    #[config(default = 4)]
    pub patch_size: usize,

    /// Total number of distinct class categories for class-conditional generation (e.g., 10 for MNIST digits 0-9).
    #[config(default = 10)]
    pub num_classes: usize,

    /// Embedding dimension (hidden state size `d_model`) used across all transformer layers.
    #[config(default = 256)]
    pub hidden_dim: usize,

    /// Total number of stacked `DiTBlock` transformer layers in the model.
    #[config(default = 6)]
    pub depth: usize,

    /// Number of parallel attention heads in Multi-Head Attention (MHA).
    /// `hidden_dim` must be divisible by `num_heads` (head dimension = `hidden_dim / num_heads`).
    #[config(default = 8)]
    pub num_heads: usize,

    /// Expansion factor for the inner feed-forward network (MLP) hidden dimension:
    /// `mlp_hidden = hidden_dim * mlp_ratio` (typically 4.0).
    #[config(default = 4.0)]
    pub mlp_ratio: f64,
}

