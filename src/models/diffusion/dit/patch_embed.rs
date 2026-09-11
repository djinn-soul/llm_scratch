use burn::module::Module;
use burn::nn::{conv::Conv2d, conv::Conv2dConfig, PaddingConfig2d};
use burn::Tensor;
use burn::tensor::backend::Backend;

/// 2D Image Patch Embedding layer.
///
/// Vision Transformers (ViT) and Diffusion Transformers (DiT) cannot directly consume
/// 2D grid pixels because standard self-attention operates on 1D token sequences.
///
/// `PatchEmbed` decomposes a 2D image into non-overlapping $p \times p$ square patches
/// and linearly projects each patch into a $D$-dimensional continuous latent token:
///
/// ```text
/// Image [B, C, H, W]
///      │
///      ▼ (Partition into non-overlapping p x p patches)
/// Grid [B, C, (H/p), p, (W/p), p]
///      │
///      ▼ (Linearly project flat pixels p*p*C to hidden dimension D)
/// Tokens [B, Num_Patches, Hidden_Dim]  where Num_Patches = (H/p) * (W/p)
/// ```
///
/// ### Why Use Conv2d Instead of Slicing & Matrix Multiplication?
/// Mathematically, sliding a 2D convolutional filter with:
/// - `kernel_size = [patch_size, patch_size]`
/// - `stride = [patch_size, patch_size]`
///
/// computes the inner product between the filter weights and each non-overlapping
/// $p \times p$ image patch in a single, highly optimized hardware kernel on modern GPUs.
#[derive(Module, Debug)]
pub struct PatchEmbed<B: Backend> {
    /// 2D convolution projecting `[C_in, H_patch, W_patch]` directly to `hidden_dim`
    pub proj: Conv2d<B>,
    /// Size of each square patch $p$
    patch_size: usize,
}

impl<B: Backend> PatchEmbed<B> {
    /// Initializes the patch embedding convolution layer.
    ///
    /// # Arguments
    /// * `in_channels` - Input image channels (e.g. 1 for MNIST, 3 for RGB)
    /// * `hidden_dim` - Target embedding dimension $D$ (`d_model`)
    /// * `patch_size` - Size of each square patch $p$ (e.g. 4)
    /// * `device` - Compute device (CPU/GPU)
    pub fn new(
        in_channels: usize,
        hidden_dim: usize,
        patch_size: usize,
        device: &B::Device,
    ) -> Self {
        // Non-overlapping convolution:
        // Input channels: in_channels, Output channels: hidden_dim
        // Kernel: [p, p], Stride: [p, p] (moves by exactly one patch size with 0 overlap)
        let proj = Conv2dConfig::new([in_channels, hidden_dim], [patch_size, patch_size])
            .with_stride([patch_size, patch_size])
            .with_padding(PaddingConfig2d::Same)
            .init(device);
        Self { proj, patch_size }
    }

    /// Forward pass: converts 4D image tensor `[B, C, H, W]` to 3D token sequence `[B, N, D]`.
    ///
    /// ### Concrete Shape Trace (MNIST Example: B=64, C=1, H=28, W=28, p=4, D=128):
    /// 1. Input image $x$: `[64, 1, 28, 28]`
    /// 2. Conv2d output: `[64, 128, 7, 7]` (where $28 / 4 = 7$ grid cells per axis)
    /// 3. Reshape grid: `[64, 128, 49]` (where $7 \times 7 = 49$ total tokens)
    /// 4. Swap dims (1, 2): `[64, 49, 128]` (matching standard Transformer sequence layout `[B, N, D]`)
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 3> {
        let feat = self.proj.forward(x);
        let [b, hidden_dim, gh, gw] = feat.dims();

        // Flatten spatial grid (gh * gw) into sequence length N, then transpose to [B, N, D]
        feat.reshape([b, hidden_dim, gh * gw]).swap_dims(1, 2)
    }
}

/// Converts a sequence of flat patch tokens back to a standard 2D image tensor.
///
/// This is the exact mathematical inverse of patchification, used at the final layer of DiT
/// to assemble the predicted noise tensor $\hat{\epsilon} \in \mathbb{R}^{B \times C \times H \times W}$.
///
/// ### Step-by-Step Dimension Permutation Trace:
/// Let $B=64, N=49, \text{patch\_dim}=16$ (since $p=4, C=1 \implies 4 \times 4 \times 1 = 16$), $H=28, W=28$:
///
/// 1. **Input**: `[B, N, patch_dim]` $\to$ `[64, 49, 16]`
/// 2. **Reshape to Grid + Patch**:
///    - Shape: `[B, gh, gw, C, p, p]` $\to$ `[64, 7, 7, 1, 4, 4]`
/// 3. **Swap Dims (2, 3)** (swap grid width with channel):
///    - Shape: `[B, gh, C, gw, p, p]` $\to$ `[64, 7, 1, 7, 4, 4]`
/// 4. **Swap Dims (1, 2)** (move channel to standard dimension 1):
///    - Shape: `[B, C, gh, gw, p, p]` $\to$ `[64, 1, 7, 7, 4, 4]`
/// 5. **Swap Dims (3, 4)** (interleave grid width with patch height):
///    - Shape: `[B, C, gh, p, gw, p]` $\to$ `[64, 1, 7, 4, 7, 4]`
/// 6. **Reshape to Final 4D Image**:
///    - Combine $(gh \times p = 7 \times 4 = 28)$ and $(gw \times p = 7 \times 4 = 28)$
///    - Output Shape: `[B, C, H, W]` $\to$ `[64, 1, 28, 28]`
pub fn unpatchify<B: Backend>(
    x: Tensor<B, 3>,
    channels: usize,
    h: usize,
    w: usize,
    patch_size: usize,
) -> Tensor<B, 4> {
    let [b, _num_patches, _patch_dim] = x.dims();
    let gh = h / patch_size; // Grid height (patches along Y-axis: e.g. 28 / 4 = 7)
    let gw = w / patch_size; // Grid width  (patches along X-axis: e.g. 28 / 4 = 7)

    // Deconstruct and re-interleave spatial dimensions
    x.reshape([b, gh, gw, channels, patch_size, patch_size])
        .swap_dims(2, 3)
        .swap_dims(1, 2)
        .swap_dims(3, 4)
        .reshape([b, channels, h, w])
}


