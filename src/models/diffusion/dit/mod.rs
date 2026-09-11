//! # Diffusion Transformer (DiT) Module
//!
//! An implementation of the Diffusion Transformer (DiT) model using the **Burn** deep learning framework.
//!
//! Reference:
//! - Paper: *"Scalable Diffusion Models with Transformers"* (Peebles & Xie, 2022) - <https://arxiv.org/abs/2212.09748>
//!
//! ### Key Concepts:
//! - **Patchification ([`PatchEmbed`])**: Turns continuous 2D images into a 1D sequence of patch tokens.
//! - **Adaptive Conditioning ([`DiTBlock`])**: Uses **adaLN-Zero** (Adaptive Layer Normalization) to inject
//!   timestep $t$ and class labels $c$ via dynamic scale ($\gamma$), shift ($\beta$), and gate ($\alpha$) vectors.
//! - **Reconstruction ([`unpatchify`])**: Inverts the patch sequence back into the spatial image grid.
//! - **Full Pipeline ([`DiffusionTransformer`])**: Stacks positional embeddings, multiple `DiTBlock`s, and final adaLN projections.

pub mod config;
pub mod dit;
pub mod dit_block;
pub mod patch_embed;

pub use config::DiTConfig;
pub use dit::DiffusionTransformer;
pub use dit_block::DiTBlock;
pub use patch_embed::{unpatchify, PatchEmbed};

