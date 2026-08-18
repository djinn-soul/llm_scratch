use burn::module::{Module, Param};

use burn::nn::{
    Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig, Relu,
};

use burn::tensor::backend::Backend;
use burn::tensor::{Distribution, Tensor};

use crate::models::diffusion::dit::config::DiTConfig;
use crate::models::diffusion::dit::patch_embed;

use super::dit_block::DiTBlock;
use super::patch_embed::PatchEmbed;
use super::patch_embed::{patchify, unpatchify, PatchEmbed};

#[derive(Module, Debug)]
pub struct DiffusionTransformer<B: Backend> {
    patch_embed: PatchEmbed<B>,
    pos_embed: Param<Tensor<B, 3>>, // [1, num_patches, hidden_dim]
    // Conditioning MLP
    t_embede_fc1: Linear<B>,
    t_embede_fc2: Linear<B>,
    class_embed: Embedding<B>,

    // Transformer Blocks
    dit_blocks: Vec<DiTBlock<B>>,
    // Final Layer
    final_norm: LayerNorm<B>,
    final_adaln: Linear<B>, //[gamma , beta]
    final_proj: Linear<B>,  // hidden_dim-> patch_size* patch_size * in_channels
    config: DiTConfig,
}

impl<B: Backend> DiffusionTransformer<B> {
    pub fn new(config: DiTConfig, device: &B::Device) -> Self {
        let num_patches = (config.img_size / config.patch_size).pow(2);
        let patch_dim = config.patch_size * config.patch_size * config.in_channels;

        let cond_dim = config.hidden_dim;
        let patch_embed = PatchEmbed::new(
            config.in_channels,
            config.hidden_dim,
            config.patch_size,
            device,
        );

        let pos_embed = Param::from_tensor(Tensor::random(
            [1, num_patches, config.hidden_dim],
            Distribution::Normal(0.0, 0.02),
            device,
        ));

        let t_embed_fc1 = LinearConfig::new(config.hidden_dim, config.hidden_dim).init(device);

        let t_embed_fc2 = LinearConfig::new(config.hidden_dim, config.hidden_dim).init(device);

        let class_embed = EmbeddingConfig::new(config.num_classes, config.hidden_dim).init(device);

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

        let final_norm = LayerNormConfig::new(config.hidden_dim).init(device);

        let final_adaln = LinearConfig::new(config.hidden_dim, config.hidden_dim).init(device);

        let final_proj = LinearConfig::new(config.hidden_dim, patch_dim).init(device);

        Self {
            patch_embed,
            pos_embed,
            t_embede_fc1,
            t_embede_fc2,
            class_embed,
            dit_blocks: blocks,
            final_norm,
            final_adaln,
            final_proj,
            config,
        }
    }

    pub fn forward(
        &self,
        x_t: Tensor<B, 4>,
        t_emb: Tensor<B, 2>,
        class_labels: Tensor<B, 1, burn::tensor::Int>,
    ) -> Tensor<B, 4> {
        let [b, _c, _h, _w] = x_t.dims();
        let d = self.config.hidden_dim;
        // 1. Compute conditioning vector y = MLP(t) + ClassEmbedding(c)
        let t_cond = self
            .t_embede_fc2
            .forward(Relu::new().forward(self.t_embede_fc1.forward(t_emb)));

        let c_cond = self.class_embed.forward(class_labels);

        let cond = t_cond + c_cond; // [B, cond_dim]

        //2. Patchify + Positional Embedding
        let mut x = self.patch_embed.forward(x_t) + self.pos_embed.val(); //[B,N,D]

        //3. Pass through DIT Blocks
        for block in &self.dit_blocks {
            x = block.forward(x, cond.clone());
        }
        // 4.Final Layernorm + adaln Modulation

        let final_params = self.final_adaln.forward(cond).unsqueeze_dim(1);
        let gamma = final_params.clone().slice([0..b, 0..1, 0..d]);
        let beta = final_params.slice([0..b, 0..1, d..2 * d]);

        x = self.final_norm.forward(x);

        x = x * gamma + beta;

        let x_patches = self.final_proj.forward(x);
        unpatchify(
            x_patches,
            self.config.in_channels,
            self.config.img_size,
            self.config.img_size,
            self.config.patch_size,
        )
    }
}
