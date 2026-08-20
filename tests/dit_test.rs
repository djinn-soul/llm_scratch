use burn::backend::ndarray::NdArray;
use burn::backend::Autodiff;
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::tensor::Tensor;

use llm_scratch_rs::models::diffusion::dit::dit_block::DiTBlock;
use llm_scratch_rs::models::diffusion::dit::patch_embed::{unpatchify, PatchEmbed};
use llm_scratch_rs::models::diffusion::dit::{DiTConfig, DiffusionTransformer};

type TestBackend = NdArray;
type TestAutodiffBackend = Autodiff<NdArray>;

#[test]
fn test_patch_embed_and_unpatchify_shapes() {
    let device = Default::default();
    let batch_size = 2;
    let in_channels = 1;
    let img_size = 28;
    let patch_size = 4;
    let hidden_dim = 64;

    let patch_embed: PatchEmbed<TestBackend> =
        PatchEmbed::new(in_channels, hidden_dim, patch_size, &device);

    let x: Tensor<TestBackend, 4> =
        Tensor::zeros([batch_size, in_channels, img_size, img_size], &device);
    let tokens: Tensor<TestBackend, 3> = patch_embed.forward(x);

    let expected_num_patches = (img_size / patch_size) * (img_size / patch_size); // 49
    assert_eq!(
        tokens.dims(),
        [batch_size, expected_num_patches, hidden_dim]
    );

    // Test unpatchify back to 4D
    let patch_dim = patch_size * patch_size * in_channels;
    let raw_patches: Tensor<TestBackend, 3> =
        Tensor::zeros([batch_size, expected_num_patches, patch_dim], &device);
    let reconstructed: Tensor<TestBackend, 4> =
        unpatchify(raw_patches, in_channels, img_size, img_size, patch_size);

    assert_eq!(
        reconstructed.dims(),
        [batch_size, in_channels, img_size, img_size]
    );
}

#[test]
fn test_dit_block_forward() {
    let device = Default::default();
    let batch_size = 2;
    let num_tokens = 49;
    let hidden_dim = 64;
    let num_heads = 4;
    let mlp_ratio = 4.0;
    let cond_dim = 64;

    let block: DiTBlock<TestBackend> =
        DiTBlock::new(hidden_dim, num_heads, mlp_ratio, cond_dim, &device);

    let x: Tensor<TestBackend, 3> = Tensor::zeros([batch_size, num_tokens, hidden_dim], &device);
    let cond: Tensor<TestBackend, 2> = Tensor::zeros([batch_size, cond_dim], &device);

    let out = block.forward(x, cond);
    assert_eq!(out.dims(), [batch_size, num_tokens, hidden_dim]);
}

#[test]
fn test_diffusion_transformer_forward_shape() {
    let device = Default::default();
    let config = DiTConfig {
        img_size: 28,
        in_channels: 1,
        patch_size: 4,
        num_classes: 10,
        hidden_dim: 64,
        depth: 2,
        num_heads: 4,
        mlp_ratio: 4.0,
    };

    let model: DiffusionTransformer<TestBackend> = DiffusionTransformer::new(config, &device);

    let batch_size = 3;
    let x_t: Tensor<TestBackend, 4> = Tensor::zeros([batch_size, 1, 28, 28], &device);
    let t_emb: Tensor<TestBackend, 2> = Tensor::zeros([batch_size, 64], &device);
    let class_labels: Tensor<TestBackend, 1, burn::tensor::Int> =
        Tensor::from_ints([0, 3, 9], &device);

    let pred_noise = model.forward(x_t, t_emb, class_labels);
    assert_eq!(pred_noise.dims(), [batch_size, 1, 28, 28]);
}

#[test]
fn test_diffusion_transformer_autodiff_step() {
    let device = Default::default();
    let config = DiTConfig {
        img_size: 28,
        in_channels: 1,
        patch_size: 4,
        num_classes: 10,
        hidden_dim: 32,
        depth: 1,
        num_heads: 2,
        mlp_ratio: 2.0,
    };

    let model: DiffusionTransformer<TestAutodiffBackend> =
        DiffusionTransformer::new(config, &device);

    let mut optimizer = AdamWConfig::new().init();

    let batch_size = 2;
    let x_t: Tensor<TestAutodiffBackend, 4> = Tensor::zeros([batch_size, 1, 28, 28], &device);
    let target_noise: Tensor<TestAutodiffBackend, 4> =
        Tensor::zeros([batch_size, 1, 28, 28], &device);
    let t_emb: Tensor<TestAutodiffBackend, 2> = Tensor::zeros([batch_size, 32], &device);
    let class_labels: Tensor<TestAutodiffBackend, 1, burn::tensor::Int> =
        Tensor::from_ints([1, 7], &device);

    // Forward
    let pred = model.forward(x_t, t_emb, class_labels);
    let loss = (pred - target_noise).powf_scalar(2.0).mean();

    // Backward
    let grads = GradientsParams::from_grads(loss.backward(), &model);

    // Optimizer step
    let updated_model = optimizer.step(1e-3, model, grads);

    // Verify model still runs after step
    let x_t_new: Tensor<TestAutodiffBackend, 4> = Tensor::zeros([batch_size, 1, 28, 28], &device);
    let t_emb_new: Tensor<TestAutodiffBackend, 2> = Tensor::zeros([batch_size, 32], &device);
    let class_labels_new: Tensor<TestAutodiffBackend, 1, burn::tensor::Int> =
        Tensor::from_ints([1, 7], &device);

    let pred_new = updated_model.forward(x_t_new, t_emb_new, class_labels_new);
    assert_eq!(pred_new.dims(), [batch_size, 1, 28, 28]);
}
