use burn::module::Module;
use burn::nn::{conv::Conv2d, conv::Conv2dConfig, PaddingConfig2d};
use burn::Tensor;

use burn::tensor::backend::Backend;

#[derive(Module, Debug)]
pub struct PatchEmbed<B: Backend> {
    pub proj: Conv2d<B>,
    patch_size: usize,
}

impl<B: Backend> PatchEmbed<B> {
    pub fn new(
        in_channels: usize,
        hidden_dim: usize,
        patch_size: usize,
        device: &B::Device,
    ) -> Self {
        let proj = Conv2dConfig::new([in_channels, hidden_dim], [patch_size, patch_size])
            .with_stride([patch_size, patch_size])
            .with_padding(PaddingConfig2d::Same)
            .init(device);
        Self { proj, patch_size }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 3> {
        let feat = self.proj.forward(x);
        let [b, hidden_dim, gh, gw] = feat.dims();

        feat.reshape([b, hidden_dim, gh * gw]).swap_dims(1, 2)
    }
}

pub fn unpatchify<B: Backend>(
    x: Tensor<B, 3>,
    channels: usize,
    h: usize,
    w: usize,
    patch_size: usize,
) -> Tensor<B, 4> {
    let [b, _num_patches, _patch_dim] = x.dims();
    let gh = h / patch_size;
    let gw = w / patch_size;

    x.reshape([b, gh, gw, channels, patch_size, patch_size])
        .swap_dims(2, 3)
        .swap_dims(1, 2)
        .swap_dims(3, 4)
        .reshape([b, channels, h, w])
}
