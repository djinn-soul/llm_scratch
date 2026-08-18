use burn::config::Config;

#[derive(Config, Debug)]
pub struct DiTConfig {
    #[config(default = 28)]
    pub img_size: usize,

    #[config(default = 1)]
    pub in_channels: usize,

    #[config(default = 4)]
    pub patch_size: usize,
    #[config(default = 10)]
    pub num_classes: usize,
    #[config(default = 256)]
    pub hidden_dim: usize,
    #[config(default = 6)]
    pub depth: usize,
    #[config(default = 8)]
    pub num_heads: usize,
    #[config(default = 4.0)]
    pub mlp_ratio: f64,
}
