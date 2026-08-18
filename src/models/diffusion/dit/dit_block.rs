use burn::module::{Module, ModuleDisplay};
use burn::nn::attention::{MhaInput, MultiHeadAttention, MultiHeadAttentionConfig};

use burn::nn::{Gelu, LayerNorm, LayerNormConfig, Linear, LinearConfig};

use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

#[derive(Module, Debug)]
pub struct DiTBlock<B: Backend> {
    norm1: LayerNorm<B>,
    attn: MultiHeadAttention<B>,
    norm2: LayerNorm<B>,
    mlp_fc1: Linear<B>,
    mlp_fc2: Linear<B>,
    mlp_act: Gelu,
    ada_ln: Linear<B>,
}

impl<B: Backend> DiTBlock<B> {
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
        let mlp_hidden = (hidden_dim as f64 * mlp_ratio) as usize;
        let fc1 = LinearConfig::new(hidden_dim, mlp_hidden).init(device);
        let fc2 = LinearConfig::new(mlp_hidden, hidden_dim).init(device);
        let act = Gelu::new();
        let ada_ln = LinearConfig::new(cond_dim, hidden_dim).init(device);

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

    pub fn forward(&self, x: Tensor<B, 3>, cond: Tensor<B, 2>) -> Tensor<B, 3> {
        let [b, n, d] = x.dims();

        let mod_params = self.ada_ln.forward(cond).unsqueeze_dim(1); //[B,1,6*D]
        let gamma1 = mod_params.clone().slice([0..b, 0..1, 0..d]);

        let beta1 = mod_params.clone().slice([0..b, 0..1, d..d * 2]);

        let alpha1 = mod_params.clone().slice([0..b, 0..1, d * 2..d * 3]);

        let gamma2 = mod_params.clone().slice([0..b, 0..1, d * 3..d * 4]);

        let beta2 = mod_params.clone().slice([0..b, 0..1, d * 4..d * 5]);

        let alpha2 = mod_params.clone().slice([0..b, 0..1, d * 5..d * 6]);

        let x_norm1 = self.norm1.forward(x.clone());


        let x_mod1 = x_norm1 * (gamma1 + 1.0) + beta1;

        let mha_in = MhaInput::self_attn(x_mod1);

        let attn_out = self.attn.forward(mha_in).context;

        let x = x+ attn_out*alpha1;

        let x_norm2 = self.norm2.forward(x.clone());

        let x_mod2 = x_norm2 * (gamma2 + 1.0) + beta2;

        let mlp_out = self.mlp_fc2.forward(self.mlp_act.forward(self.mlp_fc1.forward(x_mod2.clone())))

        let x = x + mlp_out *alpha2;

        x
    }
}
