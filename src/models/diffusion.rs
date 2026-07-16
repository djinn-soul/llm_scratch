//https://huggingface.co/learn/diffusion-course/en/unit1/3
//https://github.com/juraam/stable-diffusion-from-scratch
//https://www.kaggle.com/code/takihasan/stable-diffusion-from-scratch
//https://github.com/yousef-rafat/miniDiffusion
// https://lilianweng.github.io/posts/2021-07-11-diffusion-models/
// https://github.com/zelaki/Reading-Diffusion
// https://learnopencv.com/denoising-diffusion-probabilistic-models/
//https://sander.ai/
//https://stackoverflow.com/questions/75693493/why-the-text-embedding-or-image-embedding-generated-by-clip-model-is-768-%c3%97-n/79243065#79243065
// https://cdn.openai.com/papers/dall-e-2.pdf

pub mod attention;
pub mod cfg_training;
pub mod denoising_cnn;
pub mod denoising_cnn_5layers;
pub mod denoising_cnn_ops;
pub mod denoising_mlp;
pub mod denoising_model;
pub mod sampling;
pub mod scheduler;
pub mod time_embedding;
pub mod unet;

pub use crate::common::optimizers::MlpAdamOptimizer;
pub use attention::SpatialSelfAttention;
pub use cfg_training::{make_one_hot_cfg, one_hot_class};
pub use denoising_cnn::SimpleDenoisingCNN;
pub use denoising_cnn_5layers::SimpleDenoisingCNN5Layers;
pub use denoising_mlp::{Gradients, SimpleDenoisingMlp};
pub use denoising_model::DenoisingModel;
pub use sampling::{sample_ddpm, sample_ddpm_cond, sample_ddpm_from_noise};
pub use scheduler::BetaScheduler;
pub use time_embedding::get_time_embedding;
pub use unet::SimpleDenoisingUNet;
