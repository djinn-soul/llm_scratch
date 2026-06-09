//https://huggingface.co/learn/diffusion-course/en/unit1/3
//https://github.com/juraam/stable-diffusion-from-scratch
//https://www.kaggle.com/code/takihasan/stable-diffusion-from-scratch
//https://github.com/yousef-rafat/miniDiffusion

use anyhow::Result;
use candle_core::{DType, Device, Tensor};

pub struct BetaScheduler {
    pub steps: usize,
    pub betas: Tensor,
    pub alphas: Tensor,
    pub alphas_cumprod: Tensor,
    pub alphas_cumprod_prev: Tensor,
    pub sqrt_alphas_cumprod: Tensor,
    pub sqrt_one_minus_alphas_cumprod: Tensor,
    pub sigmas: Tensor,
}
impl BetaScheduler {
    pub fn new(steps: usize, beta_start: f64, beta_end: f64, device: &Device) -> Result<Self> {
        // 1. Generate betas as f32
        let mut betas_vec = Vec::with_capacity(steps);
        for i in 0..steps {
            let beta = beta_start + (beta_end - beta_start) * (i as f64) / ((steps - 1) as f64);
            betas_vec.push(beta as f32);
        }
        let betas = Tensor::new(betas_vec.as_slice(), device)?;

        // 2. alphas_t = 1.0 - beta_t (in F32)
        let alphas = Tensor::ones(steps, DType::F32, device)?.sub(&betas)?;

        // 3. Compute alphas_cumprod
        let mut alphas_cumprod_vec = Vec::with_capacity(steps);
        let mut cumprod = 1.0f32;
        let alphas_f32 = alphas.to_vec1::<f32>()?;
        for &alpha in &alphas_f32 {
            cumprod *= alpha;
            alphas_cumprod_vec.push(cumprod);
        }
        let alphas_cumprod = Tensor::new(alphas_cumprod_vec.as_slice(), device)?;

        // 4. Compute alphas_cumprod_prev (shifted by 1 step, t=0 is 1.0)
        let mut alphas_cumprod_prev_vec = Vec::with_capacity(steps);
        alphas_cumprod_prev_vec.push(1.0f32);
        for i in 0..(steps - 1) {
            alphas_cumprod_prev_vec.push(alphas_cumprod_vec[i]);
        }
        let alphas_cumprod_prev = Tensor::new(alphas_cumprod_prev_vec.as_slice(), device)?;

        // 5. Precompute square roots for forward noising
        let sqrt_alphas_cumprod = alphas_cumprod.sqrt()?;
        let one_minus_alpha_cumprod = Tensor::ones(steps, DType::F32, device)?.sub(&alphas_cumprod)?;
        let sqrt_one_minus_alphas_cumprod = one_minus_alpha_cumprod.sqrt()?;

        // 6. Compute sigmas (posterior standard deviation) for reverse process
        let mut sigmas_vec = Vec::with_capacity(steps);
        sigmas_vec.push(0.0f32); // t=0 is 0
        for t in 1..steps {
            let alpha_bar = alphas_cumprod_vec[t];
            let alpha_bar_prev = alphas_cumprod_prev_vec[t];
            let beta = betas_vec[t];
            let variance = ((1.0 - alpha_bar_prev) / (1.0 - alpha_bar)) * beta;
            sigmas_vec.push(variance.sqrt());
        }
        let sigmas = Tensor::new(sigmas_vec.as_slice(), device)?;

        Ok(Self {
            steps,
            betas,
            alphas,
            alphas_cumprod,
            alphas_cumprod_prev,
            sqrt_alphas_cumprod,
            sqrt_one_minus_alphas_cumprod,
            sigmas,
        })
    }
    
}
