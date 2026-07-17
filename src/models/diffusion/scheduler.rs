use anyhow::{bail, Ok, Result};
use candle_core::{DType, Device, Tensor};

// DDPM BETA SCHEDULER
//
// This struct stores the fixed noise plan used by diffusion.
//
// Forward diffusion asks:
//
// "If I start from clean data x_0, how noisy should it look at timestep t?"
//
// beta_t is the small amount of new noise added at one step.
// alpha_t = 1 - beta_t is the amount of signal kept at that step.
// alpha_bar_t = alpha_0 * alpha_1 * ... * alpha_t is the total clean signal
// still surviving after many steps.
//
// The forward noising formula is:
//
// x_t = sqrt(alpha_bar_t) * x_0
//     + sqrt(1 - alpha_bar_t) * epsilon
//
// Example:
//
// x_0 = 2.0
// epsilon = -0.5
// alpha_bar_t = 0.81
//
// x_t = 0.90 * 2.0 + 0.435 * -0.5 = 1.5825
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
        // STEP 1: make the beta schedule.
        //
        // beta_start -> beta_end means noise is added gently at first and more
        // strongly near the end.
        //
        // Example with 4 steps:
        //
        // betas = [0.0001, 0.0067, 0.0134, 0.0200]
        let mut betas_vec = Vec::with_capacity(steps);
        for i in 0..steps {
            let beta = beta_start + (beta_end - beta_start) * (i as f64) / ((steps - 1) as f64);
            betas_vec.push(beta as f32);
        }
        let betas = Tensor::new(betas_vec.as_slice(), device)?;

        // STEP 2: convert noise amount into signal keep amount.
        //
        // alpha_t = 1 - beta_t.
        //
        // If beta_t = 0.02, then alpha_t = 0.98. That means this step keeps
        // about 98 percent of the previous signal and injects about 2 percent
        // new variance.
        let alphas = Tensor::ones(steps, DType::F32, device)?.sub(&betas)?;

        // STEP 3: compute cumulative signal keep rate.
        //
        // alpha_bar_t is the running product of alphas:
        //
        // [a, b, c] -> [a, a*b, a*b*c]
        //
        // Example:
        //
        // [0.99, 0.98, 0.97] -> [0.99, 0.9702, 0.941094]
        let mut alphas_cumprod_vec = Vec::with_capacity(steps);
        let mut cumprod = 1.0f32;
        let alphas_f32 = alphas.to_vec1::<f32>()?;
        for &alpha in &alphas_f32 {
            cumprod *= alpha;
            alphas_cumprod_vec.push(cumprod);
        }
        let alphas_cumprod = Tensor::new(alphas_cumprod_vec.as_slice(), device)?;

        // STEP 4: store previous alpha_bar for reverse sampling.
        //
        // Reverse sampling uses both alpha_bar_t and alpha_bar_{t-1}. For
        // t = 0, there is no previous step, so define previous alpha_bar as
        // 1.0: the original clean signal before any noise is added.
        let mut alphas_cumprod_prev_vec = Vec::with_capacity(steps);
        alphas_cumprod_prev_vec.push(1.0f32);
        for i in 0..(steps - 1) {
            alphas_cumprod_prev_vec.push(alphas_cumprod_vec[i]);
        }
        let alphas_cumprod_prev = Tensor::new(alphas_cumprod_prev_vec.as_slice(), device)?;

        // STEP 5: precompute coefficients for q(x_t | x_0).
        //
        // These two vectors are used directly in:
        //
        // x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        let sqrt_alphas_cumprod = alphas_cumprod.sqrt()?;
        let one_minus_alpha_cumprod =
            Tensor::ones(steps, DType::F32, device)?.sub(&alphas_cumprod)?;
        let sqrt_one_minus_alphas_cumprod = one_minus_alpha_cumprod.sqrt()?;

        // STEP 6: precompute reverse-process sigma.
        //
        // sigma_t is the standard deviation for sampling x_{t-1} from x_t.
        // t = 0 gets sigma 0 because the final step should not add fresh
        // random noise.
        let mut sigmas_vec = Vec::with_capacity(steps);
        sigmas_vec.push(0.0f32);
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

    pub fn add_noise(&self, x0: &Tensor, noise: &Tensor, t: &Tensor) -> Result<Tensor> {
        // Add the correct amount of noise for each sample's timestep.
        //
        // Shapes:
        //
        // x0    [batch][data_dim]
        // noise [batch][data_dim]
        // t     [batch]
        //
        // index_select gathers one coefficient per sample:
        //
        // t = [3, 7]
        // coeffs = [sqrt_alpha_bar_3, sqrt_alpha_bar_7]
        //
        // reshape(((), 1)) changes [batch] into [batch][1], so each scalar can
        // broadcast across every pixel/feature in that sample row.
        let sqrt_alpha_bar = self
            .sqrt_alphas_cumprod
            .index_select(t, 0)?
            .reshape(((), 1))?;

        let sqrt_one_minus_alpha_bar = self
            .sqrt_one_minus_alphas_cumprod
            .index_select(t, 0)?
            .reshape(((), 1))?;

        // Forward diffusion:
        //
        // clean part = x_0   * sqrt(alpha_bar_t)
        // noise part = noise * sqrt(1 - alpha_bar_t)
        // x_t        = clean part + noise part
        //
        // Row-level example with data_dim = 2:
        //
        // x0[0]    = [2.0, -1.0]
        // noise[0] = [0.3,  0.8]
        // coeffs   = sqrt_alpha_bar=0.9, sqrt_one_minus=0.435
        // xt[0]    = [2.0*0.9 + 0.3*0.435, -1.0*0.9 + 0.8*0.435]
        let xt = x0
            .broadcast_mul(&sqrt_alpha_bar)?
            .add(&noise.broadcast_mul(&sqrt_one_minus_alpha_bar)?)?;
        Ok(xt)
    }
    pub fn new_cosine(steps: usize, device: &Device) -> Result<Self> {
        if steps == 0 {
            bail!("cosine beta scheduler requires at least one step");
        }

        let s = 0.008f64;

        // f(t) = cos(((t/T + s) / (1 + s)) * pi/2)^2
        let f = |t_val: f64| {
            let num = (t_val / steps as f64) + s;
            let denom = 1.0 + s;
            let angle = (num / denom) * (std::f64::consts::PI / 2.0);
            angle.cos().powi(2)
        };

        let mut betas_vec = Vec::with_capacity(steps);
        for t in 0..steps {
            // Each discrete beta spans [t, t + 1]. Starting with f(1)
            // keeps beta_0 > 0 and prevents 0 / sqrt(0) at reverse step zero.
            let beta = (1.0 - f(t as f64 + 1.0) / f(t as f64)).min(0.999);
            betas_vec.push(beta as f32);
        }

        // Derive alpha_bar from the clipped betas actually used by the
        // process so all stored schedule coefficients remain consistent.
        let alphas_vec: Vec<f32> = betas_vec.iter().map(|beta| 1.0 - beta).collect();
        let mut alphas_cumprod_vec = Vec::with_capacity(steps);
        let mut alpha_bar = 1.0f32;
        for alpha in &alphas_vec {
            alpha_bar *= alpha;
            alphas_cumprod_vec.push(alpha_bar);
        }

        let betas = Tensor::new(betas_vec.as_slice(), device)?;
        let alphas = Tensor::new(alphas_vec.as_slice(), device)?;
        let alphas_cumprod = Tensor::new(alphas_cumprod_vec.as_slice(), device)?;

        let mut alphas_cumprod_prev_vec = Vec::with_capacity(steps);
        alphas_cumprod_prev_vec.push(1.0f32);
        for i in 0..(steps - 1) {
            alphas_cumprod_prev_vec.push(alphas_cumprod_vec[i]);
        }
        let alphas_cumprod_prev = Tensor::new(alphas_cumprod_prev_vec.as_slice(), device)?;

        let sqrt_alphas_cumprod = alphas_cumprod.sqrt()?;
        let sqrt_one_minus_alphas_cumprod = Tensor::ones(steps, DType::F32, device)?
            .sub(&alphas_cumprod)?
            .sqrt()?;

        // Calculate posterior standard deviations
        let mut sigmas_vec = Vec::with_capacity(steps);
        sigmas_vec.push(0.0f32);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_schedule_has_finite_reverse_boundary() -> Result<()> {
        let scheduler = BetaScheduler::new_cosine(100, &Device::Cpu)?;
        let betas = scheduler.betas.to_vec1::<f32>()?;
        let alphas = scheduler.alphas.to_vec1::<f32>()?;
        let alpha_bars = scheduler.alphas_cumprod.to_vec1::<f32>()?;

        // The first assertions guard the former t=0 NaN. The product loop then
        // verifies a subtler invariant: alpha_bar must be derived from the same
        // clipped betas stored in the scheduler.
        assert!(betas[0] > 0.0);
        assert!(alpha_bars[0] < 1.0);
        assert!((betas[0] / (1.0 - alpha_bars[0]).sqrt()).is_finite());
        assert!(betas.iter().all(|value| value.is_finite()));
        assert!(alpha_bars.iter().all(|value| value.is_finite()));

        let mut product = 1.0f32;
        for (alpha, alpha_bar) in alphas.iter().zip(alpha_bars.iter()) {
            product *= alpha;
            assert!((product - alpha_bar).abs() < 1e-6);
        }
        Ok(())
    }

    #[test]
    fn cosine_schedule_rejects_zero_steps() {
        assert!(BetaScheduler::new_cosine(0, &Device::Cpu).is_err());
    }
}
