use anyhow::{Ok, Result};
use candle_core::{DType, Tensor};

use crate::models::diffusion::denoising_mlp::{Gradients, SimpleDenoisingMlp};

// ADAM OPTIMIZER FOR THE MANUAL DIFFUSION MLP
//
// This optimizer is in common::optimizers because it is training policy, not
// model structure. It still knows about SimpleDenoisingMlp because this version
// is intentionally manual and updates that model's raw Tensor fields directly.
//
// SGD uses only the current gradient:
//
// param = param - lr * grad
//
// Adam keeps two extra running memories for every parameter:
//
// m = moving average of gradients
// v = moving average of squared gradients
//
// Simple intuition:
//
// - m remembers the recent direction of travel
// - v remembers how large/noisy each gradient coordinate usually is
// - dividing by sqrt(v) makes coordinates with huge gradients take smaller
//   steps and coordinates with tiny gradients take relatively larger steps
//
// That usually makes training smoother than plain SGD for neural networks.
pub struct MlpAdamOptimizer {
    pub mw1: Tensor,
    pub vw1: Tensor,
    pub mb1: Tensor,
    pub vb1: Tensor,
    pub mw2: Tensor,
    pub vw2: Tensor,
    pub mb2: Tensor,
    pub vb2: Tensor,
    pub t: usize,
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
}

impl MlpAdamOptimizer {
    pub fn new(mlp: &SimpleDenoisingMlp, lr: f64) -> Result<Self> {
        let device = mlp.w1.device();

        // Every trainable parameter gets matching Adam state tensors.
        //
        // w1 has mw1 and vw1
        // b1 has mb1 and vb1
        // w2 has mw2 and vw2
        // b2 has mb2 and vb2
        //
        // They start at zero because there is no gradient history before the
        // first training step.
        Ok(Self {
            mw1: Tensor::zeros(mlp.w1.dims(), DType::F32, device)?,
            vw1: Tensor::zeros(mlp.w1.dims(), DType::F32, device)?,
            mb1: Tensor::zeros(mlp.b1.dims(), DType::F32, device)?,
            vb1: Tensor::zeros(mlp.b1.dims(), DType::F32, device)?,
            mw2: Tensor::zeros(mlp.w2.dims(), DType::F32, device)?,
            vw2: Tensor::zeros(mlp.w2.dims(), DType::F32, device)?,
            mb2: Tensor::zeros(mlp.b2.dims(), DType::F32, device)?,
            vb2: Tensor::zeros(mlp.b2.dims(), DType::F32, device)?,
            t: 0,
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
        })
    }

    pub fn step(&mut self, mlp: &mut SimpleDenoisingMlp, grads: &Gradients) -> Result<()> {
        // t counts optimizer steps. Adam needs this for bias correction because
        // m and v start at zero and would otherwise be too small early on.
        self.t += 1;
        let t = self.t;
        let lr = self.lr;
        let beta1 = self.beta1;
        let beta2 = self.beta2;
        let eps = self.eps;

        let adam_update =
            |param: &mut Tensor, m: &mut Tensor, v: &mut Tensor, grad: &Tensor| -> Result<()> {
                // First moment:
                //
                // m = beta1 * m + (1 - beta1) * grad
                //
                // This is a smoothed gradient direction. beta1 = 0.9 means:
                // keep 90 percent old memory, mix in 10 percent new gradient.
                let m_new = m.affine(beta1, 0.0)?.add(&grad.affine(1.0 - beta1, 0.0)?)?;

                // Second moment:
                //
                // v = beta2 * v + (1 - beta2) * grad^2
                //
                // This tracks typical squared gradient size per coordinate.
                let grad_sq = grad.sqr()?;
                let v_new = v
                    .affine(beta2, 0.0)?
                    .add(&grad_sq.affine(1.0 - beta2, 0.0)?)?;

                // Bias correction:
                //
                // Since m and v start at zero, their first few values are
                // biased low. Dividing by (1 - beta^t) corrects that startup
                // bias.
                let bc1 = 1.0 - beta1.powi(t as i32);
                let bc2 = 1.0 - beta2.powi(t as i32);
                let m_hat = m_new.affine(1.0 / bc1, 0.0)?;
                let v_hat = v_new.affine(1.0 / bc2, 0.0)?;

                // Adam update:
                //
                // update = lr * m_hat / (sqrt(v_hat) + eps)
                // param  = param - update
                //
                // eps avoids division by zero when v_hat is extremely small.
                let num = m_hat.affine(lr, 0.0)?;
                let den = v_hat.sqrt()?.affine(1.0, eps)?;
                let update = num.div(&den)?;
                *param = param.sub(&update)?;
                *m = m_new;
                *v = v_new;
                Ok(())
            };

        adam_update(&mut mlp.w1, &mut self.mw1, &mut self.vw1, &grads.dw1)?;
        adam_update(&mut mlp.b1, &mut self.mb1, &mut self.vb1, &grads.db1)?;
        adam_update(&mut mlp.w2, &mut self.mw2, &mut self.vw2, &grads.dw2)?;
        adam_update(&mut mlp.b2, &mut self.mb2, &mut self.vb2, &grads.db2)?;
        Ok(())
    }
}
