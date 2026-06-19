use anyhow::{bail, Ok, Result};
use candle_core::{DType, Tensor};

use crate::models::diffusion::denoising_model::DenoisingModel;

// ADAM OPTIMIZER FOR ANY DENOISING MODEL
//
// This optimizer works with any model that implements the `DenoisingModel`
// trait. It maintains per-parameter Adam state (first moment m and second
// moment v) in a `Vec`, matching the parameter order returned by the model's
// `params()` method.
//
// Previously this optimizer was hard-coded to `SimpleDenoisingMlp` and its
// four named parameters (w1, b1, w2, b2). The generic version uses positional
// matching instead: param[i] ↔ state[i] ↔ grad[i].
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

/// Per-parameter Adam state: first moment (m) and second moment (v).
struct ParamState {
    m: Tensor,
    v: Tensor,
}

pub struct MlpAdamOptimizer {
    states: Vec<ParamState>,
    pub t: usize,
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
}

impl MlpAdamOptimizer {
    pub fn new(model: &dyn DenoisingModel, lr: f64) -> Result<Self> {
        // Every trainable parameter gets matching Adam state tensors.
        //
        // They start at zero because there is no gradient history before the
        // first training step.
        let params = model.params();
        let mut states = Vec::with_capacity(params.len());

        for p in &params {
            let device = p.device();
            states.push(ParamState {
                m: Tensor::zeros(p.dims(), DType::F32, device)?,
                v: Tensor::zeros(p.dims(), DType::F32, device)?,
            });
        }

        Ok(Self {
            states,
            t: 0,
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
        })
    }

    pub fn step(&mut self, model: &mut dyn DenoisingModel, grads: &[Tensor]) -> Result<()> {
        // t counts optimizer steps. Adam needs this for bias correction because
        // m and v start at zero and would otherwise be too small early on.
        self.t += 1;
        let t = self.t;
        let lr = self.lr;
        let beta1 = self.beta1;
        let beta2 = self.beta2;
        let eps = self.eps;

        // Get mutable parameter references. The order matches self.states and
        // the grads slice — all three are aligned by position.
        let mut params = model.params_mut();
        if params.len() != self.states.len() {
            bail!(
                "optimizer state count ({}) does not match model parameter count ({})",
                self.states.len(),
                params.len()
            );
        }
        if grads.len() != params.len() {
            bail!(
                "gradient count ({}) does not match model parameter count ({})",
                grads.len(),
                params.len()
            );
        }

        for ((param, state), grad) in params
            .iter_mut()
            .zip(self.states.iter_mut())
            .zip(grads.iter())
        {
            // First moment:
            //
            // m = beta1 * m + (1 - beta1) * grad
            //
            // This is a smoothed gradient direction. beta1 = 0.9 means:
            // keep 90 percent old memory, mix in 10 percent new gradient.
            let m_new = state
                .m
                .affine(beta1, 0.0)?
                .add(&grad.affine(1.0 - beta1, 0.0)?)?;

            // Second moment:
            //
            // v = beta2 * v + (1 - beta2) * grad^2
            //
            // This tracks typical squared gradient size per coordinate.
            let grad_sq = grad.sqr()?;
            let v_new = state
                .v
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
            **param = param.sub(&update)?;

            state.m = m_new;
            state.v = v_new;
        }

        Ok(())
    }
}
