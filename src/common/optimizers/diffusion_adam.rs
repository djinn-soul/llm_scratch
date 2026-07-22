use anyhow::{bail, Ok, Result};
use candle_core::{DType, Tensor};

use crate::common::parameterized::Parameterized;

// ADAM OPTIMIZER FOR ANY PARAMETERIZED MODEL
//
// This optimizer works with any model implementing `Parameterized` — diffusion
// noise predictors, but equally a GPT block or a plain MLP. It maintains
// per-parameter Adam state (first moment m and second moment v) in a `Vec`,
// matching the parameter order returned by the model's `params()`.
//
// Previously this optimizer was hard-coded to `SimpleDenoisingMlp` and its four
// named parameters (w1, b1, w2, b2). Then it was widened to `DenoisingModel`,
// which still pinned it to the diffusion module. `Parameterized` is the
// narrowest contract it actually needs: read the weights, write them by name.
//
// Reads are positional (param[i] ↔ state[i] ↔ grad[i], because manual backprop
// produces gradients as an ordered Vec). Writes are by name via `set_param`,
// which is why `param_names` is captured at construction: a model whose
// parameter list changed shape underneath the optimizer fails loudly instead of
// updating whichever tensor now happens to sit at that index.
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

    /// Parameter names captured at construction, in `params()` order.
    ///
    /// Held so `step()` can write each update back by name. Re-reading them
    /// every step would defeat the purpose: the check below compares the
    /// model's current names against these, so a model that gained, lost, or
    /// reordered a parameter after the optimizer was built is rejected rather
    /// than silently paired with stale Adam moments.
    param_names: Vec<String>,

    pub t: usize,
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
}

impl MlpAdamOptimizer {
    pub fn new(model: &dyn Parameterized, lr: f64) -> Result<Self> {
        // Every trainable parameter gets matching Adam state tensors.
        //
        // They start at zero because there is no gradient history before the
        // first training step.
        let params = model.params();
        let param_names: Vec<String> = model.param_names().into_iter().map(String::from).collect();
        if param_names.len() != params.len() {
            bail!(
                "Model parameter count ({}) does not match optimizer state count ({})",
                param_names.len(),
                params.len()
            );
        }

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
            param_names,
            t: 0,
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
        })
    }

    pub fn step(&mut self, model: &dyn Parameterized, grads: &[Tensor]) -> Result<()> {
        // t counts optimizer steps. Adam needs this for bias correction because
        // m and v start at zero and would otherwise be too small early on.
        self.t += 1;
        let t = self.t;
        let lr = self.lr;
        let beta1 = self.beta1;
        let beta2 = self.beta2;
        let eps = self.eps;

        // Read the current parameter values. The order matches self.states and
        // the grads slice — all three are aligned by position.
        let params = model.params();
        if params.len() != self.states.len() {
            bail!(
                "optimizer state count ({}) does not match model parameter count ({})",
                self.states.len(),
                params.len()
            );
        }

        // The names captured at construction are the write targets. If the model
        // no longer reports the same names in the same order, the positional
        // pairing above is meaningless and every update would land on the wrong
        // tensor — with matching shapes, so nothing downstream would notice.
        let current_names = model.param_names();
        if current_names.len() != self.param_names.len()
            || current_names
                .iter()
                .zip(self.param_names.iter())
                .any(|(now, at_init)| now != at_init)
        {
            bail!(
                "model parameter names changed since the optimizer was built: {:?} != {:?}",
                current_names,
                self.param_names
            );
        }
        if grads.len() != params.len() {
            bail!(
                "gradient count ({}) does not match model parameter count ({})",
                grads.len(),
                params.len()
            );
        }

        for (((param, state), grad), name) in params
            .iter()
            .zip(self.states.iter_mut())
            .zip(grads.iter())
            .zip(self.param_names.iter())
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

            // `sub` allocates a fresh tensor, which `set_param` copies into the
            // parameter's storage in place. The model's own field observes the
            // write because it shares that storage with the VarMap entry, so
            // nothing needs reassigning here.
            model.set_param(name, &param.sub(&update)?)?;

            state.m = m_new;
            state.v = v_new;
        }

        Ok(())
    }
}

pub type DiffusionAdam = MlpAdamOptimizer;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::varstore;
    use candle_core::Device;
    use candle_nn::VarMap;

    struct TinyModel {
        varmap: VarMap,
        w: Tensor,
    }

    impl TinyModel {
        fn new(device: &Device) -> Result<Self> {
            let varmap = VarMap::new();
            let w = varstore::register(&varmap, "w", Tensor::new(&[1.0f32, 2.0], device)?)?;
            Ok(Self { varmap, w })
        }
    }

    impl Parameterized for TinyModel {
        fn varmap(&self) -> &VarMap {
            &self.varmap
        }
        fn params(&self) -> Vec<&Tensor> {
            vec![&self.w]
        }
        fn param_names(&self) -> Vec<&str> {
            vec!["w"]
        }
    }

    // The central claim of the VarMap design: an optimizer holding only a
    // `&dyn Parameterized` can move the weights, and the model's own field sees
    // it. If `register()` ever handed back a tensor that did not share storage
    // with the stored `Var`, training would run with a frozen model and no
    // error anywhere — this test is the guard against that.
    #[test]
    fn step_updates_weights_visible_through_model_field() -> Result<()> {
        let device = Device::Cpu;
        let model = TinyModel::new(&device)?;
        let mut optimizer = MlpAdamOptimizer::new(&model, 0.1)?;

        let before = model.w.copy()?.to_vec1::<f32>()?;
        let grads = vec![Tensor::new(&[1.0f32, 1.0], &device)?];
        optimizer.step(&model, &grads)?;
        let after = model.w.to_vec1::<f32>()?;

        assert_eq!(optimizer.t, 1);
        // First Adam step with bias correction moves each weight by ~lr.
        assert!(
            (before[0] - after[0]).abs() > 1e-3,
            "weight did not change: {before:?} -> {after:?}"
        );
        // Gradient is positive, so the weight must decrease.
        assert!(after[0] < before[0]);
        Ok(())
    }

    // A model whose parameter list drifted after the optimizer was built must be
    // rejected. Positional Adam state would otherwise be applied to whichever
    // tensor now sits at that index — shapes can still match, so nothing else
    // would catch it.
    #[test]
    fn step_rejects_renamed_parameters() -> Result<()> {
        let device = Device::Cpu;
        let model = TinyModel::new(&device)?;
        let mut optimizer = MlpAdamOptimizer::new(&model, 0.1)?;

        // Same shape and count, different name.
        struct Renamed(TinyModel);
        impl Parameterized for Renamed {
            fn varmap(&self) -> &VarMap {
                &self.0.varmap
            }
            fn params(&self) -> Vec<&Tensor> {
                vec![&self.0.w]
            }
            fn param_names(&self) -> Vec<&str> {
                vec!["renamed"]
            }
        }

        let grads = vec![Tensor::new(&[1.0f32, 1.0], &device)?];
        let result = optimizer.step(&Renamed(model), &grads);
        assert!(result.is_err(), "renamed parameter list must be rejected");
        Ok(())
    }
}
