use std::collections::HashMap;
use std::path::Path;

use anyhow::{bail, Ok, Result};
use candle_core::{safetensors, DType, Device, Tensor};

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

            // `.detach()` is what keeps this loop O(1) in memory rather than
            // O(steps).
            //
            // Candle records a backward graph on every tensor derived from a
            // `Var`, and the model's parameters are `Var`s. So `m_new` — built
            // from `state.m` and `grad`, both of which track — carries a graph
            // node holding *clones* of its inputs. Storing it back into
            // `state.m` therefore keeps the previous `state.m` alive, which
            // keeps the one before it alive, and so on: an unbounded chain
            // growing by one link per optimizer step, freed only at process
            // exit. Over a long run that is gigabytes of retained moments.
            //
            // Nothing ever walks that graph — this optimizer is fed gradients
            // from the model's hand-written `backward()`, not from candle's
            // autograd. `detach()` shares the same storage but drops the op,
            // so the history is released while the values are untouched.
            state.m = m_new.detach();
            state.v = v_new.detach();
        }

        Ok(())
    }
}

// ════════════════════════════════════════════════════════════════════════════
// RESUMABLE OPTIMIZER STATE
// ════════════════════════════════════════════════════════════════════════════
//
// Restoring model weights alone does not resume training. Adam's behaviour is
// determined just as much by `t` and the per-parameter moments:
//
//   - `m` / `v` are the smoothed gradient direction and magnitude. Resuming with
//     them zeroed makes the first steps after a restart effectively
//     unpreconditioned, which shows up as a visible loss spike.
//   - `t` drives bias correction, `1 / (1 - beta^t)`. Resetting it to 0 re-applies
//     the startup correction — at t=1 that scales the update by ~10x for the
//     default betas — right when the weights are already trained.
//
// Both files are keyed by parameter name, for the same reason the model
// checkpoint is: positional keys would silently pair a parameter with another
// layer's moments after any architecture edit, with shapes still matching.
impl MlpAdamOptimizer {
    /// Key holding the four scalar hyperparameters, in `HPARAM_ORDER`.
    const HPARAMS_KEY: &'static str = "hparams";
    /// Key holding the step counter `t`.
    const STEP_KEY: &'static str = "t";

    fn moment_keys(name: &str) -> (String, String) {
        (format!("m.{name}"), format!("v.{name}"))
    }

    /// Write `t`, the hyperparameters, and every `m`/`v` pair to a SafeTensors file.
    ///
    /// Tensors are moved to CPU so a checkpoint written on GPU can be resumed on
    /// a machine without one.
    pub fn save_checkpoint(&self, path: impl AsRef<Path>) -> Result<()> {
        if self.param_names.len() != self.states.len() {
            bail!(
                "optimizer holds {} names for {} states",
                self.param_names.len(),
                self.states.len()
            );
        }

        let cpu = Device::Cpu;
        let mut tensors: HashMap<String, Tensor> = HashMap::new();

        // `t` is stored as a one-element tensor because SafeTensors has no
        // scalar type. i64 is candle's widest integer dtype (there is no u64)
        // and covers any realistic step count.
        tensors.insert(
            Self::STEP_KEY.to_string(),
            Tensor::new(&[self.t as i64], &cpu)?,
        );

        // Hyperparameters travel with the state. Without them a resume could
        // silently apply different betas to moments accumulated under the old
        // ones, which changes the update rule while looking like a clean restart.
        tensors.insert(
            Self::HPARAMS_KEY.to_string(),
            Tensor::new(&[self.lr, self.beta1, self.beta2, self.eps], &cpu)?,
        );

        for (name, state) in self.param_names.iter().zip(self.states.iter()) {
            let (m_key, v_key) = Self::moment_keys(name);
            tensors.insert(m_key, state.m.to_device(&cpu)?);
            tensors.insert(v_key, state.v.to_device(&cpu)?);
        }

        // Write to a sibling temp file, then rename. Checkpoints are written
        // mid-training, so an interrupted write to the final path would leave a
        // truncated file that the next resume would happily start parsing.
        // Rename within a directory is atomic, so the destination is either the
        // previous checkpoint or the complete new one.
        let path = path.as_ref();
        let temp = path.with_extension("safetensors.tmp");
        safetensors::save(&tensors, &temp)?;
        // Windows rename fails if the destination exists, unlike POSIX.
        if path.exists() {
            std::fs::remove_file(path)?;
        }
        std::fs::rename(&temp, path)?;
        Ok(())
    }

    /// Restore `t` and the `m`/`v` moments saved by `save_checkpoint`.
    ///
    /// The optimizer must already be constructed against the same model, so the
    /// parameter names and shapes are known before anything is read.
    ///
    /// `beta1`, `beta2` and `eps` are restored, because the stored moments only
    /// mean what they mean under those values. `lr` is deliberately NOT
    /// restored: changing the learning rate on resume is a normal thing to do,
    /// and silently reverting it to the checkpointed value would override an
    /// explicit choice made by the caller.
    pub fn load_checkpoint(&mut self, path: impl AsRef<Path>, device: &Device) -> Result<()> {
        let loaded = safetensors::load(path, device)?;

        let t_tensor = loaded
            .get(Self::STEP_KEY)
            .ok_or_else(|| anyhow::anyhow!("optimizer checkpoint is missing step counter 't'"))?;
        let t = t_tensor.flatten_all()?.to_vec1::<i64>()?;
        let t = *t
            .first()
            .ok_or_else(|| anyhow::anyhow!("optimizer checkpoint has an empty step counter"))?;
        if t < 0 {
            bail!("optimizer checkpoint has a negative step counter ({t})");
        }

        let hparams = loaded
            .get(Self::HPARAMS_KEY)
            .ok_or_else(|| anyhow::anyhow!("optimizer checkpoint is missing hyperparameters"))?
            .flatten_all()?
            .to_vec1::<f64>()?;
        if hparams.len() != 4 {
            bail!(
                "optimizer checkpoint hyperparameters have {} entries, expected 4",
                hparams.len()
            );
        }

        // Read every moment into a staging vector before mutating `self`. A
        // checkpoint that fails validation half-way through would otherwise
        // leave the optimizer holding a mix of old and new state, which is worse
        // than either.
        let mut restored = Vec::with_capacity(self.states.len());
        for (name, state) in self.param_names.iter().zip(self.states.iter()) {
            let (m_key, v_key) = Self::moment_keys(name);

            let m = loaded
                .get(&m_key)
                .ok_or_else(|| anyhow::anyhow!("optimizer checkpoint is missing moment {m_key}"))?;
            let v = loaded
                .get(&v_key)
                .ok_or_else(|| anyhow::anyhow!("optimizer checkpoint is missing moment {v_key}"))?;

            if m.dims() != state.m.dims() || v.dims() != state.v.dims() {
                bail!(
                    "optimizer checkpoint shape mismatch for {name}: got m {:?} / v {:?}, expected {:?}",
                    m.dims(),
                    v.dims(),
                    state.m.dims()
                );
            }

            restored.push(ParamState {
                m: m.clone(),
                v: v.clone(),
            });
        }

        self.states = restored;
        self.t = t as usize;
        self.beta1 = hparams[1];
        self.beta2 = hparams[2];
        self.eps = hparams[3];
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

    fn temp_path(tag: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "llm-scratch-opt-{tag}-{}.safetensors",
            std::process::id()
        ))
    }

    // Round trip must restore `t` and both moments exactly. Gradients vary per
    // step so `m` and `v` end up different from each other — with a constant
    // gradient they converge to similar values and a swapped m/v key would slip
    // through.
    #[test]
    fn checkpoint_round_trip_restores_step_and_moments() -> Result<()> {
        let device = Device::Cpu;
        let model = TinyModel::new(&device)?;
        let mut saved = MlpAdamOptimizer::new(&model, 1e-3)?;

        for step in 1..=5 {
            let g = step as f32 * 0.5;
            saved.step(&model, &[Tensor::new(&[g, -g], &device)?])?;
        }
        assert_eq!(saved.t, 5);

        let path = temp_path("roundtrip");
        saved.save_checkpoint(&path)?;

        let fresh_model = TinyModel::new(&device)?;
        let mut loaded = MlpAdamOptimizer::new(&fresh_model, 1e-3)?;
        assert_eq!(loaded.t, 0);

        loaded.load_checkpoint(&path, &device)?;
        std::fs::remove_file(&path).ok();

        assert_eq!(loaded.t, 5);
        assert_eq!(loaded.beta1, saved.beta1);
        assert_eq!(loaded.beta2, saved.beta2);
        assert_eq!(loaded.eps, saved.eps);

        let m_saved = saved.states[0].m.flatten_all()?.to_vec1::<f32>()?;
        let m_loaded = loaded.states[0].m.flatten_all()?.to_vec1::<f32>()?;
        let v_saved = saved.states[0].v.flatten_all()?.to_vec1::<f32>()?;
        let v_loaded = loaded.states[0].v.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(m_saved, m_loaded);
        assert_eq!(v_saved, v_loaded);
        // Guard the guard: if m and v were equal, this test could not detect a
        // swapped key pair.
        assert_ne!(m_saved, v_saved);
        Ok(())
    }

    // Resuming must reproduce the trajectory of uninterrupted training. This is
    // the property that actually matters — equal tensors on disk mean nothing if
    // the next step diverges.
    #[test]
    fn resumed_optimizer_matches_uninterrupted_training() -> Result<()> {
        let device = Device::Cpu;
        let grad = |s: usize| -> Result<Vec<Tensor>> {
            let g = s as f32 * 0.25;
            Ok(vec![Tensor::new(&[g, -g], &device)?])
        };

        // Straight through, six steps.
        let continuous_model = TinyModel::new(&device)?;
        let mut continuous = MlpAdamOptimizer::new(&continuous_model, 1e-2)?;
        for s in 1..=6 {
            continuous.step(&continuous_model, &grad(s)?)?;
        }

        // Three steps, checkpoint, restore into a fresh optimizer, three more.
        let resumed_model = TinyModel::new(&device)?;
        let mut first_half = MlpAdamOptimizer::new(&resumed_model, 1e-2)?;
        for s in 1..=3 {
            first_half.step(&resumed_model, &grad(s)?)?;
        }
        let path = temp_path("resume");
        first_half.save_checkpoint(&path)?;

        let mut second_half = MlpAdamOptimizer::new(&resumed_model, 1e-2)?;
        second_half.load_checkpoint(&path, &device)?;
        std::fs::remove_file(&path).ok();
        for s in 4..=6 {
            second_half.step(&resumed_model, &grad(s)?)?;
        }

        let expected = continuous_model.w.flatten_all()?.to_vec1::<f32>()?;
        let actual = resumed_model.w.flatten_all()?.to_vec1::<f32>()?;
        for (e, a) in expected.iter().zip(actual.iter()) {
            assert!(
                (e - a).abs() < 1e-6,
                "resumed training diverged: {expected:?} vs {actual:?}"
            );
        }
        Ok(())
    }

    // A checkpoint from a differently shaped model must be rejected, not
    // reshaped or truncated. This is the case named keys exist to catch.
    #[test]
    fn load_checkpoint_rejects_shape_mismatch() -> Result<()> {
        let device = Device::Cpu;

        struct WideModel {
            varmap: VarMap,
            w: Tensor,
        }
        impl Parameterized for WideModel {
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

        let varmap = VarMap::new();
        // Same parameter name, three elements instead of two.
        let w = varstore::register(&varmap, "w", Tensor::new(&[1.0f32, 2.0, 3.0], &device)?)?;
        let wide = WideModel { varmap, w };

        let mut wide_opt = MlpAdamOptimizer::new(&wide, 1e-3)?;
        wide_opt.step(&wide, &[Tensor::new(&[1.0f32, 1.0, 1.0], &device)?])?;

        let path = temp_path("mismatch");
        wide_opt.save_checkpoint(&path)?;

        let narrow = TinyModel::new(&device)?;
        let mut narrow_opt = MlpAdamOptimizer::new(&narrow, 1e-3)?;
        let result = narrow_opt.load_checkpoint(&path, &device);
        std::fs::remove_file(&path).ok();

        assert!(result.is_err(), "shape mismatch must be rejected");
        // The failed load must not have left partial state behind.
        assert_eq!(narrow_opt.t, 0);
        Ok(())
    }

    // A checkpoint missing a parameter's moments must fail loudly rather than
    // resuming with that parameter's history silently zeroed.
    #[test]
    fn load_checkpoint_rejects_missing_parameter() -> Result<()> {
        let device = Device::Cpu;
        let model = TinyModel::new(&device)?;
        let mut opt = MlpAdamOptimizer::new(&model, 1e-3)?;
        opt.step(&model, &[Tensor::new(&[1.0f32, 1.0], &device)?])?;

        let path = temp_path("missing");
        opt.save_checkpoint(&path)?;

        // Optimizer built for a model whose parameter is named differently, so
        // the expected keys are absent from the file.
        struct Other {
            varmap: VarMap,
            w: Tensor,
        }
        impl Parameterized for Other {
            fn varmap(&self) -> &VarMap {
                &self.varmap
            }
            fn params(&self) -> Vec<&Tensor> {
                vec![&self.w]
            }
            fn param_names(&self) -> Vec<&str> {
                vec!["different"]
            }
        }
        let varmap = VarMap::new();
        let w = varstore::register(&varmap, "different", Tensor::new(&[1.0f32, 2.0], &device)?)?;
        let other = Other { varmap, w };

        let mut other_opt = MlpAdamOptimizer::new(&other, 1e-3)?;
        let result = other_opt.load_checkpoint(&path, &device);
        std::fs::remove_file(&path).ok();

        assert!(result.is_err(), "missing moments must be rejected");
        Ok(())
    }
}
