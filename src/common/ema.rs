// ════════════════════════════════════════════════════════════════════════════
// EXPONENTIAL MOVING AVERAGE (EMA) OF MODEL WEIGHTS
// ════════════════════════════════════════════════════════════════════════════
//
// EMA keeps a smoothed shadow copy of every trainable parameter:
//
//   shadow = decay * shadow + (1 - decay) * live
//
// Sampling from the shadow weights instead of the live ones is standard for
// diffusion models: SGD noise makes the live weights jitter around a good
// solution, and the running average sits closer to the centre of that basin.
//
// This is written against `Parameterized`, not any particular model, so it works
// for a diffusion UNet, a GPT block, or a toy MLP alike.
//
// ─────────────────────────────────────────────────────────────────────────
// WHY EVERY SNAPSHOT IS AN EXPLICIT `.copy()`
// ─────────────────────────────────────────────────────────────────────────
//
// `params()` hands back tensors that share storage with the model's VarMap
// entries, and `set_param()` writes into that storage in place. So a snapshot
// taken with `clone()` or `detach()` — both of which share storage rather than
// duplicating it — would not be a snapshot at all:
//
//   - a shadow that aliased the live weights would track them exactly, making
//     the whole average a no-op that still looks plausible in logs;
//   - a backup that aliased them would be overwritten by the very swap it
//     exists to undo, so `restore()` would put back the EMA weights.
//
// Candle also rejects the second case outright (`cannot set a variable to a
// tensor that is derived from its value`), but the first would fail silently.
// Hence `.copy()` at every point a value is captured to outlive the next write.
use anyhow::{bail, Result};
use candle_core::Tensor;

use crate::common::parameterized::Parameterized;

pub struct Ema {
    pub decay: f64,
    pub num_samples: usize,
    pub shadow_params: Vec<Tensor>,
    pub backup: Option<Vec<Tensor>>,
}

impl Ema {
    /// Seed the shadow weights from the model's current parameters.
    ///
    /// # Arguments
    ///
    /// * `target` - The model to track.
    /// * `decay` - Long-run decay rate, typically 0.999–0.9999.
    pub fn new(target: &dyn Parameterized, decay: f64) -> Result<Self> {
        let params = target.params();

        if params.is_empty() {
            bail!("Ema::new: model exposes no parameters")
        }

        let mut shadow_params = Vec::with_capacity(params.len());
        for p in &params {
            if !p.dtype().is_float() {
                bail!("EMA only supports floating point parameters");
            }
            // `.copy()`, not `.detach()` — see the storage note at the top.
            shadow_params.push(p.copy()?);
        }

        Ok(Self {
            decay,
            num_samples: 0,
            shadow_params,
            backup: None,
        })
    }

    /// Effective decay with the standard warmup ramp: `min(decay, (1+n)/(10+n))`.
    ///
    /// Without this, a decay of 0.9999 would keep the shadow pinned near its
    /// random initialisation for the first several thousand steps, so early
    /// EMA samples would be noise. The ramp starts near 0.1 and approaches the
    /// configured decay as training progresses.
    fn effective_decay(&self) -> f64 {
        let warmup_decay = (1.0 + self.num_samples as f64) / (10.0 + self.num_samples as f64);
        warmup_decay.min(self.decay)
    }

    /// Fold the current live weights into the shadow average.
    ///
    /// Call once per optimizer step, after `step()`.
    pub fn update(&mut self, target: &dyn Parameterized) -> Result<()> {
        let params = target.params();
        if params.len() != self.shadow_params.len() {
            bail!(
                "Ema::update parameter count mismatch, expected {}, got {}",
                self.shadow_params.len(),
                params.len()
            );
        }

        let decay = self.effective_decay();
        for (idx, (shadow, param)) in self.shadow_params.iter_mut().zip(params.iter()).enumerate() {
            if shadow.dims() != param.dims() {
                bail!("EMA: dimension mismatch at index {}", idx)
            }
            // shadow = decay * shadow + (1 - decay) * live
            //
            // Both `affine` calls allocate, so the result is independent storage
            // already — no copy needed here.
            let updated = shadow
                .affine(decay, 0.0)?
                .add(&param.affine(1.0 - decay, 0.0)?)?;
            *shadow = updated;
        }

        self.num_samples += 1;
        Ok(())
    }

    /// Snapshot the live weights so `restore()` can undo a `copy_to_model()`.
    ///
    /// The usual evaluation cycle is:
    ///
    ///   ema.store(&model)?;          // stash the training weights
    ///   ema.copy_to_model(&model)?;  // swap in the smoothed weights
    ///   sample(&model, ...)?;
    ///   ema.restore(&model)?;        // put the training weights back
    ///
    /// Skipping `store`/`restore` and calling `copy_to_model` alone would
    /// overwrite the weights training is still working on.
    pub fn store(&mut self, target: &dyn Parameterized) -> Result<()> {
        let params = target.params();

        let mut backup_params = Vec::with_capacity(params.len());
        for p in &params {
            // `.copy()` is mandatory: `copy_to_model` writes into this exact
            // storage, so a sharing snapshot would be destroyed by the swap.
            backup_params.push(p.copy()?);
        }
        self.backup = Some(backup_params);
        Ok(())
    }

    /// Put the weights saved by `store()` back into the model.
    pub fn restore(&mut self, target: &dyn Parameterized) -> Result<()> {
        let backup = self
            .backup
            .take()
            .ok_or_else(|| anyhow::anyhow!("Ema::restore: no backup available; call store() first"))?;

        let named = target.named_params()?;
        if named.len() != backup.len() {
            bail!(
                "Ema::restore parameter count mismatch, expected {}, got {}",
                backup.len(),
                named.len()
            );
        }
        for ((name, param), saved) in named.iter().zip(backup.iter()) {
            if param.dims() != saved.dims() {
                bail!("Ema::restore dimension mismatch for {name}");
            }
            target.set_param(name, saved)?;
        }
        Ok(())
    }

    /// Swap the shadow weights into the model for evaluation or sampling.
    ///
    /// Pair with `store()` / `restore()` unless training is finished.
    pub fn copy_to_model(&self, target: &dyn Parameterized) -> Result<()> {
        let named = target.named_params()?;
        if named.len() != self.shadow_params.len() {
            bail!(
                "Ema::copy_to_model parameter count mismatch, expected {}, got {}",
                self.shadow_params.len(),
                named.len()
            );
        }
        for ((name, param), shadow) in named.iter().zip(self.shadow_params.iter()) {
            if param.dims() != shadow.dims() {
                bail!("Ema::copy_to_model dimension mismatch for {name}");
            }
            target.set_param(name, shadow)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::varstore;
    use candle_core::Device;
    use candle_nn::VarMap;

    struct FakeModel {
        varmap: VarMap,
        w: Tensor,
        b: Tensor,
    }

    impl FakeModel {
        fn new(w: &[f32], b: &[f32], device: &Device) -> Result<Self> {
            let varmap = VarMap::new();
            let w = varstore::register(&varmap, "w", Tensor::new(w, device)?)?;
            let b = varstore::register(&varmap, "b", Tensor::new(b, device)?)?;
            Ok(Self { varmap, w, b })
        }
    }

    impl Parameterized for FakeModel {
        fn varmap(&self) -> &VarMap {
            &self.varmap
        }

        fn params(&self) -> Vec<&Tensor> {
            vec![&self.w, &self.b]
        }

        fn param_names(&self) -> Vec<&str> {
            vec!["w", "b"]
        }
    }

    #[test]
    fn ema_warmup_and_store_restore_cycle() -> Result<()> {
        let device = Device::Cpu;
        let model = FakeModel::new(&[1.0f32, 2.0], &[0.5f32], &device)?;

        let mut ema = Ema::new(&model, 0.9999)?;
        assert_eq!(ema.shadow_params[0].to_vec1::<f32>()?, vec![1.0, 2.0]);

        // Move the live weights, then fold them in.
        model.set_param("w", &Tensor::new(&[3.0f32, 4.0], &device)?)?;
        // The model's own field must observe the write: it shares storage with
        // the VarMap entry rather than being a stale copy.
        assert_eq!(model.w.to_vec1::<f32>()?, vec![3.0, 4.0]);

        ema.update(&model)?;

        // num_samples = 0 -> effective decay = 1/10 = 0.1
        // shadow = 0.1 * 1.0 + 0.9 * 3.0 = 2.8
        let shadow_w = ema.shadow_params[0].to_vec1::<f32>()?;
        assert!((shadow_w[0] - 2.8).abs() < 1e-4);

        // Non-destructive store / swap / restore cycle.
        ema.store(&model)?;
        ema.copy_to_model(&model)?;
        let copied_w = model.w.to_vec1::<f32>()?;
        assert!((copied_w[0] - 2.8).abs() < 1e-4);
        assert!((copied_w[1] - 3.8).abs() < 1e-4);

        ema.restore(&model)?;
        assert_eq!(model.w.to_vec1::<f32>()?, vec![3.0, 4.0]);

        Ok(())
    }

    // The shadow must be an independent buffer. If `Ema::new` captured the live
    // parameters with `clone()`/`detach()` instead of `copy()`, the shadow would
    // share storage with them and silently track every in-place write — an
    // average that is always exactly the live weights.
    #[test]
    fn shadow_does_not_alias_live_parameters() -> Result<()> {
        let device = Device::Cpu;
        let model = FakeModel::new(&[1.0f32, 2.0], &[0.5f32], &device)?;
        let ema = Ema::new(&model, 0.999)?;

        model.set_param("w", &Tensor::new(&[99.0f32, 99.0], &device)?)?;

        assert_eq!(ema.shadow_params[0].to_vec1::<f32>()?, vec![1.0, 2.0]);
        Ok(())
    }
}
