// ════════════════════════════════════════════════════════════════════════════
// PARAMETERIZED — the shared contract for "a thing with trainable weights"
// ════════════════════════════════════════════════════════════════════════════
//
// This trait is deliberately model-agnostic. A diffusion UNet, a GPT block, or
// a two-layer MLP all satisfy it, and everything that only needs to *read* or
// *write* weights is written against this trait instead of a concrete model:
//
//   optimizers   (`common::optimizers::MlpAdamOptimizer`)
//   EMA          (`common::ema::Ema`)
//   checkpoints  (`models::diffusion::checkpoint`)
//
// Keeping the contract here — rather than on `DenoisingModel` — is what stops
// those three from becoming diffusion-only. `DenoisingModel` extends this trait
// and adds just `forward()` / `backward()`.
//
// ─────────────────────────────────────────────────────────────────────────
// TWO VIEWS OF THE SAME WEIGHTS
// ─────────────────────────────────────────────────────────────────────────
//
// Positional (`params()`):
//   Manual backprop returns gradients as a `Vec<Tensor>`. Aligning a gradient
//   with its parameter is done by index, so the order must be stable.
//
// Named (`param_names()`, `set_param()`):
//   A checkpoint keyed by position silently reassigns weights to the wrong
//   layer the moment an architecture is edited — the shapes still match, so
//   nothing errors. Anything that outlives the process is keyed by name.
//
// `param_names()` is the single owner of the ordering contract: entry `i` names
// `params()[i]` and `backward()`'s gradient `i`. `named_params()` zips the two
// with a length check so consumers do not each re-implement it.
use anyhow::{bail, Result};
use candle_core::Tensor;
use candle_nn::VarMap;

use crate::common::varstore;

pub trait Parameterized {
    /// The VarMap holding every trainable parameter of this model.
    ///
    /// Parameters are registered with `varstore::register()`, which guarantees
    /// the `Tensor` fields returned by `params()` share storage with the stored
    /// `Var`s. That sharing is what makes `set_param()` visible to the model
    /// without reassigning any struct field.
    fn varmap(&self) -> &VarMap;

    /// Ordered references to all trainable parameters.
    ///
    /// The order must match `param_names()` and the gradient order produced by
    /// the model's backward pass.
    fn params(&self) -> Vec<&Tensor>;

    /// Human-readable name for each parameter, in `params()` order.
    ///
    /// These are the checkpoint keys, so they are part of the on-disk format:
    /// renaming one invalidates existing checkpoints.
    fn param_names(&self) -> Vec<&str>;

    /// Look up one parameter by name.
    ///
    /// For generic code that knows a parameter only by its checkpoint key. A
    /// model's own forward pass should read its cached struct fields instead —
    /// this locks the VarMap and hashes the name on every call.
    fn get(&self, name: &str) -> Result<Tensor> {
        varstore::get(self.varmap(), name)
    }

    /// Overwrite one parameter in place, by name.
    ///
    /// This is the only write path. See `common::varstore::set` for why `value`
    /// must not be derived from the parameter's own storage — a snapshot taken
    /// from `params()` has to be `.copy()`-ed before it can be written back.
    fn set_param(&self, name: &str, value: &Tensor) -> Result<()> {
        varstore::set(self.varmap(), name, value)
    }

    /// Name/tensor pairs, validated once so callers do not each `zip` blindly.
    ///
    /// `zip` truncates to the shorter side, so a model whose `param_names()`
    /// drifted out of sync with `params()` would otherwise produce a checkpoint
    /// that is silently missing its last parameters.
    fn named_params(&self) -> Result<Vec<(&str, &Tensor)>> {
        let names = self.param_names();
        let params = self.params();
        if names.len() != params.len() {
            bail!(
                "model exposes {} parameter names for {} tensors",
                names.len(),
                params.len()
            );
        }
        Ok(names.into_iter().zip(params).collect())
    }
}
