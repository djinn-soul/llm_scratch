// ════════════════════════════════════════════════════════════════════════════
// VARSTORE — named parameter registration on top of candle's VarMap
// ════════════════════════════════════════════════════════════════════════════
//
// Models in this crate do manual backpropagation, so they keep their weights as
// plain `Tensor` struct fields and use them directly in forward/backward math.
// Persistence and optimizer updates, however, want *names*: a checkpoint keyed
// by position silently reassigns weights to the wrong layer the moment an
// architecture is edited.
//
// `VarMap` supplies the naming and the safetensors round-trip. This module is
// the glue that registers an already-initialised tensor into a `VarMap`.
//
// ─────────────────────────────────────────────────────────────────────────
// THE STORAGE-SHARING RULE (read before touching this)
// ─────────────────────────────────────────────────────────────────────────
//
// `Var::from_tensor()` does NOT wrap the tensor you hand it. Internally it
// calls `make_var()`, which allocates fresh storage and copies:
//
//     let init = Tensor::randn(...)?;
//     let var  = Var::from_tensor(&init)?;   // var has its OWN storage
//     // `init` and `var` are now two independent buffers.
//
// `Var::set()` writes in place into the Var's storage. So a model field that
// kept `init` would never observe an optimizer update — training would appear
// to run while the weights stayed frozen at their initial values, which is
// exactly the kind of bug that looks like "the model just learns slowly".
//
// The only tensor that stays in sync is the one taken back out of the Var:
//
//     let field = var.as_tensor().clone();   // shares storage with `var` ✓
//
// `register()` below returns precisely that tensor, so callers cannot get this
// wrong by accident. Always store the returned value in the model field and
// drop the tensor that was passed in.
use std::collections::hash_map::Entry;

use anyhow::{bail, Result};
use candle_core::{Tensor, Var};
use candle_nn::VarMap;

/// Register `init` in `varmap` under `name` and return the model-field tensor.
///
/// The returned tensor shares storage with the stored `Var`, so later
/// `Var::set()` calls (optimizer steps, checkpoint loads, EMA swaps) are visible
/// through it. See the storage-sharing rule at the top of this file.
///
/// Registering the same name twice is rejected rather than silently overwritten:
/// a duplicate name means two different weights would collide in the checkpoint,
/// and the second one would win on save while the first is lost.
pub fn register(varmap: &VarMap, name: &str, init: Tensor) -> Result<Tensor> {
    let var = Var::from_tensor(&init)?;
    // Take the field tensor from the Var, not from `init`.
    let field = var.as_tensor().clone();

    let mut data = varmap
        .data()
        .lock()
        .map_err(|_| anyhow::anyhow!("VarMap mutex poisoned while registering {name}"))?;

    match data.entry(name.to_string()) {
        Entry::Occupied(_) => bail!("duplicate parameter name {name:?} registered in VarMap"),
        Entry::Vacant(slot) => {
            slot.insert(var);
        }
    }

    Ok(field)
}

/// Look up a registered parameter by name.
///
/// The returned tensor shares storage with the stored `Var`, so it tracks later
/// updates just like a model's own field does.
///
/// This is for code that knows a parameter only by name — checkpoint loaders,
/// tests, debugging. Model forward passes should use their cached struct fields
/// instead: this path locks the map and hashes the name on every call, which is
/// wasted work when the same weights are read every step.
pub fn get(varmap: &VarMap, name: &str) -> Result<Tensor> {
    let data = varmap
        .data()
        .lock()
        .map_err(|_| anyhow::anyhow!("VarMap mutex poisoned while reading {name}"))?;

    match data.get(name) {
        None => bail!("parameter {name:?} is not registered in VarMap"),
        Some(var) => Ok(var.as_tensor().clone()),
    }
}

/// Overwrite the value of an already-registered parameter, in place.
///
/// This is the write path for optimizers, EMA weight swaps, and checkpoint
/// restores. Because `Var::set()` mutates the shared storage, every model field
/// produced by `register()` observes the new value immediately — no need to
/// reassign struct fields.
///
/// `value` must not be derived from the parameter's own storage. Candle rejects
/// that case (`cannot set a variable to a tensor that is derived from its
/// value`), because the in-place copy would read and write the same buffer.
/// Arithmetic like `param.sub(&update)?` allocates a fresh tensor and is fine;
/// a saved snapshot of `params()` is NOT and must be `.copy()`-ed first.
pub fn set(varmap: &VarMap, name: &str, value: &Tensor) -> Result<()> {
    let data = varmap
        .data()
        .lock()
        .map_err(|_| anyhow::anyhow!("VarMap mutex poisoned while setting {name}"))?;

    match data.get(name) {
        None => bail!("parameter {name:?} is not registered in VarMap"),
        Some(var) => {
            if var.shape() != value.shape() {
                bail!(
                    "shape mismatch setting {name:?}: variable is {:?}, value is {:?}",
                    var.shape(),
                    value.shape()
                );
            }
            var.set(value)?;
            Ok(())
        }
    }
}
