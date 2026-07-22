use anyhow::Result;
use candle_core::Tensor;

pub use crate::common::parameterized::Parameterized;

// PLUGGABLE DENOISING MODEL TRAIT
//
// This trait abstracts the noise-prediction neural network used in diffusion
// training and sampling. Any architecture (MLP, UNet, Transformer, etc.) can
// implement this trait and plug into the shared training loop, optimizer, and
// sampling functions without modification.
//
// Design principles:
//
// 1. **Opaque intermediates**: `forward()` returns a `Vec<Tensor>` of cached
//    activations whose meaning is private to each implementation. The training
//    loop passes them back to `backward()` unchanged. This avoids leaking
//    architecture-specific types (like `(a1, z1)` for the MLP) into shared code.
//
// 2. **Weight access is not defined here**: parameter reads, names, and writes
//    come from `common::parameterized::Parameterized`, which this trait
//    extends. That split is intentional — optimizers, EMA, and checkpointing
//    only need `Parameterized`, so they stay usable by non-diffusion models.
//    This trait adds only the two things that are specific to a noise
//    predictor: `forward()` and `backward()`.
//
//    Note there is no `params_mut()`. Writes go through `set_param()` by name;
//    see `Parameterized` for why handing out `&mut Tensor` would let a caller
//    silently detach a weight from its VarMap entry.
//
// Example of how a training loop uses the trait:
//
//   let (pred, intermediates) = model.forward(&v)?;
//   let grads = model.backward(&v, &intermediates, &pred, &target)?;
//   optimizer.step(model, &grads)?;
pub trait DenoisingModel: Parameterized {
    /// Forward pass: predict noise from the conditioning input `v`.
    ///
    /// Returns:
    ///   - `Tensor`: the noise prediction, shape `[batch, out_dim]`
    ///   - `Vec<Tensor>`: opaque intermediate activations needed by `backward()`
    ///
    /// During inference (sampling), the intermediates can be discarded.
    fn forward(&self, v: &Tensor) -> Result<(Tensor, Vec<Tensor>)>;

    /// Manual backward pass: compute per-parameter gradients.
    ///
    /// Arguments:
    ///   - `v`:             the same input that was passed to `forward()`
    ///   - `intermediates`: the cached activations returned by `forward()`
    ///   - `pred`:          the noise prediction returned by `forward()`
    ///   - `target`:        the ground-truth noise tensor (training target)
    ///
    /// Returns one gradient `Tensor` per trainable parameter, in the same order
    /// as `params()`. The optimizer uses this correspondence to apply updates.
    fn backward(
        &self,
        v: &Tensor,
        intermediates: &[Tensor],
        pred: &Tensor,
        target: &Tensor,
    ) -> Result<Vec<Tensor>>;
}
