use crate::common::param::Param;

/// The `Optimizer` trait defines the contract every optimizer must implement.
///
/// Any optimizer (SGD, Adam, AdamW, …) just needs to implement `step()`.
/// This lets the training loop work with any optimizer without changing code:
///   `optimizer.step(&mut params);`
///
/// Training-loop order:
///   1. clear old gradients with `zero_grad()`
///   2. run forward pass
///   3. run backward pass so each `Param.grad` is filled
///   4. call `optimizer.step(&mut params)` to update `Param.data`
///
/// Design note:
///   The model owns weights and gradients through `Param`.
///   The optimizer owns update strategy and any extra state, such as momentum
///   buffers or squared-gradient averages.
pub trait Optimizer {
    /// Update all parameter weights using their stored gradients.
    /// Called once per training step, AFTER the backward pass fills `param.grad`.
    fn step(&mut self, params: &mut [&mut Param]);
}
