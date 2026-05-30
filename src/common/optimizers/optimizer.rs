use crate::common::param::Param;

/// The `Optimizer` trait defines the contract every optimizer must implement.
///
/// Any optimizer (SGD, Adam, AdamW, …) just needs to implement `step()`.
/// This lets the training loop work with any optimizer without changing code:
///   `optimizer.step(&mut params);`
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
