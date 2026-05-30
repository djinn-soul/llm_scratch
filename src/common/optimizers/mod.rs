// Optimizer module boundary:
// - `optimizer.rs` holds the shared trait used by the training loop.
// - each concrete optimizer lives in its own sibling file.
// - `Param` is re-exported here for old optimizer-facing imports, but its
//   actual home is `common::param` because model layers own parameters.
//
// This keeps public imports stable:
//   use llm_scratch_rs::common::optimizers::{Adam, ClippingStrategy, Optimizer, Param, SGD};
//
// Training-policy note:
// - `Optimizer::step()` is the public training-loop call.
// - `step()` applies `ClippingStrategy`, then calls each optimizer's `update()`.
// - concrete optimizers receive clipping in `new(...)` so configuration is
//   explicit at construction time.
//
// Learning note:
// - SGD has no memory.
// - SGDM stores direction memory (`velocity`).
// - RMSProp stores gradient-size memory (`sq_avg`).
// - Adam stores both direction memory (`m`) and gradient-size memory (`v`).
// - AdamW is Adam plus decoupled weight decay.
pub mod adam;
pub mod adam_w;
pub mod optimizer;
pub mod rmsprop;
pub mod sgd;
pub mod sgd_m;

pub use crate::common::param::Param;
pub use adam::Adam;
pub use adam_w::AdamW;
pub use optimizer::{ClippingStrategy, Optimizer};
pub use rmsprop::RMSProp;
pub use sgd::SGD;
pub use sgd_m::SGDM;
