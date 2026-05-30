// Optimizer module boundary:
// - `optimizer.rs` holds the shared trait used by the training loop.
// - each concrete optimizer lives in its own sibling file.
// - `Param` is re-exported here for old optimizer-facing imports, but its
//   actual home is `common::param` because model layers own parameters.
//
// This keeps public imports stable:
//   use llm_scratch_rs::common::optimizers::{Adam, Optimizer, Param, SGD};
//
// Learning note:
// - SGD has no memory.
// - SGDM stores direction memory (`velocity`).
// - RMSProp stores gradient-size memory (`sq_avg`).
// - Adam stores both direction memory (`m`) and gradient-size memory (`v`).
pub mod adam;
pub mod optimizer;
pub mod rmsprop;
pub mod sgd;
pub mod sgd_m;

pub use crate::common::param::Param;
pub use adam::Adam;
pub use optimizer::Optimizer;
pub use rmsprop::RMSProp;
pub use sgd::SGD;
pub use sgd_m::SGDM;
