// Optimizer module boundary:
// - `optimizer.rs` holds the shared trait used by the training loop.
// - each concrete optimizer lives in its own sibling file.
// - `Param` is re-exported here for old optimizer-facing imports, but its
//   actual home is `common::param` because model layers own parameters.
//
// This keeps the public import stable:
//   use llm_scratch_rs::common::optimizers::{Optimizer, Param, SGD};
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
