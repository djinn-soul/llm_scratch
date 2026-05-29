// Optimizer module boundary:
// - `optimizer.rs` holds the shared trait used by the training loop.
// - `param.rs` holds the shared weight/gradient container.
// - each concrete optimizer lives in its own sibling file.
//
// This keeps the public import stable:
//   use llm_scratch_rs::common::optimizers::{Optimizer, Param, SGD};
pub mod adam;
pub mod optimizer;
pub mod param;
pub mod rmsprop;
pub mod sgd;
pub mod sgd_m;

pub use adam::Adam;
pub use optimizer::Optimizer;
pub use param::Param;
pub use rmsprop::RMSProp;
pub use sgd::SGD;
pub use sgd_m::SGDM;
