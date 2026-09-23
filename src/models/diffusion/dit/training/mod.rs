//! # DiT Training, Optimization, and EMA Utilities
//!
//! Submodules:
//! - [`ema`]: Polyak Exponential Moving Average (EMA) weight tracking
//! - [`schedule`]: Linear warmup and cosine annealing learning rate scheduler
//! - [`step`]: Forward diffusion $q$-sampling and AdamW training step
//! - [`checkpoint`]: Checkpoint discovery and resumption

pub mod checkpoint;
pub mod ema;
pub mod schedule;
pub mod step;

pub use checkpoint::*;
pub use ema::*;
pub use schedule::*;
pub use step::*;
