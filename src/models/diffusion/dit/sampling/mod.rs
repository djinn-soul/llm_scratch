//! # DiT Sampling, Scheduling, and Visualization Module
//!
//! Provides modular components for diffusion inference:
//! - [`scheduler`]: Cosine and linear noise variance schedules & Fourier timestep embeddings
//! - [`ddim`]: Deterministic DDIM reverse sampling, Classifier-Free Guidance, and latent class morphing
//! - [`vis`]: Dynamic range remapping, 2x5 lookbook collages, and linear filmstrip stitching

pub mod ddim;
pub mod scheduler;
pub mod vis;

pub use ddim::*;
pub use scheduler::*;
pub use vis::*;
