//! # Exponential Moving Average (EMA) Parameter Tracking
//!
//! ## Mathematical Foundations of Model Parameter Averaging
//! Stochastic Gradient Descent (SGD) and AdamW traverse noisy, non-convex loss surfaces.
//! While gradient steps oscillate around narrow ravines, the Polyak-Ruppert Exponential Moving Average (EMA)
//! maintains a running average of the model parameters:
//!
//! $$\theta_{\text{EMA}}^{(t)} = \beta \cdot \theta_{\text{EMA}}^{(t-1)} + (1 - \beta) \cdot \theta_{\text{live}}^{(t)}$$
//!
//! where $\beta \approx 0.999$ or $0.9999$.
//!
//! ### Why EMA is Critical for Diffusion Transformers:
//! 1. **Noise Reduction on the Parameter Manifold**: High-frequency gradient oscillations are filtered out,
//!    yielding smoother score function estimates $\epsilon_\theta(x, t, c)$.
//! 2. **Prevention of Sample Mode Collapse**: Live weights may experience sudden degradation due to
//!    outlier batches; EMA preserves a stable historical consensus.
//! 3. **Superior FID & Inception Score**: In modern generative models (EDM, DiT, Stable Diffusion),
//!    the EMA checkpoint consistently outperforms the raw final optimization weights.

use std::collections::VecDeque;

use burn::module::{Module, ModuleMapper, ModuleVisitor, Param};
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

use crate::models::diffusion::dit::dit::DiffusionTransformer;

/// Heterogeneous wrapper holding parameter tensors of varying dimensionality (1D to 4D).
///
/// Erases const-generic rank $D$ so parameters can be queued in a homogeneous FIFO buffer.
pub enum AnyTensor<B: Backend> {
    D1(Tensor<B, 1>),
    D2(Tensor<B, 2>),
    D3(Tensor<B, 3>),
    D4(Tensor<B, 4>),
}

/// Visitor that collects live model parameter values into an in-memory queue.
pub struct ParamCollector<B: Backend> {
    pub queue: VecDeque<AnyTensor<B>>,
}

impl<B: Backend> ModuleVisitor<B> for ParamCollector<B> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        let val = param.val();
        let dims = val.dims();

        // Match parameter rank D (1D bias, 2D linear weight, 3D pos_embed, 4D conv filter)
        // and wrap in AnyTensor enum variant to erase the const-generic dimension for FIFO queuing
        match D {
            1 => self.queue.push_back(AnyTensor::D1(val.reshape([dims[0]]))),
            2 => self
                .queue
                .push_back(AnyTensor::D2(val.reshape([dims[0], dims[1]]))),
            3 => self
                .queue
                .push_back(AnyTensor::D3(val.reshape([dims[0], dims[1], dims[2]]))),
            4 => self.queue.push_back(AnyTensor::D4(
                val.reshape([dims[0], dims[1], dims[2], dims[3]]),
            )),
            _ => {}
        }
    }
}

/// Mapper that traverses the shadow EMA model and blends parameters from the live queue:
///
/// $$\theta_{\text{EMA}} \leftarrow \beta \cdot \theta_{\text{EMA}} + (1 - \beta) \cdot \theta_{\text{live}}$$
pub struct EmaBlender<B: Backend> {
    pub decay: f32,
    pub live_queue: VecDeque<AnyTensor<B>>,
}

impl<B: Backend> ModuleMapper<B> for EmaBlender<B> {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        // 1. Consume shadow parameter to extract its ID, raw tensor data, and mapper closure
        let (id, ema_tensor, mapper) = param.consume();
        let dims = ema_tensor.dims();

        // 2. Dequeue matching live parameter tensor from FIFO collection pass
        if let Some(live_any) = self.live_queue.pop_front() {
            // 3. Compute Polyak convex combination: ema * beta + live * (1 - beta)
            let updated = match (D, live_any) {
                (1, AnyTensor::D1(live)) => {
                    ema_tensor * self.decay + live.reshape(dims) * (1.0 - self.decay)
                }
                (2, AnyTensor::D2(live)) => {
                    ema_tensor * self.decay + live.reshape(dims) * (1.0 - self.decay)
                }
                (3, AnyTensor::D3(live)) => {
                    ema_tensor * self.decay + live.reshape(dims) * (1.0 - self.decay)
                }
                (4, AnyTensor::D4(live)) => {
                    ema_tensor * self.decay + live.reshape(dims) * (1.0 - self.decay)
                }
                _ => panic!("Shape mismatch between live and EMA parameter!"),
            };
            // 4. Reconstruct mapped Param struct preserving module graph identity
            return Param::from_mapped_value(id, updated, mapper);
        }
        Param::from_mapped_value(id, ema_tensor, mapper)
    }
}

/// Updates the shadow EMA model weights using live model parameters.
///
/// $$\theta_{\text{EMA}} \leftarrow \text{decay} \cdot \theta_{\text{EMA}} + (1 - \text{decay}) \cdot \theta_{\text{live}}$$
pub fn update_ema<B: Backend>(
    ema_model: DiffusionTransformer<B>,
    live_model: &DiffusionTransformer<B>,
    decay: f32,
) -> DiffusionTransformer<B> {
    // Pass 1: Visitor walks live model and captures all parameter tensors in deterministic FIFO order
    let mut collector = ParamCollector {
        queue: VecDeque::new(),
    };
    live_model.visit(&mut collector);

    // Pass 2: Mapper walks shadow model and blends each parameter in the exact same deterministic order
    let mut blender = EmaBlender {
        decay,
        live_queue: collector.queue,
    };
    ema_model.map(&mut blender)
}
