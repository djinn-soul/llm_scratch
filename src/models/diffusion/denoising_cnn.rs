// =============================================================================
// denoising_cnn.rs — SimpleDenoisingCNN: a 2-layer CNN noise predictor (3x3 kernels)
// =============================================================================
//
// This module implements a small convolutional denoising network that serves
// as a drop-in replacement for `SimpleDenoisingMlp` in the DDPM pipeline.
//
// Architecture (forward path):
//   Input: v = concat(x_t, time_emb, class_one_hot) → shape (B, 784 + 26)
//   1. Split v into x_t (784 dims) and cond_vec (26 dims).
//   2. Conditioning projection: Linear(26 -> 784) → shape (B, 1, 28, 28)
//   3. Channel concatenation: cat([xt_img, cond_map], dim=1) → shape (B, 2, 28, 28)
//   4. Conv1: manual_conv2d (2 → 16 channels, 3x3 kernel) + Leaky-ReLU(0.01)
//   5. Conv2: manual_conv2d (16 → 1 channel, 3x3 kernel) → reshape to (B, 784)
//
// =============================================================================

use anyhow::{bail, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarMap;

use super::denoising_cnn_ops::{manual_conv2d, manual_conv2d_backward};
use super::DenoisingModel;
use crate::common::parameterized::Parameterized;
use crate::common::varstore;

pub struct SimpleDenoisingCNN {
    /// Owns every trainable parameter under its checkpoint name; the tensor
    /// fields below share storage with its `Var`s.
    varmap: VarMap,

    pub img_dim: usize,  // flattened image size (784 for MNIST)
    pub cond_dim: usize, // conditioning vector size (time_emb_dim + class_dim)
    pub w_cond: Tensor,  // [img_dim, cond_dim]
    pub b_cond: Tensor,  // [img_dim]
    pub w1: Tensor,      // [16, 2, 3, 3]
    pub b1: Tensor,      // [16]
    pub w2: Tensor,      // [1, 16, 3, 3]
    pub b2: Tensor,      // [1]
}

impl SimpleDenoisingCNN {
    pub fn new(img_dim: usize, cond_dim: usize, device: &Device) -> Result<Self> {
        // Every parameter is registered in the VarMap; keep the tensor that
        // `register` returns, not the one passed in — only the former shares
        // storage with the stored `Var` and observes later updates.
        let varmap = VarMap::new();

        // --- Conditioning projection weights ---------------------------------
        let scale_cond = (2.0f64 / cond_dim as f64).sqrt();
        let w_cond = varstore::register(
            &varmap,
            "w_cond",
            (Tensor::randn(0.0f32, 1.0f32, (img_dim, cond_dim), device)? * scale_cond)?,
        )?;
        let b_cond = varstore::register(
            &varmap,
            "b_cond",
            Tensor::zeros(img_dim, DType::F32, device)?,
        )?;

        // --- Conv1 weights ---------------------------------------------------
        // fan_in = C_in * kH * kW = 2 * 3 * 3 = 18
        // scale1 = sqrt(2 / 18) = 0.33333
        let scale1 = (2.0f64 / (2.0 * 3.0 * 3.0)).sqrt();
        let w1 = varstore::register(
            &varmap,
            "w1",
            (Tensor::randn(0.0f32, 1.0f32, (16, 2, 3, 3), device)? * scale1)?,
        )?;
        let b1 = varstore::register(&varmap, "b1", Tensor::zeros(16, DType::F32, device)?)?;

        // --- Conv2 weights ---------------------------------------------------
        // fan_in = C_in * kH * kW = 16 * 3 * 3 = 144
        // scale2 = sqrt(2 / 144) ≈ 0.11785
        let scale2 = (2.0f64 / (16.0 * 3.0 * 3.0)).sqrt();
        let w2 = varstore::register(
            &varmap,
            "w2",
            (Tensor::randn(0.0f32, 1.0f32, (1, 16, 3, 3), device)? * scale2)?,
        )?;
        let b2 = varstore::register(&varmap, "b2", Tensor::zeros(1, DType::F32, device)?)?;

        Ok(Self {
            varmap,
            img_dim,
            cond_dim,
            w_cond,
            b_cond,
            w1,
            b1,
            w2,
            b2,
        })
    }
}

impl DenoisingModel for SimpleDenoisingCNN {
    fn forward(&self, x: &Tensor) -> Result<(Tensor, Vec<Tensor>)> {
        let device = x.device();
        let b = x.dim(0)?;

        let h = (self.img_dim as f64).sqrt() as usize;
        let w_img = h;

        // Split input
        let xt = x.narrow(1, 0, self.img_dim)?;
        let cond_vec = x.narrow(1, self.img_dim, self.cond_dim)?;

        // Project conditioning
        let cond_map = cond_vec
            .contiguous()?
            .matmul(&self.w_cond.t()?.contiguous()?)?
            .broadcast_add(&self.b_cond)?
            .reshape((b, 1, h, w_img))?;

        // Concatenate noisy image and conditioning map
        let xt_img = xt.reshape((b, 1, h, w_img))?;
        let input_cat = Tensor::cat(&[&xt_img, &cond_map], 1)?;

        // Conv1 + Leaky-ReLU
        let z1 = manual_conv2d(&input_cat, &self.w1, Some(&self.b1), &device)?;
        let a1 = z1.maximum(&z1.affine(0.01, 0.0)?)?;

        // Conv2 → output
        let z2 = manual_conv2d(&a1, &self.w2, Some(&self.b2), &device)?;
        let pred = z2.reshape((b, self.img_dim))?;

        let intermediates = vec![input_cat, z1, a1];
        Ok((pred, intermediates))
    }

    fn backward(
        &self,
        v: &Tensor,
        intermediates: &[Tensor],
        pred: &Tensor,
        target: &Tensor,
    ) -> Result<Vec<Tensor>> {
        if intermediates.len() != 3 {
            bail!(
                "SimpleDenoisingCNN expected 3 cached intermediates from forward(), got {}",
                intermediates.len()
            );
        }

        let device = v.device();
        let b = v.dim(0)?;

        let input_cat = &intermediates[0];
        let z1 = &intermediates[1];
        let a1 = &intermediates[2];

        // MSE gradient scale = 2 / (B * img_dim)
        let scale = 2.0 / (b * self.img_dim) as f64;
        let delta_pred = pred.sub(target)?.affine(scale, 0.0)?;

        // Reshape to spatial
        let h = (self.img_dim as f64).sqrt() as usize;
        let w_img = h;
        let delta_z2 = delta_pred.reshape((b, 1, h, w_img))?;

        // Conv2 backward
        let db2 = delta_z2.sum(0)?.sum(1)?.sum(1)?;
        let (delta_a1, dw2) = manual_conv2d_backward(a1, &self.w2, &delta_z2, &device)?;

        // Leaky-ReLU backward
        let relu_grad = z1.ge(0.0f32)?.to_dtype(DType::F32)?.affine(0.99, 0.01)?;
        let delta_z1 = delta_a1.mul(&relu_grad)?;

        // Conv1 backward
        let db1 = delta_z1.sum(0)?.sum(1)?.sum(1)?;
        let (delta_input_cat, dw1) =
            manual_conv2d_backward(input_cat, &self.w1, &delta_z1, &device)?;

        // Conditioning projection backward
        let delta_cond_map = delta_input_cat.narrow(1, 1, 1)?;
        let delta_cond_flat = delta_cond_map.reshape((b, self.img_dim))?;
        let db_cond = delta_cond_flat.sum(0)?;

        let cond_vec = v.narrow(1, self.img_dim, self.cond_dim)?.contiguous()?;
        let dw_cond = delta_cond_flat.t()?.contiguous()?.matmul(&cond_vec)?;

        Ok(vec![dw_cond, db_cond, dw1, db1, dw2, db2])
    }
}

impl Parameterized for SimpleDenoisingCNN {
    fn varmap(&self) -> &VarMap {
        &self.varmap
    }

    fn params(&self) -> Vec<&Tensor> {
        vec![
            &self.w_cond,
            &self.b_cond,
            &self.w1,
            &self.b1,
            &self.w2,
            &self.b2,
        ]
    }

    fn param_names(&self) -> Vec<&str> {
        vec!["w_cond", "b_cond", "w1", "b1", "w2", "b2"]
    }
}
