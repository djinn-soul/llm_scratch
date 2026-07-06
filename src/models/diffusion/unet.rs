use anyhow::{bail, Ok, Result};
use candle_core::{DType, Device, Tensor};

use super::denoising_cnn_ops::{manual_conv2d, manual_conv2d_backward};
use super::DenoisingModel;

pub struct SimpleDenoisingUNet {
    pub img_dim: usize,
    pub cond_dim: usize,
    pub w_cond: Tensor, // [16,cond_dim,1,1]
    pub b_cond: Tensor, // [16]

    pub w1: Tensor, // [16,16,3,3]
    pub b1: Tensor, // [16]

    pub w2: Tensor, //[32,32,3,3]
    pub b2: Tensor, //[32]

    pub w3: Tensor, //[16,16,32,32]
    pub b3: Tensor, // [16]

    pub w4: Tensor, // [16,16,32,32]
    pub b4: Tensor, // [16]

    pub w5: Tensor, // [1,16,3,3]
    pub b5: Tensor, // [1]
}

impl SimpleDenoisingUNet {
    pub fn new(img_dim: usize, cond_dim: usize, device: &Device) -> Result<Self> {
        let h = (img_dim as f64).sqrt() as usize;
        if h * h != img_dim {
            bail!(
                "SimpleDenoisingUNet expected img_dim to be a square image area, got {}",
                img_dim
            );
        }
        if h % 2 != 0 {
            bail!(
                "SimpleDenoisingUNet expected an even image side length for 2x2 pooling, got {}",
                h
            );
        }

        let scale_cond = (2.0f64 / cond_dim as f64).sqrt();
        let w_cond = (Tensor::randn(0.0f32, 1.0f32, (img_dim, cond_dim), device)? * scale_cond)?;
        let b_cond = Tensor::zeros(img_dim, DType::F32, device)?;
        // --- Conv1 weights (2 -> 16 channels, 3x3) ---
        let scale1 = (2.0f64 / (2.0 * 3.0 * 3.0)).sqrt();
        let w1 = (Tensor::randn(0.0f32, 1.0f32, (16, 2, 3, 3), device)? * scale1)?;
        let b1 = Tensor::zeros(16, DType::F32, device)?;
        // --- Conv2 weights (16 -> 32 channels, 3x3) ---
        let scale2 = (2.0f64 / (16.0 * 3.0 * 3.0)).sqrt();

        let w2 = (Tensor::randn(0.0f32, 1.0f32, (32, 16, 3, 3), device)? * scale2)?;
        let b2 = Tensor::zeros(32, DType::F32, device)?;
        // --- Conv3 weights (32 -> 32 channels, 3x3) ---
        let scale3 = (2.0f64 / (32.0 * 3.0 * 3.0)).sqrt();
        let w3 = (Tensor::randn(0.0f32, 1.0f32, (32, 32, 3, 3), device)? * scale3)?;
        let b3 = Tensor::zeros(32, DType::F32, device)?;
        // --- Conv4 weights (48 -> 16 channels, 3x3) ---
        let scale4 = (2.0f64 / (48.0 * 3.0 * 3.0)).sqrt();
        let w4 = (Tensor::randn(0.0f32, 1.0f32, (16, 48, 3, 3), device)? * scale4)?;
        let b4 = Tensor::zeros(16, DType::F32, device)?;
        // --- Conv5 weights (16 -> 1 channel, 3x3) ---
        let scale5 = (2.0f64 / (16.0 * 3.0 * 3.0)).sqrt();
        let w5 = (Tensor::randn(0.0f32, 1.0f32, (1, 16, 3, 3), device)? * scale5)?;
        let b5 = Tensor::zeros(1, DType::F32, device)?;
        Ok(Self {
            img_dim,
            cond_dim,
            w_cond,
            b_cond,
            w1,
            b1,
            w2,
            b2,
            w3,
            b3,
            w4,
            b4,
            w5,
            b5,
        })
    }
}

impl DenoisingModel for SimpleDenoisingUNet {
    fn forward(&self, x: &Tensor) -> Result<(Tensor, Vec<Tensor>)> {
        let device = x.device();
        let b = x.dim(0)?;
        let h = (self.img_dim as f64).sqrt() as usize;
        let w_img = h;
        let h_down = h / 2;
        let w_down = w_img / 2;

        //split input
        let xt = x.narrow(1, 0, self.img_dim)?;
        let cond_vec = x.narrow(1, self.img_dim, self.cond_dim)?;

        // project conditions: shape (B,1,H, W)
        let cond_map = cond_vec
            .contiguous()?
            .matmul(&self.w_cond.t()?.contiguous()?)?
            .broadcast_add(&self.b_cond)?
            .reshape((b, 1, h, w_img))?;

        // concate noisy image and conditing map: shape (b,1,H,w)

        let xt_img = xt.contiguous()?.reshape((b, 1, h, w_img))?;

        let input_cat = Tensor::cat(&[&xt_img, &cond_map], 1)?;

        // level 1

        let z1 = manual_conv2d(&input_cat, &self.w1, Some(&self.b1), &device)?;
        // reluleaky
        let a1 = z1.maximum(&z1.affine(0.01, 0.0)?)?;
        let a1_down = a1
            .reshape((b, 16, h_down, 2, w_down, 2))?
            .mean(5)?
            .mean(3)?;

        // level 2
        let z2 = manual_conv2d(&a1_down, &self.w2, Some(&self.b2), &device)?;
        let a2 = z2.maximum(&z2.affine(0.01, 0.0)?)?;

        // level 3

        let z3 = manual_conv2d(&a2, &self.w3, Some(&self.b3), &device)?;
        let a3 = z3.maximum(&z3.affine(0.01, 0.0)?)?;

        // decoder level(28*28)

        let a3_up = a3
            .reshape((b, 32, h_down, 1, w_down, 1))?
            .broadcast_as((b, 32, h_down, 2, w_down, 2))?
            .reshape((b, 32, h, w_img))?;

        // concate upsampled features

        let decode_cat = Tensor::cat(&[&a3_up, &a1], 1)?;
        // conv4(B,16,H,W)
        let z4 = manual_conv2d(&decode_cat, &self.w4, Some(&self.b4), &device)?;
        let a4 = z4.maximum(&z4.affine(0.01, 0.0)?)?;

        // conv5
        let z5 = manual_conv2d(&a4, &self.w5, Some(&self.b5), &device)?;
        let pred = z5.reshape((b, self.img_dim))?;
        let intermediates = vec![
            input_cat, z1, a1, a1_down, z2, a2, z3, a3, a3_up, decode_cat, z4, a4,
        ];
        Ok((pred, intermediates))
    }
    fn backward(
        &self,
        v: &Tensor,
        intermediates: &[Tensor],
        pred: &Tensor,
        target: &Tensor,
    ) -> Result<Vec<Tensor>> {
        if intermediates.len() != 12 {
            bail!(
                "SimpleDenoisingUNet expected 12 cached intermediates from forward(), got {}",
                intermediates.len()
            );
        }

        let device = v.device();
        let b = v.dim(0)?;
        let h = (self.img_dim as f64).sqrt() as usize;
        let w_img = h;
        let h_down = h / 2;
        let w_down = w_img / 2;

        let (input_cat, z1, _a1, a1_down, z2, a2, z3, _a3, _a3_up, decode_cat, z4, a4) = (
            &intermediates[0],
            &intermediates[1],
            &intermediates[2],
            &intermediates[3],
            &intermediates[4],
            &intermediates[5],
            &intermediates[6],
            &intermediates[7],
            &intermediates[8],
            &intermediates[9],
            &intermediates[10],
            &intermediates[11],
        );

        // 1. MSE gradient w.r.t predication
        let scale = 2.0 / (b * self.img_dim) as f64;

        let delta_pred = pred.sub(target)?.affine(scale, 0.0)?;

        let delta_z5 = delta_pred.reshape((b, 1, h, w_img))?;

        //2. conv5 out backward
        let db5 = delta_z5.sum(0)?.sum(1)?.sum(1)?;
        let (delta_a4, dw5) = manual_conv2d_backward(a4, &self.w5, &delta_z5, device)?;

        // 3. leaky rule backward on z4
        let relu_grad4 = z4.ge(0.0f32)?.to_dtype(DType::F32)?.affine(0.99, 0.01)?;
        let delta_z4 = delta_a4.mul(&relu_grad4)?;

        // 4. conv4 backward
        let db4 = delta_z4.sum(0)?.sum(1)?.sum(1)?;
        let (delta_decode_cat, dw4) =
            manual_conv2d_backward(decode_cat, &self.w4, &delta_z4, &device)?;

        //5.  split skip connection gradient
        let delta_a3_up = delta_decode_cat.narrow(1, 0, 32)?.contiguous()?;

        let delta_a1_from_skip = delta_decode_cat.narrow(1, 32, 16)?.contiguous()?;

        //6. Nearest Neighbour upsampling backward(sum2*2)
        let delta_a3 = delta_a3_up
            .reshape((b, 32, h_down, 2, w_down, 2))?
            .sum(5)?
            .sum(3)?;

        //7.leaky relu backward on z3

        let relu_grad3 = z3.ge(0.0f32)?.to_dtype(DType::F32)?.affine(0.99, 0.01)?;
        let delta_z3 = delta_a3.mul(&relu_grad3)?;

        // 8.botlleneck conv3 backward
        let db3 = delta_z3.sum(0)?.sum(1)?.sum(1)?;
        let (delta_a2, dw3) = manual_conv2d_backward(a2, &self.w3, &delta_z3, &device)?;

        // 9. leaky relu backward onz2
        let relu_grad2 = z2.ge(0.0f32)?.to_dtype(DType::F32)?.affine(0.99, 0.01)?;
        let delta_z2 = delta_a2.mul(&relu_grad2)?;

        //10. maxpool backward
        let db2 = delta_z2.sum(0)?.sum(1)?.sum(1)?;
        let (delta_a1_down, dw2) = manual_conv2d_backward(a1_down, &self.w2, &delta_z2, &device)?;

        // 11. average pool 2x2 backward(nearest neighbour upsample scaled gradient)
        let scaled_delta = delta_a1_down.affine(0.25, 0.0)?;

        let delta_a1_from_down = scaled_delta
            .reshape((b, 16, h_down, 1, w_down, 1))?
            .broadcast_as((b, 16, h_down, 2, w_down, 2))?
            .reshape((b, 16, h, w_img))?;

        // 12. sum gradien flow in to a1
        let delta_a1 = delta_a1_from_down.add(&delta_a1_from_skip)?;

        //13. leaky relu backward on z1
        let relu_grad1 = z1.ge(0.0f32)?.to_dtype(DType::F32)?.affine(0.99, 0.01)?;
        let delta_z1 = delta_a1.mul(&relu_grad1)?;

        // 14.conv1 backward
        let db1 = delta_z1.sum(0)?.sum(1)?.sum(1)?;
        let (delta_input_cat, dw1) =
            manual_conv2d_backward(input_cat, &self.w1, &delta_z1, device)?;

        // 15. conv1 input backward

        let delta_cond_map = delta_input_cat.narrow(1, 1, 1)?.contiguous()?;
        let delta_cond_flat = delta_cond_map.reshape((b, self.img_dim))?;
        let db_cond = delta_cond_flat.sum(0)?;
        let cond_vec = v.narrow(1, self.img_dim, self.cond_dim)?.contiguous()?;
        let dw_cond = delta_cond_flat.t()?.contiguous()?.matmul(&cond_vec)?;

        Ok(vec![
            dw_cond, db_cond, dw1, db1, dw2, db2, dw3, db3, dw4, db4, dw5, db5,
        ])
    }
    fn params(&self) -> Vec<&Tensor> {
        vec![
            &self.w_cond,
            &self.b_cond,
            &self.w1,
            &self.b1,
            &self.w2,
            &self.b2,
            &self.w3,
            &self.b3,
            &self.w4,
            &self.b4,
            &self.w5,
            &self.b5,
        ]
    }
    fn params_mut(&mut self) -> Vec<&mut Tensor> {
        vec![
            &mut self.w_cond,
            &mut self.b_cond,
            &mut self.w1,
            &mut self.b1,
            &mut self.w2,
            &mut self.b2,
            &mut self.w3,
            &mut self.b3,
            &mut self.w4,
            &mut self.b4,
            &mut self.w5,
            &mut self.b5,
        ]
    }
    fn param_names(&self) -> Vec<&str> {
        vec![
            "w_cond", "b_cond", "w1", "b1", "w2", "b2", "w3", "b3", "w4", "b4", "w5", "b5",
        ]
    }
}
