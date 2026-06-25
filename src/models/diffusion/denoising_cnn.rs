use anyhow::{bail, Result};

use super::DenoisingModel;
use candle_core::{DType, Device, Tensor};

pub struct SimpleDenoisingCNN {
    pub img_dim: usize,
    pub cond_dim: usize,
    pub w_cond: Tensor, // [img_dim, cond_dim]
    pub b_cond: Tensor, // [img_dim]
    pub w1: Tensor,     // [16, 2, 3, 3]
    pub b1: Tensor,     // [16]
    pub w2: Tensor,     // [1, 16, 3, 3]
    pub b2: Tensor,     // [1]
}

impl SimpleDenoisingCNN {
    pub fn new(img_dim: usize, cond_dim: usize, device: &Device) -> Result<Self> {
        let scale_cond = (2.0f64 / cond_dim as f64).sqrt();
        let w_cond = (Tensor::randn(0.0f32, 1.0f32, (img_dim, cond_dim), device)? * scale_cond)?;
        let b_cond = Tensor::zeros(img_dim, DType::F32, device)?;

        // conv1

        let scale1 = (2.0f64 / (2.0 * 3.0 * 3.0)).sqrt();
        let w1 = (Tensor::randn(0.0f32, 1.0f32, (16, 2, 3, 3), device)? * scale1)?;
        let b1 = Tensor::zeros(16, DType::F32, device)?;
        // 3. Conv2 weights (16 channels -> 1 channel, kernel size 3x3)
        let scale2 = (2.0f64 / (16.0 * 3.0 * 3.0)).sqrt();
        let w2 = (Tensor::randn(0.0f32, 1.0f32, (1, 16, 3, 3), device)? * scale2)?;
        let b2 = Tensor::zeros(1, DType::F32, device)?;
        Ok(Self {
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

fn manual_conv2d(x: &Tensor, w: &Tensor, bias: Option<&Tensor>, device: &Device) -> Result<Tensor> {
    let (b, c_in, h, w_img) = x.dims4()?;
    let (c_out, _, _, _) = w.dims4()?;

    let zero_row = Tensor::zeros((b, c_in, 1, w_img), DType::F32, device)?;

    let x_padded_y = Tensor::cat(&[&zero_row, x, &zero_row], 2)?;

    let zero_col = Tensor::zeros((b, c_in, h + 2, 1), DType::F32, device)?;

    let x_padded = Tensor::cat(&[&zero_col, &x_padded_y, &zero_col], 3)?;
    let mut y = Tensor::zeros((b, c_out, h, w_img), DType::F32, device)?;

    for dy in 0..3 {
        for dx in 0..3 {
            let x_slice = x_padded.narrow(2, dy, h)?.narrow(3, dx, w_img)?;

            let x_flat = x_slice.reshape((b, c_in, h * w_img))?;

            let x_perm = x_flat.permute((1, 0, 2))?.reshape((c_in, b * h * w_img))?;

            let w_slice = w
                .narrow(2, dy, 1)?
                .narrow(3, dx, 1)?
                .reshape((c_out, c_in))?;

            let out_slice = w_slice.matmul(&x_perm)?;

            let out_reshaped = out_slice
                .reshape((c_out, b, h, w_img))?
                .permute((1, 0, 2, 3))?;
            y = y.add(&out_reshaped)?;
        }
    }

    if let Some(bi) = bias {
        y = y.broadcast_add(&bi.reshape((1, c_out, 1, 1))?)?;
    }
    Ok(y)
}

fn shift_and_pad(t: &Tensor, sy: i32, sx: i32, device: &Device) -> Result<Tensor> {
    let (b, c, h, w) = t.dims4()?;
    let mut out = t.clone();
    if sy == 1 {
        let zero = Tensor::zeros((b, c, 1, w), DType::F32, device)?;
        let sliced = t.narrow(2, 0, h - 1)?;
        out = Tensor::cat(&[&zero, &sliced], 2)?;
    } else if sy == -1 {
        let zero = Tensor::zeros((b, c, 1, w), DType::F32, device)?;

        let sliced = t.narrow(2, 1, h - 1)?;
        out = Tensor::cat(&[&sliced, &zero], 2)?;
    }

    if sx == 1 {
        let zero = Tensor::zeros((b, c, h, 1), DType::F32, device)?;

        let sliced = out.narrow(3, 0, w - 1)?;
        out = Tensor::cat(&[&zero, &sliced], 3)?;
    } else if sx == -1 {
        let zero = Tensor::zeros((b, c, h, 1), DType::F32, device)?;

        let sliced = out.narrow(3, 1, w - 1)?;
        out = Tensor::cat(&[&sliced, &zero], 3)?;
    }

    Ok(out)
}
/// Helper function to perform the backward pass of manual_conv2d using shift_and_pad
fn manual_conv2d_backward(
    x: &Tensor,
    w: &Tensor,
    delta_y: &Tensor,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let (b, c_in, h, w_img) = x.dims4()?;
    let (c_out, _, _, _) = w.dims4()?;

    let mut delta_x = Tensor::zeros((b, c_in, h, w_img), DType::F32, device)?;
    let mut dw_dy_list = Vec::with_capacity(3);

    for dy in 0..3 {
        let mut dw_dx_list = Vec::with_capacity(3);
        let sy = 1 - (dy as i32);
        for dx in 0..3 {
            let sx = 1 - (dx as i32);

            let x_slice = shift_and_pad(x, sy, sx, device)?;

            let x_flat = x_slice.reshape((b, c_in, h * w_img))?;

            let x_perm = x_flat.permute((1, 0, 2))?.reshape((c_in, b * h * w_img))?;

            let delta_out_slice = delta_y
                .reshape((b, c_out, h * w_img))?
                .permute((1, 0, 2))?
                .reshape((c_out, b * h * w_img))?;

            let w_slice = w
                .narrow(2, dy, 1)?
                .narrow(3, dx, 1)?
                .reshape((c_out, c_in))?;
            let dw_slice = delta_out_slice
                .matmul(&x_perm.t()?)?
                .reshape((c_out, c_in, 1, 1))?;
            dw_dx_list.push(dw_slice);

            let dx_perm = w_slice.t()?.matmul(&delta_out_slice)?;

            let dx_slice = dx_perm
                .reshape((c_in, b, h * w_img))?
                .permute((1, 0, 2))?
                .reshape((b, c_in, h, w_img))?;

            let dx_shifted = shift_and_pad(&dx_slice, -sy, -sx, device)?;

            delta_x = delta_x.add(&dx_shifted)?;
        }
        let dw_dy = Tensor::cat(&[&dw_dx_list[0], &dw_dx_list[1], &dw_dx_list[2]], 3)?;
        dw_dy_list.push(dw_dy);
    }
    let dw = Tensor::cat(&dw_dy_list, 2)?;

    Ok((delta_x, dw))
}

impl DenoisingModel for SimpleDenoisingCNN {
    fn forward(&self, x: &Tensor) -> Result<(Tensor, Vec<Tensor>)> {
        let device = x.device();

        let b = x.dim(0)?;

        // calculate spatila shape dynamically (assumes squire images)
        let h = (self.img_dim as f64).sqrt() as usize;
        let w_img = h;

        let xt = x.narrow(1, 0, self.img_dim)?;

        let cond_vec = x.narrow(1, self.img_dim, self.cond_dim)?;

        let cond_map = cond_vec
            .matmul(&self.w_cond.t()?)?
            .broadcast_add(&self.b_cond)?
            .reshape((b, 1, h, w_img))?;

        let xt_img = xt.reshape((b, 1, h, w_img))?;

        let input_cat = Tensor::cat(&[&xt_img, &cond_map], 1)?;

        // conv 1

        let z1 = manual_conv2d(&input_cat, &self.w1, Some(&self.b1), &device)?;

        let a1 = z1.maximum(&z1.affine(0.01, 0.0)?)?;

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
                "SimpleDenoisingCNN expected 3 cached intemediates from forward(_),got{}",
                intermediates.len()
            );
        }

        let device = v.device();
        let b = v.dim(0)?;
        let h = (self.img_dim as f64).sqrt() as usize;
        let w_img = h;
        let input_cat = &intermediates[0];
        let z1 = &intermediates[1];
        let a1 = &intermediates[2];

        let scale = 2.0 / (b * self.img_dim) as f64;

        let delta_pred = pred.sub(target)?.affine(scale, 0.0)?;
        let delta_z2 = delta_pred.reshape((b, 1, h, w_img))?;

        let db2 = delta_z2.sum(0)?.sum(1)?.sum(1)?;

        let (delta_a1, dw2) = manual_conv2d_backward(a1, &self.w2, &delta_z2, &device)?;

        let relu_grad = z1.ge(0.0f32)?.to_dtype(DType::F32)?.affine(0.99, 0.01)?;
        let delta_z1 = delta_a1.mul(&relu_grad)?;

        let db1 = delta_z1.sum(0)?.sum(1)?.sum(1)?;

        let (delta_input_cat, dw1) =
            manual_conv2d_backward(input_cat, &self.w1, &delta_z1, &device)?;

        let cond_vec = v.narrow(1, self.img_dim, self.cond_dim)?;

        let delta_cond_map = delta_input_cat.narrow(1, 1, 1)?;

        let delta_cond_flat = delta_cond_map.reshape((b, self.img_dim))?;

        let db_cond = delta_cond_flat.sum(0)?;
        let dw_cond = delta_cond_flat.t()?.matmul(&cond_vec)?;

        Ok(vec![dw_cond, db_cond, dw1, db1, dw2, db2])
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
    fn params_mut(&mut self) -> Vec<&mut Tensor> {
        vec![
            &mut self.w_cond,
            &mut self.b_cond,
            &mut self.w1,
            &mut self.b1,
            &mut self.w2,
            &mut self.b2,
        ]
    }
}
