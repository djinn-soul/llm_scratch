use anyhow::{bail, Result};
use candle_core::{Device, Tensor};
use rand::RngExt;

pub fn make_one_hot_cfg(
    labels: &[u8],
    num_classes: usize,
    drop_rate: f32,
    device: &Device,
) -> Result<Tensor> {
    let mut rng = rand::rng();
    let mut hot = vec![0.0f32; labels.len() * num_classes];

    for (i, &label) in labels.iter().enumerate() {
        if rng.random::<f32>() > drop_rate {
            let idx = (i * num_classes) + label as usize;
            hot[idx] = 1.0;
        }
    }

    Ok(Tensor::from_vec(hot, (labels.len(), num_classes), device)?)
}

pub fn one_hot_class(class_label: usize, num_classes: usize, device: &Device) -> Result<Tensor> {
    if class_label >= num_classes {
        bail!(
            "class label {} is out of range for {} classes",
            class_label,
            num_classes
        );
    }

    let mut hot = vec![0.0f32; num_classes];
    hot[class_label] = 1.0;
    Ok(Tensor::new(hot.as_slice(), device)?.reshape((1, num_classes))?)
}
