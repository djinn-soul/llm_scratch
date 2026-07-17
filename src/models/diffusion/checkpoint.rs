// =============================================================================
// checkpoint.rs — architecture-aware SafeTensors persistence
// =============================================================================
//
// DenoisingModel deliberately exposes parameters as positional vectors for the
// manual optimizer. A checkpoint needs stable names as well: positions alone
// would allow an architecture edit to silently assign weights to the wrong
// layer. These helpers pair both representations and fail on contract drift.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{bail, Result};
use candle_core::{safetensors, Device, Tensor};

use super::DenoisingModel;

/// Save every trainable model tensor under its stable `param_names()` key.
///
/// The count check is intentional. `zip` truncates to the shorter iterator, so
/// without it an incomplete checkpoint could be written successfully.
pub fn save_model_checkpoint(model: &dyn DenoisingModel, path: impl AsRef<Path>) -> Result<()> {
    let names: Vec<String> = model.param_names().into_iter().map(str::to_owned).collect();
    let params = model.params();
    if names.len() != params.len() {
        bail!(
            "model exposes {} parameter names for {} tensors",
            names.len(),
            params.len()
        );
    }

    let tensors: HashMap<String, Tensor> = names
        .into_iter()
        .zip(params)
        .map(|(name, tensor)| (name.to_owned(), tensor.clone()))
        .collect();
    safetensors::save(&tensors, path)?;
    Ok(())
}

/// Restore a checkpoint into an already constructed model.
///
/// SafeTensors handles device placement while this function validates the model
/// contract: every expected name must exist and its shape must match. Unexpected
/// extra tensors are harmless because only the model's declared parameters are
/// consumed.
pub fn load_model_checkpoint(
    model: &mut dyn DenoisingModel,
    path: impl AsRef<Path>,
    device: &Device,
) -> Result<()> {
    let tensors = safetensors::load(path, device)?;
    let names: Vec<String> = model.param_names().into_iter().map(str::to_owned).collect();
    let mut params = model.params_mut();
    if names.len() != params.len() {
        bail!(
            "model exposes {} parameter names for {} tensors",
            names.len(),
            params.len()
        );
    }

    for (name, parameter) in names.iter().zip(params.iter_mut()) {
        let loaded = tensors
            .get(name.as_str())
            .ok_or_else(|| anyhow::anyhow!("checkpoint is missing parameter {name}"))?;
        if loaded.dims() != parameter.dims() {
            bail!(
                "checkpoint parameter {} has shape {:?}, expected {:?}",
                name,
                loaded.dims(),
                parameter.dims()
            );
        }
        // params_mut returns mutable references to the model fields. Replacing
        // the Tensor handle updates the actual model rather than a temporary
        // parameter list.
        **parameter = loaded.clone();
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::diffusion::SimpleDenoisingUNet;

    #[test]
    fn unet_checkpoint_round_trip_preserves_parameters() -> Result<()> {
        let device = &Device::Cpu;
        let source = SimpleDenoisingUNet::new(16, 6, device)?;
        let mut restored = SimpleDenoisingUNet::new(16, 6, device)?;
        let path = std::env::temp_dir().join(format!(
            "llm-scratch-unet-checkpoint-{}.safetensors",
            std::process::id()
        ));

        save_model_checkpoint(&source, &path)?;
        load_model_checkpoint(&mut restored, &path, device)?;
        std::fs::remove_file(&path)?;

        // Exact equality is expected: SafeTensors stores f32 values losslessly,
        // with no text conversion or quantization in the round trip.
        for (expected, actual) in source.params().into_iter().zip(restored.params()) {
            let max_diff = expected.sub(actual)?.abs()?.max_all()?.to_scalar::<f32>()?;
            assert_eq!(max_diff, 0.0);
        }
        Ok(())
    }
}
