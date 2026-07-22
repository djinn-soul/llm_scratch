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

use crate::common::parameterized::Parameterized;

/// Save every trainable model tensor under its stable `param_names()` key.
///
/// `named_params()` performs the name/tensor count check: `zip` truncates to the
/// shorter iterator, so without it an incomplete checkpoint could be written
/// successfully.
pub fn save_model_checkpoint(model: &dyn Parameterized, path: impl AsRef<Path>) -> Result<()> {
    let tensors: HashMap<String, Tensor> = model
        .named_params()?
        .into_iter()
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
    model: &dyn Parameterized,
    path: impl AsRef<Path>,
    device: &Device,
) -> Result<()> {
    let tensors = safetensors::load(path, device)?;

    for (name, parameter) in model.named_params()? {
        let loaded = tensors
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("checkpoint is missing parameter {name}"))?;
        if loaded.dims() != parameter.dims() {
            bail!(
                "checkpoint parameter {} has shape {:?}, expected {:?}",
                name,
                loaded.dims(),
                parameter.dims()
            );
        }
        // Writing by name copies into the parameter's existing storage, which
        // the model's own field shares — so the model observes the restored
        // value without any field being reassigned. This is also why the model
        // is taken by `&` and not `&mut`.
        model.set_param(name, loaded)?;
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
        let restored = SimpleDenoisingUNet::new(16, 6, device)?;
        let path = std::env::temp_dir().join(format!(
            "llm-scratch-unet-checkpoint-{}.safetensors",
            std::process::id()
        ));

        save_model_checkpoint(&source, &path)?;
        load_model_checkpoint(&restored, &path, device)?;
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
