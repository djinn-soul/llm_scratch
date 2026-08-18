// utils/mod.rs — re-exports all shared utility modules
pub mod image_io; // validated normalized-image to PNG conversion
pub mod mnist_utils; // shared MNIST download, parse, and one-hot helpers

use anyhow::Result;
use std::path::Path;

/// Ensures the parent directory of a given file path exists on disk.
/// Call this before creating/saving any file or checkpoint.
pub fn ensure_parent_dir(path: impl AsRef<Path>) -> Result<()> {
    if let Some(parent) = path.as_ref().parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    Ok(())
}
