//! # Checkpoint Discovery & State Serialization
//!
//! Provides utilities for discovering and resuming training from `.mpk` model checkpoints.

use std::path::Path;

/// Parses the integer training step index from a checkpoint file path.
///
/// Handles naming conventions such as `dit_fashion_step_05000.mpk` -> `5000`.
pub fn extract_step_from_path(path: &str) -> Option<usize> {
    let stem = Path::new(path).file_stem()?.to_str()?;

    if let Some(pos) = stem.rfind("step_") {
        stem[pos + 5..].parse::<usize>().ok()
    } else {
        stem.split(|c: char| !c.is_ascii_digit())
            .filter_map(|s| s.parse::<usize>().ok())
            .next_back()
    }
}

/// Scans a target directory for `.mpk` checkpoints and returns the highest step found.
pub fn find_latest_checkpoint(dir: &str) -> Option<(String, usize)> {
    let entries = std::fs::read_dir(dir).ok()?;
    let mut latest: Option<(String, usize)> = None;

    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) == Some("mpk") {
            let path_str = path.to_string_lossy().to_string();
            if let Some(step) = extract_step_from_path(&path_str) {
                if latest
                    .as_ref()
                    .is_none_or(|(_, max_step)| step > *max_step)
                {
                    latest = Some((path_str, step));
                }
            }
        }
    }
    latest
}
