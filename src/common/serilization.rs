use std::fs::File;
use std::io::{Read, Write};
use crate::common::optimizers::Param;

pub trait SaveableModel {
    /// Every model must implement this method to recursively return mutable references
    /// to all of its learnable parameters in a stable, deterministic order.
    fn parameters(&mut self) -> Vec<&mut Param>;

    /// Save all model weights to disk, auto-detecting JSON (.json) or high-performance Binary (default).
    fn save_weights(&mut self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        if path.ends_with(".json") {
            // ── SAVE AS JSON ─────────────────────────────────────────────────
            let params = self.parameters();
            let weights: Vec<Vec<Vec<f32>>> = params.into_iter().map(|p| p.data.clone()).collect();
            let json = serde_json::to_string(&weights)?;
            std::fs::write(path, json)?;
        } else {
            // ── SAVE AS RAW BINARY BYTES ──────────────────────────────────────
            let mut file = File::create(path)?;
            let params = self.parameters();
            
            // 1. Write total number of matrices (64-bit unsigned integer)
            file.write_all(&(params.len() as u64).to_le_bytes())?;
            
            for param in params {
                let rows = param.data.len() as u64;
                let cols = param.data[0].len() as u64;
                
                // 2. Write shape dimensions
                file.write_all(&rows.to_le_bytes())?;
                file.write_all(&cols.to_le_bytes())?;
                
                // 3. Write float elements (4 bytes each)
                for row in &param.data {
                    for &val in row {
                        file.write_all(&val.to_le_bytes())?;
                    }
                }
            }
        }
        Ok(())
    }

    /// Load weights from disk, auto-detecting JSON (.json) or high-performance Binary (default).
    fn load_weights(&mut self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        if path.ends_with(".json") {
            // ── LOAD FROM JSON ───────────────────────────────────────────────
            let json = std::fs::read_to_string(path)?;
            let weights: Vec<Vec<Vec<f32>>> = serde_json::from_str(&json)?;
            let mut params = self.parameters();
            
            assert_eq!(params.len(), weights.len(), "Weight matrices count mismatch!");
            for (param, weight) in params.iter_mut().zip(weights.into_iter()) {
                assert_eq!(param.data.len(), weight.len(), "Dimension mismatch in rows!");
                assert_eq!(param.data[0].len(), weight[0].len(), "Dimension mismatch in cols!");
                param.data = weight;
            }
        } else {
            // ── LOAD FROM RAW BINARY BYTES ───────────────────────────────────
            let mut file = File::open(path)?;
            let mut params = self.parameters();
            
            // 1. Verify total number of matrices
            let mut num_matrices_buf = [0; 8];
            file.read_exact(&mut num_matrices_buf)?;
            let num_matrices = u64::from_le_bytes(num_matrices_buf) as usize;
            assert_eq!(params.len(), num_matrices, "Weight matrices count mismatch!");
            
            for param in &mut params {
                // 2. Read and verify shape dimensions
                let mut size_buf = [0; 8];
                file.read_exact(&mut size_buf)?;
                let rows = u64::from_le_bytes(size_buf) as usize;
                file.read_exact(&mut size_buf)?;
                let cols = u64::from_le_bytes(size_buf) as usize;
                
                assert_eq!(param.data.len(), rows, "Dimension mismatch in rows!");
                assert_eq!(param.data[0].len(), cols, "Dimension mismatch in cols!");
                
                // 3. Read float bytes back into memory
                for i in 0..rows {
                    for j in 0..cols {
                        let mut val_buf = [0; 4];
                        file.read_exact(&mut val_buf)?;
                        param.data[i][j] = f32::from_le_bytes(val_buf);
                    }
                }
            }
        }
        Ok(())
    }
}
