use rand::RngExt;

pub fn add(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

// ── HELPER FUNCTIONS ────────────────────────────────────────────────────────

// Random matrix [rows][cols], values in -1.0..1.0. Used for weight init.
// Real models use smarter schemes (Xavier/Kaiming); uniform is fine for now.
pub fn random_matrix(rows: usize, cols: usize) -> Vec<Vec<f32>> {
    let mut rng = rand::rng();
    let mut matrix = Vec::new();
    for _ in 0..rows {
        let row = (0..cols).map(|_| rng.random_range(-1.0..1.0)).collect();
        matrix.push(row);
    }
    matrix
}

// Matrix addition with broadcasting
pub fn add_mat(a: &[Vec<f32>], b: &[Vec<f32>]) -> Vec<Vec<f32>> {
    a.iter().zip(b.iter()).map(|(ra, rb)| add(ra, rb)).collect()
}
