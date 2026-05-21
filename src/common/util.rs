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

// Matrix multiply: a[m][n] @ b[n][p] = result[m][p]
// Triple nested loop — O(m*n*p). Slow but clear.
//   result[i][j] = Σ_k a[i][k] * b[k][j]
pub fn matmul(a: &Vec<Vec<f32>>, b: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    let mut result: Vec<Vec<f32>> = Vec::new();
    for i in 0..a.len() {
        let mut row: Vec<f32> = Vec::new();
        for j in 0..b[0].len() {
            let mut sum = 0.0;
            // walk shared inner dimension, multiply-accumulate
            for k in 0..a[0].len() {
                sum += a[i][k] * b[k][j];
            }
            row.push(sum);
        }
        result.push(row);
    }

    result
}

// Transpose: flip rows and columns. [rows][cols] → [cols][rows]
// transposed[i][j] = matrix[j][i]. Needed to compute Q @ K^T.
pub fn mat_transpose(matrix: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    let rows = matrix.len();
    let cols = matrix[0].len();
    let mut transposed: Vec<Vec<f32>> = Vec::new();
    for i in 0..cols {
        let mut row: Vec<f32> = Vec::new();
        for j in 0..rows {
            row.push(matrix[j][i])
        }
        transposed.push(row);
    }
    transposed
}
