use rand::RngExt;

// Element-wise vector addition.
//
// Shape:
//   a = [d_model]
//   b = [d_model]
//   result = [d_model]
//
// Used heavily for residual additions and token+position embedding sums.
pub fn add(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

// ── HELPER FUNCTIONS ────────────────────────────────────────────────────────

// Random matrix [rows][cols], values in -1.0..1.0. Used for weight init.
// Real models use smarter schemes (Xavier/Kaiming); uniform is fine for now.
pub fn random_matrix(rows: usize, cols: usize) -> Vec<Vec<f32>> {
    // Result shape:
    //   rows outer Vecs, each containing cols f32 values.
    //
    // Example for rows=2, cols=3:
    //   [
    //     [w00, w01, w02],
    //     [w10, w11, w12],
    //   ]
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
    // Current behavior is row-by-row addition:
    //   result[row][col] = a[row][col] + b[row][col]
    //
    // Callers should pass matching shapes. This helper does not implement full
    // NumPy-style broadcasting; it just relies on zip truncation if shapes drift.
    a.iter().zip(b.iter()).map(|(ra, rb)| add(ra, rb)).collect()
}

// Matrix multiply: a[m][n] @ b[n][p] = result[m][p]
// Triple nested loop — O(m*n*p). Slow but clear.
//   result[i][j] = Σ_k a[i][k] * b[k][j]
pub fn matmul(a: &Vec<Vec<f32>>, b: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    // STEP 1: output has one row for every row in `a`.
    // For every row i in a:
    //   compute all p output columns.
    let mut result: Vec<Vec<f32>> = Vec::new();
    for i in 0..a.len() {
        let mut row: Vec<f32> = Vec::new();

        // STEP 2: output has one column for every column in `b`.
        for j in 0..b[0].len() {
            let mut sum = 0.0;

            // STEP 3: walk the shared inner dimension n.
            // Multiply a row element by a column element and accumulate:
            //
            //   a[i] row:       [a_i0, a_i1, a_i2]
            //   b column j:     [b_0j, b_1j, b_2j]
            //   result[i][j] = a_i0*b_0j + a_i1*b_1j + a_i2*b_2j
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
    // Example:
    //   [[1, 2, 3],
    //    [4, 5, 6]]
    //
    // becomes:
    //   [[1, 4],
    //    [2, 5],
    //    [3, 6]]
    //
    // Attention uses this for K^T so each query row can score against every key
    // row through Q @ K^T.
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
