use llm_scratch_rs::layers::layer_norm::LayerNorm;
fn main() {
    let d_model = 4;
    let mut ln = LayerNorm::new(d_model);

    // wildly different values — after norm they should be near mean=0, std=1
    let x = vec![vec![2.0_f32, 100.0, -50.0, 3.0]];
    let out = ln.forward(&x);

    println!("In:  {:?}", x[0]);
    println!("Out: {:?}", out[0]);
    // Out values: near [-0.19, 1.41, -1.41, -0.19]
}
