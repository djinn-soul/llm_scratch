use candle_core::{DType, Tensor};

// SINUSOIDAL TIMESTEP EMBEDDING
//
// The denoising model receives x_t, the noisy image/vector. But x_t alone is
// not enough. The same-looking value can mean different things at different
// timesteps:
//
// - early t: image is only a little noisy
// - late t: image is mostly noise
//
// So we give the model a second input: "which timestep is this?"
//
// A raw integer like 17 is weak as a neural-network feature. Instead, we turn t
// into many smooth sin/cos values. This is the same idea used in Transformer
// positional embeddings: different frequencies let the model recognize both
// small timestep changes and large timestep changes.
//
// Shape flow:
//
// t [batch]
// -> t_f32 [batch][1]
// -> args = t_f32 @ frequencies [batch][half_dim]
// -> concat(sin(args), cos(args)) [batch][dim]
//
// Example with dim = 4:
//
// t = [3]
// frequencies = [1.0, 0.01]
// embedding = [sin(3.0), sin(0.03), cos(3.0), cos(0.03)]
pub fn get_time_embedding(t: &Tensor, dim: usize) -> candle_core::Result<Tensor> {
    let device = t.device();
    let half_dim = dim / 2;
    let denom = (half_dim as f32).max(1.0);

    // Build a list of frequencies.
    //
    // freq[i] = exp(-i * log(10000) / half_dim)
    //
    // i = 0 gives a fast-changing wave. Larger i gives slower-changing waves.
    // The mix lets the model learn both nearby timestep and rough noise-level
    // information.
    let factor = (10000.0f32).ln() / denom;
    let arange = Tensor::arange(0u32, half_dim as u32, device)?.to_dtype(DType::F32)?;
    let frequencies = arange.affine(-factor as f64, 0.0)?.exp()?;

    // Turn [batch] into [batch][1], then matrix-multiply with [1][half_dim].
    //
    // For batch = 2 and half_dim = 3:
    //
    // t_f32       [2][1]
    // frequencies [1][3]
    // args        [2][3]
    //
    // Each row now has one timestep multiplied by every frequency.
    let t_f32 = t.to_dtype(DType::F32)?.reshape(((), 1))?;
    let args = t_f32.matmul(&frequencies.reshape((1, ()))?)?;

    // Sine and cosine carry the same frequency information with different
    // phase. Concatenating both gives a richer, smooth timestep code.
    let sin = args.sin()?;
    let cos = args.cos()?;
    Tensor::cat(&[&sin, &cos], 1)
}
