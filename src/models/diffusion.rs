//https://huggingface.co/learn/diffusion-course/en/unit1/3
//https://github.com/juraam/stable-diffusion-from-scratch
//https://www.kaggle.com/code/takihasan/stable-diffusion-from-scratch
//https://github.com/yousef-rafat/miniDiffusion
// https://lilianweng.github.io/posts/2021-07-11-diffusion-models/
// https://github.com/zelaki/Reading-Diffusion
// https://learnopencv.com/denoising-diffusion-probabilistic-models/
//https://sander.ai/
//https://stackoverflow.com/questions/75693493/why-the-text-embedding-or-image-embedding-generated-by-clip-model-is-768-%c3%97-n/79243065#79243065
// https://cdn.openai.com/papers/dall-e-2.pdf
use anyhow::{Ok, Result};
use candle_core::{DType, Device, Tensor};

// ════════════════════════════════════════════════════════════════════════════
// DDPM BETA SCHEDULER — THE FIXED NOISE PLAN
// ════════════════════════════════════════════════════════════════════════════
// Diffusion training starts with one question:
//
//   "If I start from clean data x_0, how noisy should it look at timestep t?"
//
// beta_t is the small amount of fresh noise added at one step.
// alpha_t = 1 - beta_t is the signal kept at that step.
// alpha_bar_t = alpha_0 * ... * alpha_t is the total signal still surviving.
//
//   x_t = sqrt(alpha_bar_t) * x_0
//       + sqrt(1 - alpha_bar_t) * epsilon
//
// Example: if x_0=2.0, epsilon=-0.5, alpha_bar_t=0.81:
//   x_t = 0.90*2.0 + 0.435*(-0.5) = 1.5825
//
// In a batch, x_0/noise are [batch][data_dim] and t is [batch]. The scheduler
// gathers one scalar coefficient per row and broadcasts it across data_dim.
pub struct BetaScheduler {
    pub steps: usize,
    pub betas: Tensor,
    pub alphas: Tensor,
    pub alphas_cumprod: Tensor,
    pub alphas_cumprod_prev: Tensor,
    pub sqrt_alphas_cumprod: Tensor,
    pub sqrt_one_minus_alphas_cumprod: Tensor,
    pub sigmas: Tensor,
}
impl BetaScheduler {
    pub fn new(steps: usize, beta_start: f64, beta_end: f64, device: &Device) -> Result<Self> {
        // ── STEP 1: LINEAR BETA SCHEDULE ──────────────────────────────────
        // beta_start -> beta_end means noise is added gently at first and more
        // strongly near the end. Example with 4 steps:
        //   betas = [0.0001, 0.0067, 0.0134, 0.0200]
        let mut betas_vec = Vec::with_capacity(steps);
        for i in 0..steps {
            let beta = beta_start + (beta_end - beta_start) * (i as f64) / ((steps - 1) as f64);
            betas_vec.push(beta as f32);
        }
        let betas = Tensor::new(betas_vec.as_slice(), device)?;

        // ── STEP 2: PER-STEP SIGNAL KEEP RATE ─────────────────────────────
        // alpha_t = 1 - beta_t.
        // If beta_t = 0.02, then alpha_t = 0.98, meaning this one step keeps
        // about 98 percent of the previous signal and injects about 2 percent
        // new variance.
        let alphas = Tensor::ones(steps, DType::F32, device)?.sub(&betas)?;

        // ── STEP 3: CUMULATIVE SIGNAL KEEP RATE ───────────────────────────
        // alpha_bar_t is the running product of alphas.
        // [a,b,c]-> [a,a*b,a*b*c]
        // Example: [0.99, 0.98, 0.97] -> [0.99, 0.9702, 0.941094]
        let mut alphas_cumprod_vec = Vec::with_capacity(steps);
        let mut cumprod = 1.0f32;
        let alphas_f32 = alphas.to_vec1::<f32>()?;
        for &alpha in &alphas_f32 {
            cumprod *= alpha;
            alphas_cumprod_vec.push(cumprod);
        }
        let alphas_cumprod = Tensor::new(alphas_cumprod_vec.as_slice(), device)?;

        // ── STEP 4: PREVIOUS ALPHA_BAR FOR REVERSE VARIANCE ───────────────
        // Reverse sampling uses both alpha_bar_t and alpha_bar_{t-1}. For
        // t=0, there is no previous diffusion step, so we define the previous
        // clean-signal product as 1.0.
        let mut alphas_cumprod_prev_vec = Vec::with_capacity(steps);
        alphas_cumprod_prev_vec.push(1.0f32);
        for i in 0..(steps - 1) {
            alphas_cumprod_prev_vec.push(alphas_cumprod_vec[i]);
        }
        let alphas_cumprod_prev = Tensor::new(alphas_cumprod_prev_vec.as_slice(), device)?;

        // ── STEP 5: PRECOMPUTE FOR q(x_t | x_0) ───────────────────────────
        // These two vectors are exactly the coefficients in:
        //   x_t = sqrt(alpha_bar_t) * x_0
        //       + sqrt(1 - alpha_bar_t) * noise
        let sqrt_alphas_cumprod = alphas_cumprod.sqrt()?;
        let one_minus_alpha_cumprod =
            Tensor::ones(steps, DType::F32, device)?.sub(&alphas_cumprod)?;
        let sqrt_one_minus_alphas_cumprod = one_minus_alpha_cumprod.sqrt()?;

        // ── STEP 6: REVERSE-PROCESS SIGMA ──────────────────────────────────
        // sigma_t is the standard deviation for sampling x_{t-1} from x_t.
        // t=0 gets sigma 0 because the final step should not add fresh noise.
        let mut sigmas_vec = Vec::with_capacity(steps);
        sigmas_vec.push(0.0f32); // t=0 is 0
        for t in 1..steps {
            let alpha_bar = alphas_cumprod_vec[t];
            let alpha_bar_prev = alphas_cumprod_prev_vec[t];
            let beta = betas_vec[t];
            let variance = ((1.0 - alpha_bar_prev) / (1.0 - alpha_bar)) * beta;
            sigmas_vec.push(variance.sqrt());
        }
        let sigmas = Tensor::new(sigmas_vec.as_slice(), device)?;

        Ok(Self {
            steps,
            betas,
            alphas,
            alphas_cumprod,
            alphas_cumprod_prev,
            sqrt_alphas_cumprod,
            sqrt_one_minus_alphas_cumprod,
            sigmas,
        })
    }
    pub fn add_noise(&self, x0: &Tensor, noise: &Tensor, t: &Tensor) -> Result<Tensor> {
        // Shapes: x0/noise [batch][data_dim], t [batch].
        //
        // index_select gathers one coefficient per sample:
        //   t = [3, 7]
        //   coeffs = [sqrt_alpha_bar_3, sqrt_alpha_bar_7]
        // reshape(((), 1)) turns [batch] into [batch][1] so each scalar can
        // broadcast across all data dimensions in its row.
        let sqrt_alpha_bar = self
            .sqrt_alphas_cumprod
            .index_select(t, 0)?
            .reshape(((), 1))?;

        let sqrt_one_minus_alpha_bar = self
            .sqrt_one_minus_alphas_cumprod
            .index_select(t, 0)?
            .reshape(((), 1))?;

        // Forward diffusion in one line:
        //   clean part = x_0   * sqrt(alpha_bar_t)
        //   noise part = noise * sqrt(1 - alpha_bar_t)
        //   x_t        = clean part + noise part
        // Row-level picture for data_dim = 2:
        //   x0[0]    = [2.0, -1.0]
        //   noise[0] = [0.3,  0.8]
        //   coeffs   = sqrt_alpha_bar=0.9, sqrt_one_minus=0.435
        //   xt[0]    = [2.0*0.9 + 0.3*0.435, -1.0*0.9 + 0.8*0.435]
        let xt = x0
            .broadcast_mul(&sqrt_alpha_bar)?
            .add(&noise.broadcast_mul(&sqrt_one_minus_alpha_bar)?)?;
        Ok(xt)
    }
}

// ════════════════════════════════════════════════════════════════════════════
// SINUSOIDAL TIMESTEP EMBEDDING
// ════════════════════════════════════════════════════════════════════════════
// The denoising network needs to know "how noisy" x_t is. A raw integer like
// 17 is too small and discrete, so we convert timestep t into a vector using
// sin/cos waves at many frequencies.
//
// Shape flow: t [batch] -> t_f32 [batch][1]
//   args = t_f32 @ frequencies[1][half_dim] -> [batch][half_dim]
//   concat(sin(args), cos(args)) -> [batch][dim]
//
// With dim=4, t=[3], frequencies=[1.0, 0.01]:
//   embedding = [sin(3.0), sin(0.03), cos(3.0), cos(0.03)]
pub fn get_time_embedding(t: &Tensor, dim: usize) -> candle_core::Result<Tensor> {
    let device = t.device();
    let half_dim = dim / 2;
    let denom = (half_dim as f32).max(1.0);
    // ── STEP 1: LOG-FREQUENCY SCALE ───────────────────────────────────────
    // freq[i] = exp(-i * log(10000) / half_dim)
    // Small i changes quickly with t; large i changes slowly.
    let factor = (10000.0f32).ln() / denom;
    // ── STEP 2: BUILD FREQUENCY VECTOR ────────────────────────────────────
    // arange = [0, 1, 2, ... half_dim - 1]
    let arange = Tensor::arange(0u32, half_dim as u32, device)?.to_dtype(DType::F32)?;
    // frequencies = exp(arange * -factor)
    let frequecies = arange.affine(-factor as f64, 0.0)?.exp()?;

    // ── STEP 3: OUTER PRODUCT t × frequencies ─────────────────────────────
    // t_f32 [batch][1] @ frequencies [1][half_dim] creates one row of
    // arguments per sample in the batch.
    let t_f32 = t.to_dtype(DType::F32)?.reshape(((), 1))?;

    let args = t_f32.matmul(&frequecies.reshape((1, ()))?)?;

    // ── STEP 4: PHASE PAIRING ──────────────────────────────────────────────
    // Sine and cosine use the same arguments but are phase-shifted.
    let sin = args.sin()?;
    let cos = args.cos()?;

    Tensor::cat(&[&sin, &cos], 1)
}

// ════════════════════════════════════════════════════════════════
// MANUAL BACKPROPAGATION MLP
// ════════════════════════════════════════════════════════════════
// This tiny MLP is the noise predictor:
//   input  v      = concat(x_t, time_embedding)  [batch][in_dim]
//   hidden z1     = v @ w1^T + b1               [batch][hidden_dim]
//   hidden a1     = relu(z1)                    [batch][hidden_dim]
//   pred epsilon  = a1 @ w2^T + b2              [batch][out_dim]
// During training, target is the exact noise epsilon used by add_noise().
pub struct SimpleDenoisingMlp {
    pub w1: Tensor, // [hidden_dim, in_dim]
    pub b1: Tensor, // [hidden_dim]
    pub w2: Tensor, // [out_dim, hidden_dim]
    pub b2: Tensor, // [out_dim]
}
pub struct Gradients {
    pub dw1: Tensor,
    pub db1: Tensor,
    pub dw2: Tensor,
    pub db2: Tensor,
}

impl SimpleDenoisingMlp {
    pub fn new(in_dim: usize, hidden_dim: usize, out_dim: usize, device: &Device) -> Result<Self> {
        // Build the two-layer denoising MLP's trainable parameters.
        //
        // Dimension meaning in this diffusion model:
        //
        //   in_dim     = noisy data width + timestep embedding width
        //              = concat(x_t, time_embedding) width
        //
        //   hidden_dim = number of hidden neurons after the first layer
        //
        //   out_dim    = predicted noise width
        //              = usually the same width as x_0 / x_t / epsilon
        //
        // Forward equations these parameters will later serve:
        //
        //   z1   = v  @ w1^T + b1
        //   a1   = relu(z1)
        //   pred = a1 @ w2^T + b2
        //
        // Shape plan:
        //
        //   v    [batch][in_dim]
        //   w1   [hidden_dim][in_dim]
        //   b1   [hidden_dim]
        //   z1   [batch][hidden_dim]
        //
        //   a1   [batch][hidden_dim]
        //   w2   [out_dim][hidden_dim]
        //   b2   [out_dim]
        //   pred [batch][out_dim]
        //
        // He-style scale for ReLU layers: wider inputs need smaller random
        // weights so activations do not explode at initialization.
        //
        // Why sqrt(2 / input_width)?
        //
        // Each hidden neuron sums many input values. If in_dim is large and we
        // use big random weights, the sum can become too large before training
        // even starts. ReLU also zeros roughly half of random activations, so
        // the factor 2 helps preserve a useful activation variance after ReLU.
        //
        // Small example:
        //
        //   in_dim = 8
        //   scale1 = sqrt(2 / 8) = 0.5
        //
        // A random weight like 0.70 becomes 0.70 * 0.5 = 0.35. The neuron still
        // starts random, but not so large that z1 immediately explodes.
        let scale1 = (2.0f64 / in_dim as f64).sqrt();

        // w1 maps the input vector v into hidden neurons.
        //
        // Tensor::rand(0.0, 1.0, (hidden_dim, in_dim), device) creates a matrix
        // of random numbers on the chosen device. Multiplying by scale1 shrinks
        // every random value:
        //
        //   w1[row][col] = random_0_to_1 * sqrt(2 / in_dim)
        //
        // Later, forward uses w1.t() so the matmul lines up as:
        //
        //   v [batch][in_dim] @ w1^T [in_dim][hidden_dim]
        //     -> z1 [batch][hidden_dim]
        let w1 = (Tensor::randn(0.0f32, 1.0f32, (hidden_dim, in_dim), device)? * scale1)?;

        // b1 starts at zero because the random w1 already breaks symmetry.
        //
        // Bias is one number per hidden neuron. During forward, Candle
        // broadcast_add(&b1) adds the same b1 vector to every batch row:
        //
        //   z1[row] = matmul_result[row] + b1
        let b1 = Tensor::zeros(hidden_dim, DType::F32, device)?;

        // Layer 2 receives hidden activations, not the original input, so its
        // input width is hidden_dim. That is why scale2 uses hidden_dim in the
        // denominator instead of in_dim.
        let scale2 = (2.0f64 / hidden_dim as f64).sqrt();

        // w2 maps hidden activations into the final noise prediction.
        //
        // Shape is [out_dim][hidden_dim] because forward uses w2.t():
        //
        //   a1 [batch][hidden_dim] @ w2^T [hidden_dim][out_dim]
        //      -> pred [batch][out_dim]
        //
        // Each output dimension learns a different weighted mix of hidden
        // neurons. For diffusion, each output dimension is one coordinate of
        // the predicted epsilon noise.
        let w2 = (Tensor::randn(0.0f32, 1.0f32, (out_dim, hidden_dim), device)? * scale2)?;

        // b2 is one bias per predicted noise coordinate. Starting at zero means
        // the first predictions come only from the random hidden mapping; if a
        // coordinate needs a consistent shift, training will move b2.
        let b2 = Tensor::zeros(out_dim, DType::F32, device)?;

        // Return the initialized model. `Ok(...)` is needed because every
        // Candle tensor allocation above can fail, so this constructor returns
        // Result<Self> instead of plain Self.
        Ok(Self { w1, b1, w2, b2 })
    }

    pub fn forward(&self, v: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        // Run one forward pass of the denoising MLP.
        //
        // `v` is the full conditioning input for the model. In this diffusion
        // code, it should normally be built by concatenating:
        //
        //   noisy sample x_t       [batch][data_dim]
        //   timestep embedding     [batch][time_embed_dim]
        //
        // So:
        //
        //   v [batch][in_dim]
        //   in_dim = data_dim + time_embed_dim
        //
        // The output `pred` is the model's guess for the exact noise epsilon
        // that was added by `BetaScheduler::add_noise`.
        //
        // ── LAYER 1: AFFINE + RELU ────────────────────────────────────────
        // First affine layer:
        //
        //   z1 = v @ w1^T + b1
        //
        // Why `w1.t()`?
        //
        // We store w1 as [hidden_dim][in_dim], meaning each row is one hidden
        // neuron's weight vector. Matmul needs the shared inner dimensions to
        // match, so we transpose it:
        //
        //   v      [batch][in_dim]
        //   w1     [hidden_dim][in_dim]
        //   w1^T   [in_dim][hidden_dim]
        //   result [batch][hidden_dim]
        //
        // Then `broadcast_add(&b1)` adds the same hidden bias vector to every
        // batch row:
        //
        //   z1[row][hidden] = weighted_sum(row, hidden) + b1[hidden]
        //
        // `z1` is the pre-activation value. We keep it because backward needs
        // to know which ReLU entries were positive and which were blocked.
        // v [batch][in_dim] @ w1^T [in_dim][hidden_dim]
        //   -> z1 [batch][hidden_dim]
        let z1 = v.matmul(&self.w1.t()?)?.broadcast_add(&self.b1)?;

        // ReLU turns the first layer into a nonlinear model.
        //
        // Without ReLU, both layers would collapse into one big linear layer:
        //
        //   pred = (v @ w1^T + b1) @ w2^T + b2
        //
        // A linear layer can only learn straight-line relationships. ReLU lets
        // different hidden neurons activate for different regions/timesteps,
        // which is essential for learning denoising behavior that changes with
        // noise level.
        //
        // ReLU keeps positive values and zeros negative values:
        //   z1 row = [-0.4, 1.2, 0.0]
        //   a1 row = [ 0.0, 1.2, 0.0]
        // LeakyReLU = max(0.01 * z1, z1)

        let a1 = z1.maximum(&z1.affine(0.01, 0.0)?)?;

        // ── LAYER 2: PREDICT NOISE ────────────────────────────────────────
        // Final affine layer:
        //
        //   pred = a1 @ w2^T + b2
        //
        // Here each output coordinate is one predicted noise coordinate. If
        // out_dim == data_dim, then pred has the same shape as the noise used
        // to create x_t:
        //
        //   target epsilon [batch][out_dim]
        //   pred epsilon   [batch][out_dim]
        //
        // `broadcast_add(&b2)` adds one final bias per output coordinate. That
        // lets the model shift a predicted noise coordinate up/down even when
        // the hidden activations alone are not centered correctly.
        // a1 [batch][hidden_dim] @ w2^T [hidden_dim][out_dim]
        //   -> pred [batch][out_dim], same shape as the original noise.
        let pred = a1.matmul(&self.w2.t()?)?.broadcast_add(&self.b2)?;

        // Return more than just `pred` because this implementation does manual
        // backpropagation. The backward pass needs:
        //
        //   pred: compare against target noise to get output error
        //   a1:   compute dw2 = delta2^T @ a1
        //   z1:   build the ReLU gradient mask for delta1
        //
        // If we returned only pred, backward would need to recompute a1/z1 or
        // would not know which ReLU units were active in this exact forward.
        Ok((pred, a1, z1))
    }

    pub fn backward(
        &self,
        v: &Tensor,
        a1: &Tensor,
        z1: &Tensor,
        pred: &Tensor,
        target: &Tensor,
    ) -> Result<Gradients> {
        // Mathematically exact MSE gradient, including the averaging constant over batch and output dimensions:
        //   dL/dpred = (2.0 / (batch_size * out_dim)) * (pred - target)
        let batch_size = pred.dim(0)?;
        let out_dim = pred.dim(1)?;
        let scale = 2.0 / (batch_size * out_dim) as f64;
        let delta2 = pred.sub(target)?.affine(scale, 0.0)?;
        // Layer 2 gradients: pred = a1 @ w2^T + b2
        //   dw2 = delta2^T @ a1
        //   db2 = sum rows of delta2
        //   dw2 shape = [out_dim][hidden_dim], matching w2.
        let dw2 = delta2.t()?.matmul(&a1)?;
        let db2 = delta2.sum(0)?;

        // ReLU backward: positive z1 passes gradient, negative z1 blocks it.
        // ge(0) builds that 0/1 mask with the same shape as z1.
        let relu_grad = z1.ge(0.0f32)?.to_dtype(DType::F32)?.affine(0.99, 0.01)?;

        // Move the output error back through w2, then apply the ReLU mask.
        //   delta1 = (delta2 @ w2) * relu_grad
        // delta1 shape: [batch][hidden_dim]
        let delta1 = delta2.matmul(&self.w2)?.mul(&relu_grad)?;

        // Layer 1 gradients: z1 = v @ w1^T + b1
        //   dw1 = delta1^T @ v
        //   dw1 shape = [hidden_dim][in_dim], matching w1.
        let dw1 = delta1.t()?.matmul(&v)?;

        let db1 = delta1.sum(0)?;

        Ok(Gradients { dw1, db1, dw2, db2 })
    }

    pub fn update(&mut self, grads: &Gradients, lr: f64, _batch_size: usize) -> Result<()> {
        // Gradients are already averaged over the batch and output dims inside backward,
        // so the update step simplifies to standard parameter scaling:
        //   param = param - lr * grad
        let scale = lr;

        // Why `affine(scale, 0.0)` here?
        //
        // Candle tensors do not let us write scalar multiplication with plain
        // Rust math inside `sub()`. `Tensor::affine(a, b)` is Candle's helper
        // for the elementwise transform:
        //
        //   output = input * a + b
        //
        // So:
        //
        //   grads.dw1.affine(scale, 0.0)?
        //
        // means:
        //
        //   scaled_dw1 = grads.dw1 * scale + 0.0
        //              = grads.dw1 * (lr / batch_size)
        //
        // Then the subtraction does the SGD update:
        //
        //   new_w1 = old_w1 - scaled_dw1
        //
        // Small one-weight example:
        //
        //   old_w1 = 0.80
        //   dw1    = 0.50      // summed gradient from the batch
        //   lr     = 0.10
        //   batch  = 5
        //   scale  = 0.10 / 5 = 0.02
        //
        //   dw1.affine(scale, 0.0) = 0.50 * 0.02 + 0.0 = 0.01
        //   new_w1                = 0.80 - 0.01 = 0.79
        //
        // Why subtract? The gradient points in the direction where loss
        // increases fastest. Training wants lower loss, so SGD steps in the
        // opposite direction.
        //
        // Why assign back to `self.w1`? Candle Tensor operations return new
        // tensors instead of modifying the old tensor in place, so we replace
        // each parameter with its updated value.
        // ── UPDATE LAYER 1 PARAMETERS ─────────────────────────────────────
        // Update w1, the matrix used in the first forward matmul:
        //
        //   z1 = v @ w1^T + b1
        //
        // w1 decides how each input feature contributes to each hidden neuron.
        // If one input feature helped predict too much noise, dw1 points in
        // the direction that caused that error. Subtracting scaled dw1 nudges
        // w1 so the next forward pass produces a slightly better hidden z1.
        //
        // Shape check:
        //   self.w1  [hidden_dim][in_dim]
        //   grads.dw1 [hidden_dim][in_dim]
        //
        // Because the shapes match, this is elementwise:
        //   every weight w1[row][col] gets its own gradient dw1[row][col].
        self.w1 = self.w1.sub(&grads.dw1.affine(scale, 0.0)?)?;

        // Update b1, the bias added after the first matmul:
        //
        //   z1 = v @ w1^T + b1
        //
        // b1 shifts each hidden neuron up or down before ReLU. db1 is the sum
        // of delta1 over the batch, so each hidden neuron gets one bias update.
        //
        // Shape check:
        //   self.b1   [hidden_dim]
        //   grads.db1 [hidden_dim]
        //
        // Bias has no input-feature dimension because it is broadcast across
        // every row in the batch.
        self.b1 = self.b1.sub(&grads.db1.affine(scale, 0.0)?)?;

        // ── UPDATE LAYER 2 PARAMETERS ─────────────────────────────────────
        // Update w2, the matrix used in the final prediction matmul:
        //
        //   pred = a1 @ w2^T + b2
        //
        // w2 decides how hidden activations combine into the predicted noise.
        // If pred is larger than target for an output dimension, dw2 tells us
        // which hidden activations contributed to that error. Subtracting the
        // scaled gradient reduces that error direction.
        //
        // Shape check:
        //   self.w2   [out_dim][hidden_dim]
        //   grads.dw2 [out_dim][hidden_dim]
        self.w2 = self.w2.sub(&grads.dw2.affine(scale, 0.0)?)?;

        // Update b2, the final output bias:
        //
        //   pred = a1 @ w2^T + b2
        //
        // b2 shifts the predicted noise directly. db2 is the batch-summed
        // output error `pred - target`, one number per output dimension.
        //
        // Shape check:
        //   self.b2   [out_dim]
        //   grads.db2 [out_dim]
        self.b2 = self.b2.sub(&grads.db2.affine(scale, 0.0)?)?;
        Ok(())
    }
}

// ════════════════════════════════════════════════════════════════
// MLP ADAM OPTIMIZER
// ════════════════════════════════════════════════════════════════
pub struct MlpAdamOptimizer {
    pub mw1: Tensor,
    pub vw1: Tensor,
    pub mb1: Tensor,
    pub vb1: Tensor,
    pub mw2: Tensor,
    pub vw2: Tensor,
    pub mb2: Tensor,
    pub vb2: Tensor,
    pub t: usize,
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
}

impl MlpAdamOptimizer {
    pub fn new(mlp: &SimpleDenoisingMlp, lr: f64) -> Result<Self> {
        let device = mlp.w1.device();
        Ok(Self {
            mw1: Tensor::zeros(mlp.w1.dims(), DType::F32, device)?,
            vw1: Tensor::zeros(mlp.w1.dims(), DType::F32, device)?,
            mb1: Tensor::zeros(mlp.b1.dims(), DType::F32, device)?,
            vb1: Tensor::zeros(mlp.b1.dims(), DType::F32, device)?,
            mw2: Tensor::zeros(mlp.w2.dims(), DType::F32, device)?,
            vw2: Tensor::zeros(mlp.w2.dims(), DType::F32, device)?,
            mb2: Tensor::zeros(mlp.b2.dims(), DType::F32, device)?,
            vb2: Tensor::zeros(mlp.b2.dims(), DType::F32, device)?,
            t: 0,
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
        })
    }

    pub fn step(&mut self, mlp: &mut SimpleDenoisingMlp, grads: &Gradients) -> Result<()> {
        self.t += 1;
        let t = self.t;
        let lr = self.lr;
        let beta1 = self.beta1;
        let beta2 = self.beta2;
        let eps = self.eps;
        let adam_update =
            |param: &mut Tensor, m: &mut Tensor, v: &mut Tensor, grad: &Tensor| -> Result<()> {
                // m = beta1 * m + (1 - beta1) * grad
                let m_new = m.affine(beta1, 0.0)?.add(&grad.affine(1.0 - beta1, 0.0)?)?;
                // v = beta2 * v + (1 - beta2) * grad^2
                let grad_sq = grad.sqr()?;
                let v_new = v
                    .affine(beta2, 0.0)?
                    .add(&grad_sq.affine(1.0 - beta2, 0.0)?)?;
                // Bias corrections
                let bc1 = 1.0 - beta1.powi(t as i32);
                let bc2 = 1.0 - beta2.powi(t as i32);
                let m_hat = m_new.affine(1.0 / bc1, 0.0)?;
                let v_hat = v_new.affine(1.0 / bc2, 0.0)?;
                // param = param - lr * m_hat / (sqrt(v_hat) + eps)
                let num = m_hat.affine(lr, 0.0)?;
                let den = v_hat.sqrt()?.affine(1.0, eps)?;
                let update = num.div(&den)?;
                *param = param.sub(&update)?;
                *m = m_new;
                *v = v_new;
                Ok(())
            };
        adam_update(&mut mlp.w1, &mut self.mw1, &mut self.vw1, &grads.dw1)?;
        adam_update(&mut mlp.b1, &mut self.mb1, &mut self.vb1, &grads.db1)?;
        adam_update(&mut mlp.w2, &mut self.mw2, &mut self.vw2, &grads.dw2)?;
        adam_update(&mut mlp.b2, &mut self.mb2, &mut self.vb2, &grads.db2)?;
        Ok(())
    }
}
