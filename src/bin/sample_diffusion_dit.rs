//! # Standalone Visual Sampling, CFG Guidance Sweep, and Latent Class Morphing for DiT
//!
//! ## Core Pedagogical Concepts Demonstrated:
//!
//! ### 1. Classifier-Free Guidance (CFG) Sweep: "The Prompt Volume Knob"
//! Traditional conditional diffusion computes $\epsilon_\theta(x_t, t, c)$. With CFG, we extrapolate along
//! the class conditioning vector:
//!
//! $$\tilde{\epsilon}_\theta = \epsilon_\theta(x_t, t, \emptyset) + s \cdot \Big(\epsilon_\theta(x_t, t, c) - \epsilon_\theta(x_t, t, \emptyset)\Big)$$
//!
//! By fixing the initial Gaussian noise seed $x_T \sim \mathcal{N}(0, \mathbf{I})$ across all values of $s \in [0.0, 6.0]$,
//! we hold the high-frequency spatial layout constant and isolate the pure semantic effect of guidance:
//! - $s = 0.0$: Ignores the class prompt completely; acts as the unconditional dataset prior.
//! - $s = 1.0$: Standard class-conditioned generation with realistic intra-class variation.
//! - $s = 1.5 - 2.5$: The sweet spot; sharpens silhouettes, eliminates ambiguous edges, and maximizes clarity.
//! - $s \ge 4.0$: Overdriven guidance; causes high-frequency contrast clipping and unnatural saturation.
//!
//! ### 2. Latent Class Morphing: "The Garment Blender"
//! Discrete labels $c \in \{0, \dots, 9\}$ map into continuous latent vectors $\mathbf{e}_c \in \mathbb{R}^D$.
//! By taking a linear convex combination across two class embeddings:
//!
//! $$\mathbf{e}_{\text{blend}}(\lambda) = (1 - \lambda) \cdot \mathbf{e}_{\text{from}} + \lambda \cdot \mathbf{e}_{\text{to}}, \quad \lambda \in [0.0, 1.0]$$
//!
//! and feeding $\mathbf{e}_{\text{blend}}$ through the adaLN modulation layers with fixed seed noise $x_T$,
//! the model synthesizes continuous intermediate hybrid apparel (e.g., T-shirt $\to$ Tunic $\to$ Sundress $\to$ Evening Dress).
//!
//! ### CLI Usage:
//! - `cargo run --bin sample_diffusion_dit` (generates both demonstrations)
//! - `cargo run --bin sample_diffusion_dit -- --mode cfg --class 0`
//! - `cargo run --bin sample_diffusion_dit -- --mode morph --from 0 --to 3`

use burn::module::Module;
use burn::record::CompactRecorder;
use burn::tensor::{Distribution, Tensor};

use llm_scratch_rs::models::diffusion::dit::{
    find_latest_checkpoint, sample_ddim_continuous, save_filmstrip, DiTConfig,
    DiffusionTransformer, SimpleNoiseScheduler, FASHION_CLASSES,
};

#[cfg(feature = "rocm")]
pub type MyBackend = burn::backend::Rocm;

#[cfg(all(not(feature = "rocm"), not(feature = "cpu")))]
pub type MyBackend = burn::backend::Wgpu;

#[cfg(feature = "cpu")]
pub type MyBackend = burn::backend::ndarray::NdArray;

pub fn main() -> anyhow::Result<()> {
    let device = Default::default();
    let config = DiTConfig {
        img_size: 28,
        in_channels: 1,
        patch_size: 2,
        num_classes: 11,
        hidden_dim: 256,
        depth: 6,
        num_heads: 8,
        mlp_ratio: 4.0,
    };
    let scheduler = SimpleNoiseScheduler::new_cosine(500);

    let checkpoint = std::env::args()
        .find(|a| a.ends_with(".mpk"))
        .or_else(|| find_latest_checkpoint("checkpoints").map(|(p, _)| p))
        .ok_or_else(|| anyhow::anyhow!("No checkpoint found in checkpoints/"))?;

    println!("Loading model from: {checkpoint}");
    let stem = checkpoint.trim_end_matches(".mpk");
    let recorder = CompactRecorder::new();
    let model: DiffusionTransformer<MyBackend> =
        DiffusionTransformer::new(config.clone(), &device).load_file(stem, &recorder, &device)?;

    // Class 10 is the null token used for unconditional CFG extrapolation
    let null_cond = model.get_class_embedding(10, &device);

    let args: Vec<String> = std::env::args().collect();
    let mode = args
        .iter()
        .position(|a| a == "--mode")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str());

    // ------------------------------------------------------------------------
    // Feature 1: Classifier-Free Guidance (CFG) Sweep ("Prompt Volume Knob")
    // ------------------------------------------------------------------------
    if mode.is_none() || mode == Some("cfg") {
        let class_id = args
            .iter()
            .position(|a| a == "--class")
            .and_then(|i| args.get(i + 1))
            .and_then(|s| s.parse().ok())
            .unwrap_or(0); // Default: T-shirt

        println!(
            "\n[1/2] Generating CFG Sweep for Class {}: {}...",
            class_id, FASHION_CLASSES[class_id]
        );
        let x_fixed: Tensor<MyBackend, 4> =
            Tensor::random([1, 1, 28, 28], Distribution::Normal(0.0, 1.0), &device);
        let class_cond = model.get_class_embedding(class_id, &device);

        let scales = [0.0f32, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0];
        let mut filmstrip = Vec::with_capacity(scales.len());

        for &s in &scales {
            print!("  Sampling with guidance s = {:.1}... ", s);
            let sample = sample_ddim_continuous(
                &model,
                &scheduler,
                x_fixed.clone(),
                class_cond.clone(),
                null_cond.clone(),
                50,
                s,
                config.hidden_dim,
                &device,
            );
            let pixels = sample.into_data().as_slice::<f32>().unwrap().to_vec();
            filmstrip.push(pixels);
            println!("done");
        }

        let out_path = format!("samples/dit_cfg_sweep_{}.png", FASHION_CLASSES[class_id]);
        save_filmstrip(&out_path, &filmstrip)?;
    }

    // ------------------------------------------------------------------------
    // Feature 2: Latent Class Morphing ("The Garment Blender")
    // ------------------------------------------------------------------------
    if mode.is_none() || mode == Some("morph") {
        let from_id = args
            .iter()
            .position(|a| a == "--from")
            .and_then(|i| args.get(i + 1))
            .and_then(|s| s.parse().ok())
            .unwrap_or(0); // Default: T-shirt
        let to_id = args
            .iter()
            .position(|a| a == "--to")
            .and_then(|i| args.get(i + 1))
            .and_then(|s| s.parse().ok())
            .unwrap_or(3); // Default: Dress

        println!(
            "\n[2/2] Generating Latent Morphing from {} ({}) to {} ({})...",
            from_id, FASHION_CLASSES[from_id], to_id, FASHION_CLASSES[to_id]
        );
        let x_fixed: Tensor<MyBackend, 4> =
            Tensor::random([1, 1, 28, 28], Distribution::Normal(0.0, 1.0), &device);

        let cond_a = model.get_class_embedding(from_id, &device);
        let cond_b = model.get_class_embedding(to_id, &device);

        let num_frames = 10;
        let mut filmstrip = Vec::with_capacity(num_frames);

        for step in 0..num_frames {
            let alpha = step as f32 / ((num_frames - 1) as f32);
            print!(
                "  Morph frame {}/{}: alpha = {:.2}... ",
                step + 1,
                num_frames,
                alpha
            );
            let blended_cond = cond_a.clone() * (1.0 - alpha) + cond_b.clone() * alpha;
            let sample = sample_ddim_continuous(
                &model,
                &scheduler,
                x_fixed.clone(),
                blended_cond,
                null_cond.clone(),
                50,
                2.0,
                config.hidden_dim,
                &device,
            );
            let pixels = sample.into_data().as_slice::<f32>().unwrap().to_vec();
            filmstrip.push(pixels);
            println!("done");
        }

        let out_path = format!(
            "samples/dit_morph_{}_to_{}.png",
            FASHION_CLASSES[from_id], FASHION_CLASSES[to_id]
        );
        save_filmstrip(&out_path, &filmstrip)?;
    }

    println!("\nVisual exploration completed successfully!");
    Ok(())
}
