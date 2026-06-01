// ════════════════════════════════════════════════════════════════════════════
// LEARNING RATE SCHEDULER SYSTEM
// ════════════════════════════════════════════════════════════════════════════
//
// Dynamically adjusts the optimizer's learning rate at each training step.
// Swappable strategies help prevent overshooting (unstable training) and
// improve convergence towards the bottom of local minima.
//
// Reference:
//   - https://machinelearningmastery.com/a-gentle-introduction-to-learning-rate-schedulers/
// ════════════════════════════════════════════════════════════════════════════

/// Learning rate scheduling strategies.
#[derive(Debug, Clone, Copy)]
pub enum LRScheduler {
    /// Keeps a fixed learning rate for the entire training duration.
    /// Default standard SGD/Adam behavior.
    Constant { lr: f32 },

    /// Drops the learning rate by a factor `decay_rate` every `step_size` steps.
    ///
    /// Formula:
    ///   lr = initial_lr * decay_rate ^ floor(step / step_size)
    StepDecay {
        initial_lr: f32,
        decay_rate: f32,
        step_size: usize,
    },

    /// Smoothly decays the learning rate at every step by an exponential factor.
    ///
    /// Formula:
    ///   lr = initial_lr * decay_rate ^ step
    ExponentialDecay { initial_lr: f32, decay_rate: f32 },

    /// State-of-the-art LLM schedule: Linear Warmup followed by Cosine Annealing.
    ///
    /// Formula:
    ///   - warmup: lr = max_lr * step / warmup_steps
    ///   - cosine: lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(pi * progress))
    CosineWarmup {
        max_lr: f32,
        min_lr: f32,
        warmup_steps: usize,
        total_steps: usize,
    },
}

impl LRScheduler {
    pub fn get_lr(&self, current_step: usize) -> f32 {
        match *self {
            // 1. Constant learning rate
            LRScheduler::Constant { lr } => lr,
            // 2. step decay (basic decay after every k steps)
            LRScheduler::StepDecay {
                initial_lr,
                decay_rate,
                step_size,
            } => {
                let num_decays = current_step / step_size;
                initial_lr * (decay_rate.powi(num_decays as i32))
            }
            // 3. exponential decay (smooth decay)
            LRScheduler::ExponentialDecay {
                initial_lr,
                decay_rate,
            } => initial_lr * decay_rate.powi(current_step as i32),
            // 4. cosine warmup
            LRScheduler::CosineWarmup {
                max_lr,
                min_lr,
                warmup_steps,
                total_steps,
            } => {
                if current_step < warmup_steps {
                    // linear warmup
                    max_lr * (current_step as f32 / warmup_steps as f32)
                } else {
                    // cosine annealing
                    let decay_steps = total_steps - warmup_steps;
                    let steps_into_cosine = (current_step - warmup_steps) as f32;
                    let progress = steps_into_cosine / (decay_steps.max(1) as f32);
                    min_lr
                        + 0.5 * (max_lr - min_lr) * (1.0 + (std::f32::consts::PI * progress).cos())
                }
            }
        }
    }
}
