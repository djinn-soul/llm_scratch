// ════════════════════════════════════════════════════════════════════════════
// LEARNING RATE SCHEDULER SYSTEM
// ════════════════════════════════════════════════════════════════════════════
//
// Dynamically adjusts the optimizer's learning rate at each training step.
// Swappable strategies help prevent overshooting (unstable training) and
// improve convergence towards the bottom of local minima.
//
// Training loop connection:
//   1. global_step starts at 0
//   2. each optimizer update asks scheduler.get_lr(global_step)
//   3. optimizer uses that value for exactly one parameter update
//   4. global_step increments
//
// This means the schedule is step-based, not epoch-based. If an epoch has 200
// batches, the scheduler advances 200 times during that epoch.
//
// Reference:
//   - https://machinelearningmastery.com/a-gentle-introduction-to-learning-rate-schedulers/
// ════════════════════════════════════════════════════════════════════════════

/// Learning rate scheduling strategies.
#[derive(Debug, Clone, Copy)]
pub enum LRScheduler {
    /// Keeps a fixed learning rate for the entire training duration.
    /// Default standard SGD/Adam behavior.
    ///
    /// Example:
    ///   lr = 0.001 for step 0, step 1, step 2, ... forever.
    ///
    /// Learning intuition:
    ///   Simple and predictable, but cannot be gentle at the beginning or slow
    ///   down near the end of training.
    Constant { lr: f32 },

    /// Drops the learning rate by a factor `decay_rate` every `step_size` steps.
    ///
    /// Formula:
    ///   lr = initial_lr * decay_rate ^ floor(step / step_size)
    ///
    /// Example:
    ///   initial_lr = 0.01, decay_rate = 0.5, step_size = 10
    ///
    ///   steps  0..=9  -> lr = 0.01
    ///   steps 10..=19 -> lr = 0.005
    ///   steps 20..=29 -> lr = 0.0025
    ///
    /// Learning intuition:
    ///   This is a staircase. The model trains with one LR for a while, then
    ///   suddenly takes smaller steps after each boundary.
    StepDecay {
        initial_lr: f32,
        decay_rate: f32,
        step_size: usize,
    },

    /// Smoothly decays the learning rate at every step by an exponential factor.
    ///
    /// Formula:
    ///   lr = initial_lr * decay_rate ^ step
    ///
    /// Example:
    ///   initial_lr = 0.01, decay_rate = 0.99
    ///
    ///   step 0 -> 0.010000
    ///   step 1 -> 0.009900
    ///   step 2 -> 0.009801
    ///
    /// Learning intuition:
    ///   Unlike StepDecay, every update is slightly smaller than the previous
    ///   one. This is smooth, but can shrink too aggressively if decay_rate is
    ///   far below 1.0.
    ExponentialDecay { initial_lr: f32, decay_rate: f32 },

    /// State-of-the-art LLM schedule: Linear Warmup followed by Cosine Annealing.
    ///
    /// Formula:
    ///   - warmup: lr = max_lr * step / warmup_steps
    ///   - cosine: lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(pi * progress))
    ///
    /// Example shape:
    ///   step 0                 -> near 0.0
    ///   step warmup_steps      -> max_lr
    ///   step total_steps       -> near min_lr
    ///
    /// Learning intuition:
    ///   Warmup protects early training while random weights produce unstable
    ///   gradients. Cosine decay then lowers LR smoothly, so later updates
    ///   refine instead of jumping around.
    CosineWarmup {
        max_lr: f32,
        min_lr: f32,
        warmup_steps: usize,
        total_steps: usize,
    },
}

impl LRScheduler {
    pub fn get_lr(&self, current_step: usize) -> f32 {
        // Input:
        //   current_step = number of optimizer updates already completed.
        //
        // Output:
        //   learning rate to use for the next optimizer update.
        match *self {
            // ── STRATEGY 1: CONSTANT ───────────────────────────────────────
            // No math beyond returning the configured value.
            //
            // Graph:
            //   lr
            //   |
            //   |──────────────
            //   +-------------- step
            LRScheduler::Constant { lr } => lr,

            // ── STRATEGY 2: STEP DECAY ────────────────────────────────────
            // Integer division performs floor(step / step_size).
            //
            // Example with step_size=10:
            //   current_step  0 / 10 = 0 decays
            //   current_step  9 / 10 = 0 decays
            //   current_step 10 / 10 = 1 decay
            //   current_step 20 / 10 = 2 decays
            LRScheduler::StepDecay {
                initial_lr,
                decay_rate,
                step_size,
            } => {
                let num_decays = current_step / step_size;
                initial_lr * (decay_rate.powi(num_decays as i32))
            }

            // ── STRATEGY 3: EXPONENTIAL DECAY ──────────────────────────────
            // Each step multiplies the original LR by another copy of
            // decay_rate. If decay_rate is 0.99, the LR keeps 99% of its
            // previous scale at each update.
            LRScheduler::ExponentialDecay {
                initial_lr,
                decay_rate,
            } => initial_lr * decay_rate.powi(current_step as i32),

            // ── STRATEGY 4: LINEAR WARMUP + COSINE DECAY ──────────────────
            // Used by many transformer/LLM training setups.
            LRScheduler::CosineWarmup {
                max_lr,
                min_lr,
                warmup_steps,
                total_steps,
            } => {
                if current_step < warmup_steps {
                    // PHASE A: LINEAR WARMUP
                    //
                    // Move from 0 toward max_lr in a straight line:
                    //   step=0             -> 0
                    //   step=warmup_steps  -> max_lr
                    //
                    // This avoids a large first update when model weights are
                    // random and gradients can be noisy.
                    max_lr * (current_step as f32 / warmup_steps as f32)
                } else {
                    // PHASE B: COSINE ANNEALING
                    //
                    // progress measures how far we are through the decay phase:
                    //   0.0 -> just finished warmup
                    //   1.0 -> reached total_steps
                    //
                    // cos(pi * progress):
                    //   progress=0.0 -> cos(0)  =  1, lr near max_lr
                    //   progress=1.0 -> cos(pi) = -1, lr near min_lr
                    let decay_steps = total_steps - warmup_steps;
                    let steps_into_cosine = (current_step - warmup_steps) as f32;

                    // max(1) prevents division by zero when total_steps equals
                    // warmup_steps. In that edge case, the decay phase has no
                    // real length, so we use one step as a safe denominator.
                    let progress = steps_into_cosine / (decay_steps.max(1) as f32);
                    min_lr
                        + 0.5 * (max_lr - min_lr) * (1.0 + (std::f32::consts::PI * progress).cos())
                }
            }
        }
    }
}
