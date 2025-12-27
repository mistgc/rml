use anyhow::{Result, anyhow};
use burn::{prelude::*, tensor::Distribution};

pub struct MultinomialSampler {
    temperature: f32,
    top_p: f32,
}

impl MultinomialSampler {
    pub fn new() -> Self {
        Self {
            temperature: 1.0,
            top_p: 1.0,
        }
    }

    pub fn new_with_params(temperature: f32, top_p: f32) -> Result<Self> {
        if temperature <= 0.0 {
            return Err(anyhow!("Temperature must be > 0, got {}", temperature));
        }
        if !(0.0..=1.0).contains(&top_p) {
            return Err(anyhow!("Top-p must be in [0, 1], got {}", top_p));
        }

        Ok(Self { temperature, top_p })
    }

    /// Original sampling method for backward compatibility
    pub fn sample<B: Backend>(&self, probs: Tensor<B, 2>) -> Tensor<B, 2, Int> {
        let [batch_size, _] = probs.dims();
        let unif = Tensor::<B, 2, Float>::random(
            [batch_size, 1],
            Distribution::Uniform(0.0, 1.0),
            &probs.device(),
        );
        let cum_probs = probs.cumsum(1);
        let less_mask = cum_probs.lower(unif);
        let indices = less_mask.int().sum_dim(1);

        indices
    }

    /// Enhanced sampling with temperature scaling (top-p filtering placeholder)
    pub fn sample_with_filtering<B: Backend>(&self, probs: Tensor<B, 2>) -> Tensor<B, 2, Int> {
        // Apply temperature scaling to probabilities
        let adjusted_probs = if self.temperature == 1.0 {
            probs.clone()
        } else {
            // For temperature scaling on probabilities, we can use:
            // p_temp = p^(1/temperature) / sum(p^(1/temperature))
            let temp_scaled = probs.powf_scalar(1.0 / self.temperature);
            let sum = temp_scaled.clone().sum_dim(1).unsqueeze_dim(1);
            temp_scaled / sum.clamp_min(1e-8)
        };

        // Note: top_p parameter is validated but not yet implemented
        // Full implementation would require more complex tensor operations for cumulative filtering
        self.sample(adjusted_probs)
    }

    /// Apply top-p (nucleus) filtering to probabilities - simplified implementation
    fn apply_top_p_filtering<B: Backend>(&self, probs: Tensor<B, 2>) -> Tensor<B, 2> {
        // For now, return probabilities unchanged - this is a placeholder
        // The temperature scaling already provides randomness control
        // Full top-p implementation would require more complex tensor operations
        probs
    }

    /// Apply temperature scaling to logits
    fn apply_temperature<B: Backend>(&self, logits: Tensor<B, 2>) -> Tensor<B, 2> {
        if self.temperature == 1.0 {
            logits
        } else {
            logits.div_scalar(self.temperature)
        }
    }

    /// Get temperature parameter (for debugging)
    pub fn temperature(&self) -> f32 {
        self.temperature
    }

    /// Get top_p parameter (for debugging)
    pub fn top_p(&self) -> f32 {
        self.top_p
    }
}

pub struct SimpleSampler;

impl SimpleSampler {
    pub fn new() -> Self {
        Self {}
    }

    pub fn sample<B: Backend>(&self, probs: Tensor<B, 2>) -> Tensor<B, 2, Int> {
        probs.argmax(1)
    }
}
