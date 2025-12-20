use burn::{prelude::*, tensor::Distribution};

pub struct MultinomialSampler;

impl MultinomialSampler {
    pub fn new() -> Self {
        Self {}
    }

    pub fn sample<B: Backend>(&self, probs: Tensor<B, 2, Float>) -> Tensor<B, 2, Int> {
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
}

pub struct SimpleSampler;

impl SimpleSampler {
    pub fn new() -> Self {
        Self {}
    }

    pub fn sample<B: Backend>(&self, probs: Tensor<B, 2, Float>) -> Tensor<B, 2, Int> {
        probs.argmax(1)
    }
}
