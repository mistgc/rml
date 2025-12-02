use burn::{nn, prelude::*};
use serde;

#[derive(serde::Deserialize, serde::Serialize, Debug, Clone)]
struct Qwen3Config {
    vocab_size: usize,
    hidden_size: usize,
    pad_token_id: usize,
    rms_norm_eps: f64,
}

struct Qwen3RMSNorm {}

impl Qwen3RMSNorm {
    pub fn new(hidden_size: usize, eps: f64) -> Self {
        todo!()
    }
}

struct Qwen3RotaryEmbedding {}

impl Qwen3RotaryEmbedding {
    pub fn new(config: &Qwen3Config) -> Self {
        todo!()
    }
}

struct Qwen3Model<B: Backend> {
    padding_idx: usize,
    vocab_size: usize,
    embed_tokens: nn::Embedding<B>,
    layers: Vec<nn::Linear<B>>,
    norm: Qwen3RMSNorm,
    rotary_emb: Qwen3RotaryEmbedding,
    gradient_checkpointing: bool,
    has_sliding_layers: bool,
}

impl<B: Backend> Qwen3Model<B> {
    pub fn new(config: &Qwen3Config, device: &B::Device) -> Self {
        let layers: Vec<nn::Linear<B>> = vec!();
        let has_sliding_layers = false;
        Self {
            padding_idx: config.pad_token_id,
            vocab_size: config.vocab_size,
            embed_tokens: nn::EmbeddingConfig::new(config.vocab_size, config.hidden_size).init::<B>(device),
            layers,
            norm: Qwen3RMSNorm::new(config.hidden_size, config.rms_norm_eps),
            rotary_emb: Qwen3RotaryEmbedding::new(config),
            gradient_checkpointing: false,
            has_sliding_layers,
        }.post_init()
    }

    fn post_init(&mut self) -> Self {
        todo!()
    }
}

pub struct Qwen3<B: Backend> {
    model: Qwen3Model<B>,
    config: Qwen3Config,
    lm_head: nn::Linear<B>,
}

impl<B: Backend> Qwen3<B> {
    pub fn new(config: &Qwen3Config, device: &B::Device) -> Self {
        let model = Qwen3Model::new(config, device);
        let lm_head = nn::LinearConfig::new(config.hidden_size, config.vocab_size)
            .with_bias(false)
            .init::<B>(device);

        Self {
            model,
            config: config.clone(),
            lm_head,
        }.post_init()
    }

    fn post_init(&mut self) -> Self {
        todo!()
    }
}
