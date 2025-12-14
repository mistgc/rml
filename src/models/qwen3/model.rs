use crate::models::common::Model;

use super::config::Qwen3Config;
use anyhow::{Result, anyhow};
use burn::{nn, prelude::*};
use openai_dive::v1::resources::chat::{ChatCompletionParameters, ChatCompletionResponse};
use serde::de::value;
use std::{fs, path::Path};
use tokenizers::Tokenizer;

pub struct Qwen3RMSNorm<B: Backend> {
    variance_epsilon: f32,
    weight: Tensor<B, 1>, // [hidden_size]
}

impl<B: Backend> Qwen3RMSNorm<B> {
    pub fn new(hidden_size: usize, eps: f32, device: &B::Device) -> Self {
        Self {
            variance_epsilon: eps,
            weight: Tensor::ones([hidden_size], device),
        }
    }

    pub fn forward<const D: usize>(&self, hidden_states: Tensor<B, D>) -> Tensor<B, D> {
        let input_dtype = hidden_states.dtype();
        let hidden_states_f32 = hidden_states.cast(burn::tensor::DType::F32);

        // 计算均方值: mean(x^2, dim=-1)，形状保持 [..., 1]
        let variance = hidden_states_f32.clone().powf_scalar(2.0).mean_dim(-1); // shape: [..., 1]

        // 计算 rms: sqrt(variance + epsilon)，形状与 mean_dim 输出一致
        let norm = Tensor::sqrt(variance + self.variance_epsilon); // [..., 1]

        // 归一化: x / rms
        let normalized = hidden_states_f32 / norm;

        // cast 回原始 dtype 并 apply learnable scale (self.weight 自动广播)
        let scaled = self.weight.clone().unsqueeze() * normalized;

        scaled.cast(input_dtype)
    }
}

pub fn rope_init_fn<B: Backend>(config: &Qwen3Config, device: &B::Device) -> (Tensor<B, 1>, f32) {
    let base = Tensor::from_data([config.rope_theta], device);
    let dim = config
        .head_dim
        .unwrap_or(config.hidden_size / config.num_attention_heads) as i64;
    let attention_factor: f32 = 1.0;
    let exp = Tensor::<B, 1, Int>::arange_step(0..dim, 2, device).float();
    let inv_freq = Tensor::from_data([1.0], device).div(base.powf(exp));

    (inv_freq, attention_factor)
}

pub struct Qwen3RotaryEmbedding<B: Backend> {
    rope_theta: usize,
    rope_scaling: Option<usize>,
    original_inv_freq: Tensor<B, 1>,
    attention_scaling: f32,
}

impl<B: Backend> Qwen3RotaryEmbedding<B> {
    pub fn new(config: &Qwen3Config, device: &B::Device) -> Self {
        let (inv_freq, attention_scaling) = rope_init_fn::<B>(config, device);

        Self {
            rope_theta: config.rope_theta,
            rope_scaling: config.rope_scaling.clone(),
            original_inv_freq: inv_freq,
            attention_scaling,
        }
    }

    pub fn forward(
        &self,
        inputs: Tensor<B, 3>,
        position_ids: Tensor<B, 2, Int>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let expanding_shape = [position_ids.shape()[0] as i32, -1, 1];
        let inv_freq_expanded = self
            .original_inv_freq
            .clone()
            .unsqueeze_dims::<3>(&[0, -1])
            .expand(expanding_shape);
        let position_ids_expanded = position_ids.unsqueeze_dims::<3>(&[1]);

        let freqs = inv_freq_expanded
            .matmul(position_ids_expanded.float())
            .swap_dims(1, 2);
        let emb = Tensor::cat(vec![freqs.clone(), freqs], 2);
        let cos = emb.clone().cos() * self.attention_scaling;
        let sin = emb.sin() * self.attention_scaling;

        (cos, sin)
    }
}

pub struct Qwen3Attention<B: Backend> {
    q_proj: nn::Linear<B>,
    k_proj: nn::Linear<B>,
    v_proj: nn::Linear<B>,
    o_proj: nn::Linear<B>,
    q_norm: Qwen3RMSNorm<B>,
    k_norm: Qwen3RMSNorm<B>,
    layer_idx: usize,
    num_key_value_groups: usize,
    head_dim: usize,
    scaling: f32,
}

impl<B: Backend> Qwen3Attention<B> {
    pub fn new(config: &Qwen3Config, layer_idx: usize, device: &B::Device) -> Self {
        let head_dim = config
            .head_dim
            .unwrap_or(config.hidden_size / config.num_attention_heads);
        let num_key_value_groups = config.num_attention_heads / config.num_key_value_heads;

        Self {
            scaling: head_dim.to_f32().powf(-0.5),
            head_dim,
            layer_idx,
            num_key_value_groups,
            q_proj: nn::LinearConfig::new(
                config.hidden_size,
                config.num_attention_heads * head_dim,
            )
            .with_bias(config.attention_bias)
            .init(device),
            k_proj: nn::LinearConfig::new(
                config.hidden_size,
                config.num_key_value_heads * head_dim,
            )
            .with_bias(config.attention_bias)
            .init(device),
            v_proj: nn::LinearConfig::new(
                config.hidden_size,
                config.num_key_value_heads * head_dim,
            )
            .with_bias(config.attention_bias)
            .init(device),
            o_proj: nn::LinearConfig::new(
                config.num_attention_heads * head_dim,
                config.hidden_size,
            )
            .with_bias(config.attention_bias)
            .init(device),
            q_norm: Qwen3RMSNorm::new(head_dim, config.rms_norm_eps, device),
            k_norm: Qwen3RMSNorm::new(head_dim, config.rms_norm_eps, device),
        }
    }

    pub fn forward(
        &self,
        hidden_states: Tensor<B, 3>, // [B, S, H]
        position_embeddings: (Tensor<B, 3>, Tensor<B, 3>),
        position_ids: Tensor<B, 2, Int>,
        use_cache: bool,
        attention_mask: Option<Tensor<B, 2>>,
        past_key_values: Option<Tensor<B, 2>>,
        cache_position: Option<Tensor<B, 1, Int>>,
    ) -> (Tensor<B, 3>, Option<Tensor<B, 4>>) {
        let input_shape = hidden_states.shape(); // [B, S, H]
        let hidden_shape: [i32; 4] = [
            input_shape[0].to_i32(),
            input_shape[1].to_i32(),
            -1,
            self.head_dim.to_i32(),
        ]; // [B, S, -1, D]

        let query_states = self
            .q_norm
            .forward(
                self.q_proj
                    .forward(hidden_states.clone())
                    .reshape(hidden_shape),
            )
            .swap_dims(1, 2); // [B, N, S, D]
        let key_states = self
            .k_norm
            .forward(
                self.k_proj
                    .forward(hidden_states.clone())
                    .reshape(hidden_shape),
            )
            .swap_dims(1, 2);
        let value_states = self
            .v_proj
            .forward(hidden_states.clone())
            .reshape(hidden_shape)
            .swap_dims(1, 2);

        let (cos, sin) = position_embeddings;
        let (query_states, key_states) =
            apply_rotary_pos_emb(query_states, key_states, cos.clone(), sin.clone(), None);

        let (attn_output, attn_weights) = eager_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            self.scaling,
            None,
        ); // [B, S, N, D]

        let attn_output =
            attn_output.reshape([input_shape[0].to_i32(), input_shape[1].to_i32(), -1]);
        let attn_output = self.o_proj.forward(attn_output);

        (attn_output, Some(attn_weights))
    }
}

pub fn repeat_kv<B: Backend>(hidden_states: Tensor<B, 4>, n: usize) -> Tensor<B, 4> {
    if n == 1 {
        return hidden_states;
    }
    let shape = hidden_states.dims();
    let (batch, num_key_value_heads, seq_len, head_dim) = (shape[0], shape[1], shape[2], shape[3]);

    let hidden_states = hidden_states.reshape([batch, num_key_value_heads, 1, seq_len, head_dim]);

    let hidden_states: Tensor<B, 5> = hidden_states.repeat_dim(2, n);

    hidden_states.reshape([batch, num_key_value_heads * n, seq_len, head_dim])
}

pub fn eager_attention_forward<B: Backend>(
    module: &Qwen3Attention<B>,
    q: Tensor<B, 4>, // [B, N_head, S, D]
    k: Tensor<B, 4>, // [B, N_kv, S, D]
    v: Tensor<B, 4>, // [B, N_kv, S, D]
    attn_mask: Option<Tensor<B, 2>>,
    scaling: f32,
    dropout: Option<&nn::Dropout>,
) -> (Tensor<B, 4>, Tensor<B, 4>) {
    use burn::tensor::activation::softmax;

    let key_states = repeat_kv(k.clone(), module.num_key_value_groups); // [B, N_kv, S, D] -> [B, N_head, S, D]
    let value_states = repeat_kv(v.clone(), module.num_key_value_groups); // [B, N_kv, S, D] -> [B, N_head, S, D]
    let mut attn_weights = Tensor::matmul(q, key_states.swap_dims(2, 3)).mul_scalar(scaling); // [B, N_head, S, S]

    if let Some(attn_mask) = attn_mask {
        attn_weights = attn_weights + attn_mask.unsqueeze_dim(2);
    }

    let mut attn_weights = softmax(attn_weights, 3);

    if let Some(dropout) = dropout {
        attn_weights = dropout.forward(attn_weights);
    }

    let attn_output = Tensor::matmul(attn_weights.clone(), value_states); // [B, N_head, S, D]
    let attn_output = attn_output.swap_dims(1, 2); // [B, S, N_head, D]

    (attn_output, attn_weights) // [B, S, N_head, D], [B, N_head, S, S]
}

pub fn rotate_half<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    let chunks = x.chunk(2, D - 1);
    let x1 = chunks[0].clone();
    let x2_neg = chunks[1].clone().neg();

    Tensor::cat(vec![x2_neg, x1], D - 1)
}

pub fn apply_rotary_pos_emb<B: Backend>(
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    cos: Tensor<B, 3>,
    sin: Tensor<B, 3>,
    unsqueeze_dim: Option<usize>,
) -> (Tensor<B, 4>, Tensor<B, 4>) {
    let unsqueeze_dim = unsqueeze_dim.unwrap_or(1);
    let cos = cos.unsqueeze_dim(unsqueeze_dim);
    let sin = sin.unsqueeze_dim(unsqueeze_dim);

    let q_embed = (q.clone() * cos.clone()) + (rotate_half(q) * sin.clone());
    let k_embed = (k.clone() * cos) + (rotate_half(k) * sin);

    (q_embed, k_embed)
}

pub struct Qwen3MLP<B: Backend, const D: usize> {
    hidden_size: usize,
    intermidiate_size: usize,
    gate_proj: nn::Linear<B>,
    up_proj: nn::Linear<B>,
    down_proj: nn::Linear<B>,
    act_fn: fn(Tensor<B, D>) -> Tensor<B, D>,
}

impl<B: Backend, const D: usize> Qwen3MLP<B, D> {
    pub fn new(config: &Qwen3Config, device: &B::Device) -> Self {
        Self {
            hidden_size: config.hidden_size,
            intermidiate_size: config.intermediate_size,
            gate_proj: nn::LinearConfig::new(config.hidden_size, config.intermediate_size)
                .with_bias(false)
                .init(device),
            up_proj: nn::LinearConfig::new(config.hidden_size, config.intermediate_size)
                .with_bias(false)
                .init(device),
            down_proj: nn::LinearConfig::new(config.intermediate_size, config.hidden_size)
                .with_bias(false)
                .init(device),
            act_fn: burn::tensor::activation::silu::<D, B>,
        }
    }

    pub fn forward(&self, inputs: Tensor<B, D>) -> Tensor<B, D> {
        let x = self.gate_proj.forward(inputs.clone());
        let x = (self.act_fn)(x);
        let x = x * self.up_proj.forward(inputs);
        let y = self.down_proj.forward(x);

        y
    }
}

pub struct Qwen3DecoderLayerInput<B: Backend> {
    hidden_states: Tensor<B, 3>,
    position_embeddings: (Tensor<B, 3>, Tensor<B, 3>),
    position_ids: Tensor<B, 2, Int>,
    use_cache: bool,
    attention_mask: Option<Tensor<B, 2>>,
    cache_position: Option<Tensor<B, 1, Int>>,
    past_key_values: Option<Tensor<B, 2>>,
}

pub struct Qwen3DecoderLayer<B: Backend> {
    hidden_size: usize,
    self_attn: Qwen3Attention<B>,
    mlp: Qwen3MLP<B, 3>,
    input_layernorm: Qwen3RMSNorm<B>,
    post_attention_layernorm: Qwen3RMSNorm<B>,
}

impl<B: Backend> Qwen3DecoderLayer<B> {
    pub fn new(config: &Qwen3Config, layer_idx: usize, device: &B::Device) -> Self {
        Self {
            hidden_size: config.hidden_size,
            self_attn: Qwen3Attention::<B>::new(config, layer_idx, device),
            mlp: Qwen3MLP::<B, 3>::new(config, device),
            input_layernorm: Qwen3RMSNorm::<B>::new(
                config.hidden_size,
                config.rms_norm_eps,
                device,
            ),
            post_attention_layernorm: Qwen3RMSNorm::<B>::new(
                config.hidden_size,
                config.rms_norm_eps,
                device,
            ),
        }
    }

    pub fn forward(&self, inputs: Qwen3DecoderLayerInput<B>) -> burn::Tensor<B, 3> {
        // inputs.hidden_states: [B, S, H]
        let residual = inputs.hidden_states.clone();
        let hidden_states = self.input_layernorm.forward(inputs.hidden_states);
        let (attn_output, _) = self.self_attn.forward(
            hidden_states,
            inputs.position_embeddings,
            inputs.position_ids,
            inputs.use_cache,
            inputs.attention_mask,
            inputs.past_key_values,
            inputs.cache_position,
        ); // [B, S, H]

        let attn_output = residual + attn_output; // [B, S, H]

        let residual = attn_output.clone();
        let mlp_input = self.post_attention_layernorm.forward(attn_output); // [B, S, H]
        let mlp_output = self.mlp.forward(mlp_input);
        let output = residual + mlp_output;

        output // [B, S, H]
    }
}

pub struct Qwen3Model<B: Backend> {
    padding_idx: usize,
    vocab_size: usize,
    embed_tokens: nn::Embedding<B>,
    layers: Vec<Qwen3DecoderLayer<B>>,
    norm: Qwen3RMSNorm<B>,
    rotary_emb: Qwen3RotaryEmbedding<B>,
    gradient_checkpointing: bool,
    device: B::Device,
}

impl<B: Backend> Qwen3Model<B> {
    pub fn new(config: &Qwen3Config, device: &B::Device) -> Self {
        let mut layers: Vec<Qwen3DecoderLayer<B>> = vec![];

        for idx in 0..config.num_hidden_layers {
            layers.push(Qwen3DecoderLayer::<B>::new(config, idx, device));
        }

        Self {
            device: device.clone(),
            padding_idx: config.pad_token_id,
            vocab_size: config.vocab_size,
            embed_tokens: nn::EmbeddingConfig::new(config.vocab_size, config.hidden_size)
                .init::<B>(device),
            layers,
            norm: Qwen3RMSNorm::<B>::new(config.hidden_size, config.rms_norm_eps, device),
            rotary_emb: Qwen3RotaryEmbedding::<B>::new(config, device),
            gradient_checkpointing: false,
        }
    }

    pub fn forward(&self, inputs: Qwen3Input<B>) -> (Tensor<B, 3>, Option<Tensor<B, 2>>) {
        let inputs_embeds = inputs.inputs_embeds.unwrap_or_else(|| {
            let input_ids = inputs.input_ids.unwrap_or_else(|| {
                panic!("You must specify exactly one of input_ids or inputs_embeds")
            });

            self.embed_tokens.forward(input_ids)
        });

        if inputs.use_cache && inputs.past_key_values.is_none() {
            todo!()
        }

        let cache_position = inputs.cache_position.unwrap_or_else(|| {
            let num_past_seen_tokens: i64 = inputs
                .past_key_values
                .as_ref()
                .map(|kv| kv.dims()[1].to_i64())
                .unwrap_or(0);

            Tensor::<B, 1, Int>::arange(
                num_past_seen_tokens..(num_past_seen_tokens + inputs_embeds.dims()[1].to_i64()),
                &self.device,
            )
        });

        let position_ids = inputs
            .position_ids
            .unwrap_or(cache_position.clone().unsqueeze_dim(0));

        let mut hidden_states = inputs_embeds;
        let position_embeddings = self
            .rotary_emb
            .forward(hidden_states.clone(), position_ids.clone());

        self.layers.iter().for_each(|layer| {
            // FIXME: past_key_values needs to a new struct, and here its reference should be passed in.
            hidden_states = layer.forward(Qwen3DecoderLayerInput {
                hidden_states: hidden_states.clone(),
                position_embeddings: position_embeddings.clone(),
                position_ids: position_ids.clone(),
                use_cache: inputs.use_cache,
                attention_mask: None,
                past_key_values: inputs.past_key_values.clone(),
                cache_position: Some(cache_position.clone()),
            });
        });

        let last_hidden_states = self.norm.forward(hidden_states);
        let past_key_values = None;

        (last_hidden_states, past_key_values)
    }
}

pub struct Qwen3Input<B: Backend> {
    pub input_ids: Option<Tensor<B, 2, Int>>, // [B, S], dtype: u32
    pub attention_mask: Option<Tensor<B, 2>>,
    pub position_ids: Option<Tensor<B, 2, Int>>,
    pub past_key_values: Option<Tensor<B, 2>>,
    pub inputs_embeds: Option<Tensor<B, 3>>, // [B, S, E]
    pub use_cache: bool,
    pub cache_position: Option<Tensor<B, 1, Int>>,
}

pub struct Qwen3Output<B: Backend> {
    logits: Tensor<B, 3>,
    past_key_values: Option<Tensor<B, 2>>,
}

pub struct Qwen3<B: Backend> {
    model: Qwen3Model<B>,
    tokenizer: Tokenizer,
    config: Qwen3Config,
    lm_head: nn::Linear<B>,
    device: B::Device,
}

impl<B: Backend> Qwen3<B> {
    pub fn new<P: AsRef<Path>>(model_path: P, device: &B::Device) -> Result<Self> {
        let config_path = Path::join(model_path.as_ref(), "config.json");
        let tokenizer_path = Path::join(model_path.as_ref(), "tokenizer.json");

        let tokenizer = Qwen3::<B>::create_tokenizer(tokenizer_path)?;
        let config = Qwen3::<B>::create_config(config_path)?;
        let model = Qwen3Model::new(&config, device);
        let lm_head = nn::LinearConfig::new(config.hidden_size, config.vocab_size)
            .with_bias(false)
            .init::<B>(device);

        Ok(Self {
            model,
            tokenizer,
            config: config.clone(),
            lm_head,
            device: device.clone(),
        })
    }

    fn create_config<P: AsRef<Path>>(path: P) -> Result<Qwen3Config> {
        let config_json_str = fs::read_to_string(path)?;
        let config: Qwen3Config = serde_json::from_str(&config_json_str)?;

        Ok(config)
    }

    fn create_tokenizer<P: AsRef<Path>>(path: P) -> Result<Tokenizer> {
        let tokenizer = Tokenizer::from_file(path).map_err(|e| anyhow::anyhow!(e))?;

        Ok(tokenizer)
    }
}

impl<B: Backend> Model for Qwen3<B> {
    type Input = Qwen3Input<B>;
    type Output = Qwen3Output<B>;

    fn forward(&self, input: Self::Input) -> Self::Output {
        let (hidden_states, past_key_values) = self.model.forward(input);
        let logits = self.lm_head.forward(hidden_states);

        Self::Output {
            logits,
            past_key_values,
        }
    }

    fn generate(&self, msgs: ChatCompletionParameters) -> Result<ChatCompletionResponse> {
        for m in msgs.messages {
            if let Some(text) = m.message() {
                let text = text.to_string();
                // TODO: apply chat template
                let input_ids = self
                    .tokenizer
                    .encode(text, true)
                    .map_err(|e| anyhow!(format!("Tokenizer encode err: {}", e)))?
                    .get_ids()
                    .to_vec();
                let seq_len = input_ids.len();
                let tensor_ids = Tensor::<B, 2, Int>::from_data(
                    TensorData::new(input_ids, [1, seq_len]),
                    &self.device,
                );
                let input = Self::Input {
                    input_ids: Some(tensor_ids),
                    attention_mask: None,
                    inputs_embeds: None,
                    position_ids: None,
                    past_key_values: None,
                    use_cache: false,
                    cache_position: None,
                };

                let output = self.forward(input);
                println!("logits: {:?}", output.logits);
            }
        }

        todo!()
    }
}
