use serde;

#[derive(serde::Deserialize, serde::Serialize, Debug, Clone)]
pub struct Qwen3Config {
    pub vocab_size: usize,

    pub hidden_size: usize,

    pub intermediate_size: usize,

    #[serde(default)]
    pub pad_token_id: usize,

    pub rms_norm_eps: f32,

    pub num_hidden_layers: usize,

    pub hidden_act: String,

    pub rope_theta: usize,

    pub rope_scaling: Option<usize>,

    pub head_dim: Option<usize>,

    pub num_attention_heads: usize,

    pub num_key_value_heads: usize,

    pub attention_bias: bool,

    pub attention_dropout: f32,
}
