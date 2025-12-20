use std::path::Path;

use anyhow::Result;
use openai_dive::v1::resources::chat::{ChatCompletionParameters, ChatCompletionResponse};

pub trait Model {
    fn generate(&self, msgs: ChatCompletionParameters) -> Result<ChatCompletionResponse>;
}
