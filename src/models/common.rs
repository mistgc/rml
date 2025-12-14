use openai_dive::v1::resources::chat::{ChatCompletionParameters, ChatCompletionResponse};
use anyhow::Result;

pub trait Model {
    type Input;
    type Output;

    fn forward(&self, input: Self::Input) -> Self::Output;
    fn generate(&self, msgs: ChatCompletionParameters) -> Result<ChatCompletionResponse>;
}
