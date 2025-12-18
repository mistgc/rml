use std::{fs, path::Path};

use anyhow::{Result, anyhow};
use minijinja::{Environment, Value as MinijiajaValue, context};
use openai_dive::v1::resources::chat::ChatCompletionParameters;

#[derive(Debug)]
pub struct ChatTemplate {
    template: String,
}

impl ChatTemplate {
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let tokenizer_config_json =
            fs::read_to_string(path.as_ref().join("tokenizer_config.json"))?;
        let chat_template =
            serde_json::from_str::<serde_json::Value>(&tokenizer_config_json)?["chat_template"]
                .as_str()
                .ok_or(anyhow!("Not found chat template"))?
                .to_owned();
        let fixed_template = chat_template
            .replace(
                "message.content.startswith('<tool_response>')",
                "message.content is startingwith('<tool_response>')", // replace it with `is startingwith` in minijinja
            )
            .replace(
                "message.content.endswith('</tool_response>')",
                "message.content is endingwith('</tool_response>')", // replace it with `is endingwith` in minijinja
            )
            .replace(
                "content.split('</think>')[0].rstrip('\\n').split('<think>')[-1].lstrip('\\n')",
                "((content | split('</think>'))[0] | rstrip('\\n') | split('<think>'))[-1] | lstrip('\\n')", // replace them with `split`, `rstrip` and `lstrip` custom filters
            )
            .replace(
                "content.split('</think>')[-1].lstrip('\\n')",
                "(content | split('</think>'))[-1] | lstrip('\\n')", // replace it with the custom filter
            )
            .replace(
                "reasoning_content.strip('\\n')",
                "reasoning_content | strip('\\n')", // replace it with the custom filter
            )
            .replace(
                "content.lstrip('\\n')",
                "content | lstrip('\\n')", // replace it with the custom filter
            );

        Ok(Self {
            template: fixed_template,
        })
    }

    pub fn render(&self, messages: &ChatCompletionParameters) -> Result<String> {
        let context = context! {
            messages => &messages.messages,
            tools => &messages.tools,
            add_generation_prompt => true,
        };

        let mut env = Environment::new();

        env.add_filter("tojson", |v: MinijiajaValue| {
            serde_json::to_string(&v).unwrap()
        });

        env.add_filter("split", |s: String, delimiter: String| {
            s.split(&delimiter)
                .map(|ss| ss.to_string())
                .collect::<Vec<String>>()
        });

        env.add_filter("lstrip", |s: String, chars: Option<String>| match chars {
            Some(chars_str) => s.trim_start_matches(chars_str.as_str()).to_string(),
            None => s.trim_start().to_string(),
        });

        env.add_filter("rstrip", |s: String, chars: Option<String>| match chars {
            Some(chars_str) => s.trim_end_matches(chars_str.as_str()).to_string(),
            None => s.trim_end().to_string(),
        });

        env.add_template("chat", self.template.as_ref())?;
        let rendered_messages = env
            .get_template("chat")
            .map_err(|e| anyhow!(format!("render template error {}", e)))?
            .render(context)
            .map_err(|e| anyhow!(format!("render template error {}", e)))?;

        Ok(rendered_messages)
    }
}
