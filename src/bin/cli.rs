#![recursion_limit = "256"]
use burn::backend::wgpu::WgpuDevice;
use burn::backend::Wgpu;
use openai_dive::v1::resources::chat::ChatCompletionParameters;
use rml::models::common::Model;
use rml::models::qwen3::model::Qwen3;
use serde_json;
use std::path::Path;

type Backend = Wgpu;
type Device = WgpuDevice;

fn path_exists<P: AsRef<Path>>(path: P) -> bool {
    path.as_ref().exists()
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let device = Device::default();
    let model_path = "ckpts/Qwen/Qwen3-0.6B/";
    let model = Qwen3::<Backend>::new(model_path, &device)
        .unwrap_or_else(|e| panic!("Initailizing model failed: {e:?}"));
    let message = r#"
    {
        "model": "qwen3",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Hello. How are you?"
                    }
                ]
            },
            {
                "role": "user",
                "content": "Give me a short introduction to large language model."
            }
        ]
    }
    "#;
    let message: ChatCompletionParameters = serde_json::from_str(message)?;
    let res = model.generate(message)?;
    println!("{}", serde_json::to_string_pretty(&res)?);

    Ok(())
}
