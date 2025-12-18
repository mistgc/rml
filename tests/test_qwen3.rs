use anyhow::Result;
use burn::backend::Wgpu;
use burn::backend::wgpu::WgpuDevice;
use openai_dive::v1::resources::chat::ChatCompletionParameters;
use rml::models::Qwen3;
use rml::models::common::Model;
use rml::models::qwen3::model::Qwen3Input;

type Backend = Wgpu;
type Device = WgpuDevice;

#[test]
fn qwen3_generate() -> Result<()> {
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
    model.generate(message)?;

    Ok(())
}
