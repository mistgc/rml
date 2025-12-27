#![recursion_limit = "256"]

use anyhow::Result;
use burn::backend::ndarray::NdArrayDevice;
use burn::backend::wgpu::WgpuDevice;
use burn::backend::{NdArray, Wgpu};
use openai_dive::v1::resources::chat::ChatCompletionParameters;
use rml::models::common::Model;
use rml::models::qwen3::model::Qwen3;

type Backend = Wgpu;
type Device = WgpuDevice;

type NdArrayBackend = NdArray;

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
    let res = model.generate(message)?;
    println!("response: {:?}", serde_json::to_string_pretty(&res)?);

    Ok(())
}

// #[test]
// fn qwen3_generate_ndarray() -> Result<()> {
//     let device = NdArrayDevice::Cpu;
//     let model_path = "ckpts/Qwen/Qwen3-0.6B/";
//     let model = Qwen3::<NdArrayBackend>::new(model_path, &device)
//         .unwrap_or_else(|e| panic!("Initializing model failed: {e:?}"));
//     let message = r#"
//     {
//         "model": "qwen3",
//         "messages": [
//             {
//                 "role": "user",
//                 "content": [
//                     {
//                         "type": "text",
//                         "text": "Hello. How are you?"
//                     }
//                 ]
//             },
//             {
//                 "role": "user",
//                 "content": "Give me a short introduction to large language model."
//             }
//         ]
//     }
//     "#;
//     let message: ChatCompletionParameters = serde_json::from_str(message)?;
//     let res = model.generate(message)?;
//
//     println!("response: {:?}", serde_json::to_string_pretty(&res)?);
//
//     Ok(())
// }
