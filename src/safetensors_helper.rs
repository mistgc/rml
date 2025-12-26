use anyhow::{Result, anyhow};
use burn::{
    tensor::TensorData,
};
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};

#[derive(Debug)]
pub struct SafeTensorsHelper {
    path: PathBuf,
    map: HashMap<String, burn::tensor::TensorData>,
}

impl SafeTensorsHelper {
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let data = std::fs::read(path.as_ref())?;
        let tensors = safetensors::SafeTensors::deserialize(&data)?;
        let mut map = HashMap::new();

        for (name, tensor) in tensors.tensors() {
            let burn_tensor = burn::tensor::TensorData::from_bytes_vec(
                tensor.data().to_owned(),
                tensor.shape(),
                SafeTensorsHelper::convert_dtype(tensor.dtype()),
            );

            map.insert(name, burn_tensor);
        }

        Ok(Self {
            path: path.as_ref().to_owned(),
            map,
        })
    }

    pub fn get(&self, key: &str) -> Result<TensorData> {
        let value = self
            .map
            .get(key)
            .ok_or_else(|| anyhow!(format!("The key {} is invalid.", key)))?;
        let tensor = value.clone();

        Ok(tensor)
    }

    fn convert_dtype(dtype: safetensors::Dtype) -> burn::tensor::DType {
        match dtype {
            safetensors::Dtype::BOOL => burn::tensor::DType::Bool,

            safetensors::Dtype::BF16 => burn::tensor::DType::BF16,

            safetensors::Dtype::F16 => burn::tensor::DType::F16,
            safetensors::Dtype::F32 => burn::tensor::DType::F32,
            safetensors::Dtype::F64 => burn::tensor::DType::F64,

            safetensors::Dtype::U8 => burn::tensor::DType::U8,
            safetensors::Dtype::U16 => burn::tensor::DType::U16,
            safetensors::Dtype::U32 => burn::tensor::DType::U32,
            safetensors::Dtype::U64 => burn::tensor::DType::U64,

            safetensors::Dtype::I8 => burn::tensor::DType::I8,
            safetensors::Dtype::I16 => burn::tensor::DType::I16,
            safetensors::Dtype::I32 => burn::tensor::DType::I32,
            safetensors::Dtype::I64 => burn::tensor::DType::I64,

            _ => unimplemented!(),
        }
    }
}

#[test]
fn test_safetensors_helper() {}
