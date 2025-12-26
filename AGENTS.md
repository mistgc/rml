# AGENTS.md

This file contains guidelines for agentic coding agents working in the RML (Rust Machine Learning) repository.

## Build and Development Commands

### Core Commands
- `make build` or `cargo build` - Build the project
- `make test` or `cargo test` - Run all tests with backtrace
- `make test tests::test_name` - Run a specific test
- `make check` or `cargo check` - Quick syntax/type checking
- `make clean` or `cargo clean` - Clean build artifacts

### Code Quality
- `make lint` or `cargo clippy --no-deps --all-targets -- -D warnings` - Lint with clippy (warnings are errors)
- `make fmt` or `cargo fmt --all` - Format code with rustfmt

### Running Single Tests
```bash
cargo test test_name
cargo test test_name -- --nocapture  # Show println output
cargo test --package rml --bin mod_name  # Test specific binary
```

## Project Structure

This is a Rust ML inference library with the following key components:
- `src/models/` - Model implementations (currently Qwen3)
- `src/utils/` - Utilities (samplers, etc.)
- `src/chat_template.rs` - Chat template handling
- `src/safetensors_helper.rs` - SafeTensors file operations
- `src/bin/` - CLI tools
- `tests/` - Integration tests

## Code Style Guidelines

### Imports and Dependencies
- Use `anyhow::Result` for error handling throughout
- Group imports logically: std → external → local modules
- Prefer `use` statements over fully qualified paths for frequently used items
- Use `burn` framework for ML operations with `Wgpu` backend

```rust
use anyhow::{Result, anyhow};
use burn::prelude::*;
use crate::{
    models::common::Model,
    utils::sampler::MultinomialSampler,
};
```

### Type System and Generics
- Use generic backend parameter `B: Backend` for ML modules
- Define type aliases for backend/device combinations:
```rust
type Backend = Wgpu;
type Device = WgpuDevice;
```
- Use burn's tensor types: `Tensor<B, D, Float>` and `Tensor<B, D, Int>`

### Naming Conventions
- Structs: `PascalCase` (e.g., `Qwen3RMSNorm`, `MultinomialSampler`)
- Functions/variables: `snake_case`
- Constants: `SCREAMING_SNAKE_CASE`
- Private fields: `snake_case`
- Public APIs: Descriptive names, avoid abbreviations

### Module Organization
- Each model gets its own subdirectory with `mod.rs`, `model.rs`, `config.rs`, `preprocessor.rs`
- Use `pub mod` and `pub use` to create clean public APIs
- Keep `mod.rs` files minimal with only re-exports

### Error Handling
- Use `anyhow::Result<()>` for functions that may fail
- Use `anyhow!(format!("message {}", e))` for context-rich errors
- Use `unwrap_or_else(|e| panic!("message: {:?}", e))` for initialization failures
- Return `Result` types, never panic in production code

### ML-Specific Patterns
- Model structures should derive `Module` and `Debug`
- Use `Param<Tensor<B, D>>` for learnable parameters
- Implement `new()` functions with device parameter
- Use proper tensor casting and dtype handling

```rust
#[derive(Module, Debug)]
pub struct MyLayer<B: Backend> {
    weight: Param<Tensor<B, 2>>,
}

impl<B: Backend> MyLayer<B> {
    pub fn new(device: &B::Device) -> Self {
        Self {
            weight: Param::initialized(ParamId::new(), Tensor::zeros([in_dim, out_dim], device)),
        }
    }
}
```

### Testing Guidelines
-Integration tests go in `tests/` directory
- Unit tests go in `src/` modules with `#[cfg(test)]`
- Use `#[test]` attributes, tests should return `Result<()>`
- Test with actual model paths in `ckpts/` directory
- Use descriptive test names: `qwen3_generate()`

### Traits and Interfaces
- Define common interfaces in `models/common.rs`
- Use traits for pluggable components (sampler, model, etc.)
- Keep trait methods focused and minimal

### Documentation
- Add doc comments to public APIs using `///`
- Use `# Example` sections for complex usages
- Include parameter descriptions and return value information

### File I/O and Paths
- Use `std::path::Path` and `AsRef<Path>` for file operations
- Use `fs::read_to_string()` for file reading
- Handle file not found scenarios gracefully

### Async Runtime
- Use `#[tokio::main]` for async main functions
- Tokio features are enabled with `full`

## Dependencies Overview

Core ML framework: `burn` with `wgpu` backend
Serialization: `serde`, `serde_json`
Error handling: `anyhow`
Tokenization: `tokenizers`
Template rendering: `minijinja`
HTTP/API: `openai_dive`, `modelscope`
File formats: `safetensors`

## Model Development Workflow

1. Add new model directory under `src/models/`
2. Implement `Model` trait from `models/common.rs`
3. Add configuration struct in `config.rs`
4. Implement preprocessor if needed
5. Add integration tests in `tests/`
6. Update `models/mod.rs` exports
7. Follow existing pattern for loading from SafeTensors files

Remember: This is a GPU-accelerated inference library - always pass `&Device` to model constructors and tensor operations.