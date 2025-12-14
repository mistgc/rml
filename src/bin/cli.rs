use modelscope::ModelScope;
use std::path::Path;

fn path_exists<P: AsRef<Path>>(path: P) -> bool {
    path.as_ref().exists()
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let model_id = "Qwen/Qwen3-0.6B";
    let ckpts_dir = Path::new("ckpts").join(model_id);
    if !path_exists(&ckpts_dir) {
        ModelScope::download(model_id, ckpts_dir).await?;
    }

    println!("Done.");

    Ok(())
}
