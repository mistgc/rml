use std::path::Path;
use modelscope::ModelScope;

fn path_exists<P: AsRef<Path>>(path: P) -> bool {
    path.as_ref().exists()
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("Done.");

    Ok(())
}
