// IDE-checkable full-file equivalent of --rust statement scripts.
// The runtime injects the mimir module. This path provides it to rust-analyzer.
#[allow(dead_code, non_snake_case, unused_imports)]
#[path = "src/lib.rs"]
mod mimir;
// The generated SDK uses crate::Value and crate::call when embedded as a crate.
pub use mimir::*;
fn main() -> mimir::Result<()> {
    let cfg = Architectures::default_config(&["basic_mlp".into()])?;
    Model::create(&["basic_mlp".into(), cfg])?;
    Model::allocate_params(&[])?;
    Model::init_weights(&["he".into(), 42.into()])?;
    println!("{:?}", Model::total_params(&[])?);
    Ok(())
}
