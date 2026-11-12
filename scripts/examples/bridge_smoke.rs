// Statement scripts are wrapped in fn main() -> mimir::Result<()>.
// Full Rust files with their own fn main() are also accepted.
let available = Architectures::available(&[])?;
println!("[bridge-rs] architectures: {:?}", available);
let cfg = Architectures::default_config(&["basic_mlp".into()])?;
Model::create(&["basic_mlp".into(), cfg])?;
Model::allocate_params(&[])?;
Model::init_weights(&["he".into(), 42.into()])?;
let p = Model::total_params(&[])?;
assert!(p.as_u64().unwrap() > 0);
println!("[bridge-rs] actual native parameters: {:?}", p);
assert!(Architectures::default_config(&["__missing__".into()]).is_err());
println!("[bridge-rs] ok");
