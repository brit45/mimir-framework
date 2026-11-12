#include "scriptings/Rust/rustScripting/RustScripting.hpp"
#include "scriptings/Lua/luaScripting/LuaScripting.hpp"
#include "scriptings/NativeBridge.hpp"
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>

int main() {
    if (std::system("command -v rustc >/dev/null 2>&1") != 0) return 77;
    try {
        NativeBridge::TemporaryDirectory artifacts;
        std::filesystem::create_directory(artifacts.path/"dataset");
        {std::ofstream out(artifacts.path/"dataset/tiny.ppm",std::ios::binary);out<<"P6\n2 1\n255\n";const unsigned char rgb[]={255,0,0,0,255,0};out.write(reinterpret_cast<const char*>(rgb),sizeof(rgb));}
        {std::ofstream out(artifacts.path/"dataset/tiny.txt");out<<"bonjour monde";}
        auto& ctx=RustContext::getInstance();ctx.resetRuntimeState();
        auto& lua=LuaContext::getInstance();const auto original=lua.currentModel;
        RustScripting rs;rs.setArgs("test.rs",{"argument with spaces", "é漢字"});
        rs.setSystemConfig(json{{"marker",42},{"artifacts",artifacts.path.string()}},"/tmp/conf.json","/tmp");
        if (!rs.executeScript(R"RS(
assert_eq!(arg()?[1].as_str(),Some("argument with spaces"));
assert_eq!(arg()?[2].as_str(),Some("é漢字"));
assert_eq!(conf()?["marker"].as_i64(),Some(42));
assert_eq!(conf_path()?.as_str(),Some("/tmp/conf.json"));
assert_eq!(conf_dir()?.as_str(),Some("/tmp"));
assert!(Value::parse("\"\\ud83d\\ude00\"")?.as_str()==Some("😀"));
assert!(Value::parse("[1,]").is_err());
assert!(Value::parse("01").is_err());
assert!(Value::parse("1e999").is_err());
assert_eq!(Value::from(u64::MAX).stringify()?,"18446744073709551615");
let raw=results("Architectures.default_config", &["__missing__".into()])?;
assert!(raw[0].is_null() && raw[1].as_str().is_some());
assert!(Architectures::default_config(&["__missing__".into()]).is_err());
let config=Value::object(&[("input_dim",4.into()),("hidden_dim",8.into()),("output_dim",2.into())]);
Mimir::Model::create(&["basic_mlp".into(),config])?;
model::set_hardware(&[false.into()])?;
Model::allocate_params(&[])?;Model::init_weights(&["he".into(),42.into()])?;
assert!(Model::total_params(&[])?.as_u64().unwrap()>0);
let workers:Vec<_>=(0..4).map(|_| std::thread::spawn(|| Model::total_params(&[]))).collect();
for worker in workers { assert!(worker.join().unwrap()?.as_u64().unwrap()>0); }
assert_eq!(Layers::Linear(&[])?.as_array().unwrap().len(),3);
let input=floats(&[0.1,0.2,0.3,0.4])?;
assert_eq!(Model::forward(&[input.clone(),true.into()])?.as_array().unwrap().len(),2);
Model::zero_grads(&[])?;Model::backward(&[floats(&[0.1,-0.2])?])?;
Model::optimizer_step(&[0.001.into(),"adamw".into()])?;
let trained=Model::forward(&[input.clone(),false.into()])?;
let base=conf()?["artifacts"].as_str().unwrap().to_string();
let checkpoint=format!("{}/checkpoint",base);
Serialization::save(&[checkpoint.clone().into(),"raw_folder".into()])?;
Model::init_weights(&["he".into(),123.into()])?;
Serialization::load(&[checkpoint.into(),"raw_folder".into()])?;
assert_eq!(Model::forward(&[input,false.into()])?,trained);
Tokenizer::create(&[256.into()])?;
Tokenizer::tokenize_ensure(&["bonjour monde".into()])?;
let tokens=Tokenizer::tokenize(&["bonjour monde".into()])?;
assert!(!tokens.as_array().unwrap().is_empty());
assert!(Tokenizer::detokenize(&[tokens])?.as_str().is_some());
let dataset=format!("{}/dataset",base);
let loaded=Dataset::load(&[dataset.clone().into(),2.into(),1.into(),2.into()])?;
assert_eq!(loaded[0].as_bool(),Some(true));assert_eq!(loaded[1].as_i64(),Some(1));
let item=Dataset::get(&[1.into()])?;
assert_eq!(item["text"].as_str(),Some("bonjour monde"));
assert_eq!(item["image"].as_array().unwrap().len(),6);
Dataset::release(&[1.into()])?;
let builder=Database::load(&["/missing/path".into()])?;
let failed=builder["cache"].results(&[])?;assert_eq!(failed[0].as_bool(),Some(false));
let marker=Value::object(&[("$mimir_function",1.into()),("text","é漢字 | \n".into()),("large","é漢字".repeat(30000).into())]);
let path=format!("{}/marker.json",base);
write_json(&[path.clone().into(),marker.clone()])?;
let read=read_json(&[path.into()])?;
assert_eq!(read["$mimir_function"].as_f64(),Some(1.));
assert_eq!(read["text"],marker["text"]);
assert_eq!(read["large"].as_str().unwrap().len(),"é漢字".len()*30000);
assert!(Memory::get_stats(&[])?.as_object().is_some());
assert!(Viz::backend(&[])?.as_str().is_some());
assert!(call("Model.forward", &[Value::Array(vec![f64::NAN.into()])]).is_err());
assert!(call("Tokenizer.create", &[u64::MAX.into()]).is_err());
let integral=Model::forward(&[floats(&[1.,0.,1.,0.])?,false.into()])?;
write_json(&[format!("{}/prediction.json",base).into(),integral])?;
println!("PASS Rust synchronous parity");
)RS")) throw std::runtime_error("Rust API parity failed");
        if(lua.currentModel!=original)throw std::runtime_error("Rust changed Lua context");
        {std::ifstream f(artifacts.path/"prediction.json");json stored;f>>stored;
         const auto ref=ctx.currentModel->forwardPass(std::vector<float>{1,0,1,0},false);
         if(stored.get<std::vector<float>>()!=ref)throw std::runtime_error("Rust native forward mismatch");}
        if(!ctx.currentTokenizer || !rs.executeScript("assert!(Tokenizer::vocab_size(&[])?.as_u64().unwrap()>0);"))throw std::runtime_error("lost Rust context");
        if(rs.executeScript("return Err(mimir::Error(\"intentional\".into()));"))throw std::runtime_error("Rust execution error ignored");
        if(rs.executeScript("invalid syntax !!!"))throw std::runtime_error("Rust compile error ignored");
        // Full source with sibling modules and include! remains valid in a read-only directory.
        artifacts.write("dependency.rs","pub fn answer()->u64 {42}\n");
        const auto full=artifacts.write("full.rs","mod dependency; fn main()->mimir::Result<()> {assert_eq!(dependency::answer(),42); assert!(mimir::Model::total_params(&[])?.as_u64().unwrap()>0); Ok(())}\n");
        if(!rs.loadScript(full))throw std::runtime_error("full Rust source with relative module failed");
        ctx.resetRuntimeState();
        std::cout<<"PASS Rust context persistence, errors, native equality and relative modules\n";
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
