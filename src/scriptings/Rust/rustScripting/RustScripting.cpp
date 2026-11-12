#include "RustScripting.hpp"
#include "scriptings/Lua/luaScripting/LuaScripting.hpp"
#include "scriptings/NativeBridge.hpp"
#include "RustBootstrap.hpp"
#include <fstream>
#include <sstream>
#include <regex>
#include <spawn.h>
#include <sys/wait.h>
#include <cerrno>
extern char** environ;
namespace {
// The namespace wrappers are generated from the actual native manifest each run.
// Generic mimir::call/results also support closures and future nested namespaces.
bool identifier(const std::string& s) {
    return std::regex_match(s, std::regex("[A-Za-z_][A-Za-z_0-9]*"));
}
void module(std::ostream& out, const json& object, const std::string& path) {
    if (!object.is_object()) return;
    for (auto it=object.begin();it!=object.end();++it) {
        if (!identifier(it.key())) continue;
        const auto name="r#"+it.key();
        if (it.value().is_object() && it.value().contains("$mimir_function")) {
            out << "pub fn " << name << "(args: &[crate::mimir::Value]) -> crate::mimir::Result<crate::mimir::Value> { crate::mimir::call("
                << json(path+it.key()).dump() << ", args) }\n";
        } else if (it.value().is_object()) {
            out << "#[allow(non_snake_case)] pub mod " << name << " {\n";
            module(out,it.value(),path+it.key()+".");out << "}\n";
        }
    }
}
// Ignore comments and literals when distinguishing a full crate from statements.
bool hasMain(const std::string& source) {
    std::string tokens;
    for (size_t i=0;i<source.size();) {
        if (source.compare(i,2,"//")==0) {
            const auto end=source.find('\n',i);i=end==std::string::npos?source.size():end;tokens+=' ';continue;
        }
        if (source.compare(i,2,"/*")==0) {
            int depth=1;i+=2;
            while(i<source.size() && depth) {
                if(source.compare(i,2,"/*")==0){++depth;i+=2;}
                else if(source.compare(i,2,"*/")==0){--depth;i+=2;}
                else ++i;
            }
            tokens+=' ';continue;
        }
        if (source[i]=='r') {
            size_t quote=i+1;while(quote<source.size() && source[quote]=='#')++quote;
            if(quote<source.size() && source[quote]=='"') {
                const std::string end="\""+std::string(quote-i-1,'#');
                const auto close=source.find(end,quote+1);i=close==std::string::npos?source.size():close+end.size();tokens+=' ';continue;
            }
        }
        if(source[i]=='"') {
            ++i;while(i<source.size()){if(source[i]=='\\'){i+=2;continue;}if(source[i++]=='"')break;}
            tokens+=' ';continue;
        }
        tokens+=source[i++];
    }
    return std::regex_search(tokens,std::regex("\\bfn\\s+main\\s*\\("));
}
bool compile(const std::string& source, const std::string& output) {
    std::vector<std::string> args{"rustc","--edition=2021","--crate-name","mimir_script","-C","opt-level=1",source,"-o",output};
    std::vector<char*> argv;for(auto& a:args)argv.push_back(a.data());argv.push_back(nullptr);
    pid_t pid;
    if (posix_spawnp(&pid,"rustc",nullptr,nullptr,argv.data(),environ)) return false;
    int status=0;pid_t waited;
    do {waited=waitpid(pid,&status,0);}while(waited<0 && errno==EINTR);
    return waited==pid && WIFEXITED(status) && WEXITSTATUS(status)==0;
}
}
RustScripting::RustScripting() : native_(std::make_unique<LuaScripting>()) {}
RustScripting::~RustScripting() = default;
void RustScripting::setArgs(const std::string& path,const std::vector<std::string>& args) {cacheArgs(path,args);}
void RustScripting::setSystemConfig(const json& conf,const std::string& path,const std::string& dir) {
    RustContext::getInstance().currentConfig=conf;conf_path_=path;conf_dir_=dir;
}
bool RustScripting::loadScript(const std::string& path) {
    std::ifstream file(path);
    if (!file) {RustContext::getInstance().addLog("[rust] unable to open script: "+path);return false;}
    std::ostringstream code;code<<file.rdbuf();file.close();
    return runScript(code.str(),std::filesystem::absolute(path).string());
}
bool RustScripting::executeScript(const std::string& code) {return runScript(code,"");}
bool RustScripting::runScript(const std::string& code,const std::string& source_file) {
    auto& ctx=RustContext::getInstance();
    try {
        NativeBridge::TemporaryDirectory temporary;
        return NativeBridge::execute(*native_,ctx,scriptPath(),scriptArgs(),ctx.currentConfig,
            conf_path_,conf_dir_,[&](const json& globals) {
                std::ostringstream sdk;sdk<<kMimirRustClient<<"\n";
                module(sdk,globals.at("Mimir"),"");
                sdk<<"#[allow(non_snake_case)] pub mod Mimir { pub use super::*; }\n";
                for (const auto& pair : {std::pair<const char*,const char*>{"architectures","Architectures"}, {"tokenizer","Tokenizer"}, {"dataset","Dataset"}, {"htop","Htop"}, {"viz","Viz"}})
                    sdk<<"pub use "<<pair.second<<" as "<<pair.first<<";\n";
                for (const auto* name : {"log","read_json","write_json"})
                    sdk<<"pub fn "<<name<<"(args: &[Value]) -> Result<Value> { call(\""<<name<<"\", args) }\n";
                temporary.write("mimir.rs",sdk.str());
                // Full Rust crates keep module and include paths relative to their original source.
                // Statement snippets retain the previous convenient --rust file convention.
                std::string user=code;
                if (user.rfind("#!",0)==0 && user.rfind("#![",0)!=0) user.erase(0,user.find('\n'));
                const bool full=hasMain(user);
                std::ostringstream entry;
                entry<<"#[allow(dead_code, non_snake_case, unused_imports)] #[path = "<<json((temporary.path/"mimir.rs").string()).dump()<<"] pub mod mimir;\nuse mimir::*;\n";
                if (full && !source_file.empty()) entry<<"include!("<<json(source_file).dump()<<");\n";
                else if (full) entry<<user;
                else entry<<"fn main() -> mimir::Result<()> {\n"<<user<<"\nOk(())\n}\n";
                const auto source=temporary.write("entry.rs",entry.str());
                const auto executable=(temporary.path/"script").string();
                if (!compile(source,executable)) throw std::runtime_error("rustc compilation failed");
                std::vector<std::string> args{executable};args.insert(args.end(),scriptArgs().begin(),scriptArgs().end());return args;
            },"rust");
    } catch(const std::exception& e) {ctx.addLog(std::string("[rust] execution error: ")+e.what());return false;}
}
void RustScripting::registerAPI() {}
