#include "Model.hpp"
#include "LuaScripting.hpp"
#include "Helpers.hpp"
#include "HtopDisplay.hpp"
#include "include/json.hpp"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <cstring>

using json = nlohmann::json;
namespace fs = std::filesystem;

void printUsage(const char* prog) {
    std::cout << "Usage: " << prog << " [OPTIONS]\n";
    std::cout << "\nOptions:\n";
    std::cout << "  --script <file.lua>      Execute Lua script\n";
    std::cout << "  --config <config.json>   Load configuration file\n";
    std::cout << "  --help                   Show this help message\n";
    std::cout << "\nExamples:\n";
    std::cout << "  " << prog << " --script examples/train.lua\n";
    std::cout << "  " << prog << " --config config.json --script workflow.lua\n";
}

int main(int argc, char **argv)
{
    std::cout << "╔════════════════════════════════════════╗\n";
    std::cout << "║          Mímir Framework v1.0          ║\n";
    std::cout << "║   Deep Learning C++ + Lua Scripting    ║\n";
    std::cout << "╚════════════════════════════════════════╝\n\n";
    
    std::string script_path;
    std::string config_path;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--script") == 0 && i + 1 < argc) {
            script_path = argv[++i];
        }
        else if (std::strcmp(argv[i], "--config") == 0 && i + 1 < argc) {
            config_path = argv[++i];
        }
        else if (std::strcmp(argv[i], "--help") == 0) {
            printUsage(argv[0]);
            return 0;
        }
        else {
            std::cerr << "Unknown option: " << argv[i] << "\n";
            printUsage(argv[0]);
            return 1;
        }
    }
    
    // If no script provided, run interactive mode
    if (script_path.empty()) {
        std::cout << "ℹ️  No script provided. Starting interactive Lua mode...\n";
        std::cout << "   Use --script <file.lua> to run a script\n";
        std::cout << "   Use --help for more options\n\n";
        
        // Simple REPL
        LuaScripting lua;
        
        std::cout << "Lua> ";
        std::string line;
        while (std::getline(std::cin, line)) {
            if (line == "exit" || line == "quit") break;
            
            if (!line.empty()) {
                lua.executeScript(line);
            }
            
            std::cout << "Lua> ";
        }
        
        std::cout << "\n👋 Goodbye!\n";
        return 0;
    }
    
    // Load configuration if provided
    if (!config_path.empty()) {
        std::cout << "📋 Loading configuration: " << config_path << "\n";
        try {
            std::ifstream f(config_path);
            json config;
            f >> config;
            
            auto& ctx = LuaContext::getInstance();
            ctx.currentConfig = config;
            
            std::cout << "✓ Configuration loaded\n\n";
        } catch (const std::exception& e) {
            std::cerr << "❌ Error loading config: " << e.what() << "\n";
            return 1;
        }
    }
    
    // Execute Lua script
    std::cout << "🚀 Executing script: " << script_path << "\n";
    std::cout << std::string(50, '-') << "\n\n";
    
    LuaScripting lua;
    
    if (!lua.loadScript(script_path)) {
        std::cerr << "❌ Failed to execute script\n";
        return 1;
    }
    
    std::cout << "\n" << std::string(50, '-') << "\n";
    std::cout << "✓ Script completed successfully\n";
    
    // Display logs
    auto& ctx = LuaContext::getInstance();
    if (!ctx.logs.empty()) {
        std::cout << "\n📝 Execution logs:\n";
        for (const auto& log : ctx.logs) {
            std::cout << "   " << log << "\n";
        }
    }
    
    return 0;
}
