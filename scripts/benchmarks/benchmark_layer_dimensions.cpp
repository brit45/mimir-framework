// Direct CPU kernels, with independent forward/backward timings. Run through
// run_mimir.sh with identical thread/ISA settings for before/after comparisons.
#include "runtimes/cpu/RuntimeLayerDispatch.hpp"
#include "include/json.hpp"
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

namespace {
using Clock = std::chrono::steady_clock;
using json = nlohmann::json;

void fill(std::vector<float>& values, float scale = .01f) {
    for (size_t i = 0; i < values.size(); ++i)
        values[i] = float(int((i * 13) % 37) - 18) * scale;
}

void requireFinite(const std::vector<float>& values) {
    for (float value : values)
        if (!std::isfinite(value)) throw std::runtime_error("nonfinite benchmark result");
}

template<class Run, class Prepare>
json measure(int repetitions, Run run, Prepare prepare) {
    prepare(); run(); prepare(); run();
    std::vector<double> samples;
    samples.reserve(repetitions);
    for (int i = 0; i < repetitions; ++i) {
        prepare();
        const auto start = Clock::now();
        run();
        samples.push_back(std::chrono::duration<double,std::milli>(Clock::now() - start).count());
    }
    auto sorted = samples;
    std::sort(sorted.begin(), sorted.end());
    return {{"median_ms", sorted[sorted.size()/2]}, {"samples_ms", samples}};
}
}

int main(int argc, char** argv) {
    try {
        const int dim = argc > 1 ? std::stoi(argv[1]) : 512;
        const int rows = argc > 2 ? std::stoi(argv[2]) : 16;
        const int repetitions = argc > 3 ? std::stoi(argv[3]) : 5;
        const int workers = argc > 4 ? std::stoi(argv[4]) : 1;
        const std::string filter = argc > 5 ? argv[5] : "all";
        if (dim < 1 || dim > 4096 || rows < 1 || rows > 2048 ||
            repetitions < 1 || repetitions > 1000 || workers < 1 || workers > 64)
            throw std::runtime_error("invalid benchmark dimensions");
        std::ofstream dump;
        if (const char* path = std::getenv("MIMIR_LAYER_VALUES")) {
            dump.open(path, std::ios::binary);
            if (!dump) throw std::runtime_error("cannot open value dump");
        }
        auto write = [&](const std::vector<float>& values) {
            requireFinite(values);
            if (dump.is_open())
                dump.write(reinterpret_cast<const char*>(values.data()), values.size()*sizeof(float));
        };
        Mimir::Planning::ExecutionPolicy::CpuScope scope(workers);
        json cases = json::array();
        for (const char* type : {"Linear", "SelfAttention", "CrossAttention", "LayerNorm", "RMSNorm", "Conv2d"}) {
            if (filter != "all" && filter != type) continue;
            Layer layer("probe", type, 0);
            layer.seq_len = rows;
            layer.in_features = layer.out_features = layer.embed_dim = dim;
            layer.num_heads = dim % 8 == 0 ? 8 : 1;
            layer.use_bias = true;
            size_t parameters = 0, input_elements = size_t(rows)*dim;
            if (layer.type_enum == LayerType::Linear) parameters = size_t(dim)*dim + dim;
            // Bias-free attention uses the existing native derivative contract.
            else if (layer.type_enum == LayerType::SelfAttention) parameters = 4ull*dim*dim;
            else if (layer.type_enum == LayerType::CrossAttention) parameters = 4ull*dim*dim + dim;
            else if (layer.type_enum == LayerType::LayerNorm) {
                parameters = 2ull*dim; layer.shape = {rows,dim}; layer.affine = true;
            } else if (layer.type_enum == LayerType::RMSNorm) {
                parameters = dim; layer.shape = {rows,dim};
            } else {
                // Conv2d scales channels separately: 64/128/256 at widths 512/1024/2048.
                layer.in_channels = layer.out_channels = std::max(1,dim/8);
                layer.input_height = layer.input_width = 8;
                layer.kernel_size = 3; layer.padding = 1; layer.stride = 1;
                parameters = size_t(layer.in_channels)*layer.out_channels*9 + layer.out_channels;
                input_elements = size_t(layer.in_channels)*64;
            }
            layer.weights.resize(parameters); fill(layer.weights,.003f);
            layer.grad_weights.resize(parameters);
            std::vector<float> input(input_elements), kv(size_t(rows+3)*dim);
            fill(input,.03f); fill(kv,.02f);
            std::vector<const std::vector<float>*> inputs{&input};
            if (layer.type_enum == LayerType::CrossAttention) inputs.push_back(&kv);
            std::vector<std::vector<float>> output, gradients;
            auto forward = [&] {
                if (!RuntimeLayerDispatch::cpu_forward_layer(inputs,output,layer,true))
                    throw std::runtime_error(std::string(type) + " forward failed");
            };
            forward();
            std::vector<float> upstream(output[0].size()); fill(upstream,.001f);
            const std::vector<const std::vector<float>*> grad_outputs{&upstream};
            auto backward = [&] {
                if (!RuntimeLayerDispatch::cpu_backward_layer(inputs,grad_outputs,gradients,layer,true))
                    throw std::runtime_error(std::string(type) + " backward failed");
            };
            auto clear = [&] {
                std::fill(layer.grad_weights.begin(),layer.grad_weights.end(),0.f);
                std::fill(layer.grad_bias.begin(),layer.grad_bias.end(),0.f);
            };
            const auto fwd = measure(repetitions,forward,[]{}); write(output[0]);
            const auto bwd = measure(repetitions,backward,clear);
            for (const auto& g : gradients) write(g);
            write(layer.grad_weights); write(layer.grad_bias);
            json result = {{"type",type}, {"dim",dim}, {"rows",rows}, {"parameters",parameters},
                {"input_elements",input_elements}, {"forward_ms",fwd["median_ms"]},
                {"backward_ms",bwd["median_ms"]}, {"forward_samples_ms",fwd["samples_ms"]},
                {"backward_samples_ms",bwd["samples_ms"]}};
            if (layer.type_enum == LayerType::Conv2d) result["channels"] = layer.in_channels;
            cases.push_back(std::move(result));
        }
        if (cases.empty()) throw std::runtime_error("unknown layer filter");
        if (dump.is_open() && !dump) throw std::runtime_error("value dump failed");
        std::cout << json{{"backend","CPU"}, {"workers",workers}, {"repetitions",repetitions},
            {"isa",RuntimeSimd::isaName(RuntimeSimd::selectedIsa())}, {"cases",cases}}.dump() << '\n';
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
