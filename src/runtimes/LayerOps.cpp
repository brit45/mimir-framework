#include "runtimes/LayerOps.hpp"

#include <algorithm>
#include <cmath>

namespace RuntimeLayerOps {

bool resolveUnaryOp(const LayerType type, const Layer& layer, int& op_code, float& alpha) {
    alpha = 0.01f;
    switch (type) {
        case LayerType::ReLU: op_code = 0; return true;
        case LayerType::LeakyReLU:
            op_code = 1;
            alpha = layer.leaky_relu_alpha > 0.0f ? layer.leaky_relu_alpha : 0.01f;
            return true;
        case LayerType::Sigmoid: op_code = 2; return true;
        case LayerType::Tanh: op_code = 3; return true;
        case LayerType::SiLU: op_code = 4; return true;
        case LayerType::GELU: op_code = 5; return true;
        case LayerType::Softplus: op_code = 6; return true;
        case LayerType::Mish: op_code = 7; return true;
        case LayerType::HardSigmoid: op_code = 8; return true;
        case LayerType::HardSwish: op_code = 9; return true;
        default:
            return false;
    }
}

bool resolveBinaryOp(const LayerType type, int& op_code) {
    switch (type) {
        case LayerType::Add: op_code = 0; return true;
        case LayerType::Subtract: op_code = 1; return true;
        case LayerType::Multiply: op_code = 2; return true;
        case LayerType::Divide: op_code = 3; return true;
        default:
            return false;
    }
}

void unaryForwardHost(const std::vector<float>& input, std::vector<float>& output, const int op_code, const float alpha) {
    output.resize(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        const float x = input[i];
        switch (op_code) {
            case 0: output[i] = x > 0.0f ? x : 0.0f; break;
            case 1: output[i] = x > 0.0f ? x : alpha * x; break;
            case 2: output[i] = 1.0f / (1.0f + std::exp(-x)); break;
            case 3: output[i] = std::tanh(x); break;
            case 4: {
                const float s = 1.0f / (1.0f + std::exp(-x));
                output[i] = x * s;
                break;
            }
            case 5: {
                const float c = 0.7978845608f;
                const float x3 = x * x * x;
                output[i] = 0.5f * x * (1.0f + std::tanh(c * (x + 0.044715f * x3)));
                break;
            }
            case 6: output[i] = x > 20.0f ? x : std::log1p(std::exp(x)); break;
            case 7: {
                const float sp = x > 20.0f ? x : std::log1p(std::exp(x));
                output[i] = x * std::tanh(sp);
                break;
            }
            case 8: {
                const float hs = (x + 3.0f) / 6.0f;
                output[i] = std::min(1.0f, std::max(0.0f, hs));
                break;
            }
            case 9: {
                const float hs = std::min(1.0f, std::max(0.0f, (x + 3.0f) / 6.0f));
                output[i] = x * hs;
                break;
            }
            default:
                output[i] = x;
                break;
        }
    }
}

void binaryForwardHost(const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& output, const int op_code) {
    output.resize(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        switch (op_code) {
            case 0: output[i] = a[i] + b[i]; break;
            case 1: output[i] = a[i] - b[i]; break;
            case 2: output[i] = a[i] * b[i]; break;
            case 3: {
                const float d = b[i];
                const float safe_d = std::fabs(d) < 1e-8f
                    ? (d < 0.0f ? -1e-8f : 1e-8f)
                    : d;
                output[i] = a[i] / safe_d;
                break;
            }
            default:
                output[i] = a[i];
                break;
        }
    }
}

} // namespace RuntimeLayerOps
