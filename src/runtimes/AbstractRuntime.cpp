#include "runtimes/AbstractRuntime.hpp"

#include "Layers.hpp"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstring>

namespace {
static bool tensors_are_finite(const std::vector<std::vector<float>>& tensors) {
    for (const auto& tensor : tensors) {
        for (const float value : tensor) {
            if (!std::isfinite(value)) return false;
        }
    }
    return true;
}

static inline bool env_flag_true(const char* name, bool default_value) {
    const char* v = std::getenv(name);
    if (!v) return default_value;
    if (v[0] == '\0') return default_value;

    // "0" / "false" / "no" / "off" => false
    if ((v[0] == '0' && v[1] == '\0')) return false;

    std::string s(v);
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (s == "false" || s == "no" || s == "off") return false;
    return true;
}

static inline int env_int(const char* name, int default_value) {
    const char* v = std::getenv(name);
    if (!v || v[0] == '\0') return default_value;

    errno = 0;
    char* end = nullptr;
    long val = std::strtol(v, &end, 10);
    if (errno != 0 || end == v) return default_value;
    if (val < INT_MIN) return INT_MIN;
    if (val > INT_MAX) return INT_MAX;
    return static_cast<int>(val);
}

static inline bool env_disabled(const char* name) {
    const char* v = std::getenv(name);
    if (!v) return false;
    if (v[0] == '\0') return false;
    return !(v[0] == '0' && v[1] == '\0');
}

static inline std::string make_env_name(const char* backend_upper, const char* suffix) {
    std::string n = "MIMIR_";
    n += backend_upper;
    n += suffix;
    return n;
}
} // namespace

RuntimeConfig RuntimeConfig::fromEnv(const char* backend_upper) {
    RuntimeConfig cfg;
    cfg.backend = backend_upper ? backend_upper : "";

    cfg.verbose = env_flag_true("MIMIR_ACCEL_VERBOSE", false);

    // Désactivation explicite
    {
        std::string disable_env = "MIMIR_DISABLE_";
        disable_env += (backend_upper ? backend_upper : "");
        cfg.disabled = env_disabled(disable_env.c_str());
    }

    // Fast-path Linear
    {
        cfg.linear_enabled = env_flag_true(make_env_name(backend_upper, "_LINEAR").c_str(), false);
        cfg.linear_min_ops = env_int(make_env_name(backend_upper, "_LINEAR_MIN_OPS").c_str(), 1 << 20);
    }

    // Fast-path Conv
    {
        cfg.conv_enabled  = env_flag_true(make_env_name(backend_upper, "_CONV").c_str(), false);
        cfg.conv_min_ops  = env_int(make_env_name(backend_upper, "_CONV_MIN_OPS").c_str(), 1 << 18);
    }

    // Fast-path Normalization
    {
        cfg.norm_enabled       = env_flag_true(make_env_name(backend_upper, "_NORM").c_str(), false);
        cfg.norm_min_elements  = env_int(make_env_name(backend_upper, "_NORM_MIN_ELEMS").c_str(), 1 << 12);
    }

    // Fast-path Attention
    {
        cfg.attention_enabled  = env_flag_true(make_env_name(backend_upper, "_ATTENTION").c_str(), false);
        cfg.attention_min_ops  = env_int(make_env_name(backend_upper, "_ATTENTION_MIN_OPS").c_str(), 1 << 18);
    }

    // Device index (optionnel)
    {
        const std::string device_env = make_env_name(backend_upper, "_DEVICE");
        cfg.device_index = env_int(device_env.c_str(), 0);
    }

    return cfg;
}

bool AbstractRuntime::backwardLayer(
    const std::vector<const std::vector<float>*>& inputs,
    const std::vector<const std::vector<float>*>& grad_outputs,
    std::vector<std::vector<float>>& grad_inputs,
    Layer& layer,
    bool training
) {
    (void)inputs;
    (void)grad_outputs;
    (void)grad_inputs;
    (void)layer;
    (void)training;
    return false;
}

bool AbstractRuntime::supportsForwardLayerType(LayerType type) const {
    (void)type;
    return false;
}

bool AbstractRuntime::supportsBackwardLayerType(LayerType type) const {
    (void)type;
    return false;
}

RuntimeCapabilityLevel AbstractRuntime::queryForwardCapability(LayerType type) const {
    return supportsForwardLayerType(type)
        ? RuntimeCapabilityLevel::Native
        : RuntimeCapabilityLevel::Unsupported;
}

RuntimeCapabilityLevel AbstractRuntime::queryBackwardCapability(LayerType type) const {
    return supportsBackwardLayerType(type)
        ? RuntimeCapabilityLevel::Native
        : RuntimeCapabilityLevel::Unsupported;
}

RuntimeCapabilityLevel AbstractRuntime::queryForwardOperationCapability(
    const Layer& layer,
    const std::vector<const std::vector<float>*>& inputs,
    bool training
) const {
    (void)training;
    const RuntimeCapabilityLevel capability = queryForwardCapability(layer.type_enum);
    if (!runtimeCapabilityIsNative(capability) || inputs.empty() || !inputs[0]) {
        return RuntimeCapabilityLevel::Unsupported;
    }
    return capability;
}

RuntimeCapabilityLevel AbstractRuntime::queryBackwardOperationCapability(
    const Layer& layer,
    const std::vector<const std::vector<float>*>& inputs,
    const std::vector<const std::vector<float>*>& grad_outputs,
    bool training
) const {
    (void)training;
    if (inputs.empty() || !inputs[0] || grad_outputs.empty() || !grad_outputs[0]) {
        return RuntimeCapabilityLevel::Unsupported;
    }
    return queryBackwardCapability(layer.type_enum);
}

RuntimeCapabilityLevel AbstractRuntime::queryConfiguredForwardOperationCapability(
    const Layer& layer,
    const std::vector<const std::vector<float>*>& inputs,
    bool elementwise_requires_linear_flag
) const {
    const RuntimeCapabilityLevel capability = queryForwardCapability(layer.type_enum);
    if (config_.disabled || !runtimeCapabilityIsNative(capability) ||
        inputs.empty() || !inputs[0]) return RuntimeCapabilityLevel::Unsupported;

    const auto meets = [](long long work, int threshold) {
        return work >= static_cast<long long>(std::max(0, threshold));
    };
    const auto has_weights = [&layer](size_t required) {
        return layer.getWeights() && layer.getWeightsSize() >= required;
    };
    const auto& first = *inputs[0];

    switch (layer.type_enum) {
        case LayerType::Linear: {
            if (!config_.linear_enabled || layer.in_features <= 0 || layer.out_features <= 0) {
                return RuntimeCapabilityLevel::Unsupported;
            }
            const int batch = layer.seq_len > 0
                ? layer.seq_len
                : static_cast<int>(first.size() / static_cast<size_t>(layer.in_features));
            const size_t expected_input = static_cast<size_t>(batch) * layer.in_features;
            const size_t expected_weights = static_cast<size_t>(layer.out_features) * layer.in_features +
                (layer.use_bias ? static_cast<size_t>(layer.out_features) : 0u);
            const long long work = static_cast<long long>(batch) * layer.in_features * layer.out_features;
            return batch > 0 && first.size() == expected_input && has_weights(expected_weights) &&
                    meets(work, config_.linear_min_ops)
                ? capability : RuntimeCapabilityLevel::Unsupported;
        }
        case LayerType::MatMul:
        case LayerType::BatchMatMul: {
            if (!config_.linear_enabled || inputs.size() < 2 || !inputs[1]) {
                return RuntimeCapabilityLevel::Unsupported;
            }
            const int batches = layer.type_enum == LayerType::BatchMatMul ? layer.seq_len : 1;
            const int rows = layer.in_features;
            const int inner = layer.out_features;
            const int columns = layer.embed_dim;
            if (batches <= 0 || rows <= 0 || inner <= 0 || columns <= 0) {
                return RuntimeCapabilityLevel::Unsupported;
            }
            const size_t a_size = static_cast<size_t>(batches) * rows * inner;
            const size_t b_size = static_cast<size_t>(batches) * inner * columns;
            const long long work = static_cast<long long>(batches) * rows * inner * columns;
            return first.size() == a_size && inputs[1]->size() == b_size &&
                    meets(work, config_.linear_min_ops)
                ? capability : RuntimeCapabilityLevel::Unsupported;
        }
        case LayerType::Conv2d:
        case LayerType::ConvTranspose2d: {
            if (!config_.conv_enabled) return RuntimeCapabilityLevel::Unsupported;
            const int input_channels = std::max(1, layer.in_channels);
            const int output_channels = std::max(1, layer.out_channels);
            const int height = std::max(1, layer.input_height);
            const int width = std::max(1, layer.input_width);
            const int kernel = std::max(1, layer.get_kernel_h());
            const size_t expected_input = static_cast<size_t>(input_channels) * height * width;
            const size_t expected_weights = static_cast<size_t>(output_channels) * input_channels *
                kernel * kernel + (layer.use_bias ? static_cast<size_t>(output_channels) : 0u);
            const long long work = static_cast<long long>(output_channels) * input_channels *
                kernel * kernel * height * width;
            return first.size() == expected_input && has_weights(expected_weights) &&
                    meets(work, config_.conv_min_ops)
                ? capability : RuntimeCapabilityLevel::Unsupported;
        }
        case LayerType::LayerNorm:
        case LayerType::RMSNorm: {
            if (!config_.norm_enabled || !layer.affine ||
                !meets(static_cast<long long>(first.size()), config_.norm_min_elements)) {
                return RuntimeCapabilityLevel::Unsupported;
            }
            const int normalized = layer.in_features > 0
                ? layer.in_features : static_cast<int>(first.size());
            return normalized > 0 && first.size() % static_cast<size_t>(normalized) == 0 &&
                    has_weights(static_cast<size_t>(normalized))
                ? capability : RuntimeCapabilityLevel::Unsupported;
        }
        case LayerType::SelfAttention:
        case LayerType::MultiHeadAttention: {
            if (!config_.attention_enabled) return RuntimeCapabilityLevel::Unsupported;
            const int sequence = layer.seq_len > 0 ? layer.seq_len : 1;
            const int embedding = layer.embed_dim > 0
                ? layer.embed_dim : static_cast<int>(first.size());
            const int heads = layer.num_heads > 0 ? layer.num_heads : 1;
            const long long work = 4LL * sequence * embedding * embedding;
            return embedding > 0 && heads > 0 && embedding % heads == 0 &&
                    first.size() == static_cast<size_t>(sequence) * embedding &&
                    has_weights(static_cast<size_t>(4) * embedding * embedding) &&
                    meets(work, config_.attention_min_ops)
                ? capability : RuntimeCapabilityLevel::Unsupported;
        }
        case LayerType::CrossAttention: {
            if (!config_.attention_enabled || inputs.size() < 2 || !inputs[1]) {
                return RuntimeCapabilityLevel::Unsupported;
            }
            const int heads = layer.num_heads > 0 ? layer.num_heads : 1;
            int embedding = layer.embed_dim;
            if (embedding <= 0 && layer.head_dim > 0) embedding = layer.head_dim * heads;
            if (embedding <= 0 || heads <= 0 || embedding % heads != 0 ||
                first.size() % static_cast<size_t>(embedding) != 0 ||
                inputs[1]->size() % static_cast<size_t>(embedding) != 0) {
                return RuntimeCapabilityLevel::Unsupported;
            }
            const long long query_length = static_cast<long long>(first.size() / embedding);
            const long long key_length = static_cast<long long>(inputs[1]->size() / embedding);
            const size_t required_weights = static_cast<size_t>(4) * embedding * embedding;
            return has_weights(required_weights) &&
                    meets(2LL * query_length * key_length * embedding, config_.attention_min_ops)
                ? capability : RuntimeCapabilityLevel::Unsupported;
        }
        case LayerType::Add:
        case LayerType::Subtract:
        case LayerType::Multiply:
        case LayerType::Divide:
            if ((elementwise_requires_linear_flag && !config_.linear_enabled) ||
                inputs.size() < 2 || !inputs[1] || first.empty() ||
                first.size() != inputs[1]->size()) return RuntimeCapabilityLevel::Unsupported;
            return meets(static_cast<long long>(first.size()), config_.linear_min_ops)
                ? capability : RuntimeCapabilityLevel::Unsupported;
        case LayerType::ReLU:
        case LayerType::LeakyReLU:
        case LayerType::Sigmoid:
        case LayerType::Tanh:
        case LayerType::SiLU:
        case LayerType::GELU:
        case LayerType::Softplus:
        case LayerType::Mish:
        case LayerType::HardSigmoid:
        case LayerType::HardSwish:
            if (elementwise_requires_linear_flag && !config_.linear_enabled) {
                return RuntimeCapabilityLevel::Unsupported;
            }
            return !first.empty() && meets(static_cast<long long>(first.size()), config_.linear_min_ops)
                ? capability : RuntimeCapabilityLevel::Unsupported;
        default:
            return capability;
    }
}

bool AbstractRuntime::supportsKernelFusion(LayerType producer, LayerType consumer) const {
    (void)producer;
    (void)consumer;
    return false;
}

bool AbstractRuntime::dispatchForwardLayer(
    const std::vector<AbstractRuntime*>& runtime_priority,
    const std::vector<const std::vector<float>*>& inputs,
    std::vector<std::vector<float>>& outputs,
    const Layer& layer,
    bool training,
    AbstractRuntime** selected_runtime
) {
    if (selected_runtime) *selected_runtime = nullptr;
    outputs.clear();

    for (AbstractRuntime* rt : runtime_priority) {
        switch (rt == nullptr) {
            case true:
                continue;
            case false:
                break;
        }
        switch (rt->isInitialized()) {
            case true:
                break;
            case false:
                continue;
        }
        if (!runtimeCapabilityIsNative(
                rt->queryForwardOperationCapability(layer, inputs, training))) continue;

        std::vector<std::vector<float>> local_outputs;
        if (!rt->forwardLayer(inputs, local_outputs, layer, training)) {
            continue;
        }
        if (local_outputs.empty() || !tensors_are_finite(local_outputs)) {
            continue;
        }

        outputs = std::move(local_outputs);
        if (selected_runtime) *selected_runtime = rt;
        return true;
    }

    return false;
}

bool AbstractRuntime::dispatchBackwardLayer(
    const std::vector<AbstractRuntime*>& runtime_priority,
    const std::vector<const std::vector<float>*>& inputs,
    const std::vector<const std::vector<float>*>& grad_outputs,
    std::vector<std::vector<float>>& grad_inputs,
    Layer& layer,
    bool training,
    AbstractRuntime** selected_runtime
) {
    if (selected_runtime) *selected_runtime = nullptr;
    grad_inputs.clear();

    for (AbstractRuntime* rt : runtime_priority) {
        switch (rt == nullptr) {
            case true:
                continue;
            case false:
                break;
        }
        switch (rt->isInitialized()) {
            case true:
                break;
            case false:
                continue;
        }
        if (!runtimeCapabilityIsNative(rt->queryBackwardOperationCapability(
                layer, inputs, grad_outputs, training))) continue;

        std::vector<std::vector<float>> local_grad_inputs;
        if (!rt->backwardLayer(inputs, grad_outputs, local_grad_inputs, layer, training)) {
            continue;
        }
        if (local_grad_inputs.empty() || !tensors_are_finite(local_grad_inputs)) {
            continue;
        }

        grad_inputs = std::move(local_grad_inputs);
        if (selected_runtime) *selected_runtime = rt;
        return true;
    }

    return false;
}
