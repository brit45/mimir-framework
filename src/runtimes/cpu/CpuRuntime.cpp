#include "PointwiseFusion.hpp"
#include "runtimes/cpu/CpuRuntime.hpp"

#include "SIMD_Ops.hpp"
#include "runtimes/LayerOps.hpp"
#include "runtimes/cpu/RuntimeLayerDispatch.hpp"
#include "runtimes/cpu/TensorWeightKernels.hpp"

#include <cstddef>
#include <limits>

namespace {
std::vector<float> roundToDType(const std::vector<float>& values,
                                const Mimir::DType dtype) {
    if (values.size() > static_cast<size_t>(std::numeric_limits<int>::max())) return {};
    return Mimir::TypedTensor::fromFloat32(
        values, {static_cast<int>(values.size())}, dtype).toFloat32();
}

bool linearForwardAutocast(const std::vector<const std::vector<float>*>& inputs,
                           std::vector<std::vector<float>>& outputs,
                           const Layer& layer, const Mimir::DType dtype) {
    if (inputs.empty() || !inputs[0] || !layer.getWeights() ||
        layer.in_features <= 0 || layer.out_features <= 0 ||
        (dtype != Mimir::DType::F16 && dtype != Mimir::DType::BF16)) return false;
    const auto& input = *inputs[0];
    const int in_features = layer.in_features;
    const int out_features = layer.out_features;
    const int batch = layer.seq_len > 0
        ? layer.seq_len
        : (input.size() % static_cast<size_t>(in_features) == 0
            ? static_cast<int>(input.size() / static_cast<size_t>(in_features)) : 0);
    if (batch <= 0 || input.size() != static_cast<size_t>(batch) * in_features) return false;

    const size_t weight_count = static_cast<size_t>(out_features) * in_features;
    const size_t required = weight_count + (layer.use_bias ? static_cast<size_t>(out_features) : 0);
    if (layer.getWeightsSize() < required) return false;
    std::vector<float> master_weights(layer.getWeights(), layer.getWeights() + required);
    const auto rounded_input = roundToDType(input, dtype);
    const auto rounded_weights = roundToDType(master_weights, dtype);
    if (rounded_input.size() != input.size() || rounded_weights.size() != required) return false;

    std::vector<float> result(static_cast<size_t>(batch) * out_features, 0.0f);
    SIMD::matmul_transpose_avx2(result.data(), rounded_input.data(),
        rounded_weights.data(), static_cast<size_t>(batch),
        static_cast<size_t>(out_features), static_cast<size_t>(in_features));
    if (layer.use_bias) {
        for (int row = 0; row < batch; ++row)
            for (int out = 0; out < out_features; ++out)
                result[static_cast<size_t>(row) * out_features + out] +=
                    rounded_weights[weight_count + out];
    }
    outputs.resize(1);
    outputs[0] = roundToDType(result, dtype);
    return outputs[0].size() == result.size();
}
} // namespace

bool CpuRuntime::initialize(const RuntimeConfig& cfg) {
    // CPU est toujours disponible; on conserve cfg pour homogénéité.
    config_ = cfg;
    initialized_ = true;
    return true;
}

void CpuRuntime::shutdown() {
    initialized_ = false;
}

bool CpuRuntime::computeTensorWeights(const std::vector<tensor>& input,
                                      std::vector<uint16_t>& output) {
    return supportsTensorWeights() && CpuTensorWeights::compute(input,output);
}

bool CpuRuntime::linearForward(
    const float* input,
    const float* weights,
    const float* bias_or_null,
    float* output,
    int batch,
    int in_f,
    int out_f
) {
    if (!initialized_) return false;
    if (!input || !weights || !output) return false;
    if (batch <= 0 || in_f <= 0 || out_f <= 0) return false;

    // output[batch x out_f] = input[batch x in_f] @ weights[out_f x in_f]^T
    SIMD::matmul_transpose_avx2(
        output,
        input,
        weights,
        static_cast<size_t>(batch),
        static_cast<size_t>(out_f),
        static_cast<size_t>(in_f)
    );

    if (bias_or_null) {
        for (int b = 0; b < batch; ++b) {
            float* row = output + static_cast<size_t>(b) * static_cast<size_t>(out_f);
            for (int o = 0; o < out_f; ++o) {
                row[o] += bias_or_null[o];
            }
        }
    }

    return true;
}

bool CpuRuntime::linearForwardTyped(
    const Mimir::TypedTensor& input,
    const Mimir::TypedTensor& weights,
    const Mimir::TypedTensor* bias,
    Mimir::TypedTensor& output
) {
    if (!initialized_ || input.shape().size() != 2 || weights.shape().size() != 2)
        return false;
    const int batch = input.shape()[0];
    const int in_f = input.shape()[1];
    const int out_f = weights.shape()[0];
    if (weights.shape()[1] != in_f || output.shape() != std::vector<int>({batch, out_f}))
        return false;
    if (input.dtype() != weights.dtype() || output.dtype() != input.dtype())
        return false;
    if (bias && (bias->dtype() != input.dtype() ||
                 bias->shape() != std::vector<int>({out_f})))
        return false;

    for (int row = 0; row < batch; ++row) {
        for (int out = 0; out < out_f; ++out) {
            // F64 keeps double accumulation. Lower precision floating and
            // integer kernels accumulate in at least float64/int64 range.
            long double sum = bias ? bias->get(static_cast<size_t>(out)) : 0.0L;
            for (int in = 0; in < in_f; ++in) {
                sum += static_cast<long double>(
                           input.get(static_cast<size_t>(row) * in_f + in)) *
                       static_cast<long double>(
                           weights.get(static_cast<size_t>(out) * in_f + in));
            }
            output.set(static_cast<size_t>(row) * out_f + out,
                       static_cast<double>(sum));
        }
    }
    return true;
}

bool CpuRuntime::supportsAutocast(const Layer& layer, const Mimir::DType dtype) const {
    return initialized_ && layer.type_enum == LayerType::Linear &&
        (dtype == Mimir::DType::F16 || dtype == Mimir::DType::BF16);
}

bool CpuRuntime::forwardLayer(
    const std::vector<const std::vector<float>*>& inputs,
    std::vector<std::vector<float>>& outputs,
    const Layer& layer,
    bool training
) {
    if (!initialized_) return false;
    if (!supportsForwardLayerType(layer.type_enum)) return false;
    return RuntimeLayerDispatch::cpu_forward_layer(inputs, outputs, layer, training);
}

bool CpuRuntime::forwardLayerWithContext(
    const std::vector<const std::vector<float>*>& inputs,
    std::vector<std::vector<float>>& outputs, const Layer& layer, bool training,
    const RuntimeForwardContext& context) {
    if (context.autocast_dtype != Mimir::DType::F32 &&
        layer.type_enum == LayerType::Linear) {
        outputs.clear();
        return initialized_ && linearForwardAutocast(inputs, outputs, layer,
                                                      context.autocast_dtype);
    }
    if (context.output_mask && (layer.type_enum == LayerType::Dropout ||
        layer.type_enum == LayerType::Dropout2d || layer.type_enum == LayerType::AlphaDropout)) {
        context.output_mask->clear();
        return initialized_ && RuntimeLayerDispatch::cpu_forward_layer(inputs,outputs,layer,training,context.output_mask);
    }
    return AbstractRuntime::forwardLayerWithContext(inputs,outputs,layer,training,context);
}

bool CpuRuntime::supportsForwardLayerType(const LayerType type) const {
    return RuntimeLayerDispatch::cpu_supports_forward_layer_type(type);
}

bool CpuRuntime::supportsBackwardLayerType(const LayerType type) const {
    return RuntimeLayerDispatch::cpu_supports_backward_layer_type(type);
}

RuntimeCapabilityLevel CpuRuntime::queryForwardCapability(const LayerType type) const {
    return supportsForwardLayerType(type) ? RuntimeCapabilityLevel::Native
                                          : RuntimeCapabilityLevel::Unsupported;
}

RuntimeCapabilityLevel CpuRuntime::queryBackwardCapability(const LayerType type) const {
    return supportsBackwardLayerType(type) ? RuntimeCapabilityLevel::Native
                                           : RuntimeCapabilityLevel::Unsupported;
}

bool CpuRuntime::backwardLayer(
    const std::vector<const std::vector<float>*>& inputs,
    const std::vector<const std::vector<float>*>& grad_outputs,
    std::vector<std::vector<float>>& grad_inputs,
    Layer& layer,
    bool training
) {
    if (!initialized_) return false;
    if (!supportsBackwardLayerType(layer.type_enum)) return false;
    return RuntimeLayerDispatch::cpu_backward_layer(inputs, grad_outputs, grad_inputs, layer, training);
}

bool CpuRuntime::mergeBranches(const std::vector<float>& left,
                               const std::vector<float>& right,
                               std::vector<float>& output, const Layer& layer) {
    if (!initialized_) return false;
    RuntimeLayerOps::branchMerge(left, right, output, layer.merge_op, true);
    return true;
}

RuntimeCapabilityLevel CpuRuntime::queryPointwiseFusionCapability(
    const std::vector<const Layer*>& chain,bool,bool) const {
    if(!initialized_ || chain.empty())return RuntimeCapabilityLevel::Unsupported;
    for(const auto* layer:chain)
        if(!layer || !Mimir::CpuFusion::supports(layer->type_enum) || layer->getWeightsSize()!=0 ||
            (!layer->inputs.empty() && layer->inputs.size()!=1))return RuntimeCapabilityLevel::Unsupported;
    return RuntimeCapabilityLevel::NativeOptimized;
}
bool CpuRuntime::pointwiseFusionForward(const std::vector<const Layer*>& chain,const std::vector<float>& input,
    std::vector<std::vector<float>>& output,bool retain,bool training) {
    if(!runtimeCapabilityIsNative(queryPointwiseFusionCapability(chain,false,training)))return false;
    Mimir::CpuFusion::forwardChain(chain,input,output,retain);return true;
}
bool CpuRuntime::pointwiseFusionBackward(const std::vector<const Layer*>& chain,
    const std::vector<const std::vector<float>*>& inputs,const std::vector<float>& grad,std::vector<float>& output) {
    if(chain.size()!=inputs.size() || !runtimeCapabilityIsNative(queryPointwiseFusionCapability(chain,true,true)))return false;
    for(size_t i=0;i<chain.size();++i)
        if(chain[i]->type_enum!=LayerType::Identity && (!inputs[i] || inputs[i]->size()!=grad.size()))return false;
    Mimir::CpuFusion::backwardChain(chain,inputs,grad,output);return true;
}
