#include "test_utils.hpp"
#include "runtimes/LayerOps.hpp"
#include "runtimes/cpu/CpuRuntime.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

bool matches(const std::vector<float>& actual, const std::vector<float>& expected,
             float tolerance = 2e-6f) {
    if (actual.size() != expected.size()) return false;
    for (size_t index = 0; index < actual.size(); ++index) {
        if (!std::isfinite(actual[index]) || !nearf(actual[index], expected[index], tolerance)) {
            std::cerr << "Mismatch at " << index << ": " << actual[index]
                      << " expected " << expected[index] << '\n';
            return false;
        }
    }
    return true;
}

struct ActivationCase {
    ActivationType type;
    const char* name;
    std::array<float, 5> expected;
};

template<class Function>
bool rejectsInvalidArgument(const Function& operation) {
    try { operation(); }
    catch (const std::invalid_argument&) { return true; }
    return false;
}

} // namespace

int main() {
    // Independent reference values at {-4, -1, 0, 1, 4}. Seventeen elements
    // exercise two SIMD blocks and a scalar tail with the same activation.
    const std::array<float, 5> points{-4, -1, 0, 1, 4};
    const ActivationCase activations[] = {
        {ActivationType::NONE, "none", {-4, -1, 0, 1, 4}},
        {ActivationType::RELU, "relu", {0, 0, 0, 1, 4}},
        {ActivationType::RELU6, "relu6", {0, 0, 0, 1, 4}},
        {ActivationType::LEAKY_RELU, "leaky_relu", {-0.8f, -0.2f, 0, 1, 4}},
        {ActivationType::ELU, "elu", {-0.196336872f, -0.126424112f, 0, 1, 4}},
        {ActivationType::SELU, "selu", {-1.725898628f, -1.111330738f, 0, 1.050700987f, 4.202803949f}},
        {ActivationType::GELU, "gelu", {-0.000070246f, -0.158808009f, 0, 0.841191991f, 3.999929754f}},
        {ActivationType::SWISH, "swish", {-0.071944840f, -0.268941421f, 0, 0.731058579f, 3.928055160f}},
        {ActivationType::MISH, "mish", {-0.072591741f, -0.303401461f, 0, 0.865098388f, 3.997412807f}},
        {ActivationType::TANH, "tanh", {-0.999329300f, -0.761594156f, 0, 0.761594156f, 0.999329300f}},
        {ActivationType::SIGMOID, "sigmoid", {0.017986210f, 0.268941421f, 0.5f, 0.731058579f, 0.982013790f}},
        {ActivationType::SOFTPLUS, "softplus", {0.018149928f, 0.313261688f, 0.693147181f, 1.313261688f, 4.018149928f}},
        {ActivationType::SOFTSIGN, "softsign", {-0.8f, -0.5f, 0, 0.5f, 0.8f}},
    };
    for (bool hardware : {false, true}) {
        RuntimeLayerOps::setHardwareAcceleration(hardware);
        for (const auto& activation : activations) {
            std::vector<float> input(17), expected(17);
            for (size_t index = 0; index < input.size(); ++index) {
                const size_t point = index == 16 ? 3 : index % points.size();
                input[index] = points[point];
                expected[index] = activation.expected[point];
            }
            auto by_enum = input;
            auto by_name = input;
            RuntimeLayerOps::computeActivation(by_enum, activation.type, 0.2f);
            RuntimeLayerOps::computeActivation(by_name, std::string(activation.name), 0.2f);
            TASSERT_TRUE(matches(by_enum, expected));
            TASSERT_TRUE(matches(by_name, expected));
        }
        std::vector<float> capped{-10, 0, 3, 6, 10};
        RuntimeLayerOps::computeActivation(capped, ActivationType::RELU6);
        TASSERT_TRUE(matches(capped, {0, 0, 3, 6, 6}));
        std::vector<float> probabilities(17, 1000.0f);
        RuntimeLayerOps::computeActivation(probabilities, ActivationType::SOFTMAX);
        TASSERT_TRUE(matches(probabilities, std::vector<float>(17, 1.0f / 17.0f)));
        for (ActivationType type : {ActivationType::SOFTPLUS, ActivationType::MISH}) {
            std::vector<float> extreme{-1000, 0, 1000};
            RuntimeLayerOps::computeActivation(extreme, type);
            TASSERT_TRUE(matches(extreme, {0, type == ActivationType::SOFTPLUS ? 0.693147181f : 0, 1000}));
        }
    }

    // The facade uses out-major Linear weights, matching Layer/Model storage.
    // Seventeen distinct output positions cover SIMD blocks and a scalar tail;
    // a rectangular, non-symmetric matrix exposes accidental transposition.
    for (bool hardware : {false, true}) for (bool requested : {false, true}) {
        RuntimeLayerOps::setHardwareAcceleration(hardware);
        RuntimeLayerOps::LayerParams params;
        params.in_features = 3; params.out_features = 17;
        const std::array<std::array<float,3>,5> rows{{
            {{1,2,3}}, {{-4,.5f,2}}, {{.25f,-2,1}}, {{5,0,-1}}, {{-1,3,-.5f}}
        }};
        const std::array<float,5> biases{{.25f,-.5f,1,2,-3}};
        const std::array<float,5> raw{{5.5f,10.5f,-6,-10.5f,10.75f}};
        std::vector<float> expected(17), output(50,123.f);
        for (size_t row = 0; row < 17; ++row) {
            params.weights.insert(params.weights.end(), rows[row%rows.size()].begin(), rows[row%rows.size()].end());
            params.bias.push_back(biases[row%biases.size()]);
            expected[row] = raw[row%raw.size()] + params.bias.back();
        }
        RuntimeLayerOps::computeLinear({-2,3,.5f}, output, params, requested);
        TASSERT_TRUE(matches(output, expected, 0));
        params.bias.clear();
        for (size_t row = 0; row < expected.size(); ++row) expected[row] = raw[row%raw.size()];
        RuntimeLayerOps::computeLinear({-2,3,.5f}, output, params, requested);
        TASSERT_TRUE(matches(output, expected, 0));
        const auto before = output;
        params.bias.assign(16,0.f);
        TASSERT_TRUE(rejectsInvalidArgument([&] {
            RuntimeLayerOps::computeLinear({-2,3,.5f}, output, params, requested);
        }));
        TASSERT_TRUE(output == before);
    }

    // Two heads attend to opposite tokens. Per-head scaling is 1/sqrt(2),
    // giving the analytic probability p=e/(1+e), unlike a single-head score.
    const float root2 = std::sqrt(2.f);
    const std::vector<float> query{1,0,0,1, 0,1,1,0};
    const std::vector<float> key{root2,0,root2,0, 0,root2,0,root2};
    const std::vector<float> value{2,4,10,20, 6,-2,-10,40};
    const double p = std::exp(1.0)/(1.0+std::exp(1.0));
    const std::vector<float> attention_expected{
        float(6-4*p), float(-2+6*p), float(10-20*p), float(20+20*p),
        float(2+4*p), float(4-6*p), float(-10+20*p), float(40-20*p)
    };
    for (bool hardware : {false, true}) for (bool requested : {false, true}) {
        RuntimeLayerOps::setHardwareAcceleration(hardware);
        std::vector<float> output(30,123.f);
        RuntimeLayerOps::computeAttention(query, key, value, output, 2,4,2, requested);
        TASSERT_TRUE(matches(output, attention_expected, 8e-6f));
        RuntimeLayerOps::computeAttention(std::vector<float>(8,0.f), key, value, output, 2,4,2, requested);
        TASSERT_TRUE(matches(output, {4,1,0,30, 4,1,0,30}, 0));
        const auto before = output;
        TASSERT_TRUE(rejectsInvalidArgument([&] {
            RuntimeLayerOps::computeAttention(query, key, value, output, 2,4,3, requested);
        }));
        TASSERT_TRUE(rejectsInvalidArgument([&] {
            RuntimeLayerOps::computeAttention(query, std::vector<float>(7), value, output, 2,4,2, requested);
        }));
        TASSERT_TRUE(output == before);
    }

    // Independent double softmax reference with unequal logits, large common
    // offsets, SIMD tails and both global/per-call hardware controls.
    std::vector<float> logits(17), softmax_expected(17);
    double denominator = 0;
    for (size_t i = 0; i < logits.size(); ++i) {
        logits[i] = 1000.f + float(int((i*7)%13)-6)*.25f;
        denominator += std::exp(double(logits[i])-1001.5);
    }
    for (size_t i = 0; i < logits.size(); ++i)
        softmax_expected[i] = float(std::exp(double(logits[i])-1001.5)/denominator);
    for (bool hardware : {false, true}) for (bool requested : {false, true}) {
        RuntimeLayerOps::setHardwareAcceleration(hardware);
        auto probabilities = logits;
        RuntimeLayerOps::computeActivation(probabilities, ActivationType::SOFTMAX, 0, requested);
        TASSERT_TRUE(matches(probabilities, softmax_expected));
        probabilities = logits;
        for (float& logit : probabilities) logit -= 2000.f;
        RuntimeLayerOps::computeActivation(probabilities, "softmax", 0, requested);
        TASSERT_TRUE(matches(probabilities, softmax_expected));
        double total = 0;
        for (float probability : probabilities) total += probability;
        TASSERT_TRUE(std::abs(total-1.0) < 1e-6);
    }

    for (bool hardware : {false, true}) for (bool requested : {false, true}) {
        RuntimeLayerOps::setHardwareAcceleration(hardware);
        std::vector<float> left(17), right(17), expected(17), output;
        for (size_t i = 0; i < left.size(); ++i) {
            left[i] = float(int(i)-8)*.25f;
            right[i] = float(int((i*3)%11)-5)*.5f;
        }
        for (MergeOperation operation : {MergeOperation::ADD, MergeOperation::MULTIPLY,
                                         MergeOperation::MAX, MergeOperation::AVERAGE}) {
            for (size_t i = 0; i < left.size(); ++i) {
                switch (operation) {
                    case MergeOperation::ADD: expected[i] = left[i]+right[i]; break;
                    case MergeOperation::MULTIPLY: expected[i] = left[i]*right[i]; break;
                    case MergeOperation::MAX: expected[i] = std::max(left[i],right[i]); break;
                    default: expected[i] = (left[i]+right[i])*.5f; break;
                }
            }
            RuntimeLayerOps::branchMerge(left, right, output, operation, requested);
            TASSERT_TRUE(matches(output, expected, 0));
        }
        RuntimeLayerOps::branchMerge({1,2}, {3}, output, MergeOperation::CONCATENATE, requested);
        TASSERT_TRUE(output == std::vector<float>({1,2,3}));
        RuntimeLayerOps::branchMerge({}, {3,4}, output, MergeOperation::CONCATENATE, requested);
        TASSERT_TRUE(output == std::vector<float>({3,4}));
        const auto before = output;
        TASSERT_TRUE(rejectsInvalidArgument([&] {
            RuntimeLayerOps::branchMerge({1,2}, {3}, output, MergeOperation::ADD, requested);
        }));
        TASSERT_TRUE(output == before);
    }

    RuntimeLayerOps::setHardwareAcceleration(false);
    CpuRuntime cpu;
    TASSERT_TRUE(cpu.initialize(RuntimeConfig{}));

    // The context path must still apply the configured activation after the
    // convolution has completed, including values beyond the old tanh clip.
    {
        Layer convolution("activated", "Conv2d", 2);
        convolution.in_channels = convolution.out_channels = 1;
        convolution.input_height = 1;
        convolution.input_width = 17;
        convolution.kernel_size = 1;
        convolution.use_bias = true;
        convolution.weights = {1, 0.25f};
        convolution.activation = ActivationType::TANH;
        std::vector<float> input(17), expected(17);
        for (size_t index = 0; index < input.size(); ++index) {
            input[index] = static_cast<float>(static_cast<int>(index) - 8);
            expected[index] = std::tanh(input[index] + 0.25f);
        }
        std::vector<std::vector<float>> output;
        TASSERT_TRUE(cpu.forwardLayerWithContext({&input}, output, convolution, false));
        TASSERT_TRUE(output.size() == 1 && matches(output[0], expected));
    }

    {
        RuntimeLayerOps::LayerParams params;
        params.kernel_size = 2;
        params.weights = {1, 0, 0, 1};
        params.bias = {0.5f};
        const std::vector<float> input{1, 2, 3, 4, 5, 6, 7, 8, 9};
        std::vector<float> output(20, 123.0f);
        for (int repeat = 0; repeat < 2; ++repeat) {
            RuntimeLayerOps::computeConv2D(input, output, params, 3, 3, 1, 1);
            TASSERT_TRUE(matches(output, {6.5f, 8.5f, 12.5f, 14.5f}));
        }
        params.kernel_size = 1;
        params.weights = {1, 2, -1, 0.5f};
        params.bias = {0.5f, -0.5f};
        RuntimeLayerOps::computeConv2D({1, 2, 3, 4, 10, 20, 30, 40}, output, params, 2, 2, 2, 2);
        TASSERT_TRUE(matches(output, {21.5f, 42.5f, 63.5f, 84.5f, 3.5f, 7.5f, 11.5f, 15.5f}));

        params.kernel_size = 2;
        params.weights = {1, 1, 1, 1};
        params.bias = {0.5f};
        const std::vector<float> transpose_input{1, 2, 3, 4};
        const std::vector<float> transpose_expected{1.5f, 3.5f, 2.5f, 4.5f, 10.5f, 6.5f, 3.5f, 7.5f, 4.5f};
        for (int repeat = 0; repeat < 2; ++repeat) {
            RuntimeLayerOps::computeConvTranspose2D(transpose_input, output, params, 2, 2, 1, 1);
            TASSERT_TRUE(matches(output, transpose_expected));
        }
        Layer transpose("transpose", "ConvTranspose2d", 5);
        transpose.in_channels = transpose.out_channels = 1;
        transpose.input_height = transpose.input_width = 2;
        transpose.kernel_size = 2;
        transpose.use_bias = true;
        transpose.weights = {1, 1, 1, 1, 0.5f};
        std::vector<std::vector<float>> dispatched;
        TASSERT_TRUE(cpu.forwardLayer({&transpose_input}, dispatched, transpose, false));
        TASSERT_TRUE(dispatched.size() == 1 && matches(dispatched[0], transpose_expected));
    }

    {
        const std::vector<float> input{1, 2, 3, 4, -1, -2, -3, -4};
        std::vector<float> output;
        RuntimeLayerOps::computeMaxPool2D(input, output, 2, 2, 2, 2, -1);
        TASSERT_TRUE(matches(output, {4, -1}));
        RuntimeLayerOps::computeAvgPool2D(input, output, 2, 2, 2, 2, -1);
        TASSERT_TRUE(matches(output, {2.5f, -2.5f}));

        const std::vector<float> rectangular{1, 2, 3, 4, 5, 6};
        Layer average("average", "AvgPool2d", 0);
        average.in_channels = 1;
        average.input_height = 2;
        average.input_width = 3;
        average.kernel_h = 2;
        average.kernel_w = 3;
        average.stride_w = 2;
        average.pad_h = average.pad_w = 1;
        std::vector<std::vector<float>> dispatched;
        TASSERT_TRUE(cpu.forwardLayer({&rectangular}, dispatched, average, false));
        TASSERT_TRUE(dispatched.size() == 1 && matches(dispatched[0], {1.5f, 2.5f, 3, 4, 4.5f, 5.5f}));
        Layer maximum("maximum", "MaxPool2d", 0);
        maximum.in_channels = 1;
        maximum.input_height = 2;
        maximum.input_width = 3;
        maximum.kernel_size = 2;
        maximum.padding = 1;
        TASSERT_TRUE(cpu.forwardLayer({&rectangular}, dispatched, maximum, false));
        TASSERT_TRUE(dispatched.size() == 1 && matches(dispatched[0], {1, 2, 3, 3, 4, 5, 6, 6, 4, 5, 6, 6}));
    }

    {
        // NCHW batches: the two channels must reduce across both samples.
        const std::vector<float> input{1, 3, 10, 14, 5, 7, 18, 22};
        const float s5 = std::sqrt(5.0f), s20 = std::sqrt(20.0f);
        const std::vector<float> expected{1 - 6 / s5, 1 - 2 / s5, -2 - 3 / s20, -2 - 1 / s20,
                                          1 + 2 / s5, 1 + 6 / s5, -2 + 1 / s20, -2 + 3 / s20};
        auto training = input;
        RuntimeLayerOps::computeBatchNorm(training, {2, 0.5f}, {1, -2}, {1000, 1000}, {100, 100},
                                           2, 2, 2, 0.0f, true);
        TASSERT_TRUE(matches(training, expected));
        auto inference = input;
        RuntimeLayerOps::computeBatchNorm(inference, {2, 0.5f}, {1, -2}, {4, 16}, {5, 20},
                                           2, 2, 2, 0.0f, false);
        TASSERT_TRUE(matches(inference, expected));
        Layer batch("batch", "BatchNorm2d", 4);
        batch.in_channels = 2;
        batch.input_height = 1;
        batch.input_width = 2;
        batch.weights = {2, 0.5f, 1, -2};
        batch.running_mean = {4, 16};
        batch.running_var = {5, 20};
        batch.eps = 0;
        std::vector<std::vector<float>> dispatched;
        TASSERT_TRUE(cpu.forwardLayer({&input}, dispatched, batch, false));
        TASSERT_TRUE(dispatched.size() == 1 && matches(dispatched[0], expected));

        const std::vector<float> groups{1, 3, 11, 13};
        auto normalized = groups;
        RuntimeLayerOps::computeLayerNorm(normalized, {2, 3}, {0.5f, -0.5f}, 2, 0.0f);
        TASSERT_TRUE(matches(normalized, {-1.5f, 2.5f, -1.5f, 2.5f}));
        Layer layer("layer", "LayerNorm", 2);
        layer.in_features = 2;
        layer.weights = {2, 3};
        layer.use_bias = false;
        layer.eps = 0;
        TASSERT_TRUE(cpu.forwardLayer({&groups}, dispatched, layer, false));
        TASSERT_TRUE(dispatched.size() == 1 && matches(dispatched[0], {-2, 3, -2, 3}));
        layer.affine = false;
        TASSERT_TRUE(cpu.forwardLayer({&groups}, dispatched, layer, false));
        TASSERT_TRUE(dispatched.size() == 1 && matches(dispatched[0], {-1, 1, -1, 1}));

        const std::vector<float> rms_input{3, 4, 6, 8};
        Layer rms("rms", "RMSNorm", 2);
        rms.in_features = 2;
        rms.weights = {1, 2};
        rms.eps = 0;
        TASSERT_TRUE(cpu.forwardLayer({&rms_input}, dispatched, rms, false));
        const float unit = 1.0f / std::sqrt(12.5f);
        TASSERT_TRUE(dispatched.size() == 1 && matches(dispatched[0], {3 * unit, 8 * unit, 3 * unit, 8 * unit}));
    }
    cpu.shutdown();
    RuntimeLayerOps::setHardwareAcceleration(true);
    return 0;
}
