#include "test_utils.hpp"
#include "runtimes/LayerOps.hpp"
#include "runtimes/cpu/LayerOps.hpp"
#include "runtimes/cpu/LayerOpsExt.hpp"
#include "runtimes/cpu/MatMulKernels.hpp"
#include "runtimes/cpu/RuntimeLayerDispatch.hpp"

#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <stdexcept>
#include <vector>

int main() {
    RuntimeLayerOps::setHardwareAcceleration(false);
    #ifdef _OPENMP
    omp_set_dynamic(0);
    omp_set_num_threads(4);
    #endif

    for (int in_features : {17, 1027}) for (bool bias : {false,true}) {
        const int out_features=in_features>1000?513:19;
        Layer layer;layer.type_enum=LayerType::Linear;
        layer.in_features=in_features;layer.out_features=out_features;
        layer.seq_len=1;layer.use_bias=bias;
        const size_t matrix_size=size_t(in_features)*out_features;
        layer.weights.resize(matrix_size+(bias?out_features:0),.125f);
        layer.grad_weights.resize(layer.weights.size(),.013f);
        if(bias)layer.grad_bias.resize(out_features,.02f);
        std::vector<float> x(in_features),dy(out_features);
        for(int i=0;i<in_features;++i)x[i]=float(i%13-6)*.125f;
        for(int o=0;o<out_features;++o)dy[o]=float(o-9)*.0625f;
        std::vector<std::vector<float>> dx;
        for(int accumulation=1;accumulation<=2;++accumulation){
            TASSERT_TRUE(RuntimeLayerDispatch::cpu_backward_layer({&x},{&dy},dx,layer,true));
            for(int o=0;o<out_features;++o){
                for(int i=0;i<in_features;++i)
                    TASSERT_NEAR(layer.grad_weights[size_t(o)*in_features+i],.013f+accumulation*dy[o]*x[i],1e-5f);
                if(bias){
                    TASSERT_NEAR(layer.grad_weights[matrix_size+o],.013f+accumulation*dy[o],1e-5f);
                    TASSERT_NEAR(layer.grad_bias[o],.02f+accumulation*dy[o],1e-5f);
                }
            }
            float sum=0.f;for(float g:dy)sum+=g*.125f;
            for(float value:dx[0])TASSERT_NEAR(value,sum,1e-6f);
        }
    }

    // A small graph must execute without even a serialized OpenMP region.
    // `parallel for if(false)` still enters one and adds a cost to every op.
    #ifdef _OPENMP
    {
        std::vector<int> levels(63, -1);
        RuntimeParallel::forEach(levels.size(), 1, [&](size_t i) {
            levels[i] = omp_get_level();
        });
        for (int level : levels) TASSERT_TRUE(level == 0);
    }
    #endif
    for (size_t count : {size_t(1), size_t(63), size_t(4099), size_t(65537), size_t(262147)}) {
        std::vector<float> a(count), b(count);
        for (size_t i = 0; i < count; ++i) {
            a[i] = float(int(i % 17) - 8) * .125f;
            b[i] = float(i % 5) * .25f;
        }
        const auto sum = LayerOps::add_forward(a, b);
        const auto activated = LayerOps::relu_forward(sum);
        const auto result = LayerOps::multiply_forward(activated, b);
        for (size_t i = 0; i < count; ++i) {
            TASSERT_NEAR(result[i], std::max(0.f, a[i] + b[i]) * b[i], 1e-6f);
        }
    }

    {
        Layer layer;
        layer.in_channels = 1;
        layer.out_channels = 1;
        layer.kernel_h = 1;
        layer.weights = {2.0f};
        std::vector<float> input(262144);
        for (size_t i = 0; i < input.size(); ++i) {
            input[i] = static_cast<float>(i % 101) * 0.25f;
        }

        const auto output = LayerOpsExt::conv1d_forward(input, layer);
        TASSERT_TRUE(output.size() == input.size());
        for (size_t i = 0; i < input.size(); ++i) {
            TASSERT_NEAR(output[i], input[i] * 2.0f, 1e-6f);
        }

        const auto activated = LayerOps::relu_forward(input);
        for (size_t i = 0; i < input.size(); ++i) {
            TASSERT_NEAR(activated[i], input[i], 1e-6f);
        }
    }

    {
        Layer layer;
        layer.in_channels = 1;
        layer.input_height = 2;
        layer.input_width = 2;
        layer.pad_h = 1;
        layer.pad_w = 1;
        const auto padded = LayerOpsExt::zero_pad2d_forward({1, 2, 3, 4, 99}, layer);
        const std::vector<float> expected = {0, 0, 0, 0, 0, 1, 2, 0,
                                             0, 3, 4, 0, 0, 0, 0, 0};
        TASSERT_TRUE(padded == expected);
        bool rejected = false;
        try {
            LayerOpsExt::zero_pad2d_forward({1, 2, 3}, layer);
        } catch (const std::runtime_error&) {
            rejected = true;
        }
        TASSERT_TRUE(rejected);
    }

    {
        constexpr size_t columns = (1u << 20) + 5;
        const std::vector<float> left = {1.5f, -2.0f, 0.25f, 3.0f};
        std::vector<float> right(4 * columns);
        for (size_t i = 0; i < right.size(); ++i) {
            right[i] = static_cast<float>(static_cast<int>(i % 13) - 6) * 0.125f;
        }
        std::vector<float> output(columns);

        HardwareOpt::matmul_fma_saturated(output.data(), left.data(), right.data(), 1, columns, 4);
        for (size_t column = 0; column < columns; ++column) {
            float expected = 0.0f;
            for (size_t row = 0; row < left.size(); ++row) {
                expected += left[row] * right[row * columns + column];
            }
            TASSERT_NEAR(output[column], expected, 1e-5f);
        }
        const auto first = output;
        HardwareOpt::matmul_add_fma_saturated(output.data(), left.data(), right.data(), 1, columns, 4);
        for (size_t column = 0; column < columns; ++column) {
            TASSERT_NEAR(output[column], 2.0f * first[column], 1e-5f);
        }
    }

    {
        constexpr size_t columns = (1u << 20) + 5;
        const std::vector<float> row = {1.0f, -0.5f, 2.0f, 0.25f};
        std::vector<float> matrix(4 * columns);
        std::vector<float> output(columns);
        for (size_t i = 0; i < matrix.size(); ++i) {
            matrix[i] = static_cast<float>(static_cast<int>(i % 11) - 5) * 0.1f;
        }

        SIMD::matmul_transpose_avx2(output.data(), row.data(), matrix.data(), 1, columns, row.size());
        for (size_t column = 0; column < columns; ++column) {
            float expected = 0.0f;
            for (size_t k = 0; k < row.size(); ++k) {
                expected += row[k] * matrix[column * row.size() + k];
            }
            TASSERT_NEAR(output[column], expected, 1e-5f);
        }

        RuntimeMatMul::forwardBatched(row.data(), matrix.data(), output.data(),
            1, 1, row.size(), columns);
        for (size_t column = 0; column < columns; ++column) {
            float expected = 0.0f;
            for (size_t k = 0; k < row.size(); ++k) {
                expected += row[k] * matrix[k * columns + column];
            }
            TASSERT_NEAR(output[column], expected, 1e-5f);
        }
    }

    {
        Layer layer;
        layer.type_enum = LayerType::Softmax;
        layer.axis = 1;
        layer.shape = {512, 128};
        std::vector<float> input(512 * 128);
        for (size_t i = 0; i < input.size(); ++i) {
            input[i] = static_cast<float>(static_cast<int>(i % 19) - 9) * 0.125f;
        }
        const auto output = LayerOps::softmax_forward(input, layer);
        TASSERT_TRUE(output.size() == input.size());
        for (size_t row = 0; row < 512; ++row) {
            float sum = 0.0f;
            for (size_t column = 0; column < 128; ++column) {
                sum += output[row * 128 + column];
            }
            TASSERT_NEAR(sum, 1.0f, 1e-5f);
        }
    }

    {
        std::vector<float> values = {-2.0f, 0.0f, 3.0f};
        RuntimeLayerOps::computeActivation(values, "leaky_relu", 0.1f);
        TASSERT_NEAR(values[0], -0.2f, 1e-6f);
        TASSERT_NEAR(values[1], 0.0f, 1e-6f);
        TASSERT_NEAR(values[2], 3.0f, 1e-6f);

        RuntimeLayerOps::computeActivation(values, "softmax");
        TASSERT_NEAR(values[0] + values[1] + values[2], 1.0f, 1e-6f);
        TASSERT_TRUE(values[2] > values[1] && values[1] > values[0]);
    }

    {
        RuntimeLayerOps::LayerParams params;
        params.in_features = 2;
        params.out_features = 2;
        params.weights = {1.0f, 2.0f, 3.0f, 4.0f};
        params.bias = {0.5f, -0.5f};

        std::vector<float> output;
        RuntimeLayerOps::computeLinear({2.0f, -1.0f}, output, params);
        TASSERT_TRUE(output.size() == 2);
        TASSERT_NEAR(output[0], 0.5f, 1e-6f);
        TASSERT_NEAR(output[1], 1.5f, 1e-6f);
    }

    {
        const std::vector<float> left = {1.0f, 4.0f, -2.0f};
        const std::vector<float> right = {3.0f, 2.0f, 5.0f};
        std::vector<float> merged;
        RuntimeLayerOps::branchMerge(left, right, merged, MergeOperation::AVERAGE);
        TASSERT_TRUE(merged.size() == 3);
        TASSERT_NEAR(merged[0], 2.0f, 1e-6f);
        TASSERT_NEAR(merged[1], 3.0f, 1e-6f);
        TASSERT_NEAR(merged[2], 1.5f, 1e-6f);

        std::vector<std::vector<float>> split;
        RuntimeLayerOps::branchSplit({1.0f, 2.0f, 3.0f, 4.0f, 5.0f}, split, {2, 3});
        TASSERT_TRUE(split.size() == 2);
        TASSERT_TRUE(split[0] == std::vector<float>({1.0f, 2.0f}));
        TASSERT_TRUE(split[1] == std::vector<float>({3.0f, 4.0f, 5.0f}));

        bool rejected_merge = false;
        try {
            RuntimeLayerOps::branchMerge({1.0f}, {1.0f, 2.0f}, merged, MergeOperation::ADD);
        } catch (const std::invalid_argument&) {
            rejected_merge = true;
        }
        TASSERT_TRUE(rejected_merge);

        bool rejected_split = false;
        try {
            RuntimeLayerOps::branchSplit({1.0f, 2.0f}, split, {1});
        } catch (const std::invalid_argument&) {
            rejected_split = true;
        }
        TASSERT_TRUE(rejected_split);
    }

    RuntimeLayerOps::setHardwareAcceleration(true);
    return 0;
}
