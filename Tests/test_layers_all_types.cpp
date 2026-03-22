#include "test_utils.hpp"

#include "Model.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

static void require_or_throw(bool cond, const char* msg) {
    if (!cond) throw std::runtime_error(msg);
}

static void require_size_eq(size_t actual, size_t expected, const std::string& label) {
    if (actual != expected) {
        throw std::runtime_error(label + ": expected size=" + std::to_string(expected) + ", got=" + std::to_string(actual));
    }
}

static bool all_finite(const std::vector<float>& v) {
    for (float x : v) {
        if (!std::isfinite(x)) return false;
    }
    return true;
}

static float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

static void fill_weights(Layer& layer, float value) {
    float* w = layer.getWeights();
    const size_t n = layer.getWeightsSize();
    if (!w || n == 0) return;
    for (size_t i = 0; i < n; ++i) w[i] = value;
}

static std::vector<float> run_one_layer(
    const std::string& type,
    const std::vector<std::string>& input_names,
    const std::unordered_map<std::string, std::vector<float>>& fin,
    bool training,
    void (*configure)(Layer&),
    void (*init_weights)(Layer&)
) {
    Model m;
    m.push("t0", type, 0);

    auto& layers = m.getMutableLayers();
    require_or_throw(layers.size() == 1, "run_one_layer: expected exactly 1 layer");
    Layer& L = layers[0];

    L.inputs = input_names;
    if (configure) configure(L);
    m.allocateParams();
    if (init_weights) init_weights(L);

    std::unordered_map<std::string, std::vector<int>> iin;
    return m.forwardPassNamed(fin, iin, training);
}

static void noop(Layer&) {}

int main() {
    try {
    // ------------------------------------------------------------------
    // Linear
    // ------------------------------------------------------------------
    {
        auto cfg = [](Layer& L) {
            L.in_features = 2;
            L.out_features = 3;
            L.use_bias = true;
            L.params_count = static_cast<size_t>(L.out_features) * static_cast<size_t>(L.in_features)
                + (L.use_bias ? static_cast<size_t>(L.out_features) : 0ULL);
        };
        auto init = [](Layer& L) {
            // weights: [out, in] row-major, then bias
            // y0 = 1*x0 + 2*x1 + 0
            // y1 = 0*x0 + 1*x1 + 1
            // y2 = -1*x0 + 0*x1 + 0
            float* w = L.getWeights();
            require_or_throw(w != nullptr, "Linear: weights pointer null");
            require_or_throw(L.getWeightsSize() >= 3 * 2 + 3, "Linear: weights size too small");
            w[0] = 1;  w[1] = 2;
            w[2] = 0;  w[3] = 1;
            w[4] = -1; w[5] = 0;
            w[6] = 0;  w[7] = 1;  w[8] = 0;
        };

        std::unordered_map<std::string, std::vector<float>> fin;
        fin["x"] = {1.0f, 2.0f};
        const auto out = run_one_layer("Linear", {"x"}, fin, /*training=*/false, cfg, init);
        TASSERT_TRUE(out.size() == 3);
        TASSERT_NEAR(out[0], 5.0f, 1e-6f);
        TASSERT_NEAR(out[1], 3.0f, 1e-6f);
        TASSERT_NEAR(out[2], -1.0f, 1e-6f);
    }

    // ------------------------------------------------------------------
    // Conv2d (CHW), kernel 1 -> simple scale
    // ------------------------------------------------------------------
    {
        auto cfg = [](Layer& L) {
            L.in_channels = 1;
            L.out_channels = 1;
            L.input_height = 2;
            L.input_width = 2;
            L.kernel_size = 1;
            L.stride = 1;
            L.padding = 0;
            const int k = L.get_kernel_h();
            L.params_count = static_cast<size_t>(L.out_channels) * static_cast<size_t>(L.in_channels)
                * static_cast<size_t>(k) * static_cast<size_t>(k);
        };
        auto init = [](Layer& L) {
            float* w = L.getWeights();
            require_or_throw(w != nullptr, "Conv2d: weights pointer null");
            require_or_throw(L.getWeightsSize() >= 1, "Conv2d: weights size too small");
            w[0] = 2.0f;
        };

        std::unordered_map<std::string, std::vector<float>> fin;
        fin["x"] = {1, 2, 3, 4};
        const auto out = run_one_layer("Conv2d", {"x"}, fin, false, cfg, init);
        TASSERT_TRUE(out.size() == 4);
        TASSERT_NEAR(out[0], 2.0f, 1e-6f);
        TASSERT_NEAR(out[1], 4.0f, 1e-6f);
        TASSERT_NEAR(out[2], 6.0f, 1e-6f);
        TASSERT_NEAR(out[3], 8.0f, 1e-6f);
    }

    // ------------------------------------------------------------------
    // DepthwiseConv2d (CHW), kernel 1 -> simple scale
    // ------------------------------------------------------------------
    {
        auto cfg = [](Layer& L) {
            L.in_channels = 1;
            L.input_height = 2;
            L.input_width = 2;
            L.kernel_h = 1;
            L.kernel_w = 1;
            L.stride_h = 1;
            L.stride_w = 1;
            L.pad_h = 0;
            L.pad_w = 0;
            L.use_bias = false;
            const int k = (L.kernel_h > 0) ? L.kernel_h : 3;
            L.params_count = static_cast<size_t>(L.in_channels) * static_cast<size_t>(k) * static_cast<size_t>(k)
                + (L.use_bias ? static_cast<size_t>(L.in_channels) : 0ULL);
        };
        auto init = [](Layer& L) {
            float* w = L.getWeights();
            require_or_throw(w != nullptr, "DepthwiseConv2d: weights pointer null");
            require_or_throw(L.getWeightsSize() >= 1, "DepthwiseConv2d: weights size too small");
            w[0] = 3.0f;
        };

        std::unordered_map<std::string, std::vector<float>> fin;
        fin["x"] = {1, 2, 3, 4};
        const auto out = run_one_layer("DepthwiseConv2d", {"x"}, fin, false, cfg, init);
        TASSERT_TRUE(out.size() == 4);
        TASSERT_NEAR(out[0], 3.0f, 1e-6f);
        TASSERT_NEAR(out[1], 6.0f, 1e-6f);
        TASSERT_NEAR(out[2], 9.0f, 1e-6f);
        TASSERT_NEAR(out[3], 12.0f, 1e-6f);
    }

    // ------------------------------------------------------------------
    // Conv1d, kernel 1 -> simple scale
    // ------------------------------------------------------------------
    {
        auto cfg = [](Layer& L) {
            L.in_channels = 1;
            L.out_channels = 1;
            L.kernel_h = 1;
            L.stride_h = 1;
            L.pad_h = 0;
            L.use_bias = false;
            const int k = (L.kernel_h > 0) ? L.kernel_h : 3;
            L.params_count = static_cast<size_t>(L.out_channels) * static_cast<size_t>(L.in_channels) * static_cast<size_t>(k)
                + (L.use_bias ? static_cast<size_t>(L.out_channels) : 0ULL);
        };
        auto init = [](Layer& L) {
            float* w = L.getWeights();
            require_or_throw(w != nullptr, "Conv1d: weights pointer null");
            require_or_throw(L.getWeightsSize() >= 1, "Conv1d: weights size too small");
            w[0] = 4.0f;
        };

        std::unordered_map<std::string, std::vector<float>> fin;
        fin["x"] = {1, 2, 3};
        const auto out = run_one_layer("Conv1d", {"x"}, fin, false, cfg, init);
        TASSERT_TRUE(out.size() == 3);
        TASSERT_NEAR(out[0], 4.0f, 1e-6f);
        TASSERT_NEAR(out[1], 8.0f, 1e-6f);
        TASSERT_NEAR(out[2], 12.0f, 1e-6f);
    }

    // ------------------------------------------------------------------
    // Activations (ReLU, GELU, SiLU, Tanh, Sigmoid)
    // ------------------------------------------------------------------
    {
        std::unordered_map<std::string, std::vector<float>> fin;
        fin["x"] = {-1.0f, 0.0f, 1.0f};

        const auto relu = run_one_layer("ReLU", {"x"}, fin, false, noop, nullptr);
        TASSERT_TRUE(relu.size() == 3);
        TASSERT_NEAR(relu[0], 0.0f, 1e-6f);
        TASSERT_NEAR(relu[1], 0.0f, 1e-6f);
        TASSERT_NEAR(relu[2], 1.0f, 1e-6f);

        const auto tanh_out = run_one_layer("Tanh", {"x"}, fin, false, noop, nullptr);
        TASSERT_TRUE(tanh_out.size() == 3);
        TASSERT_NEAR(tanh_out[0], std::tanh(-1.0f), 1e-6f);
        TASSERT_NEAR(tanh_out[1], 0.0f, 1e-6f);
        TASSERT_NEAR(tanh_out[2], std::tanh(1.0f), 1e-6f);

        const auto sig = run_one_layer("Sigmoid", {"x"}, fin, false, noop, nullptr);
        TASSERT_TRUE(sig.size() == 3);
        TASSERT_NEAR(sig[0], sigmoid(-1.0f), 1e-6f);
        TASSERT_NEAR(sig[1], 0.5f, 1e-6f);
        TASSERT_NEAR(sig[2], sigmoid(1.0f), 1e-6f);

        const auto silu = run_one_layer("SiLU", {"x"}, fin, false, noop, nullptr);
        TASSERT_TRUE(silu.size() == 3);
        TASSERT_NEAR(silu[1], 0.0f, 1e-6f);
        TASSERT_NEAR(silu[2], 1.0f * sigmoid(1.0f), 1e-6f);

        // GELU: on valide des points simples (0 -> 0, 1 -> ~0.8413)
        std::unordered_map<std::string, std::vector<float>> fin2;
        fin2["x"] = {0.0f, 1.0f};
        const auto gelu = run_one_layer("GELU", {"x"}, fin2, false, noop, nullptr);
        TASSERT_TRUE(gelu.size() == 2);
        TASSERT_NEAR(gelu[0], 0.0f, 1e-6f);
        TASSERT_NEAR(gelu[1], 0.8413f, 2e-3f);
    }

    // ------------------------------------------------------------------
    // Softmax / LogSoftmax
    // ------------------------------------------------------------------
    {
        std::unordered_map<std::string, std::vector<float>> fin;
        fin["x"] = {0.0f, 0.0f};

        auto cfg = [](Layer& L) {
            L.axis = -1;
        };

        const auto sm = run_one_layer("Softmax", {"x"}, fin, false, cfg, nullptr);
        TASSERT_TRUE(sm.size() == 2);
        TASSERT_NEAR(sm[0], 0.5f, 1e-6f);
        TASSERT_NEAR(sm[1], 0.5f, 1e-6f);

        const auto lsm = run_one_layer("LogSoftmax", {"x"}, fin, false, cfg, nullptr);
        TASSERT_TRUE(lsm.size() == 2);
        const float ln2 = std::log(2.0f);
        TASSERT_NEAR(lsm[0], -ln2, 1e-5f);
        TASSERT_NEAR(lsm[1], -ln2, 1e-5f);
    }

    // ------------------------------------------------------------------
    // LayerNorm (normalized_size = in_features)
    // ------------------------------------------------------------------
    {
        auto cfg = [](Layer& L) {
            L.in_features = 4;
            L.affine = true;
            L.use_bias = true;
            L.eps = 1e-6f;
            const int n = std::max(1, L.in_features);
            L.params_count = static_cast<size_t>(n) + (L.use_bias ? static_cast<size_t>(n) : 0ULL);
        };
        auto init = [](Layer& L) {
            // gamma[4] then beta[4]
            float* w = L.getWeights();
            require_or_throw(w != nullptr, "LayerNorm: weights pointer null");
            require_or_throw(L.getWeightsSize() >= 8, "LayerNorm: weights size too small");
            for (int i = 0; i < 4; ++i) w[i] = 1.0f;
            for (int i = 0; i < 4; ++i) w[4 + i] = 0.0f;
        };
        std::unordered_map<std::string, std::vector<float>> fin;
        fin["x"] = {1, 2, 3, 4};
        const auto out = run_one_layer("LayerNorm", {"x"}, fin, false, cfg, init);
        TASSERT_TRUE(out.size() == 4);
        TASSERT_TRUE(all_finite(out));

        float mean = 0.0f;
        for (float v : out) mean += v;
        mean /= 4.0f;
        float var = 0.0f;
        for (float v : out) {
            const float d = v - mean;
            var += d * d;
        }
        var /= 4.0f;
        TASSERT_NEAR(mean, 0.0f, 1e-5f);
        TASSERT_NEAR(var, 1.0f, 1e-4f);
    }

    // ------------------------------------------------------------------
    // RMSNorm (avec gamma=1)
    // ------------------------------------------------------------------
    {
        auto cfg = [](Layer& L) {
            L.affine = true;
            L.eps = 1e-6f;
            // For this test we feed 2 elements.
            L.params_count = 2;
        };
        auto init = [](Layer& L) {
            fill_weights(L, 1.0f);
        };
        std::unordered_map<std::string, std::vector<float>> fin;
        fin["x"] = {3.0f, 4.0f};
        const auto out = run_one_layer("RMSNorm", {"x"}, fin, false, cfg, init);
        TASSERT_TRUE(out.size() == 2);
        const float rms = std::sqrt((9.0f + 16.0f) / 2.0f + 1e-6f);
        TASSERT_NEAR(out[0], 3.0f / rms, 1e-5f);
        TASSERT_NEAR(out[1], 4.0f / rms, 1e-5f);
    }

    // ------------------------------------------------------------------
    // Dropout / AlphaDropout : en inference, identité
    // ------------------------------------------------------------------
    {
        std::unordered_map<std::string, std::vector<float>> fin;
        fin["x"] = {1.0f, 2.0f, 3.0f};

        auto cfg = [](Layer& L) {
            L.dropout_p = 0.9f;
        };
        const auto d0 = run_one_layer("Dropout", {"x"}, fin, /*training=*/false, cfg, nullptr);
        TASSERT_TRUE(d0 == fin["x"]);

        const auto ad0 = run_one_layer("AlphaDropout", {"x"}, fin, /*training=*/false, cfg, nullptr);
        TASSERT_TRUE(ad0 == fin["x"]);
    }

    // ------------------------------------------------------------------
    // Pooling (MaxPool2d / AvgPool2d / GlobalAvgPool2d / TokenMeanPool)
    // ------------------------------------------------------------------
    {
        std::unordered_map<std::string, std::vector<float>> fin;
        fin["x"] = {1, 2, 3, 4}; // 1ch 2x2

        auto cfg_mp = [](Layer& L) {
            L.in_channels = 1;
            L.input_height = 2;
            L.input_width = 2;
            L.kernel_size = 2;
            L.stride = 2;
            L.padding = 0;
        };
        const auto mp = run_one_layer("MaxPool2d", {"x"}, fin, false, cfg_mp, nullptr);
        require_size_eq(mp.size(), 1, "MaxPool2d");
        TASSERT_NEAR(mp[0], 4.0f, 1e-6f);

        const auto ap = run_one_layer("AvgPool2d", {"x"}, fin, false, cfg_mp, nullptr);
        require_size_eq(ap.size(), 1, "AvgPool2d");
        TASSERT_NEAR(ap[0], 2.5f, 1e-5f);

        auto cfg_gap = [](Layer& L) {
            L.in_channels = 1;
            L.input_height = 2;
            L.input_width = 2;
        };
        const auto gap = run_one_layer("GlobalAvgPool2d", {"x"}, fin, false, cfg_gap, nullptr);
        require_size_eq(gap.size(), 1, "GlobalAvgPool2d");
        TASSERT_NEAR(gap[0], 2.5f, 1e-5f);

        // TokenMeanPool: input = [seq_len, embed_dim]
        std::unordered_map<std::string, std::vector<float>> fin2;
        fin2["x"] = {
            1, 2, 3,
            4, 5, 6
        };
        auto cfg_tmp = [](Layer& L) {
            L.seq_len = 2;
            L.embed_dim = 3;
        };
        const auto tmp = run_one_layer("TokenMeanPool", {"x"}, fin2, false, cfg_tmp, nullptr);
        TASSERT_TRUE(tmp.size() == 3);
        TASSERT_NEAR(tmp[0], 2.5f, 1e-6f);
        TASSERT_NEAR(tmp[1], 3.5f, 1e-6f);
        TASSERT_NEAR(tmp[2], 4.5f, 1e-6f);
    }

    // ------------------------------------------------------------------
    // Elementwise ops (Add/Subtract/Multiply/Divide)
    // ------------------------------------------------------------------
    {
        std::unordered_map<std::string, std::vector<float>> fin;
        fin["x"] = {1, 2, 3};
        fin["y"] = {4, 5, 6};

        const auto add = run_one_layer("Add", {"x", "y"}, fin, false, noop, nullptr);
        TASSERT_TRUE(add.size() == 3);
        TASSERT_NEAR(add[0], 5.0f, 1e-6f);
        TASSERT_NEAR(add[1], 7.0f, 1e-6f);
        TASSERT_NEAR(add[2], 9.0f, 1e-6f);

        const auto sub = run_one_layer("Subtract", {"x", "y"}, fin, false, noop, nullptr);
        TASSERT_TRUE(sub.size() == 3);
        TASSERT_NEAR(sub[0], -3.0f, 1e-6f);
        TASSERT_NEAR(sub[1], -3.0f, 1e-6f);
        TASSERT_NEAR(sub[2], -3.0f, 1e-6f);

        const auto mul = run_one_layer("Multiply", {"x", "y"}, fin, false, noop, nullptr);
        TASSERT_TRUE(mul.size() == 3);
        TASSERT_NEAR(mul[0], 4.0f, 1e-6f);
        TASSERT_NEAR(mul[1], 10.0f, 1e-6f);
        TASSERT_NEAR(mul[2], 18.0f, 1e-6f);

        const auto div = run_one_layer("Divide", {"x", "y"}, fin, false, noop, nullptr);
        TASSERT_TRUE(div.size() == 3);
        TASSERT_NEAR(div[0], 0.25f, 1e-6f);
        TASSERT_NEAR(div[1], 0.4f, 1e-6f);
        TASSERT_NEAR(div[2], 0.5f, 1e-6f);
    }

    // ------------------------------------------------------------------
    // Concat / Split+Concat roundtrip
    // ------------------------------------------------------------------
    {
        // Concat simple
        {
            std::unordered_map<std::string, std::vector<float>> fin;
            fin["x"] = {1, 2};
            fin["y"] = {3};
            fin["z"] = {4, 5};
            auto cfg = [](Layer& L) { L.concat_axis = 0; };
            const auto out = run_one_layer("Concat", {"x", "y", "z"}, fin, false, cfg, nullptr);
            TASSERT_TRUE(out.size() == 5);
            TASSERT_NEAR(out[0], 1.0f, 1e-6f);
            TASSERT_NEAR(out[4], 5.0f, 1e-6f);
        }

        // Split puis Concat: [1,2,3,4] -> split [2,2] -> concat = original
        {
            Model m;
            m.push("split", "Split", 0);
            m.push("cat", "Concat", 0);
            auto& layers = m.getMutableLayers();
            TASSERT_TRUE(layers.size() == 2);
            layers[0].inputs = {"x"};
            layers[0].output = "s";
            layers[0].split_sizes = {2, 2};
            layers[0].split_axis = 0;
            layers[1].inputs = {"s_0", "s_1"};
            layers[1].output = "x";
            layers[1].concat_axis = 0;

            m.allocateParams();

            std::unordered_map<std::string, std::vector<float>> fin;
            std::unordered_map<std::string, std::vector<int>> iin;
            fin["x"] = {1, 2, 3, 4};

            const auto out = m.forwardPassNamed(fin, iin, false);
            TASSERT_TRUE(out.size() == 4);
            for (size_t i = 0; i < 4; ++i) {
                TASSERT_NEAR(out[i], fin["x"][i], 1e-6f);
            }
        }
    }

    // ------------------------------------------------------------------
    // MatMul
    // ------------------------------------------------------------------
    {
        Model m;
        m.push("mm", "MatMul", 0);
        auto& L = m.getMutableLayers()[0];
        L.inputs = {"a", "b"};
        L.output = "x";
        L.in_features = 2;   // M
        L.out_features = 3;  // K
        L.embed_dim = 2;     // N
        m.allocateParams();

        std::unordered_map<std::string, std::vector<float>> fin;
        std::unordered_map<std::string, std::vector<int>> iin;
        // A: 2x3
        fin["a"] = {
            1, 2, 3,
            4, 5, 6
        };
        // B: 3x2
        fin["b"] = {
            7, 8,
            9, 10,
            11, 12
        };
        const auto out = m.forwardPassNamed(fin, iin, false);
        TASSERT_TRUE(out.size() == 4);
        // C = A @ B
        // [ 58  64]
        // [139 154]
        TASSERT_NEAR(out[0], 58.0f, 1e-6f);
        TASSERT_NEAR(out[1], 64.0f, 1e-6f);
        TASSERT_NEAR(out[2], 139.0f, 1e-6f);
        TASSERT_NEAR(out[3], 154.0f, 1e-6f);
    }

    // ------------------------------------------------------------------
    // Embedding (float ids) + EmbeddingBag
    // ------------------------------------------------------------------
    {
        // Embedding: vocab=3, dim=2
        auto cfg = [](Layer& L) {
            L.vocab_size = 3;
            L.embed_dim = 2;
            L.padding_idx = -1;
            L.params_count = static_cast<size_t>(L.vocab_size) * static_cast<size_t>(L.embed_dim);
        };
        auto init = [](Layer& L) {
            float* w = L.getWeights();
            require_or_throw(w != nullptr, "Embedding: weights pointer null");
            require_or_throw(L.getWeightsSize() >= 6, "Embedding: weights size too small");
            // token 0 -> [0,0], token 1 -> [1,10], token 2 -> [2,20]
            w[0] = 0; w[1] = 0;
            w[2] = 1; w[3] = 10;
            w[4] = 2; w[5] = 20;
        };
        std::unordered_map<std::string, std::vector<float>> fin;
        fin["x"] = {1.0f, 2.0f};
        const auto out = run_one_layer("Embedding", {"x"}, fin, false, cfg, init);
        TASSERT_TRUE(out.size() == 4);
        TASSERT_NEAR(out[0], 1.0f, 1e-6f);
        TASSERT_NEAR(out[1], 10.0f, 1e-6f);
        TASSERT_NEAR(out[2], 2.0f, 1e-6f);
        TASSERT_NEAR(out[3], 20.0f, 1e-6f);

        // EmbeddingBag: ids [1,2,1], offsets [0,2,3] -> bag0=sum(ids[0..2))=tok1+tok2, bag1=sum(ids[2..3))=tok1
        auto cfg2 = [](Layer& L) {
            L.vocab_size = 3;
            L.embed_dim = 2;
            L.padding_idx = -1;
            L.params_count = static_cast<size_t>(L.vocab_size) * static_cast<size_t>(L.embed_dim);
        };
        std::unordered_map<std::string, std::vector<float>> fin2;
        fin2["ids"] = {1.0f, 2.0f, 1.0f};
        fin2["offs"] = {0.0f, 2.0f, 3.0f};
        const auto out2 = run_one_layer("EmbeddingBag", {"ids", "offs"}, fin2, false, cfg2, init);
        TASSERT_TRUE(out2.size() == 4);
        // bag0 = [1,10]+[2,20]=[3,30]
        // bag1 = [1,10]
        TASSERT_NEAR(out2[0], 3.0f, 1e-6f);
        TASSERT_NEAR(out2[1], 30.0f, 1e-6f);
        TASSERT_NEAR(out2[2], 1.0f, 1e-6f);
        TASSERT_NEAR(out2[3], 10.0f, 1e-6f);
    }

    // ------------------------------------------------------------------
    // PixelShuffle + ZeroPad2d (shape correctness + spot check)
    // ------------------------------------------------------------------
    {
        // PixelShuffle r=2, in_channels=4, H=W=1 -> out_channels=1, H=W=2
        auto cfg = [](Layer& L) {
            L.scale_h = 2.0f;
            L.in_channels = 4;
            L.input_height = 1;
            L.input_width = 1;
        };
        std::unordered_map<std::string, std::vector<float>> fin;
        fin["x"] = {10, 11, 12, 13};
        const auto out = run_one_layer("PixelShuffle", {"x"}, fin, false, cfg, nullptr);
        TASSERT_TRUE(out.size() == 4);
        // Layout attendu: oc=0, positions (0,0)->ic0, (0,1)->ic1, (1,0)->ic2, (1,1)->ic3
        TASSERT_NEAR(out[0], 10.0f, 1e-6f);
        TASSERT_NEAR(out[1], 11.0f, 1e-6f);
        TASSERT_NEAR(out[2], 12.0f, 1e-6f);
        TASSERT_NEAR(out[3], 13.0f, 1e-6f);

        // ZeroPad2d: 1ch 1x1 pad=1 -> 3x3, center=val
        auto cfgp = [](Layer& L) {
            L.in_channels = 1;
            L.input_height = 1;
            L.input_width = 1;
            L.pad_h = 1;
            L.pad_w = 1;
        };
        std::unordered_map<std::string, std::vector<float>> finp;
        finp["x"] = {5.0f};
        const auto pad = run_one_layer("ZeroPad2d", {"x"}, finp, false, cfgp, nullptr);
        TASSERT_TRUE(pad.size() == 9);
        TASSERT_NEAR(pad[4], 5.0f, 1e-6f); // center
        TASSERT_NEAR(pad[0], 0.0f, 1e-6f);
        TASSERT_NEAR(pad[8], 0.0f, 1e-6f);
    }

    // ------------------------------------------------------------------
    // Recurrent smoke: RNN/GRU/LSTM with zero weights -> zero output
    // ------------------------------------------------------------------
    {
        auto run_recurrent = [](const std::string& type) {
            Model m;
            m.push("r", type, 0);
            auto& L = m.getMutableLayers()[0];
            L.inputs = {"x"};
            L.output = "x";
            L.seq_len = 2;
            L.in_features = 3;
            L.out_features = 4;
            L.use_bias = true;

            // allocateParams() relies on Layer::params_count.
            const int I = L.in_features;
            const int H = L.out_features;
            if (type == "RNN") {
                L.params_count = static_cast<size_t>(H) * static_cast<size_t>(I)
                    + static_cast<size_t>(H) * static_cast<size_t>(H)
                    + static_cast<size_t>(2 * H);
            } else if (type == "GRU") {
                L.params_count = static_cast<size_t>(3 * H) * static_cast<size_t>(I)
                    + static_cast<size_t>(3 * H) * static_cast<size_t>(H)
                    + static_cast<size_t>(6 * H);
            } else {
                // LSTM
                L.params_count = static_cast<size_t>(4 * H) * static_cast<size_t>(I)
                    + static_cast<size_t>(4 * H) * static_cast<size_t>(H)
                    + static_cast<size_t>(8 * H);
            }
            m.allocateParams();
            fill_weights(L, 0.0f);

            std::unordered_map<std::string, std::vector<float>> fin;
            std::unordered_map<std::string, std::vector<int>> iin;
            fin["x"] = {
                1, 2, 3,
                4, 5, 6
            };
            const auto out = m.forwardPassNamed(fin, iin, false);
            require_or_throw(out.size() == static_cast<size_t>(L.seq_len * L.out_features), "Recurrent: output size mismatch");
            for (float v : out) {
                if (!nearf(v, 0.0f, 1e-6f)) {
                    throw std::runtime_error("Recurrent: expected zero output");
                }
            }
        };

        run_recurrent("RNN");
        run_recurrent("GRU");
        run_recurrent("LSTM");
    }

    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
