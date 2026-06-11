// test_runtime_math_attention.cpp
// Solidité mathématique du path SelfAttention : CPU + GPU optionnel.
//
// Poids layout dans le Layer (vu depuis RuntimeLayerDispatch.hpp) :
//   weights[0 .. ed*ed*3-1] = qkv_weight  (Q, K, V concaténés, [3*ed, ed])
//   weights[ed*ed*3 .. end] = out_weight   ([ed, ed])
//   params_count = 4 * ed²
//
// Propriétés vérifiées :
//   P1. Préservation de la forme : output.size() == seq_len × embed_dim
//   P2. Toutes les valeurs sont finies (pas de NaN/inf)
//   P3. Entrée nulle → sortie nulle  (QKV=0, scores=0, attn=uniforme → V=0)
//   P4. Valeur analytique exacte avec W_QKV = I, W_out = I :
//         input = [1,0,0,0 | 0,1,0,0] (2 tokens dim=4)
//         softmax scores connus → comparaison à la valeur calculée
//   P5. Non-linéarité du softmax : SelfAttn(2x) ≠ 2·SelfAttn(x) (en général)
//   P6. GPU ≈ CPU  ±1e-4

#include "test_utils.hpp"
#include "Model.hpp"

#define CHECK(cond) do { if (!(cond)) { std::cerr << "FATAL: " #cond "\n"; std::abort(); } } while(0)

#include <cmath>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef ENABLE_CUDA
#  include "runtimes/cuda/CudaRuntime.hpp"
#endif
#ifdef ENABLE_ROCM
#  include "runtimes/rocm/RocmRuntime.hpp"
#endif

static bool all_finite(const std::vector<float>& v) {
    for (float x : v) { if (!std::isfinite(x)) return false; }
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper : SelfAttention via Model
//   embed_dim = ed,  num_heads = nh,  seq_len = sl,  causal = false
//   poids = identité (W_Q = W_K = W_V = W_out = I)
// ─────────────────────────────────────────────────────────────────────────────
static std::vector<float> run_selfattn_cpu(
    const std::vector<float>& input,   // taille = sl * ed
    int ed, int nh, int sl,
    bool causal,
    const std::vector<float>& all_weights  // taille = 4 * ed * ed
) {
    Model m;
    m.push("x0", "SelfAttention", 0);
    auto& L = m.getMutableLayers()[0];
    L.embed_dim  = ed;
    L.num_heads  = nh;
    L.seq_len    = sl;
    L.causal     = causal;
    L.params_count = static_cast<size_t>(4) * static_cast<size_t>(ed) * static_cast<size_t>(ed);
    m.allocateParams();

    float* w = L.getWeights();
    CHECK(w != nullptr);
    CHECK(L.getWeightsSize() >= all_weights.size());
    for (size_t i = 0; i < all_weights.size(); ++i) w[i] = all_weights[i];

    std::unordered_map<std::string, std::vector<float>> fin;
    fin["x0"] = input;
    std::unordered_map<std::string, std::vector<int>> iin;
    return m.forwardPassNamed(fin, iin, false);
}

// Construit les poids identité 4×ed² (Q=K=V=I, out=I)
static std::vector<float> identity_weights(int ed) {
    // Matrice identité ed×ed
    std::vector<float> I(static_cast<size_t>(ed) * ed, 0.f);
    for (int i = 0; i < ed; ++i) I[static_cast<size_t>(i) * ed + i] = 1.f;

    // qkv_weight = [I; I; I]  (3 × ed²), puis out_weight = I (ed²)
    std::vector<float> weights;
    weights.reserve(4ULL * static_cast<size_t>(ed) * ed);
    for (int k = 0; k < 4; ++k)
        weights.insert(weights.end(), I.begin(), I.end());
    return weights;
}

int main() {
    const int ED = 4;   // embed_dim
    const int NH = 1;   // num_heads
    const int SL = 2;   // seq_len
    const auto W = identity_weights(ED);

    // =========================================================================
    // P1 + P2 : forme et finitude
    // =========================================================================
    {
        const std::vector<float> x(static_cast<size_t>(SL) * ED, 0.5f);
        const auto out = run_selfattn_cpu(x, ED, NH, SL, false, W);
        TASSERT_TRUE(out.size() == static_cast<size_t>(SL) * ED);
        TASSERT_TRUE(all_finite(out));
    }

    // =========================================================================
    // P3 : entrée nulle → sortie nulle
    //   Q=K=V=0, scores=0, attn uniforme, V=0 → output=0
    // =========================================================================
    {
        const std::vector<float> zeros(static_cast<size_t>(SL) * ED, 0.f);
        const auto out = run_selfattn_cpu(zeros, ED, NH, SL, false, W);
        TASSERT_TRUE(out.size() == static_cast<size_t>(SL) * ED);
        for (float v : out) TASSERT_NEAR(v, 0.f, 1e-5f);
    }

    // =========================================================================
    // P4 : valeur analytique exacte
    //   input = [e0 | e1]  (vecteurs de base canoniques dim=4)
    //   W_Q = W_K = W_V = W_out = I
    //   Q = K = V = input
    //   head_dim = 4, scale = 1/sqrt(4) = 0.5
    //   score[0,0] = Q[0]·K[0]·0.5 = 1·0.5 = 0.5
    //   score[0,1] = Q[0]·K[1]·0.5 = 0·0.5 = 0.0
    //   attn[0] = softmax([0.5, 0.0])
    //   attn[0,0] = e^0.5 / (e^0.5 + 1)
    //   attn[0,1] = 1     / (e^0.5 + 1)
    //   out[0] = attn[0,0]*e0 + attn[0,1]*e1
    //          = [attn00, attn01, 0, 0]
    // =========================================================================
    {
        // token0 = [1,0,0,0], token1 = [0,1,0,0]
        const std::vector<float> x = {
            1.f, 0.f, 0.f, 0.f,
            0.f, 1.f, 0.f, 0.f
        };
        const auto out = run_selfattn_cpu(x, ED, NH, SL, false, W);
        TASSERT_TRUE(out.size() == static_cast<size_t>(SL) * ED);
        TASSERT_TRUE(all_finite(out));

        const float exp05 = std::exp(0.5f);
        const float denom = exp05 + 1.f;
        const float attn00 = exp05 / denom;   // ≈ 0.6225
        const float attn01 = 1.f  / denom;   // ≈ 0.3775

        // Token 0 output : [attn00, attn01, 0, 0]
        TASSERT_NEAR(out[0], attn00, 1e-4f);
        TASSERT_NEAR(out[1], attn01, 1e-4f);
        TASSERT_NEAR(out[2], 0.f,    1e-4f);
        TASSERT_NEAR(out[3], 0.f,    1e-4f);
        // Token 1 output : [attn01, attn00, 0, 0]  (symétrique)
        TASSERT_NEAR(out[4], attn01, 1e-4f);
        TASSERT_NEAR(out[5], attn00, 1e-4f);
        TASSERT_NEAR(out[6], 0.f,    1e-4f);
        TASSERT_NEAR(out[7], 0.f,    1e-4f);
    }

    // =========================================================================
    // P5 : non-linéarité du softmax — SelfAttn(2x) ≠ 2·SelfAttn(x)
    // =========================================================================
    {
        const std::vector<float> x1 = {1.f, 0.f, 0.f, 0.f,  0.f, 1.f, 0.f, 0.f};
        std::vector<float> x2(x1.size());
        for (size_t i = 0; i < x1.size(); ++i) x2[i] = 2.f * x1[i];

        const auto out1 = run_selfattn_cpu(x1, ED, NH, SL, false, W);
        const auto out2 = run_selfattn_cpu(x2, ED, NH, SL, false, W);

        // Au moins un élément doit différer de 2×out1 (à cause du softmax)
        bool any_nonlinear = false;
        for (size_t i = 0; i < out1.size(); ++i) {
            if (std::fabs(out2[i] - 2.f * out1[i]) > 1e-3f) {
                any_nonlinear = true;
                break;
            }
        }
        TASSERT_TRUE(any_nonlinear);
    }

    // =========================================================================
    // Causal mask : le token 0 ne doit pas être influencé par le token 1
    //   Avec mask causal, attn[0,1] = 0 → out[0] = V[0]
    // =========================================================================
    {
        const std::vector<float> x = {
            1.f, 0.f, 0.f, 0.f,
            0.f, 1.f, 0.f, 0.f
        };
        const auto out = run_selfattn_cpu(x, ED, NH, SL, true, W);
        TASSERT_TRUE(out.size() == static_cast<size_t>(SL) * ED);
        TASSERT_TRUE(all_finite(out));
        // Token 0 : uniquement attn[0,0]=1, donc out[0..3] = V[0] = [1,0,0,0]
        TASSERT_NEAR(out[0], 1.f, 1e-4f);
        TASSERT_NEAR(out[1], 0.f, 1e-4f);
        TASSERT_NEAR(out[2], 0.f, 1e-4f);
        TASSERT_NEAR(out[3], 0.f, 1e-4f);
    }

    // =========================================================================
    // P6 – GPU CUDA ≈ CPU  (forma + finitude + accord numérique)
    // =========================================================================
#ifdef ENABLE_CUDA
    {
        CudaRuntime rt;
        RuntimeConfig cfg;
        cfg.attention_enabled = true;
        cfg.attention_min_ops = 0;
        if (!rt.initialize(cfg)) {
            std::cout << "SKIP: no CUDA device (SelfAttention)\n";
        } else {
            Model m;
            m.push("x0", "SelfAttention", 0);
            auto& L = m.getMutableLayers()[0];
            L.embed_dim   = ED;
            L.num_heads   = NH;
            L.seq_len     = SL;
            L.causal      = false;
            L.params_count = 4ULL * ED * ED;
            m.allocateParams();
            float* w = L.getWeights();
            TASSERT_TRUE(w != nullptr);
            for (size_t i = 0; i < W.size(); ++i) w[i] = W[i];

            const std::vector<float> x = {
                1.f, 0.f, 0.f, 0.f,
                0.f, 1.f, 0.f, 0.f
            };
            const std::vector<const std::vector<float>*> inputs = {&x};
            std::vector<std::vector<float>> gpu_out;
            const bool ok = rt.forwardLayer(inputs, gpu_out, L, false);
            TASSERT_TRUE(ok);
            TASSERT_TRUE(gpu_out.size() == 1);
            TASSERT_TRUE(gpu_out[0].size() == static_cast<size_t>(SL) * ED);
            TASSERT_TRUE(all_finite(gpu_out[0]));

            // Comparaison avec le CPU
            const auto cpu_ref = run_selfattn_cpu(x, ED, NH, SL, false, W);
            for (size_t i = 0; i < cpu_ref.size(); ++i) {
                TASSERT_NEAR(gpu_out[0][i], cpu_ref[i], 1e-4f);
            }
        }
    }
#endif  // ENABLE_CUDA

#ifdef ENABLE_ROCM
    {
        RocmRuntime rt;
        RuntimeConfig cfg;
        cfg.attention_enabled = true;
        cfg.attention_min_ops = 0;
        if (!rt.initialize(cfg)) {
            std::cout << "SKIP: no ROCm device (SelfAttention)\n";
        } else {
            Model m;
            m.push("x0", "SelfAttention", 0);
            auto& L = m.getMutableLayers()[0];
            L.embed_dim   = ED;
            L.num_heads   = NH;
            L.seq_len     = SL;
            L.causal      = false;
            L.params_count = 4ULL * ED * ED;
            m.allocateParams();
            float* w = L.getWeights();
            TASSERT_TRUE(w != nullptr);
            for (size_t i = 0; i < W.size(); ++i) w[i] = W[i];

            const std::vector<float> x = {
                1.f, 0.f, 0.f, 0.f,
                0.f, 1.f, 0.f, 0.f
            };
            const std::vector<const std::vector<float>*> inputs = {&x};
            std::vector<std::vector<float>> gpu_out;
            const bool ok = rt.forwardLayer(inputs, gpu_out, L, false);
            TASSERT_TRUE(ok);
            TASSERT_TRUE(gpu_out.size() == 1);
            TASSERT_TRUE(gpu_out[0].size() == static_cast<size_t>(SL) * ED);
            TASSERT_TRUE(all_finite(gpu_out[0]));

            const auto cpu_ref = run_selfattn_cpu(x, ED, NH, SL, false, W);
            for (size_t i = 0; i < cpu_ref.size(); ++i) {
                TASSERT_NEAR(gpu_out[0][i], cpu_ref[i], 1e-4f);
            }
        }
    }
#endif  // ENABLE_ROCM

    return 0;
}
