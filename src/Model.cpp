#include "Model.hpp"
#include "HardwareOpt.hpp"
#include "SIMD_Ops.hpp"
#include "Layers.hpp"
#include "MemoryGuard.hpp"
#include "DynamicTensorAllocator.hpp"
#include "VulkanCompute.hpp"
#include <fstream>
#include <iomanip>
#include <ctime>
#include <iostream>
#include <cmath>
#include <sstream>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <random>
#include <cpuid.h>

// ============================================================================
// Détection des capacités CPU au runtime
// ============================================================================

bool Model::hasAVX2() {
    static bool detected = false;
    static bool result = false;
    
    if (!detected) {
        #if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
            unsigned int eax, ebx, ecx, edx;
            if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
                result = (ebx & (1 << 5)) != 0; // EBX bit 5 = AVX2
            }
        #endif
        detected = true;
    }
    
    return result = detected;
}

bool Model::hasFMA() {
    static bool detected = false;
    static bool result = false;
    
    if (!detected) {
        #if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
            unsigned int eax, ebx, ecx, edx;
            if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
                result = (ecx & (1 << 12)) != 0; // ECX bit 12 = FMA
            }
        #endif
        detected = true;
    }
    
    return result = detected;
}

bool Model::hasF16C() {
    static bool detected = false;
    static bool result = false;
    
    if (!detected) {
        #if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
            unsigned int eax, ebx, ecx, edx;
            if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
                result = (ecx & (1 << 29)) != 0; // ECX bit 29 = F16C
            }
        #endif
        detected = true;
    }
    
    return result = detected;
}

bool Model::hasBMI2() {
    static bool detected = false;
    static bool result = false;
    
    if (!detected) {
        #if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
            unsigned int eax, ebx, ecx, edx;
            if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
                result = (ebx & (1 << 8)) != 0; // EBX bit 8 = BMI2
            }
        #endif
        detected = true;
    }
    
    return result;
}

// Global compute engine (initialized on demand)
static std::unique_ptr<VulkanCompute::ComputeEngine> g_compute_engine = nullptr;
static bool g_compute_available = false;

using json = nlohmann::json;
namespace fs = std::filesystem;

// === constructeurs / destructeurs (déjà présents) ===
Model::Model()
    : tokenizer(20000), encoder(64, 20000), hasTokenizer(true), hasEncoder(true)
{
    tw = 64; th = 64;
    // Tenter d'initialiser le compute engine
    initializeComputeEngine();
}

Model::~Model() {
    shutdownComputeEngine();
}

// ===== Hardware Acceleration =====

bool Model::hasVulkanCompute() const {
    return g_compute_available;
}

bool Model::initializeComputeEngine() {
    if (g_compute_engine) return g_compute_available; // Déjà initialisé
    
    try {
        g_compute_engine = std::make_unique<VulkanCompute::ComputeEngine>();
        g_compute_available = g_compute_engine->initialize();
        
        if (g_compute_available) {
            std::cout << "✓ Hardware acceleration enabled (Vulkan Compute)" << std::endl;
        } else {
            std::cout << "⚠ Vulkan Compute initialization failed, using CPU fallback" << std::endl;
            g_compute_engine.reset();
        }
    } catch (const std::exception& e) {
        std::cerr << "⚠ Vulkan Compute unavailable: " << e.what() << std::endl;
        g_compute_available = false;
        g_compute_engine.reset();
    }
    
    return g_compute_available;
}

void Model::shutdownComputeEngine() {
    if (g_compute_engine) {
        g_compute_engine->cleanup();
        g_compute_engine.reset();
        g_compute_available = false;
    }
}

// === méthodes utilitaires simples (déjà présentes) ===
void Model::setDensity(double d) { densityFactor = (d > 0.0 ? d : 1.0); }
double Model::getDensity() const { return densityFactor; }

void Model::push(const std::string &name, const std::string &type, size_t params_count) {
    layers.push_back({name, type, params_count});
}

size_t Model::totalParamCount() const {
    size_t s = 0;
    for (const auto &L : layers) s += L.paramsCount;
    return s;
}

void Model::allocateParams() {
    size_t tot = totalParamCount();
    
    auto& allocator = DynamicTensorAllocator::instance();
    
    std::cout << "📦 Allocation dynamique de " << tot << " tenseurs..." << std::endl;
    
    params.clear();
    params.resize(tot);
    
    // Utiliser l'allocation dynamique avec compression
    for (size_t i = 0; i < tot; ++i) {
        params[i].Weight = 0;
        params[i].Value = 0;
        // Les données seront allouées à la demande via DynamicTensorAllocator
    }
    
    std::cout << "✓ Tenseurs créés (allocation à la demande activée)" << std::endl;
}

void Model::initializeWeights(const std::string &method, unsigned int seed) {
    if (params.empty()) {
        std::cerr << "⚠️  Cannot initialize weights: params not allocated" << std::endl;
        return;
    }
    
    auto& allocator = DynamicTensorAllocator::instance();
    std::mt19937 gen(seed == 0 ? std::random_device{}() : seed);
    
    std::cout << "🎲 Initializing weights using " << method << " method (dynamic allocation)..." << std::endl;
    
    size_t offset = 0;
    size_t layer_idx = 0;
    for (const auto &layer : layers) {
        if (layer.paramsCount == 0) continue;
        
        // Afficher progression tous les 10 layers
        if (layer_idx % 10 == 0) {
            std::cout << "  Initializing layer " << layer_idx << "/" << layers.size() 
                      << " (" << layer.name << ")..." << std::endl;
        }
        layer_idx++;
        
        // Estimation fan_in/fan_out depuis paramsCount
        // Pour layer typique: params = fan_in * fan_out + fan_out (bias)
        // Approximation: fan_in ≈ fan_out ≈ sqrt(paramsCount)
        int fan_estimate = static_cast<int>(std::sqrt(static_cast<float>(layer.paramsCount)));
        int fan_in = std::max(fan_estimate, 32);   // Min 32 pour éviter std trop élevé
        int fan_out = std::max(fan_estimate, 32);
        
        float std_dev = 0.01f;
        
        if (method == "xavier" || method == "glorot") {
            // Xavier/Glorot: std = sqrt(2 / (fan_in + fan_out))
            std_dev = std::sqrt(2.0f / (fan_in + fan_out));
        }
        else if (method == "he" || method == "kaiming") {
            // He/Kaiming (optimal pour ReLU/GELU): std = sqrt(2 / fan_in)
            // Multiplier par 1.5 pour réseaux profonds (évite vanishing gradients)
            std_dev = 1.5f * std::sqrt(2.0f / fan_in);
        }
        else if (method == "normal") {
            std_dev = 0.05f;  // Augmenté de 0.02 → 0.05
        }
        
        std::normal_distribution<float> dist(0.0f, std_dev);
        
        // Estimer nombre de bias (typiquement ~1-5% des params)
        size_t num_weights = layer.paramsCount;
        size_t estimated_bias = fan_out;  // Approximation: 1 bias par neurone de sortie
        if (estimated_bias > num_weights / 10) {
            estimated_bias = num_weights / 10;  // Cap à 10% des params
        }
        size_t num_pure_weights = num_weights - estimated_bias;
        
        for (size_t i = 0; i < num_weights && (offset + i) < params.size(); ++i) {
            float value;
            
            // Bias initialisé à 0, weights normalement
            if (i >= num_pure_weights) {
                value = 0.0f;  // Bias = 0
            } else {
                value = dist(gen);  // Weights ~ N(0, std²)
            }
            
            // Clip direct sans tanh (préserve magnitude)
            value = std::clamp(value, -3.0f, 3.0f);  // ±3σ capture 99.7%
            
            // Convertir [-3, 3] → [0, 1] → uint16
            float normalized = (value + 3.0f) / 6.0f;  // [-3,3] → [0,1]
            params[offset + i].Weight = static_cast<uint16_t>(normalized * 65535.0f);
            
            // Initialiser aussi les données float si présentes
            if (!params[offset + i].data.empty()) {
                for (auto &d : params[offset + i].data) {
                    d = (i >= num_pure_weights) ? 0.0f : dist(gen);
                }
            }
        }
        
        offset += layer.paramsCount;
    }
    
    std::cout << "✓ Weights initialized (" << params.size() << " parameters)" << std::endl;
}

void Model::updateWeightsWithNoise(float learning_rate, float noise_std) {
    if (params.empty()) return;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> noise_dist(0.0f, noise_std);
    
    for (auto &param : params) {
        float current = static_cast<float>(param.Weight) / 65535.0f;
        float noise = noise_dist(gen);
        float updated = current + learning_rate * noise;
        updated = std::clamp(updated, 0.0f, 1.0f);
        param.Weight = static_cast<uint16_t>(updated * 65535.0f);
        
        // Mettre à jour aussi tensor.data
        if (!param.data.empty()) {
            for (auto &d : param.data) {
                d += learning_rate * noise_dist(gen);
            }
        }
    }
}

std::vector<uint16_t> Model::getWeights() const {
    std::vector<uint16_t> out;
    out.reserve(params.size());
    for (const auto &p : params) out.push_back(p.Weight);
    return out;
}

void Model::setTokenizer(const Tokenizer &t) { tokenizer = t; hasTokenizer = true; }
void Model::setEncoder(const Encoder &e) { encoder = e; hasEncoder = true; }

void Model::forward(std::vector<uint8_t> &out_uint8) const {
    const size_t N = static_cast<size_t>(tw) * static_cast<size_t>(th);
    out_uint8.assign(N, 0);
    if (params.empty()) return;
    for (size_t i = 0; i < N; ++i) {
        size_t idx = i % params.size();
        out_uint8[i] = params[idx].Value;
    }
}

void Model::setOutputTarget(const std::vector<uint8_t> &target) {
    // simple mapping: write target into tail of params[].Value if sizes allow
    size_t needed = target.size();
    if (needed == 0 || params.empty()) return;
    for (size_t i = 0; i < needed; ++i) {
        params[i % params.size()].Value = target[i];
    }
}

void Model::applyParamUpdate(float learning_rate) {
    // VRAIE mise à jour avec gradient descent + momentum
    static std::vector<float> velocity(params.size(), 0.0f);
    float momentum = 0.9f;
    float weight_decay = 0.0001f;
    
    if (velocity.size() != params.size()) {
        velocity.resize(params.size(), 0.0f);
    }
    
    size_t i = 0;
    
#ifdef __AVX2__
    // Vectorisation AVX2 des mises à jour de paramètres
    __m256 scale_w = _mm256_set1_ps(1.0f / 65535.0f);
    __m256 scale_v = _mm256_set1_ps(1.0f / 255.0f);
    __m256 two = _mm256_set1_ps(2.0f);
    __m256 one = _mm256_set1_ps(1.0f);
    __m256 momentum_vec = _mm256_set1_ps(momentum);
    __m256 weight_decay_vec = _mm256_set1_ps(weight_decay);
    __m256 lr_vec = _mm256_set1_ps(learning_rate);
    __m256 max_val = _mm256_set1_ps(65535.0f);
    __m256 zero = _mm256_setzero_ps();
    
    for (; i + 8 <= params.size(); i += 8) {
        // Charger weights
        float w_temp[8], v_temp[8], vel_temp[8];
        for (int j = 0; j < 8; ++j) {
            w_temp[j] = static_cast<float>(params[i + j].Weight);
            v_temp[j] = static_cast<float>(params[i + j].Value);
            vel_temp[j] = velocity[i + j];
        }
        
        __m256 cur = _mm256_loadu_ps(w_temp);
        __m256 tgt_raw = _mm256_loadu_ps(v_temp);
        __m256 vel = _mm256_loadu_ps(vel_temp);
        
        // Normaliser cur: [0,65535] -> [-1,1]
        cur = _mm256_mul_ps(cur, scale_w);
        cur = _mm256_fmsub_ps(cur, two, one);
        
        // Normaliser tgt: [0,255] -> [-1,1]
        __m256 tgt = _mm256_mul_ps(tgt_raw, scale_v);
        tgt = _mm256_fmsub_ps(tgt, two, one);
        
        // Gradient avec weight decay: grad = (tgt - cur) + weight_decay * cur
        __m256 diff = _mm256_sub_ps(tgt, cur);
        __m256 decay_term = _mm256_mul_ps(weight_decay_vec, cur);
        __m256 grad = _mm256_add_ps(diff, decay_term);
        
        // Momentum: velocity = momentum * velocity + grad
        vel = _mm256_fmadd_ps(momentum_vec, vel, grad);
        _mm256_storeu_ps(vel_temp, vel);
        for (int j = 0; j < 8; ++j) velocity[i + j] = vel_temp[j];
        
        // Mise à jour: cur += learning_rate * velocity
        cur = _mm256_fmadd_ps(lr_vec, vel, cur);
        
        // Clamp à [-1, 1]
        cur = _mm256_max_ps(_mm256_set1_ps(-1.0f), _mm256_min_ps(one, cur));
        
        // Dénormaliser: [-1,1] -> [0,1] -> [0,65535]
        cur = _mm256_add_ps(cur, one);
        cur = _mm256_div_ps(cur, two);
        cur = _mm256_mul_ps(cur, max_val);
        
        // Stocker
        _mm256_storeu_ps(w_temp, cur);
        for (int j = 0; j < 8; ++j) {
            params[i + j].Weight = static_cast<uint16_t>(std::lround(w_temp[j]));
        }
    }
#endif
    
    // Fallback scalaire pour les éléments restants
    for (; i < params.size(); ++i) {
        float cur = static_cast<float>(params[i].Weight) / 65535.0f;
        cur = cur * 2.0f - 1.0f; // [0,1] -> [-1,1]
        
        float tgt = static_cast<float>(params[i].Value) / 255.0f;
        tgt = tgt * 2.0f - 1.0f;
        
        // Gradient avec weight decay (L2 regularization)
        float grad = (tgt - cur) + weight_decay * cur;
        
        // Momentum update
        velocity[i] = momentum * velocity[i] + learning_rate * grad;
        
        // Update weight
        float upd = cur + velocity[i];
        
        // Clip pour stabilité
        upd = std::clamp(upd, -1.0f, 1.0f);
        
        // Convertir back to [0,1] puis uint16
        upd = (upd + 1.0f) / 2.0f;
        params[i].Weight = static_cast<uint16_t>(std::lround(upd * 65535.0f));
    }
}

// Multi-optimizer step (SGD, Adam, AdamW)
void Model::optimizerStep(Optimizer &opt, float learning_rate, const Gradients* gradients) {
    size_t n = params.size();
    if (n == 0) return;
    
    // Utiliser le LR decay si configuré, sinon utiliser le learning_rate fourni
    float effective_lr = learning_rate;
    if (opt.decay_strategy != LRDecayStrategy::NONE) {
        effective_lr = opt.getCurrentLR();
    }
    
    switch (opt.type) {
        case OptimizerType::SGD: {
            // Stochastic Gradient Descent (CORRIGÉ)
            // Fallback gradient: L = 0.5 * (pred - target)^2 → dL/dw = (pred - target)
            for (size_t i = 0; i < n; ++i) {
                float grad;
                if (gradients) {
                    grad = gradients->get(i);
                } else {
                    // Fallback: gradient MSE = (current - target)
                    float target = static_cast<float>(params[i].Value) / 255.0f;
                    float current = static_cast<float>(params[i].Weight) / 65535.0f;
                    grad = current - target; // CORRIGÉ: signe correct
                }
                
                float current = static_cast<float>(params[i].Weight) / 65535.0f;
                float updated = current - effective_lr * grad;
                updated = std::clamp(updated, 0.0f, 1.0f);
                params[i].Weight = static_cast<uint16_t>(std::lround(updated * 65535.0f));
            }
            break;
        }
        
        case OptimizerType::ADAM: {
            // Adam optimizer avec optimisations AVX2
            opt.ensure(n);
            opt.step += 1;
            
            const float b1 = opt.beta1, b2 = opt.beta2;
            float bias_correction1 = 1.0f - std::pow(b1, static_cast<float>(opt.step));
            float bias_correction2 = 1.0f - std::pow(b2, static_cast<float>(opt.step));
            if (bias_correction1 <= 0.0f) bias_correction1 = 1e-8f;
            if (bias_correction2 <= 0.0f) bias_correction2 = 1e-8f;
            
            // === OPTIMISATION AVX2: Vectorisation de la mise à jour Adam ===
            const __m256 b1_vec = _mm256_set1_ps(b1);
            const __m256 b2_vec = _mm256_set1_ps(b2);
            const __m256 one_minus_b1 = _mm256_set1_ps(1.0f - b1);
            const __m256 one_minus_b2 = _mm256_set1_ps(1.0f - b2);
            const __m256 bc1_vec = _mm256_set1_ps(bias_correction1);
            const __m256 bc2_vec = _mm256_set1_ps(bias_correction2);
            const __m256 eps_vec = _mm256_set1_ps(opt.eps);
            const __m256 lr_vec = _mm256_set1_ps(effective_lr);
            
            // Vectorized loop
            size_t i = 0;
            for (; i + 8 <= n; i += 8) {
                // Extraire gradients (vectorisé)
                __m256 grad_vec;
                if (gradients) {
                    float grad_buffer[8];
                    for (int j = 0; j < 8; ++j) {
                        grad_buffer[j] = gradients->get(i + j);
                    }
                    grad_vec = _mm256_loadu_ps(grad_buffer);
                } else {
                    float grad_buffer[8];
                    for (int j = 0; j < 8; ++j) {
                        float target = static_cast<float>(params[i + j].Value) / 255.0f;
                        float current = static_cast<float>(params[i + j].Weight) / 65535.0f;
                        grad_buffer[j] = target - current;
                    }
                    grad_vec = _mm256_loadu_ps(grad_buffer);
                }
                
                // Charger m et v
                __m256 m_vec = _mm256_loadu_ps(&opt.m[i]);
                __m256 v_vec = _mm256_loadu_ps(&opt.v[i]);
                
                // m = b1 * m + (1-b1) * grad
                m_vec = _mm256_fmadd_ps(b1_vec, m_vec, _mm256_mul_ps(one_minus_b1, grad_vec));
                
                // v = b2 * v + (1-b2) * grad^2
                __m256 grad_sq = _mm256_mul_ps(grad_vec, grad_vec);
                v_vec = _mm256_fmadd_ps(b2_vec, v_vec, _mm256_mul_ps(one_minus_b2, grad_sq));
                
                // m_hat = m / bias_correction1
                __m256 m_hat = _mm256_div_ps(m_vec, bc1_vec);
                
                // v_hat = v / bias_correction2
                __m256 v_hat = _mm256_div_ps(v_vec, bc2_vec);
                
                // denom = sqrt(v_hat) + eps
                __m256 denom = _mm256_add_ps(_mm256_sqrt_ps(v_hat), eps_vec);
                
                // delta = lr * (m_hat / denom)
                __m256 delta = _mm256_mul_ps(lr_vec, _mm256_div_ps(m_hat, denom));
                
                // Sauvegarder m et v
                _mm256_storeu_ps(&opt.m[i], m_vec);
                _mm256_storeu_ps(&opt.v[i], v_vec);
                
                // Mise à jour des poids (scalar car conversion uint16)
                float delta_arr[8];
                _mm256_storeu_ps(delta_arr, delta);
                for (int j = 0; j < 8; ++j) {
                    // Convertir uint16 → float dans la plage correcte [-3, 3]
                    float current = (static_cast<float>(params[i + j].Weight) / 65535.0f) * 6.0f - 3.0f;
                    float updated = current - delta_arr[j];
                    updated = std::clamp(updated, -3.0f, 3.0f);
                    // Convertir back: [-3, 3] → [0, 1] → uint16
                    float normalized = (updated + 3.0f) / 6.0f;
                    params[i + j].Weight = static_cast<uint16_t>(std::lround(normalized * 65535.0f));
                }
            }
            
            // Remaining elements (scalar)
            for (; i < n; ++i) {
                float grad;
                if (gradients) {
                    grad = gradients->get(i);
                } else {
                    float target = static_cast<float>(params[i].Value) / 255.0f;
                    float current = (static_cast<float>(params[i].Weight) / 65535.0f) * 6.0f - 3.0f;
                    grad = (target * 6.0f - 3.0f) - current;
                }
                
                opt.m[i] = b1 * opt.m[i] + (1.0f - b1) * grad;
                opt.v[i] = b2 * opt.v[i] + (1.0f - b2) * (grad * grad);
                float m_hat = opt.m[i] / bias_correction1;
                float v_hat = opt.v[i] / bias_correction2;
                float denom = std::sqrt(v_hat) + opt.eps;
                float delta = effective_lr * (m_hat / denom);
                
                // Conversion correcte uint16 ↔ float
                float current = (static_cast<float>(params[i].Weight) / 65535.0f) * 6.0f - 3.0f;
                float updated = current - delta;
                updated = std::clamp(updated, -3.0f, 3.0f);
                float normalized = (updated + 3.0f) / 6.0f;
                params[i].Weight = static_cast<uint16_t>(std::lround(normalized * 65535.0f));
            }
            break;
        }
        
        case OptimizerType::ADAMW: {
            // AdamW optimizer (Adam with decoupled weight decay)
            opt.ensure(n);
            opt.step += 1;
            
            const float b1 = opt.beta1, b2 = opt.beta2;
            float bias_correction1 = 1.0f - std::pow(b1, static_cast<float>(opt.step));
            float bias_correction2 = 1.0f - std::pow(b2, static_cast<float>(opt.step));
            if (bias_correction1 <= 0.0f) bias_correction1 = 1e-8f;
            if (bias_correction2 <= 0.0f) bias_correction2 = 1e-8f;
            
            for (size_t i = 0; i < n; ++i) {
                float grad;
                if (gradients) {
                    grad = gradients->get(i);
                } else {
                    float target = static_cast<float>(params[i].Value) / 255.0f;
                    float current = (static_cast<float>(params[i].Weight) / 65535.0f) * 6.0f - 3.0f;
                    grad = (target * 6.0f - 3.0f) - current;
                }
                
                opt.m[i] = b1 * opt.m[i] + (1.0f - b1) * grad;
                opt.v[i] = b2 * opt.v[i] + (1.0f - b2) * (grad * grad);
                float m_hat = opt.m[i] / bias_correction1;
                float v_hat = opt.v[i] / bias_correction2;
                float denom = std::sqrt(v_hat) + opt.eps;
                
                // Conversion correcte uint16 → float [-3, 3]
                float current = (static_cast<float>(params[i].Weight) / 65535.0f) * 6.0f - 3.0f;
                
                // AdamW: Weight decay appliqué directement aux poids (découplé du gradient)
                float weight_decay_term = opt.weight_decay * current;
                float adam_update = effective_lr * (m_hat / denom);
                
                float updated = current - adam_update - effective_lr * weight_decay_term;
                updated = std::clamp(updated, -3.0f, 3.0f);
                
                // Conversion back: [-3, 3] → [0, 1] → uint16
                float normalized = (updated + 3.0f) / 6.0f;
                params[i].Weight = static_cast<uint16_t>(std::lround(normalized * 65535.0f));
            }
            break;
        }
    }
}

Model::DecoderOutput Model::eval(const std::vector<uint8_t> &target) const {
    DecoderOutput out;
    std::vector<uint8_t> gen;
    forward(gen);
    if (gen.size() != target.size() || gen.empty()) { out.mse = -1.0; return out; }
    double s = 0.0;
    for (size_t i = 0; i < gen.size(); ++i) {
        double d = double(gen[i]) - double(target[i]);
        s += d * d;
    }
    out.mse = s / double(gen.size());

    if (!hasTokenizer) return out;
    size_t vs = tokenizer.getVocabSize();
    if (vs == 0) return out;
    // produce trivial logits from generated image
    out.logits.assign(vs, 0.0f);
    for (size_t i = 0; i < out.logits.size(); ++i) out.logits[i] = 1.0f / float(out.logits.size());
    // top-k tokens
    for (size_t i = 0; i < std::min<size_t>(8, out.logits.size()); ++i) out.tokens.push_back(int(i));
    return out;
}

void Model::setLastEncoding(const std::vector<float> &e) { lastEncoding = e; }

// ---------------- file helpers ----------------
// convert MagicToken vector to JSON
static json magic_tokens_to_json(const std::vector<MagicToken> &mvec) {
    json a = json::array();
    for (const auto &m : mvec) {
        json mj;
        mj["modality_mask"] = m.modality_mask;
        mj["seed"] = m.seed;
        mj["embed"] = json::array();
        for (int i = 0; i < 8; ++i) mj["embed"].push_back(m.embed[i]);
        a.push_back(mj);
    }
    return a;
}

// read magic tokens from JSON
static void json_to_magic_tokens(const json &j, std::vector<MagicToken> &outMagic) {
    if (!j.is_array()) return;
    for (const auto &m : j) {
        MagicToken mt{};
        mt.modality_mask = m.value("modality_mask", 0u);
        mt.seed = m.value("seed", 0u);
        if (m.contains("embed") && m["embed"].is_array()) {
            for (size_t i = 0; i < 8 && i < m["embed"].size(); ++i) mt.embed[i] = m["embed"][i].get<float>();
        }
        outMagic.push_back(mt);
    }
}

// ---------------- static persistence helpers ----------------
// helper: sanitize strings in id2token array (replace control chars by '<NL>' or space)
static void sanitize_id2token_json(json &tokj) {
    if (!tokj.is_object() || !tokj.contains("id2token")) return;
    try {
        auto &arr = tokj["id2token"];
        if (!arr.is_array()) return;
        for (auto &el : arr) {
            if (!el.is_string()) continue;
            std::string s = el.get<std::string>();
            bool changed = false;
            for (char &c : s) {
                if (static_cast<unsigned char>(c) <= 0x1F) { // control chars
                    changed = true;
                    c = '<NL>'; // replace with space to avoid embedded newlines
                }
            }
            if (changed) el = s;
        }
    } catch (...) { /* best-effort */ }
}

bool Model::saveCheckpoint(const Tokenizer &tokenizer, const std::vector<MagicToken> &magic_tokens, const fs::path &dir, int epoch) {
    try {
        std::string epoch_name = (epoch >= 0) ? ("epoch_" + std::to_string(epoch)) : std::string("epoch_latest");
        fs::path outdir = dir / epoch_name;
        fs::path tmpdir = outdir.string() + ".tmp";

        if (fs::exists(tmpdir)) fs::remove_all(tmpdir);
        fs::create_directories(tmpdir);

        json tokj;
        // id2Token

        tokj = tokenizer.to_json();

            // sanitize id2token entries to avoid embedded control chars that break JSON files when edited
            sanitize_id2token_json(tokj);

        // write tokenizer.json atomically (write tmp file then rename)
        {
            fs::path tf_tmp = tmpdir / "tokenizer.json.tmp";
            std::ofstream tf(tf_tmp.string(), std::ios::binary);
            if (!tf) { fs::remove_all(tmpdir); return false; }
            tf << std::setw(2) << tokj;
            tf.close();
            fs::rename(tf_tmp, tmpdir / "tokenizer.json");
        }

        // write metadata with epoch and timestamp
        json meta;
        meta["created_by"] = model_name + "::saveCheckpoint";
        meta["name"] = model_name;
        meta["timestamp"] = static_cast<long long>(std::time(nullptr));
        meta["epoch"] = epoch;
        meta["magic_tokens"] = magic_tokens_to_json(magic_tokens);
        meta["num_layers"] = layers.size();
        meta["total_params"] = totalParamCount();
        meta["image_width"] = tw;
        meta["image_height"] = th;
        {
            fs::path mf_tmp = tmpdir / "metadata.json.tmp";
            std::ofstream mf(mf_tmp.string(), std::ios::binary);
            if (!mf) { fs::remove_all(tmpdir); return false; }
            mf << std::setw(2) << meta;
            mf.close();
            fs::rename(mf_tmp, tmpdir / "metadata.json");
        }
        
        // Sauvegarder la structure des layers
        if (!saveLayersStructure(tmpdir / "layers.json")) {
            fs::remove_all(tmpdir);
            return false;
        }
        
        // Sauvegarder les embeddings de l'encoder
        if (hasEncoder && !saveEmbeddings(tmpdir / "embeddings.bin")) {
            fs::remove_all(tmpdir);
            return false;
        }
        
        // Sauvegarder les données des paramètres (tensor.data)
        if (!saveParamsData(tmpdir / "params_data.bin")) {
            fs::remove_all(tmpdir);
            return false;
        }

        // Sauvegarder les poids (weights.u16)
        {
            fs::path wf_tmp = tmpdir / "weights.u16.tmp";
            std::ofstream wf(wf_tmp.string(), std::ios::binary);
            if (!wf) { fs::remove_all(tmpdir); return false; }

            // write params[].Weight as little-endian uint16 array
            for (size_t i = 0; i < params.size(); ++i) {
                uint16_t w = params[i].Weight;
                uint8_t bytes[2];
                bytes[0] = static_cast<uint8_t>(w & 0xFF);
                bytes[1] = static_cast<uint8_t>((w >> 8) & 0xFF);
                wf.write(reinterpret_cast<const char*>(bytes), 2);
                if (!wf) { fs::remove_all(tmpdir); return false; }
            }

            wf.close();
            fs::rename(wf_tmp, tmpdir / "weights.u16");
        }

        if (fs::exists(outdir)) fs::remove_all(outdir);
        fs::rename(tmpdir, outdir);

        return true;
    } catch (...) {
        try { /* best-effort cleanup */ } catch(...) {}
        return false;
    }
}

static void write_u64_le(std::ofstream &f, uint64_t v) {
    uint8_t b[8];
    for (int i = 0; i < 8; ++i) b[i] = static_cast<uint8_t>((v >> (8 * i)) & 0xFF);
    f.write(reinterpret_cast<char*>(b), 8);
}

// writer for a set of float32 tensors into a safetensors-like file.
// Format written:
// [8 bytes little-endian u64] header_length
// [header_length bytes UTF-8 JSON header]
// [binary blob of tensors concatenated as raw little-endian float32]
//
// Header format (JSON object) follows safetensors style:
// { "metadata": {}, "tensors": { "name": { "dtype":"f32", "shape":[N], "data":[offset, length] }, ... } }
static bool write_safetensors_file(const fs::path &outpath, const std::unordered_map<std::string, std::vector<float>> &tensors, std::string *err = nullptr) {
    try {
        // prepare metadata and compute offsets
        json header;
        header["metadata"] = json::object();
        json tensors_meta = json::object();

        uint64_t offset = 0; // data offset after header
        std::vector<std::pair<const std::string*, const std::vector<float>*>> order;
        order.reserve(tensors.size());
        for (const auto &kv : tensors) order.emplace_back(&kv.first, &kv.second);

        // compute total data size to help building header offsets (we don't need it here)
        for (const auto &p : order) {
            const std::string &name = *p.first;
            const std::vector<float> &buf = *p.second;
            uint64_t byte_len = static_cast<uint64_t>(buf.size()) * sizeof(float);
            // record meta: data = [offset, length]
            json m;
            m["dtype"] = "f32";
            m["shape"] = json::array({ static_cast<uint64_t>(buf.size()) });
            m["data"] = json::array({ offset, byte_len });
            tensors_meta[name] = m;
            offset += byte_len;
        }
        header["tensors"] = tensors_meta;

        std::string header_str = header.dump();
        uint64_t header_len = static_cast<uint64_t>(header_str.size());

        // open file and write header length + header
        std::ofstream ofs(outpath.string(), std::ios::binary);
        if (!ofs) {
            if (err) *err = "failed to open output file";
            return false;
        }

        write_u64_le(ofs, header_len);
        ofs.write(header_str.data(), static_cast<std::streamsize>(header_len));

        // now write tensor data in the same order as header (order vector)
        for (const auto &p : order) {
            const std::vector<float> &buf = *p.second;
            if (!buf.empty()) {
                // write raw floats (assume host is little-endian; if not, convert)
                ofs.write(reinterpret_cast<const char*>(buf.data()), static_cast<std::streamsize>(buf.size() * sizeof(float)));
            }
        }

        ofs.close();
        return true;
    } catch (const std::exception &e) {
        if (err) *err = e.what();
        return false;
    } catch (...) {
        if (err) *err = "unknown error";
        return false;
    }
}

// Model::packToSafetensor implementation that delegates to writer above.
// Utilise une map fournie par l'appelant (nom -> float buffer).
bool Model::packToSafetensor(const fs::path &outpath, const std::unordered_map<std::string, std::vector<float>> &tensors) const {
    // create parent dir
    try {
        if (outpath.has_parent_path()) fs::create_directories(outpath.parent_path());
    } catch (...) { /* ignore */ }

    std::string err;
    if (!write_safetensors_file(outpath, tensors, &err)) {
        std::cerr << "packToSafetensor: failed to write " << outpath << " : " << err << "\n";
        return false;
    }
    return true;
}

bool Model::tryLoadExistingModel(const fs::path &ckdir, const fs::path &safep, Tokenizer &outTok, Encoder &outEnc, std::vector<MagicToken> &outMagic) {
    bool loaded_any = false;
    try {
        fs::path sjson = safep; sjson += ".json";
        if (fs::exists(sjson) && fs::is_regular_file(sjson)) {
            try {
                std::ifstream f(sjson);
                if (f) {
                    json full; f >> full;
                    if (full.contains("tokenizer")) { try { outTok.from_json(full["tokenizer"]); loaded_any = true; } catch(...) {} }
                    else if (full.contains("id2token")) { json tj; tj["id2token"] = full["id2token"]; try { outTok.from_json(tj); loaded_any = true; } catch(...) {} }
                    if (full.contains("magic_tokens")) { try { json_to_magic_tokens(full["magic_tokens"], outMagic); loaded_any = true; } catch(...) {} }
                    if (full.contains("encoder")) {
                        try {
                            auto ej = full["encoder"];
                            outEnc.dim = ej.value("dim", outEnc.dim);
                            if (ej.contains("embeddings") && ej["embeddings"].is_array()) {
                                auto &rows = ej["embeddings"];
                                outEnc.vocab_size = (int)rows.size();
                                outEnc.token_embeddings.assign((size_t)outEnc.dim * (size_t)outEnc.vocab_size, 0.0f);
                                for (size_t r = 0; r < rows.size(); ++r)
                                    for (int d = 0; d < outEnc.dim && d < (int)rows[r].size(); ++d)
                                        outEnc.token_embeddings[r * (size_t)outEnc.dim + d] = rows[r][d].get<float>();
                                loaded_any = true;
                            }
                        } catch (...) {}
                    }
                    if (loaded_any) return true;
                }
            } catch (...) {
                // fallback: safep json is invalid/corrupted, ignore and continue to checkpoint folders
            }
        }
    } catch (...) {}

    try {
        if (fs::exists(ckdir) && fs::is_directory(ckdir)) {
            int best_epoch = -1; fs::path best_dir;
            for (auto &p : fs::directory_iterator(ckdir)) {
                if (!p.is_directory()) continue;
                std::string n = p.path().filename().string();
                if (n.rfind("epoch_", 0) == 0) {
                    try { int e = std::stoi(n.substr(6)); if (e > best_epoch) { best_epoch = e; best_dir = p.path(); } } catch(...) {}
                }
            }
            if (best_epoch >= 0 && !best_dir.empty()) {
                fs::path tokp = best_dir / "tokenizer.json";
                fs::path encp = best_dir / "encoder.json";
                fs::path mp = best_dir / "metadata.json";
                fs::path layersp = best_dir / "layers.json";
                fs::path embp = best_dir / "embeddings.bin";
                fs::path paramsp = best_dir / "params_data.bin";
                
                if (fs::exists(tokp)) {
                    try {
                        std::ifstream tf(tokp);
                        json tj;
                        tf >> tj;
                        outTok.from_json(tj);
                        loaded_any = true;
                    } catch(...) {
                        // tokenizer.json is invalid -> fallback to minimal tokenizer
                        try {
                            json minimal;
                            minimal["id2token"] = json::array({ "<PAD>", "<UNK>", "<SEQ>", "<MOD>", "<MAG>", "<NL>" });
                            outTok.from_json(minimal);
                            loaded_any = true;
                        } catch(...) {}
                    }
                }
                if (fs::exists(encp)) {
                    try { std::ifstream ef(encp); json ej; ef >> ej;
                        outEnc.dim = ej.value("dim", outEnc.dim);
                        if (ej.contains("embeddings") && ej["embeddings"].is_array()) {
                            auto &rows = ej["embeddings"];
                            outEnc.vocab_size = (int)rows.size();
                            outEnc.token_embeddings.assign((size_t)outEnc.dim * (size_t)outEnc.vocab_size, 0.0f);
                            for (size_t r = 0; r < rows.size(); ++r)
                                for (int d = 0; d < outEnc.dim && d < (int)rows[r].size(); ++d)
                                    outEnc.token_embeddings[r * (size_t)outEnc.dim + d] = rows[r][d].get<float>();
                            loaded_any = true;
                        }
                    } catch(...) {}
                }
                if (fs::exists(mp)) {
                    try { std::ifstream mf(mp); json mj; mf >> mj; if (mj.contains("magic_tokens")) { json_to_magic_tokens(mj["magic_tokens"], outMagic); loaded_any = true; } } catch(...) {}
                }
                
                // Charger la structure des layers
                if (fs::exists(layersp)) {
                    try {
                        if (loadLayersStructure(layersp)) {
                            std::cout << "✓ Structure des layers chargée depuis " << layersp << std::endl;
                            loaded_any = true;
                        }
                    } catch(...) {
                        std::cerr << "⚠️ Échec du chargement de layers.json" << std::endl;
                    }
                }
                
                // Charger les embeddings
                if (fs::exists(embp)) {
                    try {
                        if (loadEmbeddings(embp)) {
                            std::cout << "✓ Embeddings chargés depuis " << embp << std::endl;
                            loaded_any = true;
                        }
                    } catch(...) {
                        std::cerr << "⚠️ Échec du chargement des embeddings" << std::endl;
                    }
                }
                
                // Charger les données des paramètres
                if (fs::exists(paramsp)) {
                    try {
                        if (loadParamsData(paramsp)) {
                            std::cout << "✓ Données des paramètres chargées depuis " << paramsp << std::endl;
                            loaded_any = true;
                        }
                    } catch(...) {
                        std::cerr << "⚠️ Échec du chargement des params_data" << std::endl;
                    }
                }
                
                if (loaded_any) return true;
            }
        }
    } catch (...) {}

    return loaded_any;
}

// ===== Implémentation des opérations de layer =====

void Model::computeConv2D(const std::vector<float>& input, std::vector<float>& output,
                         const LayerParams& params, int in_h, int in_w, int in_c, int out_c,
                         bool use_hardware) {
    use_hardware = use_hardware && global_use_hardware && hasAVX2();
    
    int out_h = (in_h + 2 * params.padding - params.dilation * (params.kernel_size - 1) - 1) / params.stride + 1;
    int out_w = (in_w + 2 * params.padding - params.dilation * (params.kernel_size - 1) - 1) / params.stride + 1;
    
    if (use_hardware && hasFMA()) {
        // Version hardware optimisée avec FMA saturé
        output.resize(out_h * out_w * out_c, 0.0f);
        
        #pragma omp parallel for collapse(3)
        for (int oc = 0; oc < out_c; ++oc) {
            for (int oh = 0; oh < out_h; ++oh) {
                for (int ow = 0; ow < out_w; ++ow) {
                    __m256 acc0 = _mm256_setzero_ps();
                    __m256 acc1 = _mm256_setzero_ps();
                    __m256 acc2 = _mm256_setzero_ps();
                    
                    int kernel_idx = 0;
                    for (int ic = 0; ic < in_c; ++ic) {
                        for (int kh = 0; kh < params.kernel_size; ++kh) {
                            for (int kw = 0; kw < params.kernel_size; kw += 3) {
                                int ih = oh * params.stride - params.padding + kh * params.dilation;
                                int iw = ow * params.stride - params.padding + kw * params.dilation;
                                
                                if (ih >= 0 && ih < in_h) {
                                    // Charger 3 valeurs pour saturer FMA (3 ops/cycle)
                                    if (kw < params.kernel_size && iw >= 0 && iw < in_w) {
                                        int in_idx = (ic * in_h + ih) * in_w + iw;
                                        __m256 in_val = _mm256_set1_ps(input[in_idx]);
                                        __m256 k_val = _mm256_set1_ps(params.weights[((oc * in_c + ic) * params.kernel_size + kh) * params.kernel_size + kw]);
                                        acc0 = _mm256_fmadd_ps(in_val, k_val, acc0);
                                    }
                                    if (kw + 1 < params.kernel_size && iw + 1 >= 0 && iw + 1 < in_w) {
                                        int in_idx = (ic * in_h + ih) * in_w + iw + 1;
                                        __m256 in_val = _mm256_set1_ps(input[in_idx]);
                                        __m256 k_val = _mm256_set1_ps(params.weights[((oc * in_c + ic) * params.kernel_size + kh) * params.kernel_size + kw + 1]);
                                        acc1 = _mm256_fmadd_ps(in_val, k_val, acc1);
                                    }
                                    if (kw + 2 < params.kernel_size && iw + 2 >= 0 && iw + 2 < in_w) {
                                        int in_idx = (ic * in_h + ih) * in_w + iw + 2;
                                        __m256 in_val = _mm256_set1_ps(input[in_idx]);
                                        __m256 k_val = _mm256_set1_ps(params.weights[((oc * in_c + ic) * params.kernel_size + kh) * params.kernel_size + kw + 2]);
                                        acc2 = _mm256_fmadd_ps(in_val, k_val, acc2);
                                    }
                                }
                            }
                        }
                    }
                    
                    // Somme horizontale des 3 accumulateurs
                    __m256 total = _mm256_add_ps(acc0, _mm256_add_ps(acc1, acc2));
                    __m128 sum_high = _mm256_extractf128_ps(total, 1);
                    __m128 sum_low = _mm256_castps256_ps128(total);
                    __m128 sum128 = _mm_add_ps(sum_low, sum_high);
                    __m128 shuf = _mm_movehdup_ps(sum128);
                    __m128 sums = _mm_add_ps(sum128, shuf);
                    shuf = _mm_movehl_ps(shuf, sums);
                    sums = _mm_add_ss(sums, shuf);
                    
                    float sum = _mm_cvtss_f32(sums);
                    if (!params.bias.empty()) sum += params.bias[oc];
                    
                    output[(oc * out_h + oh) * out_w + ow] = sum;
                }
            }
        }
    } else {
        // Version software (CPU)
        Conv::conv2d(input, output, params.weights, params.bias,
                    in_h, in_w, in_c, out_c, params.kernel_size,
                    params.stride, params.padding, params.dilation);
    }
}

void Model::computeLinear(const std::vector<float>& input, std::vector<float>& output,
                         const LayerParams& params, bool use_hardware) {
    use_hardware = use_hardware && global_use_hardware && hasAVX2();
    
    output.resize(params.out_features, 0.0f);
    
    if (use_hardware && hasFMA()) {
        // Version hardware avec FMA saturé
        SIMD::matmul_avx2(output.data(), input.data(), params.weights.data(),
                         1, params.out_features, params.in_features);
        
        // Ajouter bias
        if (!params.bias.empty()) {
            SIMD::add_vectors_avx2(output.data(), output.data(), params.bias.data(), params.out_features);
        }
    } else {
        // Version software
        for (int o = 0; o < params.out_features; ++o) {
            float sum = 0.0f;
            for (int i = 0; i < params.in_features; ++i) {
                sum += input[i] * params.weights[o * params.in_features + i];
            }
            if (!params.bias.empty()) sum += params.bias[o];
            output[o] = sum;
        }
    }
}

void Model::computeMaxPool2D(const std::vector<float>& input, std::vector<float>& output,
                            int in_h, int in_w, int channels, int kernel_size, int stride,
                            bool use_hardware) {
    use_hardware = use_hardware && global_use_hardware && hasAVX2();
    
    if (stride < 0) stride = kernel_size;
    int out_h = (in_h - kernel_size) / stride + 1;
    int out_w = (in_w - kernel_size) / stride + 1;
    
    output.resize(out_h * out_w * channels);
    
    if (use_hardware) {
        // Version hardware avec AVX2
        #pragma omp parallel for collapse(3)
        for (int c = 0; c < channels; ++c) {
            for (int oh = 0; oh < out_h; ++oh) {
                for (int ow = 0; ow < out_w; ++ow) {
                    __m256 max_vec = _mm256_set1_ps(-std::numeric_limits<float>::infinity());
                    
                    for (int kh = 0; kh < kernel_size; ++kh) {
                        for (int kw = 0; kw < kernel_size; kw += 8) {
                            int ih = oh * stride + kh;
                            int iw = ow * stride + kw;
                            
                            if (kw + 8 <= kernel_size) {
                                __m256 vals = _mm256_loadu_ps(&input[(c * in_h + ih) * in_w + iw]);
                                max_vec = _mm256_max_ps(max_vec, vals);
                            } else {
                                // Scalar fallback pour derniers éléments
                                for (int k = kw; k < kernel_size; ++k) {
                                    float val = input[(c * in_h + ih) * in_w + (ow * stride + k)];
                                    float temp[8];
                                    _mm256_storeu_ps(temp, max_vec);
                                    temp[0] = std::max(temp[0], val);
                                    max_vec = _mm256_loadu_ps(temp);
                                }
                            }
                        }
                    }
                    
                    // Horizontal max
                    float temp[8];
                    _mm256_storeu_ps(temp, max_vec);
                    float max_val = temp[0];
                    for (int i = 1; i < 8; ++i) max_val = std::max(max_val, temp[i]);
                    
                    output[(c * out_h + oh) * out_w + ow] = max_val;
                }
            }
        }
    } else {
        // Version software
        Pooling::maxpool2d(input, output, in_h, in_w, channels, kernel_size, stride);
    }
}

void Model::computeAvgPool2D(const std::vector<float>& input, std::vector<float>& output,
                            int in_h, int in_w, int channels, int kernel_size, int stride,
                            bool use_hardware) {
    use_hardware = use_hardware && global_use_hardware && hasAVX2();
    
    if (stride < 0) stride = kernel_size;
    int out_h = (in_h - kernel_size) / stride + 1;
    int out_w = (in_w - kernel_size) / stride + 1;
    
    output.resize(out_h * out_w * channels);
    
    if (use_hardware) {
        // Version hardware avec AVX2
        float inv_area = 1.0f / (kernel_size * kernel_size);
        __m256 inv_vec = _mm256_set1_ps(inv_area);
        
        #pragma omp parallel for collapse(3)
        for (int c = 0; c < channels; ++c) {
            for (int oh = 0; oh < out_h; ++oh) {
                for (int ow = 0; ow < out_w; ++ow) {
                    __m256 sum_vec = _mm256_setzero_ps();
                    
                    for (int kh = 0; kh < kernel_size; ++kh) {
                        for (int kw = 0; kw < kernel_size; kw += 8) {
                            int ih = oh * stride + kh;
                            int iw = ow * stride + kw;
                            
                            if (kw + 8 <= kernel_size) {
                                __m256 vals = _mm256_loadu_ps(&input[(c * in_h + ih) * in_w + iw]);
                                sum_vec = _mm256_add_ps(sum_vec, vals);
                            }
                        }
                    }
                    
                    // Horizontal sum
                    __m128 sum_high = _mm256_extractf128_ps(sum_vec, 1);
                    __m128 sum_low = _mm256_castps256_ps128(sum_vec);
                    __m128 sum128 = _mm_add_ps(sum_low, sum_high);
                    __m128 shuf = _mm_movehdup_ps(sum128);
                    __m128 sums = _mm_add_ps(sum128, shuf);
                    shuf = _mm_movehl_ps(shuf, sums);
                    sums = _mm_add_ss(sums, shuf);
                    
                    float sum = _mm_cvtss_f32(sums) * inv_area;
                    output[(c * out_h + oh) * out_w + ow] = sum;
                }
            }
        }
    } else {
        // Version software
        Pooling::avgpool2d(input, output, in_h, in_w, channels, kernel_size, stride);
    }
}

void Model::computeActivation(std::vector<float>& data, const std::string& activation_type,
                             float param, bool use_hardware) {
    use_hardware = use_hardware && global_use_hardware && hasAVX2();
    
    if (activation_type == "gelu" && use_hardware) {
        SIMD::gelu_forward_avx2(data.data(), data.data(), data.size());
    } else if (activation_type == "relu") {
        if (use_hardware) {
            size_t n = data.size();
            __m256 zero = _mm256_setzero_ps();
            
            #pragma omp parallel for
            for (size_t i = 0; i < n; i += 8) {
                if (i + 8 <= n) {
                    __m256 vals = _mm256_loadu_ps(&data[i]);
                    vals = _mm256_max_ps(vals, zero);
                    _mm256_storeu_ps(&data[i], vals);
                } else {
                    for (size_t j = i; j < n; ++j) {
                        data[j] = std::max(0.0f, data[j]);
                    }
                }
            }
        } else {
            relu_inplace(data);
        }
    } else if (activation_type == "leaky_relu") {
        leaky_relu_inplace(data, param);
    } else if (activation_type == "tanh") {
        tanh_inplace(data);
    } else if (activation_type == "sigmoid") {
        for (auto& v : data) v = sigmoidf(v);
    } else if (activation_type == "softmax") {
        if (use_hardware) {
            SIMD::softmax_avx2(data.data(), data.data(), data.size());
        } else {
            softmax_inplace(data);
        }
    } else if (activation_type == "elu") {
        elu_inplace(data, param);
    }
}

void Model::computeBatchNorm(std::vector<float>& data, const std::vector<float>& gamma,
                            const std::vector<float>& beta, const std::vector<float>& running_mean,
                            const std::vector<float>& running_var, int batch_size, int channels,
                            int spatial_size, float eps, bool training, bool use_hardware) {
    use_hardware = use_hardware && global_use_hardware && hasAVX2();
    
    if (use_hardware) {
        // Version hardware avec AVX2
        __m256 eps_vec = _mm256_set1_ps(eps);
        
        #pragma omp parallel for
        for (int c = 0; c < channels; ++c) {
            float mean = running_mean[c];
            float var = running_var[c];
            
            if (training) {
                // Calculer mean (AVX2)
                __m256 mean_vec = _mm256_setzero_ps();
                int total_size = batch_size * spatial_size;
                int count = 0;
                
                for (int b = 0; b < batch_size; ++b) {
                    for (int s = 0; s < spatial_size; s += 8) {
                        if (s + 8 <= spatial_size) {
                            __m256 vals = _mm256_loadu_ps(&data[b * channels * spatial_size + c * spatial_size + s]);
                            mean_vec = _mm256_add_ps(mean_vec, vals);
                            count += 8;
                        }
                    }
                }
                
                // Horizontal sum pour mean
                float temp[8];
                _mm256_storeu_ps(temp, mean_vec);
                mean = 0.0f;
                for (int i = 0; i < 8; ++i) mean += temp[i];
                for (int b = 0; b < batch_size; ++b) {
                    for (int s = count; s < spatial_size; ++s) {
                        mean += data[b * channels * spatial_size + c * spatial_size + s];
                    }
                }
                mean /= total_size;
                
                // Calculer variance
                var = 0.0f;
                for (int b = 0; b < batch_size; ++b) {
                    for (int s = 0; s < spatial_size; ++s) {
                        float diff = data[b * channels * spatial_size + c * spatial_size + s] - mean;
                        var += diff * diff;
                    }
                }
                var /= total_size;
            }
            
            __m256 mean_vec = _mm256_set1_ps(mean);
            __m256 inv_std_vec = _mm256_set1_ps(1.0f / std::sqrt(var + eps));
            __m256 gamma_vec = _mm256_set1_ps(gamma[c]);
            __m256 beta_vec = _mm256_set1_ps(beta[c]);
            
            for (int b = 0; b < batch_size; ++b) {
                for (int s = 0; s < spatial_size; s += 8) {
                    int idx = b * channels * spatial_size + c * spatial_size + s;
                    if (s + 8 <= spatial_size) {
                        __m256 vals = _mm256_loadu_ps(&data[idx]);
                        vals = _mm256_sub_ps(vals, mean_vec);
                        vals = _mm256_mul_ps(vals, inv_std_vec);
                        vals = _mm256_mul_ps(vals, gamma_vec);
                        vals = _mm256_add_ps(vals, beta_vec);
                        _mm256_storeu_ps(&data[idx], vals);
                    } else {
                        for (int i = s; i < spatial_size; ++i) {
                            int idx2 = b * channels * spatial_size + c * spatial_size + i;
                            data[idx2] = (data[idx2] - mean) * (1.0f / std::sqrt(var + eps)) * gamma[c] + beta[c];
                        }
                    }
                }
            }
        }
    } else {
        // Version software
        Normalization::batch_norm(data, gamma, beta, running_mean, running_var,
                                 batch_size, channels, spatial_size, eps, training);
    }
}

void Model::computeLayerNorm(std::vector<float>& data, const std::vector<float>& gamma,
                            const std::vector<float>& beta, int normalized_size,
                            float eps, bool use_hardware) {
    use_hardware = use_hardware && global_use_hardware && hasAVX2();
    
    if (use_hardware) {
        // Version hardware avec AVX2
        int num_groups = data.size() / normalized_size;
        __m256 eps_vec = _mm256_set1_ps(eps);
        
        #pragma omp parallel for
        for (int g = 0; g < num_groups; ++g) {
            // Calculer mean
            __m256 mean_vec = _mm256_setzero_ps();
            int base = g * normalized_size;
            
            int i = 0;
            for (; i + 8 <= normalized_size; i += 8) {
                __m256 vals = _mm256_loadu_ps(&data[base + i]);
                mean_vec = _mm256_add_ps(mean_vec, vals);
            }
            
            float temp[8];
            _mm256_storeu_ps(temp, mean_vec);
            float mean = 0.0f;
            for (int j = 0; j < 8; ++j) mean += temp[j];
            for (; i < normalized_size; ++i) mean += data[base + i];
            mean /= normalized_size;
            
            // Calculer variance
            float var = 0.0f;
            for (int i = 0; i < normalized_size; ++i) {
                float diff = data[base + i] - mean;
                var += diff * diff;
            }
            var /= normalized_size;
            
            __m256 mean_vec_bc = _mm256_set1_ps(mean);
            __m256 inv_std = _mm256_set1_ps(1.0f / std::sqrt(var + eps));
            
            // Normaliser
            for (int i = 0; i + 8 <= normalized_size; i += 8) {
                __m256 vals = _mm256_loadu_ps(&data[base + i]);
                __m256 gamma_vec = _mm256_loadu_ps(&gamma[i]);
                __m256 beta_vec = _mm256_loadu_ps(&beta[i]);
                
                vals = _mm256_sub_ps(vals, mean_vec_bc);
                vals = _mm256_mul_ps(vals, inv_std);
                vals = _mm256_mul_ps(vals, gamma_vec);
                vals = _mm256_add_ps(vals, beta_vec);
                
                _mm256_storeu_ps(&data[base + i], vals);
            }
            
            // Remaining elements
            for (int i = (normalized_size / 8) * 8; i < normalized_size; ++i) {
                data[base + i] = (data[base + i] - mean) * (1.0f / std::sqrt(var + eps)) * gamma[i] + beta[i];
            }
        }
    } else {
        // Version software
        Normalization::layer_norm(data, gamma, beta, normalized_size, eps);
    }
}

void Model::computeConvTranspose2D(const std::vector<float>& input, std::vector<float>& output,
                                  const LayerParams& params, int in_h, int in_w, int in_c, int out_c,
                                  bool use_hardware) {
    // ConvTranspose est complexe, utiliser version software
    Conv::conv_transpose2d(input, output, params.weights, params.bias,
                          in_h, in_w, in_c, out_c, params.kernel_size,
                          params.stride, params.padding);
}

void Model::computeAttention(const std::vector<float>& query, const std::vector<float>& key,
                            const std::vector<float>& value, std::vector<float>& output,
                            int seq_len, int d_model, int num_heads, bool use_hardware) {
    use_hardware = use_hardware && global_use_hardware && hasAVX2();
    
    int head_dim = d_model / num_heads;
    output.resize(seq_len * d_model, 0.0f);
    
    std::vector<float> attention_scores(seq_len * seq_len);
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    
    for (int h = 0; h < num_heads; ++h) {
        // Q * K^T avec scaling
        if (use_hardware) {
            SIMD::matmul_transpose_avx2(attention_scores.data(),
                                       &query[h * head_dim], &key[h * head_dim],
                                       seq_len, seq_len, head_dim);
        } else {
            // Scalar version
            for (int i = 0; i < seq_len; ++i) {
                for (int j = 0; j < seq_len; ++j) {
                    float sum = 0.0f;
                    for (int k = 0; k < head_dim; ++k) {
                        sum += query[(i * d_model) + h * head_dim + k] * key[(j * d_model) + h * head_dim + k];
                    }
                    attention_scores[i * seq_len + j] = sum * scale;
                }
            }
        }
        
        // Softmax sur chaque ligne
        for (int i = 0; i < seq_len; ++i) {
            std::vector<float> row(attention_scores.begin() + i * seq_len,
                                 attention_scores.begin() + (i + 1) * seq_len);
            if (use_hardware) {
                SIMD::softmax_avx2(row.data(), row.data(), seq_len);
            } else {
                softmax_inplace(row);
            }
            std::copy(row.begin(), row.end(), attention_scores.begin() + i * seq_len);
        }
        
        // Attention * V
        if (use_hardware) {
            std::vector<float> head_output(seq_len * head_dim);
            SIMD::matmul_avx2(head_output.data(), attention_scores.data(),
                            &value[h * head_dim], seq_len, head_dim, seq_len);
            
            // Copier dans output
            for (int i = 0; i < seq_len; ++i) {
                for (int j = 0; j < head_dim; ++j) {
                    output[i * d_model + h * head_dim + j] = head_output[i * head_dim + j];
                }
            }
        } else {
            for (int i = 0; i < seq_len; ++i) {
                for (int j = 0; j < head_dim; ++j) {
                    float sum = 0.0f;
                    for (int k = 0; k < seq_len; ++k) {
                        sum += attention_scores[i * seq_len + k] * value[(k * d_model) + h * head_dim + j];
                    }
                    output[i * d_model + h * head_dim + j] = sum;
                }
            }
        }
    }
}

void Model::conv2d_same(const std::vector<float> &in, std::vector<float> &out, int W, int H, const std::vector<float> &kernel, int ksize)
{

    out.assign(W * H, 0.0f);
    // Utiliser la version optimis\u00e9e si disponible
    if (global_use_hardware && hasAVX2() && hasFMA()) {
        LayerParams params;
        params.weights = kernel;
        params.kernel_size = ksize;
        params.stride = 1;
        params.padding = ksize / 2;
        
        computeConv2D(in, out, params, H, W, 1, 1, true);
        return;
    }
    
    // Fallback software
    const int khalf = ksize / 2;
    for (int y = 0; y < H; ++y)
    {
        for (int x = 0; x < W; ++x)
        {
            float sum = 0.0f;
            for (int ky = 0; ky < ksize; ++ky)
            {
                const int iy = y + ky - khalf;
                if (iy < 0 || iy >= H)
                    continue;
                for (int kx = 0; kx < ksize; ++kx)
                {
                    const int ix = x + kx - khalf;
                    if (ix < 0 || ix >= W)
                        continue;
                    sum += in[iy * W + ix] * kernel[ky * ksize + kx];
                }
            }
            out[y * W + x] = sum;
        }
    }
}

// --- Définitions vides pour méthodes virtuelles afin de fournir la vtable ---
void Model::buildBackboneUNet(int /*stages*/, int /*blocks_per_stage*/, int /*bottleneck_depth*/) { /* noop */ }
void Model::injectMagicToken(const MagicToken & /*tok*/) { /* noop */ }
void Model::buildTextBranch(const MagicToken & /*tok*/) { /* noop */ }
void Model::buildAudioBranch(const MagicToken & /*tok*/) { /* noop */ }
void Model::buildImageBranch(const MagicToken & /*tok*/) { /* noop */ }
void Model::buildVideoBranch(const MagicToken & /*tok*/) { /* noop */ }

// === Forward/Backward Pass Complet ===

std::vector<float> Model::forwardPass(const std::vector<float> &input, bool training) {
    if (params.empty() || layers.empty()) {
        std::cerr << "⚠️  Cannot perform forward pass: model not initialized" << std::endl;
        return input;
    }
    
    // Réinitialiser l'état du forward si nécessaire
    if (training) {
        forward_state.clear();
        forward_state.is_valid = true;
    }
    
    std::vector<float> x = input;
    size_t param_offset = 0;
    
    if (training) {
        forward_state.layer_inputs.reserve(layers.size());
        forward_state.layer_outputs.reserve(layers.size());
        forward_state.activations.reserve(layers.size());
    }
    
    // Pré-allouer un buffer de sortie réutilisable
    std::vector<float> layer_output;
    layer_output.reserve(x.size());
    
    // Forward pass à travers chaque layer
    for (size_t layer_idx = 0; layer_idx < layers.size(); ++layer_idx) {
        const auto &layer = layers[layer_idx];
        
        if (training) {
            forward_state.layer_inputs.push_back(x);
        }
        
        layer_output.clear(); // Réutiliser au lieu de réallouer
        
        // === DISPATCH HARDWARE ACCELERATED vs CPU ===
        bool use_gpu = g_compute_available && layer.paramsCount > 10000; // Seuil pour GPU
        
        // Traitement selon le type de layer
        if (layer.type == "Conv2d" || layer.type == "ConvTranspose2d") {
            // VRAIE convolution 2D avec kernel spatial
            const int kernel_size = 3;
            const int in_channels = 64; // À adapter selon couche
            const int out_channels = 64;
            const int height = 64, width = 64;
            const int stride = 1, padding = 1;
            
            const size_t output_size = out_channels * height * width;
            if (layer_output.capacity() < output_size) {
                layer_output.reserve(output_size);
            }
            layer_output.resize(output_size, 0.0f);
            
            if (use_gpu) {
                // === GPU PATH: Utiliser ComputeEngine (Vulkan) ===
                // Note: Pour une vraie implémentation, il faudrait créer des buffers GPU
                // et uploader les données. Ici on fait un fallback CPU pour simplicité.
                // TODO: Implémenter les kernels de convolution Vulkan
                std::cout << "⚡ GPU convolution (layer " << layer_idx << ") - TODO: implement kernels" << std::endl;
                use_gpu = false; // Fallback CPU pour l'instant
            }
            
            if (!use_gpu) {
                // === CPU PATH: Convolution parallélisée OpenMP ===
                // Convolution 2D complète (parallélisée - FORCÉE)
                #pragma omp parallel for schedule(dynamic)
                for (int oc = 0; oc < out_channels; ++oc) {
                    for (int oh = 0; oh < height; ++oh) {
                        for (int ow = 0; ow < width; ++ow) {
                            float sum = 0.0f;
                            
                            for (int ic = 0; ic < in_channels; ++ic) {
                                for (int kh = 0; kh < kernel_size; ++kh) {
                                    for (int kw = 0; kw < kernel_size; ++kw) {
                                        int ih = oh * stride + kh - padding;
                                        int iw = ow * stride + kw - padding;
                                        
                                        if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                                            int in_idx = ic * (height * width) + ih * width + iw;
                                            int w_idx = ((oc * in_channels + ic) * kernel_size + kh) * kernel_size + kw;
                                            
                                            if (in_idx < static_cast<int>(x.size()) && 
                                                (param_offset + w_idx) < params.size()) {
                                                float weight = static_cast<float>(params[param_offset + w_idx].Weight) / 65535.0f;
                                                weight = weight * 2.0f - 1.0f;
                                                sum += x[in_idx] * weight;
                                            }
                                        }
                                    }
                                }
                            }
                            
                            int out_idx = oc * (height * width) + oh * width + ow;
                            layer_output[out_idx] = sum;
                        }
                    }
                }
            }
            
            x = layer_output;
            
        } else if (layer.type == "BatchNorm2d") {
            // BatchNorm : normalisation + scale + shift
            layer_output = x;
            
            // Calculer mean et std
            float mean = 0.0f;
            for (float val : x) mean += val;
            mean /= x.size();
            
            float var = 0.0f;
            for (float val : x) {
                float diff = val - mean;
                var += diff * diff;
            }
            var /= x.size();
            float std = std::sqrt(var + 1e-5f);
            
            // Normaliser
            for (size_t i = 0; i < layer_output.size(); ++i) {
                layer_output[i] = (layer_output[i] - mean) / std;
                
                // Appliquer gamma et beta si disponibles
                if (param_offset + i < params.size()) {
                    float gamma = static_cast<float>(params[param_offset + i].Weight) / 32767.5f;
                    layer_output[i] *= gamma;
                }
            }
            
            x = layer_output;
            
        } else if (layer.type == "MaxPool2d") {
            // MaxPool : sous-échantillonnage (stride 2)
            layer_output.resize(x.size() / 2);
            for (size_t i = 0; i < layer_output.size(); ++i) {
                size_t idx1 = i * 2;
                size_t idx2 = i * 2 + 1;
                layer_output[i] = std::max(
                    idx1 < x.size() ? x[idx1] : 0.0f,
                    idx2 < x.size() ? x[idx2] : 0.0f
                );
            }
            x = layer_output;
            
        } else {
            // Layer générique : application linéaire
            layer_output = x;
            x = layer_output;
        }
        
        // Activation ReLU pour les layers de convolution
        if (layer.type == "Conv2d" || layer.type == "ConvTranspose2d") {
            for (auto &val : x) {
                val = std::max(0.0f, val); // ReLU
            }
        }
        
        if (training) {
            forward_state.layer_outputs.push_back(layer_output);
            forward_state.activations.push_back(x);
        }
        
        param_offset += layer.paramsCount;
    }
    
    if (training) {
        forward_state.final_output = x;
    }
    
    return x;
}

Gradients Model::backwardPass(const std::vector<float> &loss_gradient) {
    Gradients grads;
    
    if (!forward_state.is_valid) {
        std::cerr << "⚠️  Cannot perform backward pass: no valid forward state" << std::endl;
        return grads;
    }
    
    if (layers.empty() || params.empty()) {
        return grads;
    }
    
    std::vector<float> grad = loss_gradient;
    size_t param_offset = 0;
    
    // Calculer l'offset total des paramètres
    for (const auto &layer : layers) {
        param_offset += layer.paramsCount;
    }
    
    // Backward pass à travers chaque layer (en ordre inverse)
    for (int layer_idx = layers.size() - 1; layer_idx >= 0; --layer_idx) {
        const auto &layer = layers[layer_idx];
        param_offset -= layer.paramsCount;
        
        const auto &layer_input = forward_state.layer_inputs[layer_idx];
        const auto &layer_output = forward_state.layer_outputs[layer_idx];
        const auto &activation = forward_state.activations[layer_idx];
        
        std::vector<float> grad_input(layer_input.size(), 0.0f);
        
        if (layer.type == "Conv2d" || layer.type == "ConvTranspose2d") {
            // Backward ReLU (parallélisé)
            std::vector<float> grad_pre_relu = grad;
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < grad_pre_relu.size(); ++i) {
                if (i < activation.size() && activation[i] <= 0.0f) {
                    grad_pre_relu[i] = 0.0f; // Gradient est 0 où ReLU a coupé
                }
            }
            
            // Backward Conv : VRAIS gradients avec convolution transposée
            int kernel_size = 3;
            int in_channels = 64;
            int out_channels = 64;
            int height = 64, width = 64;
            int stride = 1, padding = 1;
            
            // Gradient des poids: dL/dW = grad_output ⊗ input (parallélisé sur oc)
            #pragma omp parallel for schedule(dynamic) collapse(2)
            for (int oc = 0; oc < out_channels; ++oc) {
                for (int ic = 0; ic < in_channels; ++ic) {
                    for (int kh = 0; kh < kernel_size; ++kh) {
                        for (int kw = 0; kw < kernel_size; ++kw) {
                            float grad_weight = 0.0f;
                            
                            for (int oh = 0; oh < height; ++oh) {
                                for (int ow = 0; ow < width; ++ow) {
                                    int ih = oh * stride + kh - padding;
                                    int iw = ow * stride + kw - padding;
                                    
                                    if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                                        int out_idx = oc * (height * width) + oh * width + ow;
                                        int in_idx = ic * (height * width) + ih * width + iw;
                                        
                                        if (out_idx < static_cast<int>(grad_pre_relu.size()) &&
                                            in_idx < static_cast<int>(layer_input.size())) {
                                            grad_weight += grad_pre_relu[out_idx] * layer_input[in_idx];
                                        }
                                    }
                                }
                            }
                            
                            int w_idx = ((oc * in_channels + ic) * kernel_size + kh) * kernel_size + kw;
                            if ((param_offset + w_idx) < params.size()) {
                                #pragma omp critical
                                grads.add(param_offset + w_idx, grad_weight);
                            }
                        }
                    }
                }
            }
            
            // Gradient de l'entrée: convolution transposée de grad avec poids flip (parallélisé)
            #pragma omp parallel for schedule(dynamic) collapse(3)
            for (int ic = 0; ic < in_channels; ++ic) {
                for (int ih = 0; ih < height; ++ih) {
                    for (int iw = 0; iw < width; ++iw) {
                        float grad_sum = 0.0f;
                        
                        for (int oc = 0; oc < out_channels; ++oc) {
                            for (int kh = 0; kh < kernel_size; ++kh) {
                                for (int kw = 0; kw < kernel_size; ++kw) {
                                    int oh = ih - kh + padding;
                                    int ow = iw - kw + padding;
                                    
                                    if (oh >= 0 && oh < height && ow >= 0 && ow < width &&
                                        oh % stride == 0 && ow % stride == 0) {
                                        oh /= stride;
                                        ow /= stride;
                                        
                                        int out_idx = oc * (height * width) + oh * width + ow;
                                        int w_idx = ((oc * in_channels + ic) * kernel_size + kh) * kernel_size + kw;
                                        
                                        if (out_idx < static_cast<int>(grad_pre_relu.size()) &&
                                            (param_offset + w_idx) < params.size()) {
                                            float weight = static_cast<float>(params[param_offset + w_idx].Weight) / 65535.0f;
                                            weight = weight * 2.0f - 1.0f;
                                            grad_sum += grad_pre_relu[out_idx] * weight;
                                        }
                                    }
                                }
                            }
                        }
                        
                        int in_idx = ic * (height * width) + ih * width + iw;
                        if (in_idx < static_cast<int>(grad_input.size())) {
                            grad_input[in_idx] = grad_sum;
                        }
                    }
                }
            }
            
            grad = grad_input;
            
        } else if (layer.type == "BatchNorm2d") {
            // Backward BatchNorm avec formule compacte standard (CORRIGÉ)
            // Formule: dx = (1/N) * gamma * invstd * (N*dY - sum(dY) - (x-mean)*invstd^2*sum(dY*(x-mean)))
            int channels = 64; // À adapter
            int height = 64, width = 64;
            int spatial_size = height * width;
            const float eps = 1e-5f;
            
            #pragma omp parallel for schedule(dynamic)
            for (int c = 0; c < channels; ++c) {
                // Calculer mean et variance pour ce canal (forward)
                float mean = 0.0f;
                for (int h = 0; h < height; ++h) {
                    for (int w = 0; w < width; ++w) {
                        int idx = c * spatial_size + h * width + w;
                        if (idx < static_cast<int>(layer_input.size())) {
                            mean += layer_input[idx];
                        }
                    }
                }
                mean /= spatial_size;
                
                float var = 0.0f;
                for (int h = 0; h < height; ++h) {
                    for (int w = 0; w < width; ++w) {
                        int idx = c * spatial_size + h * width + w;
                        if (idx < static_cast<int>(layer_input.size())) {
                            float diff = layer_input[idx] - mean;
                            var += diff * diff;
                        }
                    }
                }
                var /= spatial_size;
                float invstd = 1.0f / std::sqrt(var + eps);
                
                // Récupérer gamma (scale parameter) depuis params
                float gamma = 1.0f;
                if ((param_offset + c * 2) < params.size()) {
                    gamma = static_cast<float>(params[param_offset + c * 2].Weight) / 32767.5f;
                }
                
                // Gradient gamma: sum(dY * x_normalized)
                float grad_gamma = 0.0f;
                for (int h = 0; h < height; ++h) {
                    for (int w = 0; w < width; ++w) {
                        int idx = c * spatial_size + h * width + w;
                        if (idx < static_cast<int>(grad.size()) && 
                            idx < static_cast<int>(layer_input.size())) {
                            float x_normalized = (layer_input[idx] - mean) * invstd;
                            grad_gamma += grad[idx] * x_normalized;
                        }
                    }
                }
                if ((param_offset + c * 2) < params.size()) {
                    #pragma omp critical
                    grads.add(param_offset + c * 2, grad_gamma);
                }
                
                // Gradient beta: sum(dY)
                float grad_beta = 0.0f;
                for (int h = 0; h < height; ++h) {
                    for (int w = 0; w < width; ++w) {
                        int idx = c * spatial_size + h * width + w;
                        if (idx < static_cast<int>(grad.size())) {
                            grad_beta += grad[idx];
                        }
                    }
                }
                if ((param_offset + c * 2 + 1) < params.size()) {
                    #pragma omp critical
                    grads.add(param_offset + c * 2 + 1, grad_beta);
                }
                
                // Calculer sum(dY) et sum(dY * (x - mean))
                float sum_dy = grad_beta; // Déjà calculé
                float sum_dy_xmu = 0.0f;
                for (int h = 0; h < height; ++h) {
                    for (int w = 0; w < width; ++w) {
                        int idx = c * spatial_size + h * width + w;
                        if (idx < static_cast<int>(grad.size()) && 
                            idx < static_cast<int>(layer_input.size())) {
                            sum_dy_xmu += grad[idx] * (layer_input[idx] - mean);
                        }
                    }
                }
                
                // Gradient input avec formule compacte standard (CORRIGÉ)
                // dx = (1/N) * gamma * invstd * (N*dY - sum(dY) - (x-mean)*invstd^2*sum(dY*(x-mean)))
                float invstd2 = invstd * invstd;
                float inv_N = 1.0f / spatial_size;
                
                for (int h = 0; h < height; ++h) {
                    for (int w = 0; w < width; ++w) {
                        int idx = c * spatial_size + h * width + w;
                        if (idx < static_cast<int>(grad_input.size()) &&
                            idx < static_cast<int>(layer_input.size())) {
                            float x_mu = layer_input[idx] - mean;
                            float dx = inv_N * gamma * invstd * (
                                spatial_size * grad[idx] 
                                - sum_dy 
                                - x_mu * invstd2 * sum_dy_xmu
                            );
                            grad_input[idx] = dx;
                        }
                    }
                }
            }
            
            grad = grad_input;
            
        } else if (layer.type == "MaxPool2d") {
            // Backward MaxPool : propager le gradient au max (parallélisé)
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < grad.size(); ++i) {
                size_t idx1 = i * 2;
                size_t idx2 = i * 2 + 1;
                
                if (idx1 < layer_input.size() && idx2 < layer_input.size()) {
                    if (layer_input[idx1] >= layer_input[idx2]) {
                        grad_input[idx1] = grad[i];
                    } else {
                        grad_input[idx2] = grad[i];
                    }
                }
            }
            
            grad = grad_input;
        }
    }
    
    return grads;
}

float Model::computeLoss(const std::vector<float> &prediction, 
                        const std::vector<float> &target, 
                        const std::string &loss_type) {
    if (prediction.size() != target.size()) {
        std::cerr << "⚠️  Prediction and target size mismatch" << std::endl;
        return 0.0f;
    }
    
    float loss = 0.0f;
    
    if (loss_type == "mse") {
        // Mean Squared Error
        for (size_t i = 0; i < prediction.size(); ++i) {
            float diff = prediction[i] - target[i];
            loss += diff * diff;
        }
        loss /= prediction.size();
        
    } else if (loss_type == "mae") {
        // Mean Absolute Error
        for (size_t i = 0; i < prediction.size(); ++i) {
            loss += std::abs(prediction[i] - target[i]);
        }
        loss /= prediction.size();
        
    } else if (loss_type == "bce") {
        // Binary Cross Entropy
        for (size_t i = 0; i < prediction.size(); ++i) {
            float p = std::clamp(prediction[i], 1e-7f, 1.0f - 1e-7f);
            float t = target[i];
            loss += -(t * std::log(p) + (1.0f - t) * std::log(1.0f - p));
        }
        loss /= prediction.size();
    }
    
    return loss;
}

std::vector<float> Model::computeLossGradient(const std::vector<float> &prediction,
                                              const std::vector<float> &target,
                                              const std::string &loss_type) {
    std::vector<float> gradient(prediction.size(), 0.0f);
    
    if (prediction.size() != target.size()) {
        return gradient;
    }
    
    if (loss_type == "mse") {
        // Gradient MSE: 2(pred - target) / n avec AVX2
        size_t size = prediction.size();
        float scale = 2.0f / size;
        size_t i = 0;
        
#ifdef __AVX2__
        __m256 scale_vec = _mm256_set1_ps(scale);
        for (; i + 8 <= size; i += 8) {
            __m256 pred = _mm256_loadu_ps(&prediction[i]);
            __m256 tgt = _mm256_loadu_ps(&target[i]);
            __m256 diff = _mm256_sub_ps(pred, tgt);
            __m256 grad = _mm256_mul_ps(diff, scale_vec);
            _mm256_storeu_ps(&gradient[i], grad);
        }
#endif
        
        for (; i < size; ++i) {
            gradient[i] = scale * (prediction[i] - target[i]);
        }
        
    } else if (loss_type == "mae") {
        // Gradient MAE (CORRIGÉ): sign(pred - target) / n
        // dL/dx = +1/n si (pred - target) > 0
        //       = -1/n si (pred - target) < 0  
        //       =  0   si égalité
        for (size_t i = 0; i < prediction.size(); ++i) {
            float diff = prediction[i] - target[i];
            if (diff > 0.0f) {
                gradient[i] = 1.0f / prediction.size();
            } else if (diff < 0.0f) {
                gradient[i] = -1.0f / prediction.size();
            } else {
                gradient[i] = 0.0f; // Exactement zéro si égalité
            }
        }
        
    } else if (loss_type == "bce") {
        // Gradient BCE avec AVX2
        size_t size = prediction.size();
        float inv_size = 1.0f / size;
        size_t i = 0;
        
#ifdef __AVX2__
        __m256 eps = _mm256_set1_ps(1e-7f);
        __m256 one = _mm256_set1_ps(1.0f);
        __m256 one_minus_eps = _mm256_set1_ps(1.0f - 1e-7f);
        __m256 inv_size_vec = _mm256_set1_ps(inv_size);
        
        for (; i + 8 <= size; i += 8) {
            __m256 p = _mm256_loadu_ps(&prediction[i]);
            __m256 t = _mm256_loadu_ps(&target[i]);
            
            // Clamp p to [1e-7, 1-1e-7]
            p = _mm256_max_ps(p, eps);
            p = _mm256_min_ps(p, one_minus_eps);
            
            // grad = (p - t) / (p * (1 - p)) / size
            __m256 diff = _mm256_sub_ps(p, t);
            __m256 one_minus_p = _mm256_sub_ps(one, p);
            __m256 denom = _mm256_mul_ps(p, one_minus_p);
            __m256 grad = _mm256_div_ps(diff, denom);
            grad = _mm256_mul_ps(grad, inv_size_vec);
            
            _mm256_storeu_ps(&gradient[i], grad);
        }
#endif
        
        for (; i < size; ++i) {
            float p = std::clamp(prediction[i], 1e-7f, 1.0f - 1e-7f);
            float t = target[i];
            gradient[i] = (p - t) / (p * (1.0f - p)) * inv_size;
        }
    }
    
    return gradient;
}

// === build & autoBuildFromDataset ===

void Model::build()
{
    // Construction générique du modèle
    // Peut être surchargée pour définir une architecture spécifique
    
    // Exemple: backbone U-Net simple
    buildBackboneUNet(4, 2, 3);  // 4 stages, 2 blocs par stage, 3 blocs bottleneck
    
    std::cout << "Model::build() - Architecture construite" << std::endl;
    std::cout << "  Couches: " << layers.size() << std::endl;
    std::cout << "  Paramètres totaux: " << totalParamCount() << std::endl;
    
    // Allocation automatique des paramètres
    size_t total = totalParamCount();
    if (total > 0) {
        std::cout << "  Allocation des paramètres..." << std::endl;
        allocateParams();
        std::cout << "  ✓ " << params.size() << " paramètres alloués" << std::endl;
        
        // Initialisation automatique des poids (méthode He par défaut)
        std::cout << "  Initialisation des poids (He)..." << std::endl;
        initializeWeights("he", 0);
        std::cout << "  ✓ Poids initialisés" << std::endl;
    }
}

void Model::autoBuildFromDataset(const std::string &dataset_dir)
{
    // Analyse automatique du dataset pour construire l'architecture appropriée
    
    std::cout << "Model::autoBuildFromDataset(" << dataset_dir << ")" << std::endl;
    
    // Charger le dataset avec cache et validation flexible (min 1 modalité)
    std::vector<DatasetItem> items;
    try {
        items = loadDatasetCached(dataset_dir, 64, 64, 1);  // min_modalities = 1
    } catch (const std::exception &e) {
        std::cerr << "Erreur chargement dataset: " << e.what() << std::endl;
        // Fallback: construction par défaut
        build();
        return;
    }
    
    if (items.empty()) {
        std::cerr << "Dataset vide, construction par défaut" << std::endl;
        build();
        return;
    }
    
    std::cout << "  Items trouvés: " << items.size() << std::endl;
    
    // Analyser les modalités présentes et les linkables
    bool has_text = false;
    bool has_image = false;
    bool has_audio = false;
    bool has_video = false;
    size_t linkable_count = 0;
    
    for (const auto &item : items) {
        if (!item.text_file.empty()) has_text = true;
        if (!item.image_file.empty()) has_image = true;
        if (!item.audio_file.empty()) has_audio = true;
        if (!item.video_file.empty()) has_video = true;
        if (item.is_linked && item.countModalities() >= 2) linkable_count++;
    }
    
    std::cout << "  Modalités détectées:" << std::endl;
    std::cout << "    - Texte:  " << (has_text ? "✓" : "✗") << std::endl;
    std::cout << "    - Image:  " << (has_image ? "✓" : "✗") << std::endl;
    std::cout << "    - Audio:  " << (has_audio ? "✓" : "✗") << std::endl;
    std::cout << "    - Vidéo:  " << (has_video ? "✓" : "✗") << std::endl;
    std::cout << "    - Linkables validés: " << linkable_count << std::endl;
    
    // Construire le backbone de base
    buildBackboneUNet(4, 2, 3);
    
    // Créer les magic tokens pour chaque modalité détectée
    std::vector<MagicToken> magic_tokens;
    
    if (has_text) {
        MagicToken tok;
        tok.modality_mask = 0x01;  // bit 0 = text
        tok.seed = 42;
        for (int i = 0; i < 8; ++i) tok.embed[i] = 0.1f * (i + 1);
        magic_tokens.push_back(tok);
        buildTextBranch(tok);
        injectMagicToken(tok);
        std::cout << "  → Branche texte ajoutée" << std::endl;
    }
    
    if (has_image) {
        MagicToken tok;
        tok.modality_mask = 0x02;  // bit 1 = image
        tok.seed = 43;
        for (int i = 0; i < 8; ++i) tok.embed[i] = 0.2f * (i + 1);
        magic_tokens.push_back(tok);
        buildImageBranch(tok);
        injectMagicToken(tok);
        std::cout << "  → Branche image ajoutée" << std::endl;
    }
    
    if (has_audio) {
        MagicToken tok;
        tok.modality_mask = 0x04;  // bit 2 = audio
        tok.seed = 44;
        for (int i = 0; i < 8; ++i) tok.embed[i] = 0.3f * (i + 1);
        magic_tokens.push_back(tok);
        buildAudioBranch(tok);
        injectMagicToken(tok);
        std::cout << "  → Branche audio ajoutée" << std::endl;
    }
    
    if (has_video) {
        MagicToken tok;
        tok.modality_mask = 0x08;  // bit 3 = video
        tok.seed = 45;
        for (int i = 0; i < 8; ++i) tok.embed[i] = 0.4f * (i + 1);
        magic_tokens.push_back(tok);
        buildVideoBranch(tok);
        injectMagicToken(tok);
        std::cout << "  → Branche vidéo ajoutée" << std::endl;
    }
    
    std::cout << "  Architecture auto-construite:" << std::endl;
    std::cout << "    - Couches: " << layers.size() << std::endl;
    std::cout << "    - Paramètres: " << totalParamCount() << std::endl;
    std::cout << "    - Magic tokens: " << magic_tokens.size() << std::endl;
}

// === Nouvelles méthodes de sauvegarde/chargement ===

bool Model::saveLayersStructure(const fs::path &filepath) const {
    try {
        json layers_json = json::array();
        
        for (const auto &layer : layers) {
            json layer_obj;
            layer_obj["name"] = layer.name;
            layer_obj["type"] = layer.type;
            layer_obj["params_count"] = layer.paramsCount;
            layers_json.push_back(layer_obj);
        }
        
        json root;
        root["layers"] = layers_json;
        root["total_layers"] = layers.size();
        
        std::ofstream ofs(filepath.string());
        if (!ofs) return false;
        
        ofs << std::setw(2) << root;
        ofs.close();
        
        return true;
    } catch (const std::exception &e) {
        std::cerr << "Erreur saveLayersStructure: " << e.what() << std::endl;
        return false;
    }
}

bool Model::loadLayersStructure(const fs::path &filepath) {
    try {
        if (!fs::exists(filepath)) return false;
        
        std::ifstream ifs(filepath.string());
        if (!ifs) return false;
        
        json root;
        ifs >> root;
        
        if (!root.contains("layers") || !root["layers"].is_array()) {
            return false;
        }
        
        layers.clear();
        for (const auto &layer_obj : root["layers"]) {
            LayerDesc layer;
            layer.name = layer_obj.value("name", "");
            layer.type = layer_obj.value("type", "");
            layer.paramsCount = layer_obj.value("params_count", 0);
            layers.push_back(layer);
        }
        
        return true;
    } catch (const std::exception &e) {
        std::cerr << "Erreur loadLayersStructure: " << e.what() << std::endl;
        return false;
    }
}

bool Model::saveEmbeddings(const fs::path &filepath) const {
    try {
        std::ofstream ofs(filepath.string(), std::ios::binary);
        if (!ofs) return false;
        
        // Sauvegarder les dimensions
        uint32_t dim = encoder.dim;
        uint32_t vocab_size = encoder.vocab_size;
        ofs.write(reinterpret_cast<const char*>(&dim), sizeof(uint32_t));
        ofs.write(reinterpret_cast<const char*>(&vocab_size), sizeof(uint32_t));
        
        // Sauvegarder les embeddings
        size_t emb_size = encoder.token_embeddings.size();
        uint64_t size64 = static_cast<uint64_t>(emb_size);
        ofs.write(reinterpret_cast<const char*>(&size64), sizeof(uint64_t));
        
        if (!encoder.token_embeddings.empty()) {
            ofs.write(reinterpret_cast<const char*>(encoder.token_embeddings.data()),
                     emb_size * sizeof(float));
        }
        
        ofs.close();
        return true;
    } catch (const std::exception &e) {
        std::cerr << "Erreur saveEmbeddings: " << e.what() << std::endl;
        return false;
    }
}

bool Model::loadEmbeddings(const fs::path &filepath) {
    try {
        if (!fs::exists(filepath)) return false;
        
        std::ifstream ifs(filepath.string(), std::ios::binary);
        if (!ifs) return false;
        
        // Charger les dimensions
        uint32_t dim, vocab_size;
        ifs.read(reinterpret_cast<char*>(&dim), sizeof(uint32_t));
        ifs.read(reinterpret_cast<char*>(&vocab_size), sizeof(uint32_t));
        
        encoder.dim = dim;
        encoder.vocab_size = vocab_size;
        
        // Charger les embeddings
        uint64_t size64;
        ifs.read(reinterpret_cast<char*>(&size64), sizeof(uint64_t));
        
        if (size64 > 0) {
            encoder.token_embeddings.resize(size64);
            ifs.read(reinterpret_cast<char*>(encoder.token_embeddings.data()),
                    size64 * sizeof(float));
        }
        
        ifs.close();
        hasEncoder = true;
        return true;
    } catch (const std::exception &e) {
        std::cerr << "Erreur loadEmbeddings: " << e.what() << std::endl;
        return false;
    }
}

bool Model::saveParamsData(const fs::path &filepath) const {
    try {
        std::ofstream ofs(filepath.string(), std::ios::binary);
        if (!ofs) return false;
        
        // Nombre total de tenseurs
        uint64_t num_tensors = static_cast<uint64_t>(params.size());
        ofs.write(reinterpret_cast<const char*>(&num_tensors), sizeof(uint64_t));
        
        // Pour chaque tenseur, sauvegarder sa taille et ses données
        for (const auto &tensor : params) {
            uint64_t data_size = static_cast<uint64_t>(tensor.data.size());
            ofs.write(reinterpret_cast<const char*>(&data_size), sizeof(uint64_t));
            
            if (data_size > 0) {
                ofs.write(reinterpret_cast<const char*>(tensor.data.data()),
                         data_size * sizeof(float));
            }
            
            // Sauvegarder aussi Weight, Value, Length
            ofs.write(reinterpret_cast<const char*>(&tensor.Weight), sizeof(uint16_t));
            ofs.write(reinterpret_cast<const char*>(&tensor.Value), sizeof(uint8_t));
            ofs.write(reinterpret_cast<const char*>(&tensor.Length), sizeof(uint16_t));
            
            // Sauvegarder la position
            ofs.write(reinterpret_cast<const char*>(&tensor.Pos.X), sizeof(float));
            ofs.write(reinterpret_cast<const char*>(&tensor.Pos.Y), sizeof(float));
            ofs.write(reinterpret_cast<const char*>(&tensor.Pos.Z), sizeof(float));
            ofs.write(reinterpret_cast<const char*>(&tensor.Pos.W), sizeof(float));
        }
        
        ofs.close();
        return true;
    } catch (const std::exception &e) {
        std::cerr << "Erreur saveParamsData: " << e.what() << std::endl;
        return false;
    }
}

bool Model::loadParamsData(const fs::path &filepath) {
    try {
        if (!fs::exists(filepath)) return false;
        
        std::ifstream ifs(filepath.string(), std::ios::binary);
        if (!ifs) return false;
        
        // Lire le nombre de tenseurs
        uint64_t num_tensors;
        ifs.read(reinterpret_cast<char*>(&num_tensors), sizeof(uint64_t));
        
        auto& guard = MemoryGuard::instance();
        size_t params_overhead = num_tensors * sizeof(tensor);
        
        if (!guard.requestAllocation(params_overhead, "Model params structure")) {
            throw std::runtime_error("❌ MemoryGuard: Impossible d'allouer la structure params");
        }
        
        params.resize(num_tensors);
        
        auto& allocator = DynamicTensorAllocator::instance();
        std::cout << "📦 Chargement dynamique de " << num_tensors << " tenseurs..." << std::endl;
        
        // Charger chaque tenseur avec allocation dynamique
        size_t loaded = 0;
        size_t compressed = 0;
        
        for (size_t i = 0; i < num_tensors; ++i) {
            uint64_t data_size;
            ifs.read(reinterpret_cast<char*>(&data_size), sizeof(uint64_t));
            
            if (data_size > 0) {
                // Allouer via DynamicTensorAllocator
                auto handle = allocator.allocateTensor(data_size, "Tensor[" + std::to_string(i) + "]");
                
                if (!handle) {
                    std::cerr << "⚠️  Allocation échouée au tenseur " << i << "/" << num_tensors << std::endl;
                    std::cerr << "    Tentative de compression des tenseurs précédents..." << std::endl;
                    
                    // Compresser les tenseurs déjà chargés
                    for (size_t j = 0; j < i; ++j) {
                        if (params[j].dynamic_handle) {
                            allocator.compressTensor(
                                static_cast<DynamicTensorAllocator::TensorHandle*>(params[j].dynamic_handle));
                            compressed++;
                        }
                    }
                    
                    // Réessayer
                    handle = allocator.allocateTensor(data_size, "Tensor[" + std::to_string(i) + "]");
                    if (!handle) {
                        std::cerr << "❌ Impossible de charger le modèle même après compression!" << std::endl;
                        ifs.close();
                        return false;
                    }
                }
                
                // Charger les données
                float* data_ptr = allocator.getTensorData(handle);
                if (data_ptr) {
                    ifs.read(reinterpret_cast<char*>(data_ptr), data_size * sizeof(float));
                    params[i].dynamic_handle = handle;
                    params[i].use_dynamic_alloc = true;
                    loaded++;
                    
                    // Compresser périodiquement (tous les 100 tenseurs)
                    if (i % 100 == 0 && i > 0) {
                        allocator.compressTensor(handle);
                    }
                } else {
                    // Fallback: allocation classique
                    params[i].data.resize(data_size);
                    ifs.read(reinterpret_cast<char*>(params[i].data.data()),
                            data_size * sizeof(float));
                }
                
                // Afficher progression
                if (i % 1000 == 0 && i > 0) {
                    std::cout << "  Chargé: " << i << "/" << num_tensors 
                              << " (" << (loaded * 100 / (i + 1)) << "% dynamique, "
                              << compressed << " compressés)" << std::endl;
                }
            }
            
            // Charger Weight, Value, Length
            ifs.read(reinterpret_cast<char*>(&params[i].Weight), sizeof(uint16_t));
            ifs.read(reinterpret_cast<char*>(&params[i].Value), sizeof(uint8_t));
            ifs.read(reinterpret_cast<char*>(&params[i].Length), sizeof(uint16_t));
            
            // Charger la position
            ifs.read(reinterpret_cast<char*>(&params[i].Pos.X), sizeof(float));
            ifs.read(reinterpret_cast<char*>(&params[i].Pos.Y), sizeof(float));
            ifs.read(reinterpret_cast<char*>(&params[i].Pos.Z), sizeof(float));
            ifs.read(reinterpret_cast<char*>(&params[i].Pos.W), sizeof(float));
        }
        
        ifs.close();
        return true;
    } catch (const std::exception &e) {
        std::cerr << "Erreur loadParamsData: " << e.what() << std::endl;
        return false;
    }
}

// --- Fin ---