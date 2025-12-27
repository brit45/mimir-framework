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
#include <atomic>
#include <mutex>

// ============================================================================
// Implémentation des méthodes Layer
// ============================================================================

float* Layer::getWeights() {
    if (weight_block) return weight_block->getData();
    return weights.data();
}

const float* Layer::getWeights() const {
    if (weight_block) return weight_block->getData();
    return weights.data();
}

size_t Layer::getWeightsSize() const {
    if (weight_block) return weight_block->getSize();
    return weights.size();
}

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
    // Pattern init_once thread-safe avec atomic
    static std::atomic<bool> initialized{false};
    static std::mutex init_mutex;
    
    if (initialized.load(std::memory_order_acquire)) {
        return g_compute_available;
    }
    
    std::lock_guard<std::mutex> lock(init_mutex);
    
    // Double-check après lock
    if (initialized.load(std::memory_order_relaxed)) {
        return g_compute_available;
    }
    
    try {
        g_compute_engine = std::make_unique<VulkanCompute::ComputeEngine>();
        g_compute_available = g_compute_engine->initialize();
        
        if (g_compute_available) {
            std::cout << "✓ Vulkan Compute initialized" << std::endl;
        } else {
            std::cout << "⚠ Vulkan Compute initialization failed, using CPU fallback" << std::endl;
            g_compute_engine.reset();
        }
    } catch (const std::exception& e) {
        std::cerr << "⚠ Vulkan Compute unavailable: " << e.what() << std::endl;
        g_compute_available = false;
        g_compute_engine.reset();
    }
    
    initialized.store(true, std::memory_order_release);
    return g_compute_available;
}

void Model::shutdownComputeEngine() {
    if (g_compute_engine) {
        g_compute_engine->cleanup();
        g_compute_engine.reset();
        g_compute_available = false;
    }
}

void Model::zeroGradients() {
    // Réinitialiser tous les gradients des layers à zéro
    for (auto& layer : layers) {
        std::fill(layer.grad_weights.begin(), layer.grad_weights.end(), 0.0f);
        std::fill(layer.grad_bias.begin(), layer.grad_bias.end(), 0.0f);
    }
    
    // Réinitialiser l'état du forward pour le prochain backward
    forward_state.clear();
}

Gradients Model::getGradients() const {
    Gradients grads;
    
    // Collecter tous les gradients des layers
    size_t param_idx = 0;
    for (const auto& layer : layers) {
        // Ajouter les gradients de poids
        for (const auto& grad : layer.grad_weights) {
            grads.param_grads[param_idx++] = grad;
        }
        
        // Ajouter les gradients de biais
        for (const auto& grad : layer.grad_bias) {
            grads.param_grads[param_idx++] = grad;
        }
    }
    
    return grads;
}

// === méthodes utilitaires simples (déjà présentes) ===
void Model::setDensity(double d) { densityFactor = (d > 0.0 ? d : 1.0); }
double Model::getDensity() const { return densityFactor; }

void Model::push(const std::string &name, const std::string &type, size_t params_count) {
    Layer layer(name, type, params_count);
    
    // Si des dimensions sont configurées dans modelConfig, les appliquer
    if (modelConfig.contains("in_channels")) {
        layer.in_channels = modelConfig["in_channels"];
    }
    if (modelConfig.contains("out_channels")) {
        layer.out_channels = modelConfig["out_channels"];
    }
    if (modelConfig.contains("height")) {
        layer.input_height = modelConfig["height"];
    }
    if (modelConfig.contains("width")) {
        layer.input_width = modelConfig["width"];
    }
    if (modelConfig.contains("kernel")) {
        layer.kernel_size = modelConfig["kernel"];
    }
    if (modelConfig.contains("stride")) {
        layer.stride = modelConfig["stride"];
    }
    if (modelConfig.contains("padding")) {
        layer.padding = modelConfig["padding"];
    }
    
    // Calculer les dimensions de sortie pour Conv2D
    if ((type == "Conv2d" || type == "ConvTranspose2d") && layer.kernel_size > 0) {
        if (type == "Conv2d") {
            layer.output_height = (layer.input_height + 2 * layer.padding - layer.kernel_size) / layer.stride + 1;
            layer.output_width = (layer.input_width + 2 * layer.padding - layer.kernel_size) / layer.stride + 1;
        } else { // ConvTranspose2d
            layer.output_height = (layer.input_height - 1) * layer.stride - 2 * layer.padding + layer.kernel_size;
            layer.output_width = (layer.input_width - 1) * layer.stride - 2 * layer.padding + layer.kernel_size;
        }
    }
    
    // Détecter automatiquement le type de branche basé sur le nom du layer
    layer.detectBranchType();
    
    layers.push_back(layer);
}

size_t Model::totalParamCount() const {
    size_t s = 0;
    for (const auto &L : layers) s += L.params_count;
    return s;
}

void Model::allocateParams() {
    size_t tot = totalParamCount();
    
    auto& allocator = DynamicTensorAllocator::instance();
    
    std::cout << "📦 Allocation de " << layers.size() << " blocs de poids (" << tot << " paramètres au total)..." << std::endl;
    
    // NOUVEAU: Allouer un tensor par layer au lieu d'un tensor par paramètre
    layer_weight_blocks.clear();
    layer_weight_blocks.resize(layers.size());
    
    for (size_t i = 0; i < layers.size(); ++i) {
        size_t layer_param_count = layers[i].params_count;
        
        if (layer_param_count > 0) {
            // ⚠️ CRITIQUE: Allocation dynamique via MemoryGuard (passe par DynamicTensorAllocator)
            // Le flag 'true' force l'allocation à passer par requestAllocation()
            layer_weight_blocks[i] = tensor(layer_param_count, true);
            
            // Lier le tensor au layer
            layers[i].weight_block = &layer_weight_blocks[i];
            
            std::cout << "  Layer " << i << " (" << layers[i].name << "): " 
                      << layer_param_count << " paramètres dans 1 tensor" << std::endl;
        }
    }
    
#ifdef MIMIR_ENABLE_LEGACY_PARAMS
    // ⚠️ ATTENTION: Cette structure legacy consomme énormément de RAM!
    // std::vector<tensor> avec des millions d'entrées = explosion mémoire
    // Cette allocation ne passe PAS par MemoryGuard
    // TODO: Supprimer complètement en production
    std::cout << "⚠️  LEGACY: Allocation structure params (" << tot << " tensors)..." << std::endl;
    params.clear();
    params.resize(tot);
    for (size_t i = 0; i < tot; ++i) {
        params[i].Weight = 0;
        params[i].Value = 0;
    }
    std::cout << "⚠️  Structure legacy allouée (désactiver avec -DMIMIR_ENABLE_LEGACY_PARAMS=OFF)" << std::endl;
#else
    // Structure legacy désactivée pour économiser la RAM
    params.clear();
#endif
    
    std::cout << "✓ " << layers.size() << " blocs de poids créés (1 tensor par layer)" << std::endl;
}

void Model::initializeWeights(const std::string &method, unsigned int seed) {
    if (layer_weight_blocks.empty()) {
        std::cerr << "⚠️  Cannot initialize weights: weight blocks not allocated" << std::endl;
        return;
    }
    
    auto& allocator = DynamicTensorAllocator::instance();
    std::mt19937 gen(seed == 0 ? std::random_device{}() : seed);
    
    std::cout << "🎲 Initializing weights using " << method << " method (bloc par layer)..." << std::endl;
    
    for (size_t layer_idx = 0; layer_idx < layers.size(); ++layer_idx) {
        const auto &layer = layers[layer_idx];
        
        if (layer.params_count == 0 || !layer.weight_block) continue;
        
        // Afficher progression tous les 10 layers
        if (layer_idx % 10 == 0) {
            std::cout << "  Initializing layer " << layer_idx << "/" << layers.size() 
                      << " (" << layer.name << ")..." << std::endl;
        }
        
        // Estimation fan_in/fan_out depuis params_count
        int fan_estimate = static_cast<int>(std::sqrt(static_cast<float>(layer.params_count)));
        int fan_in = std::max(fan_estimate, 32);
        int fan_out = std::max(fan_estimate, 32);
        
        float std_dev = 0.01f;
        
        if (method == "xavier" || method == "glorot") {
            std_dev = std::sqrt(2.0f / (fan_in + fan_out));
        }
        else if (method == "he" || method == "kaiming") {
            std_dev = 1.5f * std::sqrt(2.0f / fan_in);
        }
        else if (method == "normal") {
            std_dev = 0.05f;
        }
        
        std::normal_distribution<float> dist(0.0f, std_dev);
        
        // Estimer nombre de bias
        size_t num_weights = layer.params_count;
        size_t estimated_bias = fan_out;
        if (estimated_bias > num_weights / 10) {
            estimated_bias = num_weights / 10;
        }
        size_t num_pure_weights = num_weights - estimated_bias;
        
        // NOUVEAU: Initialiser directement le weight_block du layer
        float* weights_data = layer.weight_block->getData();
        
        for (size_t i = 0; i < num_weights; ++i) {
            float value;
            
            // Bias initialisé à 0, weights normalement
            if (i >= num_pure_weights) {
                value = 0.0f;  // Bias = 0
            } else {
                value = dist(gen);  // Weights ~ N(0, std²)
            }
            
            // Clip direct sans tanh (préserve magnitude)
            value = std::clamp(value, -3.0f, 3.0f);  // ±3σ capture 99.7%
            
            weights_data[i] = value;
        }
    }
    
    std::cout << "✓ Weights initialized (" << layers.size() << " layers, " << totalParamCount() << " parameters)" << std::endl;
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
    // NOUVEAU: Utiliser les weight_blocks au lieu de params
    if (layer_weight_blocks.empty()) return;
    
    // Utiliser le LR decay si configuré, sinon utiliser le learning_rate fourni
    float effective_lr = learning_rate;
    if (opt.decay_strategy != LRDecayStrategy::NONE) {
        effective_lr = opt.getCurrentLR();
    }
    
    opt.step += 1;
    
    // NOUVEAU: Appliquer l'optimiseur sur chaque weight_block du layer
    for (size_t layer_idx = 0; layer_idx < layers.size(); ++layer_idx) {
        auto &layer = layers[layer_idx];
        
        if (!layer.weight_block || layer.params_count == 0) continue;
        if (layer.grad_weights.empty()) continue;
        
        float* weights = layer.weight_block->getData();
        size_t weight_count = layer.getWeightsSize();
        
        // S'assurer que les états Adam ont la bonne taille
        opt.ensure(weight_count);
        
        switch (opt.type) {
            case OptimizerType::SGD: {
                // SGD simple
                for (size_t i = 0; i < weight_count && i < layer.grad_weights.size(); ++i) {
                    float grad = layer.grad_weights[i];
                    weights[i] -= effective_lr * grad;
                    weights[i] = std::clamp(weights[i], -3.0f, 3.0f);
                }
                break;
            }
            
            case OptimizerType::ADAM: {
                // Adam standard
                const float b1 = opt.beta1, b2 = opt.beta2;
                float bias_correction1 = 1.0f - std::pow(b1, static_cast<float>(opt.step));
                float bias_correction2 = 1.0f - std::pow(b2, static_cast<float>(opt.step));
                if (bias_correction1 <= 0.0f) bias_correction1 = 1e-8f;
                if (bias_correction2 <= 0.0f) bias_correction2 = 1e-8f;
                
                for (size_t i = 0; i < weight_count && i < layer.grad_weights.size(); ++i) {
                    float grad = layer.grad_weights[i];
                    
                    opt.m[i] = b1 * opt.m[i] + (1.0f - b1) * grad;
                    opt.v[i] = b2 * opt.v[i] + (1.0f - b2) * grad * grad;
                    
                    float m_hat = opt.m[i] / bias_correction1;
                    float v_hat = opt.v[i] / bias_correction2;
                    
                    float denom = std::sqrt(v_hat) + opt.eps;
                    weights[i] -= effective_lr * (m_hat / denom);
                    weights[i] = std::clamp(weights[i], -3.0f, 3.0f);
                }
                break;
            }
            
            case OptimizerType::ADAMW: {
                // AdamW avec weight decay découplé
                const float b1 = opt.beta1, b2 = opt.beta2;
                float bias_correction1 = 1.0f - std::pow(b1, static_cast<float>(opt.step));
                float bias_correction2 = 1.0f - std::pow(b2, static_cast<float>(opt.step));
                if (bias_correction1 <= 0.0f) bias_correction1 = 1e-8f;
                if (bias_correction2 <= 0.0f) bias_correction2 = 1e-8f;
                
                for (size_t i = 0; i < weight_count && i < layer.grad_weights.size(); ++i) {
                    float grad = layer.grad_weights[i];
                    float current = weights[i];
                    
                    opt.m[i] = b1 * opt.m[i] + (1.0f - b1) * grad;
                    opt.v[i] = b2 * opt.v[i] + (1.0f - b2) * grad * grad;
                    
                    float m_hat = opt.m[i] / bias_correction1;
                    float v_hat = opt.v[i] / bias_correction2;
                    
                    float denom = std::sqrt(v_hat) + opt.eps;
                    float weight_decay_term = opt.weight_decay * current;
                    float adam_update = effective_lr * (m_hat / denom);
                    
                    weights[i] = current - adam_update - effective_lr * weight_decay_term;
                    weights[i] = std::clamp(weights[i], -3.0f, 3.0f);
                }
                break;
            }
        }
        
        // Réinitialiser les gradients après application
        std::fill(layer.grad_weights.begin(), layer.grad_weights.end(), 0.0f);
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
                    c = ' '; // replace with space to avoid embedded newlines
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
    // Utilise directement l'implémentation optimisée de Conv::conv2d
    // qui gère automatiquement SIMD et CPU selon la compilation
    Conv::conv2d(input, output, params.weights, params.bias,
                in_h, in_w, in_c, out_c, params.kernel_size,
                params.stride, params.padding, params.dilation);
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

// ============================= 
// Branch Operations Implementation
// =============================

void Model::computeBranchMerge(const std::vector<float>& branch1, 
                               const std::vector<float>& branch2,
                               std::vector<float>& output,
                               MergeOperation merge_op,
                               bool use_hardware) {
    use_hardware = use_hardware && global_use_hardware && hasAVX2();
    
    size_t size = branch1.size();
    output.resize(size);
    
    switch (merge_op) {
        case MergeOperation::ADD: {
            if (use_hardware) {
                #ifdef __AVX2__
                size_t i = 0;
                for (; i + 8 <= size; i += 8) {
                    __m256 a = _mm256_loadu_ps(&branch1[i]);
                    __m256 b = _mm256_loadu_ps(&branch2[i]);
                    __m256 result = _mm256_add_ps(a, b);
                    _mm256_storeu_ps(&output[i], result);
                }
                // Éléments restants
                for (; i < size; ++i) {
                    output[i] = branch1[i] + branch2[i];
                }
                #else
                for (size_t i = 0; i < size; ++i) {
                    output[i] = branch1[i] + branch2[i];
                }
                #endif
            } else {
                for (size_t i = 0; i < size; ++i) {
                    output[i] = branch1[i] + branch2[i];
                }
            }
            break;
        }
        
        case MergeOperation::MULTIPLY: {
            if (use_hardware) {
                #ifdef __AVX2__
                size_t i = 0;
                for (; i + 8 <= size; i += 8) {
                    __m256 a = _mm256_loadu_ps(&branch1[i]);
                    __m256 b = _mm256_loadu_ps(&branch2[i]);
                    __m256 result = _mm256_mul_ps(a, b);
                    _mm256_storeu_ps(&output[i], result);
                }
                for (; i < size; ++i) {
                    output[i] = branch1[i] * branch2[i];
                }
                #else
                for (size_t i = 0; i < size; ++i) {
                    output[i] = branch1[i] * branch2[i];
                }
                #endif
            } else {
                for (size_t i = 0; i < size; ++i) {
                    output[i] = branch1[i] * branch2[i];
                }
            }
            break;
        }
        
        case MergeOperation::MAX: {
            if (use_hardware) {
                #ifdef __AVX2__
                size_t i = 0;
                for (; i + 8 <= size; i += 8) {
                    __m256 a = _mm256_loadu_ps(&branch1[i]);
                    __m256 b = _mm256_loadu_ps(&branch2[i]);
                    __m256 result = _mm256_max_ps(a, b);
                    _mm256_storeu_ps(&output[i], result);
                }
                for (; i < size; ++i) {
                    output[i] = std::max(branch1[i], branch2[i]);
                }
                #else
                for (size_t i = 0; i < size; ++i) {
                    output[i] = std::max(branch1[i], branch2[i]);
                }
                #endif
            } else {
                for (size_t i = 0; i < size; ++i) {
                    output[i] = std::max(branch1[i], branch2[i]);
                }
            }
            break;
        }
        
        case MergeOperation::AVERAGE: {
            if (use_hardware) {
                #ifdef __AVX2__
                __m256 half = _mm256_set1_ps(0.5f);
                size_t i = 0;
                for (; i + 8 <= size; i += 8) {
                    __m256 a = _mm256_loadu_ps(&branch1[i]);
                    __m256 b = _mm256_loadu_ps(&branch2[i]);
                    __m256 sum = _mm256_add_ps(a, b);
                    __m256 result = _mm256_mul_ps(sum, half);
                    _mm256_storeu_ps(&output[i], result);
                }
                for (; i < size; ++i) {
                    output[i] = (branch1[i] + branch2[i]) * 0.5f;
                }
                #else
                for (size_t i = 0; i < size; ++i) {
                    output[i] = (branch1[i] + branch2[i]) * 0.5f;
                }
                #endif
            } else {
                for (size_t i = 0; i < size; ++i) {
                    output[i] = (branch1[i] + branch2[i]) * 0.5f;
                }
            }
            break;
        }
        
        case MergeOperation::CONCATENATE: {
            // Concaténation simple
            output.resize(branch1.size() + branch2.size());
            std::copy(branch1.begin(), branch1.end(), output.begin());
            std::copy(branch2.begin(), branch2.end(), output.begin() + branch1.size());
            break;
        }
        
        default: {
            // Par défaut, addition
            std::copy(branch1.begin(), branch1.end(), output.begin());
            for (size_t i = 0; i < size; ++i) {
                output[i] += branch2[i];
            }
            break;
        }
    }
}

void Model::computeBranchSplit(const std::vector<float>& input,
                               std::vector<std::vector<float>>& outputs,
                               const std::vector<int>& split_sizes) {
    outputs.resize(split_sizes.size());
    size_t offset = 0;
    
    for (size_t i = 0; i < split_sizes.size(); ++i) {
        outputs[i].resize(split_sizes[i]);
        std::copy(input.begin() + offset, 
                  input.begin() + offset + split_sizes[i], 
                  outputs[i].begin());
        offset += split_sizes[i];
    }
}

void Model::detectAndSetupBranches() {
    // Parcourir tous les layers et détecter automatiquement les types de branches
    for (auto& layer : layers) {
        layer.detectBranchType();
    }
    
    // Analyser la structure pour identifier les connexions entre branches
    for (size_t i = 0; i < layers.size(); ++i) {
        auto& layer = layers[i];
        
        // Si c'est un layer résiduel, chercher le layer source
        if (layer.branch_type == BranchType::RESIDUAL) {
            // Par convention, le shortcut se connecte généralement plusieurs layers en arrière
            // Chercher un layer avec un nom similaire mais sans "shortcut" ou "residual"
            std::string base_name = layer.name;
            size_t pos = base_name.find("_shortcut");
            if (pos == std::string::npos) {
                pos = base_name.find("_residual");
            }
            
            if (pos != std::string::npos) {
                base_name = base_name.substr(0, pos);
                
                // Chercher le layer de base correspondant
                for (int j = static_cast<int>(i) - 1; j >= 0; --j) {
                    if (layers[j].name.find(base_name) != std::string::npos && 
                        j != static_cast<int>(i)) {
                        layer.branch_sources.push_back(j);
                        layers[j].is_branch_point = true;
                        break;
                    }
                }
            }
        }
    }
    
    std::cout << "✓ Détection des branches terminée. Trouvé:" << std::endl;
    for (size_t i = 0; i < layers.size(); ++i) {
        if (layers[i].requiresBranchComputation()) {
            std::cout << "  - Layer " << i << " (" << layers[i].name << "): ";
            if (layers[i].branch_type == BranchType::RESIDUAL) {
                std::cout << "RESIDUAL";
            } else if (layers[i].branch_type == BranchType::SKIP_CONNECTION) {
                std::cout << "SKIP_CONNECTION";
            } else if (layers[i].is_branch_point) {
                std::cout << "BRANCH_POINT";
            } else if (layers[i].is_merge_point) {
                std::cout << "MERGE_POINT";
            }
            std::cout << std::endl;
        }
    }
}

void Model::executeBranchComputation(int layer_idx, 
                                    std::vector<std::vector<float>>& layer_outputs,
                                    bool training) {
    if (layer_idx < 0 || layer_idx >= static_cast<int>(layers.size())) {
        return;
    }
    
    auto& layer = layers[layer_idx];
    
    if (!layer.requiresBranchComputation()) {
        return;
    }
    
    // Si c'est un point de fusion (residual, skip connection, etc.)
    if (layer.branch_type == BranchType::RESIDUAL && !layer.branch_sources.empty()) {
        // Récupérer la sortie du layer source
        int source_idx = layer.branch_sources[0];
        if (source_idx >= 0 && source_idx < static_cast<int>(layer_outputs.size())) {
            // Fusionner avec l'opération spécifiée
            std::vector<float> merged_output;
            computeBranchMerge(layer_outputs[layer_idx], 
                             layer_outputs[source_idx],
                             merged_output,
                             layer.merge_op,
                             true);
            layer_outputs[layer_idx] = std::move(merged_output);
        }
    }
    else if (layer.branch_type == BranchType::SPLIT) {
        // Pour les splits, on doit diviser la sortie
        // Ceci sera géré au niveau du forward pass principal
    }
}

void Model::backpropThroughBranch(int layer_idx,
                                 const std::vector<float>& grad_output,
                                 std::vector<std::vector<float>>& layer_gradients) {
    if (layer_idx < 0 || layer_idx >= static_cast<int>(layers.size())) {
        return;
    }
    
    auto& layer = layers[layer_idx];
    
    if (!layer.requiresBranchComputation()) {
        return;
    }
    
    // Backprop à travers les connexions de branche
    if (layer.branch_type == BranchType::RESIDUAL && !layer.branch_sources.empty()) {
        // Pour une connexion résiduelle, le gradient se propage vers les deux branches
        int source_idx = layer.branch_sources[0];
        if (source_idx >= 0 && source_idx < static_cast<int>(layer_gradients.size())) {
            // Le gradient du résidual se propage tel quel vers la branche source
            if (layer_gradients[source_idx].empty()) {
                layer_gradients[source_idx] = grad_output;
            } else {
                // Accumuler les gradients
                for (size_t i = 0; i < grad_output.size() && i < layer_gradients[source_idx].size(); ++i) {
                    layer_gradients[source_idx][i] += grad_output[i];
                }
            }
        }
    }
}

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
    
    if (training) {
        forward_state.layer_inputs.reserve(layers.size());
        forward_state.layer_outputs.reserve(layers.size());
        forward_state.activations.reserve(layers.size());
    }
    
    // Pré-allouer un buffer de sortie réutilisable
    std::vector<float> layer_output;
    layer_output.reserve(x.size());
    
    // Stocker les sorties de tous les layers pour gérer les branches
    std::vector<std::vector<float>> all_layer_outputs;
    if (training) {
        all_layer_outputs.reserve(layers.size());
    }
    
    // Forward pass à travers chaque layer
    for (size_t layer_idx = 0; layer_idx < layers.size(); ++layer_idx) {
        const auto &layer = layers[layer_idx];
        
        if (training) {
            forward_state.layer_inputs.push_back(x);
        }
        
        layer_output.clear(); // Réutiliser au lieu de réallouer
        
        // === DISPATCH HARDWARE ACCELERATED vs CPU ===
        bool use_gpu = g_compute_available && layer.params_count > 10000; // Seuil pour GPU
        
        // Traitement selon le type de layer
        if (layer.type == "Conv2d" || layer.type == "ConvTranspose2d") {
            // VRAIE convolution 2D avec kernel spatial
            const int kernel_size = layer.kernel_size > 0 ? layer.kernel_size : 3;
            const int in_channels = layer.in_channels > 0 ? layer.in_channels : 64;
            const int out_channels = layer.out_channels > 0 ? layer.out_channels : 64;
            const int height = layer.input_height > 0 ? layer.input_height : 64;
            const int width = layer.input_width > 0 ? layer.input_width : 64;
            const int stride = layer.stride > 0 ? layer.stride : 1;
            const int padding = layer.padding;
            
            // Calculer les dimensions de sortie
            int out_height, out_width;
            if (layer.type == "Conv2d") {
                out_height = (height + 2 * padding - kernel_size) / stride + 1;
                out_width = (width + 2 * padding - kernel_size) / stride + 1;
            } else { // ConvTranspose2d
                out_height = (height - 1) * stride - 2 * padding + kernel_size;
                out_width = (width - 1) * stride - 2 * padding + kernel_size;
            }
            
            const size_t output_size = out_channels * out_height * out_width;
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
                // NOUVEAU: Récupérer les poids depuis le weight_block du layer
                const float* layer_weights = layer.getWeights();
                
                // Convolution 2D complète (parallélisée - FORCÉE)
                #pragma omp parallel for schedule(dynamic)
                for (int oc = 0; oc < out_channels; ++oc) {
                    for (int oh = 0; oh < out_height; ++oh) {
                        for (int ow = 0; ow < out_width; ++ow) {
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
                                                w_idx < static_cast<int>(layer.getWeightsSize())) {
                                                float weight = layer_weights[w_idx];
                                                sum += x[in_idx] * weight;
                                            }
                                        }
                                    }
                                }
                            }
                            
                            int out_idx = oc * (out_height * out_width) + oh * out_width + ow;
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
            
            // NOUVEAU: Récupérer gamma depuis le weight_block
            const float* layer_weights = layer.getWeights();
            
            // Normaliser
            for (size_t i = 0; i < layer_output.size(); ++i) {
                layer_output[i] = (layer_output[i] - mean) / std;
                
                // Appliquer gamma et beta si disponibles
                if (i < layer.getWeightsSize()) {
                    float gamma = layer_weights[i];
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
        
        // Stocker la sortie de ce layer pour les branches
        if (training) {
            all_layer_outputs.push_back(x);
        }
        
        // Exécuter les calculs de branche si nécessaire
        if (layer.requiresBranchComputation() && training) {
            executeBranchComputation(layer_idx, all_layer_outputs, training);
            // Mettre à jour x avec la sortie fusionnée
            x = all_layer_outputs[layer_idx];
        }
        
        if (training) {
            forward_state.layer_outputs.push_back(layer_output);
            forward_state.activations.push_back(x);
        }
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
    
    if (layers.empty() || layer_weight_blocks.empty()) {
        return grads;
    }
    
    std::vector<float> grad = loss_gradient;
    
    // Backward pass à travers chaque layer (en ordre inverse)
    for (int layer_idx = layers.size() - 1; layer_idx >= 0; --layer_idx) {
        auto &layer = layers[layer_idx];
        
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
            
            // NOUVEAU: Récupérer les poids et gradients depuis le weight_block
            const float* layer_weights = layer.getWeights();
            if (layer.grad_weights.size() != layer.getWeightsSize()) {
                layer.grad_weights.resize(layer.getWeightsSize(), 0.0f);
            }
            
            // Backward Conv : VRAIS gradients avec convolution transposée
            int kernel_size = layer.kernel_size > 0 ? layer.kernel_size : 3;
            int in_channels = layer.in_channels > 0 ? layer.in_channels : 64;
            int out_channels = layer.out_channels > 0 ? layer.out_channels : 64;
            int height = layer.input_height > 0 ? layer.input_height : 64;
            int width = layer.input_width > 0 ? layer.input_width : 64;
            int stride = layer.stride > 0 ? layer.stride : 1;
            int padding = layer.padding;
            
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
                            if (w_idx < static_cast<int>(layer.grad_weights.size())) {
                                #pragma omp critical
                                layer.grad_weights[w_idx] += grad_weight;
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
                                            w_idx < static_cast<int>(layer.getWeightsSize())) {
                                            float weight = layer_weights[w_idx];
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
            // NOUVEAU: Récupérer les poids et gradients depuis le weight_block
            const float* layer_weights = layer.getWeights();
            if (layer.grad_weights.size() != layer.getWeightsSize()) {
                layer.grad_weights.resize(layer.getWeightsSize(), 0.0f);
            }
            
            // Backward BatchNorm avec formule compacte standard (CORRIGÉ)
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
                
                // Récupérer gamma (scale parameter) depuis weight_block
                float gamma = 1.0f;
                if (c * 2 < static_cast<int>(layer.getWeightsSize())) {
                    gamma = layer_weights[c * 2];
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
                if (c * 2 < static_cast<int>(layer.grad_weights.size())) {
                    #pragma omp critical
                    layer.grad_weights[c * 2] += grad_gamma;
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
                if (c * 2 + 1 < static_cast<int>(layer.grad_weights.size())) {
                    #pragma omp critical
                    layer.grad_weights[c * 2 + 1] += grad_beta;
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
            layer_obj["params_count"] = layer.params_count;
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
            Layer layer;
            layer.name = layer_obj.value("name", "");
            layer.type = layer_obj.value("type", "");
            layer.params_count = layer_obj.value("params_count", 0);
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
        
#ifdef MIMIR_ENABLE_LEGACY_PARAMS
        // ⚠️ LEGACY: Allocation potentiellement massive
        params.resize(num_tensors);
#else
        // Structure legacy désactivée
        params.clear();
#endif
        
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