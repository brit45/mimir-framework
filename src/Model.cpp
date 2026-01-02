#include "Model.hpp"
#include "HardwareOpt.hpp"
#include "SIMD_Ops.hpp"
#include "Layers.hpp"
#include "LayerTypes.hpp"
#include "LayerOps.hpp"
#include "MemoryGuard.hpp"
#include "DynamicTensorAllocator.hpp"
#include "VulkanCompute.hpp"
#include "RuntimeAllocator.hpp"
#include "LayerOpsExt.hpp"
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
#include <unordered_set>

// ============================================================================
// Registry centralisé (via LayerTypes.hpp)
// ============================================================================

using namespace LayerRegistry;

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
    : tokenizer(20000), encoder(64, 20000), hasTokenizer(true), hasEncoder(true),
      max_ram_mb_(4096)
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

// ============================================================================
// TENSOR STORE (Multi-input/Branch Support)
// ============================================================================

const std::vector<float>& Model::getTensor(const std::string& name) const {
    auto it = tensor_store.find(name);
    if (it == tensor_store.end()) {
        std::cerr << "❌ ERROR: Tensor '" << name << "' not found in TensorStore" << std::endl;
        std::cerr << "Available tensors: ";
        for (const auto& kv : tensor_store) {
            std::cerr << "'" << kv.first << "' ";
        }
        std::cerr << std::endl;
        throw std::runtime_error("Tensor not found: " + name);
    }
    return it->second;
}

std::vector<float>& Model::getTensorMutable(const std::string& name) {
    auto it = tensor_store.find(name);
    if (it == tensor_store.end()) {
        std::cerr << "❌ ERROR: Tensor '" << name << "' not found in TensorStore" << std::endl;
        std::cerr << "Available tensors: ";
        for (const auto& kv : tensor_store) {
            std::cerr << "'" << kv.first << "' ";
        }
        std::cerr << std::endl;
        throw std::runtime_error("Tensor not found: " + name);
    }
    return it->second;
}

void Model::storeTensor(const std::string& name, const std::vector<float>& data) {
    tensor_store[name] = data;
}

void Model::storeTensor(const std::string& name, std::vector<float>&& data) {
    tensor_store[name] = std::move(data);
}

std::vector<std::string> Model::getAvailableTensors() const {
    std::vector<std::string> names;
    names.reserve(tensor_store.size());
    for (const auto& kv : tensor_store) {
        names.push_back(kv.first);
    }
    return names;
}

void Model::clearTensorStore() {
    tensor_store.clear();
}

Layer* Model::getLayerByName(const std::string& name) {
    for (auto& layer : layers) {
        if (layer.name == name) {
            return &layer;
        }
    }
    return nullptr;  // Layer not found
}

// === méthodes utilitaires simples (déjà présentes) ===
void Model::setDensity(double d) { densityFactor = (d > 0.0 ? d : 1.0); }
double Model::getDensity() const { return densityFactor; }

void Model::push(const std::string &name, const std::string &type, size_t params_count) {
    // Normaliser le type et créer le layer avec enum
    std::string normalized_type = normalize_type(type);
    Layer layer(name, normalized_type, params_count);
    
    // Le constructeur Layer a déjà converti string -> enum
    // Vérifier que c'est supporté
    if (layer.type_enum == LayerType::UNKNOWN) {
        std::cerr << "❌ ERROR: Unknown layer type '" << type << "' (normalized: '" 
                  << normalized_type << "')" << std::endl;
        log_supported_types();
        throw std::runtime_error("Unknown layer type: " + type);
    }
    
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
    if ((normalized_type == "Conv2d" || normalized_type == "ConvTranspose2d") && layer.kernel_size > 0) {
        if (normalized_type == "Conv2d") {
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
        
        // fan_in/fan_out: utiliser les dimensions réelles quand disponibles
        int fan_in = 0;
        int fan_out = 0;

        if (layer.type_enum == LayerType::Linear && layer.in_features > 0 && layer.out_features > 0) {
            fan_in = layer.in_features;
            fan_out = layer.out_features;
        } else {
            // Estimation fan_in/fan_out depuis params_count
            int fan_estimate = static_cast<int>(std::sqrt(static_cast<float>(layer.params_count)));
            fan_in = std::max(fan_estimate, 32);
            fan_out = std::max(fan_estimate, 32);
        }
        
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
        
        // Déterminer précisément la zone bias quand possible
        const size_t num_weights = layer.params_count;
        size_t num_pure_weights = num_weights;

        if (layer.type_enum == LayerType::Linear && layer.in_features > 0 && layer.out_features > 0) {
            const size_t expected_w = static_cast<size_t>(layer.in_features) * static_cast<size_t>(layer.out_features);
            const size_t expected_b = layer.use_bias ? static_cast<size_t>(layer.out_features) : 0;
            if (expected_w + expected_b == num_weights) {
                num_pure_weights = expected_w;
            } else {
                // Fallback si le comptage ne correspond pas exactement
                size_t estimated_bias = std::min(expected_b, num_weights / 10);
                num_pure_weights = num_weights - estimated_bias;
            }
        } else {
            // Heuristique générique
            size_t estimated_bias = static_cast<size_t>(fan_out);
            if (estimated_bias > num_weights / 10) {
                estimated_bias = num_weights / 10;
            }
            num_pure_weights = num_weights - estimated_bias;
        }
        
        // NOUVEAU: Initialiser directement le weight_block du layer
        float* weights_data = layer.weight_block->getData();
        
        for (size_t i = 0; i < num_weights; ++i) {
            float value;
            
            // Bias initialisé à 0, weights ~ N(0,std²)
            value = (i >= num_pure_weights) ? 0.0f : dist(gen);
            
            // Clip direct sans tanh (préserve magnitude)
            value = std::clamp(value, -3.0f, 3.0f);  // ±3σ capture 99.7%
            
            weights_data[i] = value;
        }
    }
    
    std::cout << "✓ Weights initialized (" << layers.size() << " layers, " << totalParamCount() << " parameters)" << std::endl;
}

void Model::updateWeightsWithNoise(float learning_rate, float noise_std) {
    // NOTE: Fonction obsolète utilisant l'ancienne structure params
    std::cerr << "⚠️ updateWeightsWithNoise() est obsolète" << std::endl;
}

std::vector<uint16_t> Model::getWeights() const {
    // NOTE: Fonction obsolète utilisant l'ancienne structure params
    return std::vector<uint16_t>();
}

void Model::setTokenizer(const Tokenizer &t) { tokenizer = t; hasTokenizer = true; }
void Model::setEncoder(const Encoder &e) { encoder = e; hasEncoder = true; }

void Model::forward(std::vector<uint8_t> &out_uint8) const {
    // NOTE: Fonction obsolète utilisant l'ancienne structure params
    // Utilisez forwardPass() à la place
    out_uint8.clear();
}

void Model::setOutputTarget(const std::vector<uint8_t> &target) {
    // NOTE: Fonction obsolète utilisant l'ancienne structure params
}

void Model::applyParamUpdate(float learning_rate) {
    // NOTE: Fonction obsolète - utilisez optimizerStep() pour l'entraînement moderne avec layer_weight_blocks
    std::cerr << "[DEPRECATED] applyParamUpdate() est obsolète. Utilisez optimizerStep() à la place.\n";
    return;
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
    // NOTE: Cette fonction est obsolète et a été remplacée par le module Serialization
    // Utilisez maintenant checkpoint.save() depuis Lua ou Mimir::Serialization::save_checkpoint() depuis C++
    std::cerr << "⚠️ Model::saveCheckpoint() est obsolète! Utilisez Mimir::Serialization::save_checkpoint()" << std::endl;
    std::cerr << "   Depuis Lua: checkpoint.save(model, path, {format='safetensors'})" << std::endl;
    return false;
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
                
                // NOTE: Anciennes méthodes de sauvegarde/chargement supprimées
                // Utiliser maintenant le module Serialization avec checkpoint.load()
                /*
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
                */
                
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
    // Vérifications préliminaires
    if (layers.empty()) {
        std::cerr << "⚠️  Cannot perform forward pass: no layers defined" << std::endl;
        return input;
    }
    
    if (layer_weight_blocks.empty()) {
        std::cerr << "⚠️  Cannot perform forward pass: weights not allocated" << std::endl;
        std::cerr << "    Call allocate_params() and init_weights() first" << std::endl;
        return input;
    }
    
    // ========================================================================
    // INITIALIZATION: TensorStore + Validation
    // ========================================================================
    
    // Clear et initialiser TensorStore avec l'input principal
    clearTensorStore();
    storeTensor("x", input);
    
    // VALIDATION: Vérifier que tous les layers sont supportés (une seule fois)
    static bool validated = false;
    if (!validated) {
        for (size_t i = 0; i < layers.size(); ++i) {
            if (layers[i].type_enum == LayerType::UNKNOWN) {
                std::cerr << "❌ ERROR: Unsupported layer type '" << layers[i].type 
                          << "' at index " << i << " (" << layers[i].name << ")" << std::endl;
                log_supported_types();
                throw std::runtime_error("Unsupported layer type: " + layers[i].type);
            }
        }
        validated = true;
        std::cerr << "✓ All " << layers.size() << " layers validated" << std::endl;
    }
    
    // État du forward
    if (training) {
        forward_state.clear();
        forward_state.is_valid = true;
        forward_state.layer_inputs.reserve(layers.size());
        forward_state.layer_outputs.reserve(layers.size());
        forward_state.activations.reserve(layers.size());
    }
    
    // Conservation pour backward (à migrer vers TensorStore)
    std::vector<std::vector<float>> all_layer_outputs;
    if (training) {
        all_layer_outputs.reserve(layers.size());
    }
    
    // ========================================================================
    // FORWARD PASS: Routing via TensorStore
    // ========================================================================
    
    for (size_t layer_idx = 0; layer_idx < layers.size(); ++layer_idx) {
        const auto &layer = layers[layer_idx];
        
        // ====================================================================
        // RETRIEVE INPUTS (multi-input support)
        // ====================================================================
        
        std::vector<std::string> input_names = layer.inputs;
        if (input_names.empty()) {
            input_names = {"x"};  // Default: lire "x"
        }
        
        std::vector<const std::vector<float>*> inputs;
        inputs.reserve(input_names.size());
        
        for (const auto& name : input_names) {
            try {
                inputs.push_back(&getTensor(name));
            } catch (const std::exception& e) {
                std::cerr << "❌ ERROR in layer " << layer_idx << " (" << layer.name 
                          << "): Cannot find input tensor '" << name << "'" << std::endl;
                std::cerr << "Available tensors: ";
                auto available = getAvailableTensors();
                for (const auto& t : available) std::cerr << "'" << t << "' ";
                std::cerr << std::endl;
                throw;
            }
        }
        
        // Pour compatibilité: x est toujours inputs[0]
        const std::vector<float>& x = *inputs[0];
        
        if (training) {
            forward_state.layer_inputs.push_back(x);
        }
        
        std::vector<float> layer_output;
        
        // ====================================================================
        // STRICT MODE: RuntimeAllocator
        // ====================================================================
        RuntimeAllocator allocator(MemoryGuard::instance(), max_ram_mb_);
        
        // ====================================================================
        // DISPATCH PRINCIPAL VIA SWITCH/CASE SUR LayerType (MODE STRICT)
        // ====================================================================
        
        try {
            switch (layer.type_enum) {
    
    // ====================================================================
    // CONVOLUTION
    // ====================================================================
    
    case LayerType::Conv2d:
    case LayerType::ConvTranspose2d: {
        RUNTIME_CHECK(
            layer.in_channels > 0 && layer.out_channels > 0,
            "Conv2d: in_channels and out_channels must be set"
        );
        RUNTIME_CHECK(
            layer.get_kernel_h() > 0,
            "Conv2d: kernel_size must be set"
        );
        
        const int kernel_size = layer.get_kernel_h();
        const int in_channels = layer.in_channels;
        const int out_channels = layer.out_channels;
        const int height = layer.input_height > 0 ? layer.input_height : 64;
        const int width = layer.input_width > 0 ? layer.input_width : 64;
        const int stride = layer.get_stride_h();
        const int padding = layer.get_pad_h();
        
        int out_height, out_width;
        if (layer.type_enum == LayerType::Conv2d) {
            out_height = (height + 2 * padding - kernel_size) / stride + 1;
            out_width = (width + 2 * padding - kernel_size) / stride + 1;
        } else {
            out_height = (height - 1) * stride - 2 * padding + kernel_size;
            out_width = (width - 1) * stride - 2 * padding + kernel_size;
        }
        
        // ✅ Allocation gérée
        auto output_handle = allocator.allocate_tensor(
            {out_channels, out_height, out_width},
            "float32",
            layer.name + "_output"
        );
        layer_output = output_handle.data();
        
        const float* layer_weights = layer.getWeights();
        
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
                                        sum += x[in_idx] * layer_weights[w_idx];
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
        
        // ReLU activation if specified
        if (layer.activation != ActivationType::NONE) {
            for (auto &val : layer_output) {
                val = std::max(0.0f, val);
            }
        }
        
        break;
    }
    
    case LayerType::Conv1d: {
        layer_output = LayerOpsExt::conv1d_forward(x, layer);
        break;
    }
    
    case LayerType::DepthwiseConv2d: {
        layer_output = LayerOpsExt::depthwise_conv2d_forward(x, layer);
        break;
    }
    
    // ====================================================================
    // LINEAR
    // ====================================================================
    
    case LayerType::Linear: {
        layer_output = LayerOps::linear_forward(x, layer, training);
        break;
    }
    
    case LayerType::Bilinear: {
        RUNTIME_ERROR_STRICT(
            "Bilinear layer not implemented yet. "
            "Remove from model or implement in LayerOpsExt."
        );
        break;
    }
    
    // ====================================================================
    // EMBEDDING
    // ====================================================================
    
    case LayerType::Embedding: {
        RUNTIME_ERROR_STRICT(
            "Embedding layer requires integer token input (not float vector). "
            "Current API is float-only. Use external embedding lookup or "
            "implement int-based forward API."
        );
        break;
    }
    
    case LayerType::EmbeddingBag: {
        RUNTIME_ERROR_STRICT(
            "EmbeddingBag layer not implemented. "
            "Remove from model or implement in LayerOpsExt."
        );
        break;
    }
    
    // ====================================================================
    // NORMALIZATION
    // ====================================================================
    
    case LayerType::BatchNorm2d:
    case LayerType::BatchNorm1d: {
        auto output_handle = allocator.allocate_tensor(
            {static_cast<int>(x.size())},
            "float32",
            layer.name + "_output"
        );
        std::vector<float>& layer_output = output_handle.data();
        layer_output = x;
        
        float mean = 0.0f;
        for (float val : x) mean += val;
        mean /= x.size();
        
        float var = 0.0f;
        for (float val : x) {
            float diff = val - mean;
            var += diff * diff;
        }
        var /= x.size();
        float std = std::sqrt(var + layer.eps);
        
        const float* layer_weights = layer.getWeights();
        
        for (size_t i = 0; i < layer_output.size(); ++i) {
            layer_output[i] = (layer_output[i] - mean) / std;
            if (layer.affine && i < layer.getWeightsSize()) {
                layer_output[i] *= layer_weights[i];
            }
        }
        
        break;
    }
    
    case LayerType::LayerNorm: {
        layer_output = LayerOps::layernorm_forward(x, layer, training);
        break;
    }
    
    case LayerType::GroupNorm: {
        layer_output = LayerOps::groupnorm_forward(x, layer, training);
        break;
    }
    
    case LayerType::InstanceNorm2d: {
        layer_output = LayerOpsExt::instance_norm2d_forward(x, layer);
        break;
    }
    
    case LayerType::RMSNorm: {
        layer_output = LayerOpsExt::rms_norm_forward(x, layer);
        break;
    }
    
    // ====================================================================
    // ACTIVATION
    // ====================================================================
    
    case LayerType::ReLU: {
        layer_output = LayerOps::relu_forward(x);
        break;
    }
    
    case LayerType::LeakyReLU: {
        float alpha = layer.leaky_relu_alpha > 0 ? layer.leaky_relu_alpha : 0.01f;
        layer_output = LayerOpsExt::leaky_relu_forward(x, alpha);
        break;
    }
    
    case LayerType::GELU: {
        layer_output = LayerOps::gelu_forward(x);
        break;
    }
    
    case LayerType::SiLU: {
        layer_output = LayerOps::silu_forward(x);
        break;
    }
    
    case LayerType::Tanh: {
        layer_output = LayerOps::tanh_forward(x);
        break;
    }
    
    case LayerType::Sigmoid: {
        layer_output = LayerOps::sigmoid_forward(x);
        break;
    }
    
    case LayerType::Softmax:
    case LayerType::LogSoftmax: {
        layer_output = LayerOps::softmax_forward(x, layer);
        break;
    }
    
    case LayerType::Softplus: {
        layer_output = LayerOpsExt::softplus_forward(x);
        break;
    }
    
    case LayerType::Mish: {
        layer_output = LayerOpsExt::mish_forward(x);
        break;
    }
    
    case LayerType::HardSigmoid: {
        layer_output = LayerOpsExt::hard_sigmoid_forward(x);
        break;
    }
    
    case LayerType::HardSwish: {
        layer_output = LayerOpsExt::hard_swish_forward(x);
        break;
    }
    
    // ====================================================================
    // POOLING
    // ====================================================================
    
    case LayerType::MaxPool2d: {
        RUNTIME_CHECK(
            layer.get_kernel_h() > 0,
            "MaxPool2d: kernel_size must be set"
        );
        
        const int kernel_size = layer.get_kernel_h();
        const int in_channels = layer.in_channels > 0 ? layer.in_channels : 64;
        const int height = layer.input_height > 0 ? layer.input_height : 64;
        const int width = layer.input_width > 0 ? layer.input_width : 64;
        const int stride = layer.get_stride_h();
        const int padding = layer.get_pad_h();
        
        const int out_height = (height + 2 * padding - kernel_size) / stride + 1;
        const int out_width = (width + 2 * padding - kernel_size) / stride + 1;
        
        auto output_handle = allocator.allocate_tensor(
            {in_channels, out_height, out_width},
            "float32",
            layer.name + "_output"
        );
        std::vector<float>& layer_output = output_handle.data();
        std::fill(layer_output.begin(), layer_output.end(), 
                  -std::numeric_limits<float>::infinity());
        
        #pragma omp parallel for schedule(static)
        for (int c = 0; c < in_channels; ++c) {
            for (int oh = 0; oh < out_height; ++oh) {
                for (int ow = 0; ow < out_width; ++ow) {
                    float max_val = -std::numeric_limits<float>::infinity();
                    
                    for (int kh = 0; kh < kernel_size; ++kh) {
                        for (int kw = 0; kw < kernel_size; ++kw) {
                            int ih = oh * stride + kh - padding;
                            int iw = ow * stride + kw - padding;
                            
                            if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                                int in_idx = c * (height * width) + ih * width + iw;
                                if (in_idx < static_cast<int>(x.size())) {
                                    max_val = std::max(max_val, x[in_idx]);
                                }
                            }
                        }
                    }
                    
                    int out_idx = c * (out_height * out_width) + oh * out_width + ow;
                    layer_output[out_idx] = max_val;
                }
            }
        }
        

    }
    
    case LayerType::AvgPool2d: {
        layer_output = LayerOps::avgpool2d_forward(x, layer);
        break;
    }
    
    case LayerType::AvgPool1d: {
        layer_output = LayerOpsExt::avgpool1d_forward(x, layer);
        break;
    }
    
    case LayerType::GlobalAvgPool2d:
    case LayerType::AdaptiveAvgPool2d: {
        layer_output = LayerOps::global_avgpool2d_forward(x, layer);
        break;
    }
    
    // ====================================================================
    // DROPOUT
    // ====================================================================
    
    case LayerType::Dropout:
    case LayerType::Dropout2d: {
        layer_output = LayerOps::dropout_forward(x, layer, training);
        break;
    }
    
    case LayerType::AlphaDropout: {
        RUNTIME_ERROR_STRICT(
            "AlphaDropout not implemented. Use standard Dropout or implement."
        );
        break;
    }
    
    // ====================================================================
    // SHAPE OPERATIONS
    // ====================================================================
    
    case LayerType::Flatten: {
        layer_output = LayerOps::flatten_forward(x, layer);
        break;
    }
    
    case LayerType::Reshape:
    case LayerType::View: {
        layer_output = LayerOps::reshape_forward(x, layer);
        break;
    }
    
    case LayerType::Transpose: {
        RUNTIME_CHECK(
            layer.in_features > 0 && layer.out_features > 0,
            "Transpose: in_features and out_features must be set"
        );
        
        layer_output = LayerOps::transpose_forward(
            x, layer.in_features, layer.out_features
        );
        break;
    }
    
    case LayerType::Permute: {
        RUNTIME_CHECK(
            !layer.permute_dims.empty(),
            "Permute: permute_dims must be configured"
        );
        
        std::vector<int> shape = layer.shape;
        if (shape.empty()) {
            shape = {1, static_cast<int>(x.size())};
        }
        
        layer_output = LayerOps::permute_forward(x, layer.permute_dims, shape);
        break;
    }
    
    case LayerType::Squeeze: {
        std::vector<int> input_shape = {static_cast<int>(x.size())};
        std::vector<int> output_shape;
        layer_output = LayerOpsExt::squeeze_forward(
            x, input_shape, output_shape, layer.squeeze_dim
        );
        break;
    }
    
    case LayerType::Unsqueeze: {
        std::vector<int> input_shape = {static_cast<int>(x.size())};
        std::vector<int> output_shape;
        RUNTIME_CHECK(
            layer.unsqueeze_dim >= -10 && layer.unsqueeze_dim < 10,
            "Unsqueeze: unsqueeze_dim must be set (valid range: -10 to 10)"
        );
        layer_output = LayerOpsExt::unsqueeze_forward(
            x, input_shape, output_shape, layer.unsqueeze_dim
        );
        break;
    }
    
    case LayerType::Identity: {
        layer_output = LayerOps::identity_forward(x);
        break;
    }
    
    case LayerType::Lambda: {
        RUNTIME_ERROR_STRICT(
            "Lambda layer with Lua callbacks is unsafe in strict mode. "
            "Use C++ layer implementations instead."
        );
        break;
    }
    
    // ====================================================================
    // ELEMENT-WISE OPERATIONS
    // ====================================================================
    
    case LayerType::Add: {
        RUNTIME_CHECK(
            inputs.size() >= 2,
            "Add layer requires 2 inputs, got " + std::to_string(inputs.size())
        );
        layer_output = LayerOps::add_forward(*inputs[0], *inputs[1]);
        break;
    }
    
    case LayerType::Subtract: {
        RUNTIME_CHECK(
            inputs.size() >= 2,
            "Subtract layer requires 2 inputs, got " + std::to_string(inputs.size())
        );
        layer_output = LayerOpsExt::subtract_forward(*inputs[0], *inputs[1]);
        break;
    }
    
    case LayerType::Multiply: {
        RUNTIME_CHECK(
            inputs.size() >= 2,
            "Multiply layer requires 2 inputs, got " + std::to_string(inputs.size())
        );
        layer_output = LayerOps::multiply_forward(*inputs[0], *inputs[1]);
        break;
    }
    
    case LayerType::Divide: {
        RUNTIME_CHECK(
            inputs.size() >= 2,
            "Divide layer requires 2 inputs, got " + std::to_string(inputs.size())
        );
        layer_output = LayerOpsExt::divide_forward(*inputs[0], *inputs[1]);
        break;
    }
    
    // ====================================================================
    // TENSOR OPERATIONS
    // ====================================================================
    
    case LayerType::Concat: {
        RUNTIME_CHECK(
            inputs.size() >= 2,
            "Concat requires at least 2 inputs, got " + std::to_string(inputs.size())
        );
        
        std::vector<std::vector<float>> inputs_vec;
        inputs_vec.reserve(inputs.size());
        for (const auto* inp : inputs) {
            inputs_vec.push_back(*inp);
        }
        layer_output = LayerOps::concat_forward(inputs_vec, layer.concat_axis);
        break;
    }
    
    case LayerType::Split: {
        RUNTIME_CHECK(
            !layer.split_sizes.empty(),
            "Split: split_sizes must be configured"
        );
        
        std::vector<std::vector<float>> splits = LayerOps::split_forward(
            x, layer.split_sizes, layer.split_axis
        );
        
        std::string output_base = layer.output.empty() ? "x" : layer.output;
        for (size_t i = 0; i < splits.size(); ++i) {
            std::string split_name = output_base + "_" + std::to_string(i);
            storeTensor(split_name, std::move(splits[i]));
        }
        
        layer_output = getTensor(output_base + "_0");
        break;
    }
    
    case LayerType::Chunk: {
        RUNTIME_CHECK(
            layer.num_chunks > 0,
            "Chunk: num_chunks must be set"
        );
        
        auto chunks = LayerOpsExt::chunk_forward(x, layer.num_chunks, layer.split_axis);
        
        std::string output_base = layer.output.empty() ? "x" : layer.output;
        for (size_t i = 0; i < chunks.size(); ++i) {
            storeTensor(output_base + "_" + std::to_string(i), std::move(chunks[i]));
        }
        
        layer_output = getTensor(output_base + "_0");
        break;
    }
    
    case LayerType::Stack: {
        RUNTIME_CHECK(
            inputs.size() >= 2,
            "Stack requires at least 2 inputs, got " + std::to_string(inputs.size())
        );
        
        std::vector<std::vector<float>> inputs_vec;
        for (const auto* inp : inputs) inputs_vec.push_back(*inp);
        
        layer_output = LayerOpsExt::stack_forward(inputs_vec, layer.stack_axis);
        break;
    }
    
    case LayerType::MatMul: {
        RUNTIME_CHECK(
            inputs.size() >= 2,
            "MatMul requires 2 matrix inputs, got " + std::to_string(inputs.size())
        );
        RUNTIME_CHECK(
            layer.in_features > 0 && layer.out_features > 0 && layer.embed_dim > 0,
            "MatMul: dimensions (in_features, out_features, embed_dim) must be configured"
        );
        
        int M = layer.in_features;
        int K = layer.out_features;
        int N = layer.embed_dim;
        
        layer_output = LayerOps::matmul_forward(*inputs[0], *inputs[1], M, K, N);
        break;
    }
    
    case LayerType::BatchMatMul: {
        RUNTIME_ERROR_STRICT(
            "BatchMatMul not implemented. Use MatMul with batching or implement."
        );
        break;
    }
    
    // ====================================================================
    // ATTENTION
    // ====================================================================
    
    case LayerType::SelfAttention:
    case LayerType::MultiHeadAttention: {
        RUNTIME_CHECK(
            layer.getWeights() != nullptr,
            "Attention: weights not initialized. Call allocateParams() first."
        );
        
        int seq_len = layer.seq_len > 0 ? layer.seq_len : 1;
        int embed_dim = layer.embed_dim > 0 ? layer.embed_dim : x.size();
        int num_heads = layer.num_heads > 0 ? layer.num_heads : 1;
        bool causal = layer.causal;
        
        const float* weights = layer.getWeights();
        int qkv_size = embed_dim * embed_dim * 3;
        int out_size = embed_dim * embed_dim;
        
        std::vector<float> qkv_weight(weights, weights + qkv_size);
        std::vector<float> out_weight(weights + qkv_size, weights + qkv_size + out_size);
        
        if (layer.type_enum == LayerType::SelfAttention) {
            layer_output = LayerOps::self_attention_forward(
                x, qkv_weight, out_weight, seq_len, embed_dim, num_heads, causal
            );
        } else {
            layer_output = LayerOps::multihead_attention_forward(
                x, qkv_weight, out_weight, seq_len, embed_dim, num_heads, causal
            );
        }
        break;
    }
    
    case LayerType::CrossAttention: {
        RUNTIME_CHECK(
            inputs.size() >= 2,
            "CrossAttention requires 2 inputs (query, key_value), got " +
            std::to_string(inputs.size())
        );
        
        RUNTIME_ERROR_STRICT(
            "CrossAttention: Full implementation pending. "
            "Need separate Q/K/V projections for cross-attention."
        );
        break;
    }
    
    // ====================================================================
    // UPSAMPLING
    // ====================================================================
    
    case LayerType::UpsampleNearest: {
        RUNTIME_CHECK(
            layer.out_h > 0 && layer.out_w > 0 && layer.in_channels > 0,
            "UpsampleNearest: dimensions (out_h, out_w, in_channels) must be set"
        );
        
        int in_h = layer.out_h;
        int in_w = layer.out_w;
        int channels = layer.in_channels;
        int scale_h = layer.scale_h > 0 ? layer.scale_h : 2;
        int scale_w = layer.scale_w > 0 ? layer.scale_w : 2;
        
        layer_output = LayerOps::upsample_nearest_forward(
            x, in_h, in_w, channels, scale_h, scale_w
        );
        break;
    }
    
    case LayerType::UpsampleBilinear: {
        RUNTIME_CHECK(
            layer.out_h > 0 && layer.out_w > 0 && layer.in_channels > 0,
            "UpsampleBilinear: dimensions must be set"
        );
        
        int in_h = layer.out_h;
        int in_w = layer.out_w;
        int channels = layer.in_channels;
        int out_h = in_h * 2;
        int out_w = in_w * 2;
        
        layer_output = LayerOps::upsample_bilinear_forward(
            x, in_h, in_w, channels, out_h, out_w
        );
        break;
    }
    
    case LayerType::UpsampleBicubic: {
        RUNTIME_CHECK(
            layer.out_h > 0 && layer.out_w > 0,
            "UpsampleBicubic: dimensions must be set"
        );
        
        int in_h = layer.out_h;
        int in_w = layer.out_w;
        int channels = layer.in_channels > 0 ? layer.in_channels : 3;
        int out_h = in_h * 2;
        int out_w = in_w * 2;
        
        layer_output = LayerOpsExt::upsample_bicubic_forward(
            x, in_h, in_w, channels, out_h, out_w
        );
        break;
    }
    
    case LayerType::PixelShuffle: {
        layer_output = LayerOpsExt::pixel_shuffle_forward(x, layer);
        break;
    }
    
    // ====================================================================
    // PADDING
    // ====================================================================
    
    case LayerType::ZeroPad2d: {
        layer_output = LayerOpsExt::zero_pad2d_forward(x, layer);
        break;
    }
    
    case LayerType::ReflectionPad2d: {
        layer_output = LayerOpsExt::reflection_pad2d_forward(x, layer);
        break;
    }
    
    case LayerType::ReplicationPad2d: {
        layer_output = LayerOpsExt::replication_pad2d_forward(x, layer);
        break;
    }
    
    // ====================================================================
    // RECURRENT (Hors scope - À SUPPRIMER de l'enum si non implémenté)
    // ====================================================================
    
    case LayerType::LSTM:
    case LayerType::GRU:
    case LayerType::RNN: {
        RUNTIME_ERROR_STRICT(
            "Recurrent layers (LSTM/GRU/RNN) not implemented. "
            "These are complex and require multi-timestep state handling. "
            "Decision: implement in v2.4.0 OR remove from LayerType enum."
        );
        break;
    }
    
    // ====================================================================
    // UNKNOWN / DEFAULT
    // ====================================================================
    
    case LayerType::UNKNOWN:
    default: {
        throw std::runtime_error(
            "Layer '" + layer.name + "' type '" + 
            type_to_string(layer.type_enum) + "' is UNKNOWN. " +
            "This should never happen in strict mode. Check LayerType registry."
        );
        break;
    }
}

// ============================================================================
// FIN DU SWITCH-CASE
// ============================================================================

        
        } catch (const std::exception& e) {
            std::cerr << "❌ ERROR in layer " << layer_idx << " (" << layer.name 
                      << ", type: " << type_to_string(layer.type_enum) << "): " 
                      << e.what() << std::endl;
            throw;
        }
        
        // ====================================================================
        // STORE OUTPUT (multi-output support)
        // ====================================================================
        
        std::string output_name = layer.output.empty() ? "x" : layer.output;
        storeTensor(output_name, std::move(layer_output));
        
        // Pour backward: conserver aussi dans all_layer_outputs
        if (training) {
            all_layer_outputs.push_back(getTensor(output_name));
            forward_state.layer_outputs.push_back(getTensor(output_name));
            forward_state.activations.push_back(getTensor(output_name));
        }
        
        // Gestion des branches
        if (layer.requiresBranchComputation() && training) {
            executeBranchComputation(layer_idx, all_layer_outputs, training);
            // Update tensor store avec le résultat mergé
            storeTensor(output_name, all_layer_outputs[layer_idx]);
        }
    }
    
    // Le résultat final est toujours dans "x" (ou dernier output)
    if (training) {
        forward_state.final_output = getTensor("x");
    }
    
    return getTensor("x");
}

Gradients Model::backwardPass(const std::vector<float> &loss_gradient) {
    Gradients grads;
    
    if (!forward_state.is_valid) {
        std::cerr << "⚠️  Cannot perform backward pass: no valid forward state" << std::endl;
        std::cerr << "    Call forwardPass() in training mode first" << std::endl;
        return grads;
    }
    
    if (layers.empty() || layer_weight_blocks.empty()) {
        std::cerr << "⚠️  Cannot perform backward pass: layers or weights not initialized" << std::endl;
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
        std::cout << "  ✓ " << layer_weight_blocks.size() << " blocs de poids alloués" << std::endl;
        
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

// --- Fin ---

// --- Fin ---
