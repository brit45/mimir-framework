#include "tensors.hpp"
#include "DynamicTensorAllocator.hpp"
#include "runtimes/RuntimeRouter.hpp"
#include "include/json.hpp"

#include <cstring>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>

using json = nlohmann::json;

// ============================================================================
// Implémentations tensor avec allocation dynamique
// ============================================================================

tensor::tensor(size_t size, bool dynamic) : use_dynamic_alloc(dynamic) {
    if (dynamic) {
        // 🔑 ALLOCATION DYNAMIQUE LAZY
        // Crée le handle mais n'alloue pas immédiatement la mémoire
        // La mémoire sera allouée lors du premier getData() (lazy loading)
        // Cela permet de ne pas comptabiliser la RAM avant utilisation réelle
        auto& allocator = DynamicTensorAllocator::instance();
        dynamic_handle = allocator.allocateTensor(size, "tensor_data");
        
        if (!dynamic_handle) {
            // 🛑 PANIC OOM: Impossible d'allouer même le handle
            std::cerr << "\n\u274c\u274c\u274c PANIC: OUT OF MEMORY \u274c\u274c\u274c" << std::endl;
            std::cerr << "⛔ Impossible d'allouer tensor de " << (size * sizeof(float) / 1024 / 1024) << " MB" << std::endl;
            std::cerr << "⛔ MemoryGuard a refusé l'allocation - limite atteinte" << std::endl;
            std::cerr << "\n🚨 ARRÊT CONTRÔLÉ POUR ÉVITER CRASH OS\n" << std::endl;
            
            // Afficher stats avant de quitter
            auto& guard = MemoryGuard::instance();
            guard.printStats();
            
            // Arrêt contrôlé (pas de throw pour éviter corruption)
            std::cerr << "\n⚠️  Le programme va s'arrêter proprement pour protéger l'OS..." << std::endl;
            std::exit(1);
        }
    } else {
        // Allocation classique std::vector (non contrôlée par MemoryGuard)
        // ⚠️ Utiliser dynamic=true en production pour contrôle strict!
        data.resize(size, 0.0f);
    }
}

tensor::tensor(tensor&& other) noexcept
    : Weight(other.Weight)
    , Pos(other.Pos)
    , Value(other.Value)
    , Length(other.Length)
    , data(std::move(other.data))
    , dynamic_handle(other.dynamic_handle)
    , use_dynamic_alloc(other.use_dynamic_alloc)
{
    other.dynamic_handle = nullptr;
    other.use_dynamic_alloc = false;
}

tensor& tensor::operator=(tensor&& other) noexcept {
    if (this != &other) {
        // Libérer les ressources existantes
        if (use_dynamic_alloc && dynamic_handle) {
            auto& allocator = DynamicTensorAllocator::instance();
            allocator.freeTensor(
                static_cast<DynamicTensorAllocator::TensorHandle*>(dynamic_handle));
        }
        
        // Transférer les ressources
        Weight = other.Weight;
        Pos = other.Pos;
        Value = other.Value;
        Length = other.Length;
        data = std::move(other.data);
        dynamic_handle = other.dynamic_handle;
        use_dynamic_alloc = other.use_dynamic_alloc;
        
        // Invalider l'autre
        other.dynamic_handle = nullptr;
        other.use_dynamic_alloc = false;
    }
    return *this;
}


tensor::~tensor() {
    if (use_dynamic_alloc && dynamic_handle) {
        auto& allocator = DynamicTensorAllocator::instance();
        allocator.freeTensor(
            static_cast<DynamicTensorAllocator::TensorHandle*>(dynamic_handle));
        dynamic_handle = nullptr;
    }
}

float* tensor::getData() {
    if (use_dynamic_alloc && dynamic_handle) {
        auto& allocator = DynamicTensorAllocator::instance();
        float* ptr = allocator.getTensorData(
            static_cast<DynamicTensorAllocator::TensorHandle*>(dynamic_handle));
        if (!ptr) {
            std::cerr << "\n❌❌❌ PANIC: OUT OF MEMORY (lazy load) ❌❌❌" << std::endl;
            std::cerr << "⛔ getTensorData() a échoué - MemoryGuard limite atteinte" << std::endl;
            auto& guard = MemoryGuard::instance();
            guard.printStats();
            std::cerr << "\n⚠️  Arrêt contrôlé pour éviter un crash OS..." << std::endl;
            std::exit(1);
        }
        return ptr;
    }
    return data.data();
}

const float* tensor::getData() const {
    if (use_dynamic_alloc && dynamic_handle) {
        auto& allocator = DynamicTensorAllocator::instance();
        float* ptr = allocator.getTensorData(
            static_cast<DynamicTensorAllocator::TensorHandle*>(
                const_cast<void*>(dynamic_handle)));
        if (!ptr) {
            std::cerr << "\n❌❌❌ PANIC: OUT OF MEMORY (lazy load const) ❌❌❌" << std::endl;
            std::cerr << "⛔ getTensorData() a échoué - MemoryGuard limite atteinte" << std::endl;
            auto& guard = MemoryGuard::instance();
            guard.printStats();
            std::cerr << "\n⚠️  Arrêt contrôlé pour éviter un crash OS..." << std::endl;
            std::exit(1);
        }
        return ptr;
    }
    return data.data();
}

size_t tensor::getSize() const {
    if (use_dynamic_alloc && dynamic_handle) {
        auto handle = static_cast<DynamicTensorAllocator::TensorHandle*>(
            const_cast<void*>(dynamic_handle));
        return handle->size;
    }
    return data.size();
}

// TensorSystem is an API facade; all numerical work belongs to a runtime.
TensorSystem::TensorSystem() = default;
TensorSystem::~TensorSystem() = default;

bool TensorSystem::initialize() {
    initialized = true;
    return true;
}

bool TensorSystem::computeWeights(std::vector<tensor>& tensors) {
    if (!initialized) {
        std::cerr << "TensorSystem::computeWeights: system not initialized\n";
        return false;
    }
    std::vector<uint16_t> weights;
    if (!RuntimeRouter::instance().dispatchTensorWeights(tensors, weights) ||
        weights.size() != tensors.size()) return false;
    for (size_t i = 0; i < tensors.size(); ++i) tensors[i].Weight = weights[i];
    return true;
}

// ------------------------ SafeTensors writer/reader using json-glib ------------------------

static void write_le_u32(std::ofstream &f, uint32_t v) {
    unsigned char b[4];
    b[0] = (unsigned char)(v & 0xFF);
    b[1] = (unsigned char)((v >> 8) & 0xFF);
    b[2] = (unsigned char)((v >> 16) & 0xFF);
    b[3] = (unsigned char)((v >> 24) & 0xFF);
    f.write(reinterpret_cast<const char*>(b), 4);
}

static bool read_le_u32(std::ifstream &f, uint32_t &out) {
    unsigned char b[4];
    if (!f.read(reinterpret_cast<char*>(b), 4)) return false;
    out = (uint32_t)b[0] | ((uint32_t)b[1] << 8) | ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
    return true;
}

bool save_safetensors(const std::string &fname, const std::vector<std::string>& names,
                      const std::vector<size_t>& paramsCount, const std::vector<tensor>& params)
{
    if (names.size() != paramsCount.size()) {
        std::cerr << "save_safetensors: names/paramsCount mismatch\n";
        return false;
    }

    size_t totalCount = 0;
    for (size_t c : paramsCount) totalCount += c;
    if (totalCount > params.size()) {
        std::cerr << "save_safetensors: not enough params provided\n";
        return false;
    }

    // Construction du JSON avec nlohmann/json
    json root;
    json tensors;
    
    size_t offset_bytes = 0;
    for (size_t i = 0; i < names.size(); ++i) {
        json tensor_info;
        tensor_info["dtype"] = "u16";
        tensor_info["shape"] = {paramsCount[i]};
        tensor_info["offset"] = offset_bytes;
        tensors[names[i]] = tensor_info;
        offset_bytes += paramsCount[i] * 2;
    }
    
    root["tensors"] = tensors;
    std::string json_str = root.dump();

    // Écriture du fichier
    std::ofstream fout(fname, std::ios::binary);
    if (!fout) {
        std::cerr << "save_safetensors: cannot open " << fname << " for writing\n";
        return false;
    }

    const char magic[] = "safetensors";
    fout.write(magic, 10);
    write_le_u32(fout, static_cast<uint32_t>(json_str.size()));
    fout.write(json_str.data(), static_cast<std::streamsize>(json_str.size()));

    // Section données
    size_t pidx = 0;
    for (size_t i = 0; i < paramsCount.size(); ++i) {
        for (size_t k = 0; k < paramsCount[i]; ++k) {
            uint16_t w = 0;
            if (pidx < params.size()) w = params[pidx].Weight;
            unsigned char b0 = static_cast<unsigned char>(w & 0xFF);
            unsigned char b1 = static_cast<unsigned char>((w >> 8) & 0xFF);
            fout.put(static_cast<char>(b0));
            fout.put(static_cast<char>(b1));
            ++pidx;
        }
    }

    return true;
}

bool load_safetensors(const std::string &fname, std::vector<std::string>& names,
                      std::vector<size_t>& paramsCount, std::vector<tensor>& params)
{
    names.clear();
    paramsCount.clear();
    params.clear();

    std::ifstream fin(fname, std::ios::binary);
    if (!fin) {
        std::cerr << "load_safetensors: cannot open " << fname << "\n";
        return false;
    }

    char magic[11] = {0};
    fin.read(magic, 10);
    if (fin.gcount() != 10 || std::string(magic, 10) != "safetensors") {
        std::cerr << "load_safetensors: invalid magic\n";
        return false;
    }

    uint32_t jlen = 0;
    if (!read_le_u32(fin, jlen) || jlen == 0) {
        std::cerr << "load_safetensors: invalid json length\n";
        return false;
    }

    std::string json_str;
    json_str.resize(jlen);
    fin.read(&json_str[0], jlen);
    if (static_cast<uint32_t>(fin.gcount()) != jlen) {
        std::cerr << "load_safetensors: failed to read json block\n";
        return false;
    }

    // Parse JSON avec nlohmann/json
    json root;
    try {
        root = json::parse(json_str);
    } catch (const json::exception& e) {
        std::cerr << "load_safetensors: json parse error: " << e.what() << "\n";
        return false;
    }

    if (!root.contains("tensors") || !root["tensors"].is_object()) {
        std::cerr << "load_safetensors: missing or invalid tensors object\n";
        return false;
    }

    // Extraction des informations tenseurs
    const auto& tensors = root["tensors"];
    for (const auto& [name, info] : tensors.items()) {
        if (!info.contains("shape") || !info["shape"].is_array() || 
            !info.contains("dtype") || info["dtype"] != "u16") {
            continue;
        }
        names.push_back(name);
        paramsCount.push_back(info["shape"][0]);
    }

    // Lecture section données
    fin.seekg(0, std::ios::end);
    std::streamoff file_end = fin.tellg();
    std::streamoff data_start = 10 + 4 + static_cast<std::streamoff>(jlen);
    if (data_start > file_end) {
        std::cerr << "load_safetensors: invalid data start\n";
        return false;
    }

    std::streamoff data_size = file_end - data_start;
    fin.seekg(data_start, std::ios::beg);

    std::vector<unsigned char> data;
    data.resize(static_cast<size_t>(data_size));
    fin.read(reinterpret_cast<char*>(data.data()), data_size);
    size_t got = static_cast<size_t>(fin.gcount());
    if (got != static_cast<size_t>(data_size)) {
        std::cerr << "load_safetensors: failed to read data section\n";
        return false;
    }

    // Reconstruction des params
    params.clear();
    size_t byte_idx = 0;
    for (size_t i = 0; i < paramsCount.size(); ++i) {
        for (size_t k = 0; k < paramsCount[i]; ++k) {
            uint16_t w = 0;
            if (byte_idx + 1 < data.size()) {
                w = static_cast<uint16_t>(
                    (unsigned char)data[byte_idx] | 
                    ((unsigned char)data[byte_idx + 1] << 8)
                );
            }
            // Créer tensor in-place pour éviter copie
            params.emplace_back();
            tensor& t = params.back();
            t.Weight = w;
            t.Value = 0;
            t.Length = 0;
            t.Pos = Vector4F{0.0f,0.0f,0.0f,0.0f};
            byte_idx += 2;
        }
    }

    return true;
}