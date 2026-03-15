#ifndef __TENSOR_HELPERS_HPP__
#define __TENSOR_HELPERS_HPP__

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <random>
#include <cstdint>
#include <iostream>
#include <filesystem>
#include <algorithm> // std::clamp
#include <cctype>    // std::tolower
#include <cmath>
#include <iomanip>   // dump formaté
#include <numeric>   // std::accumulate
#include <regex>
#include <unordered_set>
#include <unordered_map>
#include <stdio.h>
#include <cstring>
#include <iterator>
#include <optional>

namespace fs = std::filesystem;

#include "include/json.hpp"
using json = nlohmann::json;

#include "Sha256.hpp"

// NOTE: STB_IMAGE_IMPLEMENTATION is defined in src/stb_image_impl.cpp
// Do NOT define it here to avoid multiple definition errors
#include "stb_image.h"

// Lit tout le fichier en bytes. Retourne true si OK.
static inline bool readFileToVector(const std::string &path, std::vector<uint8_t> &out, std::string *err = nullptr)
{
    out.clear();
    try
    {
        std::ifstream ifs(path, std::ios::binary | std::ios::ate);
        if (!ifs)
        {
            if (err) *err = "open failed";
            return false;
        }
        std::streamsize sz = ifs.tellg();
        ifs.seekg(0, std::ios::beg);
        if (sz <= 0)
        {
            out.clear();
            return true;
        }
        out.resize(static_cast<size_t>(sz));
        if (!ifs.read(reinterpret_cast<char *>(out.data()), sz))
        {
            if (err) *err = "read failed";
            return false;
        }
        return true;
    }
    catch (const std::exception &e)
    {
        if (err) *err = e.what();
        return false;
    }
    catch (...)
    {
        if (err) *err = "unknown error";
        return false;
    }
}

static inline std::string bytesToString(const std::vector<uint8_t> &buf, size_t off = 0, size_t len = std::string::npos)
{
    if (off >= buf.size()) return std::string();
    size_t avail = buf.size() - off;
    if (len == std::string::npos || len > avail) len = avail;
    return std::string(reinterpret_cast<const char *>(buf.data() + off), len);
}

// remplacement de stb_image_resize par une fonction interne simple (nearest-neighbor)
static inline void resizeNearest(const unsigned char *src, int iw, int ih, int channels,
                                 unsigned char *dst, int ow, int oh)
{
    for (int y = 0; y < oh; ++y)
    {
        int sy = (y * ih) / oh;
        for (int x = 0; x < ow; ++x)
        {
            int sx = (x * iw) / ow;
            const unsigned char *s = src + (sy * iw + sx) * channels;
            unsigned char *d = dst + (y * ow + x) * channels;
            for (int c = 0; c < channels; ++c)
                d[c] = s[c];
        }
    }
}

// Redimensionnement bilinéaire (déterministe) pour images u8 interleavées.
// Utile surtout en downscale pour éviter les artefacts de nearest-neighbor.
static inline void resizeBilinear(const unsigned char* src, int iw, int ih, int channels,
                                  unsigned char* dst, int ow, int oh)
{
    if (!src || !dst || iw <= 0 || ih <= 0 || ow <= 0 || oh <= 0 || channels <= 0) return;

    // Cas dégénérés: fallback nearest.
    if (iw == 1 || ih == 1 || ow == 1 || oh == 1) {
        resizeNearest(src, iw, ih, channels, dst, ow, oh);
        return;
    }

    const float sx = (float)iw / (float)ow;
    const float sy = (float)ih / (float)oh;

    for (int y = 0; y < oh; ++y) {
        const float fy = ((float)y + 0.5f) * sy - 0.5f;
        int y0 = (int)std::floor(fy);
        float wy = fy - (float)y0;
        if (y0 < 0) { y0 = 0; wy = 0.0f; }
        int y1 = y0 + 1;
        if (y1 >= ih) { y1 = ih - 1; }

        for (int x = 0; x < ow; ++x) {
            const float fx = ((float)x + 0.5f) * sx - 0.5f;
            int x0 = (int)std::floor(fx);
            float wx = fx - (float)x0;
            if (x0 < 0) { x0 = 0; wx = 0.0f; }
            int x1 = x0 + 1;
            if (x1 >= iw) { x1 = iw - 1; }

            const unsigned char* p00 = src + ((y0 * iw + x0) * channels);
            const unsigned char* p10 = src + ((y0 * iw + x1) * channels);
            const unsigned char* p01 = src + ((y1 * iw + x0) * channels);
            const unsigned char* p11 = src + ((y1 * iw + x1) * channels);

            unsigned char* out = dst + ((y * ow + x) * channels);

            const float w00 = (1.0f - wx) * (1.0f - wy);
            const float w10 = wx * (1.0f - wy);
            const float w01 = (1.0f - wx) * wy;
            const float w11 = wx * wy;

            for (int c = 0; c < channels; ++c) {
                const float v =
                    w00 * (float)p00[c] +
                    w10 * (float)p10[c] +
                    w01 * (float)p01[c] +
                    w11 * (float)p11[c];
                int iv = (int)std::lround(v);
                iv = std::clamp(iv, 0, 255);
                out[c] = (unsigned char)iv;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Resize haute qualité (bicubique). Pour RGB, on travaille en linéaire sRGB
// afin d'éviter les artefacts gamma visibles sur les gradients.
// ---------------------------------------------------------------------------

static inline float __srgb_u8_to_linear_f(unsigned char u) {
    const float s = (float)u / 255.0f;
    // sRGB exact piecewise
    if (s <= 0.04045f) return s / 12.92f;
    return std::pow((s + 0.055f) / 1.055f, 2.4f);
}

static inline unsigned char __linear_f_to_srgb_u8(float l) {
    l = std::clamp(l, 0.0f, 1.0f);
    float s = 0.0f;
    if (l <= 0.0031308f) s = 12.92f * l;
    else s = 1.055f * std::pow(l, 1.0f / 2.4f) - 0.055f;
    int v = (int)std::lround(s * 255.0f);
    v = std::clamp(v, 0, 255);
    return (unsigned char)v;
}

static inline float __cubic_catmull_rom(float x) {
    x = std::fabs(x);
    if (x <= 1.0f) {
        return (1.5f * x - 2.5f) * x * x + 1.0f;
    }
    if (x < 2.0f) {
        return ((-0.5f * x + 2.5f) * x - 4.0f) * x + 2.0f;
    }
    return 0.0f;
}

// Bicubique générique sur u8 (sans correction gamma). Stable et déterministe.
static inline void resizeBicubicU8(const unsigned char* src, int iw, int ih, int channels,
                                   unsigned char* dst, int ow, int oh)
{
    if (!src || !dst || iw <= 0 || ih <= 0 || ow <= 0 || oh <= 0 || channels <= 0) return;
    if (iw == ow && ih == oh) {
        std::memcpy(dst, src, (size_t)iw * (size_t)ih * (size_t)channels);
        return;
    }

    const float sx = (float)iw / (float)ow;
    const float sy = (float)ih / (float)oh;

    for (int y = 0; y < oh; ++y) {
        const float fy = ((float)y + 0.5f) * sy - 0.5f;
        const int iy = (int)std::floor(fy);

        float wy[4];
        for (int j = -1; j <= 2; ++j) {
            wy[j + 1] = __cubic_catmull_rom((float)(iy + j) - fy);
        }

        for (int x = 0; x < ow; ++x) {
            const float fx = ((float)x + 0.5f) * sx - 0.5f;
            const int ix = (int)std::floor(fx);

            float wx[4];
            for (int i = -1; i <= 2; ++i) {
                wx[i + 1] = __cubic_catmull_rom((float)(ix + i) - fx);
            }

            unsigned char* out = dst + ((y * ow + x) * channels);

            for (int c = 0; c < channels; ++c) {
                float acc = 0.0f;
                float wsum = 0.0f;

                for (int j = -1; j <= 2; ++j) {
                    int syi = iy + j;
                    syi = std::clamp(syi, 0, ih - 1);
                    const float wyj = wy[j + 1];
                    for (int i = -1; i <= 2; ++i) {
                        int sxi = ix + i;
                        sxi = std::clamp(sxi, 0, iw - 1);
                        const float w = wyj * wx[i + 1];
                        const unsigned char* p = src + ((syi * iw + sxi) * channels);
                        acc += w * (float)p[c];
                        wsum += w;
                    }
                }

                if (wsum != 0.0f) acc /= wsum;
                int iv = (int)std::lround(acc);
                iv = std::clamp(iv, 0, 255);
                out[c] = (unsigned char)iv;
            }
        }
    }
}

// Bicubique RGB en linéaire sRGB (meilleure qualité perçue, surtout en downscale).
static inline void resizeBicubicRGB_SRGBLinear(const unsigned char* src, int iw, int ih,
                                               unsigned char* dst, int ow, int oh)
{
    if (!src || !dst || iw <= 0 || ih <= 0 || ow <= 0 || oh <= 0) return;
    if (iw == ow && ih == oh) {
        std::memcpy(dst, src, (size_t)iw * (size_t)ih * 3ULL);
        return;
    }

    // LUT sRGB->lin (256) pour accélérer.
    float lut[256];
    for (int i = 0; i < 256; ++i) lut[i] = __srgb_u8_to_linear_f((unsigned char)i);

    const float sx = (float)iw / (float)ow;
    const float sy = (float)ih / (float)oh;

    for (int y = 0; y < oh; ++y) {
        const float fy = ((float)y + 0.5f) * sy - 0.5f;
        const int iy = (int)std::floor(fy);

        float wy[4];
        for (int j = -1; j <= 2; ++j) {
            wy[j + 1] = __cubic_catmull_rom((float)(iy + j) - fy);
        }

        for (int x = 0; x < ow; ++x) {
            const float fx = ((float)x + 0.5f) * sx - 0.5f;
            const int ix = (int)std::floor(fx);

            float wx[4];
            for (int i = -1; i <= 2; ++i) {
                wx[i + 1] = __cubic_catmull_rom((float)(ix + i) - fx);
            }

            unsigned char* out = dst + ((y * ow + x) * 3);
            for (int c = 0; c < 3; ++c) {
                float acc = 0.0f;
                float wsum = 0.0f;
                for (int j = -1; j <= 2; ++j) {
                    int syi = std::clamp(iy + j, 0, ih - 1);
                    const float wyj = wy[j + 1];
                    for (int i = -1; i <= 2; ++i) {
                        int sxi = std::clamp(ix + i, 0, iw - 1);
                        const float w = wyj * wx[i + 1];
                        const unsigned char* p = src + ((syi * iw + sxi) * 3);
                        acc += w * lut[p[c]];
                        wsum += w;
                    }
                }
                if (wsum != 0.0f) acc /= wsum;
                out[c] = __linear_f_to_srgb_u8(acc);
            }
        }
    }
}

enum Modality : unsigned
{
    MOD_NONE = 0,
    MOD_TEXT = 1 << 0,
    MOD_AUDIO = 1 << 1,
    MOD_IMAGE = 1 << 2,
    MOD_VIDEO = 1 << 3
};

static inline unsigned detectModalities(const std::string &dataset_dir)
{
    static const std::regex re_text(R"(\.(txt|json|csv|md|xml)$)", std::regex::icase);
    static const std::regex re_audio(R"(\.(wav|flac|mp3|ogg|m4a|aac)$)", std::regex::icase);
    static const std::regex re_image(R"(\.(png|jpg|jpeg|bmp|tiff|webp)$)", std::regex::icase);
    static const std::regex re_video(R"(\.(mp4|mkv|avi|mov|webm|flv|ts)$)", std::regex::icase);

    unsigned mask = MOD_NONE;
    for (auto &p : fs::recursive_directory_iterator(dataset_dir))
    {
        if (!p.is_regular_file()) continue;
        const std::string s = p.path().string();
        if (std::regex_search(s, re_text)) mask |= MOD_TEXT;
        if (std::regex_search(s, re_audio)) mask |= MOD_AUDIO;
        if (std::regex_search(s, re_image)) mask |= MOD_IMAGE;
        if (std::regex_search(s, re_video)) mask |= MOD_VIDEO;
        if (mask == (MOD_TEXT | MOD_AUDIO | MOD_IMAGE | MOD_VIDEO)) break;
    }
    return mask;
}

struct MagicToken
{
    uint32_t modality_mask; // bits Modality
    uint32_t seed;          // déterministe d'après dataset
    float embed[8];         // petit embedding fixe (8 dim)
};

// Hash SHA256 du dataset basé sur les noms et tailles de fichiers
static inline std::string datasetSHA256Hash(const std::string &dir)
{
    std::string combined;
    std::vector<std::string> file_infos;
    
    // Collecter tous les fichiers avec leurs métadonnées
    for (auto &p : fs::recursive_directory_iterator(dir)) {
        if (!p.is_regular_file()) continue;
        
        std::string info = p.path().filename().string() + "|" + 
                          std::to_string(fs::file_size(p.path()));
        file_infos.push_back(info);
    }
    
    // Trier pour garantir un ordre déterministe
    std::sort(file_infos.begin(), file_infos.end());
    
    // Combiner toutes les infos
    for (const auto &info : file_infos) {
        combined += info + "\n";
    }
    
    // Calculer le SHA256
    return sha256(combined);
}

static inline uint32_t simpleDatasetHash(const std::string &dir)
{
    uint64_t h = 1469598103934665603ull; // FNV offset basis
    for (auto &p : fs::recursive_directory_iterator(dir))
    {
        if (!p.is_regular_file()) continue;
        auto s = p.path().filename().string();
        for (unsigned char c : s) { h ^= c; h *= 1099511628211ull; }
    }
    return static_cast<uint32_t>((h ^ (h >> 32)) & 0xFFFFFFFFu);
}

static inline MagicToken makeMagicToken(unsigned modality_mask, const std::string &dir)
{
    MagicToken t{};
    t.modality_mask = modality_mask;
    t.seed = simpleDatasetHash(dir) ^ (modality_mask * 0x9e3779b1u);
    std::mt19937 rng(t.seed);
    std::uniform_real_distribution<float> U(-1.f, 1.f);
    for (auto &e : t.embed) e = U(rng);
    return t;
}

static inline std::string sanitize_utf8(const std::string &s)
{
    const unsigned char *b = reinterpret_cast<const unsigned char *>(s.data());
    size_t n = s.size();
    std::string out;
    out.reserve(n);
    size_t i = 0;
    while (i < n)
    {
        unsigned char c = b[i];
        if (c < 0x80) { out.push_back((char)c); ++i; continue; }
        int len = 0;
        if ((c & 0xE0) == 0xC0) len = 2;
        else if ((c & 0xF0) == 0xE0) len = 3;
        else if ((c & 0xF8) == 0xF0) len = 4;
        else { out.append("\xEF\xBF\xBD"); ++i; continue; } // U+FFFD
        if (i + (size_t)len > n) { out.append("\xEF\xBF\xBD"); ++i; continue; }
        bool ok = true;
        for (int k = 1; k < len; ++k) if ((b[i + k] & 0xC0) != 0x80) { ok = false; break; }
        if (!ok) { out.append("\xEF\xBF\xBD"); ++i; continue; }
        for (int k = 0; k < len; ++k) out.push_back((char)b[i + k]);
        i += (size_t)len;
    }
    return out;
}

// image -> embedding helper (static inline)
static inline std::vector<float> imageToEmbedding(const std::vector<uint8_t> &img, int iw, int ih, int dim)
{
    std::vector<float> out((size_t)dim, 0.0f);
    if (img.empty() || iw <= 0 || ih <= 0 || dim <= 0) return out;
    uint64_t sum = 0;
    size_t N = (size_t)iw * (size_t)ih;
    size_t limit = std::min(N, img.size());
    for (size_t i = 0; i < limit; ++i) sum += img[i];
    double mean = (limit > 0) ? (double(sum) / double(limit)) : 0.0;
    float v = static_cast<float>((mean / 255.0) * 2.0 - 1.0f);
    for (int d = 0; d < dim; ++d) out[(size_t)d] = v;
    return out;
}

// helper: write uint32 little-endian (used by packToSafetensor)
static inline void write_u32_le(std::ofstream &f, uint32_t v)
{
    unsigned char b[4];
    b[0] = static_cast<unsigned char>(v & 0xFF);
    b[1] = static_cast<unsigned char>((v >> 8) & 0xFF);
    b[2] = static_cast<unsigned char>((v >> 16) & 0xFF);
    b[3] = static_cast<unsigned char>((v >> 24) & 0xFF);
    f.write(reinterpret_cast<char *>(b), 4);
}

// Gestionnaire global de mémoire RAM pour le dataset
struct DatasetMemoryManager {
    static DatasetMemoryManager& instance() {
        // Leaky singleton: évite les crashs d'ordre de destruction des statics
        // quand des DatasetItem sont détruits pendant l'atexit.
        static DatasetMemoryManager* mgr = new DatasetMemoryManager();
        return *mgr;
    }
    
    size_t max_ram_bytes = 10ULL * 1024 * 1024 * 1024; // 10 GB par défaut
    size_t current_ram_bytes = 0;
    size_t peak_ram_bytes = 0;
    std::unordered_map<void*, size_t> allocations; // Tracker les allocations
    
    void setMaxRAM(size_t bytes) { max_ram_bytes = bytes; }
    
    bool canAllocate(size_t bytes) const {
        return (current_ram_bytes + bytes) <= max_ram_bytes;
    }
    
    void trackAllocation(void* ptr, size_t bytes) {
        current_ram_bytes += bytes;
        peak_ram_bytes = std::max(peak_ram_bytes, current_ram_bytes);
        allocations[ptr] = bytes;
    }
    
    void trackDeallocation(void* ptr) {
        auto it = allocations.find(ptr);
        if (it != allocations.end()) {
            current_ram_bytes -= it->second;
            allocations.erase(it);
        }
    }
    
    size_t getCurrentRAM() const { return current_ram_bytes; }
    size_t getPeakRAM() const { return peak_ram_bytes; }
    size_t getAvailableRAM() const { 
        return max_ram_bytes > current_ram_bytes ? max_ram_bytes - current_ram_bytes : 0; 
    }
    
    float getUsagePercent() const {
        return 100.0f * static_cast<float>(current_ram_bytes) / static_cast<float>(max_ram_bytes);
    }
    
    void printStats() const {
        std::cout << "\n💾 RAM Manager Stats:" << std::endl;
        std::cout << "   Current: " << (current_ram_bytes / 1024 / 1024) << " MB" << std::endl;
        std::cout << "   Peak:    " << (peak_ram_bytes / 1024 / 1024) << " MB" << std::endl;
        std::cout << "   Max:     " << (max_ram_bytes / 1024 / 1024) << " MB" << std::endl;
        std::cout << "   Usage:   " << std::fixed << std::setprecision(1) 
                  << getUsagePercent() << "%" << std::endl;
        std::cout << "   Active allocations: " << allocations.size() << std::endl;
    }
    
private:
    DatasetMemoryManager() = default;
};

// Structure d'élément de dataset avec gestion intelligente de la RAM
struct DatasetItem {
    std::string name;
    bool is_linked = false;
    
    // Chemins (toujours présents - négligeable en RAM)
    std::string text_file;
    std::string image_file;
    std::string audio_file;
    std::string video_file;
    
    // Données (chargées à la demande avec gestion RAM)
    std::optional<std::string> text;
    std::vector<uint8_t> img;
    std::vector<uint8_t> audio_bytes;
    std::vector<uint8_t> video_bytes;

    bool img_loaded = false;
    bool audio_loaded = false;
    bool video_loaded = false;
    
    // Métadonnées
    int w = 0, h = 0;
    int img_c = 0; // 0 = inconnu, 1 = grayscale, 3 = RGB
    
    // Tracking RAM et LRU
    mutable uint64_t last_access_time = 0;
    mutable size_t estimated_ram_usage = 0;

    ~DatasetItem();
    
    // Libère toutes les données chargées
    void unload() {
        auto& mgr = DatasetMemoryManager::instance();
        
        if (text.has_value()) {
            mgr.trackDeallocation((void*)text->data());
            text.reset();
        }
        if (img_loaded) {
            void* ptr = img.empty() ? nullptr : (void*)img.data();
            if (ptr) mgr.trackDeallocation(ptr);
            std::vector<uint8_t>().swap(img);
            img_loaded = false;
        }
        if (audio_loaded) {
            void* ptr = audio_bytes.empty() ? nullptr : (void*)audio_bytes.data();
            if (ptr) mgr.trackDeallocation(ptr);
            std::vector<uint8_t>().swap(audio_bytes);
            audio_loaded = false;
        }
        if (video_loaded) {
            void* ptr = video_bytes.empty() ? nullptr : (void*)video_bytes.data();
            if (ptr) mgr.trackDeallocation(ptr);
            std::vector<uint8_t>().swap(video_bytes);
            video_loaded = false;
        }
        
        estimated_ram_usage = 0;
    }
    
    // Estime la RAM nécessaire pour charger un item
    size_t estimateRAMNeeded() const {
        size_t total = 0;
        
        if (!text_file.empty() && !text.has_value()) {
            try {
                auto fsize = fs::file_size(text_file);
                total += fsize * 2; // UTF-8 peut doubler
            } catch (...) {}
        }
        
        if (!image_file.empty() && !img_loaded) {
            // Estimation basée sur la taille cible
            const size_t c = (img_c > 0) ? static_cast<size_t>(img_c) : 1ULL;
            total += (size_t)w * (size_t)h * c;
        }
        
        if (!audio_file.empty() && !audio_loaded) {
            try {
                total += fs::file_size(audio_file);
            } catch (...) {}
        }
        
        if (!video_file.empty() && !video_loaded) {
            try {
                total += fs::file_size(video_file);
            } catch (...) {}
        }
        
        return total;
    }
    
    // Met à jour le timestamp d'accès (pour LRU)
    void touch() const {
        static uint64_t global_counter = 0;
        last_access_time = ++global_counter;
    }
    
    // Lazy loaders avec gestion RAM intelligente
    bool loadText() {
        if (text.has_value()) {
            touch();
            return true;
        }
        if (text_file.empty()) return false;
        
        auto& mgr = DatasetMemoryManager::instance();
        
        try {
            // Estimer la taille
            size_t file_size = fs::file_size(text_file);
            size_t needed = file_size * 2; // Sécurité UTF-8
            
            // Vérifier si on peut allouer
            if (!mgr.canAllocate(needed)) {
                return false; // Pas assez de RAM
            }
            
            std::ifstream f(text_file);
            if (!f) return false;
            std::ostringstream ss;
            ss << f.rdbuf();
            
            std::string loaded = sanitize_utf8(ss.str());
            size_t actual_size = loaded.size();
            
            text = std::move(loaded);
            mgr.trackAllocation((void*)text->data(), actual_size);
            estimated_ram_usage += actual_size;
            touch();
            
            return true;
        } catch (...) {
            return false;
        }
    }
    
    bool loadImage(int target_w, int target_h) {
        if (img_loaded) {
            touch();
            return true;
        }
        if (image_file.empty()) return false;
        
        auto& mgr = DatasetMemoryManager::instance();
        
        try {
            // Estimer la RAM nécessaire
            size_t needed = (size_t)target_w * (size_t)target_h * 4; // Buffer temporaire RGB + final
            
            if (!mgr.canAllocate(needed)) {
                return false;
            }
            
            int w_img = 0, h_img = 0, c = 0;
            unsigned char *data = stbi_load(image_file.c_str(), &w_img, &h_img, &c, 3);
            if (!data) return false;
            
            std::vector<unsigned char> src((size_t)w_img * (size_t)h_img * 3);
            std::memcpy(src.data(), data, src.size());
            stbi_image_free(data);
            
            std::vector<unsigned char> dst((size_t)target_w * (size_t)target_h * 3);
            // Resize haute qualité avant conversion grayscale.
            resizeBicubicRGB_SRGBLinear(src.data(), w_img, h_img, dst.data(), target_w, target_h);
            
            std::vector<uint8_t> grayscale((size_t)target_w * (size_t)target_h);
            for (int yy = 0; yy < target_h; ++yy) {
                for (int xx = 0; xx < target_w; ++xx) {
                    size_t off = ((size_t)yy * (size_t)target_w + (size_t)xx) * 3;
                    unsigned v = (unsigned)dst[off] + (unsigned)dst[off+1] + (unsigned)dst[off+2];
                    grayscale[(size_t)yy * (size_t)target_w + (size_t)xx] = static_cast<uint8_t>(v / 3);
                }
            }
            
            size_t actual_size = grayscale.size();
            img = std::move(grayscale);
            img_loaded = true;
            mgr.trackAllocation((void*)img.data(), actual_size);
            estimated_ram_usage += actual_size;
            
            w = target_w;
            h = target_h;
            img_c = 1;
            touch();
            
            return true;
        } catch (...) {
            return false;
        }
    }

    // Charge une image en RGB (3 canaux) et la redimensionne en nearest.
    // Ne modifie pas loadImage() (grayscale) pour préserver la compatibilité.
    bool loadImageRGB(int target_w, int target_h) {
        if (img_loaded && img_c == 3 && w == target_w && h == target_h) {
            touch();
            return true;
        }
        if (image_file.empty()) return false;

        auto& mgr = DatasetMemoryManager::instance();

        try {
            // Estimer la RAM nécessaire
            size_t needed = (size_t)target_w * (size_t)target_h * 3 * 2; // temporaire + final
            if (!mgr.canAllocate(needed)) {
                return false;
            }

            int w_img = 0, h_img = 0, c = 0;
            unsigned char* data = stbi_load(image_file.c_str(), &w_img, &h_img, &c, 3);
            if (!data) return false;

            std::vector<unsigned char> src((size_t)w_img * (size_t)h_img * 3);
            std::memcpy(src.data(), data, src.size());
            stbi_image_free(data);

            std::vector<uint8_t> dst((size_t)target_w * (size_t)target_h * 3);
            // Resize haute qualité (bicubique en linéaire sRGB) pour préserver au mieux les détails.
            resizeBicubicRGB_SRGBLinear(src.data(), w_img, h_img, dst.data(), target_w, target_h);

            // Si on avait déjà une image chargée, décrémenter l'ancien tracking
            if (img_loaded) {
                void* ptr = img.empty() ? nullptr : (void*)img.data();
                if (ptr) mgr.trackDeallocation(ptr);
                if (estimated_ram_usage >= img.size()) estimated_ram_usage -= img.size();
            }

            const size_t actual_size = dst.size();
            img = std::move(dst);
            img_loaded = true;
            mgr.trackAllocation((void*)img.data(), actual_size);
            estimated_ram_usage += actual_size;

            w = target_w;
            h = target_h;
            img_c = 3;
            touch();
            return true;
        } catch (...) {
            return false;
        }
    }
    
    bool loadAudio() {
        if (audio_loaded) {
            touch();
            return true;
        }
        if (audio_file.empty()) return false;
        
        auto& mgr = DatasetMemoryManager::instance();
        
        try {
            size_t file_size = fs::file_size(audio_file);
            
            if (!mgr.canAllocate(file_size)) {
                return false;
            }
            
            std::ifstream f(audio_file, std::ios::binary);
            if (!f) return false;
            std::vector<uint8_t> data(
                (std::istreambuf_iterator<char>(f)),
                std::istreambuf_iterator<char>()
            );
            
            size_t actual_size = data.size();
            audio_bytes = std::move(data);
            audio_loaded = true;
            mgr.trackAllocation((void*)audio_bytes.data(), actual_size);
            estimated_ram_usage += actual_size;
            touch();
            
            return true;
        } catch (...) {
            return false;
        }
    }
    
    bool loadVideo() {
        if (video_loaded) {
            touch();
            return true;
        }
        if (video_file.empty()) return false;
        
        auto& mgr = DatasetMemoryManager::instance();
        
        try {
            size_t file_size = fs::file_size(video_file);
            
            if (!mgr.canAllocate(file_size)) {
                return false;
            }
            
            std::ifstream f(video_file, std::ios::binary);
            if (!f) return false;
            std::vector<uint8_t> data(
                (std::istreambuf_iterator<char>(f)),
                std::istreambuf_iterator<char>()
            );
            
            size_t actual_size = data.size();
            video_bytes = std::move(data);
            video_loaded = true;
            mgr.trackAllocation((void*)video_bytes.data(), actual_size);
            estimated_ram_usage += actual_size;
            touch();
            
            return true;
        } catch (...) {
            return false;
        }
    }
    
    // Helper: compte les modalités disponibles
    int countModalities() const {
        int count = 0;
        if (!text_file.empty()) count++;
        if (!image_file.empty()) count++;
        if (!audio_file.empty()) count++;
        if (!video_file.empty()) count++;
        return count;
    }
    
    // Helper: complétude en pourcentage (0-100)
    float completeness() const {
        return (countModalities() / 4.0f) * 100.0f;
    }
    
    // Helper: taille RAM actuelle
    size_t getRAMUsage() const {
        return estimated_ram_usage;
    }
    
    // Helper: vérifie si des données sont chargées
    bool isLoaded() const {
        return text.has_value() || img_loaded || audio_loaded || video_loaded;
    }
};

// Gestionnaire de Dataset avec éviction LRU intelligente
class DatasetManager {
public:
    DatasetManager(size_t max_ram_mb = 10240) {
        auto& mgr = DatasetMemoryManager::instance();
        mgr.setMaxRAM(max_ram_mb * 1024ULL * 1024ULL);
    }
    
    // Charge les items nécessaires en libérant les plus anciens si besoin
    bool ensureLoaded(std::vector<DatasetItem>& items, const std::vector<size_t>& indices, 
                      int target_w = 64, int target_h = 64) {
        auto& mgr = DatasetMemoryManager::instance();
        
        // Calculer la RAM nécessaire
        size_t needed = 0;
        for (size_t idx : indices) {
            if (idx >= items.size()) continue;
            if (!items[idx].isLoaded()) {
                needed += items[idx].estimateRAMNeeded();
            }
        }
        
        // Si pas assez de RAM, évincer les items les moins récemment utilisés
        if (!mgr.canAllocate(needed)) {
            evictLRU(items, needed);
        }
        
        // Charger les items demandés
        bool all_loaded = true;
        for (size_t idx : indices) {
            if (idx >= items.size()) continue;
            
            auto& item = items[idx];
            
            // Charger selon les modalités disponibles
            if (!item.image_file.empty() && !item.img_loaded) {
                if (!item.loadImage(target_w, target_h)) {
                    all_loaded = false;
                }
            }
            
            if (!item.text_file.empty() && !item.text.has_value()) {
                if (!item.loadText()) {
                    all_loaded = false;
                }
            }
        }
        
        return all_loaded;
    }
    
    // Éviction LRU : libère les items les moins récemment utilisés
    void evictLRU(std::vector<DatasetItem>& items, size_t bytes_needed) {
        auto& mgr = DatasetMemoryManager::instance();
        
        // Créer une liste des items chargés triés par last_access_time
        std::vector<size_t> loaded_indices;
        for (size_t i = 0; i < items.size(); ++i) {
            if (items[i].isLoaded()) {
                loaded_indices.push_back(i);
            }
        }
        
        // Trier par access time (les plus anciens en premier)
        std::sort(loaded_indices.begin(), loaded_indices.end(), 
                  [&items](size_t a, size_t b) {
                      return items[a].last_access_time < items[b].last_access_time;
                  });
        
        // Évincer jusqu'à avoir assez de RAM
        size_t freed = 0;
        for (size_t idx : loaded_indices) {
            if (mgr.canAllocate(bytes_needed)) break;
            
            size_t item_size = items[idx].getRAMUsage();
            items[idx].unload();
            freed += item_size;
        }
        
        if (freed > 0) {
            std::cout << "  ⟳ LRU éviction: libéré " << (freed / 1024 / 1024) << " MB" << std::endl;
        }
    }
    
    // Préchargement intelligent d'un batch
    bool preloadBatch(std::vector<DatasetItem>& items, size_t batch_start, size_t batch_size,
                     int target_w = 64, int target_h = 64) {
        std::vector<size_t> indices;
        for (size_t i = 0; i < batch_size && (batch_start + i) < items.size(); ++i) {
            indices.push_back(batch_start + i);
        }
        return ensureLoaded(items, indices, target_w, target_h);
    }
    
    // Statistiques
    void printStats() const {
        DatasetMemoryManager::instance().printStats();
    }
};

// implémentation de loadDataset (parcours récursif, indexation des linkables)
// Version optimisée RAM : ne charge PAS les données, seulement les métadonnées
static inline std::vector<DatasetItem> loadDataset(const std::string &root_dir, int target_w = 64, int target_h = 64, int min_modalities = 1)
{
    static const std::regex re_image(R"(\.(png|jpg|jpeg|bmp|tiff|webp)$)", std::regex::icase);
    static const std::regex re_text(R"(\.(txt|md|json|csv)$)", std::regex::icase);
    static const std::regex re_audio(R"(\.(wav|flac|mp3|ogg|m4a|aac)$)", std::regex::icase);
    static const std::regex re_video(R"(\.(mp4|mkv|avi|mov|webm|flv|ts)$)", std::regex::icase);
    
    std::vector<DatasetItem> items;
    if (root_dir.empty()) return items;

    std::cout << "\n📂 Indexation du dataset (lazy loading activé)..." << std::endl;

    // Phase 1: Indexer tous les fichiers par clé "linkable".
    // IMPORTANT: on utilise le chemin *relatif* sans extension (et pas seulement le basename)
    // pour éviter les collisions quand plusieurs sous-dossiers contiennent le même stem.
    std::unordered_map<std::string, std::vector<fs::path>> files_by_key;
    
    for (auto &p : fs::recursive_directory_iterator(root_dir))
    {
        try {
            if (!p.is_regular_file()) continue;
            const auto path = p.path();

            fs::path rel = path;
            try {
                rel = fs::relative(path, root_dir);
            } catch (...) {
                // fallback: garder le path tel quel
            }
            rel.replace_extension("");
            const std::string key = rel.generic_string();
            files_by_key[key].push_back(path);
        } catch (...) {
            continue;
        }
    }
    
    // Phase 2: Créer les items SANS charger les données (métadonnées uniquement)
    size_t total_linkables = 0;
    size_t valid_items = 0;
    
    auto extLower = [](const fs::path& p) -> std::string {
        std::string e = p.extension().string();
        std::transform(e.begin(), e.end(), e.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return e;
    };

    auto isBetterText = [&](const fs::path& candidate, const std::string& current_path) -> bool {
        if (current_path.empty()) return true;
        const std::string ce = extLower(candidate);
        const std::string curE = extLower(fs::path(current_path));

        auto rank = [](const std::string& e) -> int {
            if (e == ".txt") return 0;
            if (e == ".md") return 1;
            if (e == ".json") return 2;
            if (e == ".csv") return 3;
            return 10;
        };
        const int r1 = rank(ce);
        const int r2 = rank(curE);
        if (r1 != r2) return r1 < r2;
        return candidate.string() < current_path;
    };

    auto isBetterImage = [&](const fs::path& candidate, const std::string& current_path) -> bool {
        if (current_path.empty()) return true;
        const std::string ce = extLower(candidate);
        const std::string curE = extLower(fs::path(current_path));

        auto rank = [](const std::string& e) -> int {
            if (e == ".png") return 0;
            if (e == ".jpg" || e == ".jpeg") return 1;
            if (e == ".webp") return 2;
            if (e == ".bmp") return 3;
            if (e == ".tiff" || e == ".tif") return 4;
            return 10;
        };
        const int r1 = rank(ce);
        const int r2 = rank(curE);
        if (r1 != r2) return r1 < r2;
        return candidate.string() < current_path;
    };

    for (const auto &[key, paths] : files_by_key)
    {
        if (paths.empty()) continue;
        
        DatasetItem item;
        item.name = key;
        item.is_linked = (paths.size() > 1);
        item.w = target_w;  // Stocker la taille cible
        item.h = target_h;
        
        if (item.is_linked) total_linkables++;
        
        // Classifier chaque fichier par type (SANS charger les données)
        for (const auto &path : paths)
        {
            const std::string pathstr = path.string();
            
            if (std::regex_search(pathstr, re_image)) {
                if (isBetterImage(path, item.image_file)) {
                    item.image_file = pathstr;
                }
            }
            else if (std::regex_search(pathstr, re_text)) {
                if (isBetterText(path, item.text_file)) {
                    item.text_file = pathstr;
                }
            }
            else if (std::regex_search(pathstr, re_audio)) {
                item.audio_file = pathstr;
            }
            else if (std::regex_search(pathstr, re_video)) {
                item.video_file = pathstr;
            }
        }
        
        // Validation: vérifier le nombre de modalités
        int modality_count = item.countModalities();
        
        if (modality_count >= min_modalities) {
            items.push_back(std::move(item));
            valid_items++;
        }
    }
    
    // Rapport de validation (sans chargement de données)
    std::cout << "\n📊 Dataset Indexing Report:" << std::endl;
    std::cout << "   Items indexés:      " << valid_items << std::endl;
    std::cout << "   Linkables détectés: " << total_linkables << std::endl;
    
    if (total_linkables > 0) {
        size_t valid_linkables = 0;
        for (const auto &item : items) {
            if (item.is_linked && item.countModalities() >= 2) {
                valid_linkables++;
            }
        }
        std::cout << "   Linkables validés:  " << valid_linkables << " ✓" << std::endl;
        std::cout << "   Ratio:              " 
                  << std::fixed << std::setprecision(1) 
                  << (100.0 * valid_linkables / total_linkables) << "%" << std::endl;
    }
    
    std::cout << "   Seuil modalités:    " << min_modalities << std::endl;
    
    // Statistiques de complétude
    if (!items.empty()) {
        float avg_completeness = 0.0f;
        for (const auto &item : items) {
            avg_completeness += item.completeness();
        }
        avg_completeness /= items.size();
        std::cout << "   Complétude moy.:    " 
                  << std::fixed << std::setprecision(1) 
                  << avg_completeness << "%" << std::endl;
    }
    
    std::cout << "\n   ⚡ Mode: Lazy loading (données chargées à la demande)" << std::endl;
    std::cout << "   💾 RAM utilisée:    0 MB (métadonnées uniquement)" << std::endl;

    return items;
}

// Cache pour datasets volumineux
struct DatasetCache {
    std::string version = "1.0";
    std::string dataset_hash; // SHA256 au lieu de uint64_t
    std::vector<DatasetItem> items;
    
    bool save(const fs::path &cache_file) const {
        try {
            json j;
            j["version"] = version;
            j["dataset_hash"] = dataset_hash;
            j["items"] = json::array();
            
            for (const auto &item : items) {
                json ji;
                ji["name"] = item.name;
                ji["is_linked"] = item.is_linked;
                ji["text_file"] = item.text_file;
                ji["image_file"] = item.image_file;
                ji["audio_file"] = item.audio_file;
                ji["video_file"] = item.video_file;
                ji["w"] = item.w;
                ji["h"] = item.h;
                j["items"].push_back(ji);
            }
            
            std::ofstream f(cache_file);
            if (!f) return false;
            f << std::setw(2) << j;
            return true;
        } catch (...) {
            return false;
        }
    }
    
    static std::optional<DatasetCache> load(const fs::path &cache_file, const std::string &expected_hash) {
        try {
            if (!fs::exists(cache_file)) return std::nullopt;
            
            std::ifstream f(cache_file);
            if (!f) return std::nullopt;
            
            json j;
            f >> j;
            
            DatasetCache cache;
            cache.version = j.value("version", "1.0");
            cache.dataset_hash = j.value("dataset_hash", std::string(""));
            
            // Vérifier le hash
            if (cache.dataset_hash != expected_hash) {
                std::cout << "⚠️  Cache invalide (hash mismatch), rechargement..." << std::endl;
                std::cout << "   Attendu: " << expected_hash.substr(0, 16) << "..." << std::endl;
                std::cout << "   Trouvé:  " << cache.dataset_hash.substr(0, 16) << "..." << std::endl;
                return std::nullopt;
            }
            
            if (!j.contains("items") || !j["items"].is_array()) {
                return std::nullopt;
            }
            
            for (const auto &ji : j["items"]) {
                DatasetItem item;
                item.name = ji.value("name", "");
                item.is_linked = ji.value("is_linked", false);
                item.text_file = ji.value("text_file", "");
                item.image_file = ji.value("image_file", "");
                item.audio_file = ji.value("audio_file", "");
                item.video_file = ji.value("video_file", "");
                item.w = ji.value("w", 0);
                item.h = ji.value("h", 0);
                cache.items.push_back(std::move(item));
            }
            
            return cache;
        } catch (...) {
            return std::nullopt;
        }
    }
};

// Variante de loadDataset avec cache et gestion RAM depuis config
static inline std::vector<DatasetItem> loadDatasetCached(
    const std::string &root_dir, 
    int target_w = 64, 
    int target_h = 64, 
    int min_modalities = 1,
    const std::string &cache_path = "dataset_cache.json",
    size_t max_ram_mb = 10240,  // Paramètre RAM
    bool lazy_loading = true)    // Flag lazy loading
{
    // Initialiser le gestionnaire RAM
    auto& mgr = DatasetMemoryManager::instance();
    mgr.setMaxRAM(max_ram_mb * 1024ULL * 1024ULL);
    
    std::cout << "\n💾 RAM Manager initialisé:" << std::endl;
    std::cout << "   Limite:      " << max_ram_mb << " MB" << std::endl;
    std::cout << "   Lazy loading: " << (lazy_loading ? "✓" : "✗") << std::endl;
    
    std::cout << "\n🔐 Calcul SHA256 du dataset..." << std::endl;
    std::string dataset_hash = datasetSHA256Hash(root_dir);
    std::cout << "   Hash: " << dataset_hash.substr(0, 16) << "..." << std::endl;
    
    // Essayer de charger depuis le cache
    auto cached = DatasetCache::load(cache_path, dataset_hash);
    if (cached.has_value()) {
        std::cout << "✓ Cache valide trouvé, chargement rapide..." << std::endl;
        
        if (lazy_loading) {
            // Mode lazy: ne rien charger maintenant
            std::cout << "  ⚡ Mode lazy: données seront chargées à la demande" << std::endl;
            std::cout << "  💾 RAM utilisée: 0 MB (métadonnées uniquement)" << std::endl;
        } else {
            // Mode eager: charger les images immédiatement
            std::cout << "  ⏳ Chargement des images..." << std::endl;
            size_t loaded_images = 0;
            size_t failed = 0;
            
            for (auto &item : cached->items) {
                if (!item.image_file.empty()) {
                    // Vérifier si on a assez de RAM
                    if (!mgr.canAllocate(target_w * target_h)) {
                        std::cout << "  ⚠️  Limite RAM atteinte, arrêt du chargement" << std::endl;
                        break;
                    }
                    
                    if (item.loadImage(target_w, target_h)) {
                        loaded_images++;
                    } else {
                        failed++;
                    }
                }
                
                // Progress tous les 100 items
                if ((loaded_images + failed) % 100 == 0) {
                    std::cout << "  Progression: " << loaded_images << " chargées, " 
                             << failed << " échecs, RAM: " 
                             << (mgr.getCurrentRAM() / 1024 / 1024) << " MB" << std::endl;
                }
            }
            
            std::cout << "  ✓ Images chargées: " << loaded_images << std::endl;
            if (failed > 0) {
                std::cout << "  ⚠️  Échecs: " << failed << std::endl;
            }
            mgr.printStats();
        }
        
        return std::move(cached->items);
    }
    
    // Pas de cache ou invalide, chargement normal
    std::cout << "⟳ Indexation du dataset..." << std::endl;
    auto items = loadDataset(root_dir, target_w, target_h, min_modalities);
    
    // Sauvegarder le cache
    DatasetCache cache;
    cache.dataset_hash = dataset_hash;
    cache.items = items;
    
    if (cache.save(cache_path)) {
        std::cout << "✓ Cache sauvegardé: " << cache_path << std::endl;
    } else {
        std::cout << "⚠️  Échec sauvegarde du cache" << std::endl;
    }
    
    return items;
}

#endif // __TENSOR_HELPERS_HPP__