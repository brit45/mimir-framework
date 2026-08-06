#include "Visualizer.hpp"
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <sstream>
#include <chrono>
#include <unordered_map>
#include <cctype>

static const char* kVizUISettingsFile = "viz_ui_settings.json";

namespace {
static inline sf::FloatRect make_rect(float x, float y, float w, float h) {
    return sf::FloatRect(sf::Vector2f(x, y), sf::Vector2f(w, h));
}

struct ParsedVizLabel {
    std::string raw;
    std::string path;
    std::string extra;
    std::vector<std::string> parts;

    std::string model;
    std::string layer_path;
    std::string layer_type;
    std::string tag;        // ex: DS/OUT/VAL/PRE/ACT/?
    std::string headline;   // ex: Dataset / Sortie / Validation / Prétraitement / Activations
    std::string short_text; // ex: unet2d/down_0/conv_in
};

static std::vector<std::string> split(const std::string& s, char delim) {
    std::vector<std::string> out;
    std::string cur;
    cur.reserve(s.size());
    for (const char c : s) {
        if (c == delim) {
            out.push_back(cur);
            cur.clear();
        } else {
            cur.push_back(c);
        }
    }
    out.push_back(cur);
    return out;
}

static std::string trim(const std::string& s) {
    size_t b = 0;
    while (b < s.size() && (s[b] == ' ' || s[b] == '\t' || s[b] == '\n' || s[b] == '\r')) ++b;
    if (b >= s.size()) return std::string();
    size_t e = s.size();
    while (e > b && (s[e - 1] == ' ' || s[e - 1] == '\t' || s[e - 1] == '\n' || s[e - 1] == '\r')) --e;
    return s.substr(b, e - b);
}

static std::string clamp_text_end(const std::string& s, size_t max_chars) {
    if (max_chars == 0) return std::string();
    if (s.size() <= max_chars) return s;
    if (max_chars <= 3) return s.substr(0, max_chars);
    return s.substr(0, max_chars - 3) + "...";
}

static std::string format_decimal(double v, int max_decimals = 8) {
    if (!std::isfinite(v)) {
        if (std::isnan(v)) return "nan";
        return (v > 0.0) ? "inf" : "-inf";
    }

    const double av = std::fabs(v);
    int prec = 0;
    if (av >= 1000.0) prec = 0;
    else if (av >= 100.0) prec = 1;
    else if (av >= 1.0) prec = 3;
    else if (av >= 0.01) prec = 5;
    else prec = std::min(max_decimals, 8);

    std::ostringstream ss;
    ss.setf(std::ios::fixed);
    ss << std::setprecision(std::max(0, prec)) << v;
    std::string s = ss.str();

    if (prec > 0) {
        while (!s.empty() && s.back() == '0') s.pop_back();
        if (!s.empty() && s.back() == '.') s.pop_back();
        if (s.empty()) s = "0";
    }

    return s;
}

static std::string basename_like(const std::string& p) {
    if (p.empty()) return p;
    const size_t pos = p.find_last_of("/\\");
    if (pos == std::string::npos) return p;
    if (pos + 1 >= p.size()) return p;
    return p.substr(pos + 1);
}

static bool contains_part(const std::vector<std::string>& parts, const std::string& needle) {
    return std::any_of(parts.begin(), parts.end(), [&](const std::string& p) { return p == needle; });
}

static std::string join_parts(const std::vector<std::string>& parts, size_t start, size_t end) {
    if (end <= start || start >= parts.size()) return {};
    end = std::min(end, parts.size());
    std::string out;
    for (size_t i = start; i < end; ++i) {
        if (parts[i].empty()) continue;
        if (!out.empty()) out += "/";
        out += parts[i];
    }
    return out;
}

static bool is_norm_type_name(const std::string& t) {
    return t == "BatchNorm1d" || t == "BatchNorm2d" || t == "InstanceNorm2d" ||
           t == "GroupNorm" || t == "LayerNorm" || t == "RMSNorm";
}

static bool is_attention_type_name(const std::string& t) {
    return t == "SelfAttention" || t == "CrossAttention" || t == "MultiHeadAttention";
}

static bool is_upsample_type_name(const std::string& t) {
    return t == "ConvTranspose2d" || t == "UpsampleNearest" || t == "UpsampleBilinear" ||
           t == "UpsampleBicubic" || t == "PixelShuffle";
}

static std::string short_layer_type_name(const std::string& t) {
    if (t == "Conv2d") return "conv";
    if (t == "ConvTranspose2d") return "deconv";
    if (t == "DepthwiseConv2d") return "dwconv";
    if (t == "GroupNorm") return "gn";
    if (t == "LayerNorm") return "ln";
    if (t == "RMSNorm") return "rms";
    if (t == "BatchNorm1d" || t == "BatchNorm2d") return "bn";
    if (t == "InstanceNorm2d") return "in";
    if (t == "SelfAttention") return "attn";
    if (t == "CrossAttention") return "xattn";
    if (t == "MultiHeadAttention") return "mha";
    if (t == "Reparameterize") return "reparam";
    if (t == "UpsampleNearest") return "up";
    if (t == "UpsampleBilinear") return "upbil";
    if (t == "UpsampleBicubic") return "upbic";
    if (t == "PixelShuffle") return "pix";
    if (t == "Add") return "add";
    if (t == "Concat") return "cat";
    if (t == "Permute") return "perm";
    if (t == "Reshape" || t == "View") return "shape";
    if (t == "Identity") return "id";
    return {};
}

static const char* heatmap_palette_name(int palette_id) {
    switch (palette_id) {
        case 1: return "TURBO";
        case 2: return "INFERNO";
        case 3: return "VIRIDIS";
        default: return "CLASSIC";
    }
}

static std::vector<uint8_t> colorize_gray_heatmap_rgb(const std::vector<uint8_t>& gray, int palette_id) {
    std::vector<uint8_t> rgb;
    rgb.resize(gray.size() * 3ULL);
    for (size_t i = 0; i < gray.size(); ++i) {
        const float t = static_cast<float>(gray[i]) / 255.0f;

        float r = 0.0f, g = 0.0f, b = 0.0f;
        if (palette_id == 1) {
            // TURBO (approximation polynomiale compacte)
            const float x = t;
            r = 34.61f + x * (1172.33f + x * (-10793.56f + x * (33300.12f + x * (-38394.49f + x * 14825.05f))));
            g = 23.31f + x * (557.33f + x * (1225.33f + x * (-3574.96f + x * (1073.77f + x * 707.56f))));
            b = 27.2f + x * (3211.1f + x * (-15327.97f + x * (27814.0f + x * (-22569.18f + x * 6838.66f))));
        } else if (palette_id == 2) {
            // INFERNO (anchors)
            const float c0r = 20.f,  c0g = 11.f,  c0b = 52.f;
            const float c1r = 121.f, c1g = 28.f,  c1b = 109.f;
            const float c2r = 219.f, c2g = 82.f,  c2b = 59.f;
            const float c3r = 250.f, c3g = 229.f, c3b = 107.f;
            if (t < 0.33f) {
                const float u = t / 0.33f;
                r = c0r + (c1r - c0r) * u;
                g = c0g + (c1g - c0g) * u;
                b = c0b + (c1b - c0b) * u;
            } else if (t < 0.66f) {
                const float u = (t - 0.33f) / 0.33f;
                r = c1r + (c2r - c1r) * u;
                g = c1g + (c2g - c1g) * u;
                b = c1b + (c2b - c1b) * u;
            } else {
                const float u = (t - 0.66f) / 0.34f;
                r = c2r + (c3r - c2r) * u;
                g = c2g + (c3g - c2g) * u;
                b = c2b + (c3b - c2b) * u;
            }
        } else if (palette_id == 3) {
            // VIRIDIS (anchors)
            const float c0r = 68.f,  c0g = 1.f,   c0b = 84.f;
            const float c1r = 59.f,  c1g = 82.f,  c1b = 139.f;
            const float c2r = 33.f,  c2g = 145.f, c2b = 140.f;
            const float c3r = 253.f, c3g = 231.f, c3b = 37.f;
            if (t < 0.33f) {
                const float u = t / 0.33f;
                r = c0r + (c1r - c0r) * u;
                g = c0g + (c1g - c0g) * u;
                b = c0b + (c1b - c0b) * u;
            } else if (t < 0.66f) {
                const float u = (t - 0.33f) / 0.33f;
                r = c1r + (c2r - c1r) * u;
                g = c1g + (c2g - c1g) * u;
                b = c1b + (c2b - c1b) * u;
            } else {
                const float u = (t - 0.66f) / 0.34f;
                r = c2r + (c3r - c2r) * u;
                g = c2g + (c3g - c2g) * u;
                b = c2b + (c3b - c2b) * u;
            }
        } else {
            // CLASSIC: froid -> neutre -> chaud
            const float neg_r = 70.0f,  neg_g = 120.0f, neg_b = 210.0f;
            const float mid_r = 175.0f, mid_g = 175.0f, mid_b = 175.0f;
            const float pos_r = 220.0f, pos_g = 65.0f,  pos_b = 65.0f;
            if (t < 0.5f) {
                const float u = t / 0.5f;
                r = neg_r + (mid_r - neg_r) * u;
                g = neg_g + (mid_g - neg_g) * u;
                b = neg_b + (mid_b - neg_b) * u;
            } else {
                const float u = (t - 0.5f) / 0.5f;
                r = mid_r + (pos_r - mid_r) * u;
                g = mid_g + (pos_g - mid_g) * u;
                b = mid_b + (pos_b - mid_b) * u;
            }
        }

        const size_t base = i * 3ULL;
        rgb[base + 0] = static_cast<uint8_t>(std::clamp(static_cast<int>(std::lround(r)), 0, 255));
        rgb[base + 1] = static_cast<uint8_t>(std::clamp(static_cast<int>(std::lround(g)), 0, 255));
        rgb[base + 2] = static_cast<uint8_t>(std::clamp(static_cast<int>(std::lround(b)), 0, 255));
    }
    return rgb;
}

static std::string strip_known_block_suffixes(const std::string& s) {
    if (s == "n" || s == "norm") return {};
    if (s == "bot_n") return "bot";
    if (s == "out_n") return "out";
    if (s.size() >= 2 && s[0] == 'n') {
        bool all_digits = true;
        for (size_t i = 1; i < s.size(); ++i) {
            if (!std::isdigit(static_cast<unsigned char>(s[i]))) {
                all_digits = false;
                break;
            }
        }
        if (all_digits) return {};
    }
    const std::array<std::string, 8> suffixes = {
        std::string("_norm"), std::string("_gn"), std::string("_ln"), std::string("_bn"),
        std::string("_in"), std::string("_rms"), std::string("/norm"), std::string("/n")
    };
    for (const auto& suffix : suffixes) {
        if (s.size() > suffix.size() && s.rfind(suffix) == s.size() - suffix.size()) {
            return s.substr(0, s.size() - suffix.size());
        }
    }
    return s;
}

static std::string canonical_path_role(const std::string& s) {
    if (s.empty()) return {};
    if (s == "self_attn" || s == "attn") return "attn";
    if (s == "cross_attn") return "xattn";
    if (s == "multi_attn") return "mha";
    if (s == "lm_head" || s == "classifier") return "head";
    if (s == "out_proj" || s == "text_proj" || s == "img_proj") return "proj";
    if (s == "final_ln" || s == "final_norm") return "final";
    if (s == "recon_to_hwc" || s == "out_to_hwc") return "out";
    if (s == "recon_chw" || s == "recon" || s == "out_concat" || s == "out_pack") return "out";
    if (s == "tok_emb") return "tok";
    if (s == "meanpool") return "pool";

    const std::array<std::pair<const char*, const char*>, 9> suffix_roles = {{
        {"_attn", "attn"},
        {"_proj", "proj"},
        {"_head", "head"},
        {"_out", "out"},
        {"_pool", "pool"},
        {"_norm", "norm"},
        {"_gn", "gn"},
        {"_ln", "ln"},
        {"_fc", "fc"},
    }};
    for (const auto& [suffix, role] : suffix_roles) {
        const std::string suf(suffix);
        if (s.size() > suf.size() && s.rfind(suf) == s.size() - suf.size()) {
            return role;
        }
    }

    return strip_known_block_suffixes(s);
}

static std::string find_label_anchor(const std::vector<std::string>& tail_parts,
                                     const std::string& alias,
                                     const std::string& role) {
    for (size_t i = tail_parts.size(); i > 0; --i) {
        const std::string token = canonical_path_role(tail_parts[i - 1]);
        if (token.empty()) continue;
        if (!alias.empty() && token == alias) continue;
        if (!role.empty() && token == role) continue;
        if (token == "vec" || token == "act" || token == "id") continue;
        return token;
    }
    return {};
}

static std::string format_block_short_text(const std::vector<std::string>& tail_parts,
                                           const std::string& tag,
                                           const std::string& alias,
                                           const std::string& layer_type,
                                           const std::string& fallback) {
    const std::string role = canonical_path_role(tail_parts.empty() ? std::string() : tail_parts.back());
    const std::string anchor = find_label_anchor(tail_parts, alias, role);

    if (tag == "N" && !alias.empty()) {
        return anchor.empty() ? alias : (anchor + "/" + alias);
    }
    if (tag == "AT" && !alias.empty()) {
        return anchor.empty() ? alias : (anchor + "/" + alias);
    }
    if (tag == "LAT" && !alias.empty()) {
        return anchor.empty() ? alias : (anchor + "/" + alias);
    }

    if (role == "proj" || role == "head" || role == "out" || role == "pool") {
        return anchor.empty() ? role : (anchor + "/" + role);
    }
    if (role == "final") {
        if (!alias.empty() && (alias == "ln" || alias == "gn" || alias == "bn" || alias == "rms")) {
            return std::string("final/") + alias;
        }
        return "final";
    }
    if (role == "fc") {
        return anchor.empty() ? std::string("fc") : (anchor + "/fc");
    }

    std::vector<std::string> normalized;
    normalized.reserve(tail_parts.size());
    for (const auto& part : tail_parts) {
        const std::string token = canonical_path_role(part);
        if (!token.empty()) normalized.push_back(token);
    }
    if (!normalized.empty()) {
        const size_t keep = std::min<size_t>(3, normalized.size());
        const size_t from = normalized.size() > keep ? (normalized.size() - keep) : 0;
        return join_parts(normalized, from, normalized.size());
    }

    (void)layer_type;
    return fallback;
}

static ParsedVizLabel parse_viz_label(const std::string& label_raw) {
    static thread_local std::unordered_map<std::string, ParsedVizLabel> label_cache;
    const auto it = label_cache.find(label_raw);
    if (it != label_cache.end()) {
        return it->second;
    }

    ParsedVizLabel out;
    out.raw = label_raw;

    // Convention: une partie "chemin" peut être suivie d'extra après " | ".
    // Ex: "vae_conv/input/dataset/rgb | /path/file.png"
    const std::string s = label_raw;
    const size_t bar = s.find("|");
    if (bar != std::string::npos) {
        out.path = trim(s.substr(0, bar));
        out.extra = trim(s.substr(bar + 1));
    } else {
        out.path = trim(s);
    }
    out.parts = split(out.path, '/');
    // Supprimer les segments vides.
    out.parts.erase(std::remove_if(out.parts.begin(), out.parts.end(), [](const std::string& p) { return p.empty(); }), out.parts.end());

    out.model = out.parts.empty() ? std::string() : out.parts.front();

    auto find_idx = [&](const std::string& v) -> int {
        for (size_t i = 0; i < out.parts.size(); ++i) {
            if (out.parts[i] == v) return static_cast<int>(i);
        }
        return -1;
    };

    const int idx_input = find_idx("input");
    const int idx_output = find_idx("output");
    const int idx_val = std::max(find_idx("validation"), find_idx("val"));
    const int idx_blocks = find_idx("blocks");
    const bool is_activation = (!out.parts.empty() && out.parts.back() == "activation");

    auto is_activation_type = [&](const std::string& t) -> bool {
        // Doit rester simple: uniquement les activations usuelles.
        return t == "ReLU" || t == "LeakyReLU" || t == "GELU" || t == "SiLU" ||
               t == "Tanh" || t == "Sigmoid" || t == "Softmax" || t == "LogSoftmax" ||
               t == "Softplus" || t == "Mish" || t == "HardSigmoid" || t == "HardSwish";
    };

    // Catégorisation heuristique (labels émis par LuaScripting + viz taps Model.cpp)
    // 1) Format viz taps: <model>/blocks/<path>/<LayerType> (ou .../<LayerType>/vec)
    if (idx_blocks >= 0) {
        // Déterminer le type (dernier segment, ou avant-dernier si suffixe /vec)
        std::string t = out.parts.empty() ? std::string() : out.parts.back();
        size_t drop = 1;
        if (t == "vec") {
            drop = 2;
            if (out.parts.size() >= 2) t = out.parts[out.parts.size() - 2];
        }

        out.layer_type = t;

        if (is_activation_type(t)) {
            out.tag = "ACT";
            out.headline = "Activations";
        } else if (is_norm_type_name(t)) {
            out.tag = "N";
            out.headline = "Norm";
        } else if (is_attention_type_name(t)) {
            out.tag = "AT";
            out.headline = "Attention";
        } else if (is_upsample_type_name(t)) {
            out.tag = "UP";
            out.headline = "Upsample";
        } else if (t == "Reparameterize") {
            out.tag = "LAT";
            out.headline = "Latent";
        } else if (t == "Add" || t == "Concat") {
            out.tag = "RES";
            out.headline = "Fusion";
        } else {
            out.tag = "L";
            out.headline = "Layer";
        }

        // Toute entree au format <model>/blocks/<layer>/<type> correspond a
        // un vrai noeud du graphe et porte le badge uniforme (L).
        out.tag = "L";
        out.headline = "Layer";

        const size_t start = static_cast<size_t>(idx_blocks + 1);
        const size_t end = (out.parts.size() > drop) ? (out.parts.size() - drop) : start;
        out.layer_path = join_parts(out.parts, start, end);

        std::vector<std::string> tail_parts;
        for (size_t i = start; i < end; ++i) tail_parts.push_back(out.parts[i]);

        const std::string alias = short_layer_type_name(t);
        out.short_text = format_block_short_text(
            tail_parts,
            out.tag,
            alias,
            t,
            out.layer_path.empty() ? std::string("blocks") : out.layer_path);
        if (out.short_text.empty()) out.short_text = out.layer_path.empty() ? std::string("blocks") : out.layer_path;
    } else if (idx_val >= 0) {
        out.tag = "VAL";
        out.headline = "Validation";
        std::string tail;
        for (size_t i = static_cast<size_t>(idx_val + 1); i < out.parts.size(); ++i) {
            if (!tail.empty()) tail += "/";
            tail += out.parts[i];
        }
        out.short_text = tail.empty() ? std::string("validation") : tail;
    } else if (idx_output >= 0) {
        out.tag = "OUT";
        out.headline = "Sortie";
        std::string tail;
        for (size_t i = static_cast<size_t>(idx_output + 1); i < out.parts.size(); ++i) {
            if (!tail.empty()) tail += "/";
            tail += out.parts[i];
        }
        out.short_text = tail.empty() ? std::string("output") : tail;
    } else if (idx_input >= 0) {
        // input/dataset vs input/* (prétraitement)
        const bool is_dataset = (idx_input + 1 < static_cast<int>(out.parts.size()) && out.parts[static_cast<size_t>(idx_input + 1)] == "dataset") ||
                                contains_part(out.parts, "dataset");
        if (is_dataset) {
            out.tag = "DS";
            out.headline = "Dataset";
        } else {
            out.tag = "PRE";
            out.headline = "Prétraitement";
        }
        std::string tail;
        for (size_t i = static_cast<size_t>(idx_input + 1); i < out.parts.size(); ++i) {
            if (!tail.empty()) tail += "/";
            tail += out.parts[i];
        }
        out.short_text = tail.empty() ? std::string("input") : tail;
    } else if (is_activation) {
        out.tag = "ACT";
        out.headline = "Activations";
        // Format attendu: model/block/component/activation (ou + profond)
        // On enlève model + activation et on garde une queue lisible.
        const size_t n = out.parts.size();
        size_t start = 1; // skip model
        size_t end = (n >= 1) ? (n - 1) : 0; // drop activation
        if (end <= start) {
            out.short_text = "activation";
        } else {
            // Garder jusqu'à 3 segments de fin (plus lisible dans les vignettes)
            const size_t keep = std::min<size_t>(3, end - start);
            const size_t from = end - keep;
            std::string tail;
            for (size_t i = from; i < end; ++i) {
                if (!tail.empty()) tail += "/";
                tail += out.parts[i];
            }
            out.short_text = tail;
        }
    } else {
        out.tag = "?";
        out.headline = "Image";
        // fallback: garder la fin du chemin
        if (!out.parts.empty()) {
            out.short_text = out.parts.back();
        }
    }

    // Amélioration: si extra est un chemin, on garde juste le basename.
    if (!out.extra.empty()) {
        const std::string bn = basename_like(out.extra);
        if (!bn.empty() && bn.size() < out.extra.size()) {
            out.extra = bn;
        }
    }

    if (label_cache.size() > 4096) {
        label_cache.clear();
    }
    label_cache.emplace(label_raw, out);
    return out;
}

static sf::Color color_for_tag(const std::string& tag) {
    if (tag == "DS") return sf::Color(120, 180, 220);
    if (tag == "PRE") return sf::Color(140, 200, 220);
    if (tag == "OUT") return sf::Color(220, 160, 120);
    if (tag == "VAL") return sf::Color(160, 220, 160);
    if (tag == "ACT") return sf::Color(160, 160, 220);
    if (tag == "N") return sf::Color(120, 205, 175);
    if (tag == "AT") return sf::Color(185, 145, 230);
    if (tag == "UP") return sf::Color(235, 170, 105);
    if (tag == "LAT") return sf::Color(230, 125, 165);
    if (tag == "RES") return sf::Color(205, 190, 120);
    if (tag == "L") return sf::Color(180, 180, 200);
    if (tag == "T") return sf::Color(180, 170, 140);
    return sf::Color(140, 140, 150);
}

static void position_sprite_centered_in_box(sf::Sprite& sprite, float x, float y, float box_size) {
    // sprite est déjà mis à l'échelle par createImageTexture().
    const auto lb = sprite.getLocalBounds();
    const auto sc = sprite.getScale();
    const float dw = lb.size.x * sc.x;
    const float dh = lb.size.y * sc.y;
    const float ox = (box_size - dw) * 0.5f;
    const float oy = (box_size - dh) * 0.5f;
    // Ajuster avec left/top au cas où localBounds n'est pas (0,0).
    const float px = x + ox - lb.position.x * sc.x;
    const float py = y + oy - lb.position.y * sc.y;
    // Snap pixel pour éviter les artefacts d'alignement/sub-pixel dans les grilles.
    sprite.setPosition(sf::Vector2f(std::round(px), std::round(py)));
}

static bool is_layer_block_preview_smoothing_candidate(const ParsedVizLabel& p, int channels) {
    if (channels != 1 && channels != 3 && channels != 4) return false;
    if (p.tag == "DS" || p.tag == "PRE" || p.tag == "OUT" || p.tag == "VAL" || p.tag == "ACT") return false;

    const std::string& path = p.layer_path;
    if (path.find("raw_in") != std::string::npos ||
        path.find("in_reshape") != std::string::npos ||
        path.find("recon") != std::string::npos ||
        path.find("to_hwc") != std::string::npos) {
        return false;
    }

    return true;
}

static std::vector<uint8_t> smooth_preview_pixels(const std::vector<uint8_t>& pixels, int w, int h, int channels) {
    if (w <= 2 || h <= 2) return pixels;
    if (channels != 1 && channels != 3 && channels != 4) return pixels;

    std::vector<uint8_t> out = pixels;
    const int color_channels = (channels == 4) ? 3 : channels;

    for (int y = 1; y < h - 1; ++y) {
        for (int x = 1; x < w - 1; ++x) {
            const size_t dst = (static_cast<size_t>(y) * static_cast<size_t>(w) + static_cast<size_t>(x)) * static_cast<size_t>(channels);
            for (int cc = 0; cc < color_channels; ++cc) {
                int acc = 0;
                for (int ky = -1; ky <= 1; ++ky) {
                    for (int kx = -1; kx <= 1; ++kx) {
                        const size_t src = (static_cast<size_t>(y + ky) * static_cast<size_t>(w) + static_cast<size_t>(x + kx)) * static_cast<size_t>(channels) + static_cast<size_t>(cc);
                        acc += static_cast<int>(pixels[src]);
                    }
                }
                out[dst + static_cast<size_t>(cc)] = static_cast<uint8_t>(std::clamp(acc / 9, 0, 255));
            }
            if (channels == 4) {
                out[dst + 3] = pixels[dst + 3];
            }
        }
    }

    return out;
}

static int infer_channels_from_buffer_size(const std::vector<uint8_t>& pixels, int w, int h, int preferred) {
    if (w <= 0 || h <= 0) return 0;
    const size_t area = static_cast<size_t>(w) * static_cast<size_t>(h);
    if (area == 0) return 0;

    const size_t n = pixels.size();
    auto exact = [&](int c) -> bool {
        return n == area * static_cast<size_t>(c);
    };
    auto at_least = [&](int c) -> bool {
        return n >= area * static_cast<size_t>(c);
    };

    // 1) Correspondance exacte prioritaire.
    if (preferred == 1 || preferred == 3 || preferred == 4) {
        if (exact(preferred)) return preferred;
    }
    if (exact(4)) return 4;
    if (exact(3)) return 3;
    if (exact(1)) return 1;

    // 2) Fallback best-effort si le buffer contient plus de données que prévu.
    // Préférer d'abord le format demandé si plausible.
    if (preferred == 1 || preferred == 3 || preferred == 4) {
        if (at_least(preferred)) return preferred;
    }
    if (at_least(4)) return 4;
    if (at_least(3)) return 3;
    if (at_least(1)) return 1;
    return 0;
}

// -----------------------------------------------------------------------------
// Live tuning slider helpers (Metrics panel)
// -----------------------------------------------------------------------------
static constexpr float kLiveLRMin = 1e-7f;
static constexpr float kLiveLRMax = 1e-3f;
static constexpr int kLiveWarmupMax = 50000;

static float clamp01(float t) {
    return std::clamp(t, 0.0f, 1.0f);
}

[[maybe_unused]] static float lr_from_t(float t) {
    const float tt = clamp01(t);
    const float logmin = std::log10(kLiveLRMin);
    const float logmax = std::log10(kLiveLRMax);
    const float v = logmin + (logmax - logmin) * tt;
    return std::pow(10.0f, v);
}

static float t_from_lr(float lr) {
    float v = lr;
    if (!std::isfinite(v)) v = kLiveLRMin;
    v = std::clamp(v, kLiveLRMin, kLiveLRMax);
    const float logmin = std::log10(kLiveLRMin);
    const float logmax = std::log10(kLiveLRMax);
    const float lv = std::log10(v);
    const float denom = std::max(1e-6f, (logmax - logmin));
    return clamp01((lv - logmin) / denom);
}

[[maybe_unused]] static int warmup_from_t(float t) {
    const float tt = clamp01(t);
    const float v = tt * static_cast<float>(kLiveWarmupMax);
    return std::max(0, static_cast<int>(std::lround(v)));
}

static float t_from_warmup(int steps) {
    const int s = std::clamp(steps, 0, kLiveWarmupMax);
    return clamp01(static_cast<float>(s) / static_cast<float>(kLiveWarmupMax));
}

[[maybe_unused]] static float klbeta_from_t(float t) {
    return clamp01(t); // 0..1
}

static float t_from_klbeta(float beta) {
    float b = beta;
    if (!std::isfinite(b)) b = 0.0f;
    return clamp01(b);
}

static uint64_t steady_now_ms() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count());
}

[[maybe_unused]] static bool is_live_input_char(uint32_t cp) {
    if (cp >= static_cast<uint32_t>('0') && cp <= static_cast<uint32_t>('9')) return true;
    return cp == static_cast<uint32_t>('.') ||
           cp == static_cast<uint32_t>('e') ||
           cp == static_cast<uint32_t>('E') ||
           cp == static_cast<uint32_t>('+') ||
           cp == static_cast<uint32_t>('-');
}
} // namespace

Visualizer::Visualizer(const json& config)
    : loss_log_file("checkpoints/loss_history.csv")
    , enabled(false)
    , window_width(1280)
    , window_height(720)
    , window_title("FLUX Model Visualization")
    , fps_limit(60)
    , show_generated_images(true)
    , show_training_progress(true)
    , show_loss_graph(true)
    , graph_history_size(200)
    , image_grid_cols(3)
    , image_grid_rows(1)
    , current_epoch(0)
    , current_total_epochs(0)
    , current_batch(0)
    , current_total_batches(0)
    , current_loss(0.0f)
    , current_avg_loss(0.0f)
    , current_lr(0.0f)
    , current_mse(0.0f)
    , current_kl(0.0f)
    , current_wass(0.0f)
    , current_ent(0.0f)
    , current_mom(0.0f)
    , current_spat(0.0f)
    , current_temp(0.0f)
    , current_timestep(0.0f)
    , current_batch_time_ms(0)
    , current_memory_mb(0)
    , current_bps(0.0f)
    , current_params(0)
    , current_grad_norm(0.0f)
    , current_grad_max(0.0f)
    , current_opt_type(0)
    , current_opt_step(0)
    , current_opt_beta1(0.0f)
    , current_opt_beta2(0.0f)
    , current_opt_eps(0.0f)
    , current_opt_weight_decay(0.0f)
    , current_val_has(false)
    , current_val_ok(false)
    , current_val_in_progress(false)
    , current_val_step(0)
    , current_val_items(0)
    , current_val_done(0)
    , current_val_total(0)
    , current_val_recon(0.0f)
    , current_val_kl(0.0f)
    , current_val_align(0.0f)
{
    if (config.contains("visualization")) {
        auto viz = config["visualization"];
        enabled = viz.value("enabled", false);
        window_width = viz.value("window_width", 1280);
        window_height = viz.value("window_height", 720);
        window_title = viz.value("window_title", "FLUX Model Visualization");
        fps_limit = viz.value("fps_limit", 60);
        show_generated_images = viz.value("show_generated_images", true);
        show_training_progress = viz.value("show_training_progress", true);
        show_loss_graph = viz.value("show_loss_graph", true);
        graph_history_size = viz.value("graph_history_size", 200);
        image_grid_cols = viz.value("image_grid_cols", 3);
        image_grid_rows = viz.value("image_grid_rows", 1);

        // CSV loss: optionnel, peut être remplacé pendant le run via AsyncMonitor::setLossLogFile().
        loss_log_file = viz.value("loss_log_file", loss_log_file);

        hide_activation_blocks = viz.value("hide_activation_blocks", false);
        hide_normalisation_blocks = viz.value("hide_normalisation_blocks", false);
        architecture_path = viz.value("architecture_path", std::string());
    }
    std::cerr << "[viz] config enabled=" << enabled
              << " window=" << window_width << "x" << window_height
              << " title='" << window_title << "'"
              << " loss_log='" << loss_log_file << "'" << std::endl;
}

void Visualizer::setLossLogFile(const std::string& filepath) {
    if (filepath.empty()) return;
    loss_log_file = filepath;
    std::cerr << "[viz] loss_log_file=" << loss_log_file << std::endl;
}

Visualizer::~Visualizer() {
    shutdown();
}

void Visualizer::shutdown() {
    if (pending_loss_log_flush_ && !loss_log_file.empty()) {
        saveLossHistory(loss_log_file);
        pending_loss_log_flush_ = false;
    }
    if (window) {
        if (window->isOpen()) {
            window->close();
        }
        window.reset();
    }
}

bool Visualizer::initialize() {
    if (!enabled) {
        return false;
    }

    try {
        window = std::make_unique<sf::RenderWindow>(
            sf::VideoMode(sf::Vector2u(static_cast<unsigned>(window_width), static_cast<unsigned>(window_height))),
            window_title,
            sf::Style::Titlebar | sf::Style::Close
        );
        // Certains environnements laissent le contexte inactif au démarrage si la
        // fenêtre n'a pas encore reçu d'interaction: forcer l'activation.
        window->setActive(true);

        // Logo du programme + splash au lancement (best-effort)
        logo_loaded_ = false;
        try {
            sf::Image logo_img;
            if (logo_img.loadFromFile("logo.png")) {
                const auto sz = logo_img.getSize();
                if (sz.x > 0 && sz.y > 0) {
                    // Certains WM ignorent les icônes trop grandes ou atypiques.
                    // On force une taille standard (64x64) via downscale nearest-neighbor.
                    const unsigned target = 64;
                    sf::Image icon_img;
                    icon_img.resize(sf::Vector2u(target, target));
                    for (unsigned yy = 0; yy < target; ++yy) {
                        for (unsigned xx = 0; xx < target; ++xx) {
                            const unsigned sx = (sz.x > 0) ? (xx * sz.x) / target : 0;
                            const unsigned sy = (sz.y > 0) ? (yy * sz.y) / target : 0;
                            icon_img.setPixel(sf::Vector2u(static_cast<unsigned int>(xx), static_cast<unsigned int>(yy)), logo_img.getPixel(sf::Vector2u(static_cast<unsigned int>(sx), static_cast<unsigned int>(sy))));
                        }
                    }
                    window->setIcon(icon_img);
                }
                if (logo_texture_.loadFromImage(logo_img)) {
                    logo_sprite_.setTexture(logo_texture_, true);
                    logo_sprite_.setColor(sf::Color(255, 255, 255, 220));
                    logo_loaded_ = true;
                    logo_clock_.restart();
                }
            }
        } catch (...) {
            logo_loaded_ = false;
        }

        window->setFramerateLimit(fps_limit);
        syncUIView();
        first_texture_sync_done_ = false;

        // Best-effort: charger une police système pour afficher les métriques.
        // Priorité: Open Sans. Fallback: polices courantes.
        // Si indisponible, on garde un rendu sans texte.
        font_loaded = false;
        const std::vector<std::string> font_candidates = {
            // Open Sans (différentes distros)
            "/usr/share/fonts/truetype/open-sans/OpenSans-Regular.ttf",
            "/usr/share/fonts/truetype/open-sans/OpenSans.ttf",
            "/usr/share/fonts/truetype/opensans/OpenSans-Regular.ttf",
            "/usr/share/fonts/truetype/opensans/OpenSans.ttf",
            "/usr/share/fonts/truetype/google-fonts/OpenSans-Regular.ttf",
            "/usr/share/fonts/truetype/google-fonts/OpenSans/OpenSans-Regular.ttf",

            // Fallbacks fréquents
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        };
        for (const auto& path : font_candidates) {
            if (font.openFromFile(path)) {
                font_loaded = true;
                break;
            }
        }

        std::cerr << "✓ Fenêtre de visualisation SFML initialisée (" 
                  << window_width << "x" << window_height << ")" << std::endl;

        // Initialiser et tenter de restaurer le dernier layout sauvegardé.
        initDefaultPanelsIfNeeded();
        try {
            json settings;
            if (loadUISettings(settings)) {
                const json* chosen = nullptr;
                const char* chosen_name = nullptr;

                if (settings.contains("last")) {
                    chosen = &settings["last"];
                    chosen_name = "last";
        std::cerr << "[viz] config enabled=" << enabled
                  << " window=" << window_width << "x" << window_height
                  << " title='" << window_title << "'"
                  << " loss_log='" << loss_log_file << "'" << std::endl;
                } else if (settings.contains("slot1")) {
                    // Compat: certains fichiers peuvent exposer un slot nommé "slot1" au top-level.
                    chosen = &settings["slot1"];
                    chosen_name = "slot1";
                } else if (settings.contains("slots") && settings["slots"].is_object() && settings["slots"].contains("1")) {
        std::cerr << "[viz] loss_log_file=" << loss_log_file << std::endl;
                    chosen = &settings["slots"]["1"];
                    chosen_name = "slots[1]";
                }

            std::cerr << "[viz] initialize skipped (disabled)" << std::endl;
                if (chosen && chosen->is_object()) {
                    // Appliquer la taille de fenêtre AVANT le layout (sinon clamp sur mauvaise taille).
                    try {
                        if (window && chosen->contains("window_width") && chosen->contains("window_height")) {
                            int ww = 0;
                            int hh = 0;
                            try {
                                ww = (*chosen)["window_width"].get<int>();
                                hh = (*chosen)["window_height"].get<int>();
                            } catch (...) {
                                ww = (int)(*chosen)["window_width"].get<double>();
                                hh = (int)(*chosen)["window_height"].get<double>();
                            }

                            ww = std::max(640, ww);
                            hh = std::max(480, hh);
                            window->setSize(sf::Vector2u(static_cast<unsigned>(ww), static_cast<unsigned>(hh)));
                            window_width = ww;
                            window_height = hh;
                            syncUIView();
                        }
                    } catch (...) {
                        // ignore resize errors
                    }

                    const bool ok = applyUILayout(*chosen);
                    if (ok && chosen_name) {
                        std::cerr << "✓ Layout appliqué par défaut (" << chosen_name << ")" << std::endl;
                    }
                }
            }
        } catch (...) {
        }
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Erreur lors de l'initialisation SFML: " << e.what() << std::endl;
        return false;
    }
}

json Visualizer::serializeUILayout() const {
    json out;
    out["version"] = 2;
    out["show_generated_images"] = show_generated_images;
    out["show_training_progress"] = show_training_progress;
    out["show_loss_graph"] = show_loss_graph;
    out["show_prompt_text"] = show_prompt_text_;
    out["heatmap_palette"] = static_cast<int>(heatmap_palette_);
    out["hide_activation_blocks"] = hide_activation_blocks;
    out["hide_normalisation_blocks"] = hide_normalisation_blocks;
    out["blocks_tips_visible_limit"] = std::max(0, blocks_tips_visible_limit_);

    // Taille de fenêtre (utile pour restaurer un slot de layout)
    out["window_width"] = window_width;
    out["window_height"] = window_height;

    json panels = json::array();
    for (size_t i = 0; i < static_cast<size_t>(PanelId::Count); ++i) {
        const PanelId id = static_cast<PanelId>(i);
        const auto& p = panels_[i];
        json jp;
        jp["id"] = static_cast<int>(id);
        jp["title"] = p.title;
        jp["x"] = p.pos.x;
        jp["y"] = p.pos.y;
        jp["w"] = p.size.x;
        jp["h"] = p.size.y;
        jp["visible"] = p.visible;
        jp["allow_drag"] = p.allow_drag;
        panels.push_back(std::move(jp));
    }
    out["panels"] = std::move(panels);
    return out;
}

bool Visualizer::applyUILayout(const json& layout) {
    initDefaultPanelsIfNeeded();

    try {
        if (layout.contains("show_generated_images")) show_generated_images = layout["show_generated_images"].get<bool>();
        if (layout.contains("show_training_progress")) show_training_progress = layout["show_training_progress"].get<bool>();
        if (layout.contains("show_loss_graph")) show_loss_graph = layout["show_loss_graph"].get<bool>();
        if (layout.contains("show_prompt_text")) show_prompt_text_ = layout["show_prompt_text"].get<bool>();
        if (layout.contains("hide_activation_blocks")) hide_activation_blocks = layout["hide_activation_blocks"].get<bool>();
        if (layout.contains("hide_normalisation_blocks")) hide_normalisation_blocks = layout["hide_normalisation_blocks"].get<bool>();
        if (layout.contains("blocks_tips_visible_limit")) {
            blocks_tips_visible_limit_ = std::max(0, layout["blocks_tips_visible_limit"].get<int>());
        }
        if (layout.contains("heatmap_palette")) {
            const int pid = std::clamp(layout["heatmap_palette"].get<int>(), 0, 3);
            heatmap_palette_ = static_cast<Visualizer::HeatmapPalette>(pid);
        }

        if (!layout.contains("panels") || !layout["panels"].is_array()) {
            // Layout incomplet: appliquer seulement les toggles.
            panels_[static_cast<size_t>(PanelId::Generated)].visible = show_generated_images;
            panels_[static_cast<size_t>(PanelId::Training)].visible = show_training_progress;
            panels_[static_cast<size_t>(PanelId::Graph)].visible = show_loss_graph;
            return true;
        }

        for (const auto& jp : layout["panels"]) {
            if (!jp.is_object()) continue;
            if (!jp.contains("id")) continue;
            const int raw_id = jp["id"].get<int>();
            if (raw_id < 0 || raw_id >= static_cast<int>(PanelId::Count)) continue;
            const size_t idx = static_cast<size_t>(raw_id);
            auto& p = panels_[idx];

            if (jp.contains("title") && jp["title"].is_string()) p.title = jp["title"].get<std::string>();
            if (jp.contains("x")) p.pos.x = jp["x"].get<float>();
            if (jp.contains("y")) p.pos.y = jp["y"].get<float>();
            if (jp.contains("w")) p.size.x = std::max(kPanelMinW, jp["w"].get<float>());
            if (jp.contains("h")) p.size.y = std::max(kPanelMinH, jp["h"].get<float>());
            if (jp.contains("visible")) p.visible = jp["visible"].get<bool>();
            if (jp.contains("allow_drag")) p.allow_drag = jp["allow_drag"].get<bool>();
        }

        // Visibilité liée aux toggles (reste la source de vérité)
        panels_[static_cast<size_t>(PanelId::Generated)].visible = show_generated_images;
        panels_[static_cast<size_t>(PanelId::Training)].visible = show_training_progress;
        panels_[static_cast<size_t>(PanelId::Graph)].visible = show_loss_graph;

        clampPanelsToWindow();
        rebuildAllTextures();
        return true;
    } catch (...) {
        return false;
    }
}

bool Visualizer::loadUISettings(json& out) const {
    out = json::object();
    try {
        std::ifstream f(kVizUISettingsFile);
        if (!f) return false;
        f >> out;
        if (!out.is_object()) {
            out = json::object();
            return false;
        }
        return true;
    } catch (...) {
        out = json::object();
        return false;
    }
}

bool Visualizer::saveUISettings(const json& s) const {
    try {
        std::ofstream f(kVizUISettingsFile);
        if (!f) return false;
        f << std::setw(2) << s << std::endl;
        return true;
    } catch (...) {
        return false;
    }
}

void Visualizer::saveUILayoutToLast() {
    json settings;
    loadUISettings(settings);
    if (!settings.is_object()) settings = json::object();
    settings["version"] = 1;
    settings["last"] = serializeUILayout();
    saveUISettings(settings);
    std::cerr << "✓ Layout sauvegardé (last) -> " << kVizUISettingsFile << std::endl;
}

void Visualizer::saveUILayoutToSlot(int slot) {
    slot = std::clamp(slot, 0, 9);
    json settings;
    loadUISettings(settings);
    if (!settings.is_object()) settings = json::object();
    settings["version"] = 1;
    if (!settings.contains("slots") || !settings["slots"].is_object()) {
        settings["slots"] = json::object();
    }
    settings["slots"][std::to_string(slot)] = serializeUILayout();
    saveUISettings(settings);
    std::cerr << "✓ Layout sauvegardé (slot " << slot << ") -> " << kVizUISettingsFile << std::endl;
}

void Visualizer::loadUILayoutFromSlot(int slot) {
    slot = std::clamp(slot, 0, 9);
    json settings;
    if (!loadUISettings(settings)) {
        std::cerr << "⚠️  Aucun settings UI trouvé: " << kVizUISettingsFile << std::endl;
        return;
    }
    try {
        if (!settings.contains("slots") || !settings["slots"].is_object()) {
            std::cerr << "⚠️  Aucun slot dans settings UI" << std::endl;
            return;
        }
        const std::string key = std::to_string(slot);
        if (!settings["slots"].contains(key)) {
            std::cerr << "⚠️  Slot UI inexistant: " << slot << std::endl;
            return;
        }

        const json& layout = settings["slots"][key];

        // Appliquer la taille de fenêtre du slot AVANT le layout (sinon clamp sur mauvaise taille).
        try {
            if (window && layout.is_object() && layout.contains("window_width") && layout.contains("window_height")) {
                int ww = 0;
                int hh = 0;
                try {
                    ww = layout["window_width"].get<int>();
                    hh = layout["window_height"].get<int>();
                } catch (...) {
                    // tolerate float/json numbers
                    ww = (int)layout["window_width"].get<double>();
                    hh = (int)layout["window_height"].get<double>();
                }

                ww = std::max(640, ww);
                hh = std::max(480, hh);
                window->setSize(sf::Vector2u(static_cast<unsigned>(ww), static_cast<unsigned>(hh)));
                window_width = ww;
                window_height = hh;
                syncUIView();
            }
        } catch (...) {
            // ignore resize errors
        }

        const bool ok = applyUILayout(layout);
        if (ok) {
            std::cerr << "✓ Layout chargé (slot " << slot << ")" << std::endl;
        } else {
            std::cerr << "⚠️  Échec application layout (slot " << slot << ")" << std::endl;
        }
    } catch (...) {
        std::cerr << "⚠️  Settings UI invalides" << std::endl;
    }
}

void Visualizer::syncUIView() {
    if (!window) return;
    // View UI en coordonnées pixels: (0..w, 0..h)
    // On la force explicitement pour éviter toute déformation au resize.
    const float ww = static_cast<float>(std::max(1, window_width));
    const float hh = static_cast<float>(std::max(1, window_height));
    sf::View v(make_rect(0.f, 0.f, ww, hh));
    v.setViewport(make_rect(0.f, 0.f, 1.f, 1.f));
    window->setView(v);
}

bool Visualizer::isOpen() const {
    return window && window->isOpen();
}

bool Visualizer::isPanelVisible(PanelId id) const {
    return panels_[static_cast<size_t>(id)].visible;
}

namespace {
sf::Color with_alpha(sf::Color c, uint8_t a) {
    c.a = a;
    return c;
}
} // namespace

sf::Color Visualizer::panelAccent(PanelId id) const {
    switch (id) {
        case PanelId::Context: return sf::Color(70, 130, 200, 255);
        case PanelId::Blocks: return sf::Color(160, 110, 210, 255);
        case PanelId::Generated: return sf::Color(70, 170, 170, 255);
        case PanelId::Training: return sf::Color(90, 180, 100, 255);
        case PanelId::Metrics: return sf::Color(210, 150, 70, 255);
        case PanelId::Graph: return sf::Color(200, 90, 90, 255);
        default: return sf::Color(140, 140, 150, 255);
    }
}

sf::FloatRect Visualizer::panelRect(PanelId id) const {
    const auto& p = panels_[static_cast<size_t>(id)];
    return make_rect(p.pos.x, p.pos.y, p.size.x, p.size.y);
}

sf::FloatRect Visualizer::panelTitleRect(PanelId id) const {
    const auto& p = panels_[static_cast<size_t>(id)];
    return make_rect(p.pos.x, p.pos.y, p.size.x, std::min(kPanelTitleH, p.size.y));
}

sf::FloatRect Visualizer::panelContentRect(PanelId id) const {
    const auto& p = panels_[static_cast<size_t>(id)];
    const float top = p.pos.y + std::min(kPanelTitleH, p.size.y);
    const float inner_w = std::max(0.f, p.size.x - 2.f * kPanelPad);
    const float inner_h = std::max(0.f, p.size.y - std::min(kPanelTitleH, p.size.y) - 2.f * kPanelPad);
    return make_rect(p.pos.x + kPanelPad, top + kPanelPad, inner_w, inner_h);
}

sf::FloatRect Visualizer::panelResizeHandleRect(PanelId id) const {
    const auto& p = panels_[static_cast<size_t>(id)];
    const float w = std::max(0.f, p.size.x);
    const float h = std::max(0.f, p.size.y);
    const float hs = std::min(kPanelResizeHandle, std::min(w, h));
    return make_rect(p.pos.x + w - hs, p.pos.y + h - hs, hs, hs);
}

sf::FloatRect Visualizer::panelCloseButtonRect(PanelId id) const {
    const auto tr = panelTitleRect(id);
    const float s = std::max(10.f, std::min(18.f, tr.size.y - 4.f));
    const float x = tr.position.x + tr.size.x - s - 4.f;
    const float y = tr.position.y + (tr.size.y - s) * 0.5f;
    return make_rect(x, y, s, s);
}

std::optional<Visualizer::PanelId> Visualizer::hitTestPanelTitle(const sf::Vector2f& mouse) const {
    // Hit-test du haut vers le bas: si des panneaux se chevauchent,
    // le dernier dessiné (graph/metrics) peut être prioritaire.
    const PanelId order[] = { PanelId::Graph, PanelId::Metrics, PanelId::Training, PanelId::Generated, PanelId::Blocks, PanelId::Context };
    for (PanelId id : order) {
        if (!isPanelVisible(id)) continue;
        const auto& p = panels_[static_cast<size_t>(id)];
        if (!p.allow_drag) continue;
        if (panelTitleRect(id).contains(mouse)) return id;
    }
    return std::nullopt;
}

std::optional<Visualizer::PanelId> Visualizer::hitTestPanelResizeHandle(const sf::Vector2f& mouse) const {
    const PanelId order[] = { PanelId::Graph, PanelId::Metrics, PanelId::Training, PanelId::Generated, PanelId::Blocks, PanelId::Context };
    for (PanelId id : order) {
        if (!isPanelVisible(id)) continue;
        const auto& p = panels_[static_cast<size_t>(id)];
        if (!p.allow_drag) continue;
        if (panelResizeHandleRect(id).contains(mouse)) return id;
    }
    return std::nullopt;
}

std::optional<Visualizer::PanelId> Visualizer::hitTestPanelCloseButton(const sf::Vector2f& mouse) const {
    const PanelId order[] = { PanelId::Graph, PanelId::Metrics, PanelId::Training, PanelId::Generated, PanelId::Blocks, PanelId::Context };
    for (PanelId id : order) {
        if (!isPanelVisible(id)) continue;
        if (panelCloseButtonRect(id).contains(mouse)) return id;
    }
    return std::nullopt;
}

void Visualizer::initCursorsIfNeeded() {
    if (cursors_loaded_) return;
    // Best-effort: certains backends peuvent refuser certains curseurs.
    cursor_arrow_ = sf::Cursor::createFromSystem(sf::Cursor::Type::Arrow);
    cursor_hand_ = sf::Cursor::createFromSystem(sf::Cursor::Type::Hand);
    cursor_cross_ = sf::Cursor::createFromSystem(sf::Cursor::Type::Cross);
    // Diagonal resize (NW-SE) pour handle bottom-right.
    cursor_resize_ = sf::Cursor::createFromSystem(sf::Cursor::Type::SizeTopLeftBottomRight);

    cursor_ok_arrow_ = cursor_arrow_.has_value();
    cursor_ok_hand_ = cursor_hand_.has_value();
    cursor_ok_cross_ = cursor_cross_.has_value();
    cursor_ok_resize_ = cursor_resize_.has_value();
    cursors_loaded_ = true;
}

void Visualizer::setCursor(CursorKind kind) {
    if (!window) return;
    if (cursor_kind_ == kind) return;
    initCursorsIfNeeded();

    const sf::Cursor* c = nullptr;
    if (kind == CursorKind::Hand && cursor_ok_hand_ && cursor_hand_.has_value()) c = &(*cursor_hand_);
    else if (kind == CursorKind::Cross && cursor_ok_cross_ && cursor_cross_.has_value()) c = &(*cursor_cross_);
    else if (kind == CursorKind::Resize && cursor_ok_resize_ && cursor_resize_.has_value()) c = &(*cursor_resize_);
    else if (cursor_ok_arrow_ && cursor_arrow_.has_value()) c = &(*cursor_arrow_);

    if (c) {
        window->setMouseCursor(*c);
    }
    cursor_kind_ = kind;
}

void Visualizer::clampPanelsToWindow() {
    const float min_margin = 4.f;
    for (auto& p : panels_) {
        if (p.size.x <= 0.f || p.size.y <= 0.f) continue;
        const float max_x = std::max(min_margin, static_cast<float>(window_width) - p.size.x - min_margin);
        const float max_y = std::max(min_margin, static_cast<float>(window_height) - p.size.y - min_margin);
        p.pos.x = std::clamp(p.pos.x, min_margin, max_x);
        p.pos.y = std::clamp(p.pos.y, min_margin, max_y);
    }
}

void Visualizer::initDefaultPanelsIfNeeded() {
    if (panels_initialized_) return;
    panels_initialized_ = true;

    const float margin = 20.f;
    const float right_w = 520.f;
    const float min_left_w = 520.f;

    const float win_w = static_cast<float>(window_width);
    const float win_h = static_cast<float>(window_height);

    const float left_w = std::clamp(win_w - right_w - 3.f * margin, min_left_w, std::max(min_left_w, win_w - 2.f * margin));
    const float right_x = margin + left_w + margin;
    const float right_safe_w = std::max(320.f, std::min(right_w, win_w - right_x - margin));

    panels_[static_cast<size_t>(PanelId::Context)] = Panel{ sf::Vector2f(margin, margin), sf::Vector2f(left_w, 280.f), "Context", true, true };
    panels_[static_cast<size_t>(PanelId::Blocks)] = Panel{ sf::Vector2f(margin, margin + 300.f), sf::Vector2f(left_w, 320.f), "Blocks / Layers", true, true };
    panels_[static_cast<size_t>(PanelId::Generated)] = Panel{ sf::Vector2f(margin, margin + 640.f), sf::Vector2f(left_w, 280.f), "Generated", true, true };

    panels_[static_cast<size_t>(PanelId::Training)] = Panel{ sf::Vector2f(right_x, margin), sf::Vector2f(right_safe_w, 70.f), "Training", true, true };
    panels_[static_cast<size_t>(PanelId::Graph)] = Panel{ sf::Vector2f(right_x, std::max(margin + 80.f, win_h - 240.f - margin)), sf::Vector2f(right_safe_w, 240.f), "Loss", true, true };
    {
        const float metrics_y = margin + 90.f;
        const float metrics_h = std::max(220.f, win_h - metrics_y - panels_[static_cast<size_t>(PanelId::Graph)].size.y - margin);
        panels_[static_cast<size_t>(PanelId::Metrics)] = Panel{ sf::Vector2f(right_x, metrics_y), sf::Vector2f(right_safe_w, metrics_h), "Metrics", true, true };
    }

    clampPanelsToWindow();
}

void Visualizer::drawPanelChrome(PanelId id) {
    if (!window) return;
    if (!isPanelVisible(id)) return;
    const auto& p = panels_[static_cast<size_t>(id)];
    if (p.size.x <= 1.f || p.size.y <= 1.f) return;

    const sf::Color accent = panelAccent(id);

    sf::RectangleShape bg(p.size);
    bg.setPosition(p.pos);
    bg.setFillColor(sf::Color(18, 18, 22, 170));
    bg.setOutlineColor(with_alpha(accent, 180));
    bg.setOutlineThickness(2);
    window->draw(bg);

    const float th = std::min(kPanelTitleH, p.size.y);
    sf::RectangleShape title(sf::Vector2f(p.size.x, th));
    title.setPosition(p.pos);
    title.setFillColor(with_alpha(accent, 120));
    window->draw(title);

    if (id == PanelId::Blocks) {
        last_blocks_hide_act_box_.reset();
        last_blocks_hide_norm_box_.reset();
        last_blocks_tips_limit_box_.reset();
    }

    if (font_loaded && !p.title.empty()) {
        sf::Text t(font);
        t.setFont(font);
        t.setCharacterSize(14);
        t.setFillColor(sf::Color(235, 235, 240));
        t.setPosition(sf::Vector2f(p.pos.x + 8.f, p.pos.y + 2.f));
        t.setString(sf::String::fromUtf8(p.title.begin(), p.title.end()));
        window->draw(t);

        // Badge de mode de rendu pour Blocks/Layers (heatmap vs réel).
        if (id == PanelId::Blocks) {
            const bool real_mode = !heatmap_mode_;
            const std::string badge = real_mode ? "REEL" : "HEATMAP";

            const sf::Color badge_fill = real_mode
                ? sf::Color(46, 122, 86, 220)
                : sf::Color(126, 78, 42, 220);
            const sf::Color badge_outline = real_mode
                ? sf::Color(126, 220, 170, 235)
                : sf::Color(234, 182, 120, 235);

            const bool smooth_on = smooth_layer_block_previews_;
            const std::string smooth_badge = smooth_on ? "LISSAGE ON" : "LISSAGE OFF";
            const sf::Color smooth_fill = smooth_on
                ? sf::Color(54, 104, 152, 220)
                : sf::Color(92, 92, 102, 220);
            const sf::Color smooth_outline = smooth_on
                ? sf::Color(140, 206, 255, 235)
                : sf::Color(156, 156, 170, 235);

            const std::string pal_badge = std::string("PAL ") + heatmap_palette_name(static_cast<int>(heatmap_palette_));
            const sf::Color pal_fill = sf::Color(72, 78, 110, 220);
            const sf::Color pal_outline = sf::Color(160, 172, 238, 235);

            const float badge_h = std::max(14.f, th - 8.f);
            const float mode_w = (badge == "HEATMAP") ? 84.f : 58.f;
            const float smooth_w = 98.f;
            const float pal_w = 100.f;
            const float filt_act_w = 84.f;
            const float filt_norm_w = 92.f;
            const float tips_w = 98.f;
            const float gap = 6.f;
            const float total_w = mode_w + gap + smooth_w + gap + pal_w + gap + filt_act_w + gap + filt_norm_w + gap + tips_w;
            const float base_x = p.pos.x + std::max(8.f, p.size.x - total_w - 28.f);
            const float badge_y = p.pos.y + (th - badge_h) * 0.5f;

            const float mode_x = base_x;
            const float smooth_x = base_x + mode_w + gap;
            const float pal_x = smooth_x + smooth_w + gap;
            const float act_x = pal_x + pal_w + gap;
            const float norm_x = act_x + filt_act_w + gap;
            const float tips_x = norm_x + filt_norm_w + gap;

            sf::RectangleShape b(sf::Vector2f(mode_w, badge_h));
            b.setPosition(sf::Vector2f(mode_x, badge_y));
            b.setFillColor(badge_fill);
            b.setOutlineColor(badge_outline);
            b.setOutlineThickness(1.f);
            window->draw(b);

            sf::Text bt(font);
            bt.setFont(font);
            bt.setCharacterSize(12);
            bt.setFillColor(sf::Color(245, 245, 248));
            bt.setPosition(sf::Vector2f(mode_x + 7.f, badge_y - 1.f));
            bt.setString(sf::String::fromUtf8(badge.begin(), badge.end()));
            window->draw(bt);

            sf::RectangleShape sb(sf::Vector2f(smooth_w, badge_h));
            sb.setPosition(sf::Vector2f(smooth_x, badge_y));
            sb.setFillColor(smooth_fill);
            sb.setOutlineColor(smooth_outline);
            sb.setOutlineThickness(1.f);
            window->draw(sb);

            sf::Text st(font);
            st.setFont(font);
            st.setCharacterSize(12);
            st.setFillColor(sf::Color(245, 245, 248));
            st.setPosition(sf::Vector2f(smooth_x + 6.f, badge_y - 1.f));
            st.setString(sf::String::fromUtf8(smooth_badge.begin(), smooth_badge.end()));
            window->draw(st);

            sf::RectangleShape pb(sf::Vector2f(pal_w, badge_h));
            pb.setPosition(sf::Vector2f(pal_x, badge_y));
            pb.setFillColor(pal_fill);
            pb.setOutlineColor(pal_outline);
            pb.setOutlineThickness(1.f);
            window->draw(pb);

            sf::Text pt(font);
            pt.setFont(font);
            pt.setCharacterSize(12);
            pt.setFillColor(sf::Color(245, 245, 248));
            pt.setPosition(sf::Vector2f(pal_x + 6.f, badge_y - 1.f));
            pt.setString(sf::String::fromUtf8(pal_badge.begin(), pal_badge.end()));
            window->draw(pt);

            auto draw_filter_badge = [&](float x, float w, const std::string& label, bool hidden,
                                         std::optional<sf::FloatRect>& out_rect) {
                const sf::Color fill = hidden ? sf::Color(145, 84, 84, 220)
                                              : sf::Color(70, 116, 86, 220);
                const sf::Color outline = hidden ? sf::Color(240, 154, 154, 235)
                                                 : sf::Color(154, 226, 186, 235);
                sf::RectangleShape fb(sf::Vector2f(w, badge_h));
                fb.setPosition(sf::Vector2f(x, badge_y));
                fb.setFillColor(fill);
                fb.setOutlineColor(outline);
                fb.setOutlineThickness(1.f);
                window->draw(fb);

                sf::Text ft(font);
                ft.setFont(font);
                ft.setCharacterSize(12);
                ft.setFillColor(sf::Color(245, 245, 248));
                ft.setPosition(sf::Vector2f(x + 6.f, badge_y - 1.f));
                ft.setString(sf::String::fromUtf8(label.begin(), label.end()));
                window->draw(ft);

                out_rect = make_rect(x, badge_y, w, badge_h);
            };

            draw_filter_badge(act_x, filt_act_w,
                              hide_activation_blocks ? "ACT OFF" : "ACT ON",
                              hide_activation_blocks,
                              last_blocks_hide_act_box_);
            draw_filter_badge(norm_x, filt_norm_w,
                              hide_normalisation_blocks ? "NORM OFF" : "NORM ON",
                              hide_normalisation_blocks,
                              last_blocks_hide_norm_box_);

            {
                const sf::Color tips_fill = blocks_tips_limit_editing_
                    ? sf::Color(76, 98, 146, 220)
                    : sf::Color(74, 74, 92, 220);
                const sf::Color tips_outline = blocks_tips_limit_editing_
                    ? sf::Color(166, 208, 255, 235)
                    : sf::Color(168, 168, 188, 235);

                sf::RectangleShape tb(sf::Vector2f(tips_w, badge_h));
                tb.setPosition(sf::Vector2f(tips_x, badge_y));
                tb.setFillColor(tips_fill);
                tb.setOutlineColor(tips_outline);
                tb.setOutlineThickness(1.f);
                window->draw(tb);

                std::string tips_value;
                if (blocks_tips_limit_editing_) {
                    tips_value = blocks_tips_limit_input_;
                } else if (blocks_tips_visible_limit_ > 0) {
                    tips_value = std::to_string(blocks_tips_visible_limit_);
                } else {
                    tips_value = "ALL";
                }
                if (tips_value.empty()) tips_value = "_";

                const std::string tips_badge = "TIPS " + tips_value;

                sf::Text tt(font);
                tt.setFont(font);
                tt.setCharacterSize(12);
                tt.setFillColor(sf::Color(245, 245, 248));
                tt.setPosition(sf::Vector2f(tips_x + 6.f, badge_y - 1.f));
                tt.setString(sf::String::fromUtf8(tips_badge.begin(), tips_badge.end()));
                window->draw(tt);

                last_blocks_tips_limit_box_ = make_rect(tips_x, badge_y, tips_w, badge_h);
            }
        }
    }

    // Poignée de redimensionnement (bas-droite)
    {
        const auto r = panelResizeHandleRect(id);
        sf::RectangleShape h(sf::Vector2f(r.size.x, r.size.y));
        h.setPosition(sf::Vector2f(r.position.x, r.position.y));
        h.setFillColor(sf::Color(80, 80, 95, 210));
        window->draw(h);
    }

    // Bouton close [X]
    {
        const auto r = panelCloseButtonRect(id);
        sf::RectangleShape b(sf::Vector2f(r.size.x, r.size.y));
        b.setPosition(sf::Vector2f(r.position.x, r.position.y));
        b.setFillColor(sf::Color(90, 40, 40, 220));
        b.setOutlineColor(sf::Color(180, 80, 80, 230));
        b.setOutlineThickness(1);
        window->draw(b);

        if (font_loaded) {
            sf::Text t(font);
            t.setFont(font);
            t.setCharacterSize(14);
            t.setFillColor(sf::Color(245, 240, 240));
            t.setPosition(sf::Vector2f(r.position.x + 4.f, r.position.y - 2.f));
            t.setString("X");
            window->draw(t);
        }
    }
}

void Visualizer::processEvents() {
    if (!window) return;

    // Garantir que les coordonnées UI (pixels) sont actives avant tout hit-test.
    syncUIView();

    auto publish_live_params = [&](bool bump_version) {
        live_lr_.store(std::clamp(live_ui_lr_, kLiveLRMin, kLiveLRMax), std::memory_order_relaxed);
        live_lr_warmup_steps_.store(std::max(0, live_ui_lr_warmup_steps_), std::memory_order_relaxed);
        live_kl_beta_.store(std::clamp(live_ui_kl_beta_, 0.0f, 1.0f), std::memory_order_relaxed);
        live_kl_warmup_steps_.store(std::max(0, live_ui_kl_warmup_steps_), std::memory_order_relaxed);
        live_kl_enabled_.store(live_ui_kl_enabled_, std::memory_order_relaxed);
        if (bump_version) {
            live_params_version_.fetch_add(1, std::memory_order_relaxed);
        }
    };

    auto apply_live_slider_from_mouse = [&](LiveDragTarget target, const sf::Vector2f& mouse) {
        auto apply_t_on_track = [&](const std::optional<sf::FloatRect>& track, auto fn) {
            if (!track.has_value()) return;
            const float denom = std::max(1.0f, track->size.x);
            const float t = clamp01((mouse.x - track->position.x) / denom);
            fn(t);
        };

        if (target == LiveDragTarget::LR) {
            apply_t_on_track(last_live_lr_track_, [&](float t) { live_ui_lr_ = lr_from_t(t); });
        } else if (target == LiveDragTarget::LRWarmup) {
            apply_t_on_track(last_live_lrwu_track_, [&](float t) { live_ui_lr_warmup_steps_ = warmup_from_t(t); });
        } else if (target == LiveDragTarget::KLBeta) {
            apply_t_on_track(last_live_klb_track_, [&](float t) { live_ui_kl_beta_ = klbeta_from_t(t); });
        } else if (target == LiveDragTarget::KLWarmup) {
            apply_t_on_track(last_live_klwu_track_, [&](float t) { live_ui_kl_warmup_steps_ = warmup_from_t(t); });
        }
    };

    auto trim_local = [](const std::string& s) -> std::string {
        size_t a = 0;
        while (a < s.size() && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
        size_t b = s.size();
        while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
        return s.substr(a, b - a);
    };

    auto commit_blocks_tips_limit = [&]() {
        const std::string trimmed = trim_local(blocks_tips_limit_input_);
        if (trimmed.empty() || trimmed == "ALL" || trimmed == "all" || trimmed == "0") {
            blocks_tips_visible_limit_ = 0;
        } else {
            try {
                const int v = std::stoi(trimmed);
                blocks_tips_visible_limit_ = std::max(0, v);
            } catch (...) {
                // Entrée invalide: conserver l'ancienne valeur.
            }
        }
        blocks_tips_limit_editing_ = false;
        blocks_tips_limit_input_.clear();
        saveUILayoutToLast();
    };

    auto commit_live_input = [&]() {
        if (live_input_target_ == LiveInputTarget::None) return;
        const std::string trimmed = trim_local(live_input_buffer_);
        bool ok = false;

        try {
            if (live_input_target_ == LiveInputTarget::LR) {
                const float v = std::stof(trimmed);
                if (std::isfinite(v)) {
                    live_ui_lr_ = std::clamp(v, kLiveLRMin, kLiveLRMax);
                    ok = true;
                }
            } else if (live_input_target_ == LiveInputTarget::KLBeta) {
                const float v = std::stof(trimmed);
                if (std::isfinite(v)) {
                    live_ui_kl_beta_ = std::clamp(v, 0.0f, 1.0f);
                    ok = true;
                }
            } else if (live_input_target_ == LiveInputTarget::LRWarmup) {
                const int v = std::stoi(trimmed);
                live_ui_lr_warmup_steps_ = std::clamp(v, 0, kLiveWarmupMax);
                ok = true;
            } else if (live_input_target_ == LiveInputTarget::KLWarmup) {
                const int v = std::stoi(trimmed);
                live_ui_kl_warmup_steps_ = std::clamp(v, 0, kLiveWarmupMax);
                ok = true;
            }
        } catch (...) {
            ok = false;
        }

        if (ok) {
            live_overrides_enabled_.store(true, std::memory_order_relaxed);
            publish_live_params(true);
            live_input_target_ = LiveInputTarget::None;
            live_input_buffer_.clear();
            live_input_error_until_ms_ = 0;
        } else {
            live_input_error_until_ms_ = steady_now_ms() + 900;
        }
    };

    while (const std::optional<sf::Event> event_opt = window->pollEvent()) {
        const sf::Event& event = *event_opt;

        if (event.is<sf::Event::Closed>()) {
            window->close();
            continue;
        }

        if (const auto* resized = event.getIf<sf::Event::Resized>()) {
            window_width = static_cast<int>(resized->size.x);
            window_height = static_cast<int>(resized->size.y);
            syncUIView();
            continue;
        }

        if (const auto* mb = event.getIf<sf::Event::MouseButtonPressed>()) {
            initDefaultPanelsIfNeeded();
            syncUIView();
            const sf::Vector2f mouse = window->mapPixelToCoords(mb->position);

            if (mb->button == sf::Mouse::Button::Left) {
                if (last_stop_button_rect_.has_value() && last_stop_button_rect_->contains(mouse)) {
                    requestStopTraining();
                    continue;
                }

                if (last_live_overrides_box_.has_value() && last_live_overrides_box_->contains(mouse)) {
                    const bool next = !live_overrides_enabled_.load(std::memory_order_relaxed);
                    live_overrides_enabled_.store(next, std::memory_order_relaxed);
                    publish_live_params(true);
                    continue;
                }

                if (last_live_kl_enable_box_.has_value() && last_live_kl_enable_box_->contains(mouse)) {
                    live_ui_kl_enabled_ = !live_ui_kl_enabled_;
                    live_overrides_enabled_.store(true, std::memory_order_relaxed);
                    publish_live_params(true);
                    continue;
                }

                auto begin_live_text_edit = [&](LiveInputTarget target, const std::string& initial_value) {
                    live_input_target_ = target;
                    live_input_buffer_ = initial_value;
                    live_input_error_until_ms_ = 0;
                };

                if (last_live_lr_value_box_.has_value() && last_live_lr_value_box_->contains(mouse)) {
                    begin_live_text_edit(LiveInputTarget::LR, format_decimal(live_ui_lr_, 10));
                    continue;
                }
                if (last_live_lrwu_value_box_.has_value() && last_live_lrwu_value_box_->contains(mouse)) {
                    begin_live_text_edit(LiveInputTarget::LRWarmup, std::to_string(live_ui_lr_warmup_steps_));
                    continue;
                }
                if (last_live_klb_value_box_.has_value() && last_live_klb_value_box_->contains(mouse)) {
                    begin_live_text_edit(LiveInputTarget::KLBeta, format_decimal(live_ui_kl_beta_, 8));
                    continue;
                }
                if (last_live_klwu_value_box_.has_value() && last_live_klwu_value_box_->contains(mouse)) {
                    begin_live_text_edit(LiveInputTarget::KLWarmup, std::to_string(live_ui_kl_warmup_steps_));
                    continue;
                }

                auto begin_live_drag_if_hit = [&](LiveDragTarget target,
                                                  const std::optional<sf::FloatRect>& track,
                                                  const std::optional<sf::FloatRect>& thumb) -> bool {
                    if ((thumb.has_value() && thumb->contains(mouse)) || (track.has_value() && track->contains(mouse))) {
                        live_dragging_ = target;
                        live_overrides_enabled_.store(true, std::memory_order_relaxed);
                        apply_live_slider_from_mouse(target, mouse);
                        publish_live_params(true);
                        return true;
                    }
                    return false;
                };

                if (begin_live_drag_if_hit(LiveDragTarget::LR, last_live_lr_track_, last_live_lr_thumb_)) continue;
                if (begin_live_drag_if_hit(LiveDragTarget::LRWarmup, last_live_lrwu_track_, last_live_lrwu_thumb_)) continue;
                if (begin_live_drag_if_hit(LiveDragTarget::KLBeta, last_live_klb_track_, last_live_klb_thumb_)) continue;
                if (begin_live_drag_if_hit(LiveDragTarget::KLWarmup, last_live_klwu_track_, last_live_klwu_thumb_)) continue;

                if (last_blocks_hide_act_box_.has_value() && last_blocks_hide_act_box_->contains(mouse)) {
                    hide_activation_blocks = !hide_activation_blocks;
                    requestResync();
                    saveUILayoutToLast();
                    continue;
                }
                if (last_blocks_hide_norm_box_.has_value() && last_blocks_hide_norm_box_->contains(mouse)) {
                    hide_normalisation_blocks = !hide_normalisation_blocks;
                    requestResync();
                    saveUILayoutToLast();
                    continue;
                }
                if (last_blocks_tips_limit_box_.has_value() && last_blocks_tips_limit_box_->contains(mouse)) {
                    blocks_tips_limit_editing_ = true;
                    blocks_tips_limit_input_ = (blocks_tips_visible_limit_ > 0) ? std::to_string(blocks_tips_visible_limit_) : std::string();
                    continue;
                }
                if (blocks_tips_limit_editing_) {
                    commit_blocks_tips_limit();
                }

                // Sélection directe par clic sur les vignettes de tips (panel Blocks).
                bool selected_tip = false;
                if (!last_block_rects_.empty()) {
                    for (int i = static_cast<int>(last_block_rects_.size()) - 1; i >= 0; --i) {
                        const auto& r = last_block_rects_[static_cast<size_t>(i)];
                        if (r.size.x <= 0.f || r.size.y <= 0.f) continue;
                        if (r.contains(mouse)) {
                            focus_target_ = FocusTarget::LayerBlock;
                            focus_block_index_ = i;
                            selected_tip = true;
                            break;
                        }
                    }
                }
                if (selected_tip) continue;

                // Sélection directe des images générées au clic.
                bool selected_generated = false;
                if (!last_generated_rects_.empty() && last_generated_rects_.size() == last_generated_indices_.size()) {
                    for (int i = static_cast<int>(last_generated_rects_.size()) - 1; i >= 0; --i) {
                        const auto& r = last_generated_rects_[static_cast<size_t>(i)];
                        if (r.size.x <= 0.f || r.size.y <= 0.f) continue;
                        if (r.contains(mouse)) {
                            focus_target_ = FocusTarget::Generated;
                            focus_generated_index_ = std::max(0, last_generated_indices_[static_cast<size_t>(i)]);
                            selected_generated = true;
                            break;
                        }
                    }
                }
                if (selected_generated) continue;

                if (const auto close_hit = hitTestPanelCloseButton(mouse)) {
                    const PanelId id = *close_hit;
                    if (id == PanelId::Generated) show_generated_images = false;
                    else if (id == PanelId::Training) show_training_progress = false;
                    else if (id == PanelId::Graph) show_loss_graph = false;
                    panels_[static_cast<size_t>(id)].visible = false;
                    saveUILayoutToLast();
                    continue;
                }

                if (isPanelVisible(PanelId::Blocks) && last_blocks_scroll_thumb_rect_.contains(mouse)) {
                    dragging_blocks_scrollbar_ = true;
                    blocks_scroll_drag_grab_y_ = mouse.y - last_blocks_scroll_thumb_rect_.position.y;
                    continue;
                }
                if (isPanelVisible(PanelId::Blocks) && last_blocks_scroll_track_rect_.contains(mouse)) {
                    const float thumb_h = std::max(1.0f, last_blocks_scroll_thumb_rect_.size.y);
                    const float t = clamp01((mouse.y - last_blocks_scroll_track_rect_.position.y - thumb_h * 0.5f) /
                                            std::max(1.0f, last_blocks_scroll_track_rect_.size.y - thumb_h));
                    blocks_scroll_y_ = t * std::max(0.0f, blocks_scroll_max_);
                    continue;
                }

                if (const auto resize_hit = hitTestPanelResizeHandle(mouse)) {
                    resizing_panel_ = true;
                    resized_panel_ = *resize_hit;
                    resize_start_mouse_ = mouse;
                    resize_start_size_ = panels_[static_cast<size_t>(resized_panel_)].size;
                    setCursor(CursorKind::Resize);
                    continue;
                }

                if (const auto drag_hit = hitTestPanelTitle(mouse)) {
                    dragging_panel_ = true;
                    dragged_panel_ = *drag_hit;
                    const auto& p = panels_[static_cast<size_t>(dragged_panel_)];
                    drag_grab_offset_ = sf::Vector2f(mouse.x - p.pos.x, mouse.y - p.pos.y);
                    setCursor(CursorKind::Hand);
                    continue;
                }
            }
            continue;
        }

        if (const auto* mm = event.getIf<sf::Event::MouseMoved>()) {
            const sf::Vector2f mouse = window->mapPixelToCoords(mm->position);

            if (dragging_panel_) {
                auto& p = panels_[static_cast<size_t>(dragged_panel_)];
                p.pos = sf::Vector2f(mouse.x - drag_grab_offset_.x, mouse.y - drag_grab_offset_.y);
                clampPanelsToWindow();
            }

            if (resizing_panel_) {
                auto& p = panels_[static_cast<size_t>(resized_panel_)];
                const float dx = mouse.x - resize_start_mouse_.x;
                const float dy = mouse.y - resize_start_mouse_.y;
                p.size.x = std::max(kPanelMinW, resize_start_size_.x + dx);
                p.size.y = std::max(kPanelMinH, resize_start_size_.y + dy);
                clampPanelsToWindow();
            }

            if (dragging_blocks_scrollbar_ && isPanelVisible(PanelId::Blocks)) {
                const float thumb_h = std::max(1.0f, last_blocks_scroll_thumb_rect_.size.y);
                const float rel = mouse.y - last_blocks_scroll_track_rect_.position.y - blocks_scroll_drag_grab_y_;
                const float denom = std::max(1.0f, last_blocks_scroll_track_rect_.size.y - thumb_h);
                const float t = clamp01(rel / denom);
                blocks_scroll_y_ = t * std::max(0.0f, blocks_scroll_max_);
            }

            if (live_dragging_ != LiveDragTarget::None) {
                apply_live_slider_from_mouse(live_dragging_, mouse);
                publish_live_params(true);
            }

            if (dragging_panel_) {
                setCursor(CursorKind::Hand);
            } else if (resizing_panel_ || dragging_blocks_scrollbar_) {
                setCursor(CursorKind::Resize);
            } else {
                bool is_hand = false;
                if (isPanelVisible(PanelId::Blocks) &&
                    (last_blocks_scroll_thumb_rect_.contains(mouse) || last_blocks_scroll_track_rect_.contains(mouse))) {
                    is_hand = true;
                }
                if (!is_hand && (hitTestPanelResizeHandle(mouse).has_value() || hitTestPanelTitle(mouse).has_value())) {
                    is_hand = true;
                }
                setCursor(is_hand ? CursorKind::Hand : CursorKind::Arrow);
            }
            continue;
        }

        if (const auto* mb_rel = event.getIf<sf::Event::MouseButtonReleased>()) {
            if (mb_rel->button == sf::Mouse::Button::Left) {
                if (dragging_panel_ || resizing_panel_ || dragging_blocks_scrollbar_) {
                    saveUILayoutToLast();
                }
                dragging_panel_ = false;
                resizing_panel_ = false;
                dragging_blocks_scrollbar_ = false;
                live_dragging_ = LiveDragTarget::None;
            }
            continue;
        }

        if (const auto* txt = event.getIf<sf::Event::TextEntered>()) {
            const uint32_t cp = static_cast<uint32_t>(txt->unicode);

            if (blocks_tips_limit_editing_) {
                if (cp == 8u) {
                    if (!blocks_tips_limit_input_.empty()) blocks_tips_limit_input_.pop_back();
                } else if (cp >= static_cast<uint32_t>('0') && cp <= static_cast<uint32_t>('9')) {
                    blocks_tips_limit_input_.push_back(static_cast<char>(cp));
                }
                continue;
            }

            if (live_input_target_ != LiveInputTarget::None) {
                if (cp == 8u) {
                    if (!live_input_buffer_.empty()) live_input_buffer_.pop_back();
                } else if (is_live_input_char(cp)) {
                    live_input_buffer_.push_back(static_cast<char>(cp));
                }
                continue;
            }
        }

        if (const auto* wheel = event.getIf<sf::Event::MouseWheelScrolled>()) {
            if (zoom_active_ || dragging_panel_ || resizing_panel_) continue;
            initDefaultPanelsIfNeeded();
            syncUIView();
            const sf::Vector2f mouse = window->mapPixelToCoords(wheel->position);
            if (isPanelVisible(PanelId::Blocks)) {
                const auto content = panelContentRect(PanelId::Blocks);
                if (content.contains(mouse)) {
                    blocks_scroll_y_ -= wheel->delta * kBlocksScrollSpeed;
                    blocks_scroll_y_ = std::clamp(blocks_scroll_y_, 0.0f, std::max(0.0f, blocks_scroll_max_));
                }
            }
            continue;
        }

        if (const auto* key = event.getIf<sf::Event::KeyPressed>()) {
            const sf::Keyboard::Key code = key->code;
            const bool ctrl = key->control;
            const bool shift = key->shift;

            if (blocks_tips_limit_editing_) {
                if (code == sf::Keyboard::Key::Enter) {
                    commit_blocks_tips_limit();
                } else if (code == sf::Keyboard::Key::Escape) {
                    blocks_tips_limit_editing_ = false;
                    blocks_tips_limit_input_.clear();
                }
                continue;
            }

            if (live_input_target_ != LiveInputTarget::None) {
                if (code == sf::Keyboard::Key::Enter) {
                    commit_live_input();
                } else if (code == sf::Keyboard::Key::Escape) {
                    live_input_target_ = LiveInputTarget::None;
                    live_input_buffer_.clear();
                    live_input_error_until_ms_ = 0;
                }
                continue;
            }

            if (code == sf::Keyboard::Key::H) show_help_overlay_ = !show_help_overlay_;
            if (code == sf::Keyboard::Key::G) {
                show_loss_graph = !show_loss_graph;
                panels_[static_cast<size_t>(PanelId::Graph)].visible = show_loss_graph;
            }
            if (code == sf::Keyboard::Key::P) show_prompt_text_ = !show_prompt_text_;
            if (code == sf::Keyboard::Key::A) {
                smooth_layer_block_previews_ = !smooth_layer_block_previews_;
                rebuildAllTextures();
            }
            if (code == sf::Keyboard::Key::M) {
                heatmap_mode_ = !heatmap_mode_;
                for (auto& img : layer_block_images) {
                    if (!img.pixels_alt.empty()) {
                        std::swap(img.pixels, img.pixels_alt);
                        std::swap(img.channels, img.channels_alt);
                    }
                    if (heatmap_mode_ && img.channels == 1) {
                        img.pixels_alt = img.pixels;
                        img.channels_alt = 1;
                        img.pixels = colorize_gray_heatmap_rgb(img.pixels_alt, static_cast<int>(heatmap_palette_));
                        img.channels = 3;
                    }
                }
                rebuildLayerBlockTextures();
            }
            if (code == sf::Keyboard::Key::K) {
                const int pid = (static_cast<int>(heatmap_palette_) + 1) % 4;
                heatmap_palette_ = static_cast<Visualizer::HeatmapPalette>(pid);
                if (heatmap_mode_) {
                    for (auto& img : layer_block_images) {
                        if (img.channels_alt == 1 && !img.pixels_alt.empty()) {
                            img.pixels = colorize_gray_heatmap_rgb(img.pixels_alt, static_cast<int>(heatmap_palette_));
                            img.channels = 3;
                        }
                    }
                }
                rebuildLayerBlockTextures();
                saveUILayoutToLast();
            }
            if (code == sf::Keyboard::Key::Z || code == sf::Keyboard::Key::Enter) zoom_active_ = !zoom_active_;
            if (code == sf::Keyboard::Key::Escape) zoom_active_ = false;
            if (code == sf::Keyboard::Key::R) {
                requestResync();
                rebuildAllTextures();
                architecture_loaded = false;
                arch_layer_names.clear();
                arch_tensor_outputs.clear();
                arch_tensor_sinks.clear();
            }
            if (code == sf::Keyboard::Key::Tab) {
                if (focus_target_ == FocusTarget::Dataset) focus_target_ = FocusTarget::Projection;
                else if (focus_target_ == FocusTarget::Projection) focus_target_ = FocusTarget::Understanding;
                else if (focus_target_ == FocusTarget::Understanding) focus_target_ = FocusTarget::LayerBlock;
                else if (focus_target_ == FocusTarget::LayerBlock) focus_target_ = FocusTarget::Generated;
                else focus_target_ = FocusTarget::Dataset;
            }
            if (code == sf::Keyboard::Key::F1) focus_target_ = FocusTarget::Dataset;
            if (code == sf::Keyboard::Key::F2) focus_target_ = FocusTarget::Projection;
            if (code == sf::Keyboard::Key::F3) focus_target_ = FocusTarget::Understanding;
            if (code == sf::Keyboard::Key::F4) focus_target_ = FocusTarget::LayerBlock;
            if (code == sf::Keyboard::Key::F5) focus_target_ = FocusTarget::Generated;

            if (ctrl && (code == sf::Keyboard::Key::Left || code == sf::Keyboard::Key::Right ||
                         code == sf::Keyboard::Key::Up || code == sf::Keyboard::Key::Down)) {
                const int step = shift ? 128 : 64;
                const auto sz = window->getSize();
                int ww = static_cast<int>(sz.x);
                int hh = static_cast<int>(sz.y);
                if (code == sf::Keyboard::Key::Left) ww -= step;
                if (code == sf::Keyboard::Key::Right) ww += step;
                if (code == sf::Keyboard::Key::Up) hh -= step;
                if (code == sf::Keyboard::Key::Down) hh += step;
                ww = std::max(640, ww);
                hh = std::max(480, hh);
                window->setSize(sf::Vector2u(static_cast<unsigned>(ww), static_cast<unsigned>(hh)));
                window_width = ww;
                window_height = hh;
                syncUIView();
            } else {
                if (focus_target_ == FocusTarget::LayerBlock &&
                    (code == sf::Keyboard::Key::Left || code == sf::Keyboard::Key::Right)) {
                    const int n = static_cast<int>(layer_block_images.size());
                    if (n > 0) {
                        if (code == sf::Keyboard::Key::Left) focus_block_index_ = (focus_block_index_ - 1 + n) % n;
                        if (code == sf::Keyboard::Key::Right) focus_block_index_ = (focus_block_index_ + 1) % n;
                    }
                } else if (focus_target_ == FocusTarget::Generated &&
                           (code == sf::Keyboard::Key::Left || code == sf::Keyboard::Key::Right)) {
                    const int n = static_cast<int>(generated_images.size());
                    if (n > 0) {
                        focus_generated_index_ = std::clamp(focus_generated_index_, 0, n - 1);
                        if (code == sf::Keyboard::Key::Left) focus_generated_index_ = (focus_generated_index_ - 1 + n) % n;
                        if (code == sf::Keyboard::Key::Right) focus_generated_index_ = (focus_generated_index_ + 1) % n;
                    }
                }
            }

            if (code == sf::Keyboard::Key::S) {
                save_chord_armed_ = true;
                save_chord_consumed_ = false;
                save_chord_armed_ms_ = static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now().time_since_epoch())
                        .count());
            }

            auto key_to_digit = [](sf::Keyboard::Key k) -> int {
                if (k >= sf::Keyboard::Key::Num0 && k <= sf::Keyboard::Key::Num9) {
                    return static_cast<int>(k) - static_cast<int>(sf::Keyboard::Key::Num0);
                }
                if (k >= sf::Keyboard::Key::Numpad0 && k <= sf::Keyboard::Key::Numpad9) {
                    return static_cast<int>(k) - static_cast<int>(sf::Keyboard::Key::Numpad0);
                }
                return -1;
            };
            const int digit = key_to_digit(code);
            if (digit >= 0) {
                const bool s_down = sf::Keyboard::isKeyPressed(sf::Keyboard::Key::S);
                const uint64_t now_ms = static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now().time_since_epoch())
                        .count());
                const bool chord_active = s_down || (save_chord_armed_ && (now_ms - save_chord_armed_ms_) < 800);
                initDefaultPanelsIfNeeded();
                if (chord_active) {
                    saveUILayoutToSlot(digit);
                    save_chord_consumed_ = true;
                } else {
                    loadUILayoutFromSlot(digit);
                }
            }
            continue;
        }

        if (const auto* key_rel = event.getIf<sf::Event::KeyReleased>()) {
            if (key_rel->code == sf::Keyboard::Key::S) {
                if (save_chord_armed_ && !save_chord_consumed_) {
                    initDefaultPanelsIfNeeded();
                    saveUILayoutToLast();
                }
                save_chord_armed_ = false;
                save_chord_consumed_ = false;
            }
            continue;
        }
    }
}

void Visualizer::requestResync() {
    resync_requested_.store(true, std::memory_order_relaxed);
}

bool Visualizer::consumeResyncRequested() {
    return resync_requested_.exchange(false, std::memory_order_relaxed);
}

void Visualizer::requestStopTraining() {
    stop_training_requested_.store(true, std::memory_order_relaxed);
}

bool Visualizer::consumeStopTrainingRequested() {
    return stop_training_requested_.exchange(false, std::memory_order_relaxed);
}

uint64_t Visualizer::liveTrainParamsVersion() const {
    return live_params_version_.load(std::memory_order_relaxed);
}

Visualizer::LiveTrainParams Visualizer::liveTrainParamsSnapshot() const {
    LiveTrainParams p;
    p.overrides_enabled = live_overrides_enabled_.load(std::memory_order_relaxed);
    p.lr = live_lr_.load(std::memory_order_relaxed);
    p.lr_warmup_steps = live_lr_warmup_steps_.load(std::memory_order_relaxed);
    p.kl_beta = live_kl_beta_.load(std::memory_order_relaxed);
    p.kl_warmup_steps = live_kl_warmup_steps_.load(std::memory_order_relaxed);
    p.kl_enabled = live_kl_enabled_.load(std::memory_order_relaxed);
    p.version = live_params_version_.load(std::memory_order_relaxed);
    return p;
}

void Visualizer::update() {
    if (!enabled) return;
    if (!window || !window->isOpen()) return;

    // Robustesse: garantir que le contexte GL est actif dans le thread Viz avant
    // toute reconstruction de texture ou tout rendu de frame.
    window->setActive(true);

    // Premier frame: re-synchroniser toutes les textures maintenant que le contexte
    // SFML est garanti valide (évite un premier rendu sans textures visibles).
    if (!first_texture_sync_done_) {
        rebuildAllTextures();
        first_texture_sync_done_ = true;
    }

    // Auto-réparation: si certains blocks ont encore une texture invalide,
    // retenter un rebuild, mais jamais a chaque frame pour garder une UI fluide.
    if (has_layer_blocks && !layer_block_images.empty()) {
        bool has_invalid_block_tex = false;
        for (const auto& img : layer_block_images) {
            if (img.pixels.empty()) continue;
            if (img.texture.getSize().x == 0 || img.texture.getSize().y == 0) {
                has_invalid_block_tex = true;
                break;
            }
        }
        if (has_invalid_block_tex) {
            const uint64_t now_ms = steady_now_ms();
            if (last_block_texture_rebuild_ms_ == 0 || (now_ms - last_block_texture_rebuild_ms_) >= 250) {
                rebuildLayerBlockTextures();
                last_block_texture_rebuild_ms_ = now_ms;
            }
        }
    }

    // Refresh auto des textures (évite un reload UI)
    {
        const uint64_t now_ms = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now().time_since_epoch())
                .count());
        if (last_auto_texture_refresh_ms_ == 0) {
            last_auto_texture_refresh_ms_ = now_ms;
        } else if ((now_ms - last_auto_texture_refresh_ms_) >= kAutoTextureRefreshPeriodMs) {
            rebuildAllTextures();
            last_auto_texture_refresh_ms_ = now_ms;
        }
    }

    // Toujours forcer la view UI avant de rendre.
    syncUIView();

    // Best-effort: si on a raté un événement Resized, ne jamais laisser
    // window_width/window_height et la view diverger (sinon étirement).
    {
        const auto sz = window->getSize();
        const int ww = static_cast<int>(sz.x);
        const int hh = static_cast<int>(sz.y);
        if (ww > 0 && hh > 0 && (ww != window_width || hh != window_height)) {
            window_width = ww;
            window_height = hh;
            syncUIView();
            // Ne pas toucher aux panneaux: ils restent tels quels.
        }
    }

    initDefaultPanelsIfNeeded();
    // Visibilité liée aux toggles
    panels_[static_cast<size_t>(PanelId::Generated)].visible = show_generated_images;
    panels_[static_cast<size_t>(PanelId::Training)].visible = show_training_progress;
    panels_[static_cast<size_t>(PanelId::Graph)].visible = show_loss_graph;

    maybeLoadArchitecture();

    window->clear(sf::Color(30, 30, 35)); // Fond sombre

    renderBackground();

    drawPanelChrome(PanelId::Context);
    renderContextImages();

    drawPanelChrome(PanelId::Blocks);
    renderLayerBlocks();

    if (show_generated_images) {
        drawPanelChrome(PanelId::Generated);
        renderGeneratedImages();
    }

    if (show_training_progress) {
        drawPanelChrome(PanelId::Training);
        renderTrainingProgress();
    }

    drawPanelChrome(PanelId::Metrics);
    renderMetrics();

    if (show_loss_graph) {
        drawPanelChrome(PanelId::Graph);
        renderLossGraph();
    }

    renderFocusOutline();

    if (zoom_active_) {
        renderZoomOverlay();
    }
    if (show_help_overlay_) {
        renderHelpOverlay();
    }

    window->display();
}

void Visualizer::renderFocusOutline() {
    if (!window) return;
    if (zoom_active_) return; // le zoom a déjà sa propre emphase

    // Ne pas afficher si le panneau associé est fermé
    if (focus_target_ == FocusTarget::LayerBlock) {
        if (!isPanelVisible(PanelId::Blocks)) return;
    } else if (focus_target_ == FocusTarget::Generated) {
        if (!isPanelVisible(PanelId::Generated)) return;
    } else {
        if (!isPanelVisible(PanelId::Context)) return;
    }

    std::optional<sf::FloatRect> r;
    if (focus_target_ == FocusTarget::Dataset) {
        r = last_dataset_rect_;
    } else if (focus_target_ == FocusTarget::Projection) {
        r = last_projection_rect_;
    } else if (focus_target_ == FocusTarget::Understanding) {
        r = last_understanding_rect_;
    } else if (focus_target_ == FocusTarget::LayerBlock) {
        if (!last_block_rects_.empty()) {
            const int n = static_cast<int>(last_block_rects_.size());
            const int idx = (n > 0) ? std::clamp(focus_block_index_, 0, n - 1) : 0;
            if (idx >= 0 && idx < n) r = last_block_rects_[static_cast<size_t>(idx)];
        }
    } else if (focus_target_ == FocusTarget::Generated) {
        if (!last_generated_rects_.empty() && last_generated_rects_.size() == last_generated_indices_.size()) {
            // Retrouver la rect correspondant à l'index sélectionné
            for (size_t i = 0; i < last_generated_indices_.size(); ++i) {
                if (last_generated_indices_[i] == focus_generated_index_) {
                    r = last_generated_rects_[i];
                    break;
                }
            }
            // Fallback: clamp sur la première rect visible
            if (!r.has_value() && !last_generated_rects_.empty()) {
                r = last_generated_rects_.front();
            }
        }
    }

    if (!r.has_value()) return;
    if (r->size.x <= 0.f || r->size.y <= 0.f) return;

    // Accent selon la zone focusée
    sf::Color col = sf::Color(245, 245, 250, 230);
    if (focus_target_ == FocusTarget::LayerBlock) {
        col = with_alpha(panelAccent(PanelId::Blocks), 235);
    } else if (focus_target_ == FocusTarget::Generated) {
        col = with_alpha(panelAccent(PanelId::Generated), 235);
    } else {
        col = with_alpha(panelAccent(PanelId::Context), 235);
    }

    // PS (user): quand on scrolle dans Blocks/Layers, ne pas laisser l'outline visible
    // si l'élément focus sort de la zone réellement visible.
    if (focus_target_ == FocusTarget::LayerBlock) {
        const auto area = panelContentRect(PanelId::Blocks);
        const float ww = static_cast<float>(std::max(1, window_width));
        const float hh = static_cast<float>(std::max(1, window_height));

        const float left = std::clamp(area.position.x, 0.0f, ww);
        const float top = std::clamp(area.position.y, 0.0f, hh);
        const float right = std::clamp(area.position.x + area.size.x, 0.0f, ww);
        const float bottom = std::clamp(area.position.y + area.size.y, 0.0f, hh);
        const float w = std::max(0.0f, right - left);
        const float h = std::max(0.0f, bottom - top);
        if (w <= 0.0f || h <= 0.0f) return;

        const sf::FloatRect clip = make_rect(left, top, w, h);
        if (!clip.findIntersection(*r).has_value()) {
            // Hors zone visible => désactiver l'overline.
            return;
        }

        const sf::View old_view = window->getView();
        sf::View v(make_rect(left, top, w, h));
        v.setViewport(make_rect(left / ww, top / hh, w / ww, h / hh));
        window->setView(v);

        sf::RectangleShape outline(sf::Vector2f(r->size.x, r->size.y));
        outline.setPosition(sf::Vector2f(r->position.x, r->position.y));
        outline.setFillColor(sf::Color::Transparent);
        outline.setOutlineColor(col);
        outline.setOutlineThickness(3);
        window->draw(outline);

        window->setView(old_view);
        return;
    }

    sf::RectangleShape outline(sf::Vector2f(r->size.x, r->size.y));
    outline.setPosition(sf::Vector2f(r->position.x, r->position.y));
    outline.setFillColor(sf::Color::Transparent);
    outline.setOutlineColor(col);
    outline.setOutlineThickness(3);
    window->draw(outline);
}

void Visualizer::rebuildAllTextures() {
    auto rebuild_one = [&](ImageData& img) {
        if (img.w <= 0 || img.h <= 0 || img.display_size <= 0) return;
        if (img.channels != 1 && img.channels != 3 && img.channels != 4) return;
        if (img.pixels.empty()) return;
        createImageTexture(img, img.w, img.h, img.channels, img.display_size);
    };

    if (has_dataset_image) rebuild_one(dataset_image);
    if (has_projection_image) rebuild_one(projection_image);
    if (has_projection_thumb_) rebuild_one(projection_thumb_);
    if (has_understanding_image) rebuild_one(understanding_image);
    for (auto& img : generated_images) rebuild_one(img);
    if (has_output_thumb_) rebuild_one(output_thumb_);
    for (size_t i = 0; i < layer_block_images.size(); ++i) {
        const std::string label = (i < layer_block_labels.size()) ? layer_block_labels[i] : std::string();
        createLayerBlockTexture(layer_block_images[i], label);
    }
}

void Visualizer::renderHelpOverlay() {
    if (!window) return;

    sf::RectangleShape bg(sf::Vector2f(static_cast<float>(window_width), static_cast<float>(window_height)));
    bg.setPosition(sf::Vector2f(0, 0));
    bg.setFillColor(sf::Color(0, 0, 0, 160));
    window->draw(bg);

    if (!font_loaded) return;
    const int x = 24;
    int y = 24;
    const int lh = 18;
    auto line = [&](const std::string& s) {
        sf::Text t(font);
        t.setFont(font);
        t.setCharacterSize(14);
        t.setFillColor(sf::Color(235, 235, 240));
        t.setPosition(sf::Vector2f(static_cast<float>(x), static_cast<float>(y)));
        t.setString(sf::String::fromUtf8(s.begin(), s.end()));
        window->draw(t);
        y += lh;
    };

    line("Aide (clavier)");
    line("H : afficher/masquer cette aide");
    line("Z ou Entrée : grossir / réduire l'image sélectionnée");
    line("Tab / F1-F5 : sélectionner (dataset / projection / understanding / blocks / generated)");
    line("←/→ : naviguer dans les blocks (si focus=blocks)");
    line("G : afficher/masquer le graph");
    line("A : activer/masquer le lissage des previews de blocks");
    line("M : basculer heatmap color\u00e9e / niveaux de gris naturels (Blocks/Layers)");
    line("K : changer la palette heatmap (CLASSIC/TURBO/INFERNO/VIRIDIS)");
    line("P : afficher/masquer le texte du prompt");
    line("R : actualiser (rebuild textures + reload architecture)");
    line("S : sauvegarder la structure UI (layout last)");
    line("S+0..9 : sauvegarder la structure UI dans un slot");
    line("0..9 : charger/appliquer un slot UI");
    line("Ctrl+←/→ : réduire/agrandir la fenêtre (X)");
    line("Ctrl+↑/↓ : réduire/agrandir la fenêtre (Y)");
    line("Esc : fermer le zoom");
    line("Souris (layout)");
    line("Glisser-déposer sur le titre d'un panneau pour le déplacer");
    line("Glisser la poignée en bas-droite pour redimensionner un panneau");
    line("Cliquer sur [X] dans l'entête pour fermer un panneau");
    line("Blocks: cliquer sur \"TIPS ...\" pour saisir une limite d'items visibles (Enter=valider, vide=ALL)");
}

void Visualizer::renderZoomOverlay() {
    if (!window) return;

    const ImageData* img = nullptr;
    std::string label;
    if (focus_target_ == FocusTarget::Dataset && has_dataset_image) {
        img = &dataset_image;
        label = dataset_label.empty() ? std::string("dataset") : dataset_label;
    } else if (focus_target_ == FocusTarget::Projection && has_projection_image) {
        img = &projection_image;
        label = projection_label.empty() ? std::string("projection") : projection_label;
    } else if (focus_target_ == FocusTarget::Understanding && has_understanding_image) {
        img = &understanding_image;
        label = understanding_label.empty() ? std::string("understanding") : understanding_label;
    } else if (focus_target_ == FocusTarget::LayerBlock && !layer_block_images.empty()) {
        const int n = static_cast<int>(layer_block_images.size());
        const int idx = (n > 0) ? std::clamp(focus_block_index_, 0, n - 1) : 0;
        img = &layer_block_images[static_cast<size_t>(idx)];
        if (idx >= 0 && idx < static_cast<int>(layer_block_labels.size())) {
            label = layer_block_labels[static_cast<size_t>(idx)];
        }
    } else if (focus_target_ == FocusTarget::Generated && !generated_images.empty()) {
        const int n = static_cast<int>(generated_images.size());
        const int idx = (n > 0) ? std::clamp(focus_generated_index_, 0, n - 1) : 0;
        img = &generated_images[static_cast<size_t>(idx)];
        label = img->prompt.empty() ? std::string("generated") : (std::string("generated | ") + img->prompt);
    }

    if (!img) return;
    const auto ts = img->texture.getSize();
    if (ts.x == 0 || ts.y == 0) return;

    if (!label.empty()) {
        label += " (" + std::to_string(img->w) + "x" + std::to_string(img->h) + "x" + std::to_string(img->channels) + ")";
    }

    sf::RectangleShape bg(sf::Vector2f(static_cast<float>(window_width), static_cast<float>(window_height)));
    bg.setPosition(sf::Vector2f(0, 0));
    bg.setFillColor(sf::Color(0, 0, 0, 200));
    window->draw(bg);

    const float pad = 28.0f;
    const float box_w = static_cast<float>(window_width) - 2.0f * pad;
    const float box_h = static_cast<float>(window_height) - 2.0f * pad - 28.0f;
    const float box = std::min(box_w, box_h);

    sf::Sprite spr(img->texture);
    const float sx = box / std::max(1.0f, static_cast<float>(ts.x));
    const float sy = box / std::max(1.0f, static_cast<float>(ts.y));
    const float sc = std::min(sx, sy);
    spr.setScale(sf::Vector2f(sc, sc));

    const float dw = static_cast<float>(ts.x) * sc;
    const float dh = static_cast<float>(ts.y) * sc;
    spr.setPosition(sf::Vector2f((static_cast<float>(window_width) - dw) * 0.5f,
                                 (static_cast<float>(window_height) - dh) * 0.5f));
    window->draw(spr);

    if (font_loaded && !label.empty()) {
        sf::Text t(font);
        t.setFont(font);
        t.setCharacterSize(14);
        t.setFillColor(sf::Color(235, 235, 240));
        t.setPosition(sf::Vector2f(pad, pad));
        const std::string s = clamp_text_end(label, 120);
        t.setString(sf::String::fromUtf8(s.begin(), s.end()));
        window->draw(t);
    }
}

void Visualizer::maybeLoadArchitecture() {
    if (architecture_loaded) return;
    if (architecture_path.empty()) return;

    const uint64_t now_ms = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count());

    // Éviter de spammer le FS si le fichier n'existe pas encore.
    if (arch_last_check_ms != 0 && (now_ms - arch_last_check_ms) < 1000) {
        return;
    }
    arch_last_check_ms = now_ms;

    std::ifstream f(architecture_path);
    if (!f.is_open()) {
        return;
    }

    try {
        json j;
        f >> j;
        if (!j.contains("layers") || !j["layers"].is_array()) {
            return;
        }

        std::unordered_set<std::string> inputs;
        arch_layer_names.clear();
        arch_tensor_outputs.clear();
        arch_tensor_sinks.clear();

        for (const auto& layer : j["layers"]) {
            if (layer.contains("name") && layer["name"].is_string()) {
                arch_layer_names.insert(layer["name"].get<std::string>());
            }
            if (layer.contains("output") && layer["output"].is_string()) {
                arch_tensor_outputs.insert(layer["output"].get<std::string>());
            }
            if (layer.contains("inputs") && layer["inputs"].is_array()) {
                for (const auto& in : layer["inputs"]) {
                    if (in.is_string()) inputs.insert(in.get<std::string>());
                }
            }
        }

        // Sinks = outputs jamais réutilisés comme input
        for (const auto& out : arch_tensor_outputs) {
            if (inputs.find(out) == inputs.end()) {
                arch_tensor_sinks.insert(out);
            }
        }

        architecture_loaded = !arch_layer_names.empty();
        if (architecture_loaded) {
            std::cerr << "✓ Visualizer: architecture chargée: " << arch_layer_names.size()
                      << " layers (" << arch_tensor_sinks.size() << " sinks)" << std::endl;
        }
    } catch (...) {
        // Best-effort: on retentera plus tard.
    }
}

void Visualizer::addGeneratedImage(const std::vector<uint8_t>& image, const std::string& prompt) {
    // Legacy wrapper: on infère un carré grayscale.
    int img_size = static_cast<int>(std::sqrt(static_cast<double>(image.size())));
    addGeneratedImage(image, img_size, img_size, 1, prompt);
}

void Visualizer::addGeneratedImage(const std::vector<uint8_t>& image, int w, int h, int channels, const std::string& prompt) {
    if (!enabled) return;

    int iw = w;
    int ih = h;
    int ic = channels;

    if (ic != 1 && ic != 3 && ic != 4) {
        // Best-effort: tenter de déduire les canaux si possible.
        ic = 0;
    }
    if (iw <= 0 || ih <= 0 || ic == 0) {
        // Best-effort: inférer un carré (grayscale ou RGB selon la taille).
        const size_t n = image.size();
        const int s1 = static_cast<int>(std::sqrt(static_cast<double>(n)));
        const int s3 = static_cast<int>(std::sqrt(static_cast<double>(n / 3)));
        if ((size_t)(s3 * s3 * 3) == n) {
            iw = s3; ih = s3; ic = 3;
        } else if ((size_t)(s1 * s1) == n) {
            iw = s1; ih = s1; ic = 1;
        }
    }

    if (iw <= 0 || ih <= 0 || (ic != 1 && ic != 3 && ic != 4)) {
        return;
    }
    const size_t expected = static_cast<size_t>(iw) * static_cast<size_t>(ih) * static_cast<size_t>(ic);
    if (image.size() < expected) {
        return;
    }

    ImageData img_data;
    img_data.pixels = image;
    img_data.prompt = prompt;
    img_data.w = iw;
    img_data.h = ih;
    img_data.channels = ic;
    img_data.display_size = 200;
    createImageTexture(img_data, iw, ih, ic, 200);

    generated_images.push_back(std::move(img_data));

    // Conserver une vignette "Sortie" (120px) pour l'afficher aussi dans Blocks/Layers.
    output_thumb_.pixels = image;
    output_thumb_.prompt = prompt;
    output_thumb_.w = iw;
    output_thumb_.h = ih;
    output_thumb_.channels = ic;
    output_thumb_.display_size = 120;
    createImageTexture(output_thumb_, iw, ih, ic, 120);
    has_output_thumb_ = true;
    output_thumb_label_ = std::string("mimir/output | ") + prompt;

    int max_images = image_grid_cols * image_grid_rows;
    if (generated_images.size() > static_cast<size_t>(max_images)) {
        generated_images.erase(generated_images.begin());
    }
}

void Visualizer::setDatasetImage(const std::vector<uint8_t>& pixels, int w, int h, int channels, const std::string& label) {
    if (!enabled) return;
    if (w <= 0 || h <= 0) return;
    if (channels != 1 && channels != 3 && channels != 4) return;

    dataset_image.pixels = pixels;
    dataset_image.prompt = label;
    dataset_label = label;
    dataset_image.w = w;
    dataset_image.h = h;
    dataset_image.channels = channels;
    dataset_image.display_size = 200;
    createImageTexture(dataset_image, w, h, channels, 200);
    has_dataset_image = true;
}

void Visualizer::setDatasetText(const std::string& raw_text, const std::string& tags, const std::string& tokenized, const std::string& encoded) {
    if (!enabled) return;
    dataset_text_raw = raw_text;
    dataset_text_tags = tags;
    dataset_text_tokens = tokenized;
    dataset_text_encoded = encoded;
    has_dataset_text = (!dataset_text_raw.empty() || !dataset_text_tags.empty() || !dataset_text_tokens.empty() || !dataset_text_encoded.empty());
}

void Visualizer::setProjectionImage(const std::vector<uint8_t>& pixels, int w, int h, int channels, const std::string& label) {
    if (!enabled) return;
    if (w <= 0 || h <= 0) return;
    if (channels != 1 && channels != 3 && channels != 4) return;

    projection_image.pixels = pixels;
    projection_image.prompt = label;
    projection_label = label;
    projection_image.w = w;
    projection_image.h = h;
    projection_image.channels = channels;
    projection_image.display_size = 200;
    createImageTexture(projection_image, w, h, channels, 200);
    has_projection_image = true;

    projection_thumb_.pixels = pixels;
    projection_thumb_.prompt = label;
    projection_thumb_.w = w;
    projection_thumb_.h = h;
    projection_thumb_.channels = channels;
    projection_thumb_.display_size = 120;
    createImageTexture(projection_thumb_, w, h, channels, 120);
    has_projection_thumb_ = true;
}

void Visualizer::setUnderstandingImage(const std::vector<uint8_t>& pixels, int w, int h, int channels, const std::string& label) {
    if (!enabled) return;
    if (w <= 0 || h <= 0) return;
    if (channels != 1 && channels != 3 && channels != 4) return;

    understanding_image.pixels = pixels;
    understanding_image.prompt = label;
    understanding_label = label;
    understanding_image.w = w;
    understanding_image.h = h;
    understanding_image.channels = channels;
    understanding_image.display_size = 200;
    createImageTexture(understanding_image, w, h, channels, 200);
    has_understanding_image = true;
}

void Visualizer::setLayerBlockImages(const std::vector<BlockFrame>& frames) {
    if (!enabled) return;

    layer_block_images.clear();
    layer_block_labels.clear();
    has_layer_blocks = false;

    if (frames.empty()) return;

    layer_block_images.reserve(frames.size());
    layer_block_labels.reserve(frames.size());

    for (const auto& f : frames) {
        if (f.w <= 0 || f.h <= 0) continue;
        if (f.pixels.empty()) continue;

        // Inference robuste des canaux pour éviter les textures décalées
        // quand les métadonnées (channels) ne collent pas au buffer.
        const int heat_channels = infer_channels_from_buffer_size(f.pixels, f.w, f.h, f.channels);
        const int real_channels = f.pixels_real.empty()
            ? 0
            : infer_channels_from_buffer_size(f.pixels_real, f.w, f.h, 1);
        if (heat_channels != 1 && heat_channels != 3 && heat_channels != 4) continue;

        const auto p = parse_viz_label(f.label);

        // PS (user): ne pas afficher les blocs d'activation.
        const bool exhaustive_model_layer = !p.model.empty() && p.tag == "L" &&
                                            f.label.find("/blocks/") != std::string::npos;
        if (hide_activation_blocks && !exhaustive_model_layer) {
            if (p.tag == "ACT" || f.label.find("/activation") != std::string::npos) {
                continue;
            }
        }

        // PS (user): ne pas afficher les blocs de normalisation.
        if (hide_normalisation_blocks && !exhaustive_model_layer) {
            if (p.tag == "N" || p.headline == "Norm" ||
                f.label.find("/norm") != std::string::npos ||
                f.label.find("/gn") != std::string::npos ||
                f.label.find("/ln") != std::string::npos ||
                f.label.find("/bn") != std::string::npos ||
                f.label.find("/in") != std::string::npos ||
                f.label.find("/rms") != std::string::npos) {
                continue;
            }
        }

        // PS (user): ne pas afficher la validation dans Blocks/Layers.
        // Best-effort: tag heuristique + substring explicite.
        if (p.tag == "VAL" || f.label.find("/validation") != std::string::npos || f.label.find("/val/") != std::string::npos) {
            continue;
        }

        // Le vecteur packé final (ex: recon||mu||logvar) n'est pas un tip image utile.
        // L'afficher comme pseudo-image carrée produit un bruit trompeur dans Outputs.
        if (!exhaustive_model_layer && f.label.find("/vec") != std::string::npos &&
            (p.layer_path.find("out_concat") != std::string::npos ||
             p.layer_path.find("out_pack") != std::string::npos)) {
            continue;
        }

        // Les concat de packing final (ex: recon||z||logvar) ne sont pas des images
        // stables à afficher: elles provoquent des artefacts visuels (répétitions/décalage).
        if (!exhaustive_model_layer && (p.layer_path.find("out_concat") != std::string::npos ||
            p.layer_path.find("out_pack") != std::string::npos)) {
            continue;
        }

        ImageData img;
        img.prompt = f.label;
        img.w = f.w;
        img.h = f.h;
        img.display_size = 120;

        if (!heatmap_mode_ && !f.pixels_real.empty()) {
            // Mode réel: respecter les canaux réellement présents dans pixels_real
            // (certaines sources peuvent fournir du RGB/RGBA et pas uniquement du 1 canal).
            if (real_channels == 1 || real_channels == 3 || real_channels == 4) {
                img.pixels       = f.pixels_real;
                img.channels     = real_channels;
                img.pixels_alt   = f.pixels;
                img.channels_alt = heat_channels;
            } else {
                // Buffer real invalide/incomplet: fallback heatmap pour éviter les artefacts.
                img.pixels       = f.pixels;
                img.channels     = heat_channels;
                img.pixels_alt   = f.pixels_real;
                img.channels_alt = (real_channels == 1 || real_channels == 3 || real_channels == 4) ? real_channels : 1;
            }
        } else {
            // Mode heatmap (par défaut) ou pas de version naturelle
            img.pixels      = f.pixels;
            img.channels    = heat_channels;
            img.pixels_alt  = f.pixels_real;
            img.channels_alt = (real_channels == 1 || real_channels == 3 || real_channels == 4) ? real_channels : 1;

            // En mode heatmap, éviter l'aspect noir/blanc sur les tips 1 canal:
            // on colorise en RGB tout en conservant la version brute pour debug.
            if (img.channels == 1) {
                img.pixels_alt = img.pixels;
                img.channels_alt = 1;
                img.pixels = colorize_gray_heatmap_rgb(img.pixels, static_cast<int>(heatmap_palette_));
                img.channels = 3;
            }
        }

        createLayerBlockTexture(img, f.label);
        layer_block_images.push_back(std::move(img));
        layer_block_labels.push_back(f.label);
    }

    has_layer_blocks = !layer_block_images.empty();
}

void Visualizer::rebuildLayerBlockTextures() {
    for (size_t i = 0; i < layer_block_images.size(); ++i) {
        const std::string label = (i < layer_block_labels.size()) ? layer_block_labels[i] : std::string();
        createLayerBlockTexture(layer_block_images[i], label);
    }
}

void Visualizer::createLayerBlockTexture(ImageData& img, const std::string& label) {
    if (img.w <= 0 || img.h <= 0 || img.display_size <= 0) return;
    if (img.channels != 1 && img.channels != 3 && img.channels != 4) return;
    if (img.pixels.empty()) return;

    const ParsedVizLabel p = parse_viz_label(label);
    const bool smooth_preview = smooth_layer_block_previews_ &&
                                is_layer_block_preview_smoothing_candidate(p, img.channels);

    if (!smooth_preview) {
        createImageTexture(img, img.w, img.h, img.channels, img.display_size);
        return;
    }

    ImageData tmp = img;
    tmp.pixels = smooth_preview_pixels(img.pixels, img.w, img.h, img.channels);
    createImageTexture(tmp, tmp.w, tmp.h, tmp.channels, tmp.display_size);
    img.texture = std::move(tmp.texture);
    img.sprite = tmp.sprite;
    img.sprite.setTexture(img.texture, true);
}

void Visualizer::updateMetrics(int epoch, int batch, float loss, float lr, float mse,
                              float kl, float wass, float ent, float mom, float spat, float temp,
                              float timestep,
                              int total_epochs, int total_batches, float avg_loss,
                              int batch_time_ms, size_t memory_mb, float bps, size_t params,
                              float grad_norm, float grad_max,
                              int opt_type, int opt_step,
                              float opt_beta1, float opt_beta2,
                              float opt_eps, float opt_weight_decay,
                              bool val_has, bool val_ok, int val_step, int val_items,
                              float val_recon, float val_kl, float val_align,
                              const std::string& recon_loss_type,
                              bool val_in_progress,
                              int val_done,
                              int val_total,
                              float kl_beta_effective) {
    current_epoch = epoch;
    current_total_epochs = total_epochs;
    current_batch = batch;
    current_total_batches = total_batches;
    current_loss = loss;
    current_avg_loss = avg_loss;
    current_lr = lr;
    current_mse = mse;
    current_recon_loss_type = recon_loss_type;
    current_kl = kl;
    current_wass = wass;
    current_ent = ent;
    current_mom = mom;
    current_spat = spat;
    current_temp = temp;
    current_timestep = timestep;
    current_batch_time_ms = batch_time_ms;
    current_memory_mb = memory_mb;
    current_bps = bps;
    current_params = params;
    current_grad_norm = grad_norm;
    current_grad_max = grad_max;
    current_opt_type = opt_type;
    current_opt_step = opt_step;
    current_opt_beta1 = opt_beta1;
    current_opt_beta2 = opt_beta2;
    current_opt_eps = opt_eps;
    current_opt_weight_decay = opt_weight_decay;

    current_kl_beta_effective = kl_beta_effective;

    current_val_has = val_has;
    current_val_ok = val_ok;
    current_val_in_progress = val_in_progress;
    current_val_step = val_step;
    current_val_items = val_items;
    current_val_done = std::max(0, val_done);
    current_val_total = std::max(0, val_total);
    current_val_recon = val_recon;
    current_val_kl = val_kl;
    current_val_align = val_align;

    if (!has_loss_stats_) {
        has_loss_stats_ = true;
        loss_min_ = loss;
        loss_max_ = loss;
    } else {
        loss_min_ = std::min(loss_min_, loss);
        loss_max_ = std::max(loss_max_, loss);
    }
    
    // Créer un record complet avec toutes les métriques
    LossRecord record;
    record.step = full_loss_history.size();
    record.epoch = epoch;
    record.total_epochs = total_epochs;
    record.batch = batch;
    record.total_batches = total_batches;
    record.loss = loss;
    record.avg_loss = avg_loss;
    record.lr = lr;
    record.batch_time_ms = batch_time_ms;
    record.bps = bps;
    record.memory_mb = memory_mb;
    record.params = params;
    record.mse = mse;
    record.kl_divergence = kl;
    record.wasserstein = wass;
    record.entropy_diff = ent;
    record.moment_mismatch = mom;
    record.spatial_coherence = spat;
    record.temporal_consistency = temp;
    record.timestep = timestep;
    record.grad_norm = grad_norm;
    record.grad_max = grad_max;
    record.opt_type = opt_type;
    record.opt_step = opt_step;
    record.opt_beta1 = opt_beta1;
    record.opt_beta2 = opt_beta2;
    record.opt_eps = opt_eps;
    record.opt_weight_decay = opt_weight_decay;
    // Métriques de validation : renseignées uniquement quand val_ok=true.
    // val_recon = loss primaire (img-space MSE pour DDPM, recon loss pour VAE).
    // val_kl    = second indicateur (eps-space MSE pour DDPM, KL pour VAE).
    record.is_val      = val_ok;
    record.val_loss    = val_ok ? val_recon : 0.f;
    record.val_mse     = val_ok ? val_kl    : 0.f;
    record.val_step_id = val_ok ? val_step  : -1;
    full_loss_history.push_back(record);
    
    // Flush CSV throttle: eviter une ecriture disque a chaque metric tick.
    pending_loss_log_flush_ = true;
    const uint64_t now_ms = steady_now_ms();
    if (!loss_log_file.empty() && (last_loss_log_flush_ms_ == 0 || (now_ms - last_loss_log_flush_ms_) >= 1000)) {
        saveLossHistory(loss_log_file);
        last_loss_log_flush_ms_ = now_ms;
        pending_loss_log_flush_ = false;
    }
}

void Visualizer::addLossPoint(float loss) {
    loss_history.push_back(loss);
    if (loss_history.size() > static_cast<size_t>(graph_history_size)) {
        loss_history.pop_front();
    }
    
    // NE PAS ajouter à full_loss_history ici - c'est déjà fait par updateMetrics()
    // Cette fonction ne fait que mettre à jour le graphique visuel
}

void Visualizer::clearImages() {
    generated_images.clear();
    has_output_thumb_ = false;
    output_thumb_.pixels.clear();
    output_thumb_label_.clear();
}

void Visualizer::setEnabled(bool en) {
    enabled = en;
}

bool Visualizer::isEnabled() const {
    return enabled;
}

// Méthodes privées de rendu

void Visualizer::renderBackground() {
    // Grille de fond subtile
    sf::RectangleShape line(sf::Vector2f(1, window_height));
    line.setFillColor(sf::Color(40, 40, 45, 100));
    
    for (int x = 0; x < window_width; x += 50) {
        line.setPosition(sf::Vector2f(x, 0));
        window->draw(line);
    }

    line.setSize(sf::Vector2f(window_width, 1));
    for (int y = 0; y < window_height; y += 50) {
        line.setPosition(sf::Vector2f(0, y));
        window->draw(line);
    }

    // Splash logo au lancement
    if (logo_loaded_) {
        const float t = logo_clock_.getElapsedTime().asSeconds();
        if (t >= 0.0f && t < logo_splash_seconds_) {
            const auto lb = logo_sprite_.getLocalBounds();
            const float iw = std::max(1.0f, lb.size.x);
            const float ih = std::max(1.0f, lb.size.y);
            const float desired = std::min<float>(
                std::min<float>(window_width, window_height) * 0.35f,
                320.0f
            );
            const float scale = desired / std::max(iw, ih);
            logo_sprite_.setScale(sf::Vector2f(scale, scale));
            logo_sprite_.setOrigin(sf::Vector2f(lb.position.x + iw * 0.5f, lb.position.y + ih * 0.5f));
            logo_sprite_.setPosition(sf::Vector2f(window_width * 0.5f, window_height * 0.5f));
            window->draw(logo_sprite_);
        }
    }
}

void Visualizer::renderContextImages() {
    // Affiche (si disponibles) l'image du dataset + la projection associée.
    const int img_display_size = 200;
    const int margin = 16;
    const auto area = panelContentRect(PanelId::Context);
    const int start_x = static_cast<int>(area.position.x);
    const int start_y = static_cast<int>(area.position.y);
    const int label_h = 20;

    // Clip au contenu du panneau (overflow: clip)
    const sf::View old_view = window->getView();
    struct ViewGuard {
        sf::RenderWindow* w;
        sf::View v;
        ~ViewGuard() {
            if (w) w->setView(v);
        }
    } guard{window.get(), old_view};
    {
        const float ww = static_cast<float>(std::max(1, window_width));
        const float hh = static_cast<float>(std::max(1, window_height));

        const float left = std::clamp(area.position.x, 0.0f, ww);
        const float top = std::clamp(area.position.y, 0.0f, hh);
        const float right = std::clamp(area.position.x + area.size.x, 0.0f, ww);
        const float bottom = std::clamp(area.position.y + area.size.y, 0.0f, hh);
        const float w = std::max(0.0f, right - left);
        const float h = std::max(0.0f, bottom - top);

        if (w > 0.0f && h > 0.0f) {
            sf::View v(make_rect(left, top, w, h));
            v.setViewport(make_rect(left / ww, top / hh, w / ww, h / hh));
            window->setView(v);
        }
    }

    auto toSfUtf8 = [](const std::string& s) -> sf::String {
        return sf::String::fromUtf8(s.begin(), s.end());
    };

    auto drawPanel = [&](int x, int y, ImageData& img, const std::string& label_text, sf::Color outline) -> sf::FloatRect {
        // Cadre
        sf::RectangleShape frame(sf::Vector2f(img_display_size + 4, img_display_size + 4));
        frame.setPosition(sf::Vector2f(x - 2, y - 2));
        frame.setFillColor(sf::Color::Transparent);
        frame.setOutlineColor(outline);
        frame.setOutlineThickness(2);
        window->draw(frame);

        const sf::FloatRect frame_rect = make_rect(static_cast<float>(x - 2), static_cast<float>(y - 2), static_cast<float>(img_display_size + 4), static_cast<float>(img_display_size + 4));

        // Image
        position_sprite_centered_in_box(img.sprite, static_cast<float>(x), static_cast<float>(y), static_cast<float>(img_display_size));
        window->draw(img.sprite);

        // Label bar
        sf::RectangleShape label(sf::Vector2f(img_display_size, label_h));
        label.setPosition(sf::Vector2f(static_cast<float>(x), static_cast<float>(y + img_display_size + 5)));
        label.setFillColor(sf::Color(50, 50, 60, 200));
        window->draw(label);

        if (font_loaded) {
            sf::Text t(font);
            t.setFont(font);
            t.setCharacterSize(14);
            t.setFillColor(sf::Color(230, 230, 235));
            t.setPosition(sf::Vector2f(static_cast<float>(x + 6), static_cast<float>(y + img_display_size + 3)));
            t.setString(toSfUtf8(label_text));
            window->draw(t);
        }

        return frame_rect;
    };

    auto apply_arch_hint = [&](ParsedVizLabel& p) {
        if (!architecture_loaded) return;
        if (!p.path.empty()) {
            if (arch_tensor_sinks.find(p.path) != arch_tensor_sinks.end()) {
                p.tag = "OUT";
                p.headline = "Sortie";
                p.short_text = p.path;
            } else if (arch_layer_names.find(p.path) != arch_layer_names.end()) {
                p.tag = "L";
                p.headline = "Layer";
                // enlever le préfixe modèle
                if (!p.model.empty() && p.path.rfind(p.model + "/", 0) == 0) {
                    p.short_text = p.path.substr(p.model.size() + 1);
                } else {
                    p.short_text = p.path;
                }
            } else if (arch_tensor_outputs.find(p.path) != arch_tensor_outputs.end()) {
                p.tag = "T";
                p.headline = "Tensor";
                p.short_text = p.path;
            }
        }
    };

    int x = start_x;
    int y = start_y;
    auto advance = [&]() {
        x += img_display_size + margin;
        if (x + img_display_size > static_cast<int>(area.position.x + area.size.x)) {
            x = start_x;
            y += img_display_size + label_h + margin;
        }
    };

    auto fits_row = [&]() {
        return (y + img_display_size + label_h) <= static_cast<int>(area.position.y + area.size.y);
    };

    if (!fits_row()) return;

    if (has_dataset_image) {
        auto p = parse_viz_label(dataset_label.empty() ? std::string("dataset") : dataset_label);
        apply_arch_hint(p);
        std::string text = "[" + p.tag + "] ";
        if (!p.model.empty()) text += p.model + " ";
        text += p.short_text;
        if (!p.extra.empty()) text += " " + p.extra;
        last_dataset_rect_ = drawPanel(x, y, dataset_image, clamp_text_end(text, 44), color_for_tag(p.tag));
        advance();
        if (!fits_row()) return;
    }
    if (has_projection_image) {
        auto p = parse_viz_label(projection_label.empty() ? std::string("projection") : projection_label);
        apply_arch_hint(p);
        std::string text = "[" + p.tag + "] ";
        if (!p.model.empty()) text += p.model + " ";
        text += p.short_text;
        if (!p.extra.empty()) text += " " + p.extra;
        last_projection_rect_ = drawPanel(x, y, projection_image, clamp_text_end(text, 44), color_for_tag(p.tag));
        advance();
        if (!fits_row()) return;
    }
    if (has_understanding_image) {
        auto p = parse_viz_label(understanding_label.empty() ? std::string("understanding") : understanding_label);
        apply_arch_hint(p);
        std::string text = "[" + p.tag + "] ";
        if (!p.model.empty()) text += p.model + " ";
        text += p.short_text;
        if (!p.extra.empty()) text += " " + p.extra;
        last_understanding_rect_ = drawPanel(x, y, understanding_image, clamp_text_end(text, 44), color_for_tag(p.tag));
    }

}

void Visualizer::renderLayerBlocks() {
    const bool has_blocks = (has_layer_blocks && !layer_block_images.empty());
    const bool show_projection = (has_projection_thumb_ && has_projection_image && !projection_thumb_.pixels.empty());
    const bool show_output = (has_output_thumb_ && !output_thumb_.pixels.empty());

    const int blocks_count = has_blocks ? static_cast<int>(layer_block_images.size()) : 0;
    if (!show_projection && !show_output && blocks_count <= 0) return;

    last_block_rects_.clear();

    const int thumb = 120;
    const int margin = 14;
    const auto area = panelContentRect(PanelId::Blocks);
    const int start_x = static_cast<int>(area.position.x);
    const int start_y = static_cast<int>(area.position.y);

    const int label_h = 18;
    const int section_h = 22;
    const int section_gap = 8;
    const int max_cols = std::max(1, static_cast<int>(area.size.x) / (thumb + margin));
    const int cell_h = thumb + label_h + margin;

    // Focus rectangles doivent correspondre aux vrais blocks (index = layer_block_images idx).
    last_block_rects_.assign(static_cast<size_t>(std::max(0, blocks_count)), make_rect(0.f, 0.f, 0.f, 0.f));

    // Clip au contenu du panneau (évite de dessiner en dehors de la zone visible pendant le scroll)
    const sf::View old_view = window->getView();
    {
        const float ww = static_cast<float>(std::max(1, window_width));
        const float hh = static_cast<float>(std::max(1, window_height));

        // Clamp à la partie visible dans la fenêtre (le panneau peut être resizé hors écran)
        const float left = std::clamp(area.position.x, 0.0f, ww);
        const float top = std::clamp(area.position.y, 0.0f, hh);
        const float right = std::clamp(area.position.x + area.size.x, 0.0f, ww);
        const float bottom = std::clamp(area.position.y + area.size.y, 0.0f, hh);
        const float w = std::max(0.0f, right - left);
        const float h = std::max(0.0f, bottom - top);

        if (w > 0.0f && h > 0.0f) {
            sf::View v(make_rect(left, top, w, h));
            v.setViewport(make_rect(left / ww, top / hh, w / ww, h / hh));
            window->setView(v);
        }
    }

    auto apply_arch_hint = [&](ParsedVizLabel& p) {
        if (!architecture_loaded) return;
        if (!p.path.empty()) {
            if (arch_tensor_sinks.find(p.path) != arch_tensor_sinks.end()) {
                p.tag = "OUT";
                p.headline = "Sortie";
                p.short_text = p.path;
            } else if (arch_layer_names.find(p.path) != arch_layer_names.end()) {
                p.tag = "L";
                p.headline = "Layer";
                // Enlever le préfixe "model/" si présent (plus lisible en vignette)
                if (!p.model.empty() && p.path.rfind(p.model + "/", 0) == 0) {
                    p.short_text = p.path.substr(p.model.size() + 1);
                } else {
                    p.short_text = p.path;
                }
            } else if (arch_tensor_outputs.find(p.path) != arch_tensor_outputs.end()) {
                p.tag = "T";
                p.headline = "Tensor";
                p.short_text = p.path;
            }
        }
    };

    struct BlockPanelEntry {
        bool is_header = false;
        bool is_extra = false;
        int block_index = -1;
        ImageData* img = nullptr;
        ParsedVizLabel parsed;
        std::string section;
        std::string text_override;
        std::string tag_override;
    };

    auto starts_with = [](const std::string& s, const std::string& prefix) {
        return s.rfind(prefix, 0) == 0;
    };

    auto section_color = [&](const std::string& section) -> sf::Color {
        if (section == "Inputs") return sf::Color(110, 145, 205, 210);
        if (section == "Text / Cond") return sf::Color(122, 188, 142, 210);
        if (section == "UNet") return sf::Color(104, 164, 216, 210);
        if (section == "VAE") return sf::Color(214, 126, 166, 210);
        if (section == "Diffusion") return sf::Color(146, 136, 222, 210);
        if (section == "Reconstruction") return sf::Color(226, 162, 106, 210);
        if (section == "ConditioningEncoder") return sf::Color(96, 168, 214, 210);
        if (section == "Down Blocks") return sf::Color(88, 176, 198, 210);
        if (section == "Bottleneck") return sf::Color(160, 118, 220, 210);
        if (section == "Backbone") return sf::Color(132, 132, 218, 210);
        if (section == "Up Blocks") return sf::Color(232, 170, 102, 210);
        if (section == "Latent") return sf::Color(214, 106, 172, 210);
        if (section == "Decoder") return sf::Color(233, 152, 92, 210);
        if (section == "Heads") return sf::Color(206, 196, 110, 210);
        if (section == "Outputs") return sf::Color(188, 188, 120, 210);
        return sf::Color(100, 100, 112, 210);
    };

    auto section_for = [&](const ParsedVizLabel& p, bool is_extra) -> std::string {
        if (is_extra) return "Outputs";
        const std::string lp = !p.layer_path.empty() ? p.layer_path : p.short_text;
        if (lp.empty()) return "Blocks";

        const auto parts = split(lp, '/');
        const std::string first = parts.empty() ? std::string() : parts.front();
        const std::string last = parts.empty() ? std::string() : parts.back();

        bool has_latent_exact = false;
        bool has_latent_prefix = false;
        bool has_latent_substr = false;

        bool has_text_substr = false;
        bool has_text_exact = false;

        bool has_input_exact = false;
        bool has_input_prefix = false;

        bool has_encoder_exact_or_prefix = false;
        bool has_decoder_exact_or_prefix = false;
        bool has_down_prefix = false;
        bool has_up_prefix = false;

        bool has_bottleneck_exact = false;
        bool has_bottleneck_prefix = false;
        bool has_bottleneck_substr = false;

        bool has_backbone_exact = false;
        bool has_backbone_prefix = false;

        bool has_head_exact = false;
        bool has_head_prefix = false;
        bool has_head_substr = false;

        bool has_output_exact = false;
        bool has_output_prefix = false;
        bool has_output_substr = false;

        bool has_diff_exact = false;
        bool has_diff_prefix = false;
        bool has_diff_substr = false;

        bool has_unet_exact = false;
        bool has_unet_substr = false;
        bool has_vae_exact = false;
        bool has_vae_substr = false;
        bool has_diffusion_exact = false;
        bool has_diffusion_substr = false;
        bool has_recon_exact = false;
        bool has_recon_substr = false;

        for (const auto& part : parts) {
            if (part == "mu" || part == "logvar" || part == "latent" || part == "z") has_latent_exact = true;
            if (starts_with(part, "mu") || starts_with(part, "logvar") || starts_with(part, "latent") || starts_with(part, "reparam")) has_latent_prefix = true;
            if (part.find("latent") != std::string::npos || part.find("logvar") != std::string::npos || part.find("reparam") != std::string::npos) has_latent_substr = true;

            if (part == "tok_emb" || part == "meanpool" || part == "pool" || part == "add_pos" || part == "text_proj") has_text_exact = true;
            if (part.find("text") != std::string::npos || part.find("token") != std::string::npos || part.find("prompt") != std::string::npos ||
                part.find("context") != std::string::npos || part.find("embed") != std::string::npos || part.find("cond") != std::string::npos) {
                has_text_substr = true;
            }

            if (part == "input" || part == "in" || part == "raw_in" || part == "raw_z" || part == "in_vec" || part == "in_hwc" || part == "in_chw" || part == "init") has_input_exact = true;
            if (starts_with(part, "input") || starts_with(part, "raw_") || starts_with(part, "in_")) has_input_prefix = true;

            if (part == "encoder" || starts_with(part, "enc") || starts_with(part, "encoder")) has_encoder_exact_or_prefix = true;
            if (part == "decoder" || starts_with(part, "dec") || starts_with(part, "decoder")) has_decoder_exact_or_prefix = true;
            if (starts_with(part, "down")) has_down_prefix = true;
            if (starts_with(part, "up")) has_up_prefix = true;

            if (part == "mid" || part == "bottleneck" || part == "bridge") has_bottleneck_exact = true;
            if (starts_with(part, "mid") || starts_with(part, "bot") || starts_with(part, "bottleneck") || starts_with(part, "bridge")) has_bottleneck_prefix = true;
            if (part.find("bot_") != std::string::npos || part.find("mid_") != std::string::npos) has_bottleneck_substr = true;

            if (part == "unet" || part == "transformer" || part == "transformer_block" || part == "backbone" || part == "trunk" || part == "layers" || part == "blocks") has_backbone_exact = true;
            if (starts_with(part, "unet") || starts_with(part, "transformer") || starts_with(part, "block") || starts_with(part, "layer")) has_backbone_prefix = true;

            if (part == "head" || part == "lm_head" || part == "classifier" || part == "proj" || part == "out_proj" || part == "text_proj") has_head_exact = true;
            if (starts_with(part, "head") || starts_with(part, "proj") || starts_with(part, "final_")) has_head_prefix = true;
            if (part.find("_head") != std::string::npos || part.find("_proj") != std::string::npos) has_head_substr = true;

            if (part == "out" || part == "output" || part == "out_pack" || part == "out_concat" || part == "recon" ||
                part == "recon_chw" || part == "recon_to_hwc" || part == "out_to_hwc" || part == "final") {
                has_output_exact = true;
            }
            if (starts_with(part, "out") || starts_with(part, "output") || starts_with(part, "recon")) has_output_prefix = true;
            if (part.find("recon") != std::string::npos || part.find("to_hwc") != std::string::npos || part.find("out_concat") != std::string::npos || part.find("out_pack") != std::string::npos) {
                has_output_substr = true;
            }

            if (part == "diff" || part == "err" || part == "resdiff" || part == "resdiff_abs") {
                has_diff_exact = true;
            }
            if (starts_with(part, "diff") || starts_with(part, "err") || starts_with(part, "resdiff")) {
                has_diff_prefix = true;
            }
            if (part.find("diff") != std::string::npos || part.find("err") != std::string::npos) {
                has_diff_substr = true;
            }

            if (part == "unet" || part == "unet2d") has_unet_exact = true;
            if (part.find("unet") != std::string::npos) has_unet_substr = true;

            if (part == "vae" || part == "vae_conv" || part == "vae_text") has_vae_exact = true;
            if (part.find("vae") != std::string::npos || part.find("decoder") != std::string::npos || part.find("encoder") != std::string::npos) {
                has_vae_substr = true;
            }

            if (part == "diffusion" || part == "ddpm") has_diffusion_exact = true;
            if (part.find("diffusion") != std::string::npos || part.find("ddpm") != std::string::npos || part.find("denoise") != std::string::npos || part.find("noise") != std::string::npos) {
                has_diffusion_substr = true;
            }

            if (part == "recon" || part == "reconstruction") has_recon_exact = true;
            if (part.find("recon") != std::string::npos || part.find("resdiff") != std::string::npos) {
                has_recon_substr = true;
            }
        }

        const bool is_latent = p.tag == "LAT" || has_latent_exact || has_latent_prefix || has_latent_substr;
        const bool is_text_cond = has_text_substr || has_text_exact;
        const bool is_input = has_input_exact || has_input_prefix;
        const bool is_encoder = first == "enc" || has_encoder_exact_or_prefix;
        const bool is_decoder = first == "dec" || has_decoder_exact_or_prefix;
        const bool is_down = has_down_prefix;
        const bool is_up = has_up_prefix;
        const bool is_bottleneck = has_bottleneck_exact || has_bottleneck_prefix || has_bottleneck_substr;
        const bool is_backbone = has_backbone_exact || has_backbone_prefix;
        const bool is_head = has_head_exact || has_head_prefix || has_head_substr;
        const bool is_diff = has_diff_exact || has_diff_prefix || has_diff_substr;
        const bool is_output = p.tag == "OUT" || has_output_exact || has_output_prefix || has_output_substr || is_diff || last == "tanh";
        const bool is_unet = has_unet_exact || has_unet_substr;
        const bool is_vae = has_vae_exact || has_vae_substr;
        const bool is_diffusion = has_diffusion_exact || has_diffusion_substr;
        const bool is_recon = has_recon_exact || has_recon_substr;

        if (is_recon) return "Reconstruction";
        if (is_diffusion) return "Diffusion";
        if (is_unet) return "UNet";
        if (is_vae) return "VAE";

        if (is_latent) {
            return "Latent";
        }
        if (is_text_cond) {
            return "Text / Cond";
        }
        if (is_input && !is_encoder && !is_decoder && !is_backbone) {
            return "Inputs";
        }
        if (is_bottleneck) {
            return "Bottleneck";
        }
        if (is_output) return "Outputs";
        if (is_encoder) return is_down ? "Down Blocks" : "ConditioningEncoder";
        if (is_down) return "Down Blocks";
        if (is_backbone && !is_up) return "Backbone";
        if (is_up) return "Up Blocks";
        if (is_decoder) return "Decoder";
        if (is_head && !is_output) return "Heads";
        if (is_input) return "Inputs";
        return "Blocks";
    };

    std::vector<BlockPanelEntry> entries;
    entries.reserve(static_cast<size_t>(blocks_count + 8));
    std::string current_section;

    auto push_header = [&](const std::string& section) {
        if (section.empty() || section == current_section) return;
        current_section = section;
        BlockPanelEntry h;
        h.is_header = true;
        h.section = section;
        entries.push_back(std::move(h));
    };

    if (show_projection || show_output) {
        push_header("Outputs");
        if (show_projection) {
            BlockPanelEntry e;
            e.is_extra = true;
            e.img = &projection_thumb_;
            e.parsed = parse_viz_label(projection_label.empty() ? std::string("mimir/output/projection") : projection_label);
            e.section = "Outputs";
            e.text_override = "Projection";
            e.tag_override = "OUT";
            entries.push_back(std::move(e));
        }
        if (show_output) {
            BlockPanelEntry e;
            e.is_extra = true;
            e.img = &output_thumb_;
            e.parsed = parse_viz_label(output_thumb_label_.empty() ? std::string("mimir/output") : output_thumb_label_);
            e.section = "Outputs";
            e.text_override = "Sortie";
            e.tag_override = "OUT";
            entries.push_back(std::move(e));
        }
    }

    for (int bi = 0; bi < blocks_count; ++bi) {
        const std::string parsed_label = (bi >= 0 && bi < static_cast<int>(layer_block_labels.size()))
            ? layer_block_labels[static_cast<size_t>(bi)]
            : std::string();
        ParsedVizLabel parsed = parse_viz_label(parsed_label);
        apply_arch_hint(parsed);
        const std::string section = section_for(parsed, false);
        push_header(section);

        BlockPanelEntry e;
        e.block_index = bi;
        e.img = &layer_block_images[static_cast<size_t>(bi)];
        e.parsed = std::move(parsed);
        e.section = section;
        entries.push_back(std::move(e));
    }

    // Règle UX: si une vignette "diff" existe, placer une vignette "recon"
    // immédiatement avant (dans la même section) pour comparaison visuelle directe.
    auto to_lower = [](std::string s) {
        std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        return s;
    };
    auto entry_key = [&](const BlockPanelEntry& e) -> std::string {
        return to_lower(e.parsed.layer_path + " " + e.parsed.path + " " + e.parsed.short_text + " " + e.text_override);
    };
    auto is_diff_entry = [&](const BlockPanelEntry& e) -> bool {
        if (e.is_header) return false;
        const std::string k = entry_key(e);
        return k.find("resdiff") != std::string::npos ||
               k.find("/diff") != std::string::npos ||
               k.find("diff_") != std::string::npos ||
               k.find(" err/") != std::string::npos;
    };
    auto is_recon_entry = [&](const BlockPanelEntry& e) -> bool {
        if (e.is_header) return false;
        const std::string k = entry_key(e);
        return (k.find("recon") != std::string::npos) && (k.find("resdiff") == std::string::npos);
    };

    for (size_t i = 0; i < entries.size(); ++i) {
        if (!is_diff_entry(entries[i])) continue;
        const std::string section = entries[i].section;

        // Déjà juste avant ?
        if (i > 0 && !entries[i - 1].is_header && entries[i - 1].section == section && is_recon_entry(entries[i - 1])) {
            continue;
        }

        // Chercher d'abord un recon dans la même section avant diff.
        size_t recon_idx = entries.size();
        for (size_t j = 0; j < i; ++j) {
            if (entries[j].is_header) continue;
            if (entries[j].section != section) continue;
            if (is_recon_entry(entries[j])) recon_idx = j;
        }

        // Sinon, chercher après diff dans la même section.
        if (recon_idx == entries.size()) {
            for (size_t j = i + 1; j < entries.size(); ++j) {
                if (entries[j].is_header) continue;
                if (entries[j].section != section) continue;
                if (is_recon_entry(entries[j])) {
                    recon_idx = j;
                    break;
                }
            }
        }

        if (recon_idx == entries.size() || recon_idx == i) continue;

        BlockPanelEntry recon = std::move(entries[recon_idx]);
        entries.erase(entries.begin() + static_cast<std::ptrdiff_t>(recon_idx));
        if (recon_idx < i) {
            --i;
        }
        entries.insert(entries.begin() + static_cast<std::ptrdiff_t>(i), std::move(recon));
    }

    if (blocks_tips_visible_limit_ > 0) {
        int remaining = blocks_tips_visible_limit_;
        std::vector<BlockPanelEntry> limited;
        limited.reserve(entries.size());

        std::optional<BlockPanelEntry> pending_header;
        for (const auto& entry : entries) {
            if (entry.is_header) {
                pending_header = entry;
                continue;
            }

            const bool keep = entry.is_extra || (remaining > 0);
            if (!keep) continue;

            if (pending_header.has_value()) {
                limited.push_back(*pending_header);
                pending_header.reset();
            }

            limited.push_back(entry);
            if (!entry.is_extra && remaining > 0) {
                --remaining;
            }
        }

        entries.swap(limited);
    }

    if (entries.empty()) return;

    std::vector<sf::Vector2f> entry_positions;
    entry_positions.reserve(entries.size());

    int layout_col = 0;
    int layout_y = start_y;
    for (const auto& entry : entries) {
        if (entry.is_header) {
            if (layout_col != 0) {
                layout_col = 0;
                layout_y += cell_h;
            }
            entry_positions.emplace_back(static_cast<float>(start_x), static_cast<float>(layout_y));
            layout_y += section_h + section_gap;
        } else {
            const int x = start_x + layout_col * (thumb + margin);
            entry_positions.emplace_back(static_cast<float>(x), static_cast<float>(layout_y));
            layout_col += 1;
            if (layout_col >= max_cols) {
                layout_col = 0;
                layout_y += cell_h;
            }
        }
    }

    const float content_h = static_cast<float>((layout_y - start_y) + ((layout_col != 0) ? cell_h : 0));

    // PS (user): MAJ en temps réel des scrollers avec les valeurs runtime.
    // Ici: si le runtime ajoute/retire des items, on conserve la position relative
    // tant que l'utilisateur n'est pas en train de drag.
    const float prev_max = blocks_scroll_max_;
    const float prev_y = blocks_scroll_y_;

    blocks_scroll_max_ = std::max(0.0f, content_h - area.size.y);
    if (!dragging_blocks_scrollbar_ && prev_max > 1e-6f) {
        const float t = std::clamp(prev_y / prev_max, 0.0f, 1.0f);
        blocks_scroll_y_ = t * blocks_scroll_max_;
    }
    blocks_scroll_y_ = std::clamp(blocks_scroll_y_, 0.0f, std::max(0.0f, blocks_scroll_max_));

    for (size_t i = 0; i < entries.size(); ++i) {
        const auto& entry = entries[i];
        const int x = static_cast<int>(std::lround(entry_positions[i].x));
        const int base_y = static_cast<int>(std::lround(entry_positions[i].y));
        const int y = base_y - static_cast<int>(std::lround((double)blocks_scroll_y_));

        if (entry.is_header) {
            if ((y - section_h) > static_cast<int>(area.position.y + area.size.y)) continue;
            if ((y + section_h + section_gap) < static_cast<int>(area.position.y)) continue;

            const float bar_w = std::max(80.0f, area.size.x - 12.0f);
            sf::RectangleShape bar(sf::Vector2f(bar_w, static_cast<float>(section_h)));
            bar.setPosition(sf::Vector2f(static_cast<float>(start_x), static_cast<float>(y)));
            bar.setFillColor(sf::Color(28, 30, 38, 220));
            bar.setOutlineColor(section_color(entry.section));
            bar.setOutlineThickness(1.0f);
            window->draw(bar);

            sf::RectangleShape accent(sf::Vector2f(8.0f, static_cast<float>(section_h)));
            accent.setPosition(sf::Vector2f(static_cast<float>(start_x), static_cast<float>(y)));
            accent.setFillColor(section_color(entry.section));
            window->draw(accent);

            if (font_loaded) {
                sf::Text t(font);
                t.setFont(font);
                t.setCharacterSize(13);
                t.setStyle(sf::Text::Bold);
                t.setFillColor(sf::Color(236, 236, 240));
                t.setPosition(sf::Vector2f(static_cast<float>(start_x + 14), static_cast<float>(y + 2)));
                t.setString(sf::String::fromUtf8(entry.section.begin(), entry.section.end()));
                window->draw(t);
            }
            continue;
        }

        if (!entry.img) continue;

        // Cull (best-effort) pour limiter les draws
        if ((y - cell_h) > static_cast<int>(area.position.y + area.size.y)) continue;
        if ((y + 2 * cell_h) < static_cast<int>(area.position.y)) continue;

        const std::string use_tag = entry.is_extra ? entry.tag_override : entry.parsed.tag;

        // Frame
        sf::RectangleShape frame(sf::Vector2f(thumb + 4, thumb + 4));
        frame.setPosition(sf::Vector2f(static_cast<float>(x - 2), static_cast<float>(y - 2)));
        frame.setFillColor(sf::Color::Transparent);
        frame.setOutlineColor(color_for_tag(use_tag));
        frame.setOutlineThickness(1);
        window->draw(frame);

        if (!entry.is_extra) {
            const int bi = entry.block_index;
            if (bi >= 0 && bi < blocks_count) {
                last_block_rects_[static_cast<size_t>(bi)] = make_rect(
                    static_cast<float>(x - 2),
                    static_cast<float>(y - 2),
                    static_cast<float>(thumb + 4),
                    static_cast<float>(thumb + 4));
            }
        }

        position_sprite_centered_in_box(entry.img->sprite, static_cast<float>(x), static_cast<float>(y), static_cast<float>(thumb));
        window->draw(entry.img->sprite);

        sf::RectangleShape label(sf::Vector2f(static_cast<float>(thumb), static_cast<float>(label_h)));
        label.setPosition(sf::Vector2f(static_cast<float>(x), static_cast<float>(y + thumb + 3)));
        label.setFillColor(sf::Color(50, 50, 60, 200));
        window->draw(label);

        if (font_loaded) {
            sf::Text t(font);
            t.setFont(font);
            t.setCharacterSize(12);
            t.setFillColor(sf::Color(230, 230, 235));
            t.setPosition(sf::Vector2f(static_cast<float>(x + 4), static_cast<float>(y + thumb + 1)));

            std::string text;
            if (entry.is_extra) {
                text = entry.text_override;
            } else if (entry.parsed.tag != "ACT") {
                text = "[" + entry.parsed.tag + "] " + entry.parsed.short_text;
            } else {
                text = entry.parsed.short_text;
            }
            text = clamp_text_end(text, 18);
            t.setString(sf::String::fromUtf8(text.begin(), text.end()));
            window->draw(t);
        }
    }

    // Restore view
    window->setView(old_view);

    // Scrollbar (slicer) cliquable (dessiné dans l'UI normale, hors view clipping)
    // NB: rects en coordonnées UI (pixels)
    last_blocks_scroll_track_rect_ = make_rect(0.f, 0.f, 0.f, 0.f);
    last_blocks_scroll_thumb_rect_ = make_rect(0.f, 0.f, 0.f, 0.f);
    if (blocks_scroll_max_ > 0.0f) {
        const float track_w = 10.0f;
        const float pad = 2.0f;
        const float tx = area.position.x + std::max(0.0f, area.size.x - track_w - pad);
        const float ty = area.position.y;
        const float th = std::max(0.0f, area.size.y);
        last_blocks_scroll_track_rect_ = make_rect(tx, ty, track_w, th);

        const float content_h2 = area.size.y + blocks_scroll_max_;
        const float visible_ratio = (content_h2 > 1.0f) ? std::clamp(area.size.y / content_h2, 0.05f, 1.0f) : 1.0f;
        const float thumb_h = std::max(24.0f, th * visible_ratio);
        const float denom = std::max(1.0f, th - thumb_h);
        const float t = (blocks_scroll_max_ > 0.0f) ? std::clamp(blocks_scroll_y_ / blocks_scroll_max_, 0.0f, 1.0f) : 0.0f;
        const float thumb_y = ty + t * denom;
        last_blocks_scroll_thumb_rect_ = make_rect(tx, thumb_y, track_w, thumb_h);

        sf::RectangleShape track(sf::Vector2f(track_w, th));
        track.setPosition(sf::Vector2f(tx, ty));
        track.setFillColor(sf::Color(35, 35, 42, 200));
        window->draw(track);

        sf::RectangleShape thumb(sf::Vector2f(track_w, thumb_h));
        thumb.setPosition(sf::Vector2f(tx, thumb_y));
        thumb.setFillColor(sf::Color(90, 90, 110, 220));
        window->draw(thumb);
    }
}

void Visualizer::renderGeneratedImages() {
    if (generated_images.empty()) return;

    const auto area = panelContentRect(PanelId::Generated);

    const int img_display_size = 200; // Taille d'affichage
    const int margin = 20;
    const int start_x = static_cast<int>(area.position.x);
    const int start_y = static_cast<int>(area.position.y);

    const int cell_w = img_display_size + margin;
    const int cell_h = img_display_size + margin + 26;
    const int cols = std::max(1, static_cast<int>(area.size.x) / std::max(1, cell_w));
    const int rows = std::max(1, static_cast<int>(area.size.y) / std::max(1, cell_h));
    const int max_items = std::max(1, cols * rows);

    const size_t n = std::min(generated_images.size(), static_cast<size_t>(max_items));
    const size_t start_idx = (generated_images.size() > n) ? (generated_images.size() - n) : 0;

    last_generated_rects_.clear();
    last_generated_indices_.clear();
    last_generated_rects_.reserve(n);
    last_generated_indices_.reserve(n);

    // Clip au contenu du panneau (overflow: clip)
    const sf::View old_view = window->getView();
    {
        const float ww = static_cast<float>(std::max(1, window_width));
        const float hh = static_cast<float>(std::max(1, window_height));

        const float left = std::clamp(area.position.x, 0.0f, ww);
        const float top = std::clamp(area.position.y, 0.0f, hh);
        const float right = std::clamp(area.position.x + area.size.x, 0.0f, ww);
        const float bottom = std::clamp(area.position.y + area.size.y, 0.0f, hh);
        const float w = std::max(0.0f, right - left);
        const float h = std::max(0.0f, bottom - top);

        if (w > 0.0f && h > 0.0f) {
            sf::View v(make_rect(left, top, w, h));
            v.setViewport(make_rect(left / ww, top / hh, w / ww, h / hh));
            window->setView(v);
        }
    }

    for (size_t i = 0; i < n; ++i) {
        auto& img = generated_images[start_idx + i];
        
        int col = static_cast<int>(i % static_cast<size_t>(cols));
        int row = static_cast<int>(i / static_cast<size_t>(cols));
        
        int x = start_x + col * cell_w;
        int y = start_y + row * cell_h;

        // Cadre
        sf::RectangleShape frame(sf::Vector2f(img_display_size + 4, img_display_size + 4));
        frame.setPosition(sf::Vector2f(x - 2, y - 2));
        frame.setFillColor(sf::Color::Transparent);
        frame.setOutlineColor(sf::Color(100, 150, 200));
        frame.setOutlineThickness(2);
        window->draw(frame);

        last_generated_rects_.push_back(make_rect(static_cast<float>(x - 2), static_cast<float>(y - 2), static_cast<float>(img_display_size + 4), static_cast<float>(img_display_size + 4)));
        last_generated_indices_.push_back(static_cast<int>(start_idx + i));

        // Image
        position_sprite_centered_in_box(img.sprite, static_cast<float>(x), static_cast<float>(y), static_cast<float>(img_display_size));
        window->draw(img.sprite);

        // Titre (simulé avec rectangle - nécessite police pour texte)
        sf::RectangleShape label(sf::Vector2f(img_display_size, 20));
        label.setPosition(sf::Vector2f(x, y + img_display_size + 5));
        label.setFillColor(sf::Color(50, 50, 60, 200));
        window->draw(label);
    }

    window->setView(old_view);
}

void Visualizer::renderTrainingProgress() {
    const auto area = panelContentRect(PanelId::Training);
    const int bar_x = static_cast<int>(area.position.x);
    const int bar_width = std::max(10, static_cast<int>(area.size.x));
    const int global_h = std::clamp(static_cast<int>(area.size.y * 0.34f), 14, 24);
    const int batch_h = std::clamp(static_cast<int>(area.size.y * 0.22f), 10, 16);
    const int gap = 8;
    const int stack_h = global_h + gap + batch_h;
    const int global_y = static_cast<int>(area.position.y + std::max(0.0f, (area.size.y - static_cast<float>(stack_h)) * 0.5f));
    const int batch_y = global_y + global_h + gap;

    const bool has_epoch_info = (current_total_epochs > 0);
    const bool has_batch_info = (current_total_batches > 0);

    const int total_epochs = std::max(1, current_total_epochs);
    const int total_batches = std::max(1, current_total_batches);

    const int epoch_index = std::clamp(current_epoch - 1, 0, std::max(0, total_epochs - 1));
    const int batch_done = std::clamp(current_batch, 0, total_batches);

    const float batch_target = has_batch_info
        ? std::clamp(static_cast<float>(batch_done) / static_cast<float>(total_batches), 0.0f, 1.0f)
        : 0.0f;
    const float global_target = has_epoch_info
        ? std::clamp((static_cast<float>(epoch_index) + batch_target) / static_cast<float>(total_epochs), 0.0f, 1.0f)
        : 0.0f;

    // Lissage visuel (monotone local) pour éviter les sauts d'affichage.
    const auto smooth_to = [](float current, float target, float alpha) {
        if (!std::isfinite(current)) current = target;
        if (target + 0.25f < current) return target; // reset rapide si run repart de plus bas
        const float v = current + (target - current) * alpha;
        return (std::fabs(v - target) < 1e-4f) ? target : v;
    };

    progress_epoch_display_ = smooth_to(progress_epoch_display_, global_target, 0.24f);
    progress_batch_display_ = smooth_to(progress_batch_display_, batch_target, 0.30f);

    const auto batch_speed_color = [&](int batch_ms) -> sf::Color {
        // 0 = rapide (vert), 1 = lent (rouge)
        const float lo = 20.0f;
        const float hi = 2000.0f;
        const float t = std::clamp((static_cast<float>(std::max(0, batch_ms)) - lo) / (hi - lo), 0.0f, 1.0f);

        const float fast_r = 124.0f, fast_g = 218.0f, fast_b = 120.0f;
        const float slow_r = 226.0f, slow_g = 116.0f, slow_b = 92.0f;

        const auto mix = [&](float a, float b) -> uint8_t {
            return static_cast<uint8_t>(std::clamp(static_cast<int>(std::lround(a + (b - a) * t)), 0, 255));
        };

        return sf::Color(mix(fast_r, slow_r), mix(fast_g, slow_g), mix(fast_b, slow_b), 245);
    };

    // Barre globale (epochs)
    sf::RectangleShape progress_bg(sf::Vector2f(static_cast<float>(bar_width), static_cast<float>(global_h)));
    progress_bg.setPosition(sf::Vector2f(static_cast<float>(bar_x), static_cast<float>(global_y)));
    progress_bg.setFillColor(sf::Color(50, 50, 60, 230));
    progress_bg.setOutlineColor(sf::Color(88, 88, 102, 235));
    progress_bg.setOutlineThickness(1.5f);
    window->draw(progress_bg);

    const float epoch_w = static_cast<float>(bar_width) * progress_epoch_display_;
    if (epoch_w > 0.0f) {
        sf::RectangleShape progress_bar(sf::Vector2f(epoch_w, static_cast<float>(global_h)));
        progress_bar.setPosition(sf::Vector2f(static_cast<float>(bar_x), static_cast<float>(global_y)));
        progress_bar.setFillColor(sf::Color(96, 184, 112, 240));
        window->draw(progress_bar);
    }

    // Barre batch (courant) sous la barre globale.
    sf::RectangleShape batch_bg(sf::Vector2f(static_cast<float>(bar_width), static_cast<float>(batch_h)));
    batch_bg.setPosition(sf::Vector2f(static_cast<float>(bar_x), static_cast<float>(batch_y)));
    batch_bg.setFillColor(sf::Color(42, 42, 52, 230));
    batch_bg.setOutlineColor(sf::Color(78, 78, 94, 220));
    batch_bg.setOutlineThickness(1.0f);
    window->draw(batch_bg);

    if (has_batch_info) {
        const float batch_w = static_cast<float>(bar_width) * progress_batch_display_;
        if (batch_w > 0.0f) {
            sf::RectangleShape batch_fg(sf::Vector2f(batch_w, static_cast<float>(batch_h)));
            batch_fg.setPosition(sf::Vector2f(static_cast<float>(bar_x), static_cast<float>(batch_y)));
            batch_fg.setFillColor(batch_speed_color(current_batch_time_ms));
            window->draw(batch_fg);
        }
    }

    // Fallback visuel si l'info d'epoch est absente: bande animée discrète.
    if (!has_epoch_info) {
        const uint64_t t = steady_now_ms();
        const float phase = static_cast<float>((t % 1600ULL) / 1600.0);
        const float seg_w = std::max(24.0f, static_cast<float>(bar_width) * 0.18f);
        const float x = static_cast<float>(bar_x) + (static_cast<float>(bar_width) + seg_w) * phase - seg_w;
        sf::RectangleShape ind(sf::Vector2f(seg_w, static_cast<float>(global_h)));
        ind.setPosition(sf::Vector2f(x, static_cast<float>(global_y)));
        ind.setFillColor(sf::Color(110, 165, 215, 120));
        window->draw(ind);
    }

    if (font_loaded) {
        int pct = static_cast<int>(std::lround(progress_epoch_display_ * 100.0f));
        pct = std::clamp(pct, 0, 100);
        std::string txt_global;
        if (has_epoch_info) {
            txt_global = std::to_string(pct) + "%";
        } else {
            txt_global = "Initialisation";
        }

        sf::Text tg(font);
        tg.setFont(font);
        tg.setCharacterSize(12);
        tg.setFillColor(sf::Color(235, 235, 240));
        tg.setString(sf::String::fromUtf8(txt_global.begin(), txt_global.end()));
        const auto bg = tg.getLocalBounds();
        const float txg = static_cast<float>(bar_x) + (static_cast<float>(bar_width) - bg.size.x) * 0.5f - bg.position.x;
        const float tyg = static_cast<float>(global_y) + (static_cast<float>(global_h) - bg.size.y) * 0.5f - bg.position.y - 1.0f;
        tg.setPosition(sf::Vector2f(std::round(txg), std::round(tyg)));
        window->draw(tg);

        std::string txt_batch;
        if (has_batch_info) {
            const int bpct = std::clamp(static_cast<int>(std::lround(progress_batch_display_ * 100.0f)), 0, 100);
            txt_batch = "batch " + std::to_string(bpct) + "%";
        } else {
            txt_batch = "batch --";
        }
        sf::Text tb(font);
        tb.setFont(font);
        tb.setCharacterSize(11);
        tb.setFillColor(sf::Color(220, 228, 210));
        tb.setString(sf::String::fromUtf8(txt_batch.begin(), txt_batch.end()));
        const auto bb = tb.getLocalBounds();
        const float txb = static_cast<float>(bar_x) + (static_cast<float>(bar_width) - bb.size.x) * 0.5f - bb.position.x;
        const float tyb = static_cast<float>(batch_y) + (static_cast<float>(batch_h) - bb.size.y) * 0.5f - bb.position.y - 1.0f;
        tb.setPosition(sf::Vector2f(std::round(txb), std::round(tyb)));
        window->draw(tb);
    }
}

void Visualizer::renderLossGraph() {
    if (!window) return;

    const bool have_full = !full_loss_history.empty();
    if (!have_full && loss_history.empty()) return;

    const auto area = panelContentRect(PanelId::Graph);
    const int graph_x = static_cast<int>(area.position.x);
    const int graph_y = static_cast<int>(area.position.y);
    const int graph_width = std::max(120, static_cast<int>(area.size.x));
    const int graph_height = std::max(120, static_cast<int>(area.size.y));

    // Fond du graphique
    sf::RectangleShape graph_bg(sf::Vector2f(static_cast<float>(graph_width), static_cast<float>(graph_height)));
    graph_bg.setPosition(sf::Vector2f(static_cast<float>(graph_x), static_cast<float>(graph_y)));
    graph_bg.setFillColor(sf::Color(20, 20, 25, 220));
    graph_bg.setOutlineColor(sf::Color(100, 100, 120));
    graph_bg.setOutlineThickness(2);
    window->draw(graph_bg);

    // Zone de plot (padding pour l'échelle)
    const float pad_l = font_loaded ? 54.f : 10.f;
    const float pad_r = 10.f;
    const float pad_t = font_loaded ? 8.f : 10.f;
    const float pad_b = font_loaded ? 22.f : 10.f;
    const float plot_x = static_cast<float>(graph_x) + pad_l;
    const float plot_y = static_cast<float>(graph_y) + pad_t;
    const float plot_w = std::max(10.f, static_cast<float>(graph_width) - pad_l - pad_r);
    const float plot_h = std::max(10.f, static_cast<float>(graph_height) - pad_t - pad_b);

    // Min/Max pour normalisation (full history par défaut)
    float min_loss = 0.0f;
    float max_loss = 1.0f;
    if (have_full && has_loss_stats_) {
        min_loss = loss_min_;
        max_loss = loss_max_;
    } else {
        // fallback: calculer sur l'historique glissant
        min_loss = *std::min_element(loss_history.begin(), loss_history.end());
        max_loss = *std::max_element(loss_history.begin(), loss_history.end());
    }
    float range = max_loss - min_loss;
    if (!(range > 0.0f)) range = 1.0f;

    // Grille + échelle Y
    const int ticks = 5;
    for (int i = 0; i < ticks; ++i) {
        const float t = (ticks <= 1) ? 0.f : (static_cast<float>(i) / static_cast<float>(ticks - 1));
        const float y = plot_y + (1.f - t) * plot_h;

        sf::RectangleShape grid(sf::Vector2f(plot_w, 1.f));
        grid.setPosition(sf::Vector2f(plot_x, y));
        grid.setFillColor(sf::Color(70, 70, 80, (i == 0 || i == ticks - 1) ? 120 : 70));
        window->draw(grid);

        if (font_loaded) {
            const float v = min_loss + t * range;
            std::ostringstream ss;
            ss << std::fixed << std::setprecision(4) << v;
            const std::string label = ss.str();
            sf::Text txt(font);
            txt.setFont(font);
            txt.setCharacterSize(12);
            txt.setFillColor(sf::Color(220, 220, 228));
            txt.setPosition(sf::Vector2f(static_cast<float>(graph_x) + 6.f, y - 8.f));
            txt.setString(sf::String::fromUtf8(label.begin(), label.end()));
            window->draw(txt);
        }
    }

    // Données à tracer: historique complet (non-glissant)
    const size_t n_total = have_full ? full_loss_history.size() : loss_history.size();
    if (n_total < 2) return;

    // Downsample: un point par pixel (approx) sur l'axe X
    const size_t max_points = static_cast<size_t>(std::max(2.f, plot_w));
    const size_t n_draw = std::min(n_total, max_points);

    auto loss_at = [&](size_t idx) -> float {
        if (have_full) return full_loss_history[idx].loss;
        return loss_history[idx];
    };

    sf::VertexArray line(sf::PrimitiveType::LineStrip, n_draw);
    for (size_t i = 0; i < n_draw; ++i) {
        const size_t idx = (n_draw <= 1) ? 0 : (i * (n_total - 1)) / (n_draw - 1);
        const float loss = loss_at(idx);
        const float x = plot_x + (static_cast<float>(i) / static_cast<float>(n_draw - 1)) * plot_w;
        const float normalized = (loss - min_loss) / range;
        const float y = plot_y + (1.f - std::clamp(normalized, 0.f, 1.f)) * plot_h;

        line[i].position = sf::Vector2f(x, y);
        line[i].color = getLossColor(loss);
    }
    window->draw(line);

    // Labels X (début/fin) + info
    if (font_loaded) {
        const size_t start_step = 0;
        const size_t end_step = (n_total > 0) ? (n_total - 1) : 0;
        auto drawSmall = [&](float x, float y, const std::string& s) {
            sf::Text t(font);
            t.setFont(font);
            t.setCharacterSize(12);
            t.setFillColor(sf::Color(220, 220, 228));
            t.setPosition(sf::Vector2f(x, y));
            t.setString(sf::String::fromUtf8(s.begin(), s.end()));
            window->draw(t);
        };

        drawSmall(plot_x, plot_y + plot_h + 4.f, std::to_string(start_step));
        const std::string end_s = std::to_string(end_step);
        drawSmall(plot_x + plot_w - 8.f * static_cast<float>(end_s.size()), plot_y + plot_h + 4.f, end_s);

        // Titre court / rappel
        std::ostringstream ss;
        ss << "steps=" << n_total;
        ss << "  min=" << std::fixed << std::setprecision(4) << min_loss;
        ss << "  max=" << std::fixed << std::setprecision(4) << max_loss;
        drawSmall(plot_x, static_cast<float>(graph_y) - 16.f, ss.str());
    }
}

void Visualizer::renderMetrics() {
    // Affichage des métriques textuelles (simulé avec rectangles colorés)
    const auto area = panelContentRect(PanelId::Metrics);
    const int panel_w = std::max(120, static_cast<int>(area.size.x));
    int metrics_x = static_cast<int>(area.position.x);
    int metrics_y = static_cast<int>(area.position.y);
    int line_height = 25;

    // Bouton stop: hitbox recalculée à chaque frame.
    last_stop_button_rect_.reset();

    // Live tuning controls: hitboxes recalculées à chaque frame.
    last_live_lr_track_.reset();
    last_live_lr_thumb_.reset();
    last_live_lrwu_track_.reset();
    last_live_lrwu_thumb_.reset();
    last_live_klb_track_.reset();
    last_live_klb_thumb_.reset();
    last_live_klwu_track_.reset();
    last_live_klwu_thumb_.reset();
    last_live_lr_value_box_.reset();
    last_live_lrwu_value_box_.reset();
    last_live_klb_value_box_.reset();
    last_live_klwu_value_box_.reset();
    last_live_kl_enable_box_.reset();
    last_live_overrides_box_.reset();

    auto drawMetricBox = [&](int y_offset, sf::Color color) {
        sf::RectangleShape box(sf::Vector2f(static_cast<float>(panel_w), 20));
        box.setPosition(sf::Vector2f(metrics_x, metrics_y + y_offset));
        box.setFillColor(color);
        window->draw(box);
    };

    // Epoch (bleu)
    drawMetricBox(0, sf::Color(50, 100, 180, 200));

    // Batch (vert)
    drawMetricBox(line_height, sf::Color(50, 150, 80, 200));

    // Loss (orange/rouge selon valeur)
    sf::Color loss_color = current_loss > 100.0f ? sf::Color(200, 80, 50, 200) : sf::Color(180, 140, 50, 200);
    drawMetricBox(line_height * 2, loss_color);

    // Learning rate (violet)
    drawMetricBox(line_height * 3, sf::Color(140, 80, 180, 200));

    // Bouton STOP (rouge) — dans le panneau Metrics (layer metrics)
    {
        const int pad = 6;
        const int btn_h = 20;
        const int btn_w = std::min(160, std::max(80, panel_w - 2 * pad));
        const int btn_x = metrics_x + panel_w - btn_w;
        const int btn_y = metrics_y;

        const bool stop_requested = stop_training_requested_.load(std::memory_order_relaxed);

        sf::RectangleShape b(sf::Vector2f((float)btn_w, (float)btn_h));
        b.setPosition(sf::Vector2f((float)btn_x, (float)btn_y));
        b.setFillColor(stop_requested ? sf::Color(180, 55, 55, 235) : sf::Color(140, 40, 40, 230));
        b.setOutlineColor(stop_requested ? sf::Color(245, 150, 150, 245) : sf::Color(210, 90, 90, 240));
        b.setOutlineThickness(1);
        window->draw(b);

        last_stop_button_rect_ = make_rect((float)btn_x, (float)btn_y, (float)btn_w, (float)btn_h);

        if (font_loaded) {
            sf::Text t(font);
            t.setFont(font);
            t.setCharacterSize(13);
            t.setFillColor(sf::Color(245, 240, 240));
            t.setPosition(sf::Vector2f((float)btn_x + 8.f, (float)btn_y + 1.f));
            t.setString(stop_requested ? "STOP demandé" : "STOP training");
            window->draw(t);
        }
    }

    // Texte best-effort
    if (font_loaded) {
        auto wrap_lines = [](const std::string& s, size_t max_chars, int max_lines) {
            std::vector<std::string> out;
            if (max_chars == 0) return out;
            const bool limit_lines = (max_lines > 0);
            std::string cur;
            cur.reserve(max_chars);
            size_t i = 0;
            while (i < s.size() && (!limit_lines || static_cast<int>(out.size()) < max_lines)) {
                // skip leading spaces
                while (i < s.size() && (s[i] == ' ' || s[i] == '\t')) ++i;
                if (i >= s.size()) break;
                cur.clear();
                while (i < s.size() && cur.size() < max_chars) {
                    const char c = s[i++];
                    if (c == '\n' || c == '\r') break;
                    cur.push_back(c);
                }
                if (!cur.empty()) out.push_back(cur);
                // skip until newline if we broke early
                while (i < s.size() && s[i] != '\n' && s[i] != '\r') {
                    if (cur.size() >= max_chars) break;
                    ++i;
                }
                while (i < s.size() && (s[i] == '\n' || s[i] == '\r')) ++i;
            }
            return out;
        };

        auto drawText = [&](int y_offset, const std::string& s) {
            sf::Text t(font);
            t.setFont(font);
            t.setCharacterSize(14);
            t.setFillColor(sf::Color(230, 230, 235));
            t.setPosition(sf::Vector2f(static_cast<float>(metrics_x + 6), static_cast<float>(metrics_y + y_offset - 2)));
            t.setString(sf::String::fromUtf8(s.begin(), s.end()));
            window->draw(t);
        };

        drawText(0, "Epoch " + std::to_string(current_epoch) + "/" + std::to_string(std::max(1, current_total_epochs)));
        drawText(line_height, "Batch " + std::to_string(current_batch) + "/" + std::to_string(std::max(1, current_total_batches)));
        {
            const float diffabs = std::sqrt(std::max(0.0f, current_mse));
            drawText(line_height * 2,
                     "Loss=" + format_decimal(current_loss, 8) +
                         " avg=" + format_decimal(current_avg_loss, 8) +
                         " diffabs=" + format_decimal(diffabs, 8) +
                         " ent=" + format_decimal(current_ent, 8));
        }
        { drawText(line_height * 3, "LR=" + format_decimal(current_lr, 10)); }

        // -------------------------
        // Live tuning controls
        // -------------------------
        if (!live_ui_inited_) {
            live_ui_lr_ = std::clamp(current_lr, kLiveLRMin, kLiveLRMax);
            live_ui_lr_warmup_steps_ = 0;
            live_ui_kl_beta_ = std::clamp(current_kl_beta_effective, 0.0f, 1.0f);
            live_ui_kl_warmup_steps_ = 0;
            live_ui_kl_enabled_ = (live_ui_kl_beta_ > 0.0f);

            // Démarre en mode NATIVE: aucun override tant que l'utilisateur n'interagit pas.
            live_overrides_enabled_.store(false, std::memory_order_relaxed);
            live_lr_.store(live_ui_lr_, std::memory_order_relaxed);
            live_lr_warmup_steps_.store(live_ui_lr_warmup_steps_, std::memory_order_relaxed);
            live_kl_beta_.store(live_ui_kl_beta_, std::memory_order_relaxed);
            live_kl_warmup_steps_.store(live_ui_kl_warmup_steps_, std::memory_order_relaxed);
            live_kl_enabled_.store(live_ui_kl_enabled_, std::memory_order_relaxed);
            // Ne pas bump la version: on ne veut pas changer le training sans interaction.
            live_ui_inited_ = true;
        }

        // PS (user): MAJ en temps réel des scrollers (sliders) avec les valeurs runtime associées.
        // En mode NATIVE, refléter current_lr/current_kl_beta_effective tant qu'on ne drag pas.
        const bool live_on_rt = live_overrides_enabled_.load(std::memory_order_relaxed);
        if (!live_on_rt && live_dragging_ == LiveDragTarget::None) {
            live_ui_lr_ = std::clamp(current_lr, kLiveLRMin, kLiveLRMax);
            live_ui_lr_warmup_steps_ = 0;
            live_ui_kl_beta_ = std::clamp(current_kl_beta_effective, 0.0f, 1.0f);
            live_ui_kl_warmup_steps_ = 0;
            live_ui_kl_enabled_ = (live_ui_kl_beta_ > 0.0f);
        }

        const int live_y0 = line_height * 4 + 6;
        const int live_row_h = 42;
        const int label_w = 82;
        const int value_w = 110;
        const float track_h = 10.f;
        const float thumb_w = 10.f;
        const float thumb_h = 16.f;
        const int rows = 5;

        auto drawSliderRow = [&](int row,
                                 const std::string& label,
                                 float t,
                                 const std::string& value,
                                 LiveInputTarget input_target,
                                 const std::string& min_label,
                                 const std::string& max_label,
                                 std::optional<sf::FloatRect>& out_track,
                                 std::optional<sf::FloatRect>& out_thumb,
                                 std::optional<sf::FloatRect>& out_value_box) {
            const float y = static_cast<float>(metrics_y + live_y0 + row * live_row_h);
            const float x0 = static_cast<float>(metrics_x + 6);
            const float x_track = static_cast<float>(metrics_x + label_w);
            const float x_value = static_cast<float>(metrics_x + panel_w - value_w);
            const float w_track = std::max(20.f, x_value - x_track - 10.f);

            // Label
            {
                sf::Text tt(font);
                tt.setFont(font);
                tt.setCharacterSize(13);
                tt.setFillColor(sf::Color(225, 225, 235));
                tt.setPosition(sf::Vector2f(x0, y - 2.f));
                tt.setString(sf::String::fromUtf8(label.begin(), label.end()));
                window->draw(tt);
            }

            // Track
            const float tt = clamp01(t);
            const float track_y = y + 7.f;
            sf::RectangleShape track(sf::Vector2f(w_track, track_h));
            track.setPosition(sf::Vector2f(x_track, track_y));
            track.setFillColor(sf::Color(55, 55, 60, 210));
            track.setOutlineColor(sf::Color(90, 90, 95, 220));
            track.setOutlineThickness(1);
            window->draw(track);

            sf::RectangleShape fill(sf::Vector2f(w_track * tt, track_h));
            fill.setPosition(sf::Vector2f(x_track, track_y));
            fill.setFillColor(sf::Color(90, 140, 200, 220));
            window->draw(fill);

            // Graduation: 5 repères visibles pour guider le réglage.
            for (int gi = 0; gi <= 4; ++gi) {
                const float gt = static_cast<float>(gi) / 4.0f;
                const float gx = x_track + gt * w_track;
                sf::RectangleShape tick(sf::Vector2f(1.0f, 5.0f));
                tick.setPosition(sf::Vector2f(gx, track_y + track_h + 1.0f));
                tick.setFillColor(sf::Color(145, 145, 160, 210));
                window->draw(tick);
            }

            if (font_loaded) {
                sf::Text tmin(font);
                tmin.setFont(font);
                tmin.setCharacterSize(10);
                tmin.setFillColor(sf::Color(178, 178, 190));
                tmin.setPosition(sf::Vector2f(x_track, track_y + track_h + 7.0f));
                tmin.setString(sf::String::fromUtf8(min_label.begin(), min_label.end()));
                window->draw(tmin);

                sf::Text tmax(font);
                tmax.setFont(font);
                tmax.setCharacterSize(10);
                tmax.setFillColor(sf::Color(178, 178, 190));
                tmax.setPosition(sf::Vector2f(x_track + w_track - 8.0f * static_cast<float>(max_label.size()), track_y + track_h + 7.0f));
                tmax.setString(sf::String::fromUtf8(max_label.begin(), max_label.end()));
                window->draw(tmax);
            }

            const float thumb_x = x_track + w_track * tt - thumb_w * 0.5f;
            sf::RectangleShape thumb(sf::Vector2f(thumb_w, thumb_h));
            thumb.setPosition(sf::Vector2f(thumb_x, track_y + track_h * 0.5f - thumb_h * 0.5f));
            thumb.setFillColor(sf::Color(210, 210, 220, 235));
            thumb.setOutlineColor(sf::Color(30, 30, 30, 220));
            thumb.setOutlineThickness(1);
            window->draw(thumb);

            out_track = make_rect(x_track, track_y, w_track, track_h);
            out_thumb = make_rect(thumb.getPosition().x, thumb.getPosition().y, thumb_w, thumb_h);

            // Cellule de saisie valeur (clic pour éditer)
            {
                const bool editing = (live_input_target_ == input_target);
                const bool invalid = editing && (live_input_error_until_ms_ > steady_now_ms());
                const float vb_h = 18.0f;
                const float vb_y = y + 1.0f;

                sf::RectangleShape vb(sf::Vector2f(static_cast<float>(value_w), vb_h));
                vb.setPosition(sf::Vector2f(x_value, vb_y));
                vb.setFillColor(editing ? sf::Color(44, 58, 74, 225) : sf::Color(34, 36, 44, 215));
                vb.setOutlineColor(invalid ? sf::Color(230, 110, 110, 235) : sf::Color(98, 104, 120, 225));
                vb.setOutlineThickness(1.0f);
                window->draw(vb);

                out_value_box = make_rect(x_value, vb_y, static_cast<float>(value_w), vb_h);

                sf::Text tt(font);
                tt.setFont(font);
                tt.setCharacterSize(13);
                tt.setFillColor(sf::Color(225, 225, 235));
                tt.setPosition(sf::Vector2f(x_value + 4.0f, y - 1.f));

                std::string shown = editing ? live_input_buffer_ : value;
                if (editing) shown += "_";
                tt.setString(sf::String::fromUtf8(shown.begin(), shown.end()));
                window->draw(tt);
            }
        };

        // Indicateur LIVE/NATIVE (cliquable)
        {
            const bool live_on = live_overrides_enabled_.load(std::memory_order_relaxed);
            const float y = static_cast<float>(metrics_y + live_y0 - 22);
            const float box_w = 130.f;
            const float box_h = 18.f;
            const float box_x = static_cast<float>(metrics_x + panel_w) - box_w - 6.f;
            const float box_y = y;

            sf::RectangleShape b(sf::Vector2f(box_w, box_h));
            b.setPosition(sf::Vector2f(box_x, box_y));
            b.setFillColor(live_on ? sf::Color(50, 110, 70, 220) : sf::Color(60, 60, 65, 210));
            b.setOutlineColor(live_on ? sf::Color(130, 220, 160, 235) : sf::Color(100, 100, 105, 220));
            b.setOutlineThickness(1);
            window->draw(b);

            last_live_overrides_box_ = make_rect(box_x, box_y, box_w, box_h);

            sf::Text tt(font);
            tt.setFont(font);
            tt.setCharacterSize(12);
            tt.setFillColor(sf::Color(245, 245, 248));
            tt.setPosition(sf::Vector2f(box_x + 6.f, box_y + 1.f));
            tt.setString(live_on ? "LIVE (click -> NATIVE)" : "NATIVE (click -> LIVE)");
            window->draw(tt);

            // Petit suffixe version (debug visuel)
            const uint64_t ver = live_params_version_.load(std::memory_order_relaxed);
            sf::Text vv(font);
            vv.setFont(font);
            vv.setCharacterSize(12);
            vv.setFillColor(sf::Color(210, 210, 215));
            vv.setPosition(sf::Vector2f(box_x + box_w - 48.f, box_y + 1.f));
            vv.setString("v=" + std::to_string(ver));
            window->draw(vv);
        }

        // LR (log scale)
        {
            drawSliderRow(0,
                          "LR",
                          t_from_lr(live_ui_lr_),
                          format_decimal(live_ui_lr_, 10),
                          LiveInputTarget::LR,
                          "1e-7",
                          "1e-3",
                          last_live_lr_track_,
                          last_live_lr_thumb_,
                          last_live_lr_value_box_);
        }

        // LR warmup
        {
            drawSliderRow(1,
                          "LR wu",
                          t_from_warmup(live_ui_lr_warmup_steps_),
                          std::to_string(live_ui_lr_warmup_steps_) + " steps",
                          LiveInputTarget::LRWarmup,
                          "0",
                          "5e4",
                          last_live_lrwu_track_,
                          last_live_lrwu_thumb_,
                          last_live_lrwu_value_box_);
        }

        // KL beta
        {
            drawSliderRow(2,
                          "KL beta",
                          t_from_klbeta(live_ui_kl_beta_),
                          format_decimal(live_ui_kl_beta_, 8),
                          LiveInputTarget::KLBeta,
                          "0",
                          "1",
                          last_live_klb_track_,
                          last_live_klb_thumb_,
                          last_live_klb_value_box_);
        }

        // KL warmup
        {
            drawSliderRow(3,
                          "KL wu",
                          t_from_warmup(live_ui_kl_warmup_steps_),
                          std::to_string(live_ui_kl_warmup_steps_) + " steps",
                          LiveInputTarget::KLWarmup,
                          "0",
                          "5e4",
                          last_live_klwu_track_,
                          last_live_klwu_thumb_,
                          last_live_klwu_value_box_);
        }

        // KL enabled checkbox
        {
            const float y = static_cast<float>(metrics_y + live_y0 + 4 * live_row_h);
            const float x0 = static_cast<float>(metrics_x + 6);
            const float box = 14.f;
            const float box_x = static_cast<float>(metrics_x + label_w);
            const float box_y = y + 4.f;

            sf::Text tt(font);
            tt.setFont(font);
            tt.setCharacterSize(13);
            tt.setFillColor(sf::Color(225, 225, 235));
            tt.setPosition(sf::Vector2f(x0, y - 2.f));
            tt.setString("KL");
            window->draw(tt);

            sf::RectangleShape cb(sf::Vector2f(box, box));
            cb.setPosition(sf::Vector2f(box_x, box_y));
            cb.setFillColor(sf::Color(55, 55, 60, 210));
            cb.setOutlineColor(sf::Color(90, 90, 95, 220));
            cb.setOutlineThickness(1);
            window->draw(cb);

            if (live_ui_kl_enabled_) {
                sf::RectangleShape mark(sf::Vector2f(box - 4.f, box - 4.f));
                mark.setPosition(sf::Vector2f(box_x + 2.f, box_y + 2.f));
                mark.setFillColor(sf::Color(90, 200, 120, 220));
                window->draw(mark);
            }

            last_live_kl_enable_box_ = make_rect(box_x, box_y, box, box);

            sf::Text st(font);
            st.setFont(font);
            st.setCharacterSize(13);
            st.setFillColor(sf::Color(225, 225, 235));
            st.setPosition(sf::Vector2f(box_x + box + 8.f, y - 2.f));
            st.setString(live_ui_kl_enabled_ ? "enabled" : "disabled");
            window->draw(st);
        }

        int extra_y = live_y0 + rows * live_row_h + 10;
        {
            std::string s = "t=" + format_decimal(current_timestep, 8) + " mse=" + format_decimal(current_mse, 8);
            if (!current_recon_loss_type.empty()) {
                s = "recon=" + current_recon_loss_type + "  " + s;
            }
            drawText(extra_y, s);
        }

        if (current_kl_beta_effective > 0.0f) {
            drawText(extra_y + line_height, "KL beta_eff=" + format_decimal(current_kl_beta_effective, 8) + "  kl=" + format_decimal(current_kl, 8));
            extra_y += line_height;
        }
        drawText(extra_y + line_height, "grad=" + format_decimal(current_grad_norm, 8) + " max=" + format_decimal(current_grad_max, 8));
        drawText(extra_y + 2 * line_height, "time=" + std::to_string(current_batch_time_ms) + "ms bps=" + format_decimal(current_bps, 8));
        drawText(extra_y + 3 * line_height, "mem=" + std::to_string(current_memory_mb) + "MB params=" + std::to_string(current_params));
        drawText(extra_y + 4 * line_height, "opt step=" + std::to_string(current_opt_step) + " wd=" + format_decimal(current_opt_weight_decay, 10));

        {
            std::string s;
            if (current_val_in_progress) {
                s = "Val: EN COURS";
                if (current_val_step > 0) s += " @" + std::to_string(current_val_step);
                if (current_val_total > 0) {
                    s += " " + std::to_string(std::clamp(current_val_done, 0, current_val_total)) + "/" + std::to_string(current_val_total);
                } else if (current_val_done > 0) {
                    s += " done=" + std::to_string(current_val_done);
                }
            } else if (!current_val_has) {
                s = "Val: —";
            } else {
                s = "Val@" + std::to_string(current_val_step) + " " + (current_val_ok ? "OK" : "FAIL") + " items=" + std::to_string(current_val_items);
            }
            drawText(extra_y + 5 * line_height, s);

            // Afficher les métriques de validation dès qu'on en a (même partiellement).
            if (current_val_has || current_val_in_progress) {
                std::ostringstream ss;
                ss << "val recon=" << format_decimal(current_val_recon, 8) << " kl=" << format_decimal(current_val_kl, 8);
                if (std::fabs(current_val_align) > 1e-9f) {
                    ss << " align=" << format_decimal(current_val_align, 8);
                }
                drawText(extra_y + 6 * line_height, ss.str());
            }
        }

        // Texte dataset (si dispo): prompt brut + tokens + résumé encodage.
        if (show_prompt_text_ && has_dataset_text) {
            int y = extra_y + 8 * line_height;
            const int y_max = static_cast<int>(area.position.y + area.size.y) - line_height;
            const size_t wrap_chars = static_cast<size_t>(std::max(30, panel_w / 8));

            auto drawWrapped = [&](const std::string& s, int max_lines_hint) {
                if (s.empty()) return;
                const int avail_lines = std::max(0, (y_max - (metrics_y + y)) / line_height);
                const int max_lines = (max_lines_hint > 0) ? std::min(max_lines_hint, avail_lines) : avail_lines;
                const auto lines = wrap_lines(s, wrap_chars, max_lines);
                for (const auto& ln : lines) {
                    if (metrics_y + y > y_max) break;
                    drawText(y, ln);
                    y += line_height;
                }
            };

            // Afficher autant que possible sans tronquer la source.
            // (La zone visible est bornée par la hauteur de la fenêtre.)
            drawWrapped("prompt: " + dataset_text_raw, 0);
            drawWrapped("tags: " + dataset_text_tags, 4);
            drawWrapped("tokens: " + dataset_text_tokens, 4);
            drawWrapped("enc: " + dataset_text_encoded, 3);
        }
    }
}

void Visualizer::createImageTexture(ImageData& img_data, int w, int h, int channels, int display_size) {
    // Créer une image SFML à partir des pixels (grayscale/RGB/RGBA)
    if (w <= 0 || h <= 0) return;
    if (channels != 1 && channels != 3 && channels != 4) return;

    if (window && window->isOpen()) {
        // Upload texture garanti avec contexte actif (évite textures noires selon WM/driver).
        window->setActive(true);
    }

    img_data.w = w;
    img_data.h = h;
    img_data.channels = channels;
    img_data.display_size = display_size;

    sf::Image sfml_image;
    sfml_image.resize(sf::Vector2u(static_cast<unsigned>(w), static_cast<unsigned>(h)));

    // Si la taille ne colle pas exactement aux canaux annoncés, tenter une correction
    // best-effort pour éviter des previews noires au premier rendu.
    const size_t area = static_cast<size_t>(w) * static_cast<size_t>(h);
    const size_t declared_stride = static_cast<size_t>(channels);
    const size_t expected_declared = area * declared_stride;
    if (img_data.pixels.size() != expected_declared) {
        const size_t n = img_data.pixels.size();
        if (n == area * 4ULL) channels = 4;
        else if (n == area * 3ULL) channels = 3;
        else if (n == area) channels = 1;
        img_data.channels = channels;
    }

    const size_t stride = static_cast<size_t>(channels);
    const size_t expected = area * stride;
    const size_t n = std::min(expected, img_data.pixels.size());

    auto at = [&](size_t idx) -> uint8_t {
        return (idx < n) ? img_data.pixels[idx] : 0;
    };

    bool force_opaque_alpha = false;
    if (channels == 4) {
        bool any_alpha = false;
        for (size_t i = 3; i < n; i += 4) {
            if (img_data.pixels[i] != 0) {
                any_alpha = true;
                break;
            }
        }
        force_opaque_alpha = !any_alpha;
    }

    for (int yy = 0; yy < h; ++yy) {
        for (int xx = 0; xx < w; ++xx) {
            const size_t base = (static_cast<size_t>(yy) * static_cast<size_t>(w) + static_cast<size_t>(xx)) * stride;
            sf::Color c;
            if (channels == 1) {
                const uint8_t g = at(base);
                c = sf::Color(g, g, g, 255);
            } else if (channels == 3) {
                c = sf::Color(at(base + 0), at(base + 1), at(base + 2), 255);
            } else {
                c = sf::Color(at(base + 0), at(base + 1), at(base + 2), force_opaque_alpha ? 255 : at(base + 3));
            }
            sfml_image.setPixel(sf::Vector2u(static_cast<unsigned int>(static_cast<unsigned>(xx)), static_cast<unsigned int>(static_cast<unsigned>(yy))), c);
        }
    }

    bool loaded = img_data.texture.loadFromImage(sfml_image);
    if (!loaded) {
        // Fallback: downscale agressif puis retry (utile si la texture dépasse les limites GPU/driver).
        const unsigned max_tex = sf::Texture::getMaximumSize();
        const unsigned src_w = static_cast<unsigned>(std::max(1, w));
        const unsigned src_h = static_cast<unsigned>(std::max(1, h));

        const unsigned hard_cap = std::max(64u, std::min(1024u, max_tex > 0 ? max_tex : 1024u));
        const unsigned dw = std::max(1u, std::min(src_w, hard_cap));
        const unsigned dh = std::max(1u, std::min(src_h, hard_cap));

        if (dw != src_w || dh != src_h) {
            sf::Image down;
            down.resize(sf::Vector2u(dw, dh));
            for (unsigned yy = 0; yy < dh; ++yy) {
                const unsigned sy = (yy * src_h) / dh;
                for (unsigned xx = 0; xx < dw; ++xx) {
                    const unsigned sx = (xx * src_w) / dw;
                    down.setPixel(sf::Vector2u(static_cast<unsigned int>(xx), static_cast<unsigned int>(yy)), sfml_image.getPixel(sf::Vector2u(static_cast<unsigned int>(sx), static_cast<unsigned int>(sy))));
                }
            }
            loaded = img_data.texture.loadFromImage(down);
        }
    }

    if (!loaded) {
        return;
    }

    img_data.texture.setSmooth(true);
    img_data.sprite.setTexture(img_data.texture, true);

    // Mettre à l'échelle pour affichage (fit-to-square)
    const float sx = static_cast<float>(display_size) / static_cast<float>(w);
    const float sy = static_cast<float>(display_size) / static_cast<float>(h);
    const float scale = std::min(sx, sy);
    img_data.sprite.setScale(sf::Vector2f(scale, scale));
}

sf::Color Visualizer::getLossColor(float loss) {
    // Gradient vert -> jaune -> rouge basé sur la loss
    if (loss < 50.0f) {
        return sf::Color(100, 200, 100); // Vert
    } else if (loss < 150.0f) {
        return sf::Color(200, 200, 100); // Jaune
    } else {
        return sf::Color(200, 100, 100); // Rouge
    }
}

void Visualizer::saveLossHistory(const std::string& filepath) const {
    
    std::ofstream file(filepath);
    
    if (!file.is_open()) {
        std::cerr << "Erreur: impossible d'ouvrir " << filepath << " pour écriture" << std::endl;
        return;
    }
    
    // En-tête CSV (métriques complètes)
    file << "step,epoch,total_epochs,batch,total_batches,loss,avg_loss,learning_rate,batch_time_ms,bps,memory_mb,params,mse,kl_divergence,wasserstein,entropy_diff,moment_mismatch,spatial_coherence,temporal_consistency,timestep,grad_norm,grad_max,opt_type,opt_step,opt_beta1,opt_beta2,opt_eps,opt_weight_decay,val_loss,val_mse,val_step" << std::endl;
    
    // Écrire tout l'historique complet (toutes les epochs et tous les steps)
    for (const auto& record : full_loss_history) {
        file << record.step << "," 
             << record.epoch << "," 
             << record.total_epochs << ","
             << record.batch << "," 
             << record.total_batches << ","
             << std::fixed << std::setprecision(6) << record.loss << ","
             << record.avg_loss << ","
             << std::scientific << record.lr << ","
             << std::fixed << record.batch_time_ms << ","
             << record.bps << ","
             << record.memory_mb << ","
             << record.params << ","
             << std::fixed << std::setprecision(6) << record.mse << ","
             << record.kl_divergence << ","
             << record.wasserstein << ","
             << record.entropy_diff << ","
             << record.moment_mismatch << ","
             << record.spatial_coherence << ","
             << record.temporal_consistency << ","
             << record.timestep << ","
             << record.grad_norm << ","
             << record.grad_max << ","
             << record.opt_type << ","
             << record.opt_step << ","
             << record.opt_beta1 << ","
             << record.opt_beta2 << ","
             // opt_eps est souvent ~1e-8 : en fixed(6) ça apparaît comme 0.000000.
             // On l'encode en scientifique pour préserver l'information.
             << std::scientific << std::setprecision(8) << record.opt_eps << ","
             << std::fixed << std::setprecision(6) << record.opt_weight_decay;
        // Colonnes de validation : vides si ce step n'est pas un step de validation.
        if (record.is_val) {
            file << "," << record.val_loss
                 << "," << record.val_mse
                 << "," << record.val_step_id;
        } else {
            file << ",,,";
        }
        file << std::endl;
    }

    file.close();
}
