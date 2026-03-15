#include "NeuroPulse.hpp"

#include "../Sha256.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

static constexpr double kPi = 3.141592653589793238462643383279502884;

static inline double clampd(double v, double lo, double hi) {
    return std::max(lo, std::min(hi, v));
}

static bool hex_to_bytes(const std::string& hex, std::vector<uint8_t>* out) {
    if (!out) return false;
    out->clear();
    if (hex.size() % 2 != 0) return false;

    auto hexval = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return 10 + (c - 'a');
        if (c >= 'A' && c <= 'F') return 10 + (c - 'A');
        return -1;
    };

    out->reserve(hex.size() / 2);
    for (size_t i = 0; i < hex.size(); i += 2) {
        int hi = hexval(hex[i]);
        int lo = hexval(hex[i + 1]);
        if (hi < 0 || lo < 0) return false;
        out->push_back(static_cast<uint8_t>((hi << 4) | lo));
    }
    return true;
}

static NeuroPulseModel::Band band_from_string(const std::string& s, bool* out_ok) {
    std::string low = s;
    std::transform(low.begin(), low.end(), low.begin(), [](unsigned char c) { return (char)std::tolower(c); });

    if (low == "delta") { if (out_ok) *out_ok = true; return NeuroPulseModel::Band::Delta; }
    if (low == "theta") { if (out_ok) *out_ok = true; return NeuroPulseModel::Band::Theta; }
    if (low == "alpha") { if (out_ok) *out_ok = true; return NeuroPulseModel::Band::Alpha; }
    if (low == "beta")  { if (out_ok) *out_ok = true; return NeuroPulseModel::Band::Beta; }
    if (low == "gamma") { if (out_ok) *out_ok = true; return NeuroPulseModel::Band::Gamma; }
    if (out_ok) *out_ok = false;
    return NeuroPulseModel::Band::Alpha;
}

static const char* band_to_string(NeuroPulseModel::Band b) {
    switch (b) {
        case NeuroPulseModel::Band::Delta: return "delta";
        case NeuroPulseModel::Band::Theta: return "theta";
        case NeuroPulseModel::Band::Alpha: return "alpha";
        case NeuroPulseModel::Band::Beta:  return "beta";
        case NeuroPulseModel::Band::Gamma: return "gamma";
    }
    return "alpha";
}

static std::pair<double, double> band_range_hz(NeuroPulseModel::Band b) {
    switch (b) {
        case NeuroPulseModel::Band::Delta: return {0.5, 4.0};
        case NeuroPulseModel::Band::Theta: return {4.0, 8.0};
        case NeuroPulseModel::Band::Alpha: return {8.0, 12.0};
        case NeuroPulseModel::Band::Beta:  return {13.0, 30.0};
        case NeuroPulseModel::Band::Gamma: return {30.0, 48.0};
    }
    return {8.0, 12.0};
}

static NeuroPulseModel::Band pick_band_auto(const std::vector<uint8_t>& digest_bytes) {
    if (digest_bytes.empty()) return NeuroPulseModel::Band::Alpha;
    const uint8_t v = digest_bytes[0];
    switch (v % 5) {
        case 0: return NeuroPulseModel::Band::Delta;
        case 1: return NeuroPulseModel::Band::Theta;
        case 2: return NeuroPulseModel::Band::Alpha;
        case 3: return NeuroPulseModel::Band::Beta;
        case 4: return NeuroPulseModel::Band::Gamma;
    }
    return NeuroPulseModel::Band::Alpha;
}

static double pick_hz_in_band(NeuroPulseModel::Band b, const std::vector<uint8_t>& digest_bytes) {
    const auto range = band_range_hz(b);
    const double lo = range.first;
    const double hi = range.second;
    const uint8_t v = digest_bytes.size() > 1 ? digest_bytes[1] : 128;
    const double u = (double)v / 255.0;
    return lo + u * (hi - lo);
}

static void write_u16_le(std::ofstream& f, uint16_t v) {
    const unsigned char b[2] = {
        (unsigned char)(v & 0xFFu),
        (unsigned char)((v >> 8) & 0xFFu),
    };
    f.write(reinterpret_cast<const char*>(b), 2);
}

static void write_u32_le(std::ofstream& f, uint32_t v) {
    const unsigned char b[4] = {
        (unsigned char)(v & 0xFFu),
        (unsigned char)((v >> 8) & 0xFFu),
        (unsigned char)((v >> 16) & 0xFFu),
        (unsigned char)((v >> 24) & 0xFFu),
    };
    f.write(reinterpret_cast<const char*>(b), 4);
}

static bool write_wav_pcm16_stereo(const std::string& path, int sample_rate, const std::vector<int16_t>& interleaved_lr, std::string* out_err) {
    if (sample_rate <= 0) {
        if (out_err) *out_err = "sample_rate invalide";
        return false;
    }
    if (interleaved_lr.size() % 2 != 0) {
        if (out_err) *out_err = "buffer audio invalide (stereo interleaved)";
        return false;
    }

    std::ofstream f(path, std::ios::binary);
    if (!f) {
        if (out_err) *out_err = "impossible d'ouvrir le fichier WAV: " + path;
        return false;
    }

    const uint16_t num_channels = 2;
    const uint16_t bits_per_sample = 16;
    const uint32_t byte_rate = (uint32_t)sample_rate * num_channels * (bits_per_sample / 8);
    const uint16_t block_align = num_channels * (bits_per_sample / 8);
    const uint32_t data_bytes = (uint32_t)(interleaved_lr.size() * sizeof(int16_t));

    f.write("RIFF", 4);
    const uint32_t riff_chunk_size = 36u + data_bytes;
    write_u32_le(f, riff_chunk_size);
    f.write("WAVE", 4);

    f.write("fmt ", 4);
    write_u32_le(f, 16u);
    write_u16_le(f, 1u);
    write_u16_le(f, num_channels);
    write_u32_le(f, (uint32_t)sample_rate);
    write_u32_le(f, byte_rate);
    write_u16_le(f, block_align);
    write_u16_le(f, bits_per_sample);

    f.write("data", 4);
    write_u32_le(f, data_bytes);
    f.write(reinterpret_cast<const char*>(interleaved_lr.data()), (std::streamsize)data_bytes);

    if (!f.good()) {
        if (out_err) *out_err = "erreur d'écriture WAV";
        return false;
    }
    return true;
}

static bool write_light_csv(const std::string& path, int fps, double duration_s, double light_hz, double depth, double phase, std::string* out_err) {
    if (fps <= 0) {
        if (out_err) *out_err = "light_fps invalide";
        return false;
    }
    if (!(duration_s > 0.0)) {
        if (out_err) *out_err = "duration_s invalide";
        return false;
    }

    std::ofstream f(path);
    if (!f) {
        if (out_err) *out_err = "impossible d'ouvrir le CSV lumière: " + path;
        return false;
    }

    f << "t_s,intensity\n";
    const int total = (int)std::ceil(duration_s * (double)fps);
    const double w = 2.0 * kPi * light_hz;

    for (int i = 0; i < total; ++i) {
        const double t = (double)i / (double)fps;
        const double s = 0.5 * (1.0 + std::sin(w * t + phase));
        const double intensity = clampd(0.5 + depth * (s - 0.5), 0.0, 1.0);
        f << t << ',' << intensity << '\n';
    }

    if (!f.good()) {
        if (out_err) *out_err = "erreur d'écriture CSV lumière";
        return false;
    }
    return true;
}

static bool write_light_csv_series(const std::string& path, int fps, const std::vector<double>& intensity, std::string* out_err) {
    if (fps <= 0) {
        if (out_err) *out_err = "light_fps invalide";
        return false;
    }
    std::ofstream f(path);
    if (!f) {
        if (out_err) *out_err = "impossible d'ouvrir le CSV lumière: " + path;
        return false;
    }
    f << "t_s,intensity\n";
    for (size_t i = 0; i < intensity.size(); ++i) {
        const double t = (double)i / (double)fps;
        f << t << ',' << clampd(intensity[i], 0.0, 1.0) << '\n';
    }
    if (!f.good()) {
        if (out_err) *out_err = "erreur d'écriture CSV lumière";
        return false;
    }
    return true;
}

static inline double lerp(double a, double b, double t) {
    return a + (b - a) * t;
}

static inline double sample_series(const std::vector<double>& s, double t, double fps) {
    if (s.empty() || !(fps > 0.0)) return 0.0;
    const double idx = t * fps;
    const int i0 = (int)std::floor(idx);
    const int i1 = i0 + 1;
    const double frac = idx - (double)i0;
    const int n = (int)s.size();
    if (i0 <= 0) return s.front();
    if (i0 >= n - 1) return s.back();
    return lerp(s[(size_t)i0], s[(size_t)i1], frac);
}

static inline unsigned int seed_from_digest(const std::vector<uint8_t>& d) {
    if (d.size() < 4) return 1u;
    const unsigned int v = ((unsigned int)d[0] << 24) | ((unsigned int)d[1] << 16) | ((unsigned int)d[2] << 8) | (unsigned int)d[3];
    return (v == 0u) ? 1u : v;
}

static inline void fill_text_features_8(const std::vector<uint8_t>& digest, float out8[8]) {
    for (int i = 0; i < 8; ++i) {
        const size_t idx = 4u + (size_t)i;
        const uint8_t b = (idx < digest.size()) ? digest[idx] : (uint8_t)(i * 31);
        out8[i] = (float)((double)b / 127.5 - 1.0); // [-1,1]
    }
}

} // namespace

void NeuroPulseModel::buildFromConfig(const Config& cfg) {
    cfg_ = cfg;

    // Build a tiny internal MLP used as a deterministic control signal generator.
    // Note: weights are allocated/initialized outside (allocateParams/initializeWeights).
    getMutableLayers().clear();
    layer_weight_blocks.clear();
    setModelName("NeuroPulseModel");

    if (!cfg_.organic_nn) {
        // No NN graph; render remains purely analytical.
        return;
    }

    modelConfig["type"] = "neuropulse";
    modelConfig["task"] = "control_mlp";

    constexpr int kInputDim = 16;
    const int hidden = std::max(4, cfg_.nn_hidden_dim);
    const int blocks = std::max(0, cfg_.nn_hidden_layers);
    const float drop = (float)std::clamp(cfg_.nn_dropout, 0.0, 0.95);
    const int out_dim = 2;

    modelConfig["nn_input_dim"] = kInputDim;
    modelConfig["nn_hidden_dim"] = hidden;
    modelConfig["nn_hidden_layers"] = blocks;
    modelConfig["nn_output_dim"] = out_dim;
    modelConfig["nn_dropout"] = drop;

    // Routing input: __input__ -> neuropulse_nn/in
    push("neuropulse_nn/raw_in", "Identity", 0);
    if (auto* L = getLayerByName("neuropulse_nn/raw_in")) {
        L->inputs = {"__input__"};
        L->output = "neuropulse_nn/in";
    }

    int cur_dim = kInputDim;
    std::string cur_tensor = "neuropulse_nn/in";

    for (int i = 0; i < blocks; ++i) {
        const std::string fc = "neuropulse_nn/fc" + std::to_string(i + 1);
        const std::string h = "neuropulse_nn/h" + std::to_string(i + 1);
        const std::string act = h + "_act";

        push(fc, "Linear", static_cast<size_t>(cur_dim) * static_cast<size_t>(hidden) + static_cast<size_t>(hidden));
        if (auto* L = getLayerByName(fc)) {
            L->inputs = {cur_tensor};
            L->output = h;
            L->in_features = cur_dim;
            L->out_features = hidden;
            L->use_bias = true;
        }

        push(fc + "/gelu", "GELU", 0);
        if (auto* L = getLayerByName(fc + "/gelu")) {
            L->inputs = {h};
            L->output = act;
        }

        if (drop > 0.0f) {
            const std::string dn = fc + "/drop";
            const std::string dout = act + "_drop";
            push(dn, "Dropout", 0);
            if (auto* L = getLayerByName(dn)) {
                L->inputs = {act};
                L->output = dout;
                L->dropout_p = drop;
            }
            cur_tensor = dout;
        } else {
            cur_tensor = act;
        }

        cur_dim = hidden;
    }

    push("neuropulse_nn/out", "Linear", static_cast<size_t>(cur_dim) * static_cast<size_t>(out_dim) + static_cast<size_t>(out_dim));
    if (auto* L = getLayerByName("neuropulse_nn/out")) {
        L->inputs = {cur_tensor};
        L->output = "neuropulse_nn/pre_tanh";
        L->in_features = cur_dim;
        L->out_features = out_dim;
        L->use_bias = true;
    }

    push("neuropulse_nn/tanh", "Tanh", 0);
    if (auto* L = getLayerByName("neuropulse_nn/tanh")) {
        L->inputs = {"neuropulse_nn/pre_tanh"};
        L->output = "x";
    }
}

NeuroPulseModel::Result NeuroPulseModel::paramsFromText(const std::string& text, const Config& cfg) {
    Result r;
    r.duration_s = cfg.duration_s;
    r.sample_rate = cfg.sample_rate;
    r.carrier_hz = cfg.carrier_hz;
    r.binaural_hz = cfg.binaural_hz;
    r.out_wav = cfg.out_wav;
    r.out_light_csv = cfg.out_light_csv;
    r.light_fps = cfg.light_fps;

    r.sha256_hex = sha256(text);
    std::vector<uint8_t> digest;
    if (!hex_to_bytes(r.sha256_hex, &digest)) {
        r.warnings.push_back("sha256 hex invalide (interne); fallback alpha 10 Hz");
        r.band = Band::Alpha;
        r.cognitive_hz = 10.0;
        r.light_hz = cfg.safe_light ? clampd(cfg.light_hz, 0.05, 2.0) : cfg.light_hz;
        return r;
    }

    r.organic_nn = cfg.organic_nn;
    r.nn_control_fps = cfg.organic_nn ? std::max(1, cfg.nn_control_fps) : 0;
    r.nn_seed = cfg.organic_nn ? (cfg.nn_seed != 0u ? cfg.nn_seed : seed_from_digest(digest)) : 0u;

    bool band_ok = false;
    Band band = Band::Alpha;
    if (!cfg.cognitive_band.empty() && cfg.cognitive_band != "auto") {
        band = band_from_string(cfg.cognitive_band, &band_ok);
        if (!band_ok) r.warnings.push_back("cognitive_band inconnu; auto appliqué");
    }
    if (!band_ok) {
        band = pick_band_auto(digest);
    }
    r.band = band;

    double hz = cfg.cognitive_hz;
    if (!(hz > 0.0)) {
        hz = pick_hz_in_band(band, digest);
    }
    r.cognitive_hz = clampd(hz, 0.5, 48.0);

    double light_hz = cfg.light_hz;
    if (cfg.safe_light) {
        light_hz = clampd(light_hz, 0.05, 2.0);
        if (cfg.light_hz > 2.0) r.warnings.push_back("light_hz clampé à 2 Hz (safe_light=true)");
    }
    r.light_hz = light_hz;

    if (cfg.safe_light && cfg.light_hz >= 3.0 && cfg.light_hz <= 60.0) {
        r.warnings.push_back("ATTENTION: light_hz dans une zone de scintillement; safe_light recommandé");
    }

    if (cfg.binaural_hz < 0.0) r.warnings.push_back("binaural_hz négatif; ignoré");
    return r;
}

bool NeuroPulseModel::renderToFiles(const std::string& text, const Config& cfg, Result* out_result, std::string* out_err) {
    Result r = paramsFromText(text, cfg);

    if (!(cfg.duration_s > 0.0) || !std::isfinite(cfg.duration_s)) {
        if (out_err) *out_err = "duration_s invalide";
        return false;
    }
    if (cfg.sample_rate < 8000 || cfg.sample_rate > 192000) {
        if (out_err) *out_err = "sample_rate invalide (8000..192000)";
        return false;
    }

    const double duration_s = cfg.duration_s;
    const int sample_rate = cfg.sample_rate;
    const int total_frames = (int)std::llround(duration_s * (double)sample_rate);
    if (total_frames <= 0) {
        if (out_err) *out_err = "durée audio trop courte";
        return false;
    }

    const double carrier_hz = clampd(cfg.carrier_hz, 40.0, 20000.0);
    const double mod_hz = r.cognitive_hz;
    const double mod_depth = clampd(cfg.audio_mod_depth, 0.0, 1.0);

    double binaural = cfg.binaural_hz;
    if (!(binaural > 0.0)) binaural = 0.0;
    binaural = clampd(binaural, 0.0, 48.0);

    const double peak = 0.2;
    const double w_car_l = 2.0 * kPi * carrier_hz;
    const double w_car_r = 2.0 * kPi * (carrier_hz + binaural);
    const double w_mod = 2.0 * kPi * mod_hz;

    std::vector<uint8_t> digest;
    hex_to_bytes(r.sha256_hex, &digest);
    const double phase_mod = digest.size() > 2 ? (2.0 * kPi * ((double)digest[2] / 255.0)) : 0.0;

    // Optional "organic" control curves via internal MLP.
    std::vector<double> ctrl0;
    std::vector<double> ctrl1;
    double ctrl_fps = 0.0;
    if (cfg.organic_nn) {
        ctrl_fps = (double)std::max(1, cfg.nn_control_fps);
        const int ctrl_n = (int)std::ceil(duration_s * ctrl_fps);
        if (ctrl_n > 1) {
            NeuroPulseModel nn;
            nn.buildFromConfig(cfg);

            // Ensure weights exist for forward.
            nn.allocateParams();
            const unsigned int seed = (cfg.nn_seed != 0u) ? cfg.nn_seed : seed_from_digest(digest);
            nn.initializeWeights("he", seed);

            ctrl0.assign((size_t)ctrl_n, 0.0);
            ctrl1.assign((size_t)ctrl_n, 0.0);

            float text8[8];
            fill_text_features_8(digest, text8);

            const double w1 = w_mod;
            const double w2 = w_mod * 0.37;
            const double w3 = w_mod * 1.73;

            const double smooth = clampd(cfg.nn_smooth, 0.0, 0.999);
            double y0_prev = 0.0;
            double y1_prev = 0.0;

            std::vector<float> in;
            in.resize(16);

            for (int i = 0; i < ctrl_n; ++i) {
                const double t = (double)i / ctrl_fps;
                const double t_norm = (duration_s > 0.0) ? clampd(t / duration_s, 0.0, 1.0) : 0.0;

                in[0] = (float)t_norm;
                in[1] = 1.0f;
                in[2] = (float)std::sin(w1 * t);
                in[3] = (float)std::cos(w1 * t);
                in[4] = (float)std::sin(w2 * t);
                in[5] = (float)std::cos(w2 * t);
                in[6] = (float)std::sin(w3 * t);
                in[7] = (float)std::cos(w3 * t);
                for (int k = 0; k < 8; ++k) in[8 + k] = text8[k];

                const auto& out = nn.forwardPassView(in, false);
                double y0 = out.size() > 0 ? (double)out[0] : 0.0;
                double y1 = out.size() > 1 ? (double)out[1] : 0.0;

                if (i == 0) {
                    y0_prev = y0;
                    y1_prev = y1;
                } else {
                    y0_prev = smooth * y0_prev + (1.0 - smooth) * y0;
                    y1_prev = smooth * y1_prev + (1.0 - smooth) * y1;
                }

                ctrl0[(size_t)i] = clampd(y0_prev, -1.0, 1.0);
                ctrl1[(size_t)i] = clampd(y1_prev, -1.0, 1.0);
            }

            r.nn_seed = seed;
            r.nn_control_fps = (int)ctrl_fps;
            r.organic_nn = true;
        } else {
            r.warnings.push_back("nn_control_fps trop bas; organic_nn ignoré");
        }
    }

    const double fade_s = 0.05;
    const int fade_n = (int)std::max(1.0, std::floor(fade_s * (double)sample_rate));

    std::vector<int16_t> pcm;
    pcm.resize((size_t)total_frames * 2u);

    for (int i = 0; i < total_frames; ++i) {
        const double t = (double)i / (double)sample_rate;

        const double organic0 = (ctrl_fps > 0.0) ? sample_series(ctrl0, t, ctrl_fps) : 0.0;
        const double organic1 = (ctrl_fps > 0.0) ? sample_series(ctrl1, t, ctrl_fps) : 0.0;
        const double strength = clampd(cfg.nn_strength, 0.0, 1.0);

        const double depth_eff = clampd(mod_depth * (1.0 + strength * organic0), 0.0, 1.0);
        const double phase_extra = strength * 0.6 * organic1; // radians
        const double env = (1.0 - depth_eff) + depth_eff * (0.5 * (1.0 + std::sin(w_mod * t + phase_mod + phase_extra)));

        double sL = std::sin(w_car_l * t) * env;
        double sR = std::sin(w_car_r * t) * env;

        double g = 1.0;
        if (i < fade_n) g = (double)i / (double)fade_n;
        const int rem = total_frames - 1 - i;
        if (rem < fade_n) g = std::min(g, (double)rem / (double)fade_n);

        sL *= (peak * g);
        sR *= (peak * g);

        const int16_t vL = (int16_t)std::llround(clampd(sL, -1.0, 1.0) * 32767.0);
        const int16_t vR = (int16_t)std::llround(clampd(sR, -1.0, 1.0) * 32767.0);

        pcm[(size_t)i * 2u + 0u] = vL;
        pcm[(size_t)i * 2u + 1u] = vR;
    }

    std::string wav_err;
    if (!write_wav_pcm16_stereo(cfg.out_wav, sample_rate, pcm, &wav_err)) {
        if (out_err) *out_err = wav_err;
        return false;
    }

    const int fps = std::max(1, cfg.light_fps);
    double light_hz = cfg.light_hz;
    if (cfg.safe_light) {
        light_hz = clampd(light_hz, 0.05, 2.0);
    }
    const double depth = clampd(cfg.light_depth, 0.0, 1.0);
    const double phase_light = digest.size() > 3 ? (2.0 * kPi * ((double)digest[3] / 255.0)) : 0.0;

    std::string light_err;
    if (ctrl_fps > 0.0 && !ctrl0.empty()) {
        const int total = (int)std::ceil(duration_s * (double)fps);
        std::vector<double> intensity;
        intensity.resize((size_t)std::max(0, total));
        const double w = 2.0 * kPi * light_hz;
        const double strength = clampd(cfg.nn_strength, 0.0, 1.0);

        for (int i = 0; i < total; ++i) {
            const double t = (double)i / (double)fps;
            const double organic0 = sample_series(ctrl0, t, ctrl_fps);
            const double organic1 = sample_series(ctrl1, t, ctrl_fps);
            const double depth_eff = clampd(depth * (1.0 + strength * 0.5 * organic0), 0.0, 1.0);

            const double s = 0.5 * (1.0 + std::sin(w * t + phase_light));
            double v = 0.5 + depth_eff * (s - 0.5);
            v += strength * 0.12 * organic1;
            intensity[(size_t)i] = clampd(v, 0.0, 1.0);
        }

        if (!write_light_csv_series(cfg.out_light_csv, fps, intensity, &light_err)) {
            if (out_err) *out_err = light_err;
            return false;
        }
    } else {
        if (!write_light_csv(cfg.out_light_csv, fps, duration_s, light_hz, depth, phase_light, &light_err)) {
            if (out_err) *out_err = light_err;
            return false;
        }
    }

    if (cfg.organic_nn) {
        r.warnings.push_back("organic_nn=true: modulation 'organique' activée (démonstration, non médicale)");
    }

    // Finalize meta
    r.sample_rate = sample_rate;
    r.duration_s = duration_s;
    r.carrier_hz = carrier_hz;
    r.binaural_hz = binaural;
    r.light_fps = fps;
    r.light_hz = light_hz;
    r.out_wav = cfg.out_wav;
    r.out_light_csv = cfg.out_light_csv;

    if (cfg.safe_light) {
        r.warnings.push_back("safe_light=true: lumière = enveloppe lisse (<=2 Hz), pas de stroboscope");
        r.warnings.push_back("Si antécédents d'épilepsie photosensible: éviter toute lumière pulsée");
    }

    if (out_result) *out_result = r;
    return true;
}
