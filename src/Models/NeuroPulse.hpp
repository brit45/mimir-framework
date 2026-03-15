#pragma once

#include "../Model.hpp"

#include <cstdint>
#include <string>
#include <vector>

// NeuroPulseModel:
// - "Utility model" that deterministically converts a text to:
//   - an audio WAV (carrier + low-frequency amplitude modulation)
//   - a light envelope CSV (intensity over time)
// - Not a medical device; do not interpret as therapy.

class NeuroPulseModel : public Model {
public:
    enum class Band {
        Delta,
        Theta,
        Alpha,
        Beta,
        Gamma,
    };

    struct Config {
        double duration_s = 10.0;
        int sample_rate = 48000;

        double carrier_hz = 220.0;
        double audio_mod_depth = 0.8; // 0..1
        double binaural_hz = 0.0;     // 0 disables

        // Cognitive modulation frequency selection.
        std::string cognitive_band = "auto"; // auto|delta|theta|alpha|beta|gamma
        double cognitive_hz = 0.0;           // 0 = auto

        // Light envelope output (values 0..1). In safe mode, smooth and clamped <= 2 Hz.
        bool safe_light = true;
        int light_fps = 60;
        double light_hz = 0.2;
        double light_depth = 0.6;

        // Organic modulation: small internal MLP that outputs a smooth control signal
        // derived from (time + text hash features). Used to gently modulate AM depth and
        // light intensity in a deterministic way.
        bool organic_nn = true;
        int nn_control_fps = 200;     // control points per second
        int nn_hidden_dim = 32;
        int nn_hidden_layers = 2;
        double nn_dropout = 0.0;      // 0..0.95 (used only in training, kept for symmetry)
        double nn_strength = 0.35;    // 0..1 (applied to modulation)
        double nn_smooth = 0.85;      // 0..0.999 (IIR smoothing in control domain)
        unsigned int nn_seed = 0;     // 0 => derived from sha256(text)

        std::string out_wav = "neuropulse.wav";
        std::string out_light_csv = "neuropulse_light.csv";
    };

    struct Result {
        std::string sha256_hex;
        Band band = Band::Alpha;
        double cognitive_hz = 10.0;
        double carrier_hz = 220.0;
        double binaural_hz = 0.0;
        int sample_rate = 48000;
        double duration_s = 10.0;
        int light_fps = 60;
        double light_hz = 0.2;

        std::string out_wav;
        std::string out_light_csv;

        bool organic_nn = false;
        unsigned int nn_seed = 0;
        int nn_control_fps = 0;

        std::vector<std::string> warnings;
    };

    NeuroPulseModel() = default;

    void buildFromConfig(const Config& cfg);
    const Config& getConfig() const { return cfg_; }

    static Result paramsFromText(const std::string& text, const Config& cfg);
    static bool renderToFiles(const std::string& text, const Config& cfg, Result* out_result, std::string* out_err);

private:
    Config cfg_;
};
