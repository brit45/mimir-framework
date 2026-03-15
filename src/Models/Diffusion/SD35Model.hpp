#pragma once

#include "../Model.hpp"

// SD35Model
// Pour l'instant: squelette/placeholder destiné à exposer une architecture "sd3_5"
// via le registry, afin de brancher progressivement la compatibilité SD3.5.
//
// Une implémentation SD3.5 complète nécessitera:
// - un backbone type MMDiT/Transformer multi-modal
// - cross-attention fonctionnel
// - embeddings/positional encodings spécifiques
// - modulation AdaLN / gating, etc.

class SD35Model : public Model {
public:
    struct Config {
        // Squelette léger mais exécutable (pas un vrai SD3.5 complet).
        // false = construire un petit backbone attention+mlp.
        bool stub_only = false;

        // Deux flux packés dans l'input: [query_stream || kv_stream]
        // Chaque stream est une séquence de tokens, aplatie en 1D.
        int q_len = 32;
        int kv_len = 32;
        int d_model = 64;
        int num_heads = 4;
        int num_layers = 2;
        int mlp_hidden = 256;
        bool causal = false;
    };

    SD35Model();

    void buildFromConfig(const Config& cfg);
    const Config& getConfig() const { return cfg_; }

    static void buildInto(Model& model, const Config& cfg);

private:
    Config cfg_;
};
