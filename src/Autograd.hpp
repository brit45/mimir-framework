#pragma once

#include <vector>
#include <unordered_map>
#include <string>
#include <cmath>

// Structure pour stocker les activations du forward pass
struct ComputationGraph {
    // Activations intermédiaires
    std::vector<float> token_embeddings;
    std::vector<float> pos_encodings;
    std::vector<float> combined_input;
    
    // Par layer transformer
    struct LayerActivations {
        std::vector<float> input;
        std::vector<float> normed1;
        std::vector<float> attn_out;
        std::vector<float> after_attn;
        std::vector<float> normed2;
        std::vector<float> ffn_out;
        std::vector<float> output;
    };
    std::vector<LayerActivations> layers;
    
    std::vector<float> final_output;
    
    // Métadonnées
    std::vector<int> input_tokens;
    int sequence_length;
    int latent_dim;
};

// Structure pour stocker les gradients
struct Gradients {
    // Gradients par paramètre (indexés par offset dans params)
    std::unordered_map<size_t, float> param_grads;
    
    // Ajouter un gradient
    void add(size_t param_idx, float grad) {
        param_grads[param_idx] += grad;
    }
    
    // Récupérer un gradient
    float get(size_t param_idx) const {
        auto it = param_grads.find(param_idx);
        return it != param_grads.end() ? it->second : 0.0f;
    }
    
    // Réinitialiser
    void zero() {
        param_grads.clear();
    }
    
    // Clipper les gradients
    void clip(float max_norm) {
        float total_norm = 0.0f;
        for (const auto& [idx, grad] : param_grads) {
            total_norm += grad * grad;
        }
        total_norm = std::sqrt(total_norm);
        
        if (total_norm > max_norm) {
            float scale = max_norm / (total_norm + 1e-6f);
            for (auto& [idx, grad] : param_grads) {
                grad *= scale;
            }
        }
    }
};

// Fonctions de backprop pour chaque opération
namespace Autograd {
    // Gradient de MSE: dL/dx = 2(x - target) / n
    inline std::vector<float> mse_backward(const std::vector<float>& pred, 
                                           const std::vector<float>& target) {
        std::vector<float> grad(pred.size());
        #pragma omp simd
        for (size_t i = 0; i < pred.size(); ++i) {
            grad[i] = 2.0f * (pred[i] - target[i]) / pred.size();
        }
        return grad;
    }
    
    // Gradient de LayerNorm
    inline std::vector<float> layernorm_backward(const std::vector<float>& grad_output,
                                                  const std::vector<float>& input,
                                                  const std::vector<float>& normalized) {
        size_t n = input.size();
        
        // Calculer mean et std du forward pass
        float mean = 0.0f;
        #pragma omp simd reduction(+:mean)
        for (size_t i = 0; i < n; ++i) mean += input[i];
        mean /= n;
        
        float var = 0.0f;
        #pragma omp simd reduction(+:var)
        for (size_t i = 0; i < n; ++i) {
            float diff = input[i] - mean;
            var += diff * diff;
        }
        var /= n;
        float std = std::sqrt(var + 1e-5f);
        
        // Gradient
        std::vector<float> grad_input(n);
        float grad_var = 0.0f;
        float grad_mean = 0.0f;
        
        #pragma omp simd reduction(+:grad_var)
        for (size_t i = 0; i < n; ++i) {
            grad_var += grad_output[i] * (input[i] - mean);
        }
        grad_var *= -0.5f / (std * std * std);
        
        #pragma omp simd reduction(+:grad_mean)
        for (size_t i = 0; i < n; ++i) {
            grad_mean += grad_output[i] * (-1.0f / std);
            grad_mean += grad_var * (-2.0f * (input[i] - mean) / n);
        }
        
        #pragma omp simd
        for (size_t i = 0; i < n; ++i) {
            grad_input[i] = grad_output[i] / std;
            grad_input[i] += grad_var * (2.0f * (input[i] - mean) / n);
            grad_input[i] += grad_mean / n;
        }
        
        return grad_input;
    }
    
    // Gradient de GELU
    inline float gelu_backward(float x, float grad_output) {
        const float sqrt_2_pi = 0.7978845608f;
        const float coeff = 0.044715f;
        
        float x_cubed = x * x * x;
        float tanh_arg = sqrt_2_pi * (x + coeff * x_cubed);
        float tanh_val = std::tanh(tanh_arg);
        
        float sech_sq = 1.0f - tanh_val * tanh_val;
        float dtanh = sqrt_2_pi * (1.0f + 3.0f * coeff * x * x) * sech_sq;
        
        float dgelu = 0.5f * (1.0f + tanh_val) + 0.5f * x * dtanh;
        
        return grad_output * dgelu;
    }
    
    // Gradient de Residual Connection: grad passe tel quel
    inline std::vector<float> residual_backward(const std::vector<float>& grad_output) {
        return grad_output; // Le gradient se propage directement
    }
}
