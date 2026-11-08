#pragma once

#include "../../Model.hpp"
#include "../../viz/LatentSpaceProjection.hpp"

#include <functional>
#include <random>
#include <string>
#include <utility>
#include <vector>

// Decoder-only language model implemented entirely with native Mimir layers.
// Inputs: token ids in "__input__". Training uses configured windows; inference
// expands the sequence dimension to fit the complete prompt and output budget.
class CausalLMModel : public Model {
public:
    struct GenerationOptions {
        int max_new_tokens = 128;
        int top_k = 40;
        int seed = 1337;
        float temperature = 0.8f;
        float top_p = 0.95f;
        int stop_id = -1;
        bool use_memory = true;
        std::string system_prompt;
    };

    using TrainingCallback = std::function<void(int, const TrainStepResult&, const Optimizer&)>;
    using TrainingControl = std::function<bool(int)>;

    struct Config {
        int vocab_size = 4096;
        int seq_len = 128; // Training window; inference sequence length is runtime-sized.
        int d_model = 256;
        int num_layers = 6;
        int num_heads = 8;
        int num_kv_heads = 2;
        int mlp_hidden = 704;
        int causal_arch_version = 2;
        // Learned token-major local mixing, using current and previous tokens
        // only. Zero disables this branch for legacy/checkpoint compatibility.
        int local_conv_kernel = 3;
        int local_conv_dilation = 1;
        // Native two-input SiLU(gate)*value, mathematically the existing FFN.
        bool fused_swiglu = true;
        // Negative selects the default contextual bottleneck; zero disables it.
        int latent_dim = -1;
        int padding_idx = 0;
        float norm_eps = 1e-5f;
        float rope_theta = 10000.0f;
    };

    CausalLMModel();
    void buildFromConfig(const Config& cfg);
    static void buildInto(Model& model, const Config& cfg);
    const Config& getConfig() const { return cfg_; }

    // The model owns its tokenizer, prepared corpus, training sampler, generator,
    // and conversation history. Call prepareTrainingCorpus before allocating params.
    void prepareTrainingCorpus(const std::string& corpus, size_t vocab_capacity,
                               int bpe_merges, const std::string& tokenizer_path,
                               int seed = 1337, bool persist_tokenizer = true);
    void savePreparedTokenizer(const std::string& tokenizer_path);
    bool hasPreparedTrainingCorpus() const { return !training_target_positions_.empty(); }
    int trainCorpus(int steps, Optimizer& optimizer, float learning_rate,
                    int seed, const TrainingCallback& callback = {},
                    const TrainingControl& continue_training = {});
    std::optional<TrainStepResult> trainStep(const TrainStepRequest& request) override;
    mimir::viz::LatentSpaceRawSnapshot captureLatentSpaceSnapshot(
        size_t optimizer_step, size_t max_token_vectors = 48,
        size_t max_sequence_positions = 64, size_t max_vectors_per_matrix = 3) const override;

    std::vector<int> generateTokenIds(const std::string& prompt,
                                      const GenerationOptions& options);
    std::string generateText(const std::string& prompt,
                             const GenerationOptions& options);
    std::string chat(const std::string& user_message, const std::string& memory_path,
                     const GenerationOptions& options);
    void resetConversationMemory(const std::string& memory_path);
    size_t conversationMemorySize(const std::string& memory_path = {});

    bool InitVizTips() override;
    bool UpdateVizTips(const Layer& layer, VizFrame& frame) override;
    const std::vector<float>& forwardPassNamedView(
        const std::unordered_map<std::string, std::vector<float>>& float_inputs,
        const std::unordered_map<std::string, std::vector<int>>& int_inputs,
        bool training = true) override;

private:
    struct ValidationSequence {
        std::vector<int> tokens;
        size_t target_begin = 0;
    };

    struct ValidationResult {
        double cross_entropy = 0.0;
        double perplexity = 0.0;
        double next_token_accuracy_pct = 0.0;
        size_t active_tokens = 0;
        size_t items = 0;
    };

    struct ConversationTurn {
        std::string user;
        std::string assistant;
    };

    std::string decorateContext(const std::string& prompt) const;
    std::vector<int> tokenizeContent(const std::string& text) const;
    ValidationResult evaluateValidation(size_t max_items);
    std::vector<int> generateFromTokenContext(std::vector<int> context,
                                              const GenerationOptions& options);
    void loadConversationMemory(const std::string& memory_path);
    void saveConversationMemory() const;

    Config cfg_;
    bool training_corpus_prepared_ = false;
    std::vector<int> training_tokens_;
    std::vector<uint8_t> training_target_mask_;
    std::vector<size_t> training_target_positions_;
    std::vector<ValidationSequence> validation_sequences_;
    std::mt19937 training_rng_{1337};
    std::string conversation_memory_path_;
    std::vector<ConversationTurn> conversation_memory_;
};
