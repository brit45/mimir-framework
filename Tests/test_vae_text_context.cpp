#include "test_utils.hpp"
#include "Models/NLP/VAETextModel.hpp"
#include "Models/Registry/ModelArchitectures.hpp"
#include "Serialization/Serialization.hpp"
#include "runtimes/losses/LossDispatch.hpp"
#include <filesystem>
#include <numeric>

int main() {
    auto cfg = ModelArchitectures::defaultConfig("vae_text");
    cfg.update({{"vocab_size", 16}, {"seq_len", 4}, {"d_model", 8},
        {"num_layers", 1}, {"num_heads", 2}, {"mlp_hidden", 16},
        {"latent_tokens", 2}, {"proj_dim", 8}, {"enable_context_heads", false},
        {"enable_conditional_encoder", false}, {"stochastic_latent", false},
        {"kl_beta", 0}, {"text_mask_ratio", 0.99}, {"content_weight", 0.2}});
    auto model = ModelArchitectures::create("vae_text", cfg);
    model->allocateParams(); model->initializeWeights("xavier", 42);
    TASSERT_TRUE(model->getLayerByName("vae_text/enc/position"));
    TASSERT_TRUE(!model->getLayerByName("vae_text/enc/block1/self_attn")->causal);
    const std::vector<int> text{2,3,4,5}, reversed{5,4,3,2};
    model->forwardPassNamedView({}, {{"text_ids", text}}, false);
    const auto first_mu = model->getTensor("vae_text/mu");
    model->forwardPassNamedView({}, {{"text_ids", reversed}}, false);
    const auto second_mu = model->getTensor("vae_text/mu");
    float difference = 0;
    for (size_t i=0;i<first_mu.size();++i) difference += std::abs(first_mu[i]-second_mu[i]);
    TASSERT_TRUE(difference > 1e-5f); // Same tokens, changed roles/order, bidirectional encoder.

    Optimizer optimizer;
    model->setVizTapsEnabled(true); model->setVizTapsLimits(64, 16);
    Model::TrainStepRequest request;
    request.optimizer = &optimizer; request.learning_rate = 0;
    request.int_inputs["text_ids"] = &text;
    request.mode = Model::TrainStepMode::Accumulate;
    auto stats = model->trainStep(request);
    TASSERT_TRUE(stats && std::isfinite(stats->loss));
    TASSERT_TRUE(stats->metrics.at("masked_tokens") > 0 && stats->metrics.at("masked_tokens") < 4);
    TASSERT_TRUE(stats->metrics.at("content_ce") > 0);
    const auto& packed = model->getTensor("x");
    const auto expected_ce = RuntimeLossGrad::text_cross_entropy(packed.data(), packed.size(), text, 4, 16, 0);
    TASSERT_NEAR(stats->metrics.at("mse"), expected_ce.loss, 1e-6f); // Targets remain clean.
    for (const auto& name : {"vae_text/enc/position", "vae_text/mu", "vae_text/content/head"}) {
        const auto& gradient = model->getLayerByName(name)->grad_weights;
        float norm = 0;
        for (float value : gradient) { TASSERT_TRUE(std::isfinite(value)); norm += std::abs(value); }
        TASSERT_TRUE(norm > 0);
    }
    bool reconstruction_tip = false, latent_tip = false;
    for (const auto& frame : model->consumeVizTaps()) {
        if (frame.label == "Texte / entrée et reconstruction") {
            reconstruction_tip = true;
            TASSERT_TRUE(frame.tensor_info.find("Texte reçu:") != std::string::npos);
            TASSERT_TRUE(frame.tensor_info.find("Reconstruction:") != std::string::npos);
            TASSERT_TRUE(frame.tensor_values.size() == frame.tensor_indices.size());
        }
        if (frame.label.find("moyenne utilisée par Lumen") != std::string::npos) latent_tip = true;
    }
    TASSERT_TRUE(reconstruction_tip && latent_tip);

    // The content CE has its own correct gradient, independently of reconstruction.
    const auto* head = model->getLayerByName("vae_text/content/head");
    const size_t bias_index = head->getWeightsSize()-16+2;
    const float analytic = head->grad_weights[bias_index];
    auto* mutable_head = model->getLayerByName("vae_text/content/head");
    const float initial = mutable_head->getWeights()[bias_index], epsilon = 1e-3f;
    auto loss = [&](float weight) {
        mutable_head->getWeights()[bias_index] = weight;
        model->zeroGradients();
        return model->trainStep(request)->loss;
    };
    const float numeric = (loss(initial+epsilon)-loss(initial-epsilon))/(2*epsilon);
    mutable_head->getWeights()[bias_index] = initial;
    TASSERT_NEAR(analytic, numeric, 1e-3f);

    // A bounded learning probe must improve reconstruction of both word orders.
    model->setVizTapsEnabled(false);
    model->modelConfig["text_mask_ratio"] = 0;
    request.mode = Model::TrainStepMode::Optimize;
    request.learning_rate = 0.003f;
    const float initial_loss = model->trainStep(request)->loss;
    for (int step=0; step<160; ++step) {
        request.int_inputs["text_ids"] = step % 2 ? &text : &reversed;
        const auto learned = model->trainStep(request);
        TASSERT_TRUE(learned && std::isfinite(learned->loss));
    }
    request.mode = Model::TrainStepMode::Accumulate;
    request.int_inputs["text_ids"] = &text;
    model->zeroGradients();
    const float final_loss = model->trainStep(request)->loss;
    TASSERT_TRUE(final_loss < initial_loss * 0.7f);
    request.int_inputs["text_ids"] = &reversed;
    model->zeroGradients();
    const float reversed_loss = model->trainStep(request)->loss;
    TASSERT_TRUE(reversed_loss < initial_loss * 0.7f);


    // Full-model and standalone decoder must use identical attention contracts.
    const auto checkpoint = std::filesystem::temp_directory_path() / "mimir_vae_text_context_test";
    std::filesystem::remove_all(checkpoint);
    Mimir::Serialization::SaveOptions save;
    save.format = Mimir::Serialization::CheckpointFormat::RawFolder;
    save.save_encoder = save.save_tokenizer = false;
    std::string error;
    TASSERT_TRUE(Mimir::Serialization::save_checkpoint(*model, checkpoint.string(), save, &error));
    const auto full = model->forwardPassNamedView({}, {{"text_ids", text}}, false);
    const auto latent = model->getTensor("vae_text/mu");
    auto decoder = ModelArchitectures::create("vae_text_decode", cfg);
    decoder->allocateParams();
    Mimir::Serialization::LoadOptions load;
    load.format = save.format; load.load_encoder = load.load_tokenizer = false;
    load.apply_model_config = false; load.strict_mode = false;
    TASSERT_TRUE(Mimir::Serialization::load_checkpoint(*decoder, checkpoint.string(), load, &error));
    const auto decoded = decoder->forwardPassNamedView({{"__input__", latent}}, {}, false);
    TASSERT_TRUE(decoded.size() == 64);
    for (size_t i=0; i<decoded.size(); ++i) TASSERT_NEAR(decoded[i], full[i], 1e-5f);
    std::filesystem::remove_all(checkpoint);
    std::cout << "learning_loss=" << initial_loss << " -> " << final_loss << " reversed=" << reversed_loss << "\n";

    // Serialized old topology remains rebuildable without new parameters.
    auto legacy = cfg;
    legacy["type"] = "vae_text";
    legacy.erase("text_context_version"); legacy.erase("contextual_encoding");
    auto old = ModelArchitectures::create("vae_text", legacy);
    TASSERT_TRUE(!old->getLayerByName("vae_text/enc/position"));
    TASSERT_TRUE(!old->getLayerByName("vae_text/content/head"));
    std::cout << "VAE_TEXT_CONTEXT_OK\n";
    return 0;
}
