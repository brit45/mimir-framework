#include "ModelArchitectures.hpp"

#include "Models/MLP/BasicMLPModel.hpp"
#include "Models/Diffusion/PonyXLDDPMModel.hpp"
#include "Models/NLP/TransformerModel.hpp"
#include "Models/Vision/ViTModel.hpp"
#include "Models/Vision/VAEModel.hpp"
#include "Models/Vision/CNN/ResNetModel.hpp"
#include "Models/Vision/UNetModel.hpp"
#include "Models/Vision/CNN/MobileNetModel.hpp"
#include "Models/Vision/CNN/VGG16Model.hpp"
#include "Models/Vision/CNN/VGG19Model.hpp"
#include "Models/Diffusion/DiffusionModel.hpp"

#include <algorithm>
#include <memory>
#include <stdexcept>

namespace ModelArchitectures {

Registry& Registry::instance() {
    static Registry inst;
    return inst;
}

void Registry::registerArchitecture(Entry entry) {
    if (entry.name.empty()) {
        throw std::runtime_error("ModelArchitectures::Registry: entry.name is empty");
    }
    if (!entry.create) {
        throw std::runtime_error("ModelArchitectures::Registry: entry.create is null for: " + entry.name);
    }
    ensureBuiltinsRegistered();
    entries_[entry.name] = std::move(entry);
}

const Entry* Registry::find(const std::string& name) const {
    ensureBuiltinsRegistered();
    auto it = entries_.find(name);
    if (it == entries_.end()) return nullptr;
    return &it->second;
}

std::vector<std::string> Registry::list() const {
    ensureBuiltinsRegistered();
    std::vector<std::string> out;
    out.reserve(entries_.size());
    for (const auto& kv : entries_) out.push_back(kv.first);
    std::sort(out.begin(), out.end());
    return out;
}

json Registry::defaultConfig(const std::string& name) const {
    ensureBuiltinsRegistered();
    auto it = entries_.find(name);
    if (it == entries_.end()) {
        throw std::runtime_error("ModelArchitectures::defaultConfig: unknown architecture: " + name);
    }
    return it->second.default_config;
}

namespace {

template <typename T>
static T jget(const json& j, const char* key, T def) {
    if (!j.is_object()) return def;
    auto it = j.find(key);
    if (it == j.end() || it->is_null()) return def;
    try {
        return it->get<T>();
    } catch (...) {
        return def;
    }
}

static void mergeInto(json& base, const json& overrides) {
    if (!base.is_object() || !overrides.is_object()) return;
    for (auto it = overrides.begin(); it != overrides.end(); ++it) {
        const std::string& key = it.key();
        const json& val = it.value();
        if (base.contains(key) && base[key].is_object() && val.is_object()) {
            mergeInto(base[key], val);
        } else {
            base[key] = val;
        }
    }
}

static BasicMLPModel::Config basicMlpCfgFromJson(const json& cfg) {
    BasicMLPModel::Config out;
    out.input_dim = jget<int>(cfg, "input_dim", out.input_dim);
    out.hidden_dim = jget<int>(cfg, "hidden_dim", out.hidden_dim);
    out.output_dim = jget<int>(cfg, "output_dim", out.output_dim);
    out.hidden_layers = jget<int>(cfg, "hidden_layers", out.hidden_layers);
    out.dropout = jget<float>(cfg, "dropout", out.dropout);
    return out;
}

static json basicMlpDefaultConfigJson() {
    BasicMLPModel::Config d;
    return json{
        {"input_dim", d.input_dim},
        {"hidden_dim", d.hidden_dim},
        {"output_dim", d.output_dim},
        {"hidden_layers", d.hidden_layers},
        {"dropout", d.dropout},
    };
}

static PonyXLDDPMModel::Config ponyCfgFromJson(const json& cfg) {

    PonyXLDDPMModel::Config out;
    
    out.d_model = jget<int>(cfg, "d_model", out.d_model);
    out.seq_len = jget<int>(cfg, "seq_len", out.seq_len);
    out.max_vocab = jget<int>(cfg, "max_vocab", out.max_vocab);

    out.image_w = jget<int>(cfg, "image_w", out.image_w);
    out.image_h = jget<int>(cfg, "image_h", out.image_h);
    out.image_c = jget<int>(cfg, "image_c", out.image_c);

    out.hidden_dim = jget<int>(cfg, "hidden_dim", out.hidden_dim);
    out.latent_dim = jget<int>(cfg, "latent_dim", out.latent_dim);

    out.blur_levels = jget<int>(cfg, "blur_levels", out.blur_levels);
    out.cfg_dropout_prob = jget<float>(cfg, "cfg_dropout_prob", out.cfg_dropout_prob);
    out.cond_image_dropout_prob = jget<float>(cfg, "cond_image_dropout_prob", out.cond_image_dropout_prob);
    out.cond_image_dropout_lr_scale = jget<float>(cfg, "cond_image_dropout_lr_scale", out.cond_image_dropout_lr_scale);
    out.cond_image_dropout_noise_std = jget<float>(cfg, "cond_image_dropout_noise_std", out.cond_image_dropout_noise_std);

    out.dropout = jget<float>(cfg, "dropout", out.dropout);
    return out;
}

static json ponyDefaultConfigJson() {
    PonyXLDDPMModel::Config d;
    return json{
        {"d_model", d.d_model},
        {"seq_len", d.seq_len},
        {"max_vocab", d.max_vocab},
        {"image_w", d.image_w},
        {"image_h", d.image_h},
        {"image_c", d.image_c},
        {"hidden_dim", d.hidden_dim},
        {"latent_dim", d.latent_dim},
        {"blur_levels", d.blur_levels},
        {"cfg_dropout_prob", d.cfg_dropout_prob},
        {"cond_image_dropout_prob", d.cond_image_dropout_prob},
        {"cond_image_dropout_lr_scale", d.cond_image_dropout_lr_scale},
        {"cond_image_dropout_noise_std", d.cond_image_dropout_noise_std},
        {"dropout", d.dropout},
    };
}

static TransformerModel::Config transformerCfgFromJson(const json& cfg) {
    TransformerModel::Config out;
    out.seq_len = jget<int>(cfg, "seq_len", out.seq_len);
    out.d_model = jget<int>(cfg, "d_model", out.d_model);
    out.vocab_size = jget<int>(cfg, "vocab_size", out.vocab_size);
    out.padding_idx = jget<int>(cfg, "padding_idx", out.padding_idx);
    out.num_layers = jget<int>(cfg, "num_layers", out.num_layers);
    out.num_heads = jget<int>(cfg, "num_heads", out.num_heads);
    out.mlp_hidden = jget<int>(cfg, "mlp_hidden", out.mlp_hidden);
    out.output_dim = jget<int>(cfg, "output_dim", out.output_dim);
    out.causal = jget<bool>(cfg, "causal", out.causal);
    return out;
}

static json transformerDefaultConfigJson() {
    TransformerModel::Config d;
    return json{{"seq_len", d.seq_len}, {"d_model", d.d_model}, {"vocab_size", d.vocab_size}, {"padding_idx", d.padding_idx}, {"num_layers", d.num_layers}, {"num_heads", d.num_heads}, {"mlp_hidden", d.mlp_hidden}, {"output_dim", d.output_dim}, {"causal", d.causal}};
}

static ViTModel::Config vitCfgFromJson(const json& cfg) {
    ViTModel::Config out;
    out.num_tokens = jget<int>(cfg, "num_tokens", out.num_tokens);
    out.d_model = jget<int>(cfg, "d_model", out.d_model);
    out.num_layers = jget<int>(cfg, "num_layers", out.num_layers);
    out.num_heads = jget<int>(cfg, "num_heads", out.num_heads);
    out.mlp_hidden = jget<int>(cfg, "mlp_hidden", out.mlp_hidden);
    out.output_dim = jget<int>(cfg, "output_dim", out.output_dim);
    out.causal = jget<bool>(cfg, "causal", out.causal);
    return out;
}

static json vitDefaultConfigJson() {
    ViTModel::Config d;
    return json{{"num_tokens", d.num_tokens}, {"d_model", d.d_model}, {"num_layers", d.num_layers}, {"num_heads", d.num_heads}, {"mlp_hidden", d.mlp_hidden}, {"output_dim", d.output_dim}, {"causal", d.causal}};
}

static VAEModel::Config vaeCfgFromJson(const json& cfg) {
    VAEModel::Config out;
    out.image_w = jget<int>(cfg, "image_w", out.image_w);
    out.image_h = jget<int>(cfg, "image_h", out.image_h);
    out.image_c = jget<int>(cfg, "image_c", out.image_c);
    out.latent_dim = jget<int>(cfg, "latent_dim", out.latent_dim);
    out.hidden_dim = jget<int>(cfg, "hidden_dim", out.hidden_dim);
    return out;
}

static json vaeDefaultConfigJson() {
    VAEModel::Config d;
    return json{{"image_w", d.image_w}, {"image_h", d.image_h}, {"image_c", d.image_c}, {"latent_dim", d.latent_dim}, {"hidden_dim", d.hidden_dim}};
}

static ResNetModel::Config resnetCfgFromJson(const json& cfg) {
    ResNetModel::Config out;
    out.image_w = jget<int>(cfg, "image_w", out.image_w);
    out.image_h = jget<int>(cfg, "image_h", out.image_h);
    out.image_c = jget<int>(cfg, "image_c", out.image_c);
    out.base_channels = jget<int>(cfg, "base_channels", out.base_channels);
    out.num_classes = jget<int>(cfg, "num_classes", out.num_classes);
    out.blocks1 = jget<int>(cfg, "blocks1", out.blocks1);
    out.blocks2 = jget<int>(cfg, "blocks2", out.blocks2);
    out.blocks3 = jget<int>(cfg, "blocks3", out.blocks3);
    out.blocks4 = jget<int>(cfg, "blocks4", out.blocks4);
    return out;
}

static json resnetDefaultConfigJson() {
    ResNetModel::Config d;
    return json{{"image_w", d.image_w}, {"image_h", d.image_h}, {"image_c", d.image_c}, {"base_channels", d.base_channels}, {"num_classes", d.num_classes}, {"blocks1", d.blocks1}, {"blocks2", d.blocks2}, {"blocks3", d.blocks3}, {"blocks4", d.blocks4}};
}

static UNetModel::Config unetCfgFromJson(const json& cfg) {
    UNetModel::Config out;
    out.image_w = jget<int>(cfg, "image_w", out.image_w);
    out.image_h = jget<int>(cfg, "image_h", out.image_h);
    out.image_c = jget<int>(cfg, "image_c", out.image_c);
    out.base_channels = jget<int>(cfg, "base_channels", out.base_channels);
    out.depth = jget<int>(cfg, "depth", out.depth);
    return out;
}

static json unetDefaultConfigJson() {
    UNetModel::Config d;
    return json{{"image_w", d.image_w}, {"image_h", d.image_h}, {"image_c", d.image_c}, {"base_channels", d.base_channels}, {"depth", d.depth}};
}

static MobileNetModel::Config mobilenetCfgFromJson(const json& cfg) {
    MobileNetModel::Config out;
    out.image_w = jget<int>(cfg, "image_w", out.image_w);
    out.image_h = jget<int>(cfg, "image_h", out.image_h);
    out.image_c = jget<int>(cfg, "image_c", out.image_c);
    out.base_channels = jget<int>(cfg, "base_channels", out.base_channels);
    out.num_classes = jget<int>(cfg, "num_classes", out.num_classes);
    return out;
}

static json mobilenetDefaultConfigJson() {
    MobileNetModel::Config d;
    return json{{"image_w", d.image_w}, {"image_h", d.image_h}, {"image_c", d.image_c}, {"base_channels", d.base_channels}, {"num_classes", d.num_classes}};
}

static VGG16Model::Config vgg16CfgFromJson(const json& cfg) {
    VGG16Model::Config out;
    out.image_w = jget<int>(cfg, "image_w", out.image_w);
    out.image_h = jget<int>(cfg, "image_h", out.image_h);
    out.image_c = jget<int>(cfg, "image_c", out.image_c);
    out.base_channels = jget<int>(cfg, "base_channels", out.base_channels);
    out.num_classes = jget<int>(cfg, "num_classes", out.num_classes);
    out.fc_hidden = jget<int>(cfg, "fc_hidden", out.fc_hidden);
    return out;
}

static json vgg16DefaultConfigJson() {
    VGG16Model::Config d;
    return json{{"image_w", d.image_w}, {"image_h", d.image_h}, {"image_c", d.image_c}, {"base_channels", d.base_channels}, {"num_classes", d.num_classes}, {"fc_hidden", d.fc_hidden}};
}

static VGG19Model::Config vgg19CfgFromJson(const json& cfg) {
    VGG19Model::Config out;
    out.image_w = jget<int>(cfg, "image_w", out.image_w);
    out.image_h = jget<int>(cfg, "image_h", out.image_h);
    out.image_c = jget<int>(cfg, "image_c", out.image_c);
    out.base_channels = jget<int>(cfg, "base_channels", out.base_channels);
    out.num_classes = jget<int>(cfg, "num_classes", out.num_classes);
    out.fc_hidden = jget<int>(cfg, "fc_hidden", out.fc_hidden);
    return out;
}

static json vgg19DefaultConfigJson() {
    VGG19Model::Config d;
    return json{{"image_w", d.image_w}, {"image_h", d.image_h}, {"image_c", d.image_c}, {"base_channels", d.base_channels}, {"num_classes", d.num_classes}, {"fc_hidden", d.fc_hidden}};
}

static DiffusionModel::Config diffusionCfgFromJson(const json& cfg) {
    DiffusionModel::Config out;
    out.image_w = jget<int>(cfg, "image_w", out.image_w);
    out.image_h = jget<int>(cfg, "image_h", out.image_h);
    out.image_c = jget<int>(cfg, "image_c", out.image_c);
    out.time_dim = jget<int>(cfg, "time_dim", out.time_dim);
    out.hidden_dim = jget<int>(cfg, "hidden_dim", out.hidden_dim);
    out.dropout = jget<float>(cfg, "dropout", out.dropout);
    return out;
}

static json diffusionDefaultConfigJson() {
    DiffusionModel::Config d;
    return json{{"image_w", d.image_w}, {"image_h", d.image_h}, {"image_c", d.image_c}, {"time_dim", d.time_dim}, {"hidden_dim", d.hidden_dim}, {"dropout", d.dropout}};
}

} // namespace

std::shared_ptr<Model> Registry::create(const std::string& name, const json& config) const {
    ensureBuiltinsRegistered();
    auto it = entries_.find(name);
    if (it == entries_.end()) {
        throw std::runtime_error("ModelArchitectures::create: unknown architecture: " + name);
    }

    json cfg = it->second.default_config;
    if (config.is_object() && !config.empty()) {
        mergeInto(cfg, config);
    }

    // Standardize metadata for serialization/rebuild.
    if (!cfg.is_object()) cfg = json::object();
    cfg["type"] = name;

    auto model = it->second.create(cfg);
    if (!model) {
        throw std::runtime_error("ModelArchitectures::create: factory returned null for: " + name);
    }
    model->modelConfig = cfg;
    return model;
}

void Registry::ensureBuiltinsRegistered() const {
    std::call_once(builtins_once_, [&]() {
        entries_.emplace(
            "basic_mlp",
            Entry{
                "basic_mlp",
                "Basic MLP (régression: input->MLP->output)",
                basicMlpDefaultConfigJson(),
                [](const json& cfg) -> std::shared_ptr<Model> {
                    auto m = std::make_shared<BasicMLPModel>();
                    m->buildFromConfig(basicMlpCfgFromJson(cfg));
                    return m;
                },
            }
        );

        entries_.emplace(
            "ponyxl_ddpm",
            Entry{
                "ponyxl_ddpm",
                "PonyXL DDPM (texte + image bruitée + timestep -> prédiction de bruit)",
                ponyDefaultConfigJson(),
                [](const json& cfg) -> std::shared_ptr<Model> {
                    auto m = std::make_shared<PonyXLDDPMModel>();
                    m->buildFromConfig(ponyCfgFromJson(cfg));
                    return m;
                },
            });

        entries_.emplace(
            "transformer",
            Entry{
                "transformer",
                "Transformer encoder (float-only: input=seq_len*d_model)",
                transformerDefaultConfigJson(),
                [](const json& cfg) -> std::shared_ptr<Model> {
                    auto m = std::make_shared<TransformerModel>();
                    m->buildFromConfig(transformerCfgFromJson(cfg));
                    return m;
                },
            }
        );

        entries_.emplace(
            "vit",
            Entry{
                "vit",
                "ViT (float-only: input=patch embeddings num_tokens*d_model)",
                vitDefaultConfigJson(),
                [](const json& cfg) -> std::shared_ptr<Model> {
                    auto m = std::make_shared<ViTModel>();
                    m->buildFromConfig(vitCfgFromJson(cfg));
                    return m;
                },
            }
        );

        entries_.emplace(
            "vae",
            Entry{
                "vae",
                "VAE-style autoencoder (output packs recon||mu||logvar)",
                vaeDefaultConfigJson(),
                [](const json& cfg) -> std::shared_ptr<Model> {
                    auto m = std::make_shared<VAEModel>();
                    m->buildFromConfig(vaeCfgFromJson(cfg));
                    return m;
                },
            }
        );

        entries_.emplace(
            "resnet",
            Entry{
                "resnet",
                "ResNet (simplified ResNet18-like)",
                resnetDefaultConfigJson(),
                [](const json& cfg) -> std::shared_ptr<Model> {
                    auto m = std::make_shared<ResNetModel>();
                    m->buildFromConfig(resnetCfgFromJson(cfg));
                    return m;
                },
            }
        );

        entries_.emplace(
            "unet",
            Entry{
                "unet",
                "UNet (simplified encoder-decoder with skip concatenations)",
                unetDefaultConfigJson(),
                [](const json& cfg) -> std::shared_ptr<Model> {
                    auto m = std::make_shared<UNetModel>();
                    m->buildFromConfig(unetCfgFromJson(cfg));
                    return m;
                },
            }
        );

        entries_.emplace(
            "mobilenet",
            Entry{
                "mobilenet",
                "MobileNetV1-style (DepthwiseConv2d + pointwise Conv2d)",
                mobilenetDefaultConfigJson(),
                [](const json& cfg) -> std::shared_ptr<Model> {
                    auto m = std::make_shared<MobileNetModel>();
                    m->buildFromConfig(mobilenetCfgFromJson(cfg));
                    return m;
                },
            }
        );

        entries_.emplace(
            "vgg16",
            Entry{
                "vgg16",
                "VGG16 (simplified, downsample via stride-2 conv)",
                vgg16DefaultConfigJson(),
                [](const json& cfg) -> std::shared_ptr<Model> {
                    auto m = std::make_shared<VGG16Model>();
                    m->buildFromConfig(vgg16CfgFromJson(cfg));
                    return m;
                },
            }
        );

        entries_.emplace(
            "vgg19",
            Entry{
                "vgg19",
                "VGG19 (simplified, downsample via stride-2 conv)",
                vgg19DefaultConfigJson(),
                [](const json& cfg) -> std::shared_ptr<Model> {
                    auto m = std::make_shared<VGG19Model>();
                    m->buildFromConfig(vgg19CfgFromJson(cfg));
                    return m;
                },
            }
        );

        entries_.emplace(
            "diffusion",
            Entry{
                "diffusion",
                "Diffusion epsilon predictor (baseline MLP: input=t_embed||x_t)",
                diffusionDefaultConfigJson(),
                [](const json& cfg) -> std::shared_ptr<Model> {
                    auto m = std::make_shared<DiffusionModel>();
                    m->buildFromConfig(diffusionCfgFromJson(cfg));
                    return m;
                },
            }
        );
    });
}

} // namespace ModelArchitectures
