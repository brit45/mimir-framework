#include "LuaScripting.hpp"
#include "Models/ModelArchitectures.hpp"
#include "AdvancedRAMManager.hpp"
#include "MemoryGuard.hpp"
#include "DynamicTensorAllocator.hpp"
#include "AsyncMonitor.hpp"
#include <fstream>
#include <sstream>
#include <iostream>

// ============================================================================
// Constructeur / Destructeur
// ============================================================================

LuaScripting::LuaScripting() {
    L = luaL_newstate();
    luaL_openlibs(L);  // Charger les bibliothèques standard Lua
    registerAPI();
}

LuaScripting::~LuaScripting() {
    if (L) {
        lua_close(L);
    }
}

// ============================================================================
// Chargement et exécution de scripts
// ============================================================================

bool LuaScripting::loadScript(const std::string& filepath) {
    if (luaL_dofile(L, filepath.c_str()) != LUA_OK) {
        std::cerr << "Erreur Lua: " << lua_tostring(L, -1) << std::endl;
        lua_pop(L, 1);
        return false;
    }
    return true;
}

bool LuaScripting::executeScript(const std::string& code) {
    if (luaL_dostring(L, code.c_str()) != LUA_OK) {
        std::cerr << "Erreur Lua: " << lua_tostring(L, -1) << std::endl;
        lua_pop(L, 1);
        return false;
    }
    return true;
}

bool LuaScripting::callFunction(const std::string& function_name) {
    lua_getglobal(L, function_name.c_str());
    if (!lua_isfunction(L, -1)) {
        std::cerr << "Fonction Lua introuvable: " << function_name << std::endl;
        lua_pop(L, 1);
        return false;
    }
    
    if (lua_pcall(L, 0, 0, 0) != LUA_OK) {
        std::cerr << "Erreur lors de l'appel de " << function_name << ": " 
                  << lua_tostring(L, -1) << std::endl;
        lua_pop(L, 1);
        return false;
    }
    
    return true;
}

// ============================================================================
// Getters / Setters
// ============================================================================

std::string LuaScripting::getString(const std::string& var_name) {
    lua_getglobal(L, var_name.c_str());
    std::string result = lua_isstring(L, -1) ? lua_tostring(L, -1) : "";
    lua_pop(L, 1);
    return result;
}

double LuaScripting::getNumber(const std::string& var_name) {
    lua_getglobal(L, var_name.c_str());
    double result = lua_isnumber(L, -1) ? lua_tonumber(L, -1) : 0.0;
    lua_pop(L, 1);
    return result;
}

bool LuaScripting::getBoolean(const std::string& var_name) {
    lua_getglobal(L, var_name.c_str());
    bool result = lua_isboolean(L, -1) ? lua_toboolean(L, -1) : false;
    lua_pop(L, 1);
    return result;
}

void LuaScripting::setString(const std::string& var_name, const std::string& value) {
    lua_pushstring(L, value.c_str());
    lua_setglobal(L, var_name.c_str());
}

void LuaScripting::setNumber(const std::string& var_name, double value) {
    lua_pushnumber(L, value);
    lua_setglobal(L, var_name.c_str());
}

void LuaScripting::setBoolean(const std::string& var_name, bool value) {
    lua_pushboolean(L, value);
    lua_setglobal(L, var_name.c_str());
}

// ============================================================================
// Enregistrement de l'API
// ============================================================================

void LuaScripting::registerAPI() {
    // ========== Table "model" ==========
    lua_newtable(L);
    
    // Gestion basique
    lua_pushcfunction(L, lua_createModel);
    lua_setfield(L, -2, "create");
    
    lua_pushcfunction(L, lua_buildModel);
    lua_setfield(L, -2, "build");
    
    lua_pushcfunction(L, lua_trainModel);
    lua_setfield(L, -2, "train");
    
    lua_pushcfunction(L, lua_inferModel);
    lua_setfield(L, -2, "infer");
    
    lua_pushcfunction(L, lua_saveModel);
    lua_setfield(L, -2, "save");
    
    lua_pushcfunction(L, lua_loadModel);
    lua_setfield(L, -2, "load");
    
    // Gestion des paramètres
    lua_pushcfunction(L, lua_allocateParams);
    lua_setfield(L, -2, "allocate_params");
    
    lua_pushcfunction(L, lua_initWeights);
    lua_setfield(L, -2, "init_weights");
    
    lua_pushcfunction(L, lua_totalParams);
    lua_setfield(L, -2, "total_params");
    
    lua_pushcfunction(L, lua_pushLayer);
    lua_setfield(L, -2, "push_layer");
    
    // Forward/Backward
    lua_pushcfunction(L, lua_forwardPass);
    lua_setfield(L, -2, "forward");
    
    lua_pushcfunction(L, lua_backwardPass);
    lua_setfield(L, -2, "backward");
    
    lua_pushcfunction(L, lua_optimizerStep);
    lua_setfield(L, -2, "optimizer_step");
    
    lua_pushcfunction(L, lua_zeroGradients);
    lua_setfield(L, -2, "zero_grads");
    
    lua_pushcfunction(L, lua_getGradients);
    lua_setfield(L, -2, "get_gradients");
    
    // Hardware
    lua_pushcfunction(L, lua_setHardwareAccel);
    lua_setfield(L, -2, "set_hardware");
    
    lua_pushcfunction(L, lua_getHardwareCaps);
    lua_setfield(L, -2, "hardware_caps");
    
    lua_setglobal(L, "model");
    
    // ========== Table "architectures" ==========
    lua_newtable(L);
    
    lua_pushcfunction(L, lua_buildUNet);
    lua_setfield(L, -2, "unet");
    
    lua_pushcfunction(L, lua_buildVAE);
    lua_setfield(L, -2, "vae");
    
    lua_pushcfunction(L, lua_buildViT);
    lua_setfield(L, -2, "vit");
    
    lua_pushcfunction(L, lua_buildGAN);
    lua_setfield(L, -2, "gan");
    
    lua_pushcfunction(L, lua_buildDiffusion);
    lua_setfield(L, -2, "diffusion");
    
    lua_pushcfunction(L, lua_buildTransformer);
    lua_setfield(L, -2, "transformer");
    
    lua_pushcfunction(L, lua_buildResNet);
    lua_setfield(L, -2, "resnet");
    
    lua_pushcfunction(L, lua_buildMobileNet);
    lua_setfield(L, -2, "mobilenet");
    
    lua_pushcfunction(L, lua_buildFlux);
    lua_setfield(L, -2, "flux");
    
    lua_setglobal(L, "architectures");
    
    // ========== Table "flux" (Flux-specific operations) ==========
    lua_newtable(L);
    
    lua_pushcfunction(L, lua_fluxGenerate);
    lua_setfield(L, -2, "generate");
    
    lua_pushcfunction(L, lua_fluxEncodeImage);
    lua_setfield(L, -2, "encode_image");
    
    lua_pushcfunction(L, lua_fluxDecodeLatent);
    lua_setfield(L, -2, "decode_latent");
    
    lua_pushcfunction(L, lua_fluxEncodeText);
    lua_setfield(L, -2, "encode_text");
    
    lua_pushcfunction(L, lua_fluxSetPromptTokenizer);
    lua_setfield(L, -2, "set_tokenizer");
    
    lua_setglobal(L, "flux");
    
    // ========== Table "FluxModel" (API moderne orientée objet) ==========
    lua_newtable(L);
    
    // Constructeur
    lua_pushcfunction(L, lua_fluxModelNew);
    lua_setfield(L, -2, "new");
    
    // Modes d'exécution
    lua_pushcfunction(L, lua_fluxTrain);
    lua_setfield(L, -2, "train");
    
    lua_pushcfunction(L, lua_fluxEval);
    lua_setfield(L, -2, "eval");
    
    lua_pushcfunction(L, lua_fluxIsTraining);
    lua_setfield(L, -2, "isTraining");
    
    // VAE
    lua_pushcfunction(L, lua_fluxEncodeImage);
    lua_setfield(L, -2, "encodeImage");
    
    lua_pushcfunction(L, lua_fluxDecodeLatent);
    lua_setfield(L, -2, "decodeLatent");
    
    // Text processing
    lua_pushcfunction(L, lua_fluxTokenizePrompt);
    lua_setfield(L, -2, "tokenizePrompt");
    
    lua_pushcfunction(L, lua_fluxEncodeText);
    lua_setfield(L, -2, "encodeText");
    
    // Diffusion
    lua_pushcfunction(L, lua_fluxPredictNoise);
    lua_setfield(L, -2, "predictNoise");
    
    lua_pushcfunction(L, lua_fluxGenerate);
    lua_setfield(L, -2, "generate");
    
    // Training
    lua_pushcfunction(L, lua_fluxComputeDiffusionLoss);
    lua_setfield(L, -2, "computeDiffusionLoss");
    
    // Configuration
    lua_pushcfunction(L, lua_fluxSetPromptTokenizer);
    lua_setfield(L, -2, "setPromptTokenizer");
    
    lua_setglobal(L, "FluxModel");
    
    // ========== Table "layers" ==========
    lua_newtable(L);
    
    lua_pushcfunction(L, lua_computeConv2D);
    lua_setfield(L, -2, "conv2d");
    
    lua_pushcfunction(L, lua_computeLinear);
    lua_setfield(L, -2, "linear");
    
    lua_pushcfunction(L, lua_computeMaxPool2D);
    lua_setfield(L, -2, "maxpool2d");
    
    lua_pushcfunction(L, lua_computeAvgPool2D);
    lua_setfield(L, -2, "avgpool2d");
    
    lua_pushcfunction(L, lua_computeActivation);
    lua_setfield(L, -2, "activation");
    
    lua_pushcfunction(L, lua_computeBatchNorm);
    lua_setfield(L, -2, "batchnorm");
    
    lua_pushcfunction(L, lua_computeLayerNorm);
    lua_setfield(L, -2, "layernorm");
    
    lua_pushcfunction(L, lua_computeAttention);
    lua_setfield(L, -2, "attention");
    
    lua_setglobal(L, "layers");
    
    // ========== Table "tokenizer" ==========
    lua_newtable(L);
    
    lua_pushcfunction(L, lua_createTokenizer);
    lua_setfield(L, -2, "create");
    
    lua_pushcfunction(L, lua_tokenize);
    lua_setfield(L, -2, "tokenize");
    
    lua_pushcfunction(L, lua_detokenize);
    lua_setfield(L, -2, "detokenize");
    
    lua_pushcfunction(L, lua_getVocabSize);
    lua_setfield(L, -2, "vocab_size");
    
    lua_pushcfunction(L, lua_saveTokenizer);
    lua_setfield(L, -2, "save");
    
    lua_pushcfunction(L, lua_loadTokenizer);
    lua_setfield(L, -2, "load");
    
    // Méthodes de manipulation du vocabulaire
    lua_pushcfunction(L, lua_addToken);
    lua_setfield(L, -2, "add_token");
    
    lua_pushcfunction(L, lua_ensureVocabFromText);
    lua_setfield(L, -2, "ensure_vocab_from_text");
    
    lua_pushcfunction(L, lua_tokenizeEnsure);
    lua_setfield(L, -2, "tokenize_ensure");
    
    // Méthodes d'accès aux tokens spéciaux
    lua_pushcfunction(L, lua_getPadId);
    lua_setfield(L, -2, "pad_id");
    
    lua_pushcfunction(L, lua_getUnkId);
    lua_setfield(L, -2, "unk_id");
    
    lua_pushcfunction(L, lua_getSeqId);
    lua_setfield(L, -2, "seq_id");
    
    lua_pushcfunction(L, lua_getModId);
    lua_setfield(L, -2, "mod_id");
    
    lua_pushcfunction(L, lua_getMagId);
    lua_setfield(L, -2, "mag_id");
    
    lua_pushcfunction(L, lua_getTokenById);
    lua_setfield(L, -2, "get_token_by_id");
    
    // Méthodes BPE
    lua_pushcfunction(L, lua_learnBPEFromCorpus);
    lua_setfield(L, -2, "learn_bpe");
    
    lua_pushcfunction(L, lua_tokenizeBPE);
    lua_setfield(L, -2, "tokenize_bpe");
    
    lua_pushcfunction(L, lua_setMaxSequenceLength);
    lua_setfield(L, -2, "set_max_length");
    
    lua_pushcfunction(L, lua_padSequence);
    lua_setfield(L, -2, "pad_sequence");
    
    lua_pushcfunction(L, lua_batchTokenize);
    lua_setfield(L, -2, "batch_tokenize");
    
    // Statistiques et analyse
    lua_pushcfunction(L, lua_printVocabStats);
    lua_setfield(L, -2, "print_stats");
    
    lua_pushcfunction(L, lua_getTokenFrequencies);
    lua_setfield(L, -2, "get_frequencies");
    
    lua_pushcfunction(L, lua_analyzeText);
    lua_setfield(L, -2, "analyze_text");
    
    lua_pushcfunction(L, lua_extractKeywords);
    lua_setfield(L, -2, "extract_keywords");
    
    lua_setglobal(L, "tokenizer");
    
    // ========== Table "dataset" ==========
    lua_newtable(L);
    
    lua_pushcfunction(L, lua_loadDataset);
    lua_setfield(L, -2, "load");
    
    lua_pushcfunction(L, lua_getDataset);
    lua_setfield(L, -2, "get");
    
    lua_pushcfunction(L, lua_prepareSequences);
    lua_setfield(L, -2, "prepare_sequences");
    
    lua_setglobal(L, "dataset");
    
    // ========== Table "memory" ==========
    lua_newtable(L);
    
    lua_pushcfunction(L, lua_memoryConfig);
    lua_setfield(L, -2, "config");
    
    lua_pushcfunction(L, lua_memoryGetStats);
    lua_setfield(L, -2, "get_stats");
    
    lua_pushcfunction(L, lua_memoryPrintStats);
    lua_setfield(L, -2, "print_stats");
    
    lua_pushcfunction(L, lua_memoryClear);
    lua_setfield(L, -2, "clear");
    
    lua_pushcfunction(L, lua_memoryGetUsage);
    lua_setfield(L, -2, "get_usage");
    
    lua_pushcfunction(L, lua_memorySetLimit);
    lua_setfield(L, -2, "set_limit");
    
    lua_setglobal(L, "memory");
    
    // ========== Table "guard" (strict memory enforcement) ==========
    lua_newtable(L);
    
    lua_pushcfunction(L, lua_guardSetLimit);
    lua_setfield(L, -2, "set_limit");
    
    lua_pushcfunction(L, lua_guardGetStats);
    lua_setfield(L, -2, "get_stats");
    
    lua_pushcfunction(L, lua_guardPrintStats);
    lua_setfield(L, -2, "print_stats");
    
    lua_pushcfunction(L, lua_guardReset);
    lua_setfield(L, -2, "reset");
    
    lua_setglobal(L, "guard");
    
    // ========== Table "MemoryGuard" (nom moderne pour guard) ==========
    lua_newtable(L);
    
    lua_pushcfunction(L, lua_guardSetLimit);
    lua_setfield(L, -2, "setLimit");
    
    lua_pushcfunction(L, lua_memoryguardGetCurrentUsage);
    lua_setfield(L, -2, "getCurrentUsage");
    
    lua_pushcfunction(L, lua_memoryguardGetPeakUsage);
    lua_setfield(L, -2, "getPeakUsage");
    
    lua_pushcfunction(L, lua_memoryguardGetLimit);
    lua_setfield(L, -2, "getLimit");
    
    lua_pushcfunction(L, lua_guardGetStats);
    lua_setfield(L, -2, "getStats");
    
    lua_pushcfunction(L, lua_guardPrintStats);
    lua_setfield(L, -2, "printStats");
    
    lua_pushcfunction(L, lua_guardReset);
    lua_setfield(L, -2, "reset");
    
    lua_setglobal(L, "MemoryGuard");
    
    // ========== Table "allocator" (dynamic tensor allocator) ==========
    lua_newtable(L);
    
    lua_pushcfunction(L, lua_allocatorConfigure);
    lua_setfield(L, -2, "configure");
    
    lua_pushcfunction(L, lua_allocatorPrintStats);
    lua_setfield(L, -2, "print_stats");
    
    lua_pushcfunction(L, lua_allocatorGetStats);
    lua_setfield(L, -2, "get_stats");
    
    lua_setglobal(L, "allocator");
    
    // ========== Table "htop" (HtopDisplay monitoring) ==========
    lua_newtable(L);
    
    lua_pushcfunction(L, lua_htopCreate);
    lua_setfield(L, -2, "create");
    
    lua_pushcfunction(L, lua_htopUpdate);
    lua_setfield(L, -2, "update");
    
    lua_pushcfunction(L, lua_htopRender);
    lua_setfield(L, -2, "render");
    
    lua_pushcfunction(L, lua_htopClear);
    lua_setfield(L, -2, "clear");
    
    lua_pushcfunction(L, lua_htopEnable);
    lua_setfield(L, -2, "enable");
    
    lua_setglobal(L, "htop");
    
    // ========== Table "viz" (Visualizer SFML) ==========
    lua_newtable(L);
    
    lua_pushcfunction(L, lua_vizCreate);
    lua_setfield(L, -2, "create");
    
    lua_pushcfunction(L, lua_vizInitialize);
    lua_setfield(L, -2, "initialize");
    
    lua_pushcfunction(L, lua_vizIsOpen);
    lua_setfield(L, -2, "is_open");
    
    lua_pushcfunction(L, lua_vizProcessEvents);
    lua_setfield(L, -2, "process_events");
    
    lua_pushcfunction(L, lua_vizUpdate);
    lua_setfield(L, -2, "update");
    
    lua_pushcfunction(L, lua_vizAddImage);
    lua_setfield(L, -2, "add_image");
    
    lua_pushcfunction(L, lua_vizUpdateMetrics);
    lua_setfield(L, -2, "update_metrics");
    
    lua_pushcfunction(L, lua_vizAddLossPoint);
    lua_setfield(L, -2, "add_loss_point");
    
    lua_pushcfunction(L, lua_vizClear);
    lua_setfield(L, -2, "clear");
    
    lua_pushcfunction(L, lua_vizSetEnabled);
    lua_setfield(L, -2, "set_enabled");
    
    lua_pushcfunction(L, lua_vizSaveLossHistory);
    lua_setfield(L, -2, "save_loss_history");
    
    lua_setglobal(L, "viz");
    
    // ========== Fonctions utilitaires globales ==========
    lua_pushcfunction(L, lua_print);
    lua_setglobal(L, "log");
    
    lua_pushcfunction(L, lua_readJSON);
    lua_setglobal(L, "read_json");
    
    lua_pushcfunction(L, lua_writeJSON);
    lua_setglobal(L, "write_json");
}

// ============================================================================
// Implémentation des fonctions Lua
// ============================================================================

int LuaScripting::lua_createModel(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    // Argument: type de modèle (string)
    const char* model_type = luaL_checkstring(L, 1);
    
    // Argument optionnel: config (table)
    json config;
    if (lua_istable(L, 2)) {
        config = luaTableToJson(L, 2);
    }
    
    try {
        // Créer un modèle de base générique
        ctx.currentModel = std::make_shared<Model>();
        
        // Stocker le type et la config pour build()
        ctx.modelType = std::string(model_type);
        ctx.modelConfig = config;
        
        // NOUVEAU: Transférer la config au modèle pour accès aux dimensions
        ctx.currentModel->modelConfig = config;
        
        ctx.addLog("Modèle créé: " + std::string(model_type));
        lua_pushboolean(L, true);
    } catch (const std::exception& e) {
        ctx.addLog("Erreur création modèle: " + std::string(e.what()));
        lua_pushboolean(L, false);
        lua_pushstring(L, e.what());
        return 2;
    }
    
    return 1;
}

int LuaScripting::lua_buildModel(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.currentModel) {
        lua_pushboolean(L, false);
        lua_pushstring(L, "Aucun modèle créé");
        return 2;
    }
    
    try {
        // Construire selon le type
        if (ctx.modelType == "transformer" || ctx.modelType == "encoder" || ctx.modelType == "decoder") {
            // Transformer / Encoder / Decoder
            ModelArchitectures::TransformerConfig tfConfig;
            
            if (!ctx.modelConfig.empty()) {
                if (ctx.modelConfig.contains("vocab_size")) 
                    tfConfig.vocab_size = ctx.modelConfig["vocab_size"];
                if (ctx.modelConfig.contains("embed_dim")) 
                    tfConfig.d_model = ctx.modelConfig["embed_dim"];
                if (ctx.modelConfig.contains("num_layers")) 
                    tfConfig.num_layers = ctx.modelConfig["num_layers"];
                if (ctx.modelConfig.contains("num_heads")) 
                    tfConfig.num_heads = ctx.modelConfig["num_heads"];
                if (ctx.modelConfig.contains("d_ff")) 
                    tfConfig.d_ff = ctx.modelConfig["d_ff"];
                if (ctx.modelConfig.contains("max_seq_len")) 
                    tfConfig.max_seq_len = ctx.modelConfig["max_seq_len"];
                if (ctx.modelConfig.contains("dropout")) 
                    tfConfig.dropout = ctx.modelConfig["dropout"];
            }
            
            ModelArchitectures::buildTransformer(*ctx.currentModel, tfConfig);
        }
        else if (ctx.modelType == "unet") {
            // UNet - Segmentation / Image generation
            ModelArchitectures::UNetConfig unetConfig;
            
            if (!ctx.modelConfig.empty()) {
                if (ctx.modelConfig.contains("input_channels"))
                    unetConfig.input_channels = ctx.modelConfig["input_channels"];
                if (ctx.modelConfig.contains("output_channels"))
                    unetConfig.output_channels = ctx.modelConfig["output_channels"];
                if (ctx.modelConfig.contains("base_channels"))
                    unetConfig.base_channels = ctx.modelConfig["base_channels"];
                if (ctx.modelConfig.contains("num_levels"))
                    unetConfig.num_levels = ctx.modelConfig["num_levels"];
                if (ctx.modelConfig.contains("blocks_per_level"))
                    unetConfig.blocks_per_level = ctx.modelConfig["blocks_per_level"];
                if (ctx.modelConfig.contains("use_attention"))
                    unetConfig.use_attention = ctx.modelConfig["use_attention"];
                if (ctx.modelConfig.contains("use_residual"))
                    unetConfig.use_residual = ctx.modelConfig["use_residual"];
                if (ctx.modelConfig.contains("dropout"))
                    unetConfig.dropout = ctx.modelConfig["dropout"];
            }
            
            ModelArchitectures::buildUNet(*ctx.currentModel, unetConfig);
        }
        else if (ctx.modelType == "vae") {
            // VAE - Variational Autoencoder
            ModelArchitectures::VAEConfig vaeConfig;
            
            if (!ctx.modelConfig.empty()) {
                if (ctx.modelConfig.contains("input_dim"))
                    vaeConfig.input_dim = ctx.modelConfig["input_dim"];
                if (ctx.modelConfig.contains("latent_dim"))
                    vaeConfig.latent_dim = ctx.modelConfig["latent_dim"];
                if (ctx.modelConfig.contains("encoder_hidden") && ctx.modelConfig["encoder_hidden"].is_array()) {
                    vaeConfig.encoder_hidden.clear();
                    for (const auto& d : ctx.modelConfig["encoder_hidden"]) {
                        vaeConfig.encoder_hidden.push_back(d);
                    }
                }
                if (ctx.modelConfig.contains("decoder_hidden") && ctx.modelConfig["decoder_hidden"].is_array()) {
                    vaeConfig.decoder_hidden.clear();
                    for (const auto& d : ctx.modelConfig["decoder_hidden"]) {
                        vaeConfig.decoder_hidden.push_back(d);
                    }
                }
            }
            
            ModelArchitectures::buildVAE(*ctx.currentModel, vaeConfig);
        }
        else if (ctx.modelType == "vit") {
            // ViT - Vision Transformer
            ModelArchitectures::ViTConfig vitConfig;
            
            if (!ctx.modelConfig.empty()) {
                if (ctx.modelConfig.contains("image_size"))
                    vitConfig.image_size = ctx.modelConfig["image_size"];
                if (ctx.modelConfig.contains("patch_size"))
                    vitConfig.patch_size = ctx.modelConfig["patch_size"];
                if (ctx.modelConfig.contains("num_classes"))
                    vitConfig.num_classes = ctx.modelConfig["num_classes"];
                if (ctx.modelConfig.contains("embed_dim"))
                    vitConfig.d_model = ctx.modelConfig["embed_dim"];
                if (ctx.modelConfig.contains("num_layers"))
                    vitConfig.num_layers = ctx.modelConfig["num_layers"];
                if (ctx.modelConfig.contains("num_heads"))
                    vitConfig.num_heads = ctx.modelConfig["num_heads"];
                if (ctx.modelConfig.contains("mlp_ratio"))
                    vitConfig.mlp_ratio = ctx.modelConfig["mlp_ratio"];
                if (ctx.modelConfig.contains("dropout"))
                    vitConfig.dropout = ctx.modelConfig["dropout"];
            }
            
            ModelArchitectures::buildViT(*ctx.currentModel, vitConfig);
        }
        else if (ctx.modelType == "generator" || ctx.modelType == "gan") {
            // GAN Generator
            ModelArchitectures::GANConfig ganConfig;
            
            if (!ctx.modelConfig.empty()) {
                if (ctx.modelConfig.contains("latent_dim"))
                    ganConfig.latent_dim = ctx.modelConfig["latent_dim"];
                if (ctx.modelConfig.contains("image_channels"))
                    ganConfig.image_channels = ctx.modelConfig["image_channels"];
                if (ctx.modelConfig.contains("image_size"))
                    ganConfig.image_size = ctx.modelConfig["image_size"];
                if (ctx.modelConfig.contains("gen_channels"))
                    ganConfig.g_base_channels = ctx.modelConfig["gen_channels"];
            }
            
            ModelArchitectures::buildGenerator(*ctx.currentModel, ganConfig);
        }
        else if (ctx.modelType == "discriminator") {
            // GAN Discriminator
            ModelArchitectures::GANConfig ganConfig;
            
            if (!ctx.modelConfig.empty()) {
                if (ctx.modelConfig.contains("image_channels"))
                    ganConfig.image_channels = ctx.modelConfig["image_channels"];
                if (ctx.modelConfig.contains("image_size"))
                    ganConfig.image_size = ctx.modelConfig["image_size"];
                if (ctx.modelConfig.contains("disc_channels"))
                    ganConfig.d_base_channels = ctx.modelConfig["disc_channels"];
            }
            
            ModelArchitectures::buildDiscriminator(*ctx.currentModel, ganConfig);
        }
        else if (ctx.modelType == "diffusion") {
            // Diffusion Model
            ModelArchitectures::DiffusionConfig diffConfig;
            
            if (!ctx.modelConfig.empty()) {
                if (ctx.modelConfig.contains("image_channels"))
                    diffConfig.image_channels = ctx.modelConfig["image_channels"];
                if (ctx.modelConfig.contains("image_size"))
                    diffConfig.image_size = ctx.modelConfig["image_size"];
                if (ctx.modelConfig.contains("model_channels"))
                    diffConfig.base_channels = ctx.modelConfig["model_channels"];
                if (ctx.modelConfig.contains("num_res_blocks"))
                    diffConfig.num_res_blocks = ctx.modelConfig["num_res_blocks"];
            }
            
            ModelArchitectures::buildDiffusion(*ctx.currentModel, diffConfig);
        }
        else if (ctx.modelType == "resnet") {
            // ResNet
            ModelArchitectures::ResNetConfig resnetConfig;
            
            if (!ctx.modelConfig.empty()) {
                if (ctx.modelConfig.contains("num_classes"))
                    resnetConfig.num_classes = ctx.modelConfig["num_classes"];
                if (ctx.modelConfig.contains("base_channels"))
                    resnetConfig.base_channels = ctx.modelConfig["base_channels"];
                if (ctx.modelConfig.contains("use_bottleneck"))
                    resnetConfig.use_bottleneck = ctx.modelConfig["use_bottleneck"];
                if (ctx.modelConfig.contains("layers") && ctx.modelConfig["layers"].is_array()) {
                    resnetConfig.layers.clear();
                    for (const auto& l : ctx.modelConfig["layers"]) {
                        resnetConfig.layers.push_back(l);
                    }
                }
            }
            
            ModelArchitectures::buildResNet(*ctx.currentModel, resnetConfig);
        }
        else if (ctx.modelType == "mobilenet") {
            // MobileNet
            ModelArchitectures::MobileNetConfig mobileConfig;
            
            if (!ctx.modelConfig.empty()) {
                if (ctx.modelConfig.contains("num_classes"))
                    mobileConfig.num_classes = ctx.modelConfig["num_classes"];
                if (ctx.modelConfig.contains("width_mult"))
                    mobileConfig.width_multiplier = ctx.modelConfig["width_mult"];
                if (ctx.modelConfig.contains("resolution"))
                    mobileConfig.resolution = ctx.modelConfig["resolution"];
            }
            
            ModelArchitectures::buildMobileNetV2(*ctx.currentModel, mobileConfig);
        }
        else {
            // Fallback: appeler build() par défaut
            ctx.currentModel->build();
        }
        
        // Allouer et initialiser les paramètres
        size_t params = ctx.currentModel->totalParamCount();
        if (params > 0) {
            ctx.currentModel->allocateParams();
            ctx.currentModel->initializeWeights("he", 0);
        }
        
        // Initialiser l'encoder si un tokenizer est présent
        if (ctx.currentTokenizer && !ctx.currentEncoder) {
            int vocab_size = ctx.currentTokenizer->getVocabSize();
            int embed_dim = 256;  // Default, ou depuis config
            if (ctx.modelConfig.contains("embed_dim")) {
                embed_dim = ctx.modelConfig["embed_dim"];
            }
            
            ctx.currentEncoder = std::make_shared<Encoder>(embed_dim, vocab_size);
            ctx.currentEncoder->initRandom();
            ctx.currentEncoder->ensureVocabSize(vocab_size);
            ctx.addLog("Encoder créé: vocab=" + std::to_string(vocab_size) + ", dim=" + std::to_string(embed_dim));
        }
        
        ctx.addLog("Modèle construit avec " + std::to_string(params) + " paramètres");
        
        lua_pushboolean(L, true);
        lua_pushinteger(L, params);
        return 2;
    } catch (const std::exception& e) {
        lua_pushboolean(L, false);
        lua_pushstring(L, e.what());
        return 2;
    }
}

int LuaScripting::lua_trainModel(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.currentModel) {
        lua_pushboolean(L, false);
        lua_pushstring(L, "Aucun modèle créé");
        return 2;
    }
    
    // Arguments: epochs (number), learning_rate (number)
    int epochs = luaL_checkinteger(L, 1);
    double lr = luaL_checknumber(L, 2);
    
    ctx.addLog("Entraînement: " + std::to_string(epochs) + " epochs, LR=" + std::to_string(lr));
    
    // Boucle d'entraînement basique
    try {
        if (ctx.currentSequences.empty()) {
            lua_pushboolean(L, false);
            lua_pushstring(L, "Aucune séquence chargée. Utilisez dataset.prepare_sequences() d'abord.");
            return 2;
        }
        
        // Instancier l'Optimizer à partir de la configuration
        Optimizer opt;
        opt.initial_lr = lr;
        
        // Type d'optimizer depuis la config (défaut: ADAMW)
        if (ctx.modelConfig.contains("optimizer")) {
            std::string opt_type = ctx.modelConfig["optimizer"];
            if (opt_type == "sgd" || opt_type == "SGD") {
                opt.type = OptimizerType::SGD;
            } else if (opt_type == "adam" || opt_type == "ADAM") {
                opt.type = OptimizerType::ADAM;
            } else if (opt_type == "adamw" || opt_type == "ADAMW") {
                opt.type = OptimizerType::ADAMW;
            }
        } else {
            opt.type = OptimizerType::ADAMW;  // Défaut
        }
        
        // Paramètres de l'optimizer depuis la config
        if (ctx.modelConfig.contains("beta1")) {
            opt.beta1 = ctx.modelConfig["beta1"];
        }
        if (ctx.modelConfig.contains("beta2")) {
            opt.beta2 = ctx.modelConfig["beta2"];
        }
        if (ctx.modelConfig.contains("epsilon")) {
            opt.eps = ctx.modelConfig["epsilon"];
        }
        if (ctx.modelConfig.contains("weight_decay")) {
            opt.weight_decay = ctx.modelConfig["weight_decay"];
        }
        
        // Paramètres de LR decay depuis la config
        if (ctx.modelConfig.contains("min_lr")) {
            opt.min_lr = ctx.modelConfig["min_lr"];
        }
        if (ctx.modelConfig.contains("decay_rate")) {
            opt.decay_rate = ctx.modelConfig["decay_rate"];
        }
        if (ctx.modelConfig.contains("decay_steps")) {
            opt.decay_steps = ctx.modelConfig["decay_steps"];
        }
        if (ctx.modelConfig.contains("warmup_steps")) {
            opt.warmup_steps = ctx.modelConfig["warmup_steps"];
        }
        
        ctx.addLog("Optimizer configuré: type=" + std::to_string(static_cast<int>(opt.type)) + 
                   ", beta1=" + std::to_string(opt.beta1) + 
                   ", beta2=" + std::to_string(opt.beta2) + 
                   ", weight_decay=" + std::to_string(opt.weight_decay));
        
        for (int epoch = 0; epoch < epochs; ++epoch) {
            float epoch_loss = 0.0f;
            
            // Training sur toutes les séquences
            for (const auto& seq : ctx.currentSequences) {
                // Forward pass (simplifié)
                std::vector<uint8_t> dummy_output;
                ctx.currentModel->forward(dummy_output);
                
                // Backward pass (simplifié)
                ctx.currentModel->optimizerStep(opt, lr, nullptr);
            }
            
            ctx.addLog("Epoch " + std::to_string(epoch + 1) + "/" + std::to_string(epochs) + " completed");
        }
        
        lua_pushboolean(L, true);
        return 1;
    } catch (const std::exception& e) {
        lua_pushboolean(L, false);
        lua_pushstring(L, e.what());
        return 2;
    }
}

int LuaScripting::lua_inferModel(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.currentModel) {
        lua_pushboolean(L, false);
        lua_pushstring(L, "Aucun modèle créé");
        return 2;
    }
    
    // Argument: input (string ou table)
    if (lua_isstring(L, 1)) {
        const char* input = lua_tostring(L, 1);
        ctx.addLog("Inférence sur: " + std::string(input));
        
        try {
            // Tokenize input
            std::vector<int> tokens;
            if (ctx.currentTokenizer) {
                tokens = ctx.currentTokenizer->tokenize(input);
            } else {
                // Simple word tokenization si pas de tokenizer
                tokens = {1, 2, 3, 4, 5}; // Placeholder
            }
            
            // Encode
            std::vector<float> encoding;
            if (ctx.currentEncoder) {
                encoding = ctx.currentEncoder->encode(tokens);
                ctx.currentModel->setLastEncoding(encoding);
            }
            
            // Forward pass
            std::vector<uint8_t> output;
            ctx.currentModel->forward(output);
            
            // Decode result
            auto result = ctx.currentModel->eval(output);
            
            // Convert tokens back to text
            if (ctx.currentTokenizer && !result.tokens.empty()) {
                std::string decoded = ctx.currentTokenizer->decode(result.tokens);
                lua_pushstring(L, decoded.c_str());
            } else {
                lua_pushstring(L, "[output tokens]");
            }
        } catch (const std::exception& e) {
            lua_pushstring(L, ("Error: " + std::string(e.what())).c_str());
        }
    } else {
        lua_pushnil(L);
    }
    
    return 1;
}

int LuaScripting::lua_saveModel(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.currentModel) {
        lua_pushboolean(L, false);
        lua_pushstring(L, "Aucun modèle créé");
        return 2;
    }
    
    const char* path = luaL_checkstring(L, 1);
    
    try {
        fs::path save_dir(path);
        
        // Créer le dossier si nécessaire
        if (!fs::exists(save_dir)) {
            fs::create_directories(save_dir);
        }
        
        // Sauvegarder le checkpoint
        std::vector<MagicToken> magic_tokens;  // Empty for now
        
        Tokenizer tokenizer = ctx.currentTokenizer ? *ctx.currentTokenizer : Tokenizer();
        
        bool success = ctx.currentModel->saveCheckpoint(
            tokenizer,
            magic_tokens,
            save_dir,
            0  // epoch 0
        );
        
        if (success) {
            ctx.addLog("Modèle sauvegardé: " + std::string(path));
            lua_pushboolean(L, true);
        } else {
            lua_pushboolean(L, false);
            lua_pushstring(L, "Erreur lors de la sauvegarde");
            return 2;
        }
    } catch (const std::exception& e) {
        lua_pushboolean(L, false);
        lua_pushstring(L, e.what());
        return 2;
    }
    
    return 1;
}

int LuaScripting::lua_loadModel(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    const char* path = luaL_checkstring(L, 1);
    
    try {
        fs::path load_dir(path);
        
        if (!fs::exists(load_dir)) {
            lua_pushboolean(L, false);
            lua_pushstring(L, "Le chemin n'existe pas");
            return 2;
        }
        
        // Chercher le fichier safetensors
        fs::path safetensor_path = load_dir / "weights.safetensors";
        if (!fs::exists(safetensor_path)) {
            safetensor_path = load_dir / "model.safetensors";
        }
        
        Tokenizer tokenizer;
        Encoder encoder;
        std::vector<MagicToken> magic_tokens;
        
        bool success = ctx.currentModel->tryLoadExistingModel(
            load_dir,
            safetensor_path,
            tokenizer,
            encoder,
            magic_tokens
        );
        
        if (success) {
            ctx.currentTokenizer = std::make_shared<Tokenizer>(tokenizer);
            ctx.currentEncoder = std::make_shared<Encoder>(encoder);
            ctx.addLog("Modèle chargé: " + std::string(path));
            lua_pushboolean(L, true);
        } else {
            lua_pushboolean(L, false);
            lua_pushstring(L, "Erreur lors du chargement");
            return 2;
        }
    } catch (const std::exception& e) {
        lua_pushboolean(L, false);
        lua_pushstring(L, e.what());
        return 2;
    }
    
    return 1;
}

// ============================================================================
// Tokenizer
// ============================================================================

int LuaScripting::lua_createTokenizer(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    int max_vocab = luaL_checkinteger(L, 1);
    
    ctx.currentTokenizer = std::make_shared<Tokenizer>(max_vocab);
    ctx.addLog("Tokenizer créé (vocab_max=" + std::to_string(max_vocab) + ")");
    
    lua_pushboolean(L, true);
    return 1;
}

int LuaScripting::lua_tokenize(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.currentTokenizer) {
        lua_pushnil(L);
        lua_pushstring(L, "Aucun tokenizer créé");
        return 2;
    }
    
    const char* text = luaL_checkstring(L, 1);
    auto tokens = ctx.currentTokenizer->tokenizeBPE(text);
    
    // Retourner une table Lua avec les tokens
    lua_newtable(L);
    for (size_t i = 0; i < tokens.size(); ++i) {
        lua_pushinteger(L, tokens[i]);
        lua_rawseti(L, -2, i + 1);  // Indices Lua commencent à 1
    }
    
    return 1;
}

int LuaScripting::lua_detokenize(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.currentTokenizer) {
        lua_pushnil(L);
        lua_pushstring(L, "Aucun tokenizer créé");
        return 2;
    }
    
    // Lire table de tokens
    luaL_checktype(L, 1, LUA_TTABLE);
    
    std::vector<int> tokens;
    lua_pushnil(L);
    while (lua_next(L, 1) != 0) {
        tokens.push_back(lua_tointeger(L, -1));
        lua_pop(L, 1);
    }
    
    std::string text = ctx.currentTokenizer->decode(tokens);
    lua_pushstring(L, text.c_str());
    
    return 1;
}

// ============================================================================
// Dataset
// ============================================================================

int LuaScripting::lua_loadDataset(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    const char* dataset_dir = luaL_checkstring(L, 1);
    
    ctx.addLog("Chargement dataset: " + std::string(dataset_dir));
    
    try {
        fs::path dataset_path(dataset_dir);
        
        if (!fs::exists(dataset_path)) {
            lua_pushboolean(L, false);
            lua_pushstring(L, "Le dossier dataset n'existe pas");
            return 2;
        }
        
        // Charger les items du dataset
        std::vector<DatasetItem> items = loadDataset(dataset_dir);
        
        if (items.empty()) {
            ctx.addLog("⚠️  Attention: Dataset vide");
        } else {
            ctx.addLog("✓ " + std::to_string(items.size()) + " items chargés");
        }
        
        // Stocker le dataset dans le contexte
        ctx.currentDataset = std::move(items);
        
        if (!ctx.currentConfig.contains("dataset")) {
            ctx.currentConfig["dataset"] = json::object();
        }
        ctx.currentConfig["dataset"]["dir"] = dataset_dir;
        ctx.currentConfig["dataset"]["num_items"] = ctx.currentDataset.size();
        
        lua_pushboolean(L, true);
        lua_pushinteger(L, ctx.currentDataset.size());
        return 2;
    } catch (const std::exception& e) {
        lua_pushboolean(L, false);
        lua_pushstring(L, e.what());
        return 2;
    }
}

int LuaScripting::lua_getDataset(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    int index = luaL_checkinteger(L, 1);
    
    if (ctx.currentDataset.empty()) {
        lua_pushnil(L);
        lua_pushstring(L, "Aucun dataset chargé. Utilisez dataset.load() d'abord.");
        return 2;
    }
    
    if (index < 1 || index > (int)ctx.currentDataset.size()) {
        lua_pushnil(L);
        lua_pushstring(L, "Index hors limites");
        return 2;
    }
    
    try {
        const auto& item = ctx.currentDataset[index - 1]; // Lua est 1-indexed
        
        // Créer une table Lua avec les informations de l'item
        lua_newtable(L);
        
        // Ajouter les chemins de fichiers
        if (!item.text_file.empty()) {
            lua_pushstring(L, item.text_file.c_str());
            lua_setfield(L, -2, "text_file");
        }
        if (!item.image_file.empty()) {
            lua_pushstring(L, item.image_file.c_str());
            lua_setfield(L, -2, "image_file");
        }
        if (!item.audio_file.empty()) {
            lua_pushstring(L, item.audio_file.c_str());
            lua_setfield(L, -2, "audio_file");
        }
        if (!item.video_file.empty()) {
            lua_pushstring(L, item.video_file.c_str());
            lua_setfield(L, -2, "video_file");
        }
        
        // Ajouter les dimensions si disponibles
        if (item.w > 0) {
            lua_pushinteger(L, item.w);
            lua_setfield(L, -2, "width");
        }
        if (item.h > 0) {
            lua_pushinteger(L, item.h);
            lua_setfield(L, -2, "height");
        }
        
        // Charger et ajouter le contenu texte si présent
        if (!item.text_file.empty() && item.text.has_value()) {
            lua_pushstring(L, item.text.value().c_str());
            lua_setfield(L, -2, "text");
        }
        
        // Note: Pour img, audio, video (binaires), on ne les retourne pas directement
        // car ce sont des vecteurs de bytes. On pourrait ajouter des fonctions dédiées si nécessaire.
        
        return 1;
    } catch (const std::exception& e) {
        lua_pushnil(L);
        lua_pushstring(L, e.what());
        return 2;
    }
}

int LuaScripting::lua_prepareSequences(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    int seq_length = luaL_checkinteger(L, 1);
    
    ctx.addLog("Préparation séquences (longueur=" + std::to_string(seq_length) + ")");
    
    try {
        if (!ctx.currentConfig.contains("dataset") || 
            !ctx.currentConfig["dataset"].contains("dir")) {
            lua_pushboolean(L, false);
            lua_pushstring(L, "Aucun dataset chargé. Utilisez dataset.load() d'abord.");
            return 2;
        }
        
        std::string dataset_dir = ctx.currentConfig["dataset"]["dir"];
        std::vector<DatasetItem> items = loadDataset(dataset_dir);
        
        // Créer des séquences à partir des items
        ctx.currentSequences.clear();
        
        for (auto& item : items) {
            // Charger le texte si nécessaire (lazy loading)
            if (!item.text_file.empty() && !item.text.has_value()) {
                item.loadText();
            }
            
            if (item.text.has_value() && !item.text.value().empty()) {
                // Tokenize le texte
                std::vector<int> tokens;
                if (ctx.currentTokenizer) {
                    tokens = ctx.currentTokenizer->tokenize(item.text.value());
                    
                    // Padding/truncation à seq_length
                    if (tokens.size() < static_cast<size_t>(seq_length)) {
                        tokens.resize(seq_length, ctx.currentTokenizer->getPadId());
                    } else if (tokens.size() > static_cast<size_t>(seq_length)) {
                        tokens.resize(seq_length);
                    }
                    
                    ctx.currentSequences.push_back(tokens);
                }
            }
        }
        
        ctx.addLog("✓ " + std::to_string(ctx.currentSequences.size()) + " séquences préparées");
        
        lua_pushboolean(L, true);
        lua_pushinteger(L, ctx.currentSequences.size());
        return 2;
    } catch (const std::exception& e) {
        lua_pushboolean(L, false);
        lua_pushstring(L, e.what());
        return 2;
    }
}

// ============================================================================
// Utilitaires
// ============================================================================

int LuaScripting::lua_print(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    const char* msg = luaL_checkstring(L, 1);
    ctx.addLog(msg);
    
    return 0;
}

int LuaScripting::lua_readJSON(lua_State* L) {
    const char* filepath = luaL_checkstring(L, 1);
    
    try {
        std::ifstream f(filepath);
        json j;
        f >> j;
        
        jsonToLuaTable(L, j);
        return 1;
    } catch (const std::exception& e) {
        lua_pushnil(L);
        lua_pushstring(L, e.what());
        return 2;
    }
}

int LuaScripting::lua_writeJSON(lua_State* L) {
    const char* filepath = luaL_checkstring(L, 1);
    luaL_checktype(L, 2, LUA_TTABLE);
    
    try {
        json j = luaTableToJson(L, 2);
        
        std::ofstream f(filepath);
        f << j.dump(2);
        
        lua_pushboolean(L, true);
    } catch (const std::exception& e) {
        lua_pushboolean(L, false);
        lua_pushstring(L, e.what());
        return 2;
    }
    
    return 1;
}

// ============================================================================
// Conversion Lua <-> JSON
// ============================================================================

json LuaScripting::luaTableToJson(lua_State* L, int index) {
    json result;
    
    // Vérifier si c'est un array ou un objet
    bool is_array = true;
    int max_index = 0;
    
    lua_pushnil(L);
    while (lua_next(L, index) != 0) {
        if (!lua_isnumber(L, -2)) {
            is_array = false;
        } else {
            int idx = lua_tointeger(L, -2);
            if (idx > max_index) max_index = idx;
        }
        lua_pop(L, 1);
    }
    
    if (is_array) {
        result = json::array();
        for (int i = 1; i <= max_index; ++i) {
            lua_rawgeti(L, index, i);
            
            if (lua_isnil(L, -1)) {
                result.push_back(nullptr);
            } else if (lua_isboolean(L, -1)) {
                result.push_back(lua_toboolean(L, -1));
            } else if (lua_isnumber(L, -1)) {
                result.push_back(lua_tonumber(L, -1));
            } else if (lua_isstring(L, -1)) {
                result.push_back(lua_tostring(L, -1));
            } else if (lua_istable(L, -1)) {
                result.push_back(luaTableToJson(L, lua_gettop(L)));
            }
            
            lua_pop(L, 1);
        }
    } else {
        result = json::object();
        lua_pushnil(L);
        while (lua_next(L, index) != 0) {
            std::string key;
            if (lua_isstring(L, -2)) {
                key = lua_tostring(L, -2);
            } else if (lua_isnumber(L, -2)) {
                key = std::to_string(lua_tointeger(L, -2));
            }
            
            if (lua_isnil(L, -1)) {
                result[key] = nullptr;
            } else if (lua_isboolean(L, -1)) {
                result[key] = lua_toboolean(L, -1);
            } else if (lua_isnumber(L, -1)) {
                result[key] = lua_tonumber(L, -1);
            } else if (lua_isstring(L, -1)) {
                result[key] = lua_tostring(L, -1);
            } else if (lua_istable(L, -1)) {
                result[key] = luaTableToJson(L, lua_gettop(L));
            }
            
            lua_pop(L, 1);
        }
    }
    
    return result;
}

void LuaScripting::jsonToLuaTable(lua_State* L, const json& j) {
    if (j.is_null()) {
        lua_pushnil(L);
    } else if (j.is_boolean()) {
        lua_pushboolean(L, j.get<bool>());
    } else if (j.is_number_integer()) {
        lua_pushinteger(L, j.get<int>());
    } else if (j.is_number_float()) {
        lua_pushnumber(L, j.get<double>());
    } else if (j.is_string()) {
        lua_pushstring(L, j.get<std::string>().c_str());
    } else if (j.is_array()) {
        lua_newtable(L);
        for (size_t i = 0; i < j.size(); ++i) {
            jsonToLuaTable(L, j[i]);
            lua_rawseti(L, -2, i + 1);
        }
    } else if (j.is_object()) {
        lua_newtable(L);
        for (auto it = j.begin(); it != j.end(); ++it) {
            lua_pushstring(L, it.key().c_str());
            jsonToLuaTable(L, it.value());
            lua_settable(L, -3);
        }
    }
}

// ============================================================================
// Nouvelles implémentations - Model API étendue
// ============================================================================

int LuaScripting::lua_allocateParams(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.currentModel) {
        lua_pushboolean(L, false);
        lua_pushstring(L, "Aucun modèle créé");
        return 2;
    }
    
    try {
        ctx.currentModel->allocateParams();
        size_t count = ctx.currentModel->totalParamCount();
        ctx.addLog("Paramètres alloués: " + std::to_string(count));
        
        lua_pushboolean(L, true);
        lua_pushinteger(L, count);
        return 2;
    } catch (const std::exception& e) {
        lua_pushboolean(L, false);
        lua_pushstring(L, e.what());
        return 2;
    }
}

int LuaScripting::lua_initWeights(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.currentModel) {
        lua_pushboolean(L, false);
        lua_pushstring(L, "Aucun modèle créé");
        return 2;
    }
    
    const char* method = luaL_optstring(L, 1, "he");
    unsigned int seed = luaL_optinteger(L, 2, 0);
    
    try {
        ctx.currentModel->initializeWeights(method, seed);
        ctx.addLog("Poids initialisés: méthode=" + std::string(method));
        
        lua_pushboolean(L, true);
        return 1;
    } catch (const std::exception& e) {
        lua_pushboolean(L, false);
        lua_pushstring(L, e.what());
        return 2;
    }
}

int LuaScripting::lua_totalParams(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.currentModel) {
        lua_pushinteger(L, 0);
        return 1;
    }
    
    lua_pushinteger(L, ctx.currentModel->totalParamCount());
    return 1;
}

int LuaScripting::lua_pushLayer(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.currentModel) {
        lua_pushboolean(L, false);
        lua_pushstring(L, "Aucun modèle créé");
        return 2;
    }
    
    const char* name = luaL_checkstring(L, 1);
    const char* type = luaL_checkstring(L, 2);
    size_t params_count = luaL_checkinteger(L, 3);
    
    ctx.currentModel->push(name, type, params_count);
    lua_pushboolean(L, true);
    return 1;
}

int LuaScripting::lua_forwardPass(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.currentModel) {
        lua_pushnil(L);
        lua_pushstring(L, "Aucun modèle créé");
        return 2;
    }
    
    // Argument 1: input (table de floats)
    luaL_checktype(L, 1, LUA_TTABLE);
    
    // Argument 2 (optionnel): training (bool, défaut: true)
    bool training = true;
    if (lua_gettop(L) >= 2 && lua_isboolean(L, 2)) {
        training = lua_toboolean(L, 2);
    }
    
    std::vector<float> input;
    lua_pushnil(L);
    while (lua_next(L, 1) != 0) {
        input.push_back(lua_tonumber(L, -1));
        lua_pop(L, 1);
    }
    
    try {
        std::vector<float> output = ctx.currentModel->forwardPass(input, training);
        
        // Retourner output comme table
        lua_newtable(L);
        for (size_t i = 0; i < output.size(); ++i) {
            lua_pushnumber(L, output[i]);
            lua_rawseti(L, -2, i + 1);
        }
        return 1;
    } catch (const std::exception& e) {
        lua_pushnil(L);
        lua_pushstring(L, e.what());
        return 2;
    }
}

int LuaScripting::lua_backwardPass(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.currentModel) {
        lua_pushboolean(L, false);
        lua_pushstring(L, "Aucun modèle créé");
        return 2;
    }
    
    // Argument: loss_gradient (table de floats)
    luaL_checktype(L, 1, LUA_TTABLE);
    
    std::vector<float> loss_grad;
    lua_pushnil(L);
    while (lua_next(L, 1) != 0) {
        loss_grad.push_back(lua_tonumber(L, -1));
        lua_pop(L, 1);
    }
    
    try {
        Gradients grads = ctx.currentModel->backwardPass(loss_grad);
        ctx.addLog("Backward pass complété");
        
        lua_pushboolean(L, true);
        return 1;
    } catch (const std::exception& e) {
        lua_pushboolean(L, false);
        lua_pushstring(L, e.what());
        return 2;
    }
}

int LuaScripting::lua_zeroGradients(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.currentModel) {
        lua_pushboolean(L, false);
        lua_pushstring(L, "Aucun modèle créé");
        return 2;
    }
    
    try {
        ctx.currentModel->zeroGradients();
        lua_pushboolean(L, true);
        return 1;
    } catch (const std::exception& e) {
        lua_pushboolean(L, false);
        lua_pushstring(L, e.what());
        return 2;
    }
}

int LuaScripting::lua_getGradients(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.currentModel) {
        lua_pushnil(L);
        lua_pushstring(L, "Aucun modèle créé");
        return 2;
    }
    
    try {
        Gradients grads = ctx.currentModel->getGradients();
        
        // Retourner les gradients comme table (ordonnée par index)
        lua_newtable(L);
        size_t lua_idx = 1;
        
        // Parcourir tous les indices dans l'ordre
        for (const auto& [param_idx, grad_value] : grads.param_grads) {
            lua_pushnumber(L, grad_value);
            lua_rawseti(L, -2, lua_idx++);
        }
        
        return 1;
    } catch (const std::exception& e) {
        lua_pushnil(L);
        lua_pushstring(L, e.what());
        return 2;
    }
}

int LuaScripting::lua_optimizerStep(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.currentModel) {
        lua_pushboolean(L, false);
        lua_pushstring(L, "Aucun modèle créé");
        return 2;
    }
    
    double lr = luaL_checknumber(L, 1);
    const char* opt_type = luaL_optstring(L, 2, "adamw");
    
    try {
        Optimizer opt;
        opt.initial_lr = lr;
        
        if (std::string(opt_type) == "sgd") {
            opt.type = OptimizerType::SGD;
        } else if (std::string(opt_type) == "adam") {
            opt.type = OptimizerType::ADAM;
        } else {
            opt.type = OptimizerType::ADAMW;
        }
        
        ctx.currentModel->optimizerStep(opt, lr, nullptr);
        lua_pushboolean(L, true);
        return 1;
    } catch (const std::exception& e) {
        lua_pushboolean(L, false);
        lua_pushstring(L, e.what());
        return 2;
    }
}

int LuaScripting::lua_setHardwareAccel(lua_State* L) {
    bool enable = lua_toboolean(L, 1);
    Model::setHardwareAcceleration(enable);
    lua_pushboolean(L, true);
    return 1;
}

int LuaScripting::lua_getHardwareCaps(lua_State* L) {
    lua_newtable(L);
    
    lua_pushboolean(L, Model::hasAVX2());
    lua_setfield(L, -2, "avx2");
    
    lua_pushboolean(L, Model::hasFMA());
    lua_setfield(L, -2, "fma");
    
    lua_pushboolean(L, Model::hasF16C());
    lua_setfield(L, -2, "f16c");
    
    lua_pushboolean(L, Model::hasBMI2());
    lua_setfield(L, -2, "bmi2");
    
    return 1;
}

// ============================================================================
// ModelArchitectures API
// ============================================================================

int LuaScripting::lua_buildUNet(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.currentModel) {
        lua_pushboolean(L, false);
        lua_pushstring(L, "Aucun modèle créé");
        return 2;
    }
    
    ModelArchitectures::UNetConfig config;
    
    if (lua_istable(L, 1)) {
        lua_getfield(L, 1, "input_channels");
        if (lua_isnumber(L, -1)) config.input_channels = lua_tointeger(L, -1);
        lua_pop(L, 1);
        
        lua_getfield(L, 1, "output_channels");
        if (lua_isnumber(L, -1)) config.output_channels = lua_tointeger(L, -1);
        lua_pop(L, 1);
        
        lua_getfield(L, 1, "base_channels");
        if (lua_isnumber(L, -1)) config.base_channels = lua_tointeger(L, -1);
        lua_pop(L, 1);
        
        lua_getfield(L, 1, "num_levels");
        if (lua_isnumber(L, -1)) config.num_levels = lua_tointeger(L, -1);
        lua_pop(L, 1);
    }
    
    try {
        ModelArchitectures::buildUNet(*ctx.currentModel, config);
        ctx.addLog("Architecture UNet construite");
        lua_pushboolean(L, true);
        return 1;
    } catch (const std::exception& e) {
        lua_pushboolean(L, false);
        lua_pushstring(L, e.what());
        return 2;
    }
}

int LuaScripting::lua_buildVAE(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.currentModel) {
        lua_pushboolean(L, false);
        lua_pushstring(L, "Aucun modèle créé");
        return 2;
    }
    
    ModelArchitectures::VAEConfig config;
    
    if (lua_istable(L, 1)) {
        lua_getfield(L, 1, "input_dim");
        if (lua_isnumber(L, -1)) config.input_dim = lua_tointeger(L, -1);
        lua_pop(L, 1);
        
        lua_getfield(L, 1, "latent_dim");
        if (lua_isnumber(L, -1)) config.latent_dim = lua_tointeger(L, -1);
        lua_pop(L, 1);
    }
    
    try {
        ModelArchitectures::buildVAE(*ctx.currentModel, config);
        ctx.addLog("Architecture VAE construite");
        lua_pushboolean(L, true);
        return 1;
    } catch (const std::exception& e) {
        lua_pushboolean(L, false);
        lua_pushstring(L, e.what());
        return 2;
    }
}

int LuaScripting::lua_buildViT(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.currentModel) {
        lua_pushboolean(L, false);
        lua_pushstring(L, "Aucun modèle créé");
        return 2;
    }
    
    ModelArchitectures::ViTConfig config;
    
    if (lua_istable(L, 1)) {
        lua_getfield(L, 1, "image_size");
        if (lua_isnumber(L, -1)) config.image_size = lua_tointeger(L, -1);
        lua_pop(L, 1);
        
        lua_getfield(L, 1, "patch_size");
        if (lua_isnumber(L, -1)) config.patch_size = lua_tointeger(L, -1);
        lua_pop(L, 1);
        
        lua_getfield(L, 1, "num_classes");
        if (lua_isnumber(L, -1)) config.num_classes = lua_tointeger(L, -1);
        lua_pop(L, 1);
        
        lua_getfield(L, 1, "d_model");
        if (lua_isnumber(L, -1)) config.d_model = lua_tointeger(L, -1);
        lua_pop(L, 1);
        
        lua_getfield(L, 1, "num_layers");
        if (lua_isnumber(L, -1)) config.num_layers = lua_tointeger(L, -1);
        lua_pop(L, 1);
    }
    
    try {
        ModelArchitectures::buildViT(*ctx.currentModel, config);
        ctx.addLog("Architecture ViT construite");
        lua_pushboolean(L, true);
        return 1;
    } catch (const std::exception& e) {
        lua_pushboolean(L, false);
        lua_pushstring(L, e.what());
        return 2;
    }
}

int LuaScripting::lua_buildGAN(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.currentModel) {
        lua_pushboolean(L, false);
        lua_pushstring(L, "Aucun modèle créé");
        return 2;
    }
    
    ModelArchitectures::GANConfig config;
    const char* model_type = luaL_optstring(L, 1, "generator");
    
    if (lua_istable(L, 2)) {
        lua_getfield(L, 2, "latent_dim");
        if (lua_isnumber(L, -1)) config.latent_dim = lua_tointeger(L, -1);
        lua_pop(L, 1);
        
        lua_getfield(L, 2, "image_size");
        if (lua_isnumber(L, -1)) config.image_size = lua_tointeger(L, -1);
        lua_pop(L, 1);
    }
    
    try {
        if (std::string(model_type) == "generator") {
            ModelArchitectures::buildGenerator(*ctx.currentModel, config);
            ctx.addLog("GAN Generator construit");
        } else {
            ModelArchitectures::buildDiscriminator(*ctx.currentModel, config);
            ctx.addLog("GAN Discriminator construit");
        }
        lua_pushboolean(L, true);
        return 1;
    } catch (const std::exception& e) {
        lua_pushboolean(L, false);
        lua_pushstring(L, e.what());
        return 2;
    }
}

int LuaScripting::lua_buildDiffusion(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.currentModel) {
        lua_pushboolean(L, false);
        lua_pushstring(L, "Aucun modèle créé");
        return 2;
    }
    
    ModelArchitectures::DiffusionConfig config;
    
    if (lua_istable(L, 1)) {
        lua_getfield(L, 1, "image_size");
        if (lua_isnumber(L, -1)) config.image_size = lua_tointeger(L, -1);
        lua_pop(L, 1);
        
        lua_getfield(L, 1, "base_channels");
        if (lua_isnumber(L, -1)) config.base_channels = lua_tointeger(L, -1);
        lua_pop(L, 1);
    }
    
    try {
        ModelArchitectures::buildDiffusion(*ctx.currentModel, config);
        ctx.addLog("Architecture Diffusion construite");
        lua_pushboolean(L, true);
        return 1;
    } catch (const std::exception& e) {
        lua_pushboolean(L, false);
        lua_pushstring(L, e.what());
        return 2;
    }
}

int LuaScripting::lua_buildTransformer(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.currentModel) {
        lua_pushboolean(L, false);
        lua_pushstring(L, "Aucun modèle créé");
        return 2;
    }
    
    ModelArchitectures::TransformerConfig config;
    
    if (lua_istable(L, 1)) {
        lua_getfield(L, 1, "vocab_size");
        if (lua_isnumber(L, -1)) config.vocab_size = lua_tointeger(L, -1);
        lua_pop(L, 1);
        
        lua_getfield(L, 1, "d_model");
        if (lua_isnumber(L, -1)) config.d_model = lua_tointeger(L, -1);
        lua_pop(L, 1);
        
        lua_getfield(L, 1, "num_layers");
        if (lua_isnumber(L, -1)) config.num_layers = lua_tointeger(L, -1);
        lua_pop(L, 1);
        
        lua_getfield(L, 1, "num_heads");
        if (lua_isnumber(L, -1)) config.num_heads = lua_tointeger(L, -1);
        lua_pop(L, 1);
    }
    
    try {
        ModelArchitectures::buildTransformer(*ctx.currentModel, config);
        ctx.addLog("Architecture Transformer construite");
        lua_pushboolean(L, true);
        return 1;
    } catch (const std::exception& e) {
        lua_pushboolean(L, false);
        lua_pushstring(L, e.what());
        return 2;
    }
}

int LuaScripting::lua_buildResNet(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.currentModel) {
        lua_pushboolean(L, false);
        lua_pushstring(L, "Aucun modèle créé");
        return 2;
    }
    
    ModelArchitectures::ResNetConfig config;
    
    if (lua_istable(L, 1)) {
        lua_getfield(L, 1, "num_classes");
        if (lua_isnumber(L, -1)) config.num_classes = lua_tointeger(L, -1);
        lua_pop(L, 1);
    }
    
    try {
        ModelArchitectures::buildResNet(*ctx.currentModel, config);
        ctx.addLog("Architecture ResNet construite");
        lua_pushboolean(L, true);
        return 1;
    } catch (const std::exception& e) {
        lua_pushboolean(L, false);
        lua_pushstring(L, e.what());
        return 2;
    }
}

int LuaScripting::lua_buildMobileNet(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.currentModel) {
        lua_pushboolean(L, false);
        lua_pushstring(L, "Aucun modèle créé");
        return 2;
    }
    
    ModelArchitectures::MobileNetConfig config;
    
    if (lua_istable(L, 1)) {
        lua_getfield(L, 1, "num_classes");
        if (lua_isnumber(L, -1)) config.num_classes = lua_tointeger(L, -1);
        lua_pop(L, 1);
        
        lua_getfield(L, 1, "width_multiplier");
        if (lua_isnumber(L, -1)) config.width_multiplier = lua_tonumber(L, -1);
        lua_pop(L, 1);
    }
    
    try {
        ModelArchitectures::buildMobileNetV2(*ctx.currentModel, config);
        ctx.addLog("Architecture MobileNetV2 construite");
        lua_pushboolean(L, true);
        return 1;
    } catch (const std::exception& e) {
        lua_pushboolean(L, false);
        lua_pushstring(L, e.what());
        return 2;
    }
}

int LuaScripting::lua_buildFlux(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.currentModel) {
        lua_pushboolean(L, false);
        lua_pushstring(L, "Aucun modèle créé");
        return 2;
    }
    
    ModelArchitectures::FluxConfig config;
    
    // Lire la configuration depuis la table Lua
    if (lua_istable(L, 1)) {
        // Dimensions générales
        lua_getfield(L, 1, "latent_channels");
        if (lua_isnumber(L, -1)) config.latent_channels = lua_tointeger(L, -1);
        lua_pop(L, 1);
        
        lua_getfield(L, 1, "latent_resolution");
        if (lua_isnumber(L, -1)) config.latent_resolution = lua_tointeger(L, -1);
        lua_pop(L, 1);
        
        lua_getfield(L, 1, "image_resolution");
        if (lua_isnumber(L, -1)) config.image_resolution = lua_tointeger(L, -1);
        lua_pop(L, 1);
        
        // VAE config
        lua_getfield(L, 1, "vae_base_channels");
        if (lua_isnumber(L, -1)) config.vae_base_channels = lua_tointeger(L, -1);
        lua_pop(L, 1);
        
        lua_getfield(L, 1, "vae_num_res_blocks");
        if (lua_isnumber(L, -1)) config.vae_num_res_blocks = lua_tointeger(L, -1);
        lua_pop(L, 1);
        
        // Text conditioning
        lua_getfield(L, 1, "text_embed_dim");
        if (lua_isnumber(L, -1)) config.text_embed_dim = lua_tointeger(L, -1);
        lua_pop(L, 1);
        
        lua_getfield(L, 1, "text_max_length");
        if (lua_isnumber(L, -1)) config.text_max_length = lua_tointeger(L, -1);
        lua_pop(L, 1);
        
        lua_getfield(L, 1, "vocab_size");
        if (lua_isnumber(L, -1)) config.vocab_size = lua_tointeger(L, -1);
        lua_pop(L, 1);
        
        // Transformer blocks
        lua_getfield(L, 1, "num_transformer_blocks");
        if (lua_isnumber(L, -1)) config.num_transformer_blocks = lua_tointeger(L, -1);
        lua_pop(L, 1);
        
        lua_getfield(L, 1, "transformer_dim");
        if (lua_isnumber(L, -1)) config.transformer_dim = lua_tointeger(L, -1);
        lua_pop(L, 1);
        
        lua_getfield(L, 1, "num_attention_heads");
        if (lua_isnumber(L, -1)) config.num_attention_heads = lua_tointeger(L, -1);
        lua_pop(L, 1);
        
        // Diffusion
        lua_getfield(L, 1, "num_timesteps");
        if (lua_isnumber(L, -1)) config.num_timesteps = lua_tointeger(L, -1);
        lua_pop(L, 1);
        
        lua_getfield(L, 1, "beta_start");
        if (lua_isnumber(L, -1)) config.beta_start = lua_tonumber(L, -1);
        lua_pop(L, 1);
        
        lua_getfield(L, 1, "beta_end");
        if (lua_isnumber(L, -1)) config.beta_end = lua_tonumber(L, -1);
        lua_pop(L, 1);
    }
    
    try {
        // Tenter de caster en FluxModel
        ModelArchitectures::FluxModel* flux_model = 
            dynamic_cast<ModelArchitectures::FluxModel*>(ctx.currentModel.get());
        
        if (flux_model) {
            flux_model->setConfig(config);
            flux_model->buildFluxArchitecture();
            ctx.addLog("Architecture Flux construite (FluxModel natif)");
        } else {
            // Sinon utiliser la fonction helper
            ModelArchitectures::buildFlux(*ctx.currentModel, config);
            ctx.addLog("Architecture Flux construite (Model générique)");
        }
        
        lua_pushboolean(L, true);
        return 1;
    } catch (const std::exception& e) {
        lua_pushboolean(L, false);
        lua_pushstring(L, e.what());
        return 2;
    }
}

// ============================================================================
// Flux-specific Operations
// ============================================================================

int LuaScripting::lua_fluxGenerate(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.currentModel) {
        lua_pushnil(L);
        lua_pushstring(L, "Aucun modèle créé");
        return 2;
    }
    
    // Tenter de caster en FluxModel
    ModelArchitectures::FluxModel* flux_model = 
        dynamic_cast<ModelArchitectures::FluxModel*>(ctx.currentModel.get());
    
    if (!flux_model) {
        lua_pushnil(L);
        lua_pushstring(L, "Le modèle actuel n'est pas un FluxModel");
        return 2;
    }
    
    // Arguments: prompt (string), num_steps (optional), guidance_scale (optional)
    const char* prompt = luaL_checkstring(L, 1);
    int num_steps = luaL_optinteger(L, 2, 50);
    float guidance_scale = luaL_optnumber(L, 3, 7.5f);
    
    try {
        ctx.addLog("Génération Flux: \"" + std::string(prompt) + 
                   "\" (steps=" + std::to_string(num_steps) + 
                   ", guidance=" + std::to_string(guidance_scale) + ")");
        
        std::vector<float> image = flux_model->generate(prompt, num_steps, guidance_scale);
        
        // Retourner l'image comme table Lua
        lua_newtable(L);
        lua_pushinteger(L, flux_model->getImageResolution());
        lua_setfield(L, -2, "resolution");
        lua_pushinteger(L, image.size());
        lua_setfield(L, -2, "size");
        
        // Ajouter les données (optionnel, pour petites images)
        // Pour grandes images, mieux vaut sauvegarder directement
        
        ctx.addLog("Génération terminée");
        return 1;
    } catch (const std::exception& e) {
        lua_pushnil(L);
        lua_pushstring(L, e.what());
        return 2;
    }
}

int LuaScripting::lua_fluxEncodeImage(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.currentModel) {
        lua_pushnil(L);
        lua_pushstring(L, "Aucun modèle créé");
        return 2;
    }
    
    ModelArchitectures::FluxModel* flux_model = 
        dynamic_cast<ModelArchitectures::FluxModel*>(ctx.currentModel.get());
    
    if (!flux_model) {
        lua_pushnil(L);
        lua_pushstring(L, "Le modèle actuel n'est pas un FluxModel");
        return 2;
    }
    
    // Argument: image data (table de floats)
    luaL_checktype(L, 1, LUA_TTABLE);
    
    std::vector<float> image;
    lua_pushnil(L);
    while (lua_next(L, 1) != 0) {
        image.push_back(lua_tonumber(L, -1));
        lua_pop(L, 1);
    }
    
    try {
        std::vector<float> latent = flux_model->encodeImage(image);
        
        // Retourner le latent comme table Lua
        lua_newtable(L);
        for (size_t i = 0; i < latent.size(); ++i) {
            lua_pushnumber(L, latent[i]);
            lua_rawseti(L, -2, i + 1);
        }
        
        return 1;
    } catch (const std::exception& e) {
        lua_pushnil(L);
        lua_pushstring(L, e.what());
        return 2;
    }
}

int LuaScripting::lua_fluxDecodeLatent(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.currentModel) {
        lua_pushnil(L);
        lua_pushstring(L, "Aucun modèle créé");
        return 2;
    }
    
    ModelArchitectures::FluxModel* flux_model = 
        dynamic_cast<ModelArchitectures::FluxModel*>(ctx.currentModel.get());
    
    if (!flux_model) {
        lua_pushnil(L);
        lua_pushstring(L, "Le modèle actuel n'est pas un FluxModel");
        return 2;
    }
    
    // Argument: latent data (table de floats)
    luaL_checktype(L, 1, LUA_TTABLE);
    
    std::vector<float> latent;
    lua_pushnil(L);
    while (lua_next(L, 1) != 0) {
        latent.push_back(lua_tonumber(L, -1));
        lua_pop(L, 1);
    }
    
    try {
        std::vector<float> image = flux_model->decodeLatent(latent);
        
        // Retourner l'image comme table Lua
        lua_newtable(L);
        for (size_t i = 0; i < image.size(); ++i) {
            lua_pushnumber(L, image[i]);
            lua_rawseti(L, -2, i + 1);
        }
        
        return 1;
    } catch (const std::exception& e) {
        lua_pushnil(L);
        lua_pushstring(L, e.what());
        return 2;
    }
}

int LuaScripting::lua_fluxEncodeText(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.currentModel) {
        lua_pushnil(L);
        lua_pushstring(L, "Aucun modèle créé");
        return 2;
    }
    
    ModelArchitectures::FluxModel* flux_model = 
        dynamic_cast<ModelArchitectures::FluxModel*>(ctx.currentModel.get());
    
    if (!flux_model) {
        lua_pushnil(L);
        lua_pushstring(L, "Le modèle actuel n'est pas un FluxModel");
        return 2;
    }
    
    // Argument: text (string) ou tokens (table)
    std::vector<int> tokens;
    
    if (lua_isstring(L, 1)) {
        const char* text = lua_tostring(L, 1);
        tokens = flux_model->tokenizePrompt(text);
    } else if (lua_istable(L, 1)) {
        lua_pushnil(L);
        while (lua_next(L, 1) != 0) {
            tokens.push_back(lua_tointeger(L, -1));
            lua_pop(L, 1);
        }
    } else {
        lua_pushnil(L);
        lua_pushstring(L, "Argument invalide: attendu string ou table");
        return 2;
    }
    
    try {
        std::vector<float> text_embedding = flux_model->encodeText(tokens);
        
        // Retourner l'embedding comme table Lua
        lua_newtable(L);
        for (size_t i = 0; i < text_embedding.size(); ++i) {
            lua_pushnumber(L, text_embedding[i]);
            lua_rawseti(L, -2, i + 1);
        }
        
        return 1;
    } catch (const std::exception& e) {
        lua_pushnil(L);
        lua_pushstring(L, e.what());
        return 2;
    }
}

int LuaScripting::lua_fluxSetPromptTokenizer(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.currentModel) {
        lua_pushboolean(L, false);
        lua_pushstring(L, "Aucun modèle créé");
        return 2;
    }
    
    ModelArchitectures::FluxModel* flux_model = 
        dynamic_cast<ModelArchitectures::FluxModel*>(ctx.currentModel.get());
    
    if (!flux_model) {
        lua_pushboolean(L, false);
        lua_pushstring(L, "Le modèle actuel n'est pas un FluxModel");
        return 2;
    }
    
    // Utiliser le tokenizer du contexte
    if (!ctx.currentTokenizer) {
        lua_pushboolean(L, false);
        lua_pushstring(L, "Aucun tokenizer disponible");
        return 2;
    }
    
    try {
        flux_model->setPromptTokenizer(ctx.currentTokenizer);
        ctx.addLog("Tokenizer assigné au modèle Flux");
        lua_pushboolean(L, true);
        return 1;
    } catch (const std::exception& e) {
        lua_pushboolean(L, false);
        lua_pushstring(L, e.what());
        return 2;
    }
}

int LuaScripting::lua_fluxTrain(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.currentModel) {
        lua_pushboolean(L, false);
        lua_pushstring(L, "Aucun modèle créé");
        return 2;
    }
    
    ModelArchitectures::FluxModel* flux_model = 
        dynamic_cast<ModelArchitectures::FluxModel*>(ctx.currentModel.get());
    
    if (!flux_model) {
        lua_pushboolean(L, false);
        lua_pushstring(L, "Le modèle actuel n'est pas un FluxModel");
        return 2;
    }
    
    flux_model->train();
    ctx.addLog("FluxModel: Mode training activé");
    lua_pushboolean(L, true);
    return 1;
}

int LuaScripting::lua_fluxEval(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.currentModel) {
        lua_pushboolean(L, false);
        lua_pushstring(L, "Aucun modèle créé");
        return 2;
    }
    
    ModelArchitectures::FluxModel* flux_model = 
        dynamic_cast<ModelArchitectures::FluxModel*>(ctx.currentModel.get());
    
    if (!flux_model) {
        lua_pushboolean(L, false);
        lua_pushstring(L, "Le modèle actuel n'est pas un FluxModel");
        return 2;
    }
    
    flux_model->eval();
    ctx.addLog("FluxModel: Mode evaluation/inference activé");
    lua_pushboolean(L, true);
    return 1;
}

int LuaScripting::lua_fluxIsTraining(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.currentModel) {
        lua_pushboolean(L, false);
        return 1;
    }
    
    ModelArchitectures::FluxModel* flux_model = 
        dynamic_cast<ModelArchitectures::FluxModel*>(ctx.currentModel.get());
    
    if (!flux_model) {
        lua_pushboolean(L, false);
        return 1;
    }
    
    lua_pushboolean(L, flux_model->isTraining());
    return 1;
}

int LuaScripting::lua_fluxTokenizePrompt(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.currentModel) {
        lua_pushnil(L);
        lua_pushstring(L, "Aucun modèle créé");
        return 2;
    }
    
    ModelArchitectures::FluxModel* flux_model = 
        dynamic_cast<ModelArchitectures::FluxModel*>(ctx.currentModel.get());
    
    if (!flux_model) {
        lua_pushnil(L);
        lua_pushstring(L, "Le modèle actuel n'est pas un FluxModel");
        return 2;
    }
    
    const char* prompt = luaL_checkstring(L, 1);
    
    try {
        std::vector<int> tokens = flux_model->tokenizePrompt(prompt);
        
        // Retourner les tokens comme table Lua
        lua_newtable(L);
        for (size_t i = 0; i < tokens.size(); ++i) {
            lua_pushinteger(L, tokens[i]);
            lua_rawseti(L, -2, i + 1);
        }
        
        return 1;
    } catch (const std::exception& e) {
        lua_pushnil(L);
        lua_pushstring(L, e.what());
        return 2;
    }
}

int LuaScripting::lua_fluxPredictNoise(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.currentModel) {
        lua_pushnil(L);
        lua_pushstring(L, "Aucun modèle créé");
        return 2;
    }
    
    ModelArchitectures::FluxModel* flux_model = 
        dynamic_cast<ModelArchitectures::FluxModel*>(ctx.currentModel.get());
    
    if (!flux_model) {
        lua_pushnil(L);
        lua_pushstring(L, "Le modèle actuel n'est pas un FluxModel");
        return 2;
    }
    
    // Arguments: noisy_latent (table), text_embedding (table), timestep (int)
    luaL_checktype(L, 1, LUA_TTABLE);
    luaL_checktype(L, 2, LUA_TTABLE);
    int timestep = luaL_checkinteger(L, 3);
    
    // Extraire noisy_latent
    std::vector<float> noisy_latent;
    lua_pushnil(L);
    while (lua_next(L, 1) != 0) {
        noisy_latent.push_back(lua_tonumber(L, -1));
        lua_pop(L, 1);
    }
    
    // Extraire text_embedding
    std::vector<float> text_embedding;
    lua_pushnil(L);
    while (lua_next(L, 2) != 0) {
        text_embedding.push_back(lua_tonumber(L, -1));
        lua_pop(L, 1);
    }
    
    try {
        std::vector<float> predicted_noise = flux_model->predictNoise(
            noisy_latent, text_embedding, timestep);
        
        // Retourner le bruit prédit comme table Lua
        lua_newtable(L);
        for (size_t i = 0; i < predicted_noise.size(); ++i) {
            lua_pushnumber(L, predicted_noise[i]);
            lua_rawseti(L, -2, i + 1);
        }
        
        return 1;
    } catch (const std::exception& e) {
        lua_pushnil(L);
        lua_pushstring(L, e.what());
        return 2;
    }
}

int LuaScripting::lua_fluxComputeDiffusionLoss(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.currentModel) {
        lua_pushnumber(L, 0.0);
        lua_pushstring(L, "Aucun modèle créé");
        return 2;
    }
    
    ModelArchitectures::FluxModel* flux_model = 
        dynamic_cast<ModelArchitectures::FluxModel*>(ctx.currentModel.get());
    
    if (!flux_model) {
        lua_pushnumber(L, 0.0);
        lua_pushstring(L, "Le modèle actuel n'est pas un FluxModel");
        return 2;
    }
    
    // Arguments: image (table), tokens (table)
    luaL_checktype(L, 1, LUA_TTABLE);
    luaL_checktype(L, 2, LUA_TTABLE);
    
    // Extraire image
    std::vector<float> image;
    lua_pushnil(L);
    while (lua_next(L, 1) != 0) {
        image.push_back(lua_tonumber(L, -1));
        lua_pop(L, 1);
    }
    
    // Extraire tokens
    std::vector<int> tokens;
    lua_pushnil(L);
    while (lua_next(L, 2) != 0) {
        tokens.push_back(lua_tointeger(L, -1));
        lua_pop(L, 1);
    }
    
    try {
        float loss = flux_model->computeDiffusionLoss(image, tokens);
        lua_pushnumber(L, loss);
        return 1;
    } catch (const std::exception& e) {
        lua_pushnumber(L, 0.0);
        lua_pushstring(L, e.what());
        return 2;
    }
}

int LuaScripting::lua_fluxModelNew(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    // Argument optionnel: configuration (table)
    ModelArchitectures::FluxConfig config;
    
    if (lua_istable(L, 1)) {
        // Lire la configuration depuis la table Lua
        lua_getfield(L, 1, "image_resolution");
        if (lua_isnumber(L, -1)) config.image_resolution = lua_tointeger(L, -1);
        lua_pop(L, 1);
        
        lua_getfield(L, 1, "latent_channels");
        if (lua_isnumber(L, -1)) config.latent_channels = lua_tointeger(L, -1);
        lua_pop(L, 1);
        
        lua_getfield(L, 1, "latent_resolution");
        if (lua_isnumber(L, -1)) config.latent_resolution = lua_tointeger(L, -1);
        lua_pop(L, 1);
        
        lua_getfield(L, 1, "vae_base_channels");
        if (lua_isnumber(L, -1)) config.vae_base_channels = lua_tointeger(L, -1);
        lua_pop(L, 1);
        
        lua_getfield(L, 1, "num_res_blocks");
        if (lua_isnumber(L, -1)) config.vae_num_res_blocks = lua_tointeger(L, -1);
        lua_pop(L, 1);
        
        lua_getfield(L, 1, "vocab_size");
        if (lua_isnumber(L, -1)) config.vocab_size = lua_tointeger(L, -1);
        lua_pop(L, 1);
        
        lua_getfield(L, 1, "text_max_length");
        if (lua_isnumber(L, -1)) config.text_max_length = lua_tointeger(L, -1);
        lua_pop(L, 1);
        
        lua_getfield(L, 1, "text_embed_dim");
        if (lua_isnumber(L, -1)) config.text_embed_dim = lua_tointeger(L, -1);
        lua_pop(L, 1);
        
        lua_getfield(L, 1, "transformer_dim");
        if (lua_isnumber(L, -1)) config.transformer_dim = lua_tointeger(L, -1);
        lua_pop(L, 1);
        
        lua_getfield(L, 1, "num_transformer_blocks");
        if (lua_isnumber(L, -1)) config.num_transformer_blocks = lua_tointeger(L, -1);
        lua_pop(L, 1);
        
        lua_getfield(L, 1, "num_attention_heads");
        if (lua_isnumber(L, -1)) config.num_attention_heads = lua_tointeger(L, -1);
        lua_pop(L, 1);
        
        lua_getfield(L, 1, "mlp_ratio");
        if (lua_isnumber(L, -1)) config.mlp_ratio = lua_tointeger(L, -1);
        lua_pop(L, 1);
        
        lua_getfield(L, 1, "timestep_embed_dim");
        if (lua_isnumber(L, -1)) config.timestep_embed_dim = lua_tointeger(L, -1);
        lua_pop(L, 1);
        
        lua_getfield(L, 1, "num_timesteps");
        if (lua_isnumber(L, -1)) config.num_timesteps = lua_tointeger(L, -1);
        lua_pop(L, 1);
        
        // vae_channel_mult (array)
        lua_getfield(L, 1, "vae_channel_mult");
        if (lua_istable(L, -1)) {
            config.vae_channel_mult.clear();
            lua_pushnil(L);
            while (lua_next(L, -2) != 0) {
                config.vae_channel_mult.push_back(lua_tointeger(L, -1));
                lua_pop(L, 1);
            }
        }
        lua_pop(L, 1);
    }
    
    try {
        // Créer un FluxModel natif
        auto flux_model = std::make_shared<ModelArchitectures::FluxModel>(config);
        flux_model->setName("FluxModel");
        flux_model->buildFluxArchitecture();
        
        // Stocker dans le contexte
        ctx.currentModel = flux_model;
        ctx.addLog("FluxModel natif créé et construit");
        
        lua_pushboolean(L, true);
        return 1;
    } catch (const std::exception& e) {
        lua_pushboolean(L, false);
        lua_pushstring(L, e.what());
        return 2;
    }
}

// ============================================================================
// Layer Operations API (stubs - implémentation complète optionnelle)
// ============================================================================

int LuaScripting::lua_computeConv2D(lua_State* L) {
    lua_pushboolean(L, false);
    lua_pushstring(L, "Non implémenté - utilisez model.forward() à la place");
    return 2;
}

int LuaScripting::lua_computeLinear(lua_State* L) {
    lua_pushboolean(L, false);
    lua_pushstring(L, "Non implémenté - utilisez model.forward() à la place");
    return 2;
}

int LuaScripting::lua_computeMaxPool2D(lua_State* L) {
    lua_pushboolean(L, false);
    lua_pushstring(L, "Non implémenté - utilisez model.forward() à la place");
    return 2;
}

int LuaScripting::lua_computeAvgPool2D(lua_State* L) {
    lua_pushboolean(L, false);
    lua_pushstring(L, "Non implémenté - utilisez model.forward() à la place");
    return 2;
}

int LuaScripting::lua_computeActivation(lua_State* L) {
    lua_pushboolean(L, false);
    lua_pushstring(L, "Non implémenté - utilisez model.forward() à la place");
    return 2;
}

int LuaScripting::lua_computeBatchNorm(lua_State* L) {
    lua_pushboolean(L, false);
    lua_pushstring(L, "Non implémenté - utilisez model.forward() à la place");
    return 2;
}

int LuaScripting::lua_computeLayerNorm(lua_State* L) {
    lua_pushboolean(L, false);
    lua_pushstring(L, "Non implémenté - utilisez model.forward() à la place");
    return 2;
}

int LuaScripting::lua_computeAttention(lua_State* L) {
    lua_pushboolean(L, false);
    lua_pushstring(L, "Non implémenté - utilisez model.forward() à la place");
    return 2;
}

// ============================================================================
// Tokenizer API étendue
// ============================================================================

int LuaScripting::lua_getVocabSize(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.currentTokenizer) {
        lua_pushinteger(L, 0);
        return 1;
    }
    
    lua_pushinteger(L, ctx.currentTokenizer->getVocabSize());
    return 1;
}

int LuaScripting::lua_saveTokenizer(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.currentTokenizer) {
        lua_pushboolean(L, false);
        lua_pushstring(L, "Aucun tokenizer créé");
        return 2;
    }
    
    const char* filepath = luaL_checkstring(L, 1);
    
    try {
        json j = ctx.currentTokenizer->to_json();
        std::ofstream f(filepath);
        f << j.dump(2);
        
        ctx.addLog("Tokenizer sauvegardé: " + std::string(filepath));
        lua_pushboolean(L, true);
        return 1;
    } catch (const std::exception& e) {
        lua_pushboolean(L, false);
        lua_pushstring(L, e.what());
        return 2;
    }
}

int LuaScripting::lua_loadTokenizer(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    const char* filepath = luaL_checkstring(L, 1);
    
    try {
        std::ifstream f(filepath);
        json j;
        f >> j;
        
        ctx.currentTokenizer = std::make_shared<Tokenizer>();
        ctx.currentTokenizer->from_json(j);
        
        ctx.addLog("Tokenizer chargé: " + std::string(filepath));
        lua_pushboolean(L, true);
        return 1;
    } catch (const std::exception& e) {
        lua_pushboolean(L, false);
        lua_pushstring(L, e.what());
        return 2;
    }
}
// Extension des méthodes Tokenizer pour l'API Lua
// À ajouter à la fin de LuaScripting.cpp avant le dernier }

// ============================================================================
// Tokenizer API - Méthodes étendues
// ============================================================================

int LuaScripting::lua_addToken(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.currentTokenizer) {
        lua_pushinteger(L, -1);
        lua_pushstring(L, "Aucun tokenizer créé");
        return 2;
    }
    
    const char* token = luaL_checkstring(L, 1);
    int id = ctx.currentTokenizer->addToken(token);
    
    lua_pushinteger(L, id);
    return 1;
}

int LuaScripting::lua_ensureVocabFromText(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.currentTokenizer) {
        lua_pushboolean(L, false);
        lua_pushstring(L, "Aucun tokenizer créé");
        return 2;
    }
    
    const char* text = luaL_checkstring(L, 1);
    ctx.currentTokenizer->ensureVocabFromText(text);
    
    lua_pushboolean(L, true);
    return 1;
}

int LuaScripting::lua_tokenizeEnsure(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.currentTokenizer) {
        lua_pushnil(L);
        lua_pushstring(L, "Aucun tokenizer créé");
        return 2;
    }
    
    const char* text = luaL_checkstring(L, 1);
    auto tokens = ctx.currentTokenizer->tokenizeEnsure(text);
    
    lua_newtable(L);
    for (size_t i = 0; i < tokens.size(); ++i) {
        lua_pushinteger(L, tokens[i]);
        lua_rawseti(L, -2, i + 1);
    }
    
    return 1;
}

int LuaScripting::lua_getPadId(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.currentTokenizer) {
        lua_pushinteger(L, 0);
        return 1;
    }
    
    lua_pushinteger(L, ctx.currentTokenizer->getPadId());
    return 1;
}

int LuaScripting::lua_getUnkId(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.currentTokenizer) {
        lua_pushinteger(L, 1);
        return 1;
    }
    
    lua_pushinteger(L, ctx.currentTokenizer->getUnkId());
    return 1;
}

int LuaScripting::lua_getSeqId(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.currentTokenizer) {
        lua_pushinteger(L, 2);
        return 1;
    }
    
    lua_pushinteger(L, ctx.currentTokenizer->getSeqId());
    return 1;
}

int LuaScripting::lua_getModId(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.currentTokenizer) {
        lua_pushinteger(L, 3);
        return 1;
    }
    
    lua_pushinteger(L, ctx.currentTokenizer->getModId());
    return 1;
}

int LuaScripting::lua_getMagId(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.currentTokenizer) {
        lua_pushinteger(L, 4);
        return 1;
    }
    
    lua_pushinteger(L, ctx.currentTokenizer->getMagId());
    return 1;
}

int LuaScripting::lua_getTokenById(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.currentTokenizer) {
        lua_pushstring(L, "");
        return 1;
    }
    
    int id = luaL_checkinteger(L, 1);
    std::string token = ctx.currentTokenizer->getTokenById(id);
    
    lua_pushstring(L, token.c_str());
    return 1;
}

int LuaScripting::lua_learnBPEFromCorpus(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.currentTokenizer) {
        lua_pushboolean(L, false);
        lua_pushstring(L, "Aucun tokenizer créé");
        return 2;
    }
    
    // Lire table de textes
    luaL_checktype(L, 1, LUA_TTABLE);
    int num_merges = luaL_optinteger(L, 2, 1000);
    
    std::vector<std::string> corpus;
    lua_pushnil(L);
    while (lua_next(L, 1) != 0) {
        corpus.push_back(lua_tostring(L, -1));
        lua_pop(L, 1);
    }
    
    ctx.currentTokenizer->learnBPEFromCorpus(corpus, num_merges);
    
    lua_pushboolean(L, true);
    return 1;
}

int LuaScripting::lua_tokenizeBPE(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.currentTokenizer) {
        lua_pushnil(L);
        lua_pushstring(L, "Aucun tokenizer créé");
        return 2;
    }
    
    const char* text = luaL_checkstring(L, 1);
    auto tokens = ctx.currentTokenizer->tokenizeBPE(text);
    
    lua_newtable(L);
    for (size_t i = 0; i < tokens.size(); ++i) {
        lua_pushinteger(L, tokens[i]);
        lua_rawseti(L, -2, i + 1);
    }
    
    return 1;
}

int LuaScripting::lua_setMaxSequenceLength(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.currentTokenizer) {
        lua_pushboolean(L, false);
        return 1;
    }
    
    int max_len = luaL_checkinteger(L, 1);
    ctx.currentTokenizer->setMaxSequenceLength(max_len);
    
    lua_pushboolean(L, true);
    return 1;
}

int LuaScripting::lua_padSequence(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.currentTokenizer) {
        lua_pushnil(L);
        lua_pushstring(L, "Aucun tokenizer créé");
        return 2;
    }
    
    // Lire table de tokens
    luaL_checktype(L, 1, LUA_TTABLE);
    int target_len = luaL_optinteger(L, 2, -1);
    
    std::vector<int> tokens;
    lua_pushnil(L);
    while (lua_next(L, 1) != 0) {
        tokens.push_back(lua_tointeger(L, -1));
        lua_pop(L, 1);
    }
    
    auto padded = ctx.currentTokenizer->padSequence(tokens, target_len);
    
    lua_newtable(L);
    for (size_t i = 0; i < padded.size(); ++i) {
        lua_pushinteger(L, padded[i]);
        lua_rawseti(L, -2, i + 1);
    }
    
    return 1;
}

int LuaScripting::lua_batchTokenize(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.currentTokenizer) {
        lua_pushnil(L);
        lua_pushstring(L, "Aucun tokenizer créé");
        return 2;
    }
    
    // Lire table de textes
    luaL_checktype(L, 1, LUA_TTABLE);
    int max_len = luaL_optinteger(L, 2, 512);
    
    std::vector<std::string> texts;
    lua_pushnil(L);
    while (lua_next(L, 1) != 0) {
        texts.push_back(lua_tostring(L, -1));
        lua_pop(L, 1);
    }
    
    auto batch = ctx.currentTokenizer->batchTokenize(texts, max_len);
    
    // Retourner table de tables
    lua_newtable(L);
    for (size_t i = 0; i < batch.size(); ++i) {
        lua_newtable(L);
        for (size_t j = 0; j < batch[i].size(); ++j) {
            lua_pushinteger(L, batch[i][j]);
            lua_rawseti(L, -2, j + 1);
        }
        lua_rawseti(L, -2, i + 1);
    }
    
    return 1;
}

int LuaScripting::lua_printVocabStats(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.currentTokenizer) {
        lua_pushboolean(L, false);
        return 1;
    }
    
    ctx.currentTokenizer->printVocabStats();
    
    lua_pushboolean(L, true);
    return 1;
}

int LuaScripting::lua_getTokenFrequencies(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.currentTokenizer) {
        lua_pushnil(L);
        lua_pushstring(L, "Aucun tokenizer créé");
        return 2;
    }
    
    const char* text = luaL_checkstring(L, 1);
    auto freqs = ctx.currentTokenizer->getTokenFrequencies(text);
    
    // Retourner table Lua
    lua_newtable(L);
    for (const auto& pair : freqs) {
        lua_pushstring(L, pair.first.c_str());
        lua_pushinteger(L, pair.second);
        lua_settable(L, -3);
    }
    
    return 1;
}

int LuaScripting::lua_analyzeText(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.currentTokenizer) {
        lua_pushnil(L);
        lua_pushstring(L, "Aucun tokenizer créé");
        return 2;
    }
    
    const char* text = luaL_checkstring(L, 1);
    auto analysis = ctx.currentTokenizer->analyzeText(text);
    
    // Retourner table Lua avec analyse
    lua_newtable(L);
    
    // entities
    lua_newtable(L);
    for (size_t i = 0; i < analysis.entities.size(); ++i) {
        lua_pushstring(L, analysis.entities[i].c_str());
        lua_rawseti(L, -2, i + 1);
    }
    lua_setfield(L, -2, "entities");
    
    // modifiers
    lua_newtable(L);
    for (size_t i = 0; i < analysis.modifiers.size(); ++i) {
        lua_pushstring(L, analysis.modifiers[i].c_str());
        lua_rawseti(L, -2, i + 1);
    }
    lua_setfield(L, -2, "modifiers");
    
    // actions
    lua_newtable(L);
    for (size_t i = 0; i < analysis.actions.size(); ++i) {
        lua_pushstring(L, analysis.actions[i].c_str());
        lua_rawseti(L, -2, i + 1);
    }
    lua_setfield(L, -2, "actions");
    
    // main_subject
    lua_pushstring(L, analysis.mainSubject.c_str());
    lua_setfield(L, -2, "main_subject");
    
    // context
    lua_pushstring(L, analysis.context.c_str());
    lua_setfield(L, -2, "context");
    
    // complexity
    lua_pushinteger(L, analysis.complexity);
    lua_setfield(L, -2, "complexity");
    
    return 1;
}

// ============================================================================
// Memory Manager API
// ============================================================================

int LuaScripting::lua_memoryConfig(lua_State* L) {
    auto& mgr = AdvancedRAMManager::instance();
    
    // Argument: table de configuration
    if (!lua_istable(L, 1)) {
        lua_pushboolean(L, false);
        lua_pushstring(L, "Argument 1 doit être une table de configuration");
        return 2;
    }
    
    AdvancedRAMManager::Config config;
    
    // max_ram_gb (en Go)
    lua_getfield(L, 1, "max_ram_gb");
    if (lua_isnumber(L, -1)) {
        double gb = lua_tonumber(L, -1);
        config.max_ram_bytes = static_cast<size_t>(gb * 1024.0 * 1024.0 * 1024.0);
    }
    lua_pop(L, 1);
    
    // enable_compression
    lua_getfield(L, 1, "enable_compression");
    if (lua_isboolean(L, -1)) {
        config.enable_compression = lua_toboolean(L, -1);
    }
    lua_pop(L, 1);
    
    // enable_async_loading
    lua_getfield(L, 1, "enable_async_loading");
    if (lua_isboolean(L, -1)) {
        config.enable_async_loading = lua_toboolean(L, -1);
    }
    lua_pop(L, 1);
    
    // enable_prediction
    lua_getfield(L, 1, "enable_prediction");
    if (lua_isboolean(L, -1)) {
        config.enable_prediction = lua_toboolean(L, -1);
    }
    lua_pop(L, 1);
    
    // enable_statistics
    lua_getfield(L, 1, "enable_statistics");
    if (lua_isboolean(L, -1)) {
        config.enable_statistics = lua_toboolean(L, -1);
    }
    lua_pop(L, 1);
    
    // preload_queue_size
    lua_getfield(L, 1, "preload_queue_size");
    if (lua_isnumber(L, -1)) {
        config.preload_queue_size = static_cast<size_t>(lua_tonumber(L, -1));
    }
    lua_pop(L, 1);
    
    // worker_threads
    lua_getfield(L, 1, "worker_threads");
    if (lua_isnumber(L, -1)) {
        config.worker_threads = static_cast<size_t>(lua_tonumber(L, -1));
    }
    lua_pop(L, 1);
    
    // Appliquer la configuration
    mgr.configure(config);
    
    std::cout << "🔧 Gestionnaire de mémoire configuré:" << std::endl;
    std::cout << "   - Limite RAM: " << (config.max_ram_bytes / 1024 / 1024 / 1024) << " GB" << std::endl;
    std::cout << "   - Compression: " << (config.enable_compression ? "activée" : "désactivée") << std::endl;
    std::cout << "   - Chargement async: " << (config.enable_async_loading ? "activé" : "désactivé") << std::endl;
    std::cout << "   - Prédiction: " << (config.enable_prediction ? "activée" : "désactivée") << std::endl;
    std::cout << "   - Worker threads: " << config.worker_threads << std::endl;
    
    lua_pushboolean(L, true);
    return 1;
}

int LuaScripting::lua_memoryGetStats(lua_State* L) {
    auto& mgr = AdvancedRAMManager::instance();
    
    // Retourner une table avec les statistiques
    lua_newtable(L);
    
    // current_mb
    size_t current = mgr.getCurrentRAM();
    lua_pushnumber(L, static_cast<double>(current) / (1024.0 * 1024.0));
    lua_setfield(L, -2, "current_mb");
    
    // peak_mb
    size_t peak = mgr.getPeakRAM();
    lua_pushnumber(L, static_cast<double>(peak) / (1024.0 * 1024.0));
    lua_setfield(L, -2, "peak_mb");
    
    // usage_percent
    float usage = mgr.getUsagePercent();
    lua_pushnumber(L, static_cast<double>(usage));
    lua_setfield(L, -2, "usage_percent");
    
    return 1;
}

int LuaScripting::lua_memoryPrintStats(lua_State* L) {
    auto& mgr = AdvancedRAMManager::instance();
    mgr.printDetailedStats();
    
    lua_pushboolean(L, true);
    return 1;
}

int LuaScripting::lua_memoryClear(lua_State* L) {
    auto& mgr = AdvancedRAMManager::instance();
    mgr.clear();
    
    std::cout << "🧹 Mémoire effacée" << std::endl;
    
    lua_pushboolean(L, true);
    return 1;
}

int LuaScripting::lua_memoryGetUsage(lua_State* L) {
    auto& mgr = AdvancedRAMManager::instance();
    
    // Retourner current_mb, peak_mb, usage_percent
    size_t current = mgr.getCurrentRAM();
    size_t peak = mgr.getPeakRAM();
    float usage = mgr.getUsagePercent();
    
    lua_pushnumber(L, static_cast<double>(current) / (1024.0 * 1024.0));
    lua_pushnumber(L, static_cast<double>(peak) / (1024.0 * 1024.0));
    lua_pushnumber(L, static_cast<double>(usage));
    
    return 3;
}

int LuaScripting::lua_memorySetLimit(lua_State* L) {
    auto& mgr = AdvancedRAMManager::instance();
    
    // Argument: limite en GB
    double gb = luaL_checknumber(L, 1);
    
    if (gb <= 0) {
        lua_pushboolean(L, false);
        lua_pushstring(L, "Limite doit être > 0");
        return 2;
    }
    
    // Créer une config avec la nouvelle limite
    AdvancedRAMManager::Config config;
    config.max_ram_bytes = static_cast<size_t>(gb * 1024.0 * 1024.0 * 1024.0);
    config.enable_compression = true;
    config.enable_async_loading = false;
    config.enable_prediction = false;
    config.enable_statistics = true;
    config.worker_threads = 2;
    
    mgr.configure(config);
    
    std::cout << "💾 Limite RAM définie à " << gb << " GB" << std::endl;
    
    lua_pushboolean(L, true);
    return 1;
}

// ============================================================================
// Memory Guard API (Strict Enforcement)
// ============================================================================

int LuaScripting::lua_guardSetLimit(lua_State* L) {
    auto& guard = MemoryGuard::instance();
    
    // Argument: peut être en GB (float) ou en bytes (très grand nombre)
    double value = luaL_checknumber(L, 1);
    
    if (value <= 0) {
        lua_pushboolean(L, false);
        lua_pushstring(L, "Limite doit être > 0");
        return 2;
    }
    
    size_t bytes;
    // Si la valeur est petite (<= 1000), on assume que c'est en GB
    if (value <= 1000.0) {
        bytes = static_cast<size_t>(value * 1024.0 * 1024.0 * 1024.0);
    } else {
        // Sinon c'est directement en bytes
        bytes = static_cast<size_t>(value);
    }
    
    guard.setLimit(bytes);
    
    lua_pushboolean(L, true);
    return 1;
}

int LuaScripting::lua_guardGetStats(lua_State* L) {
    auto& guard = MemoryGuard::instance();
    
    // Retourner une table avec les statistiques
    lua_newtable(L);
    
    // current_mb
    lua_pushnumber(L, static_cast<double>(guard.getCurrentBytes()) / (1024.0 * 1024.0));
    lua_setfield(L, -2, "current_mb");
    
    // peak_mb
    lua_pushnumber(L, static_cast<double>(guard.getPeakBytes()) / (1024.0 * 1024.0));
    lua_setfield(L, -2, "peak_mb");
    
    // limit_mb
    lua_pushnumber(L, static_cast<double>(guard.getLimit()) / (1024.0 * 1024.0));
    lua_setfield(L, -2, "limit_mb");
    
    // usage_percent
    lua_pushnumber(L, static_cast<double>(guard.getUsagePercent()));
    lua_setfield(L, -2, "usage_percent");
    
    return 1;
}

int LuaScripting::lua_guardPrintStats(lua_State* L) {
    auto& guard = MemoryGuard::instance();
    guard.printStats();
    
    lua_pushboolean(L, true);
    return 1;
}

int LuaScripting::lua_guardReset(lua_State* L) {
    auto& guard = MemoryGuard::instance();
    guard.reset();
    
    std::cout << "🔄 MemoryGuard réinitialisé" << std::endl;
    
    lua_pushboolean(L, true);
    return 1;
}

int LuaScripting::lua_memoryguardGetCurrentUsage(lua_State* L) {
    auto& guard = MemoryGuard::instance();
    lua_pushnumber(L, static_cast<double>(guard.getCurrentBytes()));
    return 1;
}

int LuaScripting::lua_memoryguardGetPeakUsage(lua_State* L) {
    auto& guard = MemoryGuard::instance();
    lua_pushnumber(L, static_cast<double>(guard.getPeakBytes()));
    return 1;
}

int LuaScripting::lua_memoryguardGetLimit(lua_State* L) {
    auto& guard = MemoryGuard::instance();
    lua_pushnumber(L, static_cast<double>(guard.getLimit()));
    return 1;
}

// ============================================================================
// Dynamic Tensor Allocator API
// ============================================================================

int LuaScripting::lua_allocatorConfigure(lua_State* L) {
    auto& allocator = DynamicTensorAllocator::instance();
    
    // Argument: table de configuration
    if (!lua_istable(L, 1)) {
        lua_pushboolean(L, false);
        lua_pushstring(L, "Argument 1 doit être une table de configuration");
        return 2;
    }
    
    // max_ram_gb
    lua_getfield(L, 1, "max_ram_gb");
    double max_ram_gb = lua_isnumber(L, -1) ? lua_tonumber(L, -1) : 10.0;
    lua_pop(L, 1);
    
    // enable_compression
    lua_getfield(L, 1, "enable_compression");
    bool enable_compression = lua_isboolean(L, -1) ? lua_toboolean(L, -1) : true;
    lua_pop(L, 1);
    
    // Configurer
    allocator.configure(static_cast<size_t>(max_ram_gb), enable_compression);
    
    std::cout << "✓ DynamicTensorAllocator configuré" << std::endl;
    std::cout << "   - Limite: " << max_ram_gb << " GB" << std::endl;
    std::cout << "   - Compression: " << (enable_compression ? "activée" : "désactivée") << std::endl;
    
    lua_pushboolean(L, true);
    return 1;
}

int LuaScripting::lua_allocatorPrintStats(lua_State* L) {
    auto& allocator = DynamicTensorAllocator::instance();
    allocator.printStats();
    
    lua_pushboolean(L, true);
    return 1;
}

int LuaScripting::lua_allocatorGetStats(lua_State* L) {
    auto& allocator = DynamicTensorAllocator::instance();
    
    // Retourner une table avec les statistiques
    lua_newtable(L);
    
    // tensor_count
    lua_pushnumber(L, static_cast<double>(allocator.getTensorCount()));
    lua_setfield(L, -2, "tensor_count");
    
    // loaded_count
    lua_pushnumber(L, static_cast<double>(allocator.getLoadedCount()));
    lua_setfield(L, -2, "loaded_count");
    
    return 1;
}

// ============================================================================
// HtopDisplay API
// ============================================================================

int LuaScripting::lua_htopCreate(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    try {
        if (!ctx.asyncMonitor) {
            ctx.asyncMonitor = std::make_shared<AsyncMonitor>();
        }
        
        // Démarrer avec htop activé
        bool enable_viz = lua_toboolean(L, 1); // Optionnel: activer viz aussi
        ctx.asyncMonitor->start(true, enable_viz);
        
        ctx.addLog("AsyncMonitor démarré (htop enabled)");
        lua_pushboolean(L, true);
    } catch (const std::exception& e) {
        lua_pushboolean(L, false);
        lua_pushstring(L, e.what());
        return 2;
    }
    
    return 1;
}

int LuaScripting::lua_htopUpdate(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.asyncMonitor) {
        lua_pushboolean(L, false);
        lua_pushstring(L, "AsyncMonitor non créé");
        return 2;
    }
    
    // Arguments: epoch, total_epochs, batch, total_batches, loss, avg_loss, lr,
    //            batch_time_ms, memory_mb, memory_freed, bps, params, timestep,
    //            kl, wass, ent, mom, spat, temp, mse
    AsyncMonitor::Metrics metrics;
    metrics.epoch = luaL_checkinteger(L, 1);
    metrics.total_epochs = luaL_checkinteger(L, 2);
    metrics.batch = luaL_checkinteger(L, 3);
    metrics.total_batches = luaL_checkinteger(L, 4);
    metrics.loss = static_cast<float>(luaL_checknumber(L, 5));
    metrics.avg_loss = static_cast<float>(luaL_checknumber(L, 6));
    metrics.lr = static_cast<float>(luaL_checknumber(L, 7));
    metrics.batch_time_ms = luaL_optinteger(L, 8, 0);
    metrics.memory_mb = static_cast<size_t>(luaL_optinteger(L, 9, 0));
    metrics.memory_freed = static_cast<size_t>(luaL_optinteger(L, 10, 0));
    metrics.bps = static_cast<float>(luaL_optnumber(L, 11, 0.0));
    metrics.params = static_cast<size_t>(luaL_optinteger(L, 12, 0));
    metrics.timestep = static_cast<float>(luaL_optnumber(L, 13, 0.0));
    metrics.kl = static_cast<float>(luaL_optnumber(L, 14, 0.0));
    metrics.wass = static_cast<float>(luaL_optnumber(L, 15, 0.0));
    metrics.ent = static_cast<float>(luaL_optnumber(L, 16, 0.0));
    metrics.mom = static_cast<float>(luaL_optnumber(L, 17, 0.0));
    metrics.spat = static_cast<float>(luaL_optnumber(L, 18, 0.0));
    metrics.temp = static_cast<float>(luaL_optnumber(L, 19, 0.0));
    metrics.mse = static_cast<float>(luaL_optnumber(L, 20, 0.0));
    
    ctx.asyncMonitor->updateMetrics(metrics);
    
    lua_pushboolean(L, true);
    return 1;
}

int LuaScripting::lua_htopRender(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    // Render est automatique dans AsyncMonitor, cette fonction est maintenant no-op
    lua_pushboolean(L, true);
    return 1;
}

int LuaScripting::lua_htopClear(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.asyncMonitor) {
        lua_pushboolean(L, false);
        lua_pushstring(L, "AsyncMonitor non créé");
        return 2;
    }
    
    auto htop = ctx.asyncMonitor->getHtop();
    if (htop) {
        htop->clearScreen();
    }
    
    lua_pushboolean(L, true);
    return 1;
}

int LuaScripting::lua_htopEnable(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.asyncMonitor) {
        lua_pushboolean(L, false);
        lua_pushstring(L, "AsyncMonitor non créé");
        return 2;
    }
    
    // Note: HtopDisplay n'a pas de setEnabled(), on peut juste démarrer/arrêter le monitor
    bool enabled = lua_toboolean(L, 1);
    if (!enabled) {
        ctx.asyncMonitor->stop();
    }
    
    lua_pushboolean(L, true);
    return 1;
}

// ============================================================================
// Visualizer API
// ============================================================================

int LuaScripting::lua_vizCreate(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    // Argument: table de configuration
    json config;
    if (lua_istable(L, 1)) {
        config = luaTableToJson(L, 1);
    }
    
    try {
        if (!ctx.asyncMonitor) {
            ctx.asyncMonitor = std::make_shared<AsyncMonitor>();
        }
        
        // Démarrer avec viz activé (et htop désactivé si pas déjà démarré)
        ctx.asyncMonitor->start(false, true, config);
        
        ctx.addLog("AsyncMonitor démarré (visualizer enabled)");
        lua_pushboolean(L, true);
    } catch (const std::exception& e) {
        lua_pushboolean(L, false);
        lua_pushstring(L, e.what());
        return 2;
    }
    
    return 1;
}

int LuaScripting::lua_vizInitialize(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.asyncMonitor) {
        lua_pushboolean(L, false);
        lua_pushstring(L, "AsyncMonitor non créé");
        return 2;
    }
    
    auto viz = ctx.asyncMonitor->getViz();
    bool success = viz && viz->isOpen();
    lua_pushboolean(L, success);
    return 1;
}

int LuaScripting::lua_vizIsOpen(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.asyncMonitor) {
        lua_pushboolean(L, false);
        return 1;
    }
    
    auto viz = ctx.asyncMonitor->getViz();
    lua_pushboolean(L, viz && viz->isOpen());
    return 1;
}

int LuaScripting::lua_vizProcessEvents(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    // Process events est automatique dans AsyncMonitor, cette fonction est maintenant no-op
    lua_pushboolean(L, true);
    return 1;
}

int LuaScripting::lua_vizUpdate(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    // Update est automatique dans AsyncMonitor, cette fonction est maintenant no-op
    lua_pushboolean(L, true);
    return 1;
}

int LuaScripting::lua_vizAddImage(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.asyncMonitor) {
        lua_pushboolean(L, false);
        lua_pushstring(L, "AsyncMonitor non créé");
        return 2;
    }
    
    // Arguments: table de pixels (uint8), prompt (string)
    if (!lua_istable(L, 1)) {
        lua_pushboolean(L, false);
        lua_pushstring(L, "Argument 1 doit être une table de pixels");
        return 2;
    }
    
    const char* prompt = luaL_optstring(L, 2, "");
    
    // Lire la table de pixels
    std::vector<uint8_t> pixels;
    lua_pushnil(L);
    while (lua_next(L, 1) != 0) {
        if (lua_isnumber(L, -1)) {
            pixels.push_back(static_cast<uint8_t>(lua_tointeger(L, -1)));
        }
        lua_pop(L, 1);
    }
    
    ctx.asyncMonitor->addImage(pixels, prompt);
    lua_pushboolean(L, true);
    return 1;
}

int LuaScripting::lua_vizUpdateMetrics(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.asyncMonitor) {
        lua_pushboolean(L, false);
        lua_pushstring(L, "AsyncMonitor non créé");
        return 2;
    }
    
    // Arguments: epoch, batch, loss, lr, mse, kl, wass, ent, mom, spat, temp
    AsyncMonitor::Metrics metrics;
    metrics.epoch = luaL_checkinteger(L, 1);
    metrics.batch = luaL_checkinteger(L, 2);
    metrics.loss = static_cast<float>(luaL_checknumber(L, 3));
    metrics.lr = static_cast<float>(luaL_checknumber(L, 4));
    metrics.mse = static_cast<float>(luaL_optnumber(L, 5, 0.0));
    metrics.kl = static_cast<float>(luaL_optnumber(L, 6, 0.0));
    metrics.wass = static_cast<float>(luaL_optnumber(L, 7, 0.0));
    metrics.ent = static_cast<float>(luaL_optnumber(L, 8, 0.0));
    metrics.mom = static_cast<float>(luaL_optnumber(L, 9, 0.0));
    metrics.spat = static_cast<float>(luaL_optnumber(L, 10, 0.0));
    metrics.temp = static_cast<float>(luaL_optnumber(L, 11, 0.0));
    
    ctx.asyncMonitor->updateMetrics(metrics);
    
    lua_pushboolean(L, true);
    return 1;
}

int LuaScripting::lua_vizAddLossPoint(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    // AddLossPoint est automatique dans AsyncMonitor via updateMetrics
    lua_pushboolean(L, true);
    return 1;
}

int LuaScripting::lua_vizClear(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.asyncMonitor) {
        lua_pushboolean(L, false);
        return 1;
    }
    
    auto viz = ctx.asyncMonitor->getViz();
    if (viz) {
        viz->clearImages();
    }
    
    lua_pushboolean(L, true);
    return 1;
}

int LuaScripting::lua_vizSetEnabled(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.asyncMonitor) {
        lua_pushboolean(L, false);
        return 1;
    }
    
    // Note: Visualizer n'a pas de setEnabled(), on peut juste démarrer/arrêter le monitor
    bool enabled = lua_toboolean(L, 1);
    if (!enabled) {
        ctx.asyncMonitor->stop();
    }
    
    lua_pushboolean(L, true);
    return 1;
}

int LuaScripting::lua_vizSaveLossHistory(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.asyncMonitor) {
        lua_pushboolean(L, false);
        lua_pushstring(L, "AsyncMonitor non créé");
        return 2;
    }
    
    const char* filepath = luaL_checkstring(L, 1);
    auto viz = ctx.asyncMonitor->getViz();
    if (viz) {
        viz->saveLossHistory(filepath);
    }
    
    lua_pushboolean(L, true);
    return 1;
}

int LuaScripting::lua_extractKeywords(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.currentTokenizer) {
        lua_pushnil(L);
        lua_pushstring(L, "Aucun tokenizer créé");
        return 2;
    }
    
    const char* text = luaL_checkstring(L, 1);
    int topN = luaL_optinteger(L, 2, 5);
    
    auto keywords = ctx.currentTokenizer->extractKeywords(text, topN);
    
    lua_newtable(L);
    for (size_t i = 0; i < keywords.size(); ++i) {
        lua_pushstring(L, keywords[i].c_str());
        lua_rawseti(L, -2, i + 1);
    }
    
    return 1;
}
