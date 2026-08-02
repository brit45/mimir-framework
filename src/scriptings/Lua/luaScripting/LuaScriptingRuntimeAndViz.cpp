#include "LuaScripting.hpp"
#include "Models/Registry/ModelArchitectures.hpp"
#include "Serialization/Serialization.hpp"
#include "Serialization/DebugJsonDump.hpp"
#include "DType.hpp"
#include "AdvancedRAMManager.hpp"
#include "MemoryGuard.hpp"
#include "DynamicTensorAllocator.hpp"
#include "AsyncMonitor.hpp"
#include "Models/Diffusion/PonyXLDDPMModel.hpp"
#include "Helpers.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <unordered_map>
#include <unordered_set>
#include <type_traits>
#include <utility>

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

// Renvoie la liste des layers du modèle courant.
// Retourne: table[] où chaque entrée est
//   { index, name, type, param_count, inputs={...}, output,
//     in_features, out_features, in_channels, out_channels,
//     kernel_size, stride, padding, seq_len, embed_dim, num_heads,
//     vocab_size, input_height, input_width }
int LuaScripting::lua_getModelLayers(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    if (!ctx.currentModel) {
        lua_newtable(L);
        return 1;
    }
    const auto& layers = ctx.currentModel->getLayers();
    lua_createtable(L, static_cast<int>(layers.size()), 0);
    for (size_t i = 0; i < layers.size(); ++i) {
        const Layer& la = layers[i];
        lua_newtable(L);

        lua_pushinteger(L, static_cast<lua_Integer>(i + 1));
        lua_setfield(L, -2, "index");
        lua_pushstring(L, la.name.c_str());
        lua_setfield(L, -2, "name");
        lua_pushstring(L, la.type.c_str());
        lua_setfield(L, -2, "type");
        lua_pushinteger(L, static_cast<lua_Integer>(la.getWeightsSize()));
        lua_setfield(L, -2, "param_count");
        lua_pushstring(L, la.output.c_str());
        lua_setfield(L, -2, "output");

        // inputs array
        lua_createtable(L, static_cast<int>(la.inputs.size()), 0);
        for (size_t j = 0; j < la.inputs.size(); ++j) {
            lua_pushstring(L, la.inputs[j].c_str());
            lua_rawseti(L, -2, static_cast<int>(j + 1));
        }
        lua_setfield(L, -2, "inputs");

        lua_pushinteger(L, la.in_features);   lua_setfield(L, -2, "in_features");
        lua_pushinteger(L, la.out_features);  lua_setfield(L, -2, "out_features");
        lua_pushinteger(L, la.in_channels);   lua_setfield(L, -2, "in_channels");
        lua_pushinteger(L, la.out_channels);  lua_setfield(L, -2, "out_channels");
        lua_pushinteger(L, la.kernel_size);   lua_setfield(L, -2, "kernel_size");
        lua_pushinteger(L, la.stride);        lua_setfield(L, -2, "stride");
        lua_pushinteger(L, la.padding);       lua_setfield(L, -2, "padding");
        lua_pushinteger(L, la.seq_len);       lua_setfield(L, -2, "seq_len");
        lua_pushinteger(L, la.embed_dim);     lua_setfield(L, -2, "embed_dim");
        lua_pushinteger(L, la.num_heads);     lua_setfield(L, -2, "num_heads");
        lua_pushinteger(L, la.vocab_size);    lua_setfield(L, -2, "vocab_size");
        lua_pushinteger(L, la.input_height);  lua_setfield(L, -2, "input_height");
        lua_pushinteger(L, la.input_width);   lua_setfield(L, -2, "input_width");
        lua_pushinteger(L, la.kernel_h);      lua_setfield(L, -2, "kernel_h");
        lua_pushinteger(L, la.kernel_w);      lua_setfield(L, -2, "kernel_w");
        lua_pushinteger(L, la.stride_h);      lua_setfield(L, -2, "stride_h");
        lua_pushinteger(L, la.stride_w);      lua_setfield(L, -2, "stride_w");
        lua_pushinteger(L, la.pad_h);         lua_setfield(L, -2, "pad_h");
        lua_pushinteger(L, la.pad_w);         lua_setfield(L, -2, "pad_w");
        lua_pushinteger(L, la.dilation);      lua_setfield(L, -2, "dilation");
        lua_pushinteger(L, la.groups);        lua_setfield(L, -2, "groups");
        lua_pushnumber(L, la.eps);            lua_setfield(L, -2, "eps");
        lua_pushinteger(L, la.num_groups);    lua_setfield(L, -2, "num_groups");
        lua_pushnumber(L, la.dropout_p);      lua_setfield(L, -2, "dropout_p");
        lua_pushinteger(L, la.axis);          lua_setfield(L, -2, "axis");
        lua_pushinteger(L, la.concat_axis);   lua_setfield(L, -2, "concat_axis");
        lua_pushinteger(L, la.split_axis);    lua_setfield(L, -2, "split_axis");
        lua_pushinteger(L, la.num_splits);    lua_setfield(L, -2, "num_splits");
        lua_pushnumber(L, la.scale_h);        lua_setfield(L, -2, "scale_h");
        lua_pushnumber(L, la.scale_w);        lua_setfield(L, -2, "scale_w");
        lua_pushinteger(L, la.out_h);         lua_setfield(L, -2, "out_h");
        lua_pushinteger(L, la.out_w);         lua_setfield(L, -2, "out_w");
        lua_pushinteger(L, la.head_dim);      lua_setfield(L, -2, "head_dim");
        lua_pushboolean(L, la.causal);        lua_setfield(L, -2, "causal");
        lua_pushboolean(L, la.use_bias);      lua_setfield(L, -2, "use_bias");
        lua_pushnumber(L, la.nms_iou_threshold);
        lua_setfield(L, -2, "nms_iou_threshold");
        lua_pushnumber(L, la.nms_score_threshold);
        lua_setfield(L, -2, "nms_score_threshold");
        lua_pushinteger(L, la.nms_max_detections);
        lua_setfield(L, -2, "nms_max_detections");
        lua_pushboolean(L, la.nms_class_agnostic);
        lua_setfield(L, -2, "nms_class_agnostic");

        auto push_int_array = [L](const std::vector<int>& values,
                                  const char* field) {
            lua_createtable(L, static_cast<int>(values.size()), 0);
            for (size_t j = 0; j < values.size(); ++j) {
                lua_pushinteger(L, values[j]);
                lua_rawseti(L, -2, static_cast<int>(j + 1));
            }
            lua_setfield(L, -2, field);
        };
        push_int_array(la.target_shape, "target_shape");
        push_int_array(la.permute_dims, "permute_dims");
        push_int_array(la.split_sizes, "split_sizes");

        lua_rawseti(L, -2, static_cast<int>(i + 1));
    }
    return 1;
}

int LuaScripting::lua_clearModelLayers(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    if (!ctx.currentModel) {
        lua_pushboolean(L, false);
        lua_pushstring(L, "Aucun modèle créé");
        return 2;
    }

    auto& layers = ctx.currentModel->getMutableLayers();
    const lua_Integer old_count = static_cast<lua_Integer>(layers.size());
    layers.clear();
    ctx.currentModel->clearVizTaps();

    lua_pushboolean(L, true);
    lua_pushinteger(L, old_count);
    return 2;
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
    const lua_Integer raw_params_count = luaL_checkinteger(L, 3);
    if (raw_params_count < 0) {
        lua_pushboolean(L, false);
        lua_pushstring(L, "params_count must be >= 0");
        return 2;
    }

    try {
        ctx.currentModel->push(
            name, type, static_cast<size_t>(raw_params_count));

        // Argument 4 optionnel: paramètres propres au nœud MPK. Cela évite
        // d'aplatir une configuration de graphe dans modelConfig.
        if (lua_istable(L, 4)) {
            Layer* layer = ctx.currentModel->getLayerByName(name);
            if (!layer) throw std::runtime_error("new layer not found");
            const json p = luaTableToJson(L, 4);
            if (!p.is_object()) {
                throw std::runtime_error("layer params must be a map");
            }

            auto set_int = [&p](const char* key, int& dst) {
                if (p.contains(key)) dst = p.at(key).get<int>();
            };
            auto set_float = [&p](const char* key, float& dst) {
                if (p.contains(key)) dst = p.at(key).get<float>();
            };
            auto set_bool = [&p](const char* key, bool& dst) {
                if (p.contains(key)) dst = p.at(key).get<bool>();
            };
            auto set_ints = [&p](const char* key, std::vector<int>& dst) {
                if (p.contains(key)) dst = p.at(key).get<std::vector<int>>();
            };

            set_int("in_features", layer->in_features);
            set_int("out_features", layer->out_features);
            set_int("in_channels", layer->in_channels);
            set_int("out_channels", layer->out_channels);
            set_int("input_height", layer->input_height);
            set_int("input_width", layer->input_width);
            set_int("output_height", layer->output_height);
            set_int("output_width", layer->output_width);
            set_int("kernel_size", layer->kernel_size);
            set_int("kernel_h", layer->kernel_h);
            set_int("kernel_w", layer->kernel_w);
            set_int("stride", layer->stride);
            set_int("stride_h", layer->stride_h);
            set_int("stride_w", layer->stride_w);
            set_int("padding", layer->padding);
            set_int("pad_h", layer->pad_h);
            set_int("pad_w", layer->pad_w);
            set_int("dilation", layer->dilation);
            set_int("dilation_h", layer->dilation_h);
            set_int("dilation_w", layer->dilation_w);
            set_int("groups", layer->groups);
            set_float("eps", layer->eps);
            set_int("num_groups", layer->num_groups);
            set_float("momentum", layer->momentum);
            set_bool("affine", layer->affine);
            set_bool("track_running_stats", layer->track_running_stats);
            set_float("dropout_p", layer->dropout_p);
            set_int("vocab_size", layer->vocab_size);
            set_int("embed_dim", layer->embed_dim);
            set_int("padding_idx", layer->padding_idx);
            set_int("axis", layer->axis);
            set_bool("use_mask", layer->use_mask);
            set_ints("target_shape", layer->target_shape);
            set_ints("shape", layer->shape);
            set_ints("permute_dims", layer->permute_dims);
            set_int("concat_axis", layer->concat_axis);
            set_int("num_splits", layer->num_splits);
            set_ints("split_sizes", layer->split_sizes);
            set_int("split_axis", layer->split_axis);
            set_float("scale_h", layer->scale_h);
            set_float("scale_w", layer->scale_w);
            set_int("out_h", layer->out_h);
            set_int("out_w", layer->out_w);
            set_int("output_h", layer->out_h);
            set_int("output_w", layer->out_w);
            set_int("num_heads", layer->num_heads);
            set_int("head_dim", layer->head_dim);
            set_int("seq_len", layer->seq_len);
            set_bool("causal", layer->causal);
            set_float("alpha", layer->alpha);
            set_float("negative_slope", layer->negative_slope);
            set_int("squeeze_dim", layer->squeeze_dim);
            set_int("unsqueeze_dim", layer->unsqueeze_dim);
            set_int("num_chunks", layer->num_chunks);
            set_int("stack_axis", layer->stack_axis);
            set_bool("use_bias", layer->use_bias);
            set_bool("trainable_parameter", layer->trainable_parameter);

            set_float("nms_iou_threshold", layer->nms_iou_threshold);
            set_float("iou_threshold", layer->nms_iou_threshold);
            set_float("nms_score_threshold", layer->nms_score_threshold);
            set_float("score_threshold", layer->nms_score_threshold);
            set_int("nms_max_detections", layer->nms_max_detections);
            set_int("max_detections", layer->nms_max_detections);
            set_bool("nms_class_agnostic", layer->nms_class_agnostic);
            set_bool("class_agnostic", layer->nms_class_agnostic);

            if (p.contains("axis")) {
                const int axis = p.at("axis").get<int>();
                layer->concat_axis = axis;
                layer->split_axis = axis;
                layer->stack_axis = axis;
            }
        }

        lua_pushboolean(L, true);
        return 1;
    } catch (const std::exception& e) {
        lua_pushboolean(L, false);
        lua_pushstring(L, e.what());
        return 2;
    }
}

int LuaScripting::lua_setLayerIO(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.currentModel) {
        lua_pushboolean(L, false);
        lua_pushstring(L, "Aucun modèle créé");
        return 2;
    }
    
    // Arguments:
    // 1. layer_name (string)
    // 2. inputs (table of strings, peut être vide)
    // 3. output (string, optionnel, défaut = "x")
    
    const char* layer_name = luaL_checkstring(L, 1);
    
    Layer* layer = ctx.currentModel->getLayerByName(layer_name);
    if (!layer) {
        lua_pushboolean(L, false);
        lua_pushfstring(L, "Layer '%s' not found", layer_name);
        return 2;
    }
    
    // Lire la table d'inputs
    layer->inputs.clear();
    if (lua_istable(L, 2)) {
        lua_pushnil(L);  // Premier key
        while (lua_next(L, 2) != 0) {
            // key à -2, value à -1
            if (lua_isstring(L, -1)) {
                layer->inputs.push_back(lua_tostring(L, -1));
            }
            lua_pop(L, 1);  // Pop value, garde key pour next
        }
    }
    
    // Lire l'output (optionnel)
    if (lua_gettop(L) >= 3 && lua_isstring(L, 3)) {
        layer->output = lua_tostring(L, 3);
    }
    
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
    
    // Argument 1: input
    // - soit un tableau (array) de floats/ints
    // - soit une table nommée { __input__ = <array_float>, text_ids = <array_int> }
    luaL_checktype(L, 1, LUA_TTABLE);
    
    // Argument 2 (optionnel): training (bool, défaut: true)
    bool training = true;
    if (lua_gettop(L) >= 2 && lua_isboolean(L, 2)) {
        training = lua_toboolean(L, 2);
    }

    auto maybe_push_viz_taps = [&]() {
        if (!ctx.asyncMonitor || !ctx.currentModel) return;
        if (ctx.asyncMonitor->getViz() == nullptr) return;

        // Auto-enable taps when viz is active.
        if (!ctx.currentModel->isVizTapsEnabled()) {
            ctx.currentModel->setVizTapsEnabled(true);
        }

        auto taps = ctx.currentModel->consumeVizTaps();
        if (taps.empty()) return;

        std::vector<Visualizer::BlockFrame> frames;
        frames.reserve(taps.size());
        for (auto& f : taps) {
            Visualizer::BlockFrame bf;
            bf.pixels = std::move(f.pixels);
            bf.w = f.w;
            bf.h = f.h;
            bf.channels = f.channels;
            bf.pixels_real = std::move(f.pixels_real);
                            bf.label = std::move(f.label);
            frames.push_back(std::move(bf));
        }
        ctx.asyncMonitor->setLayerBlockImages(frames);
    };
    
    // Support générique des entrées nommées (forwardPassNamedView):
    // { boxes={...}, scores={...}, classes={...} }, mais aussi les contrats
    // historiques { __input__={...}, text_ids={...} }.
    bool has_named_tensors = false;
    if (lua_rawlen(L, 1) == 0) {
        lua_pushnil(L);
        while (lua_next(L, 1) != 0) {
            if (lua_type(L, -2) == LUA_TSTRING && lua_istable(L, -1)) {
                has_named_tensors = true;
            }
            lua_pop(L, 1);
            if (has_named_tensors) {
                lua_pop(L, 1);
                break;
            }
        }
    }

    if (has_named_tensors) {
        std::unordered_map<std::string, std::vector<float>> fin;
        std::unordered_map<std::string, std::vector<int>> iin;

        lua_pushnil(L);
        while (lua_next(L, 1) != 0) {
            if (lua_type(L, -2) == LUA_TSTRING && lua_istable(L, -1)) {
                const std::string key = lua_tostring(L, -2);
                const size_t count = lua_rawlen(L, -1);
                if (key == "text_ids") {
                    std::vector<int> values;
                    values.reserve(count);
                    for (size_t i = 1; i <= count; ++i) {
                        lua_rawgeti(L, -1, static_cast<lua_Integer>(i));
                        values.push_back(static_cast<int>(lua_tointeger(L, -1)));
                        lua_pop(L, 1);
                    }
                    iin[key] = std::move(values);
                } else {
                    std::vector<float> values;
                    values.reserve(count);
                    for (size_t i = 1; i <= count; ++i) {
                        lua_rawgeti(L, -1, static_cast<lua_Integer>(i));
                        values.push_back(static_cast<float>(lua_tonumber(L, -1)));
                        lua_pop(L, 1);
                    }
                    fin[key] = std::move(values);
                }
            }
            lua_pop(L, 1);
        }

        if (fin.empty() && iin.empty()) {
            lua_pushnil(L);
            lua_pushstring(L, "Model.forward(named): expected named tensor arrays");
            return 2;
        }

        try {
            const std::vector<float>& output = ctx.currentModel->forwardPassNamedView(fin, iin, training);

            maybe_push_viz_taps();

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

    // Résoudre le tableau "array" à lire.
    // Si arg1 est une map {__input__=...}, on lit arg1.__input__.
    int input_index = 1;
    bool pushed_subtable = false;
    size_t n = lua_rawlen(L, input_index);
    if (n == 0) {
        lua_getfield(L, 1, "__input__");
        if (lua_istable(L, -1)) {
            input_index = lua_gettop(L);
            pushed_subtable = true;
            n = lua_rawlen(L, input_index);
        } else {
            lua_pop(L, 1);
        }
    }

    if (n == 0) {
        lua_pushnil(L);
        lua_pushstring(L, "Model.forward: expected an array of numbers or a table {__input__=<array>}");
        return 2;
    }

    // Détecter int vs float (et préserver l'ordre 1..n)
    bool all_int = true;
    for (size_t i = 1; i <= n; ++i) {
        lua_rawgeti(L, input_index, static_cast<lua_Integer>(i));
        const bool is_int = lua_isinteger(L, -1);
        lua_pop(L, 1);
        if (!is_int) {
            all_int = false;
            break;
        }
    }

    if (all_int) {
        std::vector<int> input_ids;
        input_ids.reserve(n);
        for (size_t i = 1; i <= n; ++i) {
            lua_rawgeti(L, input_index, static_cast<lua_Integer>(i));
            lua_Integer v = lua_tointeger(L, -1);
            lua_pop(L, 1);
            input_ids.push_back(static_cast<int>(v));
        }
        if (pushed_subtable) {
            lua_pop(L, 1);
            pushed_subtable = false;
        }
        try {
            std::vector<float> output = ctx.currentModel->forwardPass(input_ids, training);

            maybe_push_viz_taps();

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

    std::vector<float> input;
    input.reserve(n);
    for (size_t i = 1; i <= n; ++i) {
        lua_rawgeti(L, input_index, static_cast<lua_Integer>(i));
        input.push_back(static_cast<float>(lua_tonumber(L, -1)));
        lua_pop(L, 1);
    }
    if (pushed_subtable) {
        lua_pop(L, 1);
        pushed_subtable = false;
    }
    
    try {
        std::vector<float> output = ctx.currentModel->forwardPass(input, training);

        maybe_push_viz_taps();
        
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
        // Keep moments and step count across Lua calls and checkpoint saves.
        if (!ctx.currentModel->getSerializedOptimizer()) {
            Optimizer opt;
            opt.initial_lr = lr;
            if (std::string(opt_type) == "sgd") {
                opt.type = OptimizerType::SGD;
            } else if (std::string(opt_type) == "adam") {
                opt.type = OptimizerType::ADAM;
            } else {
                opt.type = OptimizerType::ADAMW;
            }
            const auto& cfg = ctx.currentModel->modelConfig;
            if (cfg.contains("beta1")) opt.beta1 = cfg["beta1"].get<float>();
            if (cfg.contains("beta2")) opt.beta2 = cfg["beta2"].get<float>();
            if (cfg.contains("epsilon")) opt.eps = cfg["epsilon"].get<float>();
            if (cfg.contains("weight_decay")) opt.weight_decay = cfg["weight_decay"].get<float>();
            ctx.currentModel->setSerializedOptimizer(std::move(opt));
        }
        Optimizer* opt = ctx.currentModel->getMutableSerializedOptimizer();
        if (!opt) throw std::runtime_error("optimizer state unavailable");
        opt->initial_lr = lr;
        ctx.currentModel->optimizerStep(*opt, lr, nullptr);
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

int LuaScripting::lua_setStdoutLogSuppressed(lua_State* L) {
    auto& ctx = LuaContext::getInstance();

    const int nargs = lua_gettop(L);
    if (nargs == 0) {
        lua_pushboolean(L, ctx.suppress_stdout_logs);
        return 1;
    }

    const bool enabled = lua_toboolean(L, 1);
    const bool previous = ctx.suppress_stdout_logs;
    ctx.suppress_stdout_logs = enabled;
    Model::setFrameworkLogsSuppressed(enabled);

    lua_pushboolean(L, previous);
    lua_pushboolean(L, enabled);
    return 2;
}

// ============================================================================
// Layer inspection API (read-only filters on the current model)
// ============================================================================

namespace {

static void pushLuaLayerInfo(lua_State* L, const Layer& la, size_t index) {
    lua_newtable(L);

    lua_pushinteger(L, static_cast<lua_Integer>(index + 1));
    lua_setfield(L, -2, "index");
    lua_pushstring(L, la.name.c_str());
    lua_setfield(L, -2, "name");
    lua_pushstring(L, la.type.c_str());
    lua_setfield(L, -2, "type");
    lua_pushinteger(L, static_cast<lua_Integer>(la.getWeightsSize()));
    lua_setfield(L, -2, "param_count");
    lua_pushstring(L, la.output.c_str());
    lua_setfield(L, -2, "output");

    lua_createtable(L, static_cast<int>(la.inputs.size()), 0);
    for (size_t j = 0; j < la.inputs.size(); ++j) {
        lua_pushstring(L, la.inputs[j].c_str());
        lua_rawseti(L, -2, static_cast<int>(j + 1));
    }
    lua_setfield(L, -2, "inputs");

    lua_pushinteger(L, la.in_features);   lua_setfield(L, -2, "in_features");
    lua_pushinteger(L, la.out_features);  lua_setfield(L, -2, "out_features");
    lua_pushinteger(L, la.in_channels);   lua_setfield(L, -2, "in_channels");
    lua_pushinteger(L, la.out_channels);  lua_setfield(L, -2, "out_channels");
    lua_pushinteger(L, la.kernel_size);   lua_setfield(L, -2, "kernel_size");
    lua_pushinteger(L, la.stride);        lua_setfield(L, -2, "stride");
    lua_pushinteger(L, la.padding);       lua_setfield(L, -2, "padding");
    lua_pushinteger(L, la.seq_len);       lua_setfield(L, -2, "seq_len");
    lua_pushinteger(L, la.embed_dim);     lua_setfield(L, -2, "embed_dim");
    lua_pushinteger(L, la.num_heads);     lua_setfield(L, -2, "num_heads");
    lua_pushinteger(L, la.vocab_size);    lua_setfield(L, -2, "vocab_size");
    lua_pushinteger(L, la.input_height);  lua_setfield(L, -2, "input_height");
    lua_pushinteger(L, la.input_width);   lua_setfield(L, -2, "input_width");
}

template <typename Predicate>
static int pushFilteredLayers(lua_State* L, Predicate predicate) {
    auto& ctx = LuaContext::getInstance();
    if (!ctx.currentModel) {
        lua_newtable(L);
        return 1;
    }

    const auto& layers = ctx.currentModel->getLayers();
    int out_index = 1;
    lua_newtable(L);
    for (size_t i = 0; i < layers.size(); ++i) {
        const Layer& la = layers[i];
        if (!predicate(la)) continue;
        pushLuaLayerInfo(L, la, i);
        lua_rawseti(L, -2, out_index++);
    }
    return 1;
}

static int pushLayersByExactType(lua_State* L, const std::string& requested_type) {
    const std::string normalized = LayerRegistry::normalize_type(requested_type);
    return pushFilteredLayers(L, [&](const Layer& la) {
        return LayerRegistry::normalize_type(la.type) == normalized;
    });
}

static bool isConv2dLike(const Layer& la) {
    return la.type == "Conv2d" || la.type == "ConvTranspose2d" || la.type == "DepthwiseConv2d";
}

static bool isLinearLike(const Layer& la) {
    return la.type == "Linear" || la.type == "MatMul" || la.type == "BatchMatMul" || la.type == "Bilinear";
}

static bool isMaxPoolLike(const Layer& la) {
    return la.type == "MaxPool2d" || la.type == "GlobalMaxPool2d" || la.type == "Pool1d";
}

static bool isAvgPoolLike(const Layer& la) {
    return la.type == "AvgPool2d" || la.type == "GlobalAvgPool2d";
}

static bool isActivationLike(const Layer& la) {
    static const std::unordered_set<std::string> kActivationTypes = {
        "ReLU", "GELU", "GEGLU", "SiLU", "LeakyReLU", "Tanh", "Sigmoid",
        "Softmax", "LogSoftmax", "Swish", "Mish", "Softplus", "HardSigmoid", "HardSwish"
    };
    return kActivationTypes.find(la.type) != kActivationTypes.end();
}

static bool isBatchNormLike(const Layer& la) {
    return la.type == "BatchNorm2d" || la.type == "InstanceNorm2d";
}

static bool isLayerNormLike(const Layer& la) {
    return la.type == "LayerNorm" || la.type == "GroupNorm" || la.type == "RMSNorm";
}

static bool isAttentionLike(const Layer& la) {
    return la.type == "Attention" || la.type == "MultiHeadAttention" || la.type == "SelfAttention" || la.type == "CrossAttention";
}

} // namespace

int LuaScripting::lua_layersAvailable(lua_State* L) {
    const auto types = LayerRegistry::get_all_supported_types();
    lua_createtable(L, static_cast<int>(types.size()), 0);
    for (size_t i = 0; i < types.size(); ++i) {
        lua_pushstring(L, types[i].c_str());
        lua_rawseti(L, -2, static_cast<int>(i + 1));
    }
    return 1;
}

int LuaScripting::lua_layersByType(lua_State* L) {
    std::string requested_type;
    if (lua_gettop(L) >= 1 && lua_isstring(L, 1)) {
        requested_type = lua_tostring(L, 1);
    } else if (lua_type(L, lua_upvalueindex(1)) == LUA_TSTRING) {
        requested_type = lua_tostring(L, lua_upvalueindex(1));
    }

    if (requested_type.empty()) {
        lua_newtable(L);
        return 1;
    }
    return pushLayersByExactType(L, requested_type);
}

int LuaScripting::lua_computeConv2D(lua_State* L) {
    return pushFilteredLayers(L, isConv2dLike);
}

int LuaScripting::lua_computeLinear(lua_State* L) {
    return pushFilteredLayers(L, isLinearLike);
}

int LuaScripting::lua_computeMaxPool2D(lua_State* L) {
    return pushFilteredLayers(L, isMaxPoolLike);
}

int LuaScripting::lua_computeAvgPool2D(lua_State* L) {
    return pushFilteredLayers(L, isAvgPoolLike);
}

int LuaScripting::lua_computeActivation(lua_State* L) {
    return pushFilteredLayers(L, isActivationLike);
}

int LuaScripting::lua_computeBatchNorm(lua_State* L) {
    return pushFilteredLayers(L, isBatchNormLike);
}

int LuaScripting::lua_computeLayerNorm(lua_State* L) {
    return pushFilteredLayers(L, isLayerNormLike);
}

int LuaScripting::lua_computeAttention(lua_State* L) {
    return pushFilteredLayers(L, isAttentionLike);
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

int LuaScripting::lua_getMaxVocab(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    if (!ctx.currentTokenizer) {
        lua_pushinteger(L, 0);
        return 1;
    }
    lua_pushinteger(L, static_cast<lua_Integer>(ctx.currentTokenizer->getMaxVocab()));
    return 1;
}

int LuaScripting::lua_setMaxVocab(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    if (!ctx.currentTokenizer) {
        lua_pushboolean(L, false);
        lua_pushstring(L, "Aucun tokenizer créé");
        return 2;
    }

    const lua_Integer v = luaL_checkinteger(L, 1);
    const size_t new_max = static_cast<size_t>(std::max<lua_Integer>(0, v));
    ctx.currentTokenizer->setMaxVocab(new_max);
    lua_pushboolean(L, true);
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
        const std::filesystem::path output_path(filepath);
        if (output_path.has_parent_path()) {
            std::filesystem::create_directories(output_path.parent_path());
        }
        json j = ctx.currentTokenizer->to_json();
        std::ofstream f(filepath);
        if (!f) {
            throw std::runtime_error("Impossible de créer le fichier tokenizer: " + std::string(filepath));
        }
        f << j.dump(2);
        if (!f.good()) {
            throw std::runtime_error("Échec d'écriture du tokenizer: " + std::string(filepath));
        }
        
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
        if (!f) {
            throw std::runtime_error("Fichier tokenizer introuvable ou illisible: " + std::string(filepath));
        }
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
    auto& ctx = LuaContext::getInstance();
    auto& mgr = AdvancedRAMManager::instance();
    
    // Argument: table de configuration
    if (!lua_istable(L, 1)) {
        lua_pushboolean(L, false);
        lua_pushstring(L, "Argument 1 doit être une table de configuration");
        return 2;
    }
    
    AdvancedRAMManager::Config config;
    // Spill disque: requis pour pouvoir décharger des données déjà initialisées
    // (dossier attendu par l'UX/outils)
    config.enable_disk_spill = true;
    config.spill_dir = ".mimir-spill";
    
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

    ctx.addLog("🔧 Gestionnaire de mémoire configuré:");
    ctx.addLog("   - Limite RAM: " + std::to_string(config.max_ram_bytes / 1024 / 1024 / 1024) + " GB");
    ctx.addLog(std::string("   - Compression: ") + (config.enable_compression ? "activée" : "désactivée"));
    ctx.addLog(std::string("   - Chargement async: ") + (config.enable_async_loading ? "activé" : "désactivé"));
    ctx.addLog(std::string("   - Prédiction: ") + (config.enable_prediction ? "activée" : "désactivée"));
    ctx.addLog("   - Worker threads: " + std::to_string(config.worker_threads));
    
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
    auto& ctx = LuaContext::getInstance();
    auto& mgr = AdvancedRAMManager::instance();
    mgr.clear();

    ctx.addLog("🧹 Mémoire effacée");
    
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
    auto& ctx = LuaContext::getInstance();
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
    config.enable_disk_spill = true;
    config.spill_dir = ".mimir-spill";
    config.worker_threads = 2;
    
    mgr.configure(config);

    ctx.addLog("💾 Limite RAM définie à " + std::to_string(gb) + " GB");
    
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
    auto& ctx = LuaContext::getInstance();
    auto& guard = MemoryGuard::instance();
    guard.reset();

    ctx.addLog("🔄 MemoryGuard réinitialisé");
    
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
    auto& ctx = LuaContext::getInstance();
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
    allocator.configure(max_ram_gb, enable_compression);

    ctx.addLog("✓ DynamicTensorAllocator configuré");
    ctx.addLog("   - Limite: " + std::to_string(max_ram_gb) + " GB");
    ctx.addLog(std::string("   - Compression: ") + (enable_compression ? "activée" : "désactivée"));
    
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

        // Signature supportée:
        //   - Htop.create(enable_viz?: boolean)
        //   - Htop.create({ enable_viz = bool, enable_htop = bool, viz_config = { ... } })
        bool enable_viz = false;
        bool enable_htop = true;
        json viz_config;
        std::optional<bool> csv_flag;
        std::optional<bool> csv_enabled;
        std::optional<std::string> csv_path;
        if (lua_istable(L, 1)) {
            lua_getfield(L, 1, "enable_viz");
            if (lua_isboolean(L, -1)) enable_viz = lua_toboolean(L, -1);
            lua_pop(L, 1);

            lua_getfield(L, 1, "enable_htop");
            if (lua_isboolean(L, -1)) enable_htop = lua_toboolean(L, -1);
            lua_pop(L, 1);

            // Alias de compat (certains scripts passent {viz=true}).
            lua_getfield(L, 1, "viz");
            if (!enable_viz && lua_isboolean(L, -1)) enable_viz = lua_toboolean(L, -1);
            lua_pop(L, 1);

            lua_getfield(L, 1, "viz_config");
            if (lua_istable(L, -1)) {
                viz_config = luaTableToJson(L, -1);
            }
            lua_pop(L, 1);

            // Options CSV htop (compat Lua):
            // - csv=true/false
            // - csv_enabled=true/false
            // - csv_path="..." (alias: csv_file)
            lua_getfield(L, 1, "csv");
            if (lua_isboolean(L, -1)) csv_flag = (bool)lua_toboolean(L, -1);
            lua_pop(L, 1);

            lua_getfield(L, 1, "csv_enabled");
            if (lua_isboolean(L, -1)) csv_enabled = (bool)lua_toboolean(L, -1);
            lua_pop(L, 1);

            lua_getfield(L, 1, "csv_path");
            if (lua_isstring(L, -1)) csv_path = std::string(lua_tostring(L, -1));
            lua_pop(L, 1);

            if (!csv_path.has_value()) {
                lua_getfield(L, 1, "csv_file");
                if (lua_isstring(L, -1)) csv_path = std::string(lua_tostring(L, -1));
                lua_pop(L, 1);
            }
        } else if (!lua_isnoneornil(L, 1)) {
            enable_viz = lua_toboolean(L, 1);
        }

        // Si l'utilisateur demande la viz sans fournir de config, on force l'activation.
        // (Le Visualizer est disabled par défaut si visualization.enabled n'est pas true.)
        if (enable_viz) {
            if (!viz_config.contains("visualization")) viz_config["visualization"] = json::object();
            if (!viz_config["visualization"].contains("enabled")) {
                viz_config["visualization"]["enabled"] = true;
            }
        }

        ctx.asyncMonitor->start(enable_htop, enable_viz, viz_config);

        // Appliquer les options CSV côté HtopDisplay si présent.
        // Par défaut, AsyncMonitor désactive le CSV Htop si la Viz est active,
        // pour éviter les écritures concurrentes.
        if (enable_htop) {
            auto h = ctx.asyncMonitor->getHtop();
            if (h) {
                if (csv_path.has_value() && !csv_path->empty()) {
                    h->setCsvLogFile(*csv_path);
                }

                bool enable_csv = !enable_viz;
                if (csv_path.has_value()) {
                    // Si l'utilisateur fournit un chemin, activer par défaut.
                    enable_csv = true;
                }
                if (csv_flag.has_value()) {
                    enable_csv = *csv_flag;
                }
                if (csv_enabled.has_value()) {
                    enable_csv = *csv_enabled;
                }

                h->setCsvEnabled(enable_csv);
            }
        }

        ctx.addLog(std::string("AsyncMonitor démarré (") + (enable_htop ? "htop enabled" : "htop disabled") + ")");
        lua_pushboolean(L, true);

        // Remonter un warning sans casser les scripts (ok=true + msg).
        if (enable_viz && !ctx.asyncMonitor->vizInitOk()) {
            const std::string err = ctx.asyncMonitor->vizInitError();
            if (!err.empty()) {
                lua_pushstring(L, (std::string("Viz init failed: ") + err).c_str());
                return 2;
            }
        }
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
    
    AsyncMonitor::Metrics metrics;

    // Signature supportée:
    //   - Htop.update(tbl)
    //   - Htop.update(epoch, total_epochs, batch, total_batches, loss, avg_loss, lr, ...)
    if (lua_istable(L, 1)) {
        auto get_int = [&](const char* key, int def) -> int {
            lua_getfield(L, 1, key);
            int v = lua_isnumber(L, -1) ? (int)lua_tointeger(L, -1) : def;
            lua_pop(L, 1);
            return v;
        };
        auto get_num = [&](const char* key, float def) -> float {
            lua_getfield(L, 1, key);
            float v = lua_isnumber(L, -1) ? (float)lua_tonumber(L, -1) : def;
            lua_pop(L, 1);
            return v;
        };
        auto get_str = [&](const char* key, const char* def) -> std::string {
            lua_getfield(L, 1, key);
            std::string v = lua_isstring(L, -1) ? std::string(lua_tostring(L, -1)) : std::string(def);
            lua_pop(L, 1);
            return v;
        };

        metrics.epoch = get_int("epoch", 0);
        metrics.total_epochs = get_int("total_epochs", get_int("totalEpochs", 0));
        metrics.batch = get_int("batch", 0);
        metrics.total_batches = get_int("total_batches", get_int("totalBatches", 0));

        // Compat: certains scripts utilisent `step`.
        const int step = get_int("step", 0);
        if (metrics.batch == 0 && step > 0) metrics.batch = step;

        metrics.loss = get_num("loss", 0.0f);
        metrics.avg_loss = get_num("avg_loss", get_num("avgLoss", 0.0f));
        metrics.lr = get_num("lr", 0.0f);
        metrics.batch_time_ms = get_int("batch_time_ms", get_int("batchTimeMs", 0));
        metrics.memory_mb = (size_t)std::max(0, get_int("memory_mb", get_int("memoryMb", 0)));
        metrics.memory_freed = (size_t)std::max(0, get_int("memory_freed", get_int("memoryFreed", 0)));
        metrics.bps = get_num("bps", 0.0f);
        metrics.params = (size_t)std::max(0, get_int("params", 0));
        metrics.timestep = get_num("timestep", 0.0f);
        metrics.kl = get_num("kl", 0.0f);
        metrics.wass = get_num("wass", 0.0f);
        metrics.ent = get_num("ent", 0.0f);
        metrics.mom = get_num("mom", 0.0f);
        metrics.spat = get_num("spat", 0.0f);
        metrics.temp = get_num("temp", 0.0f);
        metrics.mse = get_num("mse", 0.0f);
        metrics.grad_norm = get_num("grad_norm", get_num("gradNorm", 0.0f));
        metrics.grad_max = get_num("grad_max", get_num("gradMax", 0.0f));
        metrics.recon_loss_type = get_str("recon_loss_type", "");
        if (metrics.recon_loss_type.empty()) {
            metrics.recon_loss_type = get_str("reconLoss", "");
        }

        // Optimizer (optionnel) : soit champs plats, soit sous-table `optimizer`.
        metrics.opt_type = get_int("opt_type", get_int("optType", 0));
        metrics.opt_step = get_int("opt_step", get_int("optStep", 0));
        metrics.opt_beta1 = get_num("opt_beta1", get_num("optBeta1", 0.0f));
        metrics.opt_beta2 = get_num("opt_beta2", get_num("optBeta2", 0.0f));
        metrics.opt_eps = get_num("opt_eps", get_num("optEps", 0.0f));
        metrics.opt_weight_decay = get_num("opt_weight_decay", get_num("optWeightDecay", 0.0f));

        auto parse_opt_type = [&](const std::string& s) -> int {
            std::string t = s;
            std::transform(t.begin(), t.end(), t.begin(), [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
            if (t == "sgd") return 0;
            if (t == "adam") return 1;
            if (t == "adamw") return 2;
            return metrics.opt_type;
        };

        // Support: opt_type="adamw" (string)
        lua_getfield(L, 1, "opt_type");
        if (lua_isstring(L, -1)) {
            metrics.opt_type = parse_opt_type(lua_tostring(L, -1));
        }
        lua_pop(L, 1);
        lua_getfield(L, 1, "optType");
        if (lua_isstring(L, -1)) {
            metrics.opt_type = parse_opt_type(lua_tostring(L, -1));
        }
        lua_pop(L, 1);

        lua_getfield(L, 1, "optimizer");
        if (lua_isstring(L, -1)) {
            // Support: optimizer="adamw" (string)
            metrics.opt_type = parse_opt_type(lua_tostring(L, -1));
        } else if (lua_istable(L, -1)) {
            lua_getfield(L, -1, "type");
            if (lua_isnumber(L, -1)) metrics.opt_type = (int)lua_tointeger(L, -1);
            else if (lua_isstring(L, -1)) metrics.opt_type = parse_opt_type(lua_tostring(L, -1));
            lua_pop(L, 1);
            lua_getfield(L, -1, "step");
            if (lua_isnumber(L, -1)) metrics.opt_step = (int)lua_tointeger(L, -1);
            lua_pop(L, 1);
            lua_getfield(L, -1, "beta1");
            if (lua_isnumber(L, -1)) metrics.opt_beta1 = (float)lua_tonumber(L, -1);
            lua_pop(L, 1);
            lua_getfield(L, -1, "beta2");
            if (lua_isnumber(L, -1)) metrics.opt_beta2 = (float)lua_tonumber(L, -1);
            lua_pop(L, 1);
            lua_getfield(L, -1, "eps");
            if (lua_isnumber(L, -1)) metrics.opt_eps = (float)lua_tonumber(L, -1);
            lua_pop(L, 1);
            lua_getfield(L, -1, "weight_decay");
            if (lua_isnumber(L, -1)) metrics.opt_weight_decay = (float)lua_tonumber(L, -1);
            lua_pop(L, 1);
        }
        lua_pop(L, 1);
    } else {
        // Arguments positionnels (legacy)
        // epoch, total_epochs, batch, total_batches, loss, avg_loss, lr,
        // batch_time_ms, memory_mb, memory_freed, bps, params, timestep,
        // kl, wass, ent, mom, spat, temp, mse, grad_norm, grad_max,
        // [opt_type, opt_step, opt_beta1, opt_beta2, opt_eps, opt_weight_decay]
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
        metrics.grad_norm = static_cast<float>(luaL_optnumber(L, 21, 0.0));
        metrics.grad_max = static_cast<float>(luaL_optnumber(L, 22, 0.0));

        // Optimizer (optionnel)
        metrics.opt_type = luaL_optinteger(L, 23, 0);
        metrics.opt_step = luaL_optinteger(L, 24, 0);
        metrics.opt_beta1 = static_cast<float>(luaL_optnumber(L, 25, 0.0));
        metrics.opt_beta2 = static_cast<float>(luaL_optnumber(L, 26, 0.0));
        metrics.opt_eps = static_cast<float>(luaL_optnumber(L, 27, 0.0));
        metrics.opt_weight_decay = static_cast<float>(luaL_optnumber(L, 28, 0.0));
    }
    
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

    // Si l'utilisateur appelle explicitement Viz.create(), on active la viz par défaut.
    if (!config.contains("visualization")) config["visualization"] = json::object();
    if (!config["visualization"].contains("enabled")) {
        config["visualization"]["enabled"] = true;
    }
    
    try {
        if (!ctx.asyncMonitor) {
            ctx.asyncMonitor = std::make_shared<AsyncMonitor>();
        }
        
        // Démarrer avec viz activé (et htop désactivé si pas déjà démarré)
        ctx.asyncMonitor->start(false, true, config);

        if (!ctx.asyncMonitor->vizInitOk()) {
            lua_pushboolean(L, false);
            const std::string err = ctx.asyncMonitor->vizInitError();
            lua_pushstring(L, err.empty() ? "Visualizer init failed" : err.c_str());
            return 2;
        }
        
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
    
    // Arguments (compat):
    //  - add_image(pixels_table, prompt)
    //  - add_image(pixels_table, prompt, w, h, channels)
    //  - add_image(pixels_table, w, h, channels, prompt)
    if (!lua_istable(L, 1)) {
        lua_pushboolean(L, false);
        lua_pushstring(L, "Argument 1 doit être une table de pixels");
        return 2;
    }

    int w = 0;
    int h = 0;
    int channels = 0;
    std::string prompt;

    if (lua_isnumber(L, 2)) {
        // (pixels, w, h, channels, prompt)
        w = luaL_checkinteger(L, 2);
        h = luaL_checkinteger(L, 3);
        channels = luaL_optinteger(L, 4, 0);
        prompt = luaL_optstring(L, 5, "");
    } else {
        // (pixels, prompt[, w, h, channels])
        prompt = luaL_optstring(L, 2, "");
        if (lua_isnumber(L, 3)) {
            w = luaL_checkinteger(L, 3);
            h = luaL_checkinteger(L, 4);
            channels = luaL_optinteger(L, 5, 0);
        }
    }
    
    // Lire la table de pixels
    std::vector<uint8_t> pixels;
    lua_pushnil(L);
    while (lua_next(L, 1) != 0) {
        if (lua_isnumber(L, -1)) {
            pixels.push_back(static_cast<uint8_t>(lua_tointeger(L, -1)));
        }
        lua_pop(L, 1);
    }
    
    ctx.asyncMonitor->addImage(pixels, w, h, channels, prompt);
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
    
    // Signature supportée:
    //   - Viz.update_metrics(tbl)  (recommandé)
    //   - Viz.update_metrics(epoch, batch, loss, lr, mse, kl, wass, ent, mom, spat, temp, kl_beta_effective[, time_ms, mem_mb, bps, params])
    // Notes compat:
    //   - tbl.time / tbl.time_ms -> batch_time_ms
    //   - tbl.mem / tbl.mem_mb   -> memory_mb
    AsyncMonitor::Metrics metrics;

    if (lua_istable(L, 1)) {
        auto get_int = [&](const char* key, int def) -> int {
            lua_getfield(L, 1, key);
            int v = lua_isnumber(L, -1) ? (int)lua_tointeger(L, -1) : def;
            lua_pop(L, 1);
            return v;
        };
        auto get_num = [&](const char* key, float def) -> float {
            lua_getfield(L, 1, key);
            float v = lua_isnumber(L, -1) ? (float)lua_tonumber(L, -1) : def;
            lua_pop(L, 1);
            return v;
        };
        auto get_str = [&](const char* key, const char* def) -> std::string {
            lua_getfield(L, 1, key);
            std::string v = lua_isstring(L, -1) ? std::string(lua_tostring(L, -1)) : std::string(def);
            lua_pop(L, 1);
            return v;
        };

        metrics.epoch = get_int("epoch", 0);
        metrics.total_epochs = get_int("total_epochs", get_int("totalEpochs", 0));
        metrics.batch = get_int("batch", 0);
        metrics.total_batches = get_int("total_batches", get_int("totalBatches", 0));
        const int step = get_int("step", 0);
        if (metrics.batch == 0 && step > 0) metrics.batch = step;

        metrics.loss = get_num("loss", 0.0f);
        metrics.avg_loss = get_num("avg_loss", get_num("avgLoss", 0.0f));
        metrics.lr = get_num("lr", 0.0f);

        metrics.batch_time_ms = get_int(
            "batch_time_ms",
            get_int("batchTimeMs", get_int("time_ms", get_int("timeMs", get_int("time", 0))))
        );
        metrics.memory_mb = (size_t)std::max(
            0,
            get_int("memory_mb", get_int("memoryMb", get_int("mem_mb", get_int("memMb", get_int("mem", 0)))))
        );
        metrics.bps = get_num("bps", get_num("batches_per_sec", get_num("batchesPerSec", 0.0f)));
        metrics.params = (size_t)std::max(0, get_int("params", 0));

        metrics.mse = get_num("mse", 0.0f);
        metrics.kl = get_num("kl", 0.0f);
        metrics.wass = get_num("wass", 0.0f);
        metrics.ent = get_num("ent", 0.0f);
        metrics.mom = get_num("mom", 0.0f);
        metrics.spat = get_num("spat", 0.0f);
        metrics.temp = get_num("temp", 0.0f);
        metrics.timestep = get_num("timestep", 0.0f);
        metrics.grad_norm = get_num("grad_norm", get_num("gradNorm", 0.0f));
        metrics.grad_max = get_num("grad_max", get_num("gradMax", 0.0f));
        metrics.kl_beta_effective = get_num("kl_beta_effective", get_num("klBetaEffective", 0.0f));
        metrics.recon_loss_type = get_str("recon_loss_type", "");
        if (metrics.recon_loss_type.empty()) {
            metrics.recon_loss_type = get_str("reconLoss", "");
        }
    } else {
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
        metrics.kl_beta_effective = static_cast<float>(luaL_optnumber(L, 12, 0.0));
        metrics.batch_time_ms = luaL_optinteger(L, 13, 0);
        {
            lua_Integer v = luaL_optinteger(L, 14, 0);
            if (v < 0) v = 0;
            metrics.memory_mb = (size_t)v;
        }
        metrics.bps = static_cast<float>(luaL_optnumber(L, 15, 0.0));
        {
            lua_Integer v = luaL_optinteger(L, 16, 0);
            if (v < 0) v = 0;
            metrics.params = (size_t)v;
        }
    }
    
    ctx.asyncMonitor->updateMetrics(metrics);
    
    lua_pushboolean(L, true);
    return 1;
}

int LuaScripting::lua_vizSetValidation(lua_State* L) {
    auto& ctx = LuaContext::getInstance();

    if (!ctx.asyncMonitor) {
        lua_pushboolean(L, false);
        lua_pushstring(L, "AsyncMonitor non créé");
        return 2;
    }

    if (!lua_istable(L, 1)) {
        lua_pushboolean(L, false);
        lua_pushstring(L, "Argument 1 doit être une table (ex: {in_progress=true, step=123, done=1, total=8})");
        return 2;
    }

    auto getBoolField = [&](const char* key, bool def) -> bool {
        lua_getfield(L, 1, key);
        bool v = def;
        if (lua_isboolean(L, -1)) v = lua_toboolean(L, -1);
        lua_pop(L, 1);
        return v;
    };
    auto getIntField = [&](const char* key, int def) -> int {
        lua_getfield(L, 1, key);
        int v = def;
        if (lua_isnumber(L, -1)) v = static_cast<int>(lua_tointeger(L, -1));
        lua_pop(L, 1);
        return v;
    };
    auto getNumField = [&](const char* key, float def) -> float {
        lua_getfield(L, 1, key);
        float v = def;
        if (lua_isnumber(L, -1)) v = static_cast<float>(lua_tonumber(L, -1));
        lua_pop(L, 1);
        return v;
    };

    const bool in_progress = getBoolField("in_progress", false);
    const int step = getIntField("step", 0);
    const int done = getIntField("done", 0);
    const int total = getIntField("total", 0);
    const bool has = getBoolField("has", false);
    const bool ok = getBoolField("ok", true);
    const float recon = getNumField("recon", 0.0f);
    const float kl = getNumField("kl", 0.0f);
    const float align = getNumField("align", 0.0f);

    ctx.asyncMonitor->updateValidation(in_progress, step, done, total, has, ok, recon, kl, align);

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
