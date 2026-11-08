#pragma once

#include "../Layers.hpp"
#include "../runtimes/PrimerGeometry.hpp"
#include "../include/json.hpp"
#include <stdexcept>
#include <utility>

namespace Mimir::Serialization::LayerMetadata {
inline constexpr int version=1;
inline bool convolutionND(const Layer& layer) {
    return layer.type_enum==LayerType::Conv3d || layer.type_enum==LayerType::Conv4d;
}
inline bool containsConvolutionND(const nlohmann::json& layers) {
    for(const auto& entry:layers) {
        const std::string type=entry.value("type","");
        if(LayerRegistry::string_to_type(LayerRegistry::normalize_type(type))==LayerType::Conv3d ||
           LayerRegistry::string_to_type(LayerRegistry::normalize_type(type))==LayerType::Conv4d)return true;
    }
    return false;
}
inline bool requiresCompleteArchitecture(const nlohmann::json& layers) {
    if (containsConvolutionND(layers)) return true;
    for (const auto& entry : layers) {
        const auto type = LayerRegistry::string_to_type(LayerRegistry::normalize_type(entry.value("type", "")));
        if (type == LayerType::CausalDepthwiseConv1d || type == LayerType::SwiGLU || RuntimePrimer::supported(type)) return true;
    }
    return false;
}
inline const auto& spatialIntegers() {
    static const std::pair<const char*,int Layer::*> fields[]={
        {"in_channels",&Layer::in_channels},{"out_channels",&Layer::out_channels},
        {"input_time",&Layer::input_time},{"input_depth",&Layer::input_depth},
        {"input_height",&Layer::input_height},{"input_width",&Layer::input_width},
        {"output_time",&Layer::output_time},{"output_depth",&Layer::output_depth},
        {"output_height",&Layer::output_height},{"output_width",&Layer::output_width},
        {"kernel_size",&Layer::kernel_size},{"kernel_t",&Layer::kernel_t},
        {"kernel_d",&Layer::kernel_d},{"kernel_h",&Layer::kernel_h},{"kernel_w",&Layer::kernel_w},
        {"stride",&Layer::stride},{"stride_t",&Layer::stride_t},{"stride_d",&Layer::stride_d},
        {"stride_h",&Layer::stride_h},{"stride_w",&Layer::stride_w},
        {"padding",&Layer::padding},{"pad_t",&Layer::pad_t},{"pad_d",&Layer::pad_d},
        {"pad_h",&Layer::pad_h},{"pad_w",&Layer::pad_w},
        {"dilation",&Layer::dilation},{"dilation_t",&Layer::dilation_t},
        {"dilation_d",&Layer::dilation_d},{"dilation_h",&Layer::dilation_h},
        {"dilation_w",&Layer::dilation_w},{"groups",&Layer::groups}
    };
    return fields;
}
inline const auto& commonIntegers() {
    static const std::pair<const char*,int Layer::*> fields[]={
        {"in_features",&Layer::in_features},{"out_features",&Layer::out_features},
        {"num_groups",&Layer::num_groups},{"vocab_size",&Layer::vocab_size},
        {"embed_dim",&Layer::embed_dim},{"padding_idx",&Layer::padding_idx},{"axis",&Layer::axis},
        {"nms_max_detections",&Layer::nms_max_detections},{"skip_input_index",&Layer::skip_input_index},
        {"concat_axis",&Layer::concat_axis},{"num_splits",&Layer::num_splits},{"split_axis",&Layer::split_axis},
        {"out_h",&Layer::out_h},{"out_w",&Layer::out_w},{"num_heads",&Layer::num_heads},
        {"num_kv_heads",&Layer::num_kv_heads},{"head_dim",&Layer::head_dim},{"seq_len",&Layer::seq_len},
        {"patch_dim",&Layer::patch_dim},{"num_patches",&Layer::num_patches},{"seq_text",&Layer::seq_text},
        {"squeeze_dim",&Layer::squeeze_dim},{"unsqueeze_dim",&Layer::unsqueeze_dim},
        {"num_chunks",&Layer::num_chunks},{"stack_axis",&Layer::stack_axis},{"branch_target",&Layer::branch_target}
    };
    return fields;
}
inline const auto& commonFloats() {
    static const std::pair<const char*,float Layer::*> fields[]={
        {"eps",&Layer::eps},{"momentum",&Layer::momentum},{"dropout_p",&Layer::dropout_p},
        {"nms_iou_threshold",&Layer::nms_iou_threshold},{"nms_score_threshold",&Layer::nms_score_threshold},
        {"scale_h",&Layer::scale_h},{"scale_w",&Layer::scale_w},{"rope_theta",&Layer::rope_theta},
        {"alpha",&Layer::alpha},{"negative_slope",&Layer::negative_slope},
        {"leaky_relu_alpha",&Layer::leaky_relu_alpha},{"activation_param",&Layer::activation_param}
    };
    return fields;
}
inline const auto& commonBooleans() {
    static const std::pair<const char*,bool Layer::*> fields[]={
        {"trainable_parameter",&Layer::trainable_parameter},{"affine",&Layer::affine},
        {"track_running_stats",&Layer::track_running_stats},{"use_mask",&Layer::use_mask},
        {"nms_class_agnostic",&Layer::nms_class_agnostic},{"causal",&Layer::causal},
        {"use_bias",&Layer::use_bias},{"is_branch_point",&Layer::is_branch_point},
        {"is_merge_point",&Layer::is_merge_point}
    };
    return fields;
}
inline const auto& spatialVectors() {
    static const std::pair<const char*,std::vector<int> Layer::*> fields[]={
        {"input_spatial_shape",&Layer::input_spatial_shape},{"kernel_shape",&Layer::kernel_shape},
        {"stride_shape",&Layer::stride_shape},{"padding_shape",&Layer::padding_shape},
        {"dilation_shape",&Layer::dilation_shape},{"shape",&Layer::shape}
    };
    return fields;
}
inline const auto& commonIntegerVectors() {
    static const std::pair<const char*,std::vector<int> Layer::*> fields[]={
        {"target_shape",&Layer::target_shape},{"permute_dims",&Layer::permute_dims},
        {"split_sizes",&Layer::split_sizes},{"branch_sources",&Layer::branch_sources}
    };
    return fields;
}
inline const auto& commonFloatVectors() {
    static const std::pair<const char*,std::vector<float> Layer::*> fields[]={
        {"running_mean",&Layer::running_mean},{"running_var",&Layer::running_var}
    };
    return fields;
}
inline const auto& commonStrings() {
    static const std::pair<const char*,std::string Layer::*> fields[]={
        {"name",&Layer::name},{"type",&Layer::type},{"shared_weights_from",&Layer::shared_weights_from},
        {"output",&Layer::output}
    };
    return fields;
}
inline const auto& commonStringVectors() {
    static const std::pair<const char*,std::vector<std::string> Layer::*> fields[]={
        {"inputs",&Layer::inputs}
    };
    return fields;
}
inline const auto& commonSizes() {
    static const std::pair<const char*,size_t Layer::*> fields[]={{"params_count",&Layer::params_count}};
    return fields;
}
inline const auto& commonDTypes() {
    static const std::pair<const char*,::Mimir::DType Layer::*> fields[]={
        {"dtype",&Layer::dtype},{"accumulation_dtype",&Layer::accumulation_dtype}
    };
    return fields;
}
template<class Fields>
inline void writeFields(nlohmann::json& object,const Layer& layer,const Fields& fields) {
    for(const auto& [name,member]:fields)object[name]=layer.*member;
}
template<class T,size_t N>
inline void readFields(Layer& layer,const nlohmann::json& object,const std::pair<const char*,T Layer::*> (&fields)[N]) {
    for(const auto& [name,member]:fields)if(object.contains(name))layer.*member=object.at(name).template get<T>();
}
template<class Fields>
inline bool containsFields(const nlohmann::json& object,const Fields& fields) {
    for(const auto& [name,member]:fields){(void)member;if(!object.contains(name))return false;}
    return true;
}
template<class Enum>
inline void readEnum(Layer& layer,const nlohmann::json& object,const char* name,Enum Layer::* member,Enum last) {
    if(!object.contains(name))return;
    const int value=object.at(name).get<int>();
    if(value<0 || value>static_cast<int>(last))throw std::invalid_argument(std::string("Invalid layer ")+name+" in checkpoint");
    layer.*member=static_cast<Enum>(value);
}
inline void readDTypes(Layer& layer,const nlohmann::json& object) {
    for(const auto& [name,member]:commonDTypes())if(object.contains(name)) {
        const std::string encoded=object.at(name).get<std::string>();
        const auto dtype=::Mimir::parse_dtype_safetensors(encoded);
        if(dtype==::Mimir::DType::UNKNOWN && encoded!="unknown")throw std::invalid_argument(std::string("Invalid layer ")+name+" in checkpoint");
        layer.*member=dtype;
    }
}
// Configuration only: no weights/gradients, tensor arithmetic or backend calls.
inline void writeCommon(nlohmann::json& object,const Layer& layer) {
    object["metadata_version"]=version;
    writeFields(object,layer,spatialIntegers());writeFields(object,layer,commonIntegers());
    writeFields(object,layer,commonFloats());writeFields(object,layer,commonBooleans());
    writeFields(object,layer,spatialVectors());writeFields(object,layer,commonIntegerVectors());
    writeFields(object,layer,commonFloatVectors());writeFields(object,layer,commonStrings());
    writeFields(object,layer,commonStringVectors());writeFields(object,layer,commonSizes());
    for(const auto& [name,member]:commonDTypes())object[name]=::Mimir::dtype_to_string(layer.*member);
    object["activation"]=static_cast<int>(layer.activation);
    object["branch_type"]=static_cast<int>(layer.branch_type);
    object["merge_op"]=static_cast<int>(layer.merge_op);
}
inline void readCommon(Layer& layer,const nlohmann::json& object) {
    readFields(layer,object,spatialIntegers());readFields(layer,object,commonIntegers());
    readFields(layer,object,commonFloats());readFields(layer,object,commonBooleans());
    readFields(layer,object,spatialVectors());readFields(layer,object,commonIntegerVectors());
    readFields(layer,object,commonFloatVectors());readFields(layer,object,commonStrings());
    readFields(layer,object,commonStringVectors());readFields(layer,object,commonSizes());
    readDTypes(layer,object);
    readEnum(layer,object,"activation",&Layer::activation,ActivationType::SOFTSIGN);
    readEnum(layer,object,"branch_type",&Layer::branch_type,BranchType::MERGE);
    readEnum(layer,object,"merge_op",&Layer::merge_op,MergeOperation::ATTENTION_WEIGHTED);
    layer.type=LayerRegistry::normalize_type(layer.type);
    layer.type_enum=LayerRegistry::string_to_type(layer.type);
}
inline bool complete(const nlohmann::json& object) {
    return object.is_object() && object.contains("metadata_version") &&
        object.at("metadata_version").is_number_integer() && object.at("metadata_version")==version &&
        containsFields(object,spatialIntegers()) && containsFields(object,commonIntegers()) &&
        containsFields(object,commonFloats()) && containsFields(object,commonBooleans()) &&
        containsFields(object,spatialVectors()) && containsFields(object,commonIntegerVectors()) &&
        containsFields(object,commonFloatVectors()) && containsFields(object,commonStrings()) &&
        containsFields(object,commonStringVectors()) && containsFields(object,commonSizes()) &&
        containsFields(object,commonDTypes()) && object.contains("activation") &&
        object.contains("branch_type") && object.contains("merge_op");
}
inline bool completeArchitecture(const nlohmann::json& architecture) {
    if(!architecture.is_object() || !architecture.contains("layer_metadata_version") ||
       !architecture.at("layer_metadata_version").is_number_integer() || architecture.at("layer_metadata_version")!=version || !architecture.contains("layers") ||
       !architecture.at("layers").is_array())return false;
    for(const auto& entry:architecture.at("layers"))if(!complete(entry))return false;
    return true;
}

// Retain the ND-only helpers used by DebugJSON and prebuilt legacy loaders.
// A caller-provided non-ND topology keeps its established override semantics.
inline void writeConvolutionND(nlohmann::json& object,const Layer& layer) {
    if(!convolutionND(layer))return;
    writeFields(object,layer,spatialIntegers());writeFields(object,layer,spatialVectors());
    object["use_bias"]=layer.use_bias;object["activation"]=static_cast<int>(layer.activation);
    object["activation_param"]=layer.activation_param;
}
inline void readConvolutionND(Layer& layer,const nlohmann::json& object) {
    if(!convolutionND(layer))return;
    readFields(layer,object,spatialIntegers());readFields(layer,object,spatialVectors());
    if(object.contains("use_bias"))layer.use_bias=object.at("use_bias").get<bool>();
    readEnum(layer,object,"activation",&Layer::activation,ActivationType::SOFTSIGN);
    if(object.contains("activation_param"))layer.activation_param=object.at("activation_param").get<float>();
    if(object.contains("inputs"))layer.inputs=object.at("inputs").get<std::vector<std::string>>();
    if(object.contains("output"))layer.output=object.at("output").get<std::string>();
    readDTypes(layer,object);
}
} // namespace Mimir::Serialization::LayerMetadata
