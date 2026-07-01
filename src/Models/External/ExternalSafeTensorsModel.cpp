#include "ExternalSafeTensorsModel.hpp"

#include "../../DType.hpp"

#include <algorithm>
#include <fstream>
#include <stdexcept>

namespace {

using json = nlohmann::json;

struct ParsedTensorHeader {
    std::string name;
    Mimir::DType dtype = Mimir::DType::UNKNOWN;
    std::vector<size_t> shape;
};

static uint64_t read_u64_le(std::ifstream& f) {
    uint8_t bytes[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    f.read(reinterpret_cast<char*>(bytes), 8);
    uint64_t value = 0;
    for (int i = 7; i >= 0; --i) {
        value = (value << 8) | static_cast<uint64_t>(bytes[i]);
    }
    return value;
}

static std::vector<ParsedTensorHeader> parse_safetensors_header(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open SafeTensors file: " + path);
    }

    const uint64_t header_len = read_u64_le(file);
    if (!file || header_len == 0 || header_len > 100ULL * 1024ULL * 1024ULL) {
        throw std::runtime_error("Invalid SafeTensors header length: " + path);
    }

    std::vector<char> header_data(static_cast<size_t>(header_len));
    file.read(header_data.data(), static_cast<std::streamsize>(header_data.size()));
    if (!file) {
        throw std::runtime_error("Failed to read SafeTensors header: " + path);
    }

    const json header = json::parse(std::string(header_data.begin(), header_data.end()));
    std::vector<ParsedTensorHeader> tensors;
    tensors.reserve(header.size());
    for (auto it = header.begin(); it != header.end(); ++it) {
        if (it.key() == "__metadata__") continue;
        if (!it.value().is_object()) continue;
        if (!it.value().contains("shape") || !it.value().contains("dtype")) continue;

        ParsedTensorHeader tensor;
        tensor.name = it.key();
        tensor.dtype = Mimir::parse_dtype_safetensors(it.value()["dtype"].get<std::string>());
        tensor.shape = it.value()["shape"].get<std::vector<size_t>>();
        tensors.push_back(std::move(tensor));
    }
    return tensors;
}

static size_t safe_numel(const std::vector<size_t>& shape) {
    size_t total = 1;
    for (size_t dim : shape) {
        if (dim == 0) return 0;
        if (total > (static_cast<size_t>(-1) / dim)) {
            throw std::runtime_error("Tensor shape overflow while computing numel");
        }
        total *= dim;
    }
    return total;
}

static bool starts_with_any(const std::string& s, const std::vector<std::string>& prefixes) {
    if (prefixes.empty()) return true;
    for (const std::string& p : prefixes) {
        if (!p.empty() && s.rfind(p, 0) == 0) return true;
    }
    return false;
}

static bool excluded_by_any(const std::string& s, const std::vector<std::string>& prefixes) {
    for (const std::string& p : prefixes) {
        if (!p.empty() && s.rfind(p, 0) == 0) return true;
    }
    return false;
}

} // namespace

ExternalSafeTensorsModel::ExternalSafeTensorsModel() {
    setModelName("ExternalSafeTensorsBase");
}

void ExternalSafeTensorsModel::buildFromConfig(const Config& cfg) {
    cfg_ = cfg;
    buildInto(*this, cfg_);
}

void ExternalSafeTensorsModel::buildInto(Model& model, const Config& cfg) {
    if (cfg.source_safetensors.empty()) {
        throw std::runtime_error("ExternalSafeTensorsModel: source_safetensors is required");
    }

    const auto tensors = parse_safetensors_header(cfg.source_safetensors);

    model.getMutableLayers().clear();
    model.setModelName("ExternalSafeTensorsBase");
    model.modelConfig = json::object();
    model.modelConfig["type"] = "external_safetensors_base";
    model.modelConfig["source_safetensors"] = cfg.source_safetensors;
    model.modelConfig["include_prefixes"] = cfg.include_prefixes;
    model.modelConfig["exclude_prefixes"] = cfg.exclude_prefixes;
    model.modelConfig["max_tensors"] = cfg.max_tensors;

    int selected_count = 0;
    size_t selected_params = 0;
    for (const ParsedTensorHeader& tensor : tensors) {
        if (!starts_with_any(tensor.name, cfg.include_prefixes)) continue;
        if (excluded_by_any(tensor.name, cfg.exclude_prefixes)) continue;
        if (cfg.max_tensors > 0 && selected_count >= cfg.max_tensors) break;

        const size_t params_count = safe_numel(tensor.shape);
        model.push(tensor.name, "Identity", params_count);
        if (auto* layer = model.getLayerByName(tensor.name)) {
            layer->inputs = {"__input__"};
            layer->output = tensor.name;
        }
        selected_params += params_count;
        ++selected_count;
    }

    if (selected_count <= 0) {
        throw std::runtime_error("ExternalSafeTensorsModel: no tensor selected from " + cfg.source_safetensors);
    }

    model.push("external_safetensors_base/out", "Identity", 0);
    if (auto* layer = model.getLayerByName("external_safetensors_base/out")) {
        layer->inputs = {"__input__"};
        layer->output = "x";
    }

    model.modelConfig["selected_tensors"] = selected_count;
    model.modelConfig["selected_params"] = selected_params;
}