#ifndef __TENSOR_VIZ_TEXT_PAYLOAD_HPP__
#define __TENSOR_VIZ_TEXT_PAYLOAD_HPP__

#include "Model.hpp"
#include "PromptParsing.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace VizTextPayload {

struct Payload {
    std::string prompt;
    std::string tags;
    std::string tokens;
    std::string encoding;
};

inline std::string escapeTokenPiece(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        switch (c) {
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default: out.push_back(c); break;
        }
    }
    return out;
}

inline std::string formatTokens(const Tokenizer* tok, const std::vector<int>& token_ids, size_t max_tokens = 24) {
    if (!tok || token_ids.empty()) return std::string();

    std::ostringstream oss;
    const size_t n = std::min(max_tokens, token_ids.size());
    for (size_t i = 0; i < n; ++i) {
        if (i > 0) oss << " | ";
        const int tid = token_ids[i];
        std::string piece;
        try {
            piece = tok->getTokenById(tid);
        } catch (...) {
            piece.clear();
        }
        piece = escapeTokenPiece(piece);
        if (piece.size() > 20) piece = piece.substr(0, 17) + "...";
        oss << tid;
        if (!piece.empty()) oss << ":" << piece;
    }
    if (token_ids.size() > max_tokens) {
        oss << " | ...("
            << (token_ids.size() - max_tokens)
            << " more)";
    }
    return oss.str();
}

inline std::string formatEncoding(const ConditioningEncoder* enc, const std::vector<int>& token_ids, size_t max_head = 8) {
    if (!enc || token_ids.empty()) return std::string();

    try {
        std::vector<float> values;
        enc->encodeInto(values, token_ids);
        if (values.empty()) return "dim=0";

        double sum = 0.0;
        double n2 = 0.0;
        float vmin = values[0];
        float vmax = values[0];
        for (float x : values) {
            sum += static_cast<double>(x);
            n2 += static_cast<double>(x) * static_cast<double>(x);
            if (x < vmin) vmin = x;
            if (x > vmax) vmax = x;
        }

        const double mean = sum / static_cast<double>(std::max<size_t>(1, values.size()));
        const double l2 = std::sqrt(std::max(0.0, n2));

        std::ostringstream oss;
        oss << std::fixed << std::setprecision(4);
        oss << "dim=" << values.size()
            << " mean=" << mean
            << " min=" << vmin
            << " max=" << vmax
            << " l2=" << l2
            << " head=[";

        const size_t hn = std::min(max_head, values.size());
        for (size_t i = 0; i < hn; ++i) {
            if (i > 0) oss << ",";
            oss << values[i];
        }
        if (values.size() > max_head) oss << ",...";
        oss << "]";
        return oss.str();
    } catch (...) {
        return "encode_error";
    }
}

inline Payload buildDatasetTextPayload(
    const Model* model,
    const Tokenizer* tok,
    const std::string& prompt,
    const std::vector<int>* ids_hint,
    int pad_id,
    bool caption_kv_enable = false,
    bool caption_structured_enable = true,
    bool caption_structured_canonicalize = true
) {
    Payload out;

    PromptParsing::ParseOptions options;
    options.kv_enable = caption_kv_enable;
    options.structured_enable = caption_structured_enable;
    options.structured_canonicalize = caption_structured_canonicalize;

    const PromptParsing::PromptAnalysis analysis = PromptParsing::analyzePrompt(prompt, options);
    out.prompt = analysis.display_prompt.empty() ? prompt : analysis.display_prompt;
    out.tags = analysis.tags;

    if (!tok || out.prompt.empty()) return out;

    std::vector<int> ids;
    if (ids_hint && !ids_hint->empty()) {
        ids = *ids_hint;
    } else {
        try {
            ids = tok->tokenize(out.prompt);
        } catch (...) {
            ids.clear();
        }
    }

    if (pad_id >= 0) {
        while (!ids.empty() && ids.back() == pad_id) ids.pop_back();
    }

    out.tokens = formatTokens(tok, ids);

    const ConditioningEncoder* enc = nullptr;
    if (model) enc = &model->getEncoder();
    out.encoding = formatEncoding(enc, ids);
    return out;
}

} // namespace VizTextPayload

#endif