#pragma once

#include "../PrimerGeometry.hpp"
#include "../FiniteCheck.hpp"
#include "Parallel.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace CpuPrimer {
inline void requireFinite(const std::vector<float>& values) {
    if (!Mimir::RuntimeChecks::allFinite(values.data(), values.size()))
        throw std::invalid_argument("Primer: non-finite tensor/result");
}
inline void requireInput(const std::vector<float>& values) {
    RuntimePrimer::checkedStorage(values.size());
    requireFinite(values);
}
inline void requireGradient(const std::vector<float>& input, const std::vector<float>& gradient) {
    requireInput(input);
    if (input.size() != gradient.size())
        throw std::invalid_argument("Primer: gradient shape mismatch");
    requireFinite(gradient);
}
inline size_t workEstimate(size_t first, size_t second) {
    return first && second > std::numeric_limits<size_t>::max()/first
        ? std::numeric_limits<size_t>::max() : first*second;
}
inline float sign(float value) {
    return value > 0.f ? 1.f : (value < 0.f ? -1.f : 0.f);
}
inline void validateUnaryDomain(const std::vector<float>& input, int code, bool backward) {
    if (code < 0 || code > 9) throw std::invalid_argument("Primer: unknown unary operation");
    if (code == 4 || (backward && code == 2))
        for (float value : input) if (value == 0.f)
            throw std::invalid_argument("Primer: singular logarithm/square-root derivative at zero");
}
inline void unaryForward(const std::vector<float>& input, std::vector<float>& output, int code) {
    requireInput(input); validateUnaryDomain(input, code, false);
    std::vector<float> result(input.size());
    RuntimeParallel::forEach(input.size(), RuntimeParallel::workers(input.size()), [&](size_t index) {
        const float x = input[index];
        switch (code) {
        case 0: result[index] = x > 0.f ? x*x : 0.f; break;
        case 1: result[index] = x*x; break;
        case 2: result[index] = std::sqrt(std::abs(x)); break;
        case 3: result[index] = std::exp(x); break;
        case 4: result[index] = std::log(std::abs(x)); break;
        case 5: result[index] = std::abs(x); break;
        case 6: result[index] = x == 0.f ? 0.f : 1.f/x; break;
        case 7: result[index] = sign(x); break;
        case 8: result[index] = std::sin(x); break;
        case 9: result[index] = std::cos(x); break;
        }
    });
    requireFinite(result);
    output = std::move(result);
}
inline void unaryBackward(const std::vector<float>& input, const std::vector<float>& grad_output,
                          std::vector<float>& grad_input, int code) {
    requireGradient(input, grad_output); validateUnaryDomain(input, code, true);
    std::vector<float> result(input.size());
    RuntimeParallel::forEach(input.size(), RuntimeParallel::workers(input.size()), [&](size_t index) {
        const float x = input[index], dy = grad_output[index];
        switch (code) {
        case 0: result[index] = x > 0.f ? dy*(2.f*x) : 0.f; break;
        case 1: result[index] = dy*(2.f*x); break;
        case 2: result[index] = ((0.5f*dy)*sign(x))/std::sqrt(std::abs(x)); break;
        case 3: result[index] = dy*std::exp(x); break;
        case 4: result[index] = dy/x; break;
        case 5: result[index] = dy*sign(x); break;
        case 6: result[index] = x == 0.f ? 0.f : -(dy/x)/x; break;
        case 7: result[index] = 0.f; break;
        case 8: result[index] = dy*std::cos(x); break;
        case 9: result[index] = (-dy)*std::sin(x); break;
        }
    });
    requireFinite(result);
    grad_input = std::move(result);
}

inline size_t checkedBinary(const std::vector<float>& first, const std::vector<float>& second, int code) {
    if (code < 0 || code > 1) throw std::invalid_argument("Primer: unknown binary operation");
    requireInput(first); requireInput(second);
    if (first.size() != second.size() && first.size() != 1 && second.size() != 1)
        throw std::invalid_argument("Primer maximum/minimum: shapes must match or one input must be scalar");
    return std::max(first.size(), second.size());
}
inline void binaryForward(const std::vector<float>& first, const std::vector<float>& second,
                          std::vector<float>& output, int code) {
    const size_t count = checkedBinary(first, second, code);
    std::vector<float> result(count);
    RuntimeParallel::forEach(count, RuntimeParallel::workers(count), [&](size_t index) {
        const float a = first[first.size() == 1 ? 0 : index];
        const float b = second[second.size() == 1 ? 0 : index];
        result[index] = code == 0 ? (a >= b ? a : b) : (a <= b ? a : b);
    });
    requireFinite(result);
    output = std::move(result);
}
inline float binaryGradient(float first, float second, float gradient, int code, bool first_input) {
    if (first == second) return gradient*0.5f;
    const bool choose_first = code == 0 ? first > second : first < second;
    return choose_first == first_input ? gradient : 0.f;
}
inline void binaryBackward(const std::vector<float>& first, const std::vector<float>& second,
                           const std::vector<float>& grad_output,
                           std::vector<float>& grad_first, std::vector<float>& grad_second, int code) {
    const size_t count = checkedBinary(first, second, code);
    if (grad_output.size() != count)
        throw std::invalid_argument("Primer maximum/minimum: gradient shape mismatch");
    requireFinite(grad_output);
    std::vector<float> first_result(first.size(), 0.f), second_result(second.size(), 0.f);
    auto gather = [&](std::vector<float>& result, bool first_input) {
        RuntimeParallel::forEach(result.size(), RuntimeParallel::workers(count), [&](size_t index) {
            // Scalar adjoints own the complete increasing-index reduction.
            const size_t begin = result.size() == 1 ? 0 : index;
            const size_t end = result.size() == 1 ? count : index+1;
            float sum = 0.f;
            for (size_t position = begin; position < end; ++position) {
                const float a = first[first.size() == 1 ? 0 : position];
                const float b = second[second.size() == 1 ? 0 : position];
                sum += binaryGradient(a, b, grad_output[position], code, first_input);
            }
            result[index] = sum;
        });
    };
    gather(first_result, true); gather(second_result, false);
    requireFinite(first_result); requireFinite(second_result);
    grad_first = std::move(first_result);
    grad_second = std::move(second_result);
}

inline void reductionForward(const std::vector<float>& input, std::vector<float>& output, const Layer& layer) {
    requireInput(input);
    const auto geometry = RuntimePrimer::checkedReduction(layer, input.size());
    const int code = RuntimePrimer::reductionCode(layer.type_enum);
    std::vector<float> result(geometry.output_size);
    RuntimeParallel::forEach(result.size(), RuntimeParallel::workers(input.size()), [&](size_t index) {
        const size_t base = (index/geometry.inner)*geometry.axis_size*geometry.inner + index%geometry.inner;
        float value = code == 4 ? 1.f : ((code == 2 || code == 3) ? input[base] : 0.f);
        for (size_t axis = 0; axis < geometry.axis_size; ++axis) {
            const float x = input[base+axis*geometry.inner];
            if (code == 2) { if (x < value) value = x; }
            else if (code == 3) { if (x > value) value = x; }
            else if (code == 4) value *= x;
            else value += x;
        }
        result[index] = code == 1 ? value/static_cast<float>(geometry.axis_size) : value;
    });
    requireFinite(result);
    output = std::move(result);
}
inline void reductionBackward(const std::vector<float>& input, const std::vector<float>& grad_output,
                              std::vector<float>& grad_input, const Layer& layer) {
    requireInput(input);
    const auto geometry = RuntimePrimer::checkedReduction(layer, input.size());
    if (grad_output.size() != geometry.output_size)
        throw std::invalid_argument("Primer reduction: gradient shape mismatch");
    requireFinite(grad_output);
    const int code = RuntimePrimer::reductionCode(layer.type_enum);
    std::vector<float> extrema;
    std::vector<size_t> ties;
    if (code == 2 || code == 3) {
        reductionForward(input, extrema, layer);
        ties.resize(geometry.output_size, 0);
        RuntimeParallel::forEach(ties.size(), RuntimeParallel::workers(input.size()), [&](size_t index) {
            const size_t base = (index/geometry.inner)*geometry.axis_size*geometry.inner + index%geometry.inner;
            size_t count = 0;
            for (size_t axis = 0; axis < geometry.axis_size; ++axis)
                if (input[base+axis*geometry.inner] == extrema[index]) ++count;
            ties[index] = count;
        });
    }
    std::vector<float> result(input.size());
    const size_t work = code == 4 ? workEstimate(input.size(), geometry.axis_size) : input.size();
    RuntimeParallel::forEach(input.size(), RuntimeParallel::workers(work), [&](size_t index) {
        const size_t axis = (index/geometry.inner)%geometry.axis_size;
        const size_t outer = index/(geometry.axis_size*geometry.inner);
        const size_t inner = index%geometry.inner;
        const size_t output_index = outer*geometry.inner+inner;
        const float dy = grad_output[output_index];
        if (code == 0) result[index] = dy;
        else if (code == 1) result[index] = dy/static_cast<float>(geometry.axis_size);
        else if (code == 2 || code == 3)
            result[index] = input[index] == extrema[output_index]
                ? dy/static_cast<float>(ties[output_index]) : 0.f;
        else {
            const size_t base = outer*geometry.axis_size*geometry.inner+inner;
            float product = 1.f;
            for (size_t other = 0; other < geometry.axis_size; ++other)
                if (other != axis) product *= input[base+other*geometry.inner];
            result[index] = dy*product;
        }
    });
    requireFinite(result);
    grad_input = std::move(result);
}

inline void triangleForward(const std::vector<float>& input, std::vector<float>& output, const Layer& layer) {
    requireInput(input);
    const auto geometry = RuntimePrimer::checkedTriangle(layer, input.size());
    std::vector<float> result(input.size());
    RuntimeParallel::forEach(input.size(), RuntimeParallel::workers(input.size()), [&](size_t index) {
        const size_t matrix_index = index%(geometry.rows*geometry.columns);
        const size_t row = matrix_index/geometry.columns, column = matrix_index%geometry.columns;
        result[index] = column <= row ? input[index] : 0.f;
    });
    requireFinite(result);
    output = std::move(result);
}
inline void triangleBackward(const std::vector<float>& input, const std::vector<float>& grad_output,
                             std::vector<float>& grad_input, const Layer& layer) {
    requireGradient(input, grad_output);
    triangleForward(grad_output, grad_input, layer);
}
} // namespace CpuPrimer
