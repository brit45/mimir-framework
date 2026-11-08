#pragma once

#include "../Layers.hpp"
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

// Classification and checked tensor metadata only; no tensor arithmetic.
namespace RuntimePrimer {
inline int unaryCode(LayerType type) {
    switch (type) {
    case LayerType::SquaredReLU: return 0;
    case LayerType::Square: return 1;
    case LayerType::SqrtAbs: return 2;
    case LayerType::Exp: return 3;
    case LayerType::LogAbs: return 4;
    case LayerType::Abs: return 5;
    case LayerType::ReciprocalNoNan: return 6;
    case LayerType::Sign: return 7;
    case LayerType::Sin: return 8;
    case LayerType::Cos: return 9;
    default: return -1;
    }
}
inline int binaryCode(LayerType type) {
    switch (type) {
    case LayerType::Maximum: return 0;
    case LayerType::Minimum: return 1;
    default: return -1;
    }
}
inline int reductionCode(LayerType type) {
    switch (type) {
    case LayerType::ReduceSum: return 0;
    case LayerType::ReduceMean: return 1;
    case LayerType::ReduceMin: return 2;
    case LayerType::ReduceMax: return 3;
    case LayerType::ReduceProd: return 4;
    default: return -1;
    }
}
inline bool lowerTriangularType(LayerType type) {
    return type == LayerType::LowerTriangular;
}
inline bool supported(LayerType type) {
    return unaryCode(type) >= 0 || binaryCode(type) >= 0 ||
        reductionCode(type) >= 0 || lowerTriangularType(type);
}
inline size_t checkedProduct(size_t first, size_t second) {
    constexpr size_t limit = static_cast<size_t>(std::numeric_limits<std::ptrdiff_t>::max()) / sizeof(float);
    if (first && second > limit / first)
        throw std::invalid_argument("Primer: tensor dimensions exceed storage bounds");
    return first * second;
}
inline void checkedStorage(size_t count) {
    if (!count) throw std::invalid_argument("Primer: empty tensors are unsupported");
    (void)checkedProduct(count, 1);
}

struct Reduction {
    size_t outer = 0;
    size_t axis_size = 0;
    size_t inner = 0;
    size_t input_size = 0;
    size_t output_size = 0;
    std::vector<int> output_shape;
};
inline Reduction checkedReduction(const Layer& layer, size_t input_size) {
    if (reductionCode(layer.type_enum) < 0)
        throw std::invalid_argument("Primer reduction: unexpected layer type");
    Reduction geometry;
    geometry.input_size = input_size;
    if (layer.shape.empty()) {
        checkedStorage(input_size);
        if (layer.axis != 0 && layer.axis != -1)
            throw std::invalid_argument("Primer reduction: a flat vector has only axis 0/-1");
        geometry.outer = geometry.inner = geometry.output_size = 1;
        geometry.axis_size = input_size;
        geometry.output_shape = {1};
        return geometry;
    }
    const long long rank = static_cast<long long>(layer.shape.size());
    long long axis = layer.axis;
    if (axis < 0) axis += rank;
    if (axis < 0 || axis >= rank)
        throw std::invalid_argument("Primer reduction: axis outside tensor rank");
    size_t count = 1;
    geometry.outer = geometry.inner = 1;
    geometry.output_shape = layer.shape;
    for (size_t index = 0; index < layer.shape.size(); ++index) {
        const int dimension = layer.shape[index];
        if (dimension <= 0)
            throw std::invalid_argument("Primer reduction: nonpositive tensor dimension");
        count = checkedProduct(count, static_cast<size_t>(dimension));
        if (index < static_cast<size_t>(axis))
            geometry.outer = checkedProduct(geometry.outer, static_cast<size_t>(dimension));
        else if (index > static_cast<size_t>(axis))
            geometry.inner = checkedProduct(geometry.inner, static_cast<size_t>(dimension));
        else geometry.axis_size = static_cast<size_t>(dimension);
    }
    if (input_size && count != input_size)
        throw std::invalid_argument("Primer reduction: shape differs from input storage");
    geometry.input_size = count;
    geometry.output_shape[static_cast<size_t>(axis)] = 1;
    geometry.output_size = checkedProduct(geometry.outer, geometry.inner);
    return geometry;
}

struct Triangle {
    size_t batch = 0;
    size_t rows = 0;
    size_t columns = 0;
    size_t input_size = 0;
};
inline Triangle checkedTriangle(const Layer& layer, size_t input_size) {
    if (!lowerTriangularType(layer.type_enum) || layer.shape.size() < 2)
        throw std::invalid_argument("LowerTriangular: explicit tensor shape of rank >=2 required");
    Triangle geometry;
    geometry.batch = 1;
    geometry.input_size = input_size;
    const size_t rank = layer.shape.size();
    for (size_t index = 0; index < rank; ++index) {
        if (layer.shape[index] <= 0)
            throw std::invalid_argument("LowerTriangular: nonpositive tensor dimension");
        const size_t dimension = static_cast<size_t>(layer.shape[index]);
        if (index + 2 < rank) geometry.batch = checkedProduct(geometry.batch, dimension);
        else if (index + 2 == rank) geometry.rows = dimension;
        else geometry.columns = dimension;
    }
    const size_t count = checkedProduct(checkedProduct(geometry.batch, geometry.rows), geometry.columns);
    if (input_size && count != input_size)
        throw std::invalid_argument("LowerTriangular: shape differs from input storage");
    geometry.input_size = count;
    return geometry;
}
} // namespace RuntimePrimer
