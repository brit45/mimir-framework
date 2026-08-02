#pragma once

#include <vector>

#include "Layers.hpp"

namespace RuntimeLayerOps {

bool resolveUnaryOp(LayerType type, const Layer& layer, int& op_code, float& alpha);
bool resolveBinaryOp(LayerType type, int& op_code);

void unaryForwardHost(const std::vector<float>& input, std::vector<float>& output, int op_code, float alpha);
void binaryForwardHost(const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& output, int op_code);

} // namespace RuntimeLayerOps
