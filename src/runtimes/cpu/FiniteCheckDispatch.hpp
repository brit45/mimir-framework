#pragma once
#include <cstddef>

namespace Mimir::RuntimeChecks {
// Exact host-buffer classification with runtime ISA selection. IEEE exponent
// bits detect NaN/Inf without arithmetic, including signaling NaNs. Non-IEEE
// builds use std::isfinite; short views use the scalar implementation.
bool allFiniteDispatched(const float* values,std::size_t count);
}
