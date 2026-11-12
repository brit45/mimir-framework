// Compile independently of mimir_core's GUI defines. AsyncMonitor must be able
// to publish the same metrics API in a no-GUI build, including on Windows.
#include "AsyncMonitor.hpp"
#include <type_traits>
#include <utility>
static_assert(std::is_same_v<decltype(std::declval<Visualizer&>().setModelMetrics(
    std::declval<const std::map<std::string,double>&>())),void>);
