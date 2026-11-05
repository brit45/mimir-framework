#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace Mimir::Planning::ExecutionPolicy
{
    // Zero means automatic latency-based selection, not a hidden size cutoff.
    inline constexpr int automatic_minimum = 0;
    enum class Work
    {
        Linear,
        Convolution,
        Normalization,
        Attention
    };
    enum class CpuAlgorithm
    {
        Reference,
        PointwiseConvolution
    };
    struct CpuChoice
    {
        int workers = 1;
        CpuAlgorithm algorithm = CpuAlgorithm::Reference;
    };
    CpuChoice cpuChoice();
    class CpuScope
    {
        CpuChoice previous_;

    public:
        explicit CpuScope(int workers = 1, CpuAlgorithm algorithm = CpuAlgorithm::Reference);
        ~CpuScope();
        CpuScope(const CpuScope &) = delete;
        CpuScope &operator=(const CpuScope &) = delete;
    };
    int cpuWorkers(size_t independent_work);
    std::vector<int> cpuCandidates(bool ordered_reduction = false);
    std::vector<CpuAlgorithm> cpuAlgorithms(bool pointwise_convolution);
    int configuredMinimum(const char *backend, Work kind);
    template <class WorkSize>
    inline bool runtimeWorkAllowed(WorkSize work, int explicit_minimum)
    {
        if constexpr (std::is_signed<WorkSize>::value)
            if (work < 0)
                return false;
        return explicit_minimum <= automatic_minimum || uint64_t(work) >= uint64_t(explicit_minimum);
    }
    // These are protection/geometry bounds, never speed-selection thresholds.
    // The portable serial path must remain bounded against GPU watchdog stalls.
    struct Limits
    {
        static constexpr size_t portable_serial_elements = 65536;
        static constexpr uint64_t portable_serial_work = 4 * 1024 * 1024;
        static constexpr size_t portable_parallel_invocations = 16 * 1024 * 1024;
        static constexpr size_t portable_staged_elements = 8 * 1024 * 1024;
        static constexpr size_t portable_parallel_elements = 64 * 1024 * 1024;
        static constexpr uint64_t portable_parallel_work = 128 * 1024 * 1024;
        static constexpr uint64_t portable_reduction_span = 1024 * 1024;
        static constexpr uint64_t portable_scatter_span = 4096;
        static constexpr size_t calibration_cache_entries = 2048;
        static constexpr size_t calibration_refresh = 1024;
        static constexpr size_t pointwise_tile_elements = 512; // SIMD cache tile, not an eligibility gate
    };
    size_t calibrationBudget();
    size_t cacheBytes();
    int convolutionTileRows(size_t reduction, size_t channels, int spatial);
}
