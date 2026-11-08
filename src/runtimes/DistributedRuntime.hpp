#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace Mimir {

// Process-wide collective context used by data-parallel training.
// MPI is optional at build time; without it initialize() gives an actionable error.
class DistributedRuntime {
public:
    static DistributedRuntime& instance();

    bool initialize();
    void shutdown();
    bool initialized() const { return initialized_; }
    int rank() const { return rank_; }
    int worldSize() const { return world_size_; }
    bool isMainProcess() const { return rank_ == 0; }
    void barrier();
    void requireIdentical(uint64_t signature, const char* subject);
    void broadcast(std::vector<float>& values, int root = 0);
    void allReduceMax(std::vector<int>& values);
    void allReduceMean(std::vector<float>& values);

private:
    DistributedRuntime() = default;
    ~DistributedRuntime();
    DistributedRuntime(const DistributedRuntime&) = delete;
    DistributedRuntime& operator=(const DistributedRuntime&) = delete;

    bool initialized_ = false;
    bool owns_mpi_ = false;
    int rank_ = 0;
    int world_size_ = 1;
};

} // namespace Mimir
