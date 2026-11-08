#include "runtimes/DistributedRuntime.hpp"

#include <algorithm>
#include <climits>
#include <stdexcept>
#include <string>

#ifdef MIMIR_HAS_MPI
#include <mpi.h>
#endif

namespace Mimir {

DistributedRuntime& DistributedRuntime::instance() {
    static DistributedRuntime runtime;
    return runtime;
}

DistributedRuntime::~DistributedRuntime() { shutdown(); }

bool DistributedRuntime::initialize() {
    if (initialized_) return true;
#ifndef MIMIR_HAS_MPI
    throw std::runtime_error(
        "DDP is unavailable: rebuild Mimir with -DENABLE_MPI=ON and an MPI C++ implementation");
#else
    int was_initialized = 0;
    MPI_Initialized(&was_initialized);
    if (!was_initialized) {
        int argc = 0;
        char** argv = nullptr;
        int provided = MPI_THREAD_SINGLE;
        const int rc = MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
        if (rc != MPI_SUCCESS)
            throw std::runtime_error("MPI_Init_thread failed while starting Mimir DDP");
        owns_mpi_ = true;
        if (provided < MPI_THREAD_FUNNELED) {
            MPI_Finalize();
            owns_mpi_ = false;
            throw std::runtime_error("MPI implementation does not provide MPI_THREAD_FUNNELED");
        }
    }
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
    if (MPI_Comm_rank(MPI_COMM_WORLD, &rank_) != MPI_SUCCESS ||
        MPI_Comm_size(MPI_COMM_WORLD, &world_size_) != MPI_SUCCESS) {
        if (owns_mpi_) MPI_Finalize();
        owns_mpi_ = false;
        throw std::runtime_error("Could not query the MPI world for Mimir DDP");
    }
    if (world_size_ < 1) {
        if (owns_mpi_) MPI_Finalize();
        owns_mpi_ = false;
        rank_ = 0;
        world_size_ = 1;
        throw std::runtime_error("MPI returned an invalid world size");
    }
    initialized_ = true;
    return true;
#endif
}

void DistributedRuntime::shutdown() {
    if (!initialized_) return;
#ifdef MIMIR_HAS_MPI
    int finalized = 0;
    MPI_Finalized(&finalized);
    if (owns_mpi_ && !finalized) MPI_Finalize();
#endif
    initialized_ = false;
    owns_mpi_ = false;
    rank_ = 0;
    world_size_ = 1;
}

void DistributedRuntime::barrier() {
    if (!initialized_) throw std::runtime_error("DDP is not initialized");
#ifdef MIMIR_HAS_MPI
    if (MPI_Barrier(MPI_COMM_WORLD) != MPI_SUCCESS)
        throw std::runtime_error("MPI_Barrier failed");
#endif
}

void DistributedRuntime::requireIdentical(uint64_t signature, const char* subject) {
    if (!initialized_) throw std::runtime_error("DDP is not initialized");
    if (world_size_ <= 1) return;
#ifdef MIMIR_HAS_MPI
    unsigned long long local = static_cast<unsigned long long>(signature);
    unsigned long long minimum = 0, maximum = 0;
    if (MPI_Allreduce(&local, &minimum, 1, MPI_UNSIGNED_LONG_LONG, MPI_MIN,
                      MPI_COMM_WORLD) != MPI_SUCCESS ||
        MPI_Allreduce(&local, &maximum, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX,
                      MPI_COMM_WORLD) != MPI_SUCCESS)
        throw std::runtime_error("MPI_Allreduce failed while checking DDP topology");
    if (minimum != maximum)
        throw std::runtime_error(std::string("DDP ") + (subject ? subject : "topology") +
                                 " differs between ranks");
#else
    (void)signature;
    (void)subject;
#endif
}

void DistributedRuntime::broadcast(std::vector<float>& values, const int root) {
    if (!initialized_) throw std::runtime_error("DDP is not initialized");
    if (root < 0 || root >= world_size_)
        throw std::invalid_argument("DDP broadcast root is outside the process group");
    if (world_size_ <= 1 || values.empty()) return;
#ifdef MIMIR_HAS_MPI
    size_t offset = 0;
    while (offset < values.size()) {
        const size_t remaining = values.size() - offset;
        const int count = static_cast<int>(std::min(remaining,
            static_cast<size_t>(INT_MAX)));
        if (MPI_Bcast(values.data() + offset, count, MPI_FLOAT, root,
                      MPI_COMM_WORLD) != MPI_SUCCESS)
            throw std::runtime_error("MPI_Bcast failed while synchronizing DDP parameters");
        offset += static_cast<size_t>(count);
    }
#else
    (void)root;
#endif
}

void DistributedRuntime::allReduceMax(std::vector<int>& values) {
    if (!initialized_) throw std::runtime_error("DDP is not initialized");
    if (world_size_ <= 1 || values.empty()) return;
#ifdef MIMIR_HAS_MPI
    size_t offset = 0;
    while (offset < values.size()) {
        const size_t remaining = values.size() - offset;
        const int count = static_cast<int>(std::min(remaining,
            static_cast<size_t>(INT_MAX)));
        if (MPI_Allreduce(MPI_IN_PLACE, values.data() + offset, count, MPI_INT,
                          MPI_MAX, MPI_COMM_WORLD) != MPI_SUCCESS)
            throw std::runtime_error("MPI_Allreduce failed while synchronizing DDP gradient presence");
        offset += static_cast<size_t>(count);
    }
#endif
}

void DistributedRuntime::allReduceMean(std::vector<float>& values) {
    if (!initialized_) throw std::runtime_error("DDP is not initialized");
    if (world_size_ <= 1 || values.empty()) return;
#ifdef MIMIR_HAS_MPI
    size_t offset = 0;
    while (offset < values.size()) {
        const size_t remaining = values.size() - offset;
        const int count = static_cast<int>(std::min(remaining,
            static_cast<size_t>(INT_MAX)));
        if (MPI_Allreduce(MPI_IN_PLACE, values.data() + offset, count, MPI_FLOAT,
                          MPI_SUM, MPI_COMM_WORLD) != MPI_SUCCESS)
            throw std::runtime_error("MPI_Allreduce failed while synchronizing DDP gradients");
        offset += static_cast<size_t>(count);
    }
    const float scale = 1.0f / static_cast<float>(world_size_);
    for (float& value : values) value *= scale;
#endif
}

} // namespace Mimir
