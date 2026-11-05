#pragma once
#include "runtimes/AbstractRuntime.hpp"
#include <functional>
#include <limits>
#include <string>
#include <vector>

namespace Mimir::Planning::FusionCalibration {
using Buffers=std::vector<std::vector<float>>;
struct Candidate {
    AbstractRuntime* runtime=nullptr; // nullptr denotes the ordinary routed chain
    std::function<bool(Buffers&)> execute;
    int cpu_workers=1;
};
struct Request {
    std::vector<const Layer*> layers;
    const std::vector<float>* input=nullptr;
    std::vector<const std::vector<float>*> snapshots;
    std::vector<AbstractRuntime*> available;
    uint64_t generation=0;
    bool backward=false,training=false,retain=false;
    Mimir::DType dtype=Mimir::DType::F32;
};
struct Alternative {std::string runtime;double seconds=0;bool exact=false;int cpu_workers=1;};
struct Decision {
    size_t key=0;
    bool measured=false, exact=false, fused=false, cache_hit=false;
    AbstractRuntime* runtime=nullptr;
    int cpu_workers=1;
    double unfused_seconds=0,fused_seconds=std::numeric_limits<double>::infinity();
    std::string reason,selected_runtime;
    std::vector<Alternative> alternatives;
};
// These are legal program candidates, not an execution choice. Only select()
// approves a fused kernel, after real host-to-host timing and exact parity.
bool pointwiseCandidate(LayerType type);
// Candidate execution callbacks may be empty for a cached lookup. Their
// runtime/worker metadata still participates in the complete decision key.
bool lookup(const Request&,const std::vector<Candidate>&,Decision&);
Decision select(const Request&,const std::vector<Candidate>&);
void invalidate(size_t key);
void clearProfiles();
const Decision* profile(const Layer* first,bool backward);
}
