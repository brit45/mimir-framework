#pragma once

#include "Helpers.hpp"
#include <memory>

class Tokenizer;

// Host-side decoding and deterministic conversion only. Workers never access
// Lua, model state, augmentation RNG, or mutable tokenizer vocabulary.
class DatasetPrefetcher {
    struct State;
public:
    enum class Format { Text, RGB, HWC, CHW, FloatCHW, MediaHWC };
    struct Options {
        int width=64,height=64;
        size_t ahead=8, max_bytes=64ULL*1024*1024;
        std::shared_ptr<const Tokenizer> tokenizer;
        size_t minimum_tokens=0;
        int pad_id=0;
        static Options fromEnvironment(int width,int height);
    };
    struct Sample {
        DatasetItem item;
        std::vector<float> input;
        std::vector<int> tokens;
        size_t index=0;
        bool tokenized=false;
        std::string error;
    };
    struct Stats {
        size_t capacity=0,queued=0,ready=0,leased=0,loading=0,resident_bytes=0,peak_bytes=0;
        size_t requests=0,hits=0,waits=0,prepared=0,recycled=0,errors=0;
        size_t buffer_growths=0;
        double wait_ms=0,prepare_ms=0;
        bool enabled=false;
    };
    class Lease {
        friend class DatasetPrefetcher;
        std::shared_ptr<State> state_;
        size_t slot_=0;
        Lease(std::shared_ptr<State> state,size_t slot);
    public:
        Lease()=default;
        ~Lease();
        Lease(const Lease&)=delete;
        Lease& operator=(const Lease&)=delete;
        Lease(Lease&&) noexcept;
        Lease& operator=(Lease&&) noexcept;
        explicit operator bool() const { return bool(state_); }
        Sample& operator*() const;
        Sample* operator->() const;
        void reset() noexcept;
    };

    // Sources are metadata snapshots. Reloading/destroying the original vector
    // cannot invalidate a running decoder or a leased sample.
    DatasetPrefetcher(const std::vector<DatasetItem>& items,Options options);
    ~DatasetPrefetcher();
    DatasetPrefetcher(const DatasetPrefetcher&)=delete;
    DatasetPrefetcher& operator=(const DatasetPrefetcher&)=delete;
    Lease acquire(size_t index,Format format=Format::RGB);
    void setOrder(std::vector<size_t> order,Format format=Format::RGB);
    void invalidate(size_t index);
    Stats stats() const;
    bool matchesDimensions(int width,int height) const;
    bool preparesTokens() const;
    void stop() noexcept;
private:
    std::shared_ptr<State> state_;
};
