#include "DatasetPrefetcher.hpp"
#include "Tokenizer.hpp"

#include <chrono>
#include <condition_variable>
#include <cerrno>
#include <cstdlib>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace {
size_t setting(const char* name,size_t fallback) {
    const char* value=std::getenv(name);
    if(!value)return fallback;
    if(!*value || *value=='-')throw std::invalid_argument(std::string(name)+": expected a nonnegative integer");
    char* end=nullptr;errno=0;
    const auto parsed=std::strtoull(value,&end,10);
    if(errno || *end || parsed>std::numeric_limits<size_t>::max())
        throw std::invalid_argument(std::string(name)+": invalid byte/sample count");
    return size_t(parsed);
}
using Clock=std::chrono::steady_clock;
double milliseconds(Clock::time_point start) {
    return std::chrono::duration<double,std::milli>(Clock::now()-start).count();
}
}

DatasetPrefetcher::Options DatasetPrefetcher::Options::fromEnvironment(int width,int height) {
    Options result;result.width=width;result.height=height;
    result.ahead=setting("MIMIR_DATASET_PREFETCH",result.ahead);
    result.max_bytes=setting("MIMIR_DATASET_PREFETCH_BYTES",result.max_bytes);
    return result;
}

struct DatasetPrefetcher::State {
    struct Source {
        std::string name,text_file,text_inline,annotation_file,image_file,mask_file,audio_file,video_file;
        bool linked=false;
        size_t text_bytes=0;
    };
    struct Key {
        size_t index=0;Format format=Format::RGB;
        bool operator==(const Key& other) const {return index==other.index && format==other.format;}
    };
    enum class Stage { Idle,Queued,Loading,Ready,Leased,Recycle };
    struct Slot {
        Sample sample;
        Key key;
        Stage stage=Stage::Idle;
        bool urgent=false,discard=false;
        size_t resident=0;
    };
    Options options;
    std::vector<Source> sources;
    std::vector<std::unique_ptr<Slot>> slots;
    std::vector<size_t> order;
    std::vector<Key> wanted;
    size_t cursor=0,next_sequential=0;
    Format order_format=Format::RGB,sequential_format=Format::RGB;
    bool explicit_order=false,stopping=false;
    std::exception_ptr failure;
    mutable std::mutex mutex;
    std::condition_variable work,changed;
    std::thread worker;
    Stats counters;

    State(const std::vector<DatasetItem>& items,Options config):options(std::move(config)) {
        if(options.width<=0 || options.height<=0 || size_t(options.width)>std::numeric_limits<size_t>::max()/size_t(options.height)/18)
            throw std::invalid_argument("dataset prefetch: invalid image dimensions");
        if(options.ahead>256)throw std::invalid_argument("dataset prefetch: ahead must be in [0,256]");
        counters.enabled=options.ahead && options.max_bytes && !items.empty();
        if(!counters.enabled)return;
        sources.reserve(items.size());
        for(const auto& item:items) {
            Source source;
            source.name=item.name;source.text_file=item.text_file;source.text_inline=item.text_inline;
            if(source.text_file.empty() && source.text_inline.empty() && item.text)source.text_inline=*item.text;
            source.annotation_file=item.annotation_file;source.image_file=item.image_file;source.mask_file=item.mask_file;
            source.audio_file=item.audio_file;source.video_file=item.video_file;source.linked=item.is_linked;
            source.text_bytes=source.text_inline.size();
            if(source.text_inline.empty() && !source.text_file.empty()) {
                std::error_code error;const auto bytes=fs::file_size(source.text_file,error);
                if(!error && bytes<=std::numeric_limits<size_t>::max()/2)source.text_bytes=size_t(bytes);
            }
            sources.push_back(std::move(source));
        }
        // One consumer sample plus the fixed look-ahead window. Slots are
        // stable: DatasetItem's tracked optional/text addresses must not move.
        counters.capacity=options.ahead;
        for(size_t i=0;i<options.ahead+1;++i)slots.emplace_back(std::make_unique<Slot>());
        wanted.reserve(options.ahead);
        worker=std::thread([this]{run();});
    }
    ~State() {stop();for(auto& slot:slots)clean(*slot,true);}

    size_t residentLocked() const {
        size_t total=0;for(const auto& slot:slots)total+=slot->resident;return total;
    }
    size_t estimate(const Key& key) const {
        const auto& source=sources[key.index];
        size_t bytes=source.text_bytes*2;
        if(options.tokenizer) {
            const size_t tokens=std::max(source.text_bytes,options.minimum_tokens);
            if(tokens>(std::numeric_limits<size_t>::max()-bytes)/sizeof(int))return std::numeric_limits<size_t>::max();
            bytes+=tokens*sizeof(int);
        }
        if(key.format==Format::MediaHWC && !source.video_file.empty() && !source.mask_file.empty())
            bytes+=size_t(options.width)*options.height;
        if(key.format!=Format::Text && !source.image_file.empty() &&
            !(key.format==Format::MediaHWC && !source.video_file.empty())) {
            const size_t pixels=size_t(options.width)*options.height;
            bytes+=pixels*(key.format==Format::RGB?6:18);
            if(!source.mask_file.empty())bytes+=pixels;
        }
        return bytes;
    }
    bool wantedLocked(const Key& key) const {
        return std::find(wanted.begin(),wanted.end(),key)!=wanted.end();
    }
    size_t findLocked(const Key& key) const {
        for(size_t i=0;i<slots.size();++i) {
            const auto& slot=*slots[i];
            if(slot.key==key && (slot.stage==Stage::Ready || slot.stage==Stage::Queued ||
                (slot.stage==Stage::Loading && !slot.discard)))return i;
        }
        return slots.size();
    }
    size_t availableLocked(bool urgent) const {
        for(Stage stage:{Stage::Idle,Stage::Recycle,Stage::Queued,Stage::Ready})
            for(size_t i=0;i<slots.size();++i) {
                const auto& slot=*slots[i];
                if(!urgent && slot.urgent)continue; // Never replace a waiting consumer with speculative work.
                if(slot.stage==stage && (urgent || stage==Stage::Idle || stage==Stage::Recycle || !wantedLocked(slot.key)))return i;
            }
        return slots.size();
    }
    void queueLocked(size_t index,const Key& key,bool urgent) {
        auto& slot=*slots[index];slot.key=key;slot.stage=Stage::Queued;slot.urgent=urgent;slot.discard=false;
    }
    void fillLocked() {
        wanted.clear();
        for(size_t i=0;i<options.ahead;++i) {
            const size_t position=explicit_order?cursor+i:next_sequential+i;
            if(position>=(explicit_order?order.size():sources.size()))break;
            const Key key{explicit_order?order[position]:position,explicit_order?order_format:sequential_format};
            if(std::find(wanted.begin(),wanted.end(),key)==wanted.end())wanted.push_back(key);
        }
        for(auto& ptr:slots) {
            auto& slot=*ptr;
            if((slot.stage==Stage::Queued || slot.stage==Stage::Ready) && !slot.urgent && !wantedLocked(slot.key))
                slot.stage=Stage::Recycle;
        }
        for(const auto& key:wanted) {
            if(findLocked(key)!=slots.size())continue;
            bool held=false;for(const auto& slot:slots)held|=slot->stage==Stage::Leased && slot->key==key;
            if(held)continue;
            const size_t free=availableLocked(false);
            if(free==slots.size())break;
            queueLocked(free,key,false);
        }
        work.notify_one();
    }

    // Called only on the producer, except release after an explicit shutdown.
    void clean(Slot& slot,bool trim) {
        slot.sample.item.unload();slot.sample.error.clear();slot.sample.tokenized=false;
        // Keep the float view's size as well as its capacity: the next full
        // overwrite must not zero-initialize an already allocated image again.
        slot.sample.tokens.clear();
        auto& memory=DatasetMemoryManager::instance();
        if(trim) {
            memory.trackDeallocation(&slot.sample.input);memory.trackDeallocation(&slot.sample.tokens);
            std::vector<float>().swap(slot.sample.input);std::vector<int>().swap(slot.sample.tokens);
        }
    }
    size_t actualBytes(const Slot& slot) const {
        return slot.sample.item.getRAMUsage()+slot.sample.input.capacity()*sizeof(float)+slot.sample.tokens.capacity()*sizeof(int);
    }
    bool reclaimForDemand(size_t active,const Key& key) {
        bool reclaimed=false;
        for(size_t i=0;i<slots.size();++i) {
            if(i==active)continue;
            {
                std::lock_guard<std::mutex> lock(mutex);
                const size_t used=residentLocked()-slots[active]->resident,bytes=estimate(key);
                if(used<=options.max_bytes && bytes<=options.max_bytes-used &&
                    DatasetMemoryManager::instance().canAllocate(bytes))break;
                auto& slot=*slots[i];
                if(!slot.resident || slot.stage==Stage::Leased || slot.stage==Stage::Loading)continue;
                slot.stage=Stage::Loading;
            }
            // Drop only idle/speculative data. A live batch is never a victim.
            clean(*slots[i],true);reclaimed=true;
            {
                std::lock_guard<std::mutex> lock(mutex);
                slots[i]->resident=0;slots[i]->stage=Stage::Idle;slots[i]->urgent=false;++counters.recycled;
            }
        }
        return reclaimed;
    }
    void prepare(Slot& slot,const Key& key) {
        auto& sample=slot.sample;const auto& source=sources[key.index];auto& item=sample.item;
        sample.index=key.index;item.name=source.name;item.is_linked=source.linked;
        item.text_file=source.text_file;item.text_inline=source.text_inline;item.annotation_file=source.annotation_file;
        item.image_file=source.image_file;item.mask_file=source.mask_file;item.audio_file=source.audio_file;item.video_file=source.video_file;
        item.w=options.width;item.h=options.height;item.img_c=0;
        if(!source.text_file.empty() || !source.text_inline.empty()) {
            if(!item.loadText())sample.error=item.annotation_error.empty()?"Cannot load dataset text: "+source.text_file:item.annotation_error;
        }
        if(options.tokenizer && item.text) {
            auto ids=options.tokenizer->tokenize(*item.text);
            const size_t count=std::max(ids.size(),options.minimum_tokens);
            const size_t old=sample.tokens.capacity();
            if(count>std::numeric_limits<size_t>::max()/sizeof(int) ||
                (count>old && !DatasetMemoryManager::instance().canAllocate((count-old)*sizeof(int))))
                throw std::runtime_error("Dataset RAM limit: token buffer");
            sample.tokens.assign(ids.begin(),ids.end());sample.tokenized=true;
            if(sample.tokens.size()<options.minimum_tokens)sample.tokens.resize(options.minimum_tokens,options.pad_id);
            DatasetMemoryManager::instance().trackAllocation(&sample.tokens,sample.tokens.capacity()*sizeof(int));
        }
        if(key.format==Format::Text || key.format==Format::RGB) {
            DatasetMemoryManager::instance().trackDeallocation(&sample.input);
            std::vector<float>().swap(sample.input);
        }
        if(key.format==Format::MediaHWC && !source.video_file.empty()) {
            sample.input.clear();
            // The trainer streams every video frame. Its optional cover image
            // must not become the model input or leave a stale RAM registration
            // when the frame stream replaces item.img.
            if(!source.mask_file.empty() && !item.loadMask(options.width,options.height))
                sample.error="Cannot decode dataset mask: "+source.mask_file;
            return;
        }
        if(key.format==Format::Text || source.image_file.empty()){sample.input.clear();return;}
        if(!item.loadImageRGB(options.width,options.height)) {
            sample.input.clear();sample.error="Cannot decode dataset image: "+source.image_file;return;
        }
        if(!source.mask_file.empty() && !item.loadMask(options.width,options.height))sample.error="Cannot decode dataset mask: "+source.mask_file;
        if(key.format==Format::RGB)return;
        const size_t count=item.img.size(),old=sample.input.capacity()*sizeof(float);
        if(count>std::numeric_limits<size_t>::max()/sizeof(float))throw std::length_error("dataset input size overflow");
        const size_t bytes=count*sizeof(float);
        if(bytes>old && !DatasetMemoryManager::instance().canAllocate(bytes-old)) {
            sample.input.clear();sample.error="Dataset RAM limit: normalized input buffer";return;
        }
        sample.input.resize(count);
        DatasetMemoryManager::instance().trackAllocation(&sample.input,sample.input.capacity()*sizeof(float));
        const size_t pixels=size_t(options.width)*options.height;
        for(size_t i=0;i<count;++i) {
            const float value=key.format==Format::FloatCHW?float(item.img[i])/127.5f-1.f:
                float(double(item.img[i])/127.5-1.0);
            const size_t destination=(key.format==Format::HWC || key.format==Format::MediaHWC)?i:(i%3)*pixels+i/3;
            sample.input[destination]=value;
        }
    }

    void run() noexcept {
#ifdef _OPENMP
        // Keep preparation from creating another model-sized OpenMP team.
        // These ICVs are local to this producer thread.
        omp_set_num_threads(1);
#endif
        try {for(;;) {
            size_t index=slots.size();Key key;bool recycle=false,demanded=false;
            {
                std::unique_lock<std::mutex> lock(mutex);
                work.wait(lock,[&] {
                    if(stopping)return true;
                    for(bool urgent:{true,false}) {
                        if(!urgent)for(size_t i=0;i<slots.size();++i)if(slots[i]->stage==Stage::Recycle) {
                            index=i;recycle=true;return true;
                        }
                        for(size_t i=0;i<slots.size();++i)if(slots[i]->stage==Stage::Queued && slots[i]->urgent==urgent) {
                            const size_t bytes=estimate(slots[i]->key),used=residentLocked()-slots[i]->resident;
                            // Only a demanded sample may exceed the look-ahead
                            // byte budget. It must still fit dataset RAM.
                            if(urgent || (used<=options.max_bytes && bytes<=options.max_bytes-used &&
                                DatasetMemoryManager::instance().canAllocate(bytes))) {index=i;return true;}
                        }
                    }
                    return false;
                });
                if(stopping)break;
                auto& slot=*slots[index];key=slot.key;demanded=slot.urgent;slot.stage=Stage::Loading;
            }
            auto& slot=*slots[index];const auto start=Clock::now();
            const size_t retained=slot.sample.input.capacity()*sizeof(float)+slot.sample.tokens.capacity()*sizeof(int);
            clean(slot,retained>options.max_bytes/slots.size());
            if(demanded)reclaimForDemand(index,key);
            const size_t input_capacity=slot.sample.input.capacity(),token_capacity=slot.sample.tokens.capacity();
            if(!recycle) {
                try{prepare(slot,key);}catch(const std::exception& e){slot.sample.input.clear();slot.sample.error=e.what();}
                catch(...){slot.sample.input.clear();slot.sample.error="Unknown dataset preparation error";}
                {
                    std::lock_guard<std::mutex> lock(mutex);demanded=slot.urgent;
                }
                if(demanded && !slot.sample.error.empty() && reclaimForDemand(index,key)) {
                    // An in-flight speculative read may have become demanded
                    // under RAM pressure. Retry once after evicting idle data.
                    slot.sample.error.clear();
                    try{prepare(slot,key);}catch(const std::exception& e){slot.sample.input.clear();slot.sample.error=e.what();}
                    catch(...){slot.sample.input.clear();slot.sample.error="Unknown dataset preparation error";}
                }
            }
            {
                std::lock_guard<std::mutex> lock(mutex);
                slot.resident=actualBytes(slot);
                counters.buffer_growths+=slot.sample.input.capacity()>input_capacity;
                counters.buffer_growths+=slot.sample.tokens.capacity()>token_capacity;
                if(recycle) {slot.stage=Stage::Idle;slot.urgent=false;++counters.recycled;}
                else {
                    ++counters.prepared;if(!slot.sample.error.empty())++counters.errors;
                    counters.prepare_ms+=milliseconds(start);
                    slot.stage=slot.discard?Stage::Recycle:Stage::Ready;
                }
                counters.peak_bytes=std::max(counters.peak_bytes,residentLocked());
                if(!stopping)fillLocked();
            }
            changed.notify_all();work.notify_one();
        }}catch(...) {
            // Allocation/control failures must wake the consumer, never escape
            // a std::thread entry point and terminate the training process.
            std::lock_guard<std::mutex> lock(mutex);
            failure=std::current_exception();stopping=true;++counters.errors;
        }
        // Queued work is abandoned at shutdown; do not decode the rest of an epoch.
        for(auto& ptr:slots) {
            bool held;
            {std::lock_guard<std::mutex> lock(mutex);held=ptr->stage==Stage::Leased;}
            if(!held) {
                clean(*ptr,true);
                std::lock_guard<std::mutex> lock(mutex);ptr->resident=0;ptr->stage=Stage::Idle;
            }
        }
        changed.notify_all();
    }
    void release(size_t index) noexcept {
        std::lock_guard<std::mutex> lock(mutex);
        auto& slot=*slots[index];slot.urgent=false;slot.stage=Stage::Recycle;
        if(!stopping)fillLocked();
        work.notify_one();changed.notify_all();
    }
    void stop() noexcept {
        {std::lock_guard<std::mutex> lock(mutex);stopping=true;}
        work.notify_all();changed.notify_all();
        if(worker.joinable())worker.join();
    }
};

DatasetPrefetcher::DatasetPrefetcher(const std::vector<DatasetItem>& items,Options options)
    :state_(std::make_shared<State>(items,std::move(options))) {}
DatasetPrefetcher::~DatasetPrefetcher(){stop();}
DatasetPrefetcher::Lease::Lease(std::shared_ptr<State> state,size_t slot):state_(std::move(state)),slot_(slot) {}
DatasetPrefetcher::Lease::~Lease(){reset();}
DatasetPrefetcher::Lease::Lease(Lease&& other) noexcept:state_(std::move(other.state_)),slot_(other.slot_) {}
DatasetPrefetcher::Lease& DatasetPrefetcher::Lease::operator=(Lease&& other) noexcept {
    if(this!=&other){reset();state_=std::move(other.state_);slot_=other.slot_;}return *this;
}
DatasetPrefetcher::Sample& DatasetPrefetcher::Lease::operator*() const{return state_->slots[slot_]->sample;}
DatasetPrefetcher::Sample* DatasetPrefetcher::Lease::operator->() const{return &**this;}
void DatasetPrefetcher::Lease::reset() noexcept {
    if(state_){state_->release(slot_);state_.reset();}
}
DatasetPrefetcher::Lease DatasetPrefetcher::acquire(size_t index,Format format) {
    auto& state=*state_;
    if(!state.counters.enabled)return {};
    if(index>=state.sources.size())throw std::out_of_range("dataset prefetch index out of range");
    const State::Key key{index,format};const auto start=Clock::now();bool waited=false;
    std::unique_lock<std::mutex> lock(state.mutex);++state.counters.requests;
    for(;;) {
        if(state.stopping) {
            if(state.failure)std::rethrow_exception(state.failure);
            throw std::runtime_error("dataset prefetch stopped");
        }
        size_t slot=state.findLocked(key);
        if(slot!=state.slots.size() && state.slots[slot]->stage==State::Stage::Ready) {
            state.slots[slot]->stage=State::Stage::Leased;
            if(!waited)++state.counters.hits;else state.counters.wait_ms+=milliseconds(start);
            if(state.explicit_order) {
                if(state.cursor<state.order.size() && state.order[state.cursor]==index && state.order_format==format)++state.cursor;
            } else {state.next_sequential=index+1;state.sequential_format=format;}
            state.fillLocked();
            return Lease(state_,slot);
        }
        if(slot==state.slots.size()) {
            slot=state.availableLocked(true);
            if(slot!=state.slots.size())state.queueLocked(slot,key,true);
            else {
                bool all_held=true;for(const auto& entry:state.slots)all_held&=entry->stage==State::Stage::Leased;
                if(all_held)throw std::runtime_error("dataset prefetch: all slots are held by active samples");
            }
        } else state.slots[slot]->urgent=true;
        if(!waited){++state.counters.waits;waited=true;}
        state.work.notify_one();state.changed.wait(lock);
    }
}
void DatasetPrefetcher::setOrder(std::vector<size_t> order,Format format) {
    auto& state=*state_;if(!state.counters.enabled)return;
    for(size_t index:order)if(index>=state.sources.size())throw std::out_of_range("dataset prefetch order index out of range");
    std::lock_guard<std::mutex> lock(state.mutex);
    state.order=std::move(order);state.cursor=0;state.explicit_order=!state.order.empty();state.order_format=format;
    if(!state.explicit_order)state.next_sequential=state.sources.size();
    state.fillLocked();
}
void DatasetPrefetcher::invalidate(size_t index) {
    auto& state=*state_;std::lock_guard<std::mutex> lock(state.mutex);
    for(auto& slot:state.slots)if(slot->key.index==index) {
        if(slot->stage==State::Stage::Loading)slot->discard=true;
        else if(slot->stage==State::Stage::Ready || slot->stage==State::Stage::Queued)slot->stage=State::Stage::Recycle;
    }
    state.work.notify_one();state.changed.notify_all();
}
DatasetPrefetcher::Stats DatasetPrefetcher::stats() const {
    auto& state=*state_;std::lock_guard<std::mutex> lock(state.mutex);Stats result=state.counters;
    for(const auto& slot:state.slots) {
        result.queued+=slot->stage==State::Stage::Queued;result.ready+=slot->stage==State::Stage::Ready;
        result.leased+=slot->stage==State::Stage::Leased;result.loading+=slot->stage==State::Stage::Loading;
        result.resident_bytes+=slot->resident;
    }
    return result;
}
void DatasetPrefetcher::stop() noexcept {if(state_)state_->stop();}
bool DatasetPrefetcher::matchesDimensions(int width,int height) const {
    return state_->options.width==width && state_->options.height==height;
}
bool DatasetPrefetcher::preparesTokens() const {return bool(state_->options.tokenizer);}
