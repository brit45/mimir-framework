#include "test_utils.hpp"
#include "DatasetPrefetcher.hpp"
#include "Tokenizer.hpp"
#include <chrono>
#include <cstring>
#include <fstream>
#include <set>
#include <thread>
#if defined(__unix__)
#include <sys/stat.h>
#endif

using Format=DatasetPrefetcher::Format;
using Clock=std::chrono::steady_clock;
static void check(bool ok,const char* message) {if(!ok)throw std::runtime_error(message);}
template<class Predicate> void until(Predicate ready) {
    const auto end=Clock::now()+std::chrono::seconds(5);
    while(!ready()) {
        if(Clock::now()>end)throw std::runtime_error("dataset producer did not make progress");
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}
struct Fixture {
    fs::path root=fs::temp_directory_path()/("mimir_prefetch_"+std::to_string(Clock::now().time_since_epoch().count()));
    std::vector<DatasetItem> items;
    Fixture() {
        fs::create_directories(root);items.resize(12);
        for(size_t i=0;i<items.size();++i) {
            auto& item=items[i];item.name=std::to_string(i);item.w=13;item.h=7;
            item.image_file=(root/(item.name+".ppm")).string();
            item.mask_file=(root/(item.name+"_mask.pgm")).string();
            item.text_file=(root/(item.name+".txt")).string();
            std::ofstream image(item.image_file,std::ios::binary);image<<"P6\n13 7\n255\n";
            for(size_t p=0;p<13*7*3;++p)image.put(char((p*7+i*13)%256));
            std::ofstream mask(item.mask_file,std::ios::binary);mask<<"P5\n13 7\n255\n";
            for(size_t p=0;p<13*7;++p)mask.put(char((p+i)%2?255:0));
            std::ofstream text(item.text_file);text<<"Une image numéro "<<i<<". Texte de contrôle.";
        }
    }
    ~Fixture(){std::error_code error;fs::remove_all(root,error);}
};
static void reference(const DatasetItem& source,int width,int height,DatasetItem& item) {
    // Metadata-only copy: do not copy loaded buffers with RAM registrations.
    item.name=source.name;item.image_file=source.image_file;item.mask_file=source.mask_file;
    item.text_file=source.text_file;item.text_inline=source.text_inline;item.annotation_file=source.annotation_file;
    check(item.loadImageRGB(width,height),"reference image failed");
    check(item.loadMask(width,height),"reference mask failed");check(item.loadText(),"reference text failed");
}
static void parity(Fixture& fixture) {
    for(Format format:{Format::RGB,Format::HWC,Format::CHW,Format::FloatCHW,Format::MediaHWC}) {
        DatasetPrefetcher::Options options;options.width=11;options.height=5;options.ahead=2;options.max_bytes=1024*1024;
        DatasetPrefetcher queue(fixture.items,options);const std::vector<size_t> order{7,2,2,11,0,5,9};
        queue.setOrder(order,format);
        until([&]{return queue.stats().ready==2;}); // Decoding starts without an acquire call.
        for(size_t index:order) {
            auto sample=queue.acquire(index,format);DatasetItem want;reference(fixture.items[index],11,5,want);
            check(sample->index==index && sample->item.image_file==fixture.items[index].image_file,"source identity/order changed");
            check(sample->item.img==want.img && sample->item.mask==want.mask && sample->item.text==want.text,"decoded payload changed");
            if(format!=Format::RGB) {
                std::vector<float> expected(want.img.size());
                for(size_t i=0;i<want.img.size();++i) {
                    const size_t dst=(format==Format::HWC || format==Format::MediaHWC)?i:(i%3)*55+i/3;
                    expected[dst]=format==Format::FloatCHW?float(want.img[i])/127.5f-1.f:float(double(want.img[i])/127.5-1.0);
                }
                check(sample->input.size()==expected.size() && !std::memcmp(sample->input.data(),expected.data(),expected.size()*sizeof(float)),
                    "normalization/permutation changed bits");
            }
            const auto s=queue.stats();
            check(s.queued+s.ready+s.leased+s.loading<=3,"look-ahead exceeded fixed slots");
        }
        check(queue.stats().hits>=1,"prepared first sample was not a cache hit");
    }
    std::cout<<"DATASET_PREFETCH_PARITY_OK RGB, HWC, CHW, distinct float/double rounding, masks, text, shuffled duplicates\n";
}
static void leasesAndReuse(Fixture& fixture) {
    DatasetPrefetcher::Options options;options.width=13;options.height=7;options.ahead=2;options.max_bytes=1024*1024;
    DatasetPrefetcher queue(fixture.items,options);
    queue.setOrder({0,1,2,3,4,5},Format::HWC);
    auto held=queue.acquire(0,Format::HWC);const auto pixels=held->item.img;const auto input=held->input;
    for(size_t i:{9u,3u,7u,1u,0u}) {
        auto nested=queue.acquire(i,Format::HWC);
        check(held->item.img==pixels && held->input==input,"live sample overwritten by producer/validation");
    }
    held.reset();queue.setOrder({},Format::HWC);
    // Repeated use has a bounded number of reusable conversion buffers.
    std::vector<DatasetPrefetcher::Lease> warm_samples;
    for(size_t i=0;i<3;++i)warm_samples.push_back(queue.acquire(i,Format::HWC));
    queue.setOrder({},Format::HWC);warm_samples.clear();
    const auto warm=queue.stats().buffer_growths;
    for(size_t i=0;i<96;++i){auto sample=queue.acquire(i%12,Format::HWC);}
    check(queue.stats().buffer_growths==warm,"warm input buffers were reallocated");
    bool rejected=false;try{queue.setOrder({fixture.items.size()},Format::HWC);}catch(const std::out_of_range&){rejected=true;}
    check(rejected,"invalid order accepted");
    std::cout<<"DATASET_PREFETCH_LEASE_OK nested validation, stable live buffers, warm reuse, invalid indices\n";
}
static void textAndErrors(Fixture& fixture) {
    auto tokenizer=std::make_shared<Tokenizer>();
    for(auto& item:fixture.items){item.loadText();tokenizer->ensureVocabFromText(*item.text);item.unload();}
    const auto before_vocab=tokenizer->getVocabSize();
    DatasetPrefetcher::Options options;options.width=13;options.height=7;options.ahead=2;
    options.tokenizer=std::make_shared<const Tokenizer>(*tokenizer);options.minimum_tokens=64;options.pad_id=tokenizer->getPadId();
    DatasetPrefetcher queue(fixture.items,options);queue.setOrder({4,8},Format::Text);
    for(size_t i:{4u,8u}) {
        auto sample=queue.acquire(i,Format::Text);auto ids=tokenizer->tokenize(*sample->item.text);ids.resize(64,options.pad_id);
        check(sample->tokenized && sample->tokens==ids && sample->input.empty() && !sample->item.img_loaded,"text preparation changed");
    }
    check(tokenizer->getVocabSize()==before_vocab,"background tokenizer mutated vocabulary");
    std::vector<DatasetItem> invalid(3);
    invalid[0].image_file=(fixture.root/"missing.ppm").string();
    invalid[1].image_file=(fixture.root/"bad.ppm").string();std::ofstream(invalid[1].image_file)<<"corrupt";
    invalid[2].text_inline="texte inline";
    invalid[2].image_file=fixture.items[0].image_file;
    DatasetPrefetcher errors(invalid,options);errors.setOrder({0,1,2},Format::RGB);
    {auto sample=errors.acquire(2,Format::HWC);check(sample->input.size()==273,"warm conversion failed");}
    for(size_t i:{0u,1u}){auto sample=errors.acquire(i,Format::HWC);check(!sample->item.img_loaded && sample->input.empty() && !sample->error.empty(),"decode failure exposed stale converted input");}
    auto valid=errors.acquire(2,Format::Text);check(valid->item.text==std::optional<std::string>("texte inline"),"decode failure poisoned later requests");
    std::cout<<"DATASET_PREFETCH_TEXT_OK immutable tokenizer, padding, missing/corrupt files, recovery\n";
}
static void budgetAndLifetime(Fixture& fixture) {
    DatasetPrefetcher::Options options;options.width=13;options.height=7;options.ahead=2;options.max_bytes=1;
    DatasetPrefetcher::Lease retained;
    {
        std::vector<DatasetItem> metadata(1);metadata[0].image_file=fixture.items[3].image_file;
        DatasetPrefetcher queue(metadata,options);metadata.clear(); // Producer owns its path snapshots.
        retained=queue.acquire(0,Format::HWC);
        check(retained->item.img_loaded && retained->input.size()==273,"demanded oversized item made no progress");
        check(queue.stats().leased==1 && queue.stats().ready==0,"byte cap preloaded extra oversized items");
        queue.stop();check(retained->item.img_loaded,"shutdown invalidated an outstanding lease");
    }
    retained.reset(); // Must unregister retained conversion buffers, even after shutdown.
    {
        std::vector<DatasetItem> media(1);media[0].image_file=fixture.items[3].image_file;
        media[0].video_file=(fixture.root/"stream.mkv").string();media[0].mask_file=fixture.items[3].mask_file;
        DatasetPrefetcher queue(media,options);auto sample=queue.acquire(0,Format::MediaHWC);
        check(!sample->item.img_loaded && sample->input.empty() && sample->item.mask_loaded,
            "video cover was substituted for streamed training frames");
    }
    const size_t limit=DatasetMemoryManager::instance().getMaxRAM();
    DatasetMemoryManager::instance().setMaxRAM(1800);
    {
        options.max_bytes=1024*1024;DatasetPrefetcher tight(fixture.items,options);
        tight.setOrder({0,1,2,3,4,5},Format::HWC);
        for(size_t i=0;i<6;++i) {
            auto sample=tight.acquire(i,Format::HWC);
            check(sample->item.img_loaded && sample->input.size()==273,"speculative buffers starved a demanded sample under dataset RAM limit");
        }
    }
    DatasetMemoryManager::instance().setMaxRAM(limit);
    options.ahead=0;DatasetPrefetcher disabled(fixture.items,options);
    check(!disabled.stats().enabled && !disabled.acquire(0),"disabled prefetch did background work");
    std::cout<<"DATASET_PREFETCH_BUDGET_OK oversized demanded sample, metadata lifetime, lease after shutdown, disabled mode\n";
}
static void blockedProducer(Fixture& fixture) {
#if defined(__unix__)
    const auto pipe=fixture.root/"annotation.pipe",stub=fixture.root/"annotation.source";
    check(::mkfifo(pipe.c_str(),0600)==0,"cannot create decoder synchronization fixture");
    std::ofstream(stub)<<"{}";
    std::vector<DatasetItem> items(2);items[0].text_file=stub.string();items[0].annotation_file=pipe.string();
    items[1].text_inline="ready after replan";
    DatasetPrefetcher::Options options;options.ahead=1;
    DatasetPrefetcher queue(items,options);queue.setOrder({0},Format::Text);
    until([&]{return queue.stats().loading==1;});
    // These operations must remain available while the producer blocks on I/O.
    queue.setOrder({1},Format::Text);
    {std::ofstream writer(pipe);writer<<"{\"captions\":{\"primary\":\"unblocked\"}}";}
    auto sample=queue.acquire(1,Format::Text);
    check(sample->item.text==std::optional<std::string>("ready after replan"),"obsolete I/O result replaced replanned sample");
    std::cout<<"DATASET_PREFETCH_ASYNC_OK metadata/control remain available during blocked I/O\n";
#endif
}
int main() {
    try {
        auto& memory=DatasetMemoryManager::instance();const auto baseline=memory.getCurrentRAM(),limit=memory.getMaxRAM();
        memory.setMaxRAM(64*1024*1024);
        {Fixture fixture;parity(fixture);leasesAndReuse(fixture);textAndErrors(fixture);budgetAndLifetime(fixture);blockedProducer(fixture);}
        check(memory.getCurrentRAM()==baseline,"dataset allocations leaked after producer shutdown");memory.setMaxRAM(limit);
        std::cout<<"DATASET_PREFETCH_OK bounded producer, exact values, reuse, cancellation and complete cleanup\n";
    }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
