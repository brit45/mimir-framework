#include "Model.hpp"
#include <atomic>
#include <cstring>
#include <thread>
#include <stdexcept>

void check(bool condition,const char* message){if(!condition)throw std::runtime_error(message);}
int main(){
    Model::setFrameworkLogsSuppressed(true);
    Model model;
    std::vector<float> source{0.f,-0.f,1.234375f,-2.71875f,7.f};
    for(const char* name:{"float32","float16","bfloat16","float64"}) {
        model.clearTensorStore();model.setDefaultDType(name);
        const auto dtype=Mimir::parse_dtype(name);
        const auto expected=Mimir::TypedTensor::fromFloat32(source,{5},dtype);
        model.storeTensor("value",source);
        check(model.hasTypedTensor("value")&&model.typed_tensor_store.empty(),"store eagerly duplicated the tensor");
        model.setDefaultDType("float32");
        const auto& snapshot=model.getTypedTensor("value");
        check(snapshot.dtype()==dtype&&snapshot.size_bytes()==expected.size_bytes(),"snapshot lost store-time dtype");
        check(std::memcmp(snapshot.data(),expected.data(),expected.size_bytes())==0,"snapshot changed scalar bits");
        const auto saved=snapshot;
        model.getTensorMutable("value")[2]=9.f;
        check(std::memcmp(snapshot.data(),expected.data(),expected.size_bytes())==0,"mutable float access changed typed snapshot semantics");
        model.storeTensor("value",std::vector<float>{6.f});
        check(model.typed_tensor_store.empty()&&model.getTypedTensor("value").numel()==1,"overwrite kept stale cache");
        check(saved.numel()==5&&std::memcmp(saved.data(),expected.data(),expected.size_bytes())==0,"owned snapshot aliases overwritten storage");
    }
    model.clearTensorStore();model.storeTensor("untouched",source);
    model.getTensorMutable("untouched")[0]=12.f;
    check(model.getTypedTensor("untouched").get(0)==0.f,"deferred snapshot was not frozen before mutable access");
    model.storeTensor("empty",std::vector<float>{});
    check(model.hasTypedTensor("empty")&&model.getTypedTensor("empty").numel()==0,"empty snapshot lost");
    model.clearTensorStore();
    check(!model.hasTypedTensor("empty")&&model.tensor_storage_dtypes_.empty(),"clear retained snapshot metadata");
    model.storeTensor("threaded",source);
    std::atomic<bool> okay{true};std::vector<std::thread> readers;
    for(int i=0;i<8;++i)readers.emplace_back([&]{for(int j=0;j<20;++j){
        if(model.getTypedTensor("threaded").get(2)!=source[2]||!model.hasTypedTensor("threaded")||model.hasTypedTensor("missing"))okay=false;
    }});
    for(auto& reader:readers)reader.join();
    check(okay&&model.typed_tensor_store.size()==1,"concurrent readers corrupted lazy cache");
    model.setDefaultDType("int32");
    bool rejected=false;
    try{model.storeTensor("invalid",std::vector<float>{std::numeric_limits<float>::quiet_NaN()});}
    catch(const std::invalid_argument&){rejected=true;}
    check(rejected,"integer finiteness validation was deferred");
    model.storeTensor("integer",source);
    check(model.typed_tensor_store.count("integer")==1,"integer conversion timing changed");
    std::cout<<"PASS typed snapshots: lazy floating storage, exact dtype/bits, invalidation, ownership and concurrent reads\n";
}
