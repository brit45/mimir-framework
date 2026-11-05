#pragma once
#include "Layers.hpp"
#include "runtimes/LayerOps.hpp"
#include "Planning/ExecutionPolicy.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace Mimir::GpuFusion {
inline bool supported(const std::vector<const Layer*>& chain) {
    if(chain.size()<2 || chain.size()>Planning::ExecutionPolicy::Limits::portable_reduction_span)return false;
    for(const auto* layer:chain) {
        int op;float alpha;
        if(!layer || layer->getWeightsSize()!=0 || layer->inputs.size()>1 ||
           (layer->type_enum!=LayerType::Identity && !RuntimeLayerOps::resolveUnaryOp(layer->type_enum,*layer,op,alpha)))return false;
    }
    return true;
}
// One arena gives each device invocation exclusive ownership of its element
// through the entire chain. No intermediate device/host transfer is needed.
struct Program {
    std::vector<float> memory;
    int n=0,stages=0,input=1,snapshots=0,operations=0,output=0;
    size_t output_count=0;
    bool retain=false,backward=false;
    bool prepare(const std::vector<const Layer*>& chain,const std::vector<float>& x,
                 bool keep,const std::vector<const std::vector<float>*>* saved=nullptr) {
        if(!supported(chain))return false;
        backward=saved!=nullptr;retain=keep;
        const size_t count=x.size(),depth=chain.size();
        if(backward) {
            if(saved->size()!=depth)return false;
            for(size_t j=0;j<depth;++j)
                if(chain[j]->type_enum!=LayerType::Identity && (!(*saved)[j] || (*saved)[j]->size()!=count))return false;
        }
        const size_t limit=std::min(size_t(std::numeric_limits<int>::max()),Planning::ExecutionPolicy::Limits::portable_staged_elements);
        const size_t rows=backward?depth+2:keep?depth+1:2;
        if(depth>(limit-1)/2 || count>(limit-1-depth*2)/rows ||
           count>Planning::ExecutionPolicy::Limits::portable_parallel_invocations)return false;
        n=int(count);stages=int(depth);output_count=count*(keep&&!backward?depth:1);
        memory.assign(1+rows*count+depth*2,0.f);
        std::copy(x.begin(),x.end(),memory.begin()+input);
        snapshots=int(1+count);operations=int(1+count+(backward?depth*count:0));
        output=operations+int(depth*2);
        for(size_t j=0;j<depth;++j) {
            int op=10;float alpha=.01f;
            if(chain[j]->type_enum!=LayerType::Identity)RuntimeLayerOps::resolveUnaryOp(chain[j]->type_enum,*chain[j],op,alpha);
            memory[operations+2*j]=float(op);memory[operations+2*j+1]=alpha;
            if(backward && chain[j]->type_enum!=LayerType::Identity)
                std::copy((*saved)[j]->begin(),(*saved)[j]->end(),memory.begin()+snapshots+j*count);
        }
        return true;
    }
    void forwardResult(std::vector<std::vector<float>>& destination) const {
        std::vector<std::vector<float>> result(retain?size_t(stages):1);
        for(size_t j=0;j<result.size();++j)
            result[j].assign(memory.begin()+output+j*n,memory.begin()+output+(j+1)*n);
        destination=std::move(result);
    }
    void backwardResult(std::vector<float>& destination) const {
        destination.assign(memory.begin()+output,memory.begin()+output+n);
    }
};
}
