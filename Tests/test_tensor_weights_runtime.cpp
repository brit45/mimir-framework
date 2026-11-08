#include "tensors.hpp"
#include "runtimes/cpu/CpuRuntime.hpp"
#include "runtimes/RuntimeRouter.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

static void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}

// Original public CPU formula retained independently as a numerical oracle.
static uint16_t original(const tensor& t){
    const float r2=t.Pos.X*t.Pos.X+t.Pos.Y*t.Pos.Y+t.Pos.Z*t.Pos.Z;
    const float r=std::sqrt(r2);
    const float spatial=std::tanh(std::min(std::max(r*.5f,-3.f),3.f));
    const float value=float(t.Value)*(1.f/255.f),length=float(t.Length)*(1.f/1024.f);
    const float velocity=std::tanh(value*2.f-1.f);
    const float density=std::tanh(std::min(std::max(length,0.f),1.f)*2.f-1.f);
    const float w=t.Pos.W,high=w*.5f,mid=w*.1f,low=w*.02f;
    const float s1=std::sin(high),c1=std::cos(high),s2=std::sin(mid),c2=std::cos(mid),s3=std::sin(low),c3=std::cos(low);
    const float context=std::tanh(w*.05f);
    const float features=.5f*s1+.35f*c1+.30f*s2+.20f*c2+.15f*s3+.10f*c3;
    const float sum=std::min(std::max(context+.6f*features,-1.f),1.f);
    const float gate=1.f/(1.f+std::exp(-2.2f*sum));
    const float spatial_weight=(1.f-gate)*.95f+.05f,dynamic_weight=gate*.90f;
    const float combined=spatial_weight*spatial+dynamic_weight*(velocity-.7f*density);
    float normalized=1.f/(1.f+std::exp(-2.f*combined));normalized=.98f*normalized+.01f;
    return uint16_t(std::max(0,std::min(int(std::floor(normalized*65535.f+.5f)),65535)));
}

int main(){
    try{
        auto& router=RuntimeRouter::instance();router.setActivators({},{},{},{},{});
        router.setRuntimes(nullptr,nullptr,nullptr,nullptr,nullptr);
        std::vector<tensor> tensors(257);
        std::vector<std::vector<float>> payloads;
        std::vector<uint16_t> expected,initial;
        std::vector<Vector4F> positions;
        for(size_t i=0;i<tensors.size();++i){
            auto& t=tensors[i];
            t.Pos={float(int(i%11)-5)*.31f,float(int(i%7)-3)*.73f,float(i%5)*1.7f,float(int(i)-128)*7.1f};
            t.Value=uint8_t(i);t.Length=uint16_t(i*251);t.Weight=uint16_t(900+i);
            t.data={float(i),-float(i),.125f};
            if(i==0)t.Pos={0,0,0,0};
            if(i==1)t.Pos={1e20f,-1e20f,1e20f,1e20f};
            expected.push_back(original(t));initial.push_back(t.Weight);
            payloads.push_back(t.data);positions.push_back(t.Pos);
        }
        auto unchanged=[&](const std::vector<uint16_t>& weights){
            for(size_t i=0;i<tensors.size();++i)require(tensors[i].Weight==weights[i],"partial tensor weight commit");
        };
        TensorSystem system;
        require(!system.computeWeights(tensors),"public facade requires initialization");unchanged(initial);
        CpuRuntime cpu;std::vector<uint16_t> output{11,22};
        require(!cpu.computeTensorWeights(tensors,output),"CPU operation requires initialization");
        require(output==std::vector<uint16_t>({11,22}),"CPU uninitialized output changed");
        AbstractRuntime* selected=&cpu;
        require(router.dispatchTensorWeights(tensors,output,&selected),"standalone dispatch");
        require(selected && std::string(selected->name())=="CPU" && output==expected,"standalone CPU numerical result");
        require(system.initialize() && system.initialize() && system.computeWeights(tensors),"public routed weights");
        unchanged(expected);
        router.setRuntimes(nullptr,nullptr,nullptr,nullptr,&cpu);
        selected=&cpu;output={11,22};
        require(!router.dispatchTensorWeights(tensors,output,&selected) && !selected,"configured uninitialized CPU must be refused");
        require(output==std::vector<uint16_t>({11,22}),"declined dispatch changed output");
        require(cpu.initialize({}) && cpu.supportsTensorWeights(),"CPU supports tensor weights");
        require(router.dispatchTensorWeights(tensors,output,&selected) && selected==&cpu && output==expected,"configured CPU route");
        for(size_t i=0;i<tensors.size();++i){
            const auto& t=tensors[i];const auto& p=positions[i];
            require(t.data==payloads[i] && t.Value==uint8_t(i) && t.Length==uint16_t(i*251),"payload metadata modified");
            require(t.Pos.X==p.X && t.Pos.Y==p.Y && t.Pos.Z==p.Z && t.Pos.W==p.W,"position modified");
        }
        for(float invalid:{std::numeric_limits<float>::quiet_NaN(),std::numeric_limits<float>::infinity(),-std::numeric_limits<float>::infinity()}){
            for(int axis=0;axis<4;++axis){
                auto& p=tensors.back().Pos;
                float* coordinate=axis==0?&p.X:axis==1?&p.Y:axis==2?&p.Z:&p.W;
                const float saved=*coordinate;*coordinate=invalid;
                output={11,22};selected=&cpu;
                require(!cpu.computeTensorWeights(tensors,output),"nonfinite CPU input accepted");
                require(output==std::vector<uint16_t>({11,22}),"CPU failure changed result");
                require(!router.dispatchTensorWeights(tensors,output,&selected) && !selected,"invalid routed input accepted");
                require(output==std::vector<uint16_t>({11,22}),"router failure changed result");
                require(!system.computeWeights(tensors),"public invalid input accepted");unchanged(expected);
                *coordinate=saved;
            }
        }
        std::vector<tensor> empty;
        require(system.computeWeights(empty) && router.dispatchTensorWeights(empty,output,&selected) && output.empty(),"empty batch");
        RuntimeConfig disabled;disabled.disabled=true;cpu.initialize(disabled);output={77};
        require(!router.dispatchTensorWeights(tensors,output,&selected) && !selected && output==std::vector<uint16_t>({77}),"disabled CPU ignored");
        router.setRuntimes(nullptr,nullptr,nullptr,nullptr,nullptr);
        std::cout<<"TENSOR_WEIGHTS_RUNTIME_OK records="<<tensors.size()<<'\n';
        return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
