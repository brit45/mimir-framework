#include "runtimes/GradientCommit.hpp"
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>

static void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
static bool exact(const std::vector<float>& a,const std::vector<float>& b){
    return a.size()==b.size() && (a.empty() || std::memcmp(a.data(),b.data(),a.size()*sizeof(float))==0);
}
int main(){
    try{
        Layer layer;layer.weights.assign(4,.5f);layer.grad_weights={1,2,3,4};layer.grad_bias={5};
        require(RuntimeGradientCommit::parameters(layer,{.5f,-.5f,1.f},{2.f}),"valid separate bias commit");
        require(layer.grad_weights==std::vector<float>({1.5f,1.5f,4.f,6.f}) && layer.grad_bias==std::vector<float>({7.f}),"separate bias counted incorrectly");
        require(RuntimeGradientCommit::parameters(layer,{1,2,3,4},{4},true),"valid packed bias commit");
        require(layer.grad_weights==std::vector<float>({2.5f,3.5f,7.f,10.f}) && layer.grad_bias==std::vector<float>({11.f}),"packed bias counted twice");
        const float max=std::numeric_limits<float>::max(),nan=std::numeric_limits<float>::quiet_NaN();
        auto refused=[&](const std::vector<float>& dw,const std::vector<float>& db,bool packed=false){
            auto before=layer.grad_weights,bias=layer.grad_bias;
            require(!RuntimeGradientCommit::parameters(layer,dw,db,packed),"invalid commit accepted");
            require(exact(before,layer.grad_weights) && exact(bias,layer.grad_bias),"failed commit modified gradients");
        };
        refused({1,2,nan},{});refused({1,2,3},{std::numeric_limits<float>::infinity()});
        refused({1,2,3,4,5},{});refused({1},{1,2,3,4,5});
        layer.grad_weights={1,2,max,4};refused({1,2,max},{});
        layer.grad_weights={1,2,3,max};refused({1,2,3},{max});
        layer.grad_weights={1,2,3,4};layer.grad_bias={max};refused({1,2,3},{max});
        layer.grad_weights={1,2,nan,4};refused({1,2,3},{});
        layer.grad_weights={1,2,3,4};layer.grad_bias={nan};refused({1,2,3},{});
        std::cout<<"GRADIENT_COMMIT_TRANSACTIONAL_OK\n";return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
