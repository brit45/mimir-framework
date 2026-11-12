#include "Planning/RuntimeRoute.hpp"
#include <cstdlib>
#include <iostream>
#include <new>
#include <stdexcept>
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic ignored "-Wmismatched-new-delete"
#endif
namespace {bool counting=false;size_t allocations=0;}
void* operator new(size_t n){if(void* p=std::malloc(n?n:1)){if(counting)++allocations;return p;}throw std::bad_alloc();}
void* operator new[](size_t n){return ::operator new(n);}
void operator delete(void* p) noexcept {std::free(p);}
void operator delete[](void* p) noexcept {std::free(p);}
void operator delete(void* p,size_t) noexcept {std::free(p);}
void operator delete[](void* p,size_t) noexcept {std::free(p);}
int main(){
    using Mimir::Planning::Calibration::RuntimeRoute;
    int storage[8]{};std::vector<AbstractRuntime*> input;
    for(auto& v:storage)input.push_back(reinterpret_cast<AbstractRuntime*>(&v));
    counting=true;
    RuntimeRoute common(input.data(),input.data()+6),copied=common;
    common.erase(common.begin()+2);common.push_back(input[2]);
    const bool okay=common.size()==6&&common[2]==input[3]&&common[5]==input[2]&&copied[2]==input[2];
    counting=false;
    if(!okay||allocations)throw std::runtime_error("six-backend route allocated or lost order");
    RuntimeRoute large(input);auto large_copy=large;
    if(large.size()!=8||!std::equal(large.begin(),large.end(),input.begin()))return 1;
    large.erase(large.begin());large.erase(large.begin()+6);
    if(large.size()!=6||large.front()!=input[1])return 2;
    large.clear();large.push_back(input[0]);
    if(large.front()!=input[0]||large_copy.size()!=8)return 3;
    RuntimeRoute moved(std::move(large_copy));
    if(moved.size()!=8||!large_copy.empty())return 4;
    std::cout<<"PASS inline routes, overflow, copy, move, erase, and zero warm allocations\n";
}
