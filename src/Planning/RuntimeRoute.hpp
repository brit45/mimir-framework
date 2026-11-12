#pragma once
#include <algorithm>
#include <array>
#include <cstddef>
#include <vector>
#include <utility>
class AbstractRuntime;

namespace Mimir::Planning::Calibration {
// The router has six native backend slots. Keep their per-dispatch snapshot on
// the stack, while still accepting larger candidate lists from C++ callers.
class RuntimeRoute {
    std::array<AbstractRuntime*,6> inline_{};
    std::vector<AbstractRuntime*> overflow_;
    size_t size_=0;
    bool overflow_mode_=false;
public:
    using iterator=AbstractRuntime**;
    using const_iterator=AbstractRuntime* const*;
    RuntimeRoute()=default;
    RuntimeRoute(const RuntimeRoute&)=default;
    RuntimeRoute& operator=(const RuntimeRoute&)=default;
    RuntimeRoute(RuntimeRoute&& other) noexcept :inline_(other.inline_),overflow_(std::move(other.overflow_)),
        size_(std::exchange(other.size_,0)),overflow_mode_(std::exchange(other.overflow_mode_,false)) {}
    RuntimeRoute& operator=(RuntimeRoute&& other) noexcept {
        if(this!=&other){inline_=other.inline_;overflow_=std::move(other.overflow_);
            size_=std::exchange(other.size_,0);overflow_mode_=std::exchange(other.overflow_mode_,false);}
        return *this;
    }
    RuntimeRoute(const std::vector<AbstractRuntime*>& values) {assign(values.data(),values.size());}
    RuntimeRoute(AbstractRuntime* const* first,AbstractRuntime* const* last) {assign(first,first==last?0:size_t(last-first));}
    RuntimeRoute& operator=(const std::vector<AbstractRuntime*>& values) {assign(values.data(),values.size());return *this;}
    void assign(AbstractRuntime* const* values,size_t count) {
        if(count<=inline_.size()) {
            if(count)std::copy_n(values,count,inline_.data());
            overflow_.clear();overflow_mode_=false;
        }else {
            overflow_.assign(values,values+count);overflow_mode_=true;
        }
        size_=count;
    }
    size_t size() const {return size_;}
    bool empty() const {return !size_;}
    iterator begin() {return overflow_mode_?overflow_.data():inline_.data();}
    iterator end() {return begin()+size_;}
    const_iterator begin() const {return overflow_mode_?overflow_.data():inline_.data();}
    const_iterator end() const {return begin()+size_;}
    AbstractRuntime*& operator[](size_t i) {return begin()[i];}
    AbstractRuntime* const& operator[](size_t i) const {return begin()[i];}
    AbstractRuntime* front() const {return (*this)[0];}
    void clear() {overflow_.clear();overflow_mode_=false;size_=0;}
    void push_back(AbstractRuntime* runtime) {
        if(!overflow_mode_ && size_<inline_.size()) {inline_[size_++]=runtime;return;}
        if(!overflow_mode_) {
            overflow_.assign(inline_.begin(),inline_.end());
            overflow_.push_back(runtime);overflow_mode_=true;
        }else overflow_.push_back(runtime);
        ++size_;
    }
    iterator erase(iterator position) {
        const size_t index=size_t(position-begin());
        if(overflow_mode_)overflow_.erase(overflow_.begin()+index);
        else std::move(begin()+index+1,end(),begin()+index);
        --size_;return begin()+index;
    }
};
}
