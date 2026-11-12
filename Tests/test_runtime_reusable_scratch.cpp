#include "runtimes/cpu/ReusableScratch.hpp"
#include "runtimes/cpu/FlatJoinKernels.hpp"
#include <atomic>
#include <iostream>
#include <thread>
#include <stdexcept>

namespace {
void check(bool value,const char* message) {if(!value)throw std::runtime_error(message);}
void buffers() {
    using namespace RuntimeScratch;
    auto& pool=Pool::local(); pool.clear();
    auto& guard=MemoryGuard::instance();
    const auto initial=guard.getCurrentBytes(), count=pool.allocations();
    float *first=nullptr,*second=nullptr;
    {
        FloatBuffer a(128,7.f);first=a.data();
        check(guard.getCurrentBytes()==initial+128*sizeof(float),"scratch not accounted");
        FloatBuffer nested(128,-3.f);second=nested.data();
        check(nested.data()!=a.data(),"nested lease aliases live data");
        check(a[0]==7.f&&a[127]==7.f,"nested lease corrupted outer buffer");
    }
    check(pool.allocations()==count+2,"cold allocations wrong");
    const auto warm=pool.allocations();
    {
        FloatBuffer a(64);
        check(a.data()==first||a.data()==second,"reusable buffer shape wrong");
        for(float v:a)check(v==0.f,"dirty cached memory exposed as zero initialized");
        a[0]=9.f;a.resize(120);
        check(a[0]==9.f&&a[119]==0.f,"resize prefix or zero tail changed");
        a.resize(4);a.resize(100);
        check(a[0]==9.f&&a[4]==0.f,"shrink/regrow exposes stale data");
        FloatBuffer b(16,2.f);a.swap(b);
        check(a.size()==16&&a[0]==2.f&&b.size()==100&&b[0]==9.f,"swap lost lease ownership");
        try {b.resize(std::numeric_limits<size_t>::max());check(false,"overflow accepted");}
        catch(const std::length_error&) {}
        check(b.size()==100&&b[0]==9.f,"failed grow corrupted live values");
    }
    check(pool.allocations()==warm,"warm working set allocated again");
    pool.clear();check(guard.getCurrentBytes()==initial,"clear leaked scratch memory");
    {FloatBuffer ready(128,1.f);}
    guard.blockAllocations();
    try {
        FloatBuffer warm_buffer(128,3.f);
        check(warm_buffer[127]==3.f,"reuse under allocation freeze failed");
    }catch(...){guard.blockAllocations(false);throw;}
    guard.blockAllocations(false);pool.clear();
    const auto limit=guard.getLimit();
    struct Restore {MemoryGuard& guard;size_t limit;~Restore(){Pool::local().clear();guard.setLimit(limit);}} restore{guard,limit};
    guard.setLimit(16*1024*1024);
    {
        FloatBuffer large(Pool::budget()/sizeof(float)+1,0.f);
        check(large.size()>0,"oversized lease failed");
    }
    check(pool.cachedBytes()==0,"oversized lease retained beyond cache budget");
    {FloatBuffer retained(100,1.f);}
    check(pool.cachedBytes()<=Pool::budget(),"cache exceeds retention bound");
    guard.setLimit(1024);
    {FloatBuffer tiny(1,0.f);}
    check(pool.cachedBytes()<=Pool::budget(),"lowered memory limit not respected");
    pool.clear();
    check(guard.getCurrentBytes()==initial,"budget changes leaked memory");
}
void threadIsolation() {
    using namespace RuntimeScratch;
    auto& pool=Pool::local();pool.clear();
    FloatBuffer outer(32,11.f);
    std::atomic<bool> okay{true};
    std::thread worker([&]{
        auto& local=Pool::local();
        {FloatBuffer inner(32,-4.f);
            if(inner.data()==outer.data())okay=false;
            for(float v:inner)if(v!=-4.f)okay=false;
        }
        if(!local.cachedBytes())okay=false;
    });
    worker.join();
    check(okay&&outer[0]==11.f&&outer[31]==11.f,"thread-local ownership failed");
    check(pool.cachedBytes()==0,"other thread populated caller's cache");
}
void arenas() {
    using namespace RuntimeScratch;
    Pool::local().clear();
    {
        FloatArena workspace({17,31,0,8});
        auto a=workspace.take(17),b=workspace.take(31),empty=workspace.take(0),c=workspace.take(8);
        a.fill(2.f);b.fill(3.f);c.fill(4.f);
        check(a[16]==2.f&&b[30]==3.f&&c[7]==4.f&&empty.size()==0,"arena slices overlap");
        a.resize(17);
        try {a.resize(18);check(false,"slice shape overflow accepted");}catch(const std::length_error&){}
        try {workspace.take(1);check(false,"arena overflow accepted");}catch(const std::length_error&){}
    }
    const auto allocations=Pool::local().allocations();
    {FloatArena warmed({17,31,0,8});check(warmed.take(17).size()==17,"warm arena shape");}
    check(Pool::local().allocations()==allocations,"warm arena allocated");
    Pool::local().clear();
    try {FloatArena huge({std::numeric_limits<size_t>::max(),1});check(false,"arena integer overflow accepted");}
    catch(const std::length_error&){}
}

void joins() {
    std::vector<float> a{1,2},b{3,4},out;
    std::vector<const std::vector<float>*> inputs{&a,&b};
    check(RuntimeFlatJoin::forward(inputs,out,false),"join failed");
    const float* storage=out.data();
    for(int i=0;i<8;++i)check(RuntimeFlatJoin::forward(inputs,out,false)&&out.data()==storage,"join reallocated warmed output");
    check(out==std::vector<float>({1,2,3,4}),"join values changed");
    check(RuntimeFlatJoin::forward({&a,&b},a,false)&&a==out,"aliased join corrupted source");
    const auto before=out;
    check(!RuntimeFlatJoin::forward({&a,nullptr},out,false)&&out==before,"invalid join changed output");
    const std::vector<float> empty;
    check(RuntimeFlatJoin::forward({&empty,&empty},out,false)&&out.empty(),"empty join failed");
}
}
int main() {
    try {buffers();threadIsolation();RuntimeScratch::Pool::local().clear();arenas();joins();}
    catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
    std::cout<<"PASS reusable scratch: initialization, resize, nested leases, threads, accounting, budgets and aliasing\n";
}
