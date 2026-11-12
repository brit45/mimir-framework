#include "MetricsHistory.hpp"
#include <chrono>
#include <iostream>
#include <iterator>
#include <stdexcept>
std::string bytes(const std::filesystem::path& path){std::ifstream f(path,std::ios::binary);return {std::istreambuf_iterator<char>(f),{}};}
int main(){
    const auto root=std::filesystem::temp_directory_path()/("mimir-csv-batch-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(root);
    struct Cleanup{std::filesystem::path path;~Cleanup(){std::filesystem::remove_all(path);}} cleanup{root};
    std::vector<MetricsRecord> records(512);
    for(size_t i=0;i<records.size();++i){auto& r=records[i];r.step=i;r.loss=float(i)/1024;r.lr=.001f;
        if(i%13==0){r.is_val=true;r.val_step_id=int(i);r.val_loss=.125f;r.val_feedback="reward, \"exact\"";}}
    const auto one=root/"one.csv",batch=root/"batch.csv";
    auto start=std::chrono::steady_clock::now();
    for(const auto& r:records)if(!MetricsCsv::append(one.string(),r))return 1;
    const double single_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    start=std::chrono::steady_clock::now();
    if(!MetricsCsv::appendRange(batch.string(),records.begin(),records.end()))return 2;
    const double batch_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    if(bytes(one)!=bytes(batch)||MetricsCsv::read(batch.string()).size()!=records.size())return 3;
    const auto empty=root/"empty.csv";
    if(!MetricsCsv::appendRange(empty.string(),records.end(),records.end())||std::filesystem::exists(empty))return 4;
    // Both forms must retain legacy columns and repair the same missing newline.
    std::string legacy;
    for(int i=0;i<33;++i){if(i)legacy+=',';legacy+="field"+std::to_string(i);}
    {std::ofstream a(one),b(batch);a<<legacy;b<<legacy;}
    for(const auto& r:records)if(!MetricsCsv::append(one.string(),r))return 5;
    if(!MetricsCsv::appendRange(batch.string(),records.begin(),records.end())||bytes(one)!=bytes(batch))return 6;
    {std::ofstream f(batch);f<<"unsupported,header";}
    if(MetricsCsv::appendRange(batch.string(),records.begin(),records.end())||bytes(batch)!="unsupported,header")return 7;
    std::cout<<"PASS CSV exact batch equivalence; 512 records single_ms="<<single_ms<<" batch_ms="<<batch_ms<<'\n';
}
