#include "T5Model.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <limits>

T5Model::T5Model() { setModelName("T5Model"); setHasEncoder(false); }
void T5Model::buildFromConfig(const Config& cfg) { buildInto(*this, cfg); }

void T5Model::buildInto(Model& m, const Config& c) {
    if (c.vocab_size < 2 || c.seq_len < 1 || c.d_model < 1 || c.num_heads < 1 ||
        c.d_model % c.num_heads || c.num_layers < 1 || c.mlp_hidden < 1 ||
        c.relative_attention_num_buckets < 4 || c.relative_attention_num_buckets % 4 ||
        c.relative_attention_max_distance <= c.relative_attention_num_buckets / 4 ||
        !(c.norm_eps > 0) || !std::isfinite(c.norm_eps) ||
        c.padding_idx < 0 || c.padding_idx >= c.vocab_size)
        throw std::invalid_argument("T5: invalid dimensions, buckets, padding or epsilon");
    const auto limit=std::numeric_limits<int>::max();
    if (static_cast<long long>(c.seq_len)*c.seq_len > limit/c.num_heads ||
        static_cast<long long>(c.seq_len)*std::max(c.vocab_size,c.d_model) > limit)
        throw std::invalid_argument("T5: tensor dimensions exceed runtime limits");
    const int s=c.seq_len, d=c.d_model, h=c.num_heads, hd=d/h;
    m.getMutableLayers().clear(); m.setHasEncoder(false); m.setModelName("T5Model");
    m.modelConfig={{"type","t5"},{"task","text_denoising"},{"vocab_size",c.vocab_size},
        {"seq_len",s},{"d_model",d},{"num_layers",c.num_layers},{"num_heads",h},
        {"mlp_hidden",c.mlp_hidden},{"padding_idx",c.padding_idx},{"norm_eps",c.norm_eps},
        {"relative_attention_num_buckets",c.relative_attention_num_buckets},
        {"relative_attention_max_distance",c.relative_attention_max_distance},
        {"encoder_only",c.encoder_only},{"input_dim",s},{"output_dim",s*(c.encoder_only?d:c.vocab_size)},
        {"initialize_from_layer_weights",true},{"frozen_layer_prefixes",{"t5/fixed/"}}};
    auto op=[&](const std::string& n,const std::string& type,std::vector<std::string> in,size_t count=0)->Layer* {
        m.push(n,type,count); auto* l=m.getLayerByName(n); l->inputs=std::move(in); l->output=n; return l;
    };
    auto linear=[&](const std::string& n,const std::string& in,int a,int b) {
        auto* l=op(n,"Linear",{in},size_t(a)*b); l->in_features=a;l->out_features=b;l->seq_len=s;l->use_bias=false;
        return n;
    };
    auto norm=[&](const std::string& n,const std::string& in) {
        auto* l=op(n,"RMSNorm",{in},d);l->in_features=d;l->eps=c.norm_eps;l->affine=true;l->use_bias=false;return n;
    };
    auto permute=[&](const std::string& n,const std::string& in,std::vector<int> shape,std::vector<int> dims) {
        auto* l=op(n,"Permute",{in});l->shape=std::move(shape);l->permute_dims=std::move(dims);return n;
    };
    auto bmm=[&](const std::string& n,const std::string& a,const std::string& b,int rows,int inner,int cols) {
        auto* l=op(n,"BatchMatMul",{a,b});l->seq_len=h;l->in_features=rows;l->out_features=inner;l->embed_dim=cols;return n;
    };
    auto* emb=op("t5/shared","Embedding",{"text_ids"},size_t(c.vocab_size)*d);
    emb->vocab_size=c.vocab_size;emb->embed_dim=d;emb->padding_idx=c.padding_idx;emb->seq_len=s;
    auto* mask=op("t5/fixed/padding_mask","Embedding",{"text_ids"},c.vocab_size);
    mask->vocab_size=c.vocab_size;mask->embed_dim=1;mask->padding_idx=-1;mask->seq_len=s;
    mask->weights.assign(c.vocab_size,0.0f);mask->weights[c.padding_idx]=-1e9f;
    // Bucket indices are immutable graph data. Relative bias is learned and shared across blocks.
    auto* indices=op("t5/fixed/buckets","Constant",{},size_t(s)*s);
    indices->weights.resize(size_t(s)*s);
    const int half=c.relative_attention_num_buckets/2, exact=half/2;
    for(int i=0;i<s;++i) for(int j=0;j<s;++j) {
        int delta=j-i, dist=std::abs(delta), bucket=dist;
        if(dist>=exact) bucket=std::min(half-1,exact+int(std::log(float(dist)/exact)/
            std::log(float(c.relative_attention_max_distance)/exact)*(half-exact)));
        indices->weights[size_t(i)*s+j]=float(bucket+(delta>0?half:0));
    }
    auto* bias=op("t5/relative_bias","Embedding",{"t5/fixed/buckets"},size_t(c.relative_attention_num_buckets)*h);
    bias->vocab_size=c.relative_attention_num_buckets;bias->embed_dim=h;bias->padding_idx=-1;
    permute("t5/position_bias","t5/relative_bias",{s,s,h},{2,0,1});
    std::string x="t5/shared";
    for(int i=0;i<c.num_layers;++i) {
        const std::string p="t5/encoder/block"+std::to_string(i);
        const auto n=norm(p+"/norm1",x);
        auto q=linear(p+"/q",n,d,d),k=linear(p+"/k",n,d,d),v=linear(p+"/v",n,d,d);
        q=permute(p+"/qh",q,{s,h,hd},{1,0,2});
        k=permute(p+"/kh",k,{s,h,hd},{1,2,0});
        v=permute(p+"/vh",v,{s,h,hd},{1,0,2});
        // T5 uses unscaled dot products, with learned relative scalar bias.
        auto scores=bmm(p+"/scores",q,k,s,hd,s);
        op(p+"/biased","Add",{scores,"t5/position_bias"});
        op(p+"/masked","Add",{p+"/biased","t5/fixed/padding_mask"});
        auto* sm=op(p+"/probs","Softmax",{p+"/masked"});sm->shape={h,s,s};sm->axis=2;
        auto ctx=bmm(p+"/context",p+"/probs",v,s,s,hd);
        ctx=permute(p+"/tokens",ctx,{h,s,hd},{1,0,2});
        auto out=linear(p+"/o",ctx,d,d);
        op(p+"/res1","Add",{x,out});
        auto ff=linear(p+"/wi",norm(p+"/norm2",p+"/res1"),d,c.mlp_hidden);
        op(p+"/relu","ReLU",{ff});
        ff=linear(p+"/wo",p+"/relu",c.mlp_hidden,d);
        op(p+"/res2","Add",{p+"/res1",ff});x=p+"/res2";
    }
    x=norm("t5/encoder/final_norm",x);
    if(!c.encoder_only) {
        auto* head=op("t5/lm_head","Linear",{x});head->in_features=d;head->out_features=c.vocab_size;
        head->seq_len=s;head->use_bias=false;head->shared_weights_from="t5/shared";x="t5/lm_head";
    }
    op("t5/output","Identity",{x})->output="x";
}
