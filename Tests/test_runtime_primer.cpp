#include "test_utils.hpp"
#include "Model.hpp"
#include "Serialization/LayerMetadata.hpp"
#include "Serialization/Serialization.hpp"
#include "runtimes/RuntimeRouter.hpp"
#include "runtimes/cpu/CpuRuntime.hpp"
#include "runtimes/vulkan/VulkanRuntime.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void check(bool value, const std::string& message) { if (!value) throw std::runtime_error(message); }
void near(double actual, double expected, const std::string& label, double tolerance = 7e-5) {
    check(std::isfinite(actual) && std::isfinite(expected) &&
          std::abs(actual - expected) <= tolerance * std::max(1., std::abs(expected)),
          label + ": " + std::to_string(actual) + " vs " + std::to_string(expected));
}
template<class T> void compare(const std::vector<float>& actual, const std::vector<T>& expected,
                               const std::string& label, double tolerance = 7e-5) {
    check(actual.size() == expected.size(), label + " shape");
    for (size_t i = 0; i < actual.size(); ++i) near(actual[i], expected[i], label + "[" + std::to_string(i) + "]", tolerance);
}
bool sameBits(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && (a.empty() || !std::memcmp(a.data(), b.data(), a.size() * sizeof(float)));
}

using Inputs = std::vector<const std::vector<float>*>;
std::vector<float> forward(RuntimeRouter& router, AbstractRuntime& runtime, Layer& layer, const Inputs& inputs) {
    check(runtimeCapabilityIsNative(runtime.queryForwardOperationCapability(layer, inputs, true)), layer.type + " forward not native");
    std::vector<std::vector<float>> outputs;
    AbstractRuntime* selected = nullptr;
    check(router.dispatchForwardLayer(inputs, outputs, layer, true, &selected), layer.type + " forward declined");
    check(selected == &runtime && outputs.size() == 1, layer.type + " forward silently fell back");
    return outputs[0];
}
std::vector<std::vector<float>> backward(RuntimeRouter& router, AbstractRuntime& runtime, Layer& layer,
                                        const Inputs& inputs, const std::vector<float>& gradient) {
    check(runtimeCapabilityIsNative(runtime.queryBackwardOperationCapability(layer, inputs, {&gradient}, true)), layer.type + " backward not native");
    std::vector<std::vector<float>> gradients;
    AbstractRuntime* selected = nullptr;
    check(router.dispatchBackwardLayer(inputs, {&gradient}, gradients, layer, true, &selected), layer.type + " backward declined");
    check(selected == &runtime && gradients.size() == inputs.size(), layer.type + " backward silently fell back");
    return gradients;
}

struct Unary {
    const char* name;
    std::function<double(double)> value;
    std::function<double(double)> derivative;
    bool singular_zero = false;
};
const std::vector<Unary>& unaryCases() {
    static const std::vector<Unary> cases{
        {"SquaredReLU", [](double x) { return x > 0 ? x*x : 0.; }, [](double x) { return x > 0 ? 2*x : 0.; }},
        {"Square", [](double x) { return x*x; }, [](double x) { return 2*x; }},
        {"SqrtAbs", [](double x) { return std::sqrt(std::abs(x)); }, [](double x) { return (x > 0 ? 1. : -1.)/(2*std::sqrt(std::abs(x))); }, true},
        {"Exp", [](double x) { return std::exp(x); }, [](double x) { return std::exp(x); }},
        {"LogAbs", [](double x) { return std::log(std::abs(x)); }, [](double x) { return 1/x; }, true},
        {"Abs", [](double x) { return std::abs(x); }, [](double x) { return x > 0 ? 1. : x < 0 ? -1. : 0.; }},
        {"ReciprocalNoNan", [](double x) { return x == 0 ? 0. : 1/x; }, [](double x) { return x == 0 ? 0. : -1/(x*x); }},
        {"Sign", [](double x) { return x > 0 ? 1. : x < 0 ? -1. : 0.; }, [](double) { return 0.; }},
        {"Sin", [](double x) { return std::sin(x); }, [](double x) { return std::cos(x); }},
        {"Cos", [](double x) { return std::cos(x); }, [](double x) { return -std::sin(x); }}
    };
    return cases;
}

void finiteDifference(RuntimeRouter& router, AbstractRuntime& runtime, Layer& layer,
                       std::vector<std::vector<float>> values, const std::vector<float>& upstream,
                       const std::vector<std::vector<float>>& derivatives,
                       const std::function<bool(size_t,size_t)>& allowed = {}) {
    Inputs inputs; for (const auto& value : values) inputs.push_back(&value);
    auto loss = [&]() {
        const auto output = forward(router, runtime, layer, inputs);
        check(output.size() == upstream.size(), "finite difference output shape");
        double result = 0.; for (size_t i = 0; i < output.size(); ++i) result += double(output[i]) * upstream[i];
        return result;
    };
    for (size_t input = 0; input < values.size(); ++input) {
        const size_t length = values[input].size();
        for (size_t index : {size_t(0), length/2, length-1}) {
            if (allowed && !allowed(input, index)) continue;
            const float old = values[input][index], epsilon = .001f;
            values[input][index] = old + epsilon; const double plus = loss();
            values[input][index] = old - epsilon; const double minus = loss(); values[input][index] = old;
            near((plus-minus)/(2*epsilon), derivatives[input][index], layer.type + " finite difference", .003);
        }
    }
}

void unary(RuntimeRouter& router, AbstractRuntime& runtime) {
    for (const auto& operation : unaryCases()) for (size_t length : {size_t(1),size_t(17),size_t(257)}) {
        Layer layer(operation.name, operation.name, 0);
        std::vector<float> input(length), upstream(length);
        std::vector<double> reference(length), derivative(length);
        const float points[] = {-4.f,-1.5f,-.25f,-0.f,0.f,.25f,1.5f,4.f};
        for (size_t i = 0; i < length; ++i) {
            input[i] = points[i % 8];
            if (operation.singular_zero && input[i] == 0.f) input[i] = .75f;
            upstream[i] = .25f * std::cos(float(i) * .27f);
            reference[i] = operation.value(input[i]);
            derivative[i] = operation.derivative(input[i]) * upstream[i];
        }
        compare(forward(router, runtime, layer, {&input}), reference, layer.type + " independent oracle");
        const auto gradient = backward(router, runtime, layer, {&input}, upstream);
        compare(gradient[0], derivative, layer.type + " independent adjoint");
        if (length == 17)
            finiteDifference(router, runtime, layer, {input}, upstream, gradient,
                             [&](size_t, size_t i) { return input[i] != 0.f; });
    }
    Layer rectified("rectified", "SquaredReLU", 0);
    std::vector<float> input{-std::numeric_limits<float>::max(), -0.f, 0.f}, gradient(3, 1.f);
    compare(forward(router, runtime, rectified, {&input}), std::vector<float>(3), "negative extreme squared ReLU", 0.);
    compare(backward(router, runtime, rectified, {&input}, gradient)[0], std::vector<float>(3), "negative extreme squared ReLU derivative", 0.);
    Layer square_root("zero_root", "SqrtAbs", 0); input = {0.f};
    compare(forward(router, runtime, square_root, {&input}), std::vector<float>{0.f}, "zero root forward", 0.);
}

void binary(RuntimeRouter& router, AbstractRuntime& runtime) {
    for (const auto* type : {"Maximum","Minimum"}) for (size_t length : {size_t(1),size_t(17),size_t(257)})
        for (int broadcast : {0,1,2}) {
            Layer layer(type, type, 0);
            std::vector<float> left(broadcast == 1 ? 1 : length), right(broadcast == 2 ? 1 : length), upstream(length);
            for (size_t i = 0; i < left.size(); ++i) left[i] = float(int(i%7)-3)*.5f;
            for (size_t i = 0; i < right.size(); ++i) right[i] = float(int((i+2)%7)-3)*.5f;
            if (broadcast) { if (broadcast == 1) left[0] = 0.f; else right[0] = 0.f; }
            else right[0] = left[0]; // Explicit equal-value derivative convention.
            std::vector<double> expected(length), dl(left.size()), dr(right.size());
            const bool maximum = std::string(type) == "Maximum";
            for (size_t i = 0; i < length; ++i) {
                const size_t a = left.size() == 1 ? 0 : i, b = right.size() == 1 ? 0 : i;
                const double x = left[a], y = right[b];
                // Keep the ordinary scalar-gather fixture free of cancellation;
                // conservative GPU rejection of ambiguous zeros is tested below.
                upstream[i] = .125f * float(broadcast ? 1 + int(i%9) : int(i%9)-4);
                expected[i] = maximum ? std::max(x,y) : std::min(x,y);
                const double fraction = x == y ? .5 : (maximum ? x > y : x < y) ? 1. : 0.;
                dl[a] += upstream[i]*fraction; dr[b] += upstream[i]*(1-fraction);
            }
            compare(forward(router, runtime, layer, {&left,&right}), expected, std::string(type) + " broadcast");
            const auto gradient = backward(router, runtime, layer, {&left,&right}, upstream);
            compare(gradient[0], dl, std::string(type) + " dleft"); compare(gradient[1], dr, std::string(type) + " dright");
            if (length == 17 && !broadcast)
                finiteDifference(router, runtime, layer, {left,right}, upstream, gradient,
                                 [&](size_t, size_t i) { return left[i] != right[i]; });
        }
}

struct ReductionOracle {
    std::vector<double> output, gradient;
};
ReductionOracle reductionOracle(const std::string& operation, const std::vector<int>& original_shape, int requested_axis,
                                 const std::vector<float>& input, const std::vector<float>& upstream = {}) {
    const auto shape = original_shape.empty() ? std::vector<int>{static_cast<int>(input.size())} : original_shape;
    const int axis = requested_axis < 0 ? requested_axis + static_cast<int>(shape.size()) : requested_axis;
    std::vector<size_t> strides(shape.size(),1);
    for (size_t i = shape.size()-1; i > 0; --i) strides[i-1] = strides[i]*shape[i];
    size_t count = 1; for (size_t i = 0; i < shape.size(); ++i) if (int(i) != axis) count *= shape[i];
    std::vector<std::vector<size_t>> groups(count);
    for (size_t i = 0; i < input.size(); ++i) {
        size_t target = 0;
        for (size_t d = 0; d < shape.size(); ++d) if (int(d) != axis)
            target = target * shape[d] + (i/strides[d])%shape[d];
        groups[target].push_back(i);
    }
    ReductionOracle result{std::vector<double>(count), std::vector<double>(input.size())};
    for (size_t group = 0; group < groups.size(); ++group) {
        const auto& indices = groups[group];
        double value = operation == "ReduceProd" ? 1. : operation == "ReduceMin" ?
                       std::numeric_limits<double>::infinity() : operation == "ReduceMax" ?
                       -std::numeric_limits<double>::infinity() : 0.;
        for (size_t index : indices) {
            if (operation == "ReduceProd") value *= input[index];
            else if (operation == "ReduceMin") value = std::min(value, double(input[index]));
            else if (operation == "ReduceMax") value = std::max(value, double(input[index]));
            else value += input[index];
        }
        if (operation == "ReduceMean") value /= indices.size();
        result.output[group] = value;
        if (upstream.empty()) continue;
        size_t ties = 0;
        if (operation == "ReduceMin" || operation == "ReduceMax")
            for (size_t index : indices) ties += input[index] == value;
        for (size_t index : indices) {
            double derivative = 1.;
            if (operation == "ReduceMean") derivative /= indices.size();
            else if (operation == "ReduceMin" || operation == "ReduceMax") derivative = input[index] == value ? 1./ties : 0.;
            else if (operation == "ReduceProd") {
                derivative = 1.; for (size_t other : indices) if (other != index) derivative *= input[other];
            }
            result.gradient[index] = derivative*upstream[group];
        }
    }
    return result;
}

void reductionCase(RuntimeRouter& router, AbstractRuntime& runtime, const char* type,
                    const std::vector<int>& shape, int axis, const std::vector<float>& input, bool numerical) {
    Layer layer(type, type, 0); layer.shape = shape; layer.axis = axis;
    const auto initial = reductionOracle(type, shape, axis, input);
    std::vector<float> upstream(initial.output.size());
    for (size_t i = 0; i < upstream.size(); ++i) upstream[i] = .25f + .125f*float(i%5);
    const auto oracle = reductionOracle(type, shape, axis, input, upstream);
    compare(forward(router, runtime, layer, {&input}), oracle.output, std::string(type) + " axes oracle");
    const auto gradient = backward(router, runtime, layer, {&input}, upstream);
    compare(gradient[0], oracle.gradient, std::string(type) + " reduction adjoint");
    if (numerical) finiteDifference(router, runtime, layer, {input}, upstream, gradient);
}
void reductions(RuntimeRouter& router, AbstractRuntime& runtime) {
    for (const auto* type : {"ReduceSum","ReduceMean","ReduceMin","ReduceMax","ReduceProd"}) {
        std::vector<float> input(24);
        for (size_t i = 0; i < input.size(); ++i) input[i] = .625f + float(i)*.03125f;
        for (int axis : {0,1,2,-1,-2,-3}) reductionCase(router, runtime, type, {2,3,4}, axis, input, axis >= 0);
        for (size_t length : {size_t(1),size_t(17),size_t(257)}) {
            std::vector<float> vector(length);
            for (size_t i = 0; i < vector.size(); ++i) vector[i] = .75f + float(i%7)*.03125f;
            for (int axis : {0,-1}) reductionCase(router, runtime, type, {}, axis, vector, false);
        }
    }
    for (const auto* type : {"ReduceMin","ReduceMax"}) {
        Layer layer(type, type, 0); layer.shape = {2,3}; layer.axis = 1;
        std::vector<float> input = std::string(type) == "ReduceMin" ?
            std::vector<float>{1,1,4, 3,3,3} : std::vector<float>{4,4,1, 3,3,3};
        const std::vector<float> upstream{6,9}, expected{3,3,0, 3,3,3};
        const double tolerance = std::string(runtime.name()) == "CPU" ? 0. :
            3. * std::numeric_limits<float>::epsilon();
        compare(backward(router, runtime, layer, {&input}, upstream)[0], expected, "equal reduction values split gradient", tolerance);
    }
    for (const auto& input : {std::vector<float>{0,2,3}, std::vector<float>{0,0,3}}) {
        Layer layer("zero_safe_product", "ReduceProd", 0); layer.axis = -1;
        const std::vector<float> upstream{1.f};
        const auto oracle = reductionOracle(layer.type, {}, -1, input, upstream);
        compare(forward(router, runtime, layer, {&input}), oracle.output, "zero safe product");
        const auto actual = backward(router, runtime, layer, {&input}, upstream)[0];
        compare(actual, oracle.gradient, "zero safe product adjoint");
    }
}

void triangles(RuntimeRouter& router, AbstractRuntime& runtime) {
    for (const auto& shape : {std::vector<int>{1,1}, std::vector<int>{3,5}, std::vector<int>{2,4,3}, std::vector<int>{2,3,2,5}}) {
        Layer layer("triangle", "LowerTriangular", 0); layer.shape = shape;
        size_t count = 1; for (int extent : shape) count *= extent;
        std::vector<float> input(count), upstream(count), expected(count), derivative(count);
        const size_t rows = shape[shape.size()-2], columns = shape.back();
        for (size_t i = 0; i < count; ++i) {
            input[i] = float(i+1)*.125f; upstream[i] = float(int(i%11)-5)*.25f;
            const bool lower = i%columns <= (i/columns)%rows;
            expected[i] = lower ? input[i] : 0.f; derivative[i] = lower ? upstream[i] : 0.f;
        }
        compare(forward(router, runtime, layer, {&input}), expected, "rectangular triangle", 0.);
        const auto gradient = backward(router, runtime, layer, {&input}, upstream);
        compare(gradient[0], derivative, "rectangular triangle adjoint", 0.);
        if (shape == std::vector<int>{3,5}) finiteDifference(router, runtime, layer, {input}, upstream, gradient);
    }
}

void reject(AbstractRuntime& runtime, Layer& layer, const Inputs& inputs, const std::vector<float>& upstream,
            bool test_forward = true, bool test_backward = true) {
    const std::vector<std::vector<float>> sentinel{{17.f,-23.f}};
    const auto old_weights = layer.weights, old_dw = layer.grad_weights, old_db = layer.grad_bias;
    if (test_forward) {
        auto output = sentinel;
        check(!runtime.forwardLayer(inputs, output, layer, true), layer.type + " invalid forward accepted");
        check(output == sentinel, layer.type + " invalid forward changed output");
    }
    if (test_backward) {
        auto output = sentinel;
        check(!runtime.backwardLayer(inputs, {&upstream}, output, layer, true), layer.type + " invalid backward accepted");
        check(output == sentinel, layer.type + " invalid backward changed input gradients");
    }
    check(sameBits(layer.weights,old_weights) && sameBits(layer.grad_weights,old_dw) && sameBits(layer.grad_bias,old_db),
          layer.type + " invalid operation changed parameter buffers");
}
void rejectionCases(AbstractRuntime& runtime) {
    const float nan = std::numeric_limits<float>::quiet_NaN(), large = std::numeric_limits<float>::max();
    const std::vector<const char*> names{"SquaredReLU","Square","SqrtAbs","Exp","LogAbs","Abs","ReciprocalNoNan","Sign","Sin","Cos",
        "Maximum","Minimum","ReduceSum","ReduceMean","ReduceMin","ReduceMax","ReduceProd","LowerTriangular"};
    for (const auto* name : names) {
        Layer layer(name,name,0); std::vector<float> input{1,2,3,4}, second{2,1,4,3};
        if (layer.type == "LowerTriangular") layer.shape = {2,2};
        const bool binary = layer.type == "Maximum" || layer.type == "Minimum";
        const bool reduction = layer.type.find("Reduce") == 0;
        std::vector<float> upstream(reduction ? 1 : 4, .5f);
        Inputs inputs = binary ? Inputs{&input,&second} : Inputs{&input};
        input[0] = nan; reject(runtime, layer, inputs, upstream); input[0] = 1;
        upstream[0] = nan; reject(runtime, layer, inputs, upstream, false, true); upstream[0] = .5f;
        auto short_gradient = upstream; short_gradient.push_back(.5f);
        reject(runtime, layer, inputs, short_gradient, false, true);
        reject(runtime, layer, {}, upstream);
        layer.weights = {1}; layer.params_count = 1; reject(runtime, layer, inputs, upstream);
        layer.weights.clear(); layer.params_count = 0;
        if (binary) { second.resize(3); reject(runtime, layer, inputs, upstream); }
        if (reduction) {
            layer.shape = {2,3}; reject(runtime, layer, inputs, upstream);
            layer.shape = {2,0,2}; reject(runtime, layer, inputs, upstream);
            layer.shape = {2,2}; layer.axis = 2; reject(runtime, layer, inputs, upstream);
            layer.axis = -3; reject(runtime, layer, inputs, upstream);
            layer.shape.clear(); layer.axis = 1; reject(runtime, layer, inputs, upstream);
        }
        if (layer.type == "LowerTriangular") {
            layer.shape.clear(); reject(runtime, layer, inputs, upstream);
            layer.shape = {4}; reject(runtime, layer, inputs, upstream);
            layer.shape = {2,3}; reject(runtime, layer, inputs, upstream);
        }
    }
    std::vector<float> input{0}, gradient{1};
    Layer logarithm("domain", "LogAbs", 0); reject(runtime, logarithm, {&input}, gradient);
    Layer root("root_domain", "SqrtAbs", 0); reject(runtime, root, {&input}, gradient, false, true);
    for (const auto* name : {"SquaredReLU","Square","Exp","ReduceSum","ReduceProd"}) {
        Layer layer(name,name,0); input = std::string(name).find("Reduce") == 0 ? std::vector<float>{large,large} : std::vector<float>{large};
        gradient.assign(1,1.f);
        // Sum/product can have a finite local adjoint even when their forward
        // result overflows, so only the overflowing result must be declined.
        reject(runtime, layer, {&input}, gradient, true, std::string(name).find("Reduce") != 0);
    }
    // A finite output can still have an overflowing adjoint.
    Layer square("gradient_overflow", "Square", 0); input = {2.f}; gradient = {large};
    reject(runtime, square, {&input}, gradient, false, true);
    Layer reciprocal("reciprocal_overflow", "ReciprocalNoNan", 0); input = {std::numeric_limits<float>::denorm_min()}; gradient = {1};
    reject(runtime, reciprocal, {&input}, gradient);
}

void subnormalFallback(RuntimeRouter& router, AbstractRuntime& gpu, CpuRuntime& cpu) {
    const float tiny = std::numeric_limits<float>::denorm_min();
    std::vector<float> input{tiny,-tiny,1.f}, upstream{.5f,.25f,.125f};
    for (const auto* name : {"Abs","Sign"}) {
        Layer layer(name,name,0);
        reject(gpu,layer,{&input},upstream);
        check(!runtimeCapabilityIsNative(gpu.queryForwardOperationCapability(layer,{&input},true)),"GPU accepted subnormal capability");
        check(!runtimeCapabilityIsNative(gpu.queryBackwardOperationCapability(layer,{&input},{&upstream},true)),"GPU accepted subnormal backward capability");
        std::vector<std::vector<float>> output,gradient; AbstractRuntime*selected=nullptr;
        check(router.dispatchForwardLayer({&input},output,layer,true,&selected) && selected==&cpu,"subnormal forward did not explicitly fall back to CPU");
        compare(output[0],std::string(name)=="Abs"?std::vector<float>{tiny,tiny,1.f}:std::vector<float>{1.f,-1.f,1.f},"subnormal CPU result",0.);
        check(router.dispatchBackwardLayer({&input},{&upstream},gradient,layer,true,&selected) && selected==&cpu,"subnormal backward did not explicitly fall back to CPU");
        compare(gradient[0],std::string(name)=="Abs"?std::vector<float>{.5f,-.25f,.125f}:std::vector<float>(3),"subnormal CPU adjoint",0.);
    }
    Layer absolute("tiny_gradient","Abs",0);
    input={1.f};upstream={tiny};reject(gpu,absolute,{&input},upstream,false,true);
    std::vector<std::vector<float>> gradient;AbstractRuntime*selected=nullptr;
    check(router.dispatchBackwardLayer({&input},{&upstream},gradient,absolute,true,&selected) && selected==&cpu,
          "subnormal upstream did not explicitly fall back to CPU");
    compare(gradient[0],upstream,"subnormal upstream preserved by CPU",0.);
}

// Absolute tolerances suitable for ordinary tensors would allow a flushed zero
// to pass. Compare generated tiny values against independently rounded FP32
// values with a relative tolerance and only a few subnormal ULPs of slack.
void compareTiny(const std::vector<float>& actual,const std::vector<double>& expected,const std::string& label) {
    check(actual.size()==expected.size(),label+" shape");
    for(size_t i=0;i<actual.size();++i) {
        const float rounded=static_cast<float>(expected[i]);
        const double tolerance=1e-5*std::abs(double(rounded))+8.*std::numeric_limits<float>::denorm_min();
        check(std::isfinite(actual[i]) && std::abs(double(actual[i])-rounded)<=tolerance,
              label+" index="+std::to_string(i)+" silently flushed/miscomputed a tiny value");
        if(rounded!=0.f)check(actual[i]!=0.f,label+" lost a nonzero value");
    }
}
void guardedCase(RuntimeRouter& router,AbstractRuntime& runtime,CpuRuntime& cpu,Layer& layer,
                 std::vector<std::vector<float>> values,const std::vector<float>& upstream,
                 const std::vector<double>& expected,const std::vector<std::vector<double>>& expected_gradients) {
    Inputs inputs;for(const auto& value:values)inputs.push_back(&value);
    std::vector<std::vector<float>> reference,reference_gradients;
    check(cpu.forwardLayer(inputs,reference,layer,true),layer.type+" CPU edge forward");
    compareTiny(reference[0],expected,layer.type+" independent tiny forward");
    check(cpu.backwardLayer(inputs,{&upstream},reference_gradients,layer,true),layer.type+" CPU edge backward");
    check(reference_gradients.size()==expected_gradients.size(),"edge adjoint count");
    for(size_t i=0;i<expected_gradients.size();++i)
        compareTiny(reference_gradients[i],expected_gradients[i],layer.type+" independent tiny adjoint");
    if(&runtime==&cpu)return;

    check(runtimeCapabilityIsNative(runtime.queryForwardCapability(layer.type_enum)) &&
          runtimeCapabilityIsNative(runtime.queryBackwardCapability(layer.type_enum)),"guarded operation lacks native type capability");
    const std::vector<std::vector<float>> sentinel{{37.f,-41.f}};
    auto direct=sentinel;
    const bool accepted=runtime.forwardLayer(inputs,direct,layer,true);
    if(accepted)compareTiny(direct[0],expected,layer.type+" native tiny forward");
    else check(direct==sentinel,layer.type+" guarded native refusal changed forward output");
    router.setRuntimes(nullptr,nullptr,&runtime,nullptr,&cpu);
    std::vector<std::vector<float>> routed;AbstractRuntime*selected=nullptr;
    check(router.dispatchForwardLayer(inputs,routed,layer,true,&selected),layer.type+" guarded forward could not fall back");
    check(selected==(accepted?&runtime:static_cast<AbstractRuntime*>(&cpu)),layer.type+" forward fallback selection hidden");
    compareTiny(routed[0],expected,layer.type+" routed tiny forward");

    direct=sentinel;
    const bool backward_accepted=runtime.backwardLayer(inputs,{&upstream},direct,layer,true);
    if(backward_accepted) {
        check(direct.size()==expected_gradients.size(),"native tiny adjoint count");
        for(size_t i=0;i<expected_gradients.size();++i)compareTiny(direct[i],expected_gradients[i],layer.type+" native tiny adjoint");
    }else check(direct==sentinel,layer.type+" guarded native refusal changed backward output");
    router.setRuntimes(nullptr,nullptr,&runtime,nullptr,&cpu);
    check(router.dispatchBackwardLayer(inputs,{&upstream},routed,layer,true,&selected),layer.type+" guarded backward could not fall back");
    check(selected==(backward_accepted?&runtime:static_cast<AbstractRuntime*>(&cpu)),layer.type+" backward fallback selection hidden");
    check(routed.size()==expected_gradients.size(),"routed tiny adjoint count");
    for(size_t i=0;i<expected_gradients.size();++i)compareTiny(routed[i],expected_gradients[i],layer.type+" routed tiny adjoint");
}
void generatedSubnormals(RuntimeRouter& router,AbstractRuntime& runtime,CpuRuntime& cpu) {
    const float small=1e-20f,normal=std::numeric_limits<float>::min();
    Layer square("generated_subnormal_square","Square",0);
    guardedCase(router,runtime,cpu,square,{{small}},{1.f},{double(small)*small},{{2.*small}});
    guardedCase(router,runtime,cpu,square,{{small}},{small},{double(small)*small},{{2.*small*small}});
    Layer exponential("generated_subnormal_exp","Exp",0);
    guardedCase(router,runtime,cpu,exponential,{{-90.f}},{1.f},{std::exp(-90.)},{{std::exp(-90.)}});
    Layer product("generated_subnormal_product","ReduceProd",0);
    guardedCase(router,runtime,cpu,product,{{small,small}},{1.f},{double(small)*small},{{small,small}});
    // The forward rounds to zero, while two derivatives remain nonzero. A y/x
    // implementation loses them; multiplying all other factors preserves them.
    const float very_small=1e-30f;
    guardedCase(router,runtime,cpu,product,{{very_small,very_small,1.f}},{1.f},
                {double(very_small)*very_small},{{very_small,very_small,double(very_small)*very_small}});
    Layer maximum("generated_subnormal_tie","Maximum",0);
    guardedCase(router,runtime,cpu,maximum,{{1.f},{1.f}},{normal},{1.},{{double(normal)*.5},{double(normal)*.5}});
    Layer mean("generated_subnormal_mean_adjoint","ReduceMean",0);
    guardedCase(router,runtime,cpu,mean,{{1.f,2.f}},{normal},{1.5},{{double(normal)*.5,double(normal)*.5}});

    // Exact cancellations share the conservative zero guard. They may execute
    // natively or decline transactionally, but the routed mathematical result
    // and the selected CPU fallback must remain visible and correct.
    for(const auto* type:{"ReduceSum","ReduceMean"}) {
        Layer cancellation(type,type,0);const bool average=std::string(type)=="ReduceMean";
        guardedCase(router,runtime,cpu,cancellation,{{-1.f,1.f}},{1.f},{0.},{{average?.5:1.,average?.5:1.}});
    }
    for(const auto* type:{"Maximum","Minimum"}) {
        Layer cancellation(type,type,0);const float other=std::string(type)=="Maximum"?0.f:2.f;
        guardedCase(router,runtime,cpu,cancellation,{{1.f},{other,other}},{1.f,-1.f},{1.,1.},{{0.},{0.,0.}});
    }
}

void buildGraph(Model& model) {
    model.setModelName("PrimerMathGraph"); model.setDefaultDType("float32");
    model.push("bias","Constant",1);
    auto& bias = model.getMutableLayers().back(); bias.inputs.clear(); bias.output = "bias"; bias.trainable_parameter = true;
    model.push("offset","Add",0);
    auto& offset = model.getMutableLayers().back(); offset.inputs = {"__input__","bias"}; offset.output = "offset";
    std::string previous = "offset";
    for (const auto* type : {"SquaredReLU","Square","SqrtAbs","Exp","LogAbs","Abs","ReciprocalNoNan","Sign","Sin","Cos"}) {
        model.push(type,type,0); auto& layer = model.getMutableLayers().back();
        layer.inputs = {previous}; layer.output = type; previous = type;
    }
    for (const auto* type : {"Maximum","Minimum"}) {
        model.push(type,type,0); auto& layer = model.getMutableLayers().back();
        layer.inputs = {previous,"offset"}; layer.output = type; previous = type;
    }
    const std::vector<std::vector<int>> shapes{{2,2,3},{2,2,1},{2,1,1},{1,1,1},{1,1,1}};
    const int axes[] = {2,1,0,-1,-2};
    size_t index = 0;
    for (const auto* type : {"ReduceSum","ReduceMean","ReduceMin","ReduceMax","ReduceProd"}) {
        model.push(type,type,0); auto& layer = model.getMutableLayers().back();
        layer.inputs = {previous}; layer.output = type; previous = type;
        layer.shape = shapes[index]; layer.axis = axes[index++];
    }
    model.push("triangle","LowerTriangular",0);
    auto& triangle = model.getMutableLayers().back(); triangle.inputs = {previous}; triangle.output = "x"; triangle.shape = {1,1};
}
struct TemporaryDirectory {
    std::filesystem::path path = std::filesystem::temp_directory_path() /
        ("mimir_primer_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    ~TemporaryDirectory() { std::error_code error; std::filesystem::remove_all(path,error); }
};
void modelChecks() {
    for (const auto* backend : {"MIMIR_DISABLE_VULKAN","MIMIR_DISABLE_OPENCL","MIMIR_DISABLE_CUDA","MIMIR_DISABLE_ROCM"}) setTestEnvironment(backend,"1");
    setTestEnvironment("MIMIR_PLANNER_FUSION","0"); setTestEnvironment("MIMIR_PLANNER_BUFFER_REUSE","0");
    // Two branches reading the same input must sum both real adjoints.
    Model branches;
    branches.push("square","Square",0); branches.push("rectified","SquaredReLU",0); branches.push("sum","Add",0);
    auto& layers = branches.getMutableLayers();
    layers[0].inputs={"__input__"}; layers[0].output="square";
    layers[1].inputs={"__input__"}; layers[1].output="rectified";
    layers[2].inputs={"square","rectified"}; layers[2].output="x";
    branches.allocateParams(); branches.zeroGradients();
    const auto output = branches.forwardPass(std::vector<float>{-2,-0.f,0,2},true);
    compare(output,std::vector<float>{4,0,0,8},"branch graph output",0.);
    branches.backwardPass({1,2,3,4});
    compare(branches.getLastInputGradient(),std::vector<float>{-4,0,0,32},"shared input summed adjoints",0.);

    Model source; buildGraph(source); source.allocateParams();
    source.getLayerByName("bias")->getWeights()[0]=.125f;
    std::vector<float> input(12); for(size_t i=0;i<input.size();++i)input[i]=.5f+float(i)*.03125f;
    source.zeroGradients(); const auto expected=source.forwardPass(input,true); source.backwardPass({.75f});
    const auto dx=source.getLastInputGradient(); const auto dw=source.getLayerByName("bias")->grad_weights;
    check(!dw.empty() && std::isfinite(dw[0]) && std::abs(dw[0])>0,"graph trainable constant has no gradient");
    using namespace Mimir::Serialization;
    TemporaryDirectory directory;
    for(auto format:{CheckpointFormat::RawFolder,CheckpointFormat::SafeTensors}) {
        const auto path=directory.path/(format==CheckpointFormat::RawFolder?"raw":"model.safetensors");
        SaveOptions save;save.format=format;save.save_encoder=save.save_tokenizer=false;save.include_git_info=false;
        std::string error;check(save_checkpoint(source,path.string(),save,&error),"save primer graph: "+error);
        for(bool prebuilt:{false,true}) {
            Model restored;if(prebuilt){buildGraph(restored);restored.allocateParams();}
            LoadOptions load;load.format=format;load.load_encoder=load.load_tokenizer=false;load.strict_mode=true;
            check(load_checkpoint(restored,path.string(),load,&error),"load primer graph: "+error);
            check(restored.getLayers().size()==source.getLayers().size(),"checkpoint lost stateless math nodes");
            for(size_t i=0;i<source.getLayers().size();++i) {
                json before,after;LayerMetadata::writeCommon(before,source.getLayers()[i]);LayerMetadata::writeCommon(after,restored.getLayers()[i]);
                check(before==after,"checkpoint primer metadata: "+source.getLayers()[i].name);
            }
            restored.zeroGradients();compare(restored.forwardPass(input,true),expected,"checkpoint graph output",0.);
            restored.backwardPass({.75f});compare(restored.getLastInputGradient(),dx,"checkpoint graph input gradient",0.);
            compare(restored.getLayerByName("bias")->grad_weights,dw,"checkpoint graph parameter gradient",0.);
        }
    }
    std::cout<<"PRIMER_MODEL_SERIALIZATION_OK\n";
}
} // namespace

int main(int argc,char**argv) {
    try {
        setTestEnvironment("MIMIR_RUNTIME_AUTOTUNE","0");
        setTestEnvironment("MIMIR_PLANNER_MODE","static");
        const std::string mode=argc>1?argv[1]:"--cpu";
        if(mode=="--model"){modelChecks();return 0;}
        const bool gpu=mode=="--vulkan";
        RuntimeConfig cfg;cfg.conv_enabled=cfg.linear_enabled=cfg.norm_enabled=cfg.attention_enabled=true;
        cfg.conv_min_ops=cfg.linear_min_ops=cfg.norm_min_elements=cfg.attention_min_ops=0;
        CpuRuntime cpu;check(cpu.initialize(cfg),"CPU initialize");AbstractRuntime*runtime=&cpu;
#ifdef ENABLE_VULKAN
        VulkanRuntime vulkan;
        if(gpu){if(!vulkan.initialize(cfg)){std::cout<<"SKIP Vulkan unavailable\n";return 77;}runtime=&vulkan;}
#else
        if(gpu){std::cout<<"SKIP Vulkan not compiled\n";return 77;}
#endif
        auto&router=RuntimeRouter::instance();router.setActivators({},{},{},{},{});
        router.setRuntimes(nullptr,nullptr,gpu?runtime:nullptr,nullptr,&cpu);
        unary(router,*runtime);binary(router,*runtime);reductions(router,*runtime);triangles(router,*runtime);rejectionCases(*runtime);
        if(gpu)subnormalFallback(router,*runtime,cpu);
        generatedSubnormals(router,*runtime,cpu);
        router.setRuntimes(nullptr,nullptr,nullptr,nullptr,nullptr);
        std::cout<<"PRIMER_RUNTIME_OK backend="<<(gpu?"VULKAN":"CPU")<<"\n";return 0;
    }catch(const std::exception&error){std::cerr<<"PRIMER_RUNTIME_FAILED: "<<error.what()<<'\n';return 1;}
}
