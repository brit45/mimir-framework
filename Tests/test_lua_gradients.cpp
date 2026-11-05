#include "test_utils.hpp"
#include "scriptings/Lua/luaScripting/LuaScripting.hpp"
#include <memory>
#include <vector>

int main() {
    LuaScripting lua;
    auto& context = LuaContext::getInstance();
    context.resetRuntimeState();
    TASSERT_TRUE(lua.executeScript("local g,err=Mimir.Model.get_gradients(); assert(g==nil and type(err)=='string')"));

    auto model = std::make_shared<Model>();
    model->push("first", "Linear", 0);
    model->push("empty", "Identity", 0);
    model->push("last", "Linear", 0);
    auto& layers = model->getMutableLayers();
    layers[0].grad_weights = {1.f, -2.f, 3.f};
    layers[0].grad_bias = {4.f, -5.f};
    layers[2].grad_weights = {6.f, 7.f};
    layers[2].grad_bias = {-8.f};
    context.currentModel = model;
    TASSERT_TRUE(lua.executeScript(R"(
        local actual = assert(Mimir.Model.get_gradients())
        local expected = {1,-2,3,4,-5,6,7,-8}
        assert(#actual == #expected)
        for i,v in ipairs(expected) do assert(actual[i] == v, 'gradient index '..i) end
        actual[1] = 999 -- The returned table must not alias native gradients.
        assert(Mimir.Model.get_gradients()[1] == 1)
    )"));
    const auto indexed = model->getGradients();
    const std::vector<float> expected = {1,-2,3,4,-5,6,7,-8};
    for (size_t i=0; i<expected.size(); ++i) TASSERT_TRUE(indexed.get(i) == expected[i]);

    // Stress the array part, including its non-power-of-two tail.
    for (auto& layer : layers) { layer.grad_weights.clear(); layer.grad_bias.clear(); }
    layers[2].grad_weights.resize(131075);
    for (size_t i=0; i<layers[2].grad_weights.size(); ++i)
        layers[2].grad_weights[i] = float(int(i%101)-50)*.125f;
    TASSERT_TRUE(lua.executeScript(R"(
        local g=assert(Mimir.Model.get_gradients())
        assert(#g==131075)
        for i=1,#g do assert(g[i]==((i-1)%101-50)*.125) end
    )"));
    layers[2].grad_weights.clear();
    TASSERT_TRUE(lua.executeScript("assert(#Mimir.Model.get_gradients()==0)"));
    TASSERT_TRUE(lua.executeScript(R"(
        local B=dofile('scripts/benchmarks/common.lua')
        local values={0,-1,1,1e300}
        assert(B.vector(values,4)==values)
        for _,bad in ipairs({math.huge,-math.huge,0/0,false,'1',{}}) do
            local ok,err=pcall(B.vector,{1,bad})
            assert(not ok and err:find('NaN/Inf à l’indice 2',1,true))
        end
        assert(not pcall(B.vector,{}))
        assert(not pcall(B.vector,{1},2))
    )"));
    context.resetRuntimeState();
    return 0;
}
