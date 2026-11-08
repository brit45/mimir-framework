#include "AsyncMonitor.hpp"
#include "Models/Registry/ModelArchitectures.hpp"
#include "viz/LatentSpaceProjection.hpp"
#include "scriptings/Lua/luaScripting/LuaScripting.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <thread>

namespace {
using namespace mimir::viz;
void check(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
LatentSpaceRawSnapshot fixture(uint64_t generation,const std::string& name="Custom model",const std::string& architecture="custom") {
    LatentSpaceRawSnapshot raw;raw.generation=generation;raw.model_name=name;raw.architecture=architecture;
    raw.optimizer_step=23;raw.feature_dimension=4;
    const std::vector<std::vector<float>> values{{1,0},{0,1,1},{-1,0,0,1},{0,-1,0,-1}};
    for(size_t i=0;i<values.size();++i) {
        LatentVectorSample sample;sample.label="vector "+std::to_string(i);sample.values=values[i];
        sample.original_dimension=values[i].size();sample.layer=int(i);
        double squared=0;for(float value:sample.values)squared+=double(value)*value;
        sample.norm=float(std::sqrt(squared));raw.vectors.push_back(std::move(sample));
    }
    return raw;
}
void minimalProjectionCases() {
    auto raw=fixture(1);
    raw.d_model=0; // Generic projections do not depend on a language-model width.
    const auto heterogeneous=projectLatentSpace3D(raw);
    check(heterogeneous.points.size()==4,"heterogeneous feature dimensions rejected");
    std::array<double,3> axis_energy{{0,0,0}};
    for(const auto& point:heterogeneous.points)for(size_t axis=0;axis<3;++axis) {
        const double value=point.position[axis];
        check(std::isfinite(value),"non-finite heterogeneous projection");
        axis_energy[axis]+=value*value;
    }
    const double total_energy=axis_energy[0]+axis_energy[1]+axis_energy[2];
    check(total_energy>0,"nondegenerate fixture collapsed to origin");
    // Four centered samples have rank <=3, so these axes contain all variance.
    // A separate normalization of each axis would destroy these energy ratios.
    for(size_t axis=0;axis<3;++axis)
        check(std::abs(axis_energy[axis]/total_energy-double(heterogeneous.explained_variance_percent[axis])/100.0)<1e-3,
              "display coordinates distort PCA explained variance");
    raw.vectors.resize(1);
    auto one=projectLatentSpace3D(raw);
    check(one.points.size()==1,"single-vector model cannot be projected");
    for(float value:one.points[0].position)check(value==0,"single vector must be at the origin");
    raw=fixture(1);raw.vectors.resize(2);
    auto two=projectLatentSpace3D(raw);
    check(two.points.size()==2,"two-vector model cannot be projected");
    for(const auto& point:two.points) {
        for(float value:point.position)check(std::isfinite(value),"two-vector projection is non-finite");
        check(point.position[1]==0 && point.position[2]==0,"missing PCA axes are not zero");
    }
    raw=fixture(1);
    for(auto& sample:raw.vectors){std::fill(sample.values.begin(),sample.values.end(),0.f);sample.norm=0;}
    auto zero=projectLatentSpace3D(raw);
    check(zero.points.size()==raw.vectors.size(),"zero-valued parameters were omitted");
    for(const auto& point:zero.points)for(float value:point.position)check(value==0,"all-zero parameters must project to the origin");
    raw=fixture(1);raw.vectors[0].values[0]=std::numeric_limits<float>::quiet_NaN();
    raw.vectors[1].values[0]=std::numeric_limits<float>::infinity();
    auto finite=projectLatentSpace3D(raw);
    check(finite.points.size()==2,"invalid vectors were projected or valid vectors lost");
}
void staleControlCase() {
    LatentSpaceControl control;
    const uint64_t old_generation=control.resetSession("Old MLP","basic_mlp",true);
    control.requestSnapshot();uint64_t token=0;
    check(control.beginCapture(token) && token==old_generation,"request was not consumed");
    auto old=fixture(token,"Old MLP","basic_mlp");
    const auto stale_projection=projectLatentSpace3D(old);
    check(control.submitRawSnapshot(old),"current-generation snapshot rejected");
    const auto worker_snapshot=control.takeRawSnapshot();check(bool(worker_snapshot),"worker did not take old snapshot");
    // Leave another pending snapshot as well: reset must clear its mailbox.
    check(control.submitRawSnapshot(old),"pending old snapshot rejected");
    const uint64_t current=control.resetSession("New VAE","vae",true);
    // Simulate the exact worker race: it already owns immutable old data when
    // the UI switches the model, before entering the Projecting state.
    check(!control.beginProjection(worker_snapshot->generation,worker_snapshot->revision),"old worker started projecting after model reset");
    check(control.state.load()==LatentSpaceControl::State::Ready,"old worker overwrote new Ready state");
    check(current!=old_generation,"new model reused snapshot generation");
    check(!control.raw_ready.load() && !control.takeRawSnapshot() && !control.projectionSnapshot(),"new model kept old raw/projection");
    check(!control.submitRawSnapshot(old),"stale raw snapshot accepted");
    check(!control.publishProjection(stale_projection),"stale projection accepted");
    control.fail("stale failure",old_generation);
    check(control.state.load()==LatentSpaceControl::State::Ready,"stale failure replaced current state");
    control.requestSnapshot();check(control.beginCapture(token) && token==current,"new model request rejected");
    auto fresh=fixture(token,"New VAE","vae");
    check(control.submitRawSnapshot(fresh),"fresh raw rejected");
    const auto consumed=control.takeRawSnapshot();check(consumed && consumed->generation==current,"fresh raw mailbox lost generation");
    check(control.publishProjection(projectLatentSpace3D(*consumed)),"fresh projection rejected");
    auto projected=control.projectionSnapshot();
    check(projected && projected->model_name=="New VAE" && projected->architecture=="vae","wrong model projection published");
    // A newer capture of the same model also supersedes in-flight work.
    check(control.submitRawSnapshot(fixture(current)),"same-session raw rejected");
    const auto superseded=control.takeRawSnapshot();
    check(superseded && control.beginProjection(current,superseded->revision),"old projection did not begin");
    control.requestSnapshot();check(control.beginCapture(token,true),"forced refresh while projecting rejected");
    check(control.submitRawSnapshot(fixture(current)),"replacement raw rejected");
    check(!control.beginProjection(current,superseded->revision),"superseded revision began projecting");
    check(!control.publishProjection(projectLatentSpace3D(*superseded)),"superseded same-model projection accepted");
    const auto newest=control.takeRawSnapshot();check(newest && newest->revision>superseded->revision,"raw revision did not advance");
    check(control.publishProjection(projectLatentSpace3D(*newest)),"newest revision publication rejected");
    control.fail("superseded failure",current,superseded->revision);
    check(control.state.load()==LatentSpaceControl::State::Complete,"superseded failure replaced fresh projection");
    // Inference is eligible outside training; stale model hooks must not steal
    // a pending request belonging to the new model.
    control.training_active.store(false);control.requestSnapshot();
    check(!control.beginCapture(token,false,old_generation) && control.requested.load(),"stale model consumed fresh request");
    check(control.beginCapture(token),"inactive training prevented inference capture");
}
void realtimeControlCase() {
    using Clock=LatentSpaceControl::Clock;
    const auto start=Clock::time_point{}+std::chrono::seconds(10);
    LatentSpaceControl control;
    auto generation=control.resetSession("Realtime model","custom",true);
    uint64_t token=0,revision=0;
    auto begin=[&](Clock::time_point now) {return control.beginCapture(token,false,generation,now,&revision);};
    auto complete=[&](size_t step) {
        auto raw=fixture(token);raw.revision=revision;raw.optimizer_step=step;
        check(control.submitRawSnapshot(std::move(raw)),"reserved realtime raw rejected");
        const auto captured=control.takeRawSnapshot();check(bool(captured),"realtime raw mailbox lost capture");
        check(control.beginProjection(captured->generation,captured->revision),"realtime projection did not begin");
        check(control.publishProjection(projectLatentSpace3D(*captured)),"current realtime projection rejected");
        return captured;
    };
    check(!control.realtime_enabled.load() && !control.view_active.load(),"realtime enabled by default");
    check(!begin(start),"disabled realtime started capture without a request");
    control.setViewActive(true);
    check(!begin(start),"open view enabled realtime without checkbox");
    control.setRealtimeEnabled(true);
    check(!control.requested.load() && begin(start),"realtime did not start automatically");
    const uint64_t first_revision=revision;
    check(!begin(start+std::chrono::milliseconds(1)),"automatic capture superseded an unfinished capture");
    auto first=fixture(token);first.revision=first_revision;
    check(control.submitRawSnapshot(std::move(first)),"first automatic snapshot rejected");
    check(!begin(start+std::chrono::milliseconds(2)),"automatic capture superseded a pending raw snapshot");
    const auto pending=control.takeRawSnapshot();check(bool(pending),"automatic snapshot disappeared");
    check(!begin(start+std::chrono::milliseconds(3)),"taking the mailbox released the in-flight guard too soon");
    check(control.beginProjection(pending->generation,pending->revision),"automatic PCA did not begin");
    check(!begin(start+std::chrono::milliseconds(4)),"automatic refresh starved an in-progress PCA");
    check(control.publishProjection(projectLatentSpace3D(*pending)),"blocked automatic refresh invalidated current publication");
    check(!begin(start+std::chrono::milliseconds(249)),"realtime cadence shorter than 250 ms");
    control.setRealtimeEnabled(true);control.setViewActive(true);
    check(!begin(start+std::chrono::milliseconds(249)),"idempotent setters bypassed realtime cadence");
    check(begin(start+LatentSpaceControl::realtime_interval),"realtime capture refused at cadence boundary");
    auto second=fixture(token);second.revision=revision;second.optimizer_step=2;
    check(control.submitRawSnapshot(std::move(second)),"second automatic raw rejected");
    const auto in_progress=control.takeRawSnapshot();check(bool(in_progress),"second automatic capture lost");
    check(control.beginProjection(token,in_progress->revision),"second automatic PCA refused");
    control.setRealtimeEnabled(false);control.setViewActive(false);
    check(!begin(start+std::chrono::milliseconds(250)),"disabled/closed view admitted automatic capture");
    check(control.publishProjection(projectLatentSpace3D(*in_progress)),"disabling realtime cancelled an in-progress publication");
    check(!begin(start+std::chrono::milliseconds(250)),"disabled realtime refreshed after publication");
    // Manual refresh and forced snapshots retain their original behavior.
    control.requestSnapshot();
    check(begin(start+std::chrono::milliseconds(251)),"manual request obeyed realtime cadence or view gates");
    const auto old=complete(3);
    check(control.beginCapture(token,true,generation,start+std::chrono::milliseconds(252),&revision),"forced capture obeyed realtime gates");
    complete(4);
    control.setRealtimeEnabled(true);
    check(!begin(start+std::chrono::milliseconds(252)),"realtime ran with closed view");
    control.setViewActive(true);
    generation=control.resetSession("Replacement model","custom_new",true);
    check(control.realtime_enabled.load() && control.view_active.load(),"model reset lost realtime/view preferences");
    check(!control.beginCapture(token,false,old->generation,start+std::chrono::milliseconds(253)),"stale model admitted automatic capture");
    check(!control.submitRawSnapshot(*old) && !control.publishProjection(projectLatentSpace3D(*old)),"stale realtime snapshot replaced a new session");
    check(begin(start+std::chrono::milliseconds(253)),"model reset did not reset automatic cadence");
    complete(5);
    check(control.projectionSnapshot()->generation==generation,"automatic reset publication used stale generation");
}
template<class Predicate>
void waitFor(Predicate predicate,const char* failure) {
    const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(5);
    while(!predicate() && std::chrono::steady_clock::now()<deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    check(predicate(),failure);
}
void monitorPublicationCase() {
    AsyncMonitor monitor;
    for(const char* architecture:{"basic_mlp","vae","vae_conv","transformer","causal_lm","unregistered_custom"}) {
        monitor.configureLatentSpace(json{{"type",architecture}});
        check(monitor.latentSpaceControl()->available.load(),"3D control still gated to one architecture");
    }
    auto model=ModelArchitectures::create("basic_mlp",json{{"input_dim",4},{"hidden_dim",8},{"output_dim",3},{"hidden_layers",1},{"dropout",0.0}});
    model->allocateParams();model->initializeWeights("xavier",882);
    monitor.configureLatentSpace(model->modelConfig);
    const auto control=monitor.latentSpaceControl();model->bindLatentSpaceControl(control);
    control->requestSnapshot();
    const std::vector<float> input{-.4f,.2f,.7f,.1f};
    model->forwardPass(input,false); // Real generic Model safe point -> monitor -> PCA mailbox.
    waitFor([&]{return control->state.load()==LatentSpaceControl::State::Complete;},"inference did not publish a projection");
    auto projection=control->projectionSnapshot();
    check(projection && projection->architecture=="basic_mlp" && !projection->points.empty(),"headless monitor published wrong/empty model");
    check(projection->generation==control->generation.load(),"monitor published wrong generation");
    model->forwardPass(input,true);
    model->backwardPass(std::vector<float>(3,.1f));
    control->requestSnapshot();
    Optimizer optimizer;optimizer.type=OptimizerType::SGD;
    model->optimizerStep(optimizer,.001f); // Actual optimizer safe point, after weight updates.
    waitFor([&]{const auto p=control->projectionSnapshot();return p && p->optimizer_step==size_t(optimizer.step);},
            "optimizer step did not publish its learned weights");
    projection=control->projectionSnapshot();
    check(projection->optimizer_step==size_t(optimizer.step) && optimizer.step>0,"optimizer snapshot has stale step");
    // Resetting an architecture while an old result exists clears that result.
    monitor.configureLatentSpace(json{{"type","vae"}});
    check(!control->projectionSnapshot(),"reconfiguration displayed previous architecture");
    uint64_t token=0;check(control->beginCapture(token,true),"explicit capture after reset rejected");
    monitor.submitLatentSpaceSnapshot(fixture(token,"Current VAE","vae"));
    waitFor([&]{const auto p=control->projectionSnapshot();return p && p->generation==token;},"generic raw-to-PCA worker publication failed");
    projection=control->projectionSnapshot();
    check(projection->model_name=="Current VAE" && projection->architecture=="vae","old model replaced fresh projection");
    monitor.stop();
}

void realtimeOptimizerPublicationCase() {
    AsyncMonitor monitor;
    auto model=ModelArchitectures::create("basic_mlp",json{{"input_dim",4},{"hidden_dim",8},{"output_dim",3},{"hidden_layers",1},{"dropout",0.0}});
    model->allocateParams();model->initializeWeights("xavier",177);
    monitor.configureLatentSpace(model->modelConfig);
    const auto control=monitor.latentSpaceControl();model->bindLatentSpaceControl(control);
    control->setViewActive(true);control->setRealtimeEnabled(true);
    const auto generation=control->generation.load();
    const auto found=std::find_if(model->getLayers().begin(),model->getLayers().end(),[](const Layer& layer){return layer.getWeightsSize()>0;});
    check(found!=model->getLayers().end(),"realtime fixture has no learned parameters");
    const auto* parameters=found->getWeights();const size_t parameter_count=found->getWeightsSize();
    const std::vector<float> input{-.4f,.2f,.7f,.1f};
    Optimizer optimizer;optimizer.type=OptimizerType::SGD;
    uint64_t previous_revision=0;
    for(size_t step=1;step<=2;++step) {
        model->zeroGradients();model->forwardPass(input,true);model->backwardPass(std::vector<float>(3,.1f));
        const std::vector<float> before(parameters,parameters+parameter_count);
        if(step==2)std::this_thread::sleep_for(LatentSpaceControl::realtime_interval+std::chrono::milliseconds(20));
        check(!control->requested.load(),"automatic fixture unexpectedly requested a manual capture");
        model->optimizerStep(optimizer,.001f);
        check(size_t(optimizer.step)==step,"realtime optimizer did not update its step");
        check(!std::equal(before.begin(),before.end(),parameters),"realtime fixture did not update learned weights");
        waitFor([&]{const auto p=control->projectionSnapshot();return p && p->optimizer_step==step;},
                "automatic optimizer update did not publish fresh learned vectors");
        const auto p=control->projectionSnapshot();
        check(p->generation==generation && p->revision>previous_revision && !p->points.empty(),"automatic publication is stale or empty");
        previous_revision=p->revision;
    }
    control->setRealtimeEnabled(false);
    model->captureRequestedLatentSpace(optimizer.step);
    check(control->projectionSnapshot()->revision==previous_revision && !control->raw_ready.load(),"disabled realtime enqueued another snapshot");
    control->setRealtimeEnabled(true);control->setViewActive(false);
    model->captureRequestedLatentSpace(optimizer.step);
    check(control->projectionSnapshot()->revision==previous_revision && !control->raw_ready.load(),"closed view enqueued another snapshot");
    monitor.stop();
}

void scriptingBindingCase() {
    LuaScripting lua;
    auto& context=LuaContext::getInstance();
    auto monitor=std::make_shared<AsyncMonitor>();
    context.asyncMonitor=monitor;
    monitor->start(false,false);
    check(lua.executeScript(R"(
        assert(Mimir.Model.create("basic_mlp", {
            input_dim=4,hidden_dim=6,output_dim=3,hidden_layers=1,dropout=0
        }))
        assert(Mimir.Model.allocate_params())
        assert(Mimir.Model.init_weights("xavier",51))
    )"),"Lua model setup failed");
    auto control=monitor->latentSpaceControl();
    const auto old_generation=control->generation.load();
    control->requestSnapshot();
    check(lua.executeScript("assert(Mimir.Model.forward({1,2,3,4},false))"),"Lua forward failed");
    waitFor([&]{return control->state.load()==LatentSpaceControl::State::Complete;},
            "Lua model binding did not publish its projection");
    check(control->projectionSnapshot()->architecture=="basic_mlp","Lua projection used another architecture");
    check(lua.executeScript(R"(
        assert(Mimir.Model.create_empty("identity_graph",{}))
        assert(Mimir.Model.push_layer("identity","Identity",0))
        assert(Mimir.Model.allocate_params())
    )"),"Lua custom graph setup failed");
    check(control->generation.load()!=old_generation && !control->projectionSnapshot(),
          "Lua model replacement kept the old projection");
    control->requestSnapshot();
    check(lua.executeScript("assert(Mimir.Model.forward({1,2,3,4},false))"),"Lua custom forward failed");
    waitFor([&]{return control->state.load()==LatentSpaceControl::State::Complete;},
            "parameter-free custom graph did not publish its activations");
    check(control->projectionSnapshot()->architecture=="identity_graph","custom graph projection used stale metadata");
    monitor->stop();
    context.asyncMonitor.reset();
    context.currentModel.reset();
}
}
int main() {
    try {
        Model::setFrameworkLogsSuppressed(true);
        minimalProjectionCases();staleControlCase();realtimeControlCase();monitorPublicationCase();realtimeOptimizerPublicationCase();scriptingBindingCase();
        std::cout<<"GENERIC_LATENT_CONTROL_PUBLICATION_OK heterogeneous PCA, minimal models, stale sessions, realtime cadence, inference/optimizer safe points\n";
        return 0;
    } catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
