#include "test_utils.hpp"
#include "Model.hpp"
#include "AsyncMonitor.hpp"
#include <thread>
#include <chrono>
#include <cmath>

static size_t index_of(const std::shared_ptr<LiveModelConfig>& config, const std::string& key) {
    const auto rows = config->snapshot();
    for (size_t i=0; i<rows.size(); ++i) if(rows[i].key==key) return i;
    throw std::runtime_error("missing row " + key);
}
int main(int argc, char** argv) {
    Model model;
    model.modelConfig = {{"type","basic_mlp"},{"hidden_dim",16},{"training",{{"epochs",10}}}};
    model.push("linear", "Linear", 1);
    auto& layer = model.getMutableLayers()[0];
    layer.inputs={"__input__"}; layer.output="x";
    layer.in_features=layer.out_features=1; layer.use_bias=false;
    model.allocateParams(); layer.getWeights()[0]=1;
    Optimizer opt;
    opt.type=OptimizerType::ADAMW; opt.decay_strategy=LRDecayStrategy::NONE;
    opt.initial_lr=0.1f; opt.weight_decay=0;
    model.publishRuntimeConfiguration(&opt);
    auto config=model.runtimeConfiguration();
    std::string error;
    TASSERT_TRUE(!config->request("hidden_dim","32",error));
    TASSERT_TRUE(!config->request("training/epochs","2",error));
    TASSERT_TRUE(!config->request("unknown","2",error));
    TASSERT_TRUE(!config->request("grad_clip_norm","nan",error));
    TASSERT_TRUE(!config->request("grad_clip_norm","-1",error));
    TASSERT_TRUE(!config->request("beta1","1",error));
    TASSERT_TRUE(!config->request("epsilon","0",error));
    TASSERT_TRUE(config->request("weight_decay","0.5",error));
    TASSERT_TRUE(config->request("grad_clip_norm","1",error));
    TASSERT_TRUE(!model.modelConfig.contains("grad_clip_norm"));
    // Forward consumes model changes; optimizer changes wait for its boundary.
    model.forwardPass(std::vector<float>{100},true);
    TASSERT_TRUE(model.modelConfig["grad_clip_norm"]==1);
    TASSERT_TRUE(opt.weight_decay==0);
    model.backwardPass({1});
    model.optimizerStep(opt,0.1f,nullptr);
    TASSERT_NEAR(opt.weight_decay,0.5f,1e-6f);
    TASSERT_NEAR(layer.getWeights()[0],0.85f,1e-5f);
    TASSERT_TRUE(model.modelConfig["hidden_dim"]==16);

    AsyncMonitor monitor;
    monitor.bindRuntimeConfiguration(config);
    monitor.updateRuntimeTrainParams(0.001f,0,0,0,"mse");
#ifdef ENABLE_VIZ
    if(argc>1 && std::string(argv[1])=="--viz") {
        Visualizer viz(json{{"visualization", {{"enabled",true},{"window_width",1000},
            {"window_height",720},{"window_title","Mimir configuration test"},{"fps_limit",30}}}});
        viz.setLossLogEnabled(false);
        viz.setRuntimeConfiguration(config);
        TASSERT_TRUE(viz.initialize());
        bool applied=false;
        std::cout << "CONFIG_READY " << index_of(config,"learning_rate") << std::endl;
        const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(30);
        while(viz.isOpen() && std::chrono::steady_clock::now()<deadline) {
            viz.processEvents(); viz.update();
            const auto live=monitor.liveTrainParamsSnapshot();
            if(live.overrides_enabled && std::abs(live.lr-0.02f)<1e-6f) {applied=true;break;}
        }
        viz.shutdown();
        TASSERT_TRUE(applied);
        std::cout << "CONFIG_VIZ_APPLIED" << std::endl;
        return 0;
    }
#endif
    if(argc>1 && std::string(argv[1])=="--htop") {
        // PTY integration: C opens, arrows choose learning_rate, Enter edits,
        // typing 0.02 + Enter submits, and the training thread acknowledges.
        monitor.configureMetricsCsv("",false);
        monitor.start(true,false);
        bool applied=false;
        const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(30);
        while(std::chrono::steady_clock::now()<deadline) {
            const auto live=monitor.liveTrainParamsSnapshot();
            if(live.overrides_enabled && std::abs(live.lr-0.02f)<1e-6f) {applied=true;break;}
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        monitor.stop();
        TASSERT_TRUE(applied);
        return 0;
    }
    RuntimeConfigEditor editor;
    TASSERT_TRUE(!editor.visible);
    editor.toggle();
    editor.selected=index_of(config,"learning_rate");
    editor.enter(*config);
    TASSERT_TRUE(editor.editing);
    for(char c:std::string("0.02")) editor.text(c);
    editor.enter(*config);
    TASSERT_TRUE(!editor.editing);
    TASSERT_TRUE(config->snapshot()[index_of(config,"learning_rate")].pending.has_value());
    const auto live=monitor.liveTrainParamsSnapshot();
    TASSERT_TRUE(live.overrides_enabled);
    TASSERT_NEAR(live.lr,0.02f,1e-7f);
    monitor.updateRuntimeTrainParams(0.0001f,0,0,0,"mse");
    TASSERT_NEAR(monitor.liveTrainParamsSnapshot().lr,0.02f,1e-7f);
    TASSERT_TRUE(!config->snapshot()[index_of(config,"learning_rate")].pending.has_value());
    TASSERT_TRUE(!config->request("lr_warmup_steps","2.5",error));
    TASSERT_TRUE(!config->request("validation_enabled","1",error));
    TASSERT_TRUE(config->request("validation_enabled","true",error));
    TASSERT_TRUE(monitor.validationEnabledSnapshot());
    editor.enter(*config);
    editor.text('7');
    editor.cancel();
    TASSERT_NEAR(monitor.liveTrainParamsSnapshot().lr,0.02f,1e-7f);
    editor.cancel();
    TASSERT_TRUE(!editor.visible);

    // Publishing a runtime snapshot cannot discard an unconsumed user edit.
    TASSERT_TRUE(config->request("beta2","0.95",error));
    model.publishRuntimeConfiguration(&opt);
    model.applyRuntimeOptimizerConfiguration(opt);
    TASSERT_NEAR(opt.beta2,0.95f,1e-7f);
    TASSERT_TRUE(!config->snapshot()[index_of(config,"beta2")].pending.has_value());
    Model dropout;
    dropout.push("dropout", "Dropout", 0);
    dropout.getMutableLayers()[0].inputs={"__input__"};
    dropout.allocateParams();
    dropout.publishRuntimeConfiguration();
    TASSERT_TRUE(dropout.runtimeConfiguration()->request("dropout","0",error));
    TASSERT_TRUE(dropout.forwardPass(std::vector<float>{1,2,3},true)==std::vector<float>({1,2,3}));
    TASSERT_TRUE(dropout.getLayers()[0].dropout_p==0);
    return 0;
}
