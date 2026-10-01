cmake_minimum_required(VERSION 3.15)

function(mimir_add_test test_name src_file)
    add_executable(${test_name} ${src_file} test_log_stub.cpp)
    target_compile_features(${test_name} PRIVATE cxx_std_17)
    target_link_libraries(${test_name} PRIVATE mimir_core)
    add_test(NAME ${test_name} COMMAND ${test_name})
    set_tests_properties(${test_name} PROPERTIES TIMEOUT 15)
endfunction()

mimir_add_test(LRDecayTest.Warmup               test_lr_decay_warmup.cpp)
mimir_add_test(LRDecayTest.Boundaries           test_lr_decay_boundaries.cpp)
mimir_add_test(SerializationTest.RoundTrip      test_serialization_roundtrip.cpp)
mimir_add_test(SerializationTest.Deterministic  test_serialization_deterministic.cpp)
mimir_add_test(SerializationTest.CorruptedInput test_serialization_corrupted.cpp)

mimir_add_test(RegistryTest.CfgFromConfig        test_registry_cfg_from_config.cpp)
mimir_add_test(RegistryTest.CreateFromConfig     test_registry_create_from_config.cpp)
mimir_add_test(LuaTest.CreateFromConfig          test_lua_create_from_config.cpp)

mimir_add_test(FrameworkTest.TokenizerBasics      test_tokenizer_basics.cpp)
mimir_add_test(FrameworkTest.TokenizerBPE         test_tokenizer_bpe.cpp)
mimir_add_test(FrameworkTest.EncoderBasics        test_encoder_basics.cpp)
mimir_add_test(FrameworkTest.MemoryGuard          test_memory_guard.cpp)
mimir_add_test(FrameworkTest.OverrideParsing      test_override_parsing.cpp)
mimir_add_test(RegistryTest.Aliases               test_registry_aliases.cpp)
mimir_add_test(LuaTest.ApiBasics                  test_lua_api_basics.cpp)
set_tests_properties(LuaTest.ApiBasics PROPERTIES
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
)
mimir_add_test(LuaTest.ModelDType                 test_lua_dtype_api.cpp)
mimir_add_test(LuaTest.CreateFromConfigErrors     test_lua_create_from_config_errors.cpp)
add_test(
    NAME LuaTest.CausalLMTokenizer
    COMMAND $<TARGET_FILE:mimir> --lua scripts/tests/test_causal_lm_tokenizer_bootstrap.lua
)
set_tests_properties(LuaTest.CausalLMTokenizer PROPERTIES
    TIMEOUT 15
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
)

mimir_add_test(SerializationTest.SafeTensorsModel test_serialization_safetensors_model.cpp)
mimir_add_test(SerializationTest.SafeTensorsCompliance test_safetensors_compliance.cpp)
mimir_add_test(SerializationTest.RawFolderModel   test_serialization_rawfolder_model.cpp)
mimir_add_test(SerializationTest.DebugJsonModel    test_serialization_debug_json_model.cpp)
mimir_add_test(SerializationTest.StrictMissingAssets test_serialization_strict_missing_assets.cpp)

# Bridge API multi-runtime
mimir_add_test(BridgeTest.API test_bridge_api.cpp)
set_tests_properties(BridgeTest.API PROPERTIES TIMEOUT 120)

mimir_add_test(FrameworkTest.TokenizerJsonControlChars test_tokenizer_json_control_chars.cpp)
mimir_add_test(FrameworkTest.OverrideErrors           test_override_errors.cpp)

mimir_add_test(RuntimeTest.NamedRouting              test_layers_named_routing.cpp)
mimir_add_test(RuntimeTest.EmbeddingIntInputs         test_runtime_embedding_int_inputs.cpp)
mimir_add_test(RuntimeTest.ResidualMerge              test_runtime_residual_merge.cpp)
mimir_add_test(RuntimeTest.AllLayerTypes              test_layers_all_types.cpp)
mimir_add_test(ModelTest.WeightBlocks                test_model_weight_blocks.cpp)
mimir_add_test(ModelTest.TrainStepFreeze             test_model_train_step_freeze.cpp)
mimir_add_test(ModelTest.TrainStepDispatch           test_model_train_step_dispatch.cpp)
mimir_add_test(ModelTest.ExtendedOptimizers           test_optimizers_extended.cpp)

mimir_add_test(FrameworkTest.DType                   test_dtype.cpp)
mimir_add_test(FrameworkTest.PlannerLifetimes        test_planner_lifetimes.cpp)
mimir_add_test(FrameworkTest.PlannerFusionAndScratch test_planner_fusion_and_scratch.cpp)
mimir_add_test(FrameworkTest.RuntimeCapabilities    test_runtime_capabilities.cpp)
mimir_add_test(FrameworkTest.ExecutionPlan          test_execution_plan.cpp)
mimir_add_test(FrameworkTest.PlannerEquivalence     test_planner_equivalence.cpp)
mimir_add_test(ModelTest.VAEValidationFeedback     test_vae_validation_feedback.cpp)

mimir_add_test(RuntimeTest.MathLinear      test_runtime_math_linear.cpp)
mimir_add_test(RuntimeTest.MathConv2d     test_runtime_math_conv2d.cpp)
mimir_add_test(RuntimeTest.MathNorms      test_runtime_math_norms.cpp)
mimir_add_test(RuntimeTest.MathAttention  test_runtime_math_attention.cpp)
mimir_add_test(RuntimeTest.LayerOps       test_runtime_layer_ops.cpp)
mimir_add_test(RuntimeTest.PixelShuffle   test_runtime_pixel_shuffle.cpp)
mimir_add_test(RuntimeTest.NMS            test_runtime_nms.cpp)
mimir_add_test(AutogradTest.Numerical     test_autograd_numerical.cpp)
mimir_add_test(AutogradTest.RuntimeDispatch test_runtime_autograd_dispatch.cpp)
mimir_add_test(ModelTest.ReconLosses      test_recon_losses.cpp)
mimir_add_test(ModelTest.VAEConvContract  test_vae_conv_contract.cpp)
mimir_add_test(ModelTest.LumenDiffusionStructure test_lumen_diffusion_structure.cpp)
mimir_add_test(VizTest.TextPayload             test_viz_text_payload.cpp)
mimir_add_test(MetricsTest.HtopCsv             test_htop_metrics_csv.cpp)
if(NOT MIMIR_VIZ_BACKEND STREQUAL "NONE")
    mimir_add_test(VizTest.MetricsCsv          test_viz_metrics_csv.cpp)
endif()
mimir_add_test(ModelTest.VGG16FeatEncNorm test_vgg16_feat_enc_norm.cpp)

# Le runtime FPGA n'est pas encore composé dans le routeur global. Ces tests
# compilent néanmoins ses composants isolés afin d'éviter que le prototype
# matériel dérive silencieusement.
add_executable(RuntimeTest.FpgaDetection
    test_runtime_fpga_detection.cpp
    test_log_stub.cpp
    ../src/runtimes/fpga/FpgaRuntime.cpp
)
target_compile_features(RuntimeTest.FpgaDetection PRIVATE cxx_std_17)
target_link_libraries(RuntimeTest.FpgaDetection PRIVATE mimir_core)
add_test(NAME RuntimeTest.FpgaDetection COMMAND RuntimeTest.FpgaDetection)
set_tests_properties(RuntimeTest.FpgaDetection PROPERTIES TIMEOUT 15)

add_executable(RuntimeTest.FpgaValidationRunner
    test_fpga_validation_runner.cpp
    test_log_stub.cpp
    ../src/runtimes/fpga/FpgaRuntime.cpp
    ../src/runtimes/fpga/FpgaValidationRunner.cpp
)
target_compile_features(RuntimeTest.FpgaValidationRunner PRIVATE cxx_std_17)
target_link_libraries(RuntimeTest.FpgaValidationRunner PRIVATE mimir_core)
add_test(NAME RuntimeTest.FpgaValidationRunner COMMAND RuntimeTest.FpgaValidationRunner)
set_tests_properties(RuntimeTest.FpgaValidationRunner PROPERTIES TIMEOUT 15)

# Spawn-only regression: no display, SFML, Qt or GTK needed.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_executable(VizTest.HostEnvironment test_viz_host_environment.cpp ../src/viz/ProcessHost.cpp)
    target_include_directories(VizTest.HostEnvironment PRIVATE ../src)
    target_link_libraries(VizTest.HostEnvironment PRIVATE Threads::Threads)
    add_test(NAME VizTest.HostEnvironment COMMAND VizTest.HostEnvironment)
    set_tests_properties(VizTest.HostEnvironment PROPERTIES TIMEOUT 15)
endif()

# Native GPU gradient parity. Missing devices are explicit CTest skips.
if(ENABLE_OPENCL OR ENABLE_VULKAN)
    add_executable(GpuBackwardParity test_gpu_backward_parity.cpp test_log_stub.cpp)
    target_link_libraries(GpuBackwardParity PRIVATE mimir_core)
    foreach(backend OPENCL VULKAN)
        if(ENABLE_${backend})
            string(TOLOWER "${backend}" backend_arg)
            add_test(NAME RuntimeTest.${backend}BackwardParity COMMAND GpuBackwardParity ${backend_arg})
            set_tests_properties(RuntimeTest.${backend}BackwardParity PROPERTIES
                SKIP_RETURN_CODE 77 TIMEOUT 120 WORKING_DIRECTORY "${CMAKE_BINARY_DIR}")
        endif()
    endforeach()
endif()

mimir_add_test(AutogradTest.ModelRuntime test_model_runtime_autograd.cpp)

mimir_add_test(AutogradTest.Convolution test_conv_autograd.cpp)

mimir_add_test(FrameworkTest.DatasetRGB test_dataset_rgb.cpp)

mimir_add_test(SerializationTest.ActiveState test_checkpoint_active_state.cpp)
set_tests_properties(SerializationTest.ActiveState PROPERTIES TIMEOUT 60 ENVIRONMENT "OMP_NUM_THREADS=1")

mimir_add_test(ModelTest.SkipConnections test_skip_connections.cpp)
add_test(NAME ModelTest.SkipConnectionsUnplanned COMMAND ModelTest.SkipConnections)
set_tests_properties(ModelTest.SkipConnections ModelTest.SkipConnectionsUnplanned PROPERTIES TIMEOUT 30)
set_tests_properties(ModelTest.SkipConnections PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1;MIMIR_ENABLE_PLANNER=1")
set_tests_properties(ModelTest.SkipConnectionsUnplanned PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1;MIMIR_ENABLE_PLANNER=0")

mimir_add_test(ModelTest.LiveConfiguration test_live_model_config.cpp)
set_tests_properties(ModelTest.LiveConfiguration PROPERTIES TIMEOUT 30 ENVIRONMENT "OMP_NUM_THREADS=1")
