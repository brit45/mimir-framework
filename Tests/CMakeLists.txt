cmake_minimum_required(VERSION 3.15)

function(mimir_add_test test_name src_file)
    add_executable(${test_name} ${src_file} test_log_stub.cpp)
    target_compile_features(${test_name} PRIVATE cxx_std_17)
    target_link_libraries(${test_name} PRIVATE mimir_core)
    add_test(NAME ${test_name} COMMAND ${test_name})
    set_tests_properties(${test_name} PROPERTIES TIMEOUT 15)
endfunction()

# Compile-only regressions: exercise both optimizer preprocessing branches even
# when the compiler defaults to amd64-v3, and the no-GUI monitor API separately.
add_library(CompileTest.HeadlessVisualizer OBJECT test_headless_visualizer_api.cpp)
target_compile_features(CompileTest.HeadlessVisualizer PRIVATE cxx_std_17)
target_include_directories(CompileTest.HeadlessVisualizer PRIVATE ../src ../src/include ../src/runtimes/cpu)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang" AND
   CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
    foreach(isa IN ITEMS AVX2 Portable)
        add_library(CompileTest.Optimizer${isa} OBJECT ../src/runtimes/optimizers/OptimizerHost.cpp)
        target_include_directories(CompileTest.Optimizer${isa} PRIVATE ../src ../src/include ../src/runtimes/cpu)
        target_compile_definitions(CompileTest.Optimizer${isa} PRIVATE MIMIR_SIMD_ENABLED=1)
        if(TARGET OpenMP::OpenMP_CXX)
            target_link_libraries(CompileTest.Optimizer${isa} PRIVATE OpenMP::OpenMP_CXX)
        endif()
        target_compile_features(CompileTest.Optimizer${isa} PRIVATE cxx_std_17)
    endforeach()
    target_compile_options(CompileTest.OptimizerAVX2 PRIVATE -mavx2)
    target_compile_options(CompileTest.OptimizerPortable PRIVATE -mno-avx2)
endif()

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
mimir_add_test(LuaTest.Gradients                  test_lua_gradients.cpp)
set_tests_properties(LuaTest.Gradients PROPERTIES WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
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
add_test(NAME LuaTest.ConvND
    COMMAND $<TARGET_FILE:mimir> --lua scripts/tests/test_conv_nd.lua)
set_tests_properties(LuaTest.ConvND PROPERTIES
    TIMEOUT 15 WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    ENVIRONMENT "MIMIR_DISABLE_VULKAN=1;MIMIR_DISABLE_OPENCL=1;MIMIR_DISABLE_ROCM=1")
add_test(
    NAME LuaTest.CausalLMTrainingSmoke
    COMMAND $<TARGET_FILE:mimir> --lua scripts/tests/test_causal_lm_smoke.lua
)
set_tests_properties(LuaTest.CausalLMTrainingSmoke PROPERTIES
    TIMEOUT 15
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
)

mimir_add_test(SerializationTest.SafeTensorsModel test_serialization_safetensors_model.cpp)
mimir_add_test(SerializationTest.SafeTensorsCompliance test_safetensors_compliance.cpp)
mimir_add_test(SerializationTest.RawFolderModel   test_serialization_rawfolder_model.cpp)
mimir_add_test(SerializationTest.DebugJsonModel    test_serialization_debug_json_model.cpp)
mimir_add_test(SerializationTest.StrictMissingAssets test_serialization_strict_missing_assets.cpp)

# Bridge API multi-runtime
if(ENABLE_SCRIPTING_RUST)
    find_program(MIMIR_RUST_TEST_COMPILER rustc)
    if(MIMIR_RUST_TEST_COMPILER)
        add_test(NAME BridgeTest.RustConf
            COMMAND ${CMAKE_COMMAND} -DMIMIR_BIN=$<TARGET_FILE:mimir>
                -P ${CMAKE_CURRENT_SOURCE_DIR}/test_rust_conf.cmake)
        add_test(NAME BridgeTest.RustSmoke
            COMMAND $<TARGET_FILE:mimir> --rust scripts/examples/bridge_smoke.rs)
        set_tests_properties(BridgeTest.RustConf BridgeTest.RustSmoke PROPERTIES
            TIMEOUT 120 WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
    endif()
    mimir_add_test(BridgeTest.RustParity test_rust_bridge.cpp)
    set_tests_properties(BridgeTest.RustParity PROPERTIES TIMEOUT 120 SKIP_RETURN_CODE 77 WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
endif()
find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
    add_test(NAME BridgeTest.IDEHelpers
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/tools/generate_ide_helpers.py
            --binary $<TARGET_FILE:mimir> --check)
    set_tests_properties(BridgeTest.IDEHelpers PROPERTIES TIMEOUT 30 WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
endif()
mimir_add_test(BridgeTest.API test_bridge_api.cpp)
set_tests_properties(BridgeTest.API PROPERTIES TIMEOUT 120)
if(ENABLE_SCRIPTING_JS)
    find_program(MIMIR_JS_TEST_NODE node)
    if(MIMIR_JS_TEST_NODE)
        add_test(NAME BridgeTest.JSConf
            COMMAND ${CMAKE_COMMAND} -DMIMIR_BIN=$<TARGET_FILE:mimir>
                -P ${CMAKE_CURRENT_SOURCE_DIR}/test_js_conf.cmake)
        add_test(NAME BridgeTest.JSSmoke
            COMMAND $<TARGET_FILE:mimir> --js scripts/examples/bridge_smoke.js)
        set_tests_properties(BridgeTest.JSConf BridgeTest.JSSmoke PROPERTIES
            TIMEOUT 15 WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
    endif()
endif()


mimir_add_test(FrameworkTest.TokenizerJsonControlChars test_tokenizer_json_control_chars.cpp)
mimir_add_test(FrameworkTest.OverrideErrors           test_override_errors.cpp)

mimir_add_test(RuntimeTest.NamedRouting              test_layers_named_routing.cpp)
mimir_add_test(RuntimeTest.EmbeddingIntInputs         test_runtime_embedding_int_inputs.cpp)
mimir_add_test(RuntimeTest.ResidualMerge              test_runtime_residual_merge.cpp)
mimir_add_test(RuntimeTest.AllLayerTypes              test_layers_all_types.cpp)
mimir_add_test(ModelTest.WeightBlocks                test_model_weight_blocks.cpp)
mimir_add_test(ModelTest.TrainStepFreeze             test_model_train_step_freeze.cpp)
mimir_add_test(ModelTest.TrainStepDispatch           test_model_train_step_dispatch.cpp)
mimir_add_test(RuntimeTest.ReconstructionLoss test_reconstruction_runtime.cpp)
set_tests_properties(RuntimeTest.ReconstructionLoss PROPERTIES TIMEOUT 180)
foreach(backend OPENCL VULKAN CUDA ROCM)
    if(ENABLE_${backend})
        add_test(NAME RuntimeTest.${backend}ReconstructionLoss COMMAND RuntimeTest.ReconstructionLoss ${backend})
        set_tests_properties(RuntimeTest.${backend}ReconstructionLoss PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 180 LABELS "GPU;${backend}")
    endif()
endforeach()
add_test(NAME RuntimeTest.TPUReconstructionLoss COMMAND RuntimeTest.ReconstructionLoss TPU)
set_tests_properties(RuntimeTest.TPUReconstructionLoss PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 1800 LABELS "TPU")
mimir_add_test(RuntimeTest.OptimizerDispatch test_optimizer_runtime.cpp)
set_tests_properties(RuntimeTest.OptimizerDispatch PROPERTIES TIMEOUT 60)
foreach(backend OPENCL VULKAN CUDA ROCM)
    if(ENABLE_${backend})
        add_test(NAME RuntimeTest.${backend}Optimizers COMMAND RuntimeTest.OptimizerDispatch ${backend})
        set_tests_properties(RuntimeTest.${backend}Optimizers PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 180 LABELS "GPU;${backend}")
    endif()
endforeach()
mimir_add_test(ModelTest.ExtendedOptimizers           test_optimizers_extended.cpp)

mimir_add_test(FrameworkTest.DType                   test_dtype.cpp)
mimir_add_test(FrameworkTest.PlannerLifetimes        test_planner_lifetimes.cpp)
mimir_add_test(FrameworkTest.PlannerFusionAndScratch test_planner_fusion_and_scratch.cpp)
mimir_add_test(FrameworkTest.RuntimeCapabilities    test_runtime_capabilities.cpp)
mimir_add_test(FrameworkTest.ExecutionPlan          test_execution_plan.cpp)
mimir_add_test(FrameworkTest.ExecutionCalibration test_execution_calibration.cpp)
mimir_add_test(FrameworkTest.AttentionCalibration test_attention_calibration.cpp)
set_tests_properties(FrameworkTest.AttentionCalibration PROPERTIES TIMEOUT 30 ENVIRONMENT "OMP_DYNAMIC=FALSE")
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(FrameworkTest.AttentionCalibration PRIVATE -ffp-contract=off)
endif()
foreach(backend IN ITEMS OPENCL VULKAN)
    if(ENABLE_${backend})
        add_test(NAME RuntimeTest.${backend}AttentionCalibration COMMAND FrameworkTest.AttentionCalibration ${backend})
        set_tests_properties(RuntimeTest.${backend}AttentionCalibration PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 120)
    endif()
endforeach()
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
mimir_add_test(ModelTest.VAEKLControls test_vae_kl_controls.cpp)
mimir_add_test(ModelTest.VAEConvLogvar test_vae_conv_logvar.cpp)
set_tests_properties(ModelTest.VAEConvLogvar PROPERTIES TIMEOUT 90 ENVIRONMENT
    "OMP_NUM_THREADS=1;MIMIR_DISABLE_CUDA=1;MIMIR_DISABLE_ROCM=1;MIMIR_DISABLE_VULKAN=1;MIMIR_DISABLE_OPENCL=1;MIMIR_RUNTIME_AUTOTUNE=0")
mimir_add_test(ModelTest.VAETextContext test_vae_text_context.cpp)
mimir_add_test(ModelTest.VAEConvLatentContext test_vae_conv_latent.cpp)
mimir_add_test(ModelTest.LumenDiffusionStructure test_lumen_diffusion_structure.cpp)
set_tests_properties(
    ModelTest.VAEConvLatentContext
    ModelTest.LumenDiffusionStructure
    PROPERTIES TIMEOUT 120)
mimir_add_test(VizTest.TextPayload             test_viz_text_payload.cpp)
mimir_add_test(MetricsTest.AsyncHistory test_async_monitor_history.cpp)
mimir_add_test(LuaTest.VAETextVizBridge test_lua_vae_text_viz.cpp)
mimir_add_test(MetricsTest.GradientDiagnostics test_gradient_diagnostics.cpp)
mimir_add_test(MetricsTest.HtopCsv             test_htop_metrics_csv.cpp)
if(NOT MIMIR_VIZ_BACKEND STREQUAL "NONE")
    mimir_add_test(VizTest.MetricsCsv          test_viz_metrics_csv.cpp)
endif()
mimir_add_test(ModelTest.VGG16FeatEncNorm test_vgg16_feat_enc_norm.cpp)
set_tests_properties(ModelTest.VGG16FeatEncNorm PROPERTIES TIMEOUT 120)
add_executable(VizTest.AnnotationFeatureBounds test_annotation_feature_bounds.cpp)
target_include_directories(VizTest.AnnotationFeatureBounds PRIVATE ../src ../src/include)
add_test(NAME VizTest.AnnotationFeatureBounds COMMAND VizTest.AnnotationFeatureBounds)
set_tests_properties(VizTest.AnnotationFeatureBounds PROPERTIES TIMEOUT 15)
mimir_add_test(ModelTest.MedicalDetector test_medical_detector_model.cpp)

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
mimir_add_test(FrameworkTest.DatasetPrefetch test_dataset_prefetch.cpp)
set_tests_properties(FrameworkTest.DatasetPrefetch PROPERTIES TIMEOUT 30)
add_test(NAME LuaTest.DatasetPrefetch COMMAND $<TARGET_FILE:mimir> --lua scripts/tests/test_dataset_prefetch.lua)
set_tests_properties(LuaTest.DatasetPrefetch PROPERTIES TIMEOUT 30 WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    ENVIRONMENT "MIMIR_DATASET_PREFETCH=3;MIMIR_DISABLE_ROCM=1;MIMIR_DISABLE_CUDA=1;MIMIR_DISABLE_OPENCL=1;MIMIR_DISABLE_VULKAN=1")
if(Python3_Interpreter_FOUND)
    add_test(NAME LuaTest.DatasetTrainingPrefetch COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_SOURCE_DIR}/scripts/tests/test_dataset_training_prefetch.py --binary $<TARGET_FILE:mimir>)
    set_tests_properties(LuaTest.DatasetTrainingPrefetch PROPERTIES TIMEOUT 180 WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
endif()
mimir_add_test(FrameworkTest.DatasetSymlinks test_dataset_symlinks.cpp)
set_tests_properties(FrameworkTest.DatasetSymlinks PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 30)
mimir_add_test(FrameworkTest.GifDisposal test_gif_disposal.cpp)
mimir_add_test(FrameworkTest.DatasetAnnotations test_dataset_annotations.cpp)
set_tests_properties(FrameworkTest.DatasetAnnotations PROPERTIES TIMEOUT 60 ENVIRONMENT "OMP_NUM_THREADS=1")

mimir_add_test(SerializationTest.ActiveState test_checkpoint_active_state.cpp)
set_tests_properties(SerializationTest.ActiveState PROPERTIES TIMEOUT 60 ENVIRONMENT "OMP_NUM_THREADS=1")

mimir_add_test(ModelTest.SkipConnections test_skip_connections.cpp)
add_test(NAME ModelTest.SkipConnectionsUnplanned COMMAND ModelTest.SkipConnections)
set_tests_properties(ModelTest.SkipConnections ModelTest.SkipConnectionsUnplanned PROPERTIES TIMEOUT 30)
set_tests_properties(ModelTest.SkipConnections PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1;MIMIR_ENABLE_PLANNER=1")
set_tests_properties(ModelTest.SkipConnectionsUnplanned PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1;MIMIR_ENABLE_PLANNER=0")

mimir_add_test(ModelTest.LiveConfiguration test_live_model_config.cpp)
mimir_add_test(VizTest.ModelGraphSnapshot test_model_graph_snapshot.cpp)
mimir_add_test(VizTest.HtopModelGraph test_htop_model_graph.cpp)
if(UNIX AND NOT APPLE)
    target_link_libraries(VizTest.HtopModelGraph PRIVATE util)
endif()
set_tests_properties(VizTest.HtopModelGraph PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 30)

# Metadata regression runs on CPU-only builders too.
mimir_add_test(RuntimeTest.CudaLinearCapability test_cuda_linear_capability.cpp)
if(NOT ENABLE_CUDA)
    target_sources(RuntimeTest.CudaLinearCapability PRIVATE ../src/runtimes/cuda/CudaRuntime.cpp)
endif()

set_tests_properties(ModelTest.LiveConfiguration PROPERTIES TIMEOUT 30 ENVIRONMENT "OMP_NUM_THREADS=1")

mimir_add_test(FrameworkTest.BBoxSupervision test_bbox_supervision.cpp)
set_tests_properties(FrameworkTest.BBoxSupervision PROPERTIES TIMEOUT 60 ENVIRONMENT "OMP_NUM_THREADS=1")

mimir_add_test(ModelTest.VisionBBox test_vision_bbox.cpp)
set_tests_properties(ModelTest.VisionBBox PROPERTIES TIMEOUT 120 ENVIRONMENT "OMP_NUM_THREADS=1")

# Exercise the shared vector icon offscreen even in a headless framework build.
find_package(PkgConfig QUIET)
# pkg-config describes the build host; never leak Linux Cairo include paths
# into a Windows cross compiler. These offscreen tests use native Cairo only.
if(PkgConfig_FOUND AND NOT CMAKE_CROSSCOMPILING)
    pkg_check_modules(BBOX_BADGE_CAIRO QUIET IMPORTED_TARGET cairo)
    if(BBOX_BADGE_CAIRO_FOUND)
        add_executable(VizTest.ModelGraphLanes test_model_graph_lanes.cpp ../src/viz/SoftwareGraphics.cpp)
        target_compile_features(VizTest.ModelGraphLanes PRIVATE cxx_std_17)
        target_include_directories(VizTest.ModelGraphLanes PRIVATE ../src)
        target_link_libraries(VizTest.ModelGraphLanes PRIVATE PkgConfig::BBOX_BADGE_CAIRO)
        add_test(NAME VizTest.ModelGraphLanes COMMAND VizTest.ModelGraphLanes)
        set_tests_properties(VizTest.ModelGraphLanes PROPERTIES TIMEOUT 15)
        add_executable(VizTest.DatasetBBoxBadge test_viz_dataset_bbox_badge.cpp ../src/viz/SoftwareGraphics.cpp)
        target_compile_features(VizTest.DatasetBBoxBadge PRIVATE cxx_std_17)
        target_include_directories(VizTest.DatasetBBoxBadge PRIVATE ../src)
        target_link_libraries(VizTest.DatasetBBoxBadge PRIVATE PkgConfig::BBOX_BADGE_CAIRO)
        add_test(NAME VizTest.DatasetBBoxBadge COMMAND VizTest.DatasetBBoxBadge)
        set_tests_properties(VizTest.DatasetBBoxBadge PROPERTIES TIMEOUT 15)
    endif()
endif()

# Real event dispatcher with an offscreen window; skip when SFML has no display.
mimir_add_test(VizTest.ButtonEvents test_viz_events.cpp)
add_test(NAME VizTest.KLControls COMMAND VizTest.ButtonEvents --kl-only)
set_tests_properties(VizTest.KLControls PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 30)
set_tests_properties(VizTest.ButtonEvents PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 30)

# Matrix launch geometry and batch strides must execute on the selected GPU.
if(ENABLE_CUDA OR ENABLE_ROCM OR ENABLE_OPENCL OR ENABLE_VULKAN)
    add_executable(RuntimeTest.GpuParallelMatMul test_gpu_parallel_matmul.cpp test_log_stub.cpp)
    target_compile_features(RuntimeTest.GpuParallelMatMul PRIVATE cxx_std_17)
    target_link_libraries(RuntimeTest.GpuParallelMatMul PRIVATE mimir_core)
    foreach(backend CUDA ROCM OPENCL VULKAN)
        if(ENABLE_${backend})
            add_test(NAME RuntimeTest.${backend}ParallelMatMul COMMAND RuntimeTest.GpuParallelMatMul ${backend})
            set_tests_properties(RuntimeTest.${backend}ParallelMatMul PROPERTIES
                SKIP_RETURN_CODE 77 TIMEOUT 180 LABELS "GPU;${backend}")
        endif()
    endforeach()
endif()

# Native CUDA parity must execute on a CUDA device; CPU fallback is not a pass.
if(ENABLE_CUDA AND CUDAToolkit_FOUND)
    mimir_add_test(RuntimeTest.CudaNativeOps test_cuda_native_ops.cpp)
    set_tests_properties(RuntimeTest.CudaNativeOps PROPERTIES TIMEOUT 120 LABELS "CUDA;GPU")
endif()

mimir_add_test(RuntimeTest.PortableWorkBudget test_portable_work_budget.cpp)
mimir_add_test(RuntimeTest.TensorWeights test_tensor_weights_runtime.cpp)
mimir_add_test(RuntimeTest.GradientCommit test_gradient_commit.cpp)
if(ENABLE_OPENCL OR ENABLE_VULKAN)
    add_executable(PortableNativeOps test_portable_native_ops.cpp test_log_stub.cpp)
    target_compile_features(PortableNativeOps PRIVATE cxx_std_17)
    target_link_libraries(PortableNativeOps PRIVATE mimir_core)
    foreach(backend OPENCL VULKAN)
        if(ENABLE_${backend})
            add_test(NAME RuntimeTest.${backend}NativeOps COMMAND PortableNativeOps ${backend})
            set_tests_properties(RuntimeTest.${backend}NativeOps PROPERTIES
                SKIP_RETURN_CODE 77 TIMEOUT 180 LABELS "GPU;${backend}" WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
        endif()
    endforeach()
endif()

mimir_add_test(ModelTest.GanStyle test_gan_style_model.cpp)
set_tests_properties(ModelTest.GanStyle PROPERTIES TIMEOUT 60 ENVIRONMENT "OMP_NUM_THREADS=1")

# Exercise real loader/serializer entrypoints against a serial firmware emulator.
find_package(Python3 QUIET COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
    foreach(mode host legacy corrupt-crc)
        if(mode STREQUAL "host")
            set(fpga_test_args "")
        else()
            set(fpga_test_args --${mode})
        endif()
        add_test(NAME RuntimeTest.FpgaIo.${mode}
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/tests/test_fpga_io.py ${fpga_test_args})
        set_tests_properties(RuntimeTest.FpgaIo.${mode} PROPERTIES TIMEOUT 45)
    endforeach()
    find_program(MIMIR_IVERILOG iverilog)
    find_program(MIMIR_VVP vvp)
    if(MIMIR_IVERILOG AND MIMIR_VVP)
        add_test(NAME RuntimeTest.FpgaIo.RTL
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/tests/test_fpga_io_rtl.py)
        set_tests_properties(RuntimeTest.FpgaIo.RTL PROPERTIES TIMEOUT 45)
    endif()
endif()

# Reference fixture has no dependency on JAX. Its output is also consumed by the
# optional Python differential test and the optional TPU runtime test.
mimir_add_test(RuntimeTest.TpuCpuFixture test_tpu_cpu_fixture.cpp)
set_tests_properties(RuntimeTest.TpuCpuFixture PROPERTIES TIMEOUT 60)
set(MIMIR_TPU_TEST_PYTHON "" CACHE FILEPATH "Python with JAX for TPU reference differential tests")
if(MIMIR_TPU_TEST_PYTHON AND EXISTS "${MIMIR_TPU_TEST_PYTHON}")
    add_test(NAME RuntimeTest.ReconstructionJaxReference
        COMMAND "${MIMIR_TPU_TEST_PYTHON}" "${CMAKE_CURRENT_SOURCE_DIR}/test_reconstruction_tpu_reference.py"
                $<TARGET_FILE:RuntimeTest.ReconstructionLoss>)
    set_tests_properties(RuntimeTest.ReconstructionJaxReference PROPERTIES
        TIMEOUT 180 ENVIRONMENT "OMP_NUM_THREADS=1;JAX_PLATFORMS=cpu" LABELS "JAX")
    add_test(NAME RuntimeTest.OptimizerJaxReference
        COMMAND "${MIMIR_TPU_TEST_PYTHON}" "${CMAKE_CURRENT_SOURCE_DIR}/test_optimizer_tpu_reference.py"
                $<TARGET_FILE:RuntimeTest.OptimizerDispatch>)
    set_tests_properties(RuntimeTest.OptimizerJaxReference PROPERTIES
        TIMEOUT 120 ENVIRONMENT "OMP_NUM_THREADS=1;JAX_PLATFORMS=cpu" LABELS "JAX")
    add_test(NAME RuntimeTest.TpuReference
        COMMAND "${MIMIR_TPU_TEST_PYTHON}" "${CMAKE_CURRENT_SOURCE_DIR}/test_tpu_reference.py"
                $<TARGET_FILE:RuntimeTest.TpuCpuFixture>)
    set_tests_properties(RuntimeTest.TpuReference PROPERTIES
        TIMEOUT 300 ENVIRONMENT "OMP_NUM_THREADS=1" LABELS "TPU;COLAB")
elseif(MIMIR_TPU_TEST_PYTHON)
    message(WARNING "MIMIR_TPU_TEST_PYTHON does not point to an existing file; optional TPU reference tests are disabled.")
endif()
mimir_add_test(RuntimeTest.TpuLifecycle test_tpu_lifecycle.cpp)
mimir_add_test(RuntimeTest.TpuModelTrain test_tpu_model_train.cpp)
set_tests_properties(RuntimeTest.TpuModelTrain PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 300 LABELS "TPU")

# Odd feature sizes, batched reductions and accumulated packed gradients.
mimir_add_test(RuntimeTest.LinearBackwardTiled test_linear_backward_tiled.cpp)
mimir_add_test(RuntimeTest.TypedConversions test_typed_runtime.cpp)
mimir_add_test(RuntimeTest.ConvolutionScratch test_convolution_scratch.cpp)
mimir_add_test(RuntimeTest.AllocatorScratch test_allocator_scratch.cpp)
mimir_add_test(RuntimeTest.MatMulBackward test_matmul_backward.cpp)
mimir_add_test(RuntimeTest.FiniteCheck test_finite_check.cpp)
mimir_add_test(RuntimeTest.RouteRefresh test_route_refresh.cpp)
mimir_add_test(RuntimeTest.BatchMatMulForward test_batch_matmul_forward.cpp)
mimir_add_test(RuntimeTest.FlatJoin test_flat_join.cpp)
mimir_add_test(RuntimeTest.AttentionTiled test_attention_tiled.cpp)
mimir_add_test(RuntimeTest.OptimizerSelection test_optimizer_selection.cpp)

# Opt-in latency benchmark: includes the complete public training/inference path.
add_executable(mimir_bench_whole_model EXCLUDE_FROM_ALL
    ../scripts/benchmarks/benchmark_whole_model.cpp test_log_stub.cpp)
target_compile_features(mimir_bench_whole_model PRIVATE cxx_std_17)
target_link_libraries(mimir_bench_whole_model PRIVATE mimir_core)
add_executable(mimir_bench_dataset_prefetch EXCLUDE_FROM_ALL
    ../scripts/benchmarks/benchmark_dataset_prefetch.cpp test_log_stub.cpp)
target_compile_features(mimir_bench_dataset_prefetch PRIVATE cxx_std_17)
target_link_libraries(mimir_bench_dataset_prefetch PRIVATE mimir_core)
add_executable(mimir_bench_layer_dimensions EXCLUDE_FROM_ALL
    ../scripts/benchmarks/benchmark_layer_dimensions.cpp test_log_stub.cpp)
target_compile_features(mimir_bench_layer_dimensions PRIVATE cxx_std_17)
target_link_libraries(mimir_bench_layer_dimensions PRIVATE mimir_core)
add_executable(mimir_bench_vgg16_whole_model EXCLUDE_FROM_ALL
    ../scripts/benchmarks/benchmark_vgg16_whole_model.cpp test_log_stub.cpp)
target_link_libraries(mimir_bench_vgg16_whole_model PRIVATE mimir_core)

mimir_add_test(RuntimeTest.OptimizerCompact test_optimizer_compact.cpp)
# Optional OpenCL numerical reference, including software devices. This test
# proves kernel parity, not physical GPU execution or throughput.
find_path(MIMIR_TEST_OPENCL_INCLUDE CL/cl.h)
find_library(MIMIR_TEST_OPENCL_LIBRARY OpenCL)
if(MIMIR_TEST_OPENCL_INCLUDE AND MIMIR_TEST_OPENCL_LIBRARY)
    target_sources(RuntimeTest.OptimizerCompact PRIVATE optimizer_opencl_reference.cpp)
    target_compile_definitions(RuntimeTest.OptimizerCompact PRIVATE MIMIR_TEST_OPENCL)
    target_include_directories(RuntimeTest.OptimizerCompact PRIVATE ${MIMIR_TEST_OPENCL_INCLUDE})
    target_link_libraries(RuntimeTest.OptimizerCompact PRIVATE ${MIMIR_TEST_OPENCL_LIBRARY})
    add_test(NAME RuntimeTest.OptimizerCompactOpenCL COMMAND RuntimeTest.OptimizerCompact opencl)
    set_tests_properties(RuntimeTest.OptimizerCompactOpenCL PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 180)
endif()

# Compile the embedded source even on CPU-only/MSVC runners, without OpenCL SDK.
add_executable(RuntimeTest.EmbeddedShaderSource test_embedded_shader_source.cpp)
target_include_directories(RuntimeTest.EmbeddedShaderSource PRIVATE ../src)
target_compile_definitions(RuntimeTest.EmbeddedShaderSource PRIVATE ENABLE_OPENCL)
add_test(NAME RuntimeTest.EmbeddedShaderSource COMMAND RuntimeTest.EmbeddedShaderSource)
set_tests_properties(RuntimeTest.EmbeddedShaderSource PROPERTIES TIMEOUT 15 WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")

add_executable(RuntimeTest.SoftwareFP16 test_fp16_software.cpp ../src/runtimes/cpu/SimdDispatch.cpp)
target_include_directories(RuntimeTest.SoftwareFP16 PRIVATE ../src)
if(OpenMP_CXX_FOUND)
    target_link_libraries(RuntimeTest.SoftwareFP16 PRIVATE OpenMP::OpenMP_CXX)
endif()
add_test(NAME RuntimeTest.SoftwareFP16 COMMAND RuntimeTest.SoftwareFP16)
set_tests_properties(RuntimeTest.SoftwareFP16 PROPERTIES TIMEOUT 15 ENVIRONMENT "MIMIR_CPU_ISA=scalar")

add_test(NAME LuaTest.CheckpointResumeResolver
    COMMAND $<TARGET_FILE:mimir> --lua scripts/tests/test_checkpoint_resume.lua)
add_test(NAME LuaTest.TrainingResume
    COMMAND $<TARGET_FILE:mimir> --lua scripts/tests/test_training_resume.lua)
set_tests_properties(LuaTest.CheckpointResumeResolver LuaTest.TrainingResume PROPERTIES
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" TIMEOUT 120 ENVIRONMENT "OMP_NUM_THREADS=1")

add_test(NAME LuaTest.VAEConvInferenceCompat
    COMMAND $<TARGET_FILE:mimir> --lua scripts/tests/test_vae_conv_inference_compat.lua)
set_tests_properties(LuaTest.VAEConvInferenceCompat PROPERTIES
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" TIMEOUT 180 ENVIRONMENT
    "OMP_NUM_THREADS=1;MIMIR_DISABLE_CUDA=1;MIMIR_DISABLE_ROCM=1;MIMIR_DISABLE_VULKAN=1;MIMIR_DISABLE_OPENCL=1;MIMIR_RUNTIME_AUTOTUNE=0;MIMIR_VAE_INFERENCE_TEST_BIN=$<TARGET_FILE:mimir>")

mimir_add_test(RuntimeTest.Fusion test_runtime_fusion.cpp)
set_tests_properties(RuntimeTest.Fusion PROPERTIES TIMEOUT 120 ENVIRONMENT
    "OMP_NUM_THREADS=1;MIMIR_DISABLE_CUDA=1;MIMIR_DISABLE_ROCM=1;MIMIR_DISABLE_VULKAN=1;MIMIR_DISABLE_OPENCL=1")
add_test(NAME RuntimeTest.VAEConvTrainingFusion
    COMMAND ${CMAKE_COMMAND} -DMIMIR_FUSION_TEST=$<TARGET_FILE:RuntimeTest.Fusion>
        -P ${CMAKE_CURRENT_SOURCE_DIR}/test_vae_fusion_parity.cmake)
set_tests_properties(RuntimeTest.VAEConvTrainingFusion PROPERTIES TIMEOUT 240)


mimir_add_test(PlannerTest.FusionCalibration test_fusion_calibration.cpp)
set_tests_properties(PlannerTest.FusionCalibration PROPERTIES TIMEOUT 120 ENVIRONMENT "OMP_NUM_THREADS=1")

mimir_add_test(PlannerTest.ExecutionPolicy test_execution_policy.cpp)
set_tests_properties(PlannerTest.ExecutionPolicy PROPERTIES TIMEOUT 120 ENVIRONMENT "OMP_NUM_THREADS=4")

add_executable(RuntimeTest.GpuPointwiseFusion test_gpu_pointwise_fusion.cpp test_log_stub.cpp)
target_compile_features(RuntimeTest.GpuPointwiseFusion PRIVATE cxx_std_17)
target_link_libraries(RuntimeTest.GpuPointwiseFusion PRIVATE mimir_core)
foreach(backend VULKAN OPENCL CUDA ROCM)
    add_test(NAME RuntimeTest.${backend}PointwiseFusion COMMAND RuntimeTest.GpuPointwiseFusion ${backend})
    set_tests_properties(RuntimeTest.${backend}PointwiseFusion PROPERTIES
        SKIP_RETURN_CODE 77 TIMEOUT 180 LABELS "GPU;${backend}"
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" ENVIRONMENT "OMP_NUM_THREADS=1;MIMIR_RUNTIME_AUTOTUNE=1")
endforeach()

add_test(NAME RuntimeTest.GpuFusionPacking COMMAND RuntimeTest.GpuPointwiseFusion --packing)
set_tests_properties(RuntimeTest.GpuFusionPacking PROPERTIES TIMEOUT 15)

add_test(NAME LuaTest.LumenInferenceUI
    COMMAND $<TARGET_FILE:mimir> --lua scripts/tests/test_lumen_inference_ui.lua)
add_test(NAME LuaTest.LumenInferenceScript
    COMMAND $<TARGET_FILE:mimir> --lua scripts/tests/test_lumen_inference_script.lua)
set_tests_properties(LuaTest.LumenInferenceUI LuaTest.LumenInferenceScript PROPERTIES
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" TIMEOUT 30)

mimir_add_test(ModelTest.T5 test_t5.cpp)
set_tests_properties(ModelTest.T5 PROPERTIES TIMEOUT 60)
add_test(NAME LuaTest.T5 COMMAND $<TARGET_FILE:mimir> --lua scripts/tests/test_t5.lua)
set_tests_properties(LuaTest.T5 PROPERTIES WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" TIMEOUT 30)

mimir_add_test(RuntimeTest.LayerBoundary test_runtime_layer_boundary.cpp)

# Baseline executable: optional ISA kernels are isolated in the dispatcher TU.
add_executable(RuntimeTest.SimdDispatch test_runtime_simd.cpp ../src/runtimes/cpu/SimdDispatch.cpp)
target_include_directories(RuntimeTest.SimdDispatch PRIVATE ../src)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    # The independent reference must preserve separate multiply/add as well,
    # including compilers configured to target FMA-capable hosts by default.
    target_compile_options(RuntimeTest.SimdDispatch PRIVATE -ffp-contract=off)
endif()
if(OpenMP_CXX_FOUND)
    target_link_libraries(RuntimeTest.SimdDispatch PRIVATE OpenMP::OpenMP_CXX)
endif()
add_test(NAME RuntimeTest.SimdDispatch COMMAND RuntimeTest.SimdDispatch)
add_executable(RuntimeTest.FiniteDispatch test_runtime_finite_dispatch.cpp
    ../src/runtimes/cpu/FiniteCheckDispatch.cpp ../src/runtimes/cpu/SimdDispatch.cpp)
target_compile_features(RuntimeTest.FiniteDispatch PRIVATE cxx_std_17)
target_include_directories(RuntimeTest.FiniteDispatch PRIVATE ../src)
if(OpenMP_CXX_FOUND)
    target_link_libraries(RuntimeTest.FiniteDispatch PRIVATE OpenMP::OpenMP_CXX)
endif()
add_test(NAME RuntimeTest.FiniteDispatch COMMAND RuntimeTest.FiniteDispatch)
set_tests_properties(RuntimeTest.FiniteDispatch PROPERTIES TIMEOUT 30)
add_executable(RuntimeTest.AttentionBackward test_attention_backward.cpp
    ../src/runtimes/cpu/AttentionBackward.cpp ../src/runtimes/cpu/SimdDispatch.cpp)
target_compile_features(RuntimeTest.AttentionBackward PRIVATE cxx_std_17)
target_include_directories(RuntimeTest.AttentionBackward PRIVATE ../src)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(RuntimeTest.AttentionBackward PRIVATE -ffp-contract=off)
endif()
if(OpenMP_CXX_FOUND)
    target_link_libraries(RuntimeTest.AttentionBackward PRIVATE OpenMP::OpenMP_CXX)
endif()
add_test(NAME RuntimeTest.AttentionBackward COMMAND RuntimeTest.AttentionBackward)
set_tests_properties(RuntimeTest.AttentionBackward PROPERTIES TIMEOUT 30 ENVIRONMENT "OMP_DYNAMIC=FALSE")
mimir_add_test(RuntimeTest.AttentionLayers test_attention_layers.cpp)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(RuntimeTest.AttentionLayers PRIVATE -ffp-contract=off)
endif()
set_tests_properties(RuntimeTest.AttentionLayers PROPERTIES TIMEOUT 30 ENVIRONMENT "OMP_DYNAMIC=FALSE")
add_test(NAME RuntimeTest.SimdInvalidISA COMMAND RuntimeTest.SimdDispatch --expect-env-rejection)
set_tests_properties(RuntimeTest.SimdDispatch PROPERTIES TIMEOUT 30)
set_tests_properties(RuntimeTest.SimdInvalidISA PROPERTIES TIMEOUT 15 ENVIRONMENT "MIMIR_CPU_ISA=unknown-isa")

mimir_add_test(RuntimeTest.ConvND test_runtime_conv_nd.cpp)
set_tests_properties(RuntimeTest.ConvND PROPERTIES TIMEOUT 60)
find_program(MIMIR_RUNTIME_AUDIT_PYTHON NAMES python3)
if(MIMIR_RUNTIME_AUDIT_PYTHON)
    add_test(NAME RuntimeTest.DispatchBoundary
        COMMAND ${MIMIR_RUNTIME_AUDIT_PYTHON}
            ${CMAKE_SOURCE_DIR}/scripts/dev/audit_runtime_dispatch.py
            --output ${CMAKE_BINARY_DIR}/runtime_dispatch_audit.json)
    set_tests_properties(RuntimeTest.DispatchBoundary PROPERTIES TIMEOUT 15)
endif()

mimir_add_test(VizTest.GenericLatentSpace test_latent_space_generic.cpp)
mimir_add_test(VizTest.GenericLatentControl test_latent_space_control.cpp)
set_tests_properties(VizTest.GenericLatentSpace VizTest.GenericLatentControl PROPERTIES
    TIMEOUT 60 ENVIRONMENT "OMP_NUM_THREADS=2;MIMIR_DISABLE_VULKAN=1;MIMIR_DISABLE_OPENCL=1;MIMIR_DISABLE_CUDA=1;MIMIR_DISABLE_ROCM=1;MIMIR_RUNTIME_AUTOTUNE=0;MIMIR_PLANNER_FUSION=0;MIMIR_PLANNER_BUFFER_REUSE=0")
add_test(NAME VizTest.GenericProjection COMMAND VizTest.ButtonEvents --projection-only)
set_tests_properties(VizTest.GenericProjection PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 30)
add_test(NAME VizTest.ScrollPadding COMMAND VizTest.ButtonEvents --scroll-padding-only)
set_tests_properties(VizTest.ScrollPadding PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 30)
add_test(NAME VizTest.PanelManager COMMAND VizTest.ButtonEvents --panel-manager-only)
set_tests_properties(VizTest.PanelManager PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 30)
add_test(NAME VizTest.Export3D COMMAND VizTest.ButtonEvents --export-3d-only)
set_tests_properties(VizTest.Export3D PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 60)

mimir_add_test(ModelTest.VAEConvModes test_vae_conv_modes.cpp)
set_tests_properties(ModelTest.VAEConvModes PROPERTIES TIMEOUT 90
    ENVIRONMENT "OMP_NUM_THREADS=1;MIMIR_DISABLE_VULKAN=1;MIMIR_DISABLE_OPENCL=1;MIMIR_DISABLE_CUDA=1;MIMIR_DISABLE_ROCM=1;MIMIR_RUNTIME_AUTOTUNE=0;MIMIR_PLANNER_FUSION=0;MIMIR_PLANNER_BUFFER_REUSE=0")

mimir_add_test(RuntimeTest.CausalSequence test_runtime_causal_sequence.cpp)
add_test(NAME RuntimeTest.VulkanCausalSequence COMMAND RuntimeTest.CausalSequence --vulkan)
set_tests_properties(RuntimeTest.CausalSequence PROPERTIES TIMEOUT 60 ENVIRONMENT "OMP_NUM_THREADS=1;MIMIR_RUNTIME_AUTOTUNE=0")
set_tests_properties(RuntimeTest.VulkanCausalSequence PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 120
    LABELS "GPU;VULKAN" ENVIRONMENT "OMP_NUM_THREADS=1;MIMIR_RUNTIME_AUTOTUNE=0")
mimir_add_test(ModelTest.CausalSequenceLayers test_model_causal_sequence.cpp)
set_tests_properties(ModelTest.CausalSequenceLayers PROPERTIES TIMEOUT 90
    ENVIRONMENT "OMP_NUM_THREADS=1;MIMIR_DISABLE_VULKAN=1;MIMIR_DISABLE_OPENCL=1;MIMIR_DISABLE_CUDA=1;MIMIR_DISABLE_ROCM=1;MIMIR_RUNTIME_AUTOTUNE=0;MIMIR_PLANNER_FUSION=0;MIMIR_PLANNER_BUFFER_REUSE=0")

mimir_add_test(RuntimeTest.PrimerOps test_runtime_primer.cpp)
add_test(NAME RuntimeTest.VulkanPrimerOps COMMAND RuntimeTest.PrimerOps --vulkan)
add_test(NAME ModelTest.PrimerOps COMMAND RuntimeTest.PrimerOps --model)
set_tests_properties(RuntimeTest.PrimerOps PROPERTIES TIMEOUT 90 ENVIRONMENT "OMP_NUM_THREADS=1;MIMIR_RUNTIME_AUTOTUNE=0")
set_tests_properties(RuntimeTest.VulkanPrimerOps PROPERTIES TIMEOUT 180 SKIP_RETURN_CODE 77
    LABELS "GPU;VULKAN" ENVIRONMENT "OMP_NUM_THREADS=1;MIMIR_RUNTIME_AUTOTUNE=0")
set_tests_properties(ModelTest.PrimerOps PROPERTIES TIMEOUT 60
    ENVIRONMENT "OMP_NUM_THREADS=1;MIMIR_DISABLE_VULKAN=1;MIMIR_DISABLE_OPENCL=1;MIMIR_DISABLE_CUDA=1;MIMIR_DISABLE_ROCM=1;MIMIR_RUNTIME_AUTOTUNE=0")
add_test(NAME LuaTest.PrimerOps COMMAND $<TARGET_FILE:mimir> --lua scripts/tests/test_primer_ops.lua)
set_tests_properties(LuaTest.PrimerOps PROPERTIES TIMEOUT 45 WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    ENVIRONMENT "OMP_NUM_THREADS=1;MIMIR_DISABLE_VULKAN=1;MIMIR_DISABLE_OPENCL=1;MIMIR_DISABLE_CUDA=1;MIMIR_DISABLE_ROCM=1;MIMIR_RUNTIME_AUTOTUNE=0")

add_executable(VizTest.LatentCorrelationMath test_latent_correlations.cpp)
target_compile_features(VizTest.LatentCorrelationMath PRIVATE cxx_std_17)
target_include_directories(VizTest.LatentCorrelationMath PRIVATE ../src)
add_test(NAME VizTest.LatentCorrelationMath COMMAND VizTest.LatentCorrelationMath)
set_tests_properties(VizTest.LatentCorrelationMath PROPERTIES TIMEOUT 30)
add_executable(VizTest.OrganicNetwork test_organic_network.cpp)
target_compile_features(VizTest.OrganicNetwork PRIVATE cxx_std_17)
target_include_directories(VizTest.OrganicNetwork PRIVATE ../src)
add_test(NAME VizTest.OrganicNetwork COMMAND VizTest.OrganicNetwork)
add_executable(VizTest.LatentSpaceExport test_latent_space_export.cpp)
target_compile_features(VizTest.LatentSpaceExport PRIVATE cxx_std_17)
target_include_directories(VizTest.LatentSpaceExport PRIVATE ../src)
add_test(NAME VizTest.LatentSpaceExport COMMAND VizTest.LatentSpaceExport)
set_tests_properties(VizTest.LatentSpaceExport PROPERTIES TIMEOUT 30)
mimir_add_test(VizTest.LatentCorrelationControl test_latent_correlation_control.cpp)
mimir_add_test(ModelTest.LatentCorrelations test_model_latent_correlations.cpp)
set_tests_properties(VizTest.LatentCorrelationControl ModelTest.LatentCorrelations PROPERTIES
    TIMEOUT 90 ENVIRONMENT "OMP_NUM_THREADS=1;MIMIR_DISABLE_VULKAN=1;MIMIR_DISABLE_OPENCL=1;MIMIR_DISABLE_CUDA=1;MIMIR_DISABLE_ROCM=1;MIMIR_RUNTIME_AUTOTUNE=0")

mimir_add_test(ModelTest.NeuronNetwork test_neuron_network.cpp)
mimir_add_test(VizTest.NeuronNetworkControl test_neuron_network_control.cpp)
set_tests_properties(ModelTest.NeuronNetwork VizTest.NeuronNetworkControl PROPERTIES
    TIMEOUT 90 ENVIRONMENT "OMP_NUM_THREADS=1;MIMIR_DISABLE_VULKAN=1;MIMIR_DISABLE_OPENCL=1;MIMIR_DISABLE_CUDA=1;MIMIR_DISABLE_ROCM=1;MIMIR_RUNTIME_AUTOTUNE=0;MIMIR_PLANNER_FUSION=0;MIMIR_PLANNER_BUFFER_REUSE=0")

mimir_add_test(ModelTest.LiveWeightNetwork test_live_weight_network.cpp)
mimir_add_test(VizTest.LiveWeightControl test_live_weight_control.cpp)
set_tests_properties(ModelTest.LiveWeightNetwork VizTest.LiveWeightControl PROPERTIES
    TIMEOUT 90 ENVIRONMENT "OMP_NUM_THREADS=1;MIMIR_DISABLE_VULKAN=1;MIMIR_DISABLE_OPENCL=1;MIMIR_DISABLE_CUDA=1;MIMIR_DISABLE_ROCM=1;MIMIR_RUNTIME_AUTOTUNE=0;MIMIR_PLANNER_FUSION=0;MIMIR_PLANNER_BUFFER_REUSE=0")

add_executable(RuntimeTest.ReusableScratch test_runtime_reusable_scratch.cpp)
target_compile_features(RuntimeTest.ReusableScratch PRIVATE cxx_std_17)
target_include_directories(RuntimeTest.ReusableScratch PRIVATE ../src)
target_link_libraries(RuntimeTest.ReusableScratch PRIVATE Threads::Threads)
add_test(NAME RuntimeTest.ReusableScratch COMMAND RuntimeTest.ReusableScratch)
set_tests_properties(RuntimeTest.ReusableScratch PROPERTIES TIMEOUT 30)

add_executable(RuntimeBench.Buffers bench_runtime_buffers.cpp test_log_stub.cpp)
target_compile_features(RuntimeBench.Buffers PRIVATE cxx_std_17)
target_link_libraries(RuntimeBench.Buffers PRIVATE mimir_core)

mimir_add_test(ModelTest.TypedSnapshots test_model_typed_snapshots.cpp)
add_executable(FrameworkTest.RuntimeRoute test_runtime_route.cpp)
target_compile_features(FrameworkTest.RuntimeRoute PRIVATE cxx_std_17)
target_include_directories(FrameworkTest.RuntimeRoute PRIVATE ../src)
add_test(NAME FrameworkTest.RuntimeRoute COMMAND FrameworkTest.RuntimeRoute)
add_executable(MetricsTest.BatchCsv test_metrics_batch.cpp)
target_compile_features(MetricsTest.BatchCsv PRIVATE cxx_std_17)
target_include_directories(MetricsTest.BatchCsv PRIVATE ../src)
add_test(NAME MetricsTest.BatchCsv COMMAND MetricsTest.BatchCsv)
if(ENABLE_OPENCL)
    add_executable(RuntimeBench.OpenCLBuffers bench_opencl_buffers.cpp)
    target_compile_features(RuntimeBench.OpenCLBuffers PRIVATE cxx_std_17)
    target_compile_definitions(RuntimeBench.OpenCLBuffers PRIVATE MIMIR_VERIFY_OPENCL_CACHE)
    target_link_libraries(RuntimeBench.OpenCLBuffers PRIVATE mimir_core)
    add_test(NAME RuntimeTest.OpenCLBufferReuse COMMAND RuntimeBench.OpenCLBuffers)
    set_tests_properties(RuntimeTest.OpenCLBufferReuse PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 120)
endif()

mimir_add_test(RuntimeTest.OptimizerPortableSimd test_optimizer_portable_simd.cpp)

set_tests_properties(RuntimeTest.OptimizerPortableSimd PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=3;OMP_DYNAMIC=FALSE")
if(ENABLE_OPENCL AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_executable(RuntimeTest.OpenCLUploadLifetime test_opencl_upload_lifetime.cpp)
    target_compile_features(RuntimeTest.OpenCLUploadLifetime PRIVATE cxx_std_17)
    target_link_libraries(RuntimeTest.OpenCLUploadLifetime PRIVATE mimir_core)
    target_link_options(RuntimeTest.OpenCLUploadLifetime PRIVATE
        "-Wl,--wrap=clEnqueueWriteBuffer,--wrap=clSetKernelArg,--wrap=clEnqueueReadBuffer,--wrap=clFinish")
    add_test(NAME RuntimeTest.OpenCLUploadLifetime COMMAND RuntimeTest.OpenCLUploadLifetime)
    set_tests_properties(RuntimeTest.OpenCLUploadLifetime PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 120)
    add_executable(RuntimeTest.OpenCLBufferFailures test_opencl_buffer_failures.cpp)
    target_compile_features(RuntimeTest.OpenCLBufferFailures PRIVATE cxx_std_17)
    target_link_libraries(RuntimeTest.OpenCLBufferFailures PRIVATE mimir_core)
    target_link_options(RuntimeTest.OpenCLBufferFailures PRIVATE "-Wl,--wrap=clCreateBuffer")
    add_test(NAME RuntimeTest.OpenCLBufferFailures COMMAND RuntimeTest.OpenCLBufferFailures)
    set_tests_properties(RuntimeTest.OpenCLBufferFailures PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 120)
endif()
