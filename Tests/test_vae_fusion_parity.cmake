# Fresh processes give the private Reparameterize RNG identical initial state.
foreach(mode 0 1)
    set(output "${CMAKE_CURRENT_BINARY_DIR}/vae-fusion-parity-${mode}.json")
    execute_process(
        COMMAND "${CMAKE_COMMAND}" -E env OMP_NUM_THREADS=1
            MIMIR_DISABLE_VULKAN=1 MIMIR_DISABLE_OPENCL=1
            MIMIR_DISABLE_CUDA=1 MIMIR_DISABLE_ROCM=1
            MIMIR_ENABLE_PLANNER=1 MIMIR_ENABLE_FUSION_TRAIN=1
            MIMIR_PLANNER_MODE=cost MIMIR_RUNTIME_AUTOTUNE=1
            "${MIMIR_FUSION_TEST}" --vae-branch-parity "${mode}" "${output}"
        RESULT_VARIABLE result OUTPUT_QUIET ERROR_VARIABLE diagnostics TIMEOUT 120)
    if(NOT "${result}" STREQUAL "0")
        message(FATAL_ERROR "VAEConv fusion=${mode} failed: ${result}\n${diagnostics}")
    endif()
    file(SHA256 "${output}" hash_${mode})
endforeach()
if(NOT "${hash_0}" STREQUAL "${hash_1}")
    message(FATAL_ERROR "Stochastic VAEConv outputs, gradients or Lion weights differ with fusion")
endif()
file(REMOVE "${CMAKE_CURRENT_BINARY_DIR}/vae-fusion-parity-0.json"
    "${CMAKE_CURRENT_BINARY_DIR}/vae-fusion-parity-1.json")
message(STATUS "Stochastic VAEConv fusion: exact outputs, gradients and Lion weights over three steps")
