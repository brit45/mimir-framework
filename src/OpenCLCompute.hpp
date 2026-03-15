#ifndef __OPENCL_COMPUTE_HPP__
#define __OPENCL_COMPUTE_HPP__

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#ifdef ENABLE_OPENCL
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#endif

namespace OpenCLCompute {

// Moteur OpenCL minimal, pensé pour coexister avec Vulkan.
// Objectif immédiat: accélérer certaines opérations (ex: Linear) au runtime.
class ComputeEngine {
public:
    ComputeEngine() = default;
    ~ComputeEngine() { cleanup(); }

    ComputeEngine(const ComputeEngine&) = delete;
    ComputeEngine& operator=(const ComputeEngine&) = delete;

    bool initialize() {
#ifndef ENABLE_OPENCL
        initialized_ = false;
        return false;
#else
        if (initialized_) return true;

        cl_int err = CL_SUCCESS;
        cl_uint num_platforms = 0;
        err = clGetPlatformIDs(0, nullptr, &num_platforms);
        if (err != CL_SUCCESS || num_platforms == 0) {
            return false;
        }

        std::vector<cl_platform_id> platforms(num_platforms);
        err = clGetPlatformIDs(num_platforms, platforms.data(), nullptr);
        if (err != CL_SUCCESS) {
            return false;
        }

        // Choix: GPU d'abord, sinon CPU
        cl_platform_id chosen_platform = nullptr;
        cl_device_id chosen_device = nullptr;
        auto pick = [&](cl_device_type t) {
            for (cl_uint i = 0; i < num_platforms; ++i) {
                cl_uint dev_count = 0;
                if (clGetDeviceIDs(platforms[i], t, 0, nullptr, &dev_count) == CL_SUCCESS && dev_count > 0) {
                    std::vector<cl_device_id> devs(dev_count);
                    if (clGetDeviceIDs(platforms[i], t, dev_count, devs.data(), nullptr) == CL_SUCCESS) {
                        chosen_platform = platforms[i];
                        chosen_device = devs[0];
                        return true;
                    }
                }
            }
            return false;
        };
        if (!pick(CL_DEVICE_TYPE_GPU)) {
            (void)pick(CL_DEVICE_TYPE_CPU);
        }

        if (!chosen_device) {
            return false;
        }

        platform_ = chosen_platform;
        device_ = chosen_device;

        context_ = clCreateContext(nullptr, 1, &device_, nullptr, nullptr, &err);
        if (err != CL_SUCCESS || !context_) {
            cleanup();
            return false;
        }

        queue_ = clCreateCommandQueueWithProperties(context_, device_, nullptr, &err);
        if (err != CL_SUCCESS || !queue_) {
            cleanup();
            return false;
        }

        if (!buildProgramAndKernels()) {
            cleanup();
            return false;
        }

        initialized_ = true;
        return true;
#endif
    }

    bool isInitialized() const { return initialized_; }

    void cleanup() {
#ifdef ENABLE_OPENCL
        if (kernel_linear_) { clReleaseKernel(kernel_linear_); kernel_linear_ = nullptr; }
        if (program_) { clReleaseProgram(program_); program_ = nullptr; }
        if (queue_) { clReleaseCommandQueue(queue_); queue_ = nullptr; }
        if (context_) { clReleaseContext(context_); context_ = nullptr; }
        platform_ = nullptr;
        device_ = nullptr;
#endif
        initialized_ = false;
    }

    // Linear: input [batch, in_f] ; weights [out_f, in_f] ; bias [out_f]
    // output [batch, out_f]
    bool linearForward(
        const float* input,
        const float* weights,
        const float* bias_or_null,
        float* output,
        int batch,
        int in_f,
        int out_f
    ) {
#ifndef ENABLE_OPENCL
        (void)input; (void)weights; (void)bias_or_null; (void)output; (void)batch; (void)in_f; (void)out_f;
        return false;
#else
        if (!initialized_ || !context_ || !queue_ || !kernel_linear_) return false;
        if (!input || !weights || !output) return false;
        if (batch <= 0 || in_f <= 0 || out_f <= 0) return false;

        cl_int err = CL_SUCCESS;
        const size_t bytes_in = static_cast<size_t>(batch) * static_cast<size_t>(in_f) * sizeof(float);
        const size_t bytes_w = static_cast<size_t>(out_f) * static_cast<size_t>(in_f) * sizeof(float);
        const size_t bytes_b = static_cast<size_t>(out_f) * sizeof(float);
        const size_t bytes_out = static_cast<size_t>(batch) * static_cast<size_t>(out_f) * sizeof(float);

        // Note: allocation simple par appel. Optimisable via cache/pool ensuite.
        cl_mem buf_in = clCreateBuffer(context_, CL_MEM_READ_ONLY, bytes_in, nullptr, &err);
        if (err != CL_SUCCESS || !buf_in) return false;
        cl_mem buf_w = clCreateBuffer(context_, CL_MEM_READ_ONLY, bytes_w, nullptr, &err);
        if (err != CL_SUCCESS || !buf_w) { clReleaseMemObject(buf_in); return false; }
        cl_mem buf_b = clCreateBuffer(context_, CL_MEM_READ_ONLY, bytes_b, nullptr, &err);
        if (err != CL_SUCCESS || !buf_b) { clReleaseMemObject(buf_in); clReleaseMemObject(buf_w); return false; }
        cl_mem buf_out = clCreateBuffer(context_, CL_MEM_WRITE_ONLY, bytes_out, nullptr, &err);
        if (err != CL_SUCCESS || !buf_out) { clReleaseMemObject(buf_in); clReleaseMemObject(buf_w); clReleaseMemObject(buf_b); return false; }

        err = clEnqueueWriteBuffer(queue_, buf_in, CL_TRUE, 0, bytes_in, input, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) { release4(buf_in, buf_w, buf_b, buf_out); return false; }
        err = clEnqueueWriteBuffer(queue_, buf_w, CL_TRUE, 0, bytes_w, weights, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) { release4(buf_in, buf_w, buf_b, buf_out); return false; }

        if (bias_or_null) {
            err = clEnqueueWriteBuffer(queue_, buf_b, CL_TRUE, 0, bytes_b, bias_or_null, 0, nullptr, nullptr);
        } else {
            zeroBiasScratch_.assign(static_cast<size_t>(out_f), 0.0f);
            err = clEnqueueWriteBuffer(queue_, buf_b, CL_TRUE, 0, bytes_b, zeroBiasScratch_.data(), 0, nullptr, nullptr);
        }
        if (err != CL_SUCCESS) { release4(buf_in, buf_w, buf_b, buf_out); return false; }

        err  = clSetKernelArg(kernel_linear_, 0, sizeof(cl_mem), &buf_in);
        err |= clSetKernelArg(kernel_linear_, 1, sizeof(cl_mem), &buf_w);
        err |= clSetKernelArg(kernel_linear_, 2, sizeof(cl_mem), &buf_b);
        err |= clSetKernelArg(kernel_linear_, 3, sizeof(cl_mem), &buf_out);
        err |= clSetKernelArg(kernel_linear_, 4, sizeof(int), &batch);
        err |= clSetKernelArg(kernel_linear_, 5, sizeof(int), &in_f);
        err |= clSetKernelArg(kernel_linear_, 6, sizeof(int), &out_f);
        if (err != CL_SUCCESS) { release4(buf_in, buf_w, buf_b, buf_out); return false; }

        // Global 2D: (batch, out_f)
        const size_t global[2] = { static_cast<size_t>(batch), static_cast<size_t>(out_f) };
        err = clEnqueueNDRangeKernel(queue_, kernel_linear_, 2, nullptr, global, nullptr, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) { release4(buf_in, buf_w, buf_b, buf_out); return false; }

        err = clEnqueueReadBuffer(queue_, buf_out, CL_TRUE, 0, bytes_out, output, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) { release4(buf_in, buf_w, buf_b, buf_out); return false; }

        release4(buf_in, buf_w, buf_b, buf_out);
        return true;
#endif
    }

private:
#ifdef ENABLE_OPENCL
    cl_platform_id platform_ = nullptr;
    cl_device_id device_ = nullptr;
    cl_context context_ = nullptr;
    cl_command_queue queue_ = nullptr;
    cl_program program_ = nullptr;
    cl_kernel kernel_linear_ = nullptr;

    std::vector<float> zeroBiasScratch_;

    static void release4(cl_mem a, cl_mem b, cl_mem c, cl_mem d) {
        if (a) clReleaseMemObject(a);
        if (b) clReleaseMemObject(b);
        if (c) clReleaseMemObject(c);
        if (d) clReleaseMemObject(d);
    }

    static void printBuildLog(cl_program prog, cl_device_id device) {
        size_t log_size = 0;
        clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        if (log_size == 0) return;
        std::string log(log_size, '\0');
        clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
        std::cerr << "OpenCL build log:\n" << log << "\n";
    }

    bool buildProgramAndKernels() {
        cl_int err = CL_SUCCESS;
        const char* src = R"CLC(
__kernel void linear_forward(
    __global const float* input,      // [batch, in_f]
    __global const float* weights,    // [out_f, in_f]
    __global const float* bias,       // [out_f]
    __global float* output,           // [batch, out_f]
    int batch,
    int in_f,
    int out_f
) {
    const int b = (int)get_global_id(0);
    const int o = (int)get_global_id(1);
    if (b >= batch || o >= out_f) return;
    float acc = bias[o];
    const int in_base = b * in_f;
    const int w_base = o * in_f;
    for (int k = 0; k < in_f; ++k) {
        acc += input[in_base + k] * weights[w_base + k];
    }
    output[b * out_f + o] = acc;
}
)CLC";

        const size_t src_len = std::strlen(src);
        program_ = clCreateProgramWithSource(context_, 1, &src, &src_len, &err);
        if (err != CL_SUCCESS || !program_) return false;

        err = clBuildProgram(program_, 1, &device_, nullptr, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            printBuildLog(program_, device_);
            return false;
        }

        kernel_linear_ = clCreateKernel(program_, "linear_forward", &err);
        if (err != CL_SUCCESS || !kernel_linear_) return false;
        return true;
    }
#endif

    bool initialized_ = false;
};

} // namespace OpenCLCompute

#endif // __OPENCL_COMPUTE_HPP__