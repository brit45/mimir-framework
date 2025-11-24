#ifndef __TENSORS_LIB_HPP__
#define __TENSORS_LIB_HPP__

#include <iostream>
#include <vector>
#include <cstdint>
#define CL_TARGET_OPENCL_VERSION 300 // ou 300 si tu veux explicite 3.0
#include <CL/cl.h>

struct Vector4F {

    float           X = 0.0f;
    float           Y = 0.0f;
    float           Z = 0.0f;
    float           W = 0.0f;
};

struct tensor {
    
    std::uint16_t   Weight = 0;
    Vector4F        Pos;
    std::uint8_t    Value = 0;
    std::uint16_t   Length = 0;
    
    // Extension pour supporter les paramètres float du modèle
    std::vector<float> data;
    
    // Constructeur par défaut
    tensor() = default;
    
    // Constructeur avec taille
    explicit tensor(size_t size) : data(size, 0.0f) {}
    
    // Constructeur avec données
    explicit tensor(const std::vector<float>& values) : data(values) {}
};

class TensorSystem {
public:
    TensorSystem();
    ~TensorSystem();

    // Initialize OpenCL context and kernel
    bool initialize();

    // Compute weights for a vector of tensors using OpenCL
    bool computeWeights(std::vector<tensor>& tensors);

private:
    cl_platform_id platform_id = nullptr;
    cl_device_id device_id = nullptr;
    cl_context context = nullptr;
    cl_command_queue command_queue = nullptr;
    cl_program program = nullptr;
    cl_kernel kernel = nullptr;

    bool initialized = false;

    // Helper to load and build OpenCL kernel
    bool buildKernel(const char* kernelSource);

    // OpenCL kernel source for weight calculation (define in .cpp)
    static const char* weightKernelSource;
};


#endif // __TENSORS_LIB_HPP__