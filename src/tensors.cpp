#include "tensors.hpp"
#include "DynamicTensorAllocator.hpp"
#include "include/json.hpp"

#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>

using json = nlohmann::json;

// ============================================================================
// Implémentations tensor avec allocation dynamique
// ============================================================================

tensor::tensor(size_t size, bool dynamic) : use_dynamic_alloc(dynamic) {
    if (dynamic) {
        // 🔑 ALLOCATION DYNAMIQUE LAZY
        // Crée le handle mais n'alloue pas immédiatement la mémoire
        // La mémoire sera allouée lors du premier getData() (lazy loading)
        // Cela permet de ne pas comptabiliser la RAM avant utilisation réelle
        auto& allocator = DynamicTensorAllocator::instance();
        dynamic_handle = allocator.allocateTensor(size, "tensor_data");
        
        if (!dynamic_handle) {
            // 🛑 PANIC OOM: Impossible d'allouer même le handle
            std::cerr << "\n\u274c\u274c\u274c PANIC: OUT OF MEMORY \u274c\u274c\u274c" << std::endl;
            std::cerr << "⛔ Impossible d'allouer tensor de " << (size * sizeof(float) / 1024 / 1024) << " MB" << std::endl;
            std::cerr << "⛔ MemoryGuard a refusé l'allocation - limite atteinte" << std::endl;
            std::cerr << "\n🚨 ARRÊT CONTRÔLÉ POUR ÉVITER CRASH OS\n" << std::endl;
            
            // Afficher stats avant de quitter
            auto& guard = MemoryGuard::instance();
            guard.printStats();
            
            // Arrêt contrôlé (pas de throw pour éviter corruption)
            std::cerr << "\n⚠️  Le programme va s'arrêter proprement pour protéger l'OS..." << std::endl;
            std::exit(1);
        }
    } else {
        // Allocation classique std::vector (non contrôlée par MemoryGuard)
        // ⚠️ Utiliser dynamic=true en production pour contrôle strict!
        data.resize(size, 0.0f);
    }
}

tensor::tensor(tensor&& other) noexcept
    : Weight(other.Weight)
    , Pos(other.Pos)
    , Value(other.Value)
    , Length(other.Length)
    , data(std::move(other.data))
    , dynamic_handle(other.dynamic_handle)
    , use_dynamic_alloc(other.use_dynamic_alloc)
{
    other.dynamic_handle = nullptr;
    other.use_dynamic_alloc = false;
}

tensor& tensor::operator=(tensor&& other) noexcept {
    if (this != &other) {
        // Libérer les ressources existantes
        if (use_dynamic_alloc && dynamic_handle) {
            auto& allocator = DynamicTensorAllocator::instance();
            allocator.freeTensor(
                static_cast<DynamicTensorAllocator::TensorHandle*>(dynamic_handle));
        }
        
        // Transférer les ressources
        Weight = other.Weight;
        Pos = other.Pos;
        Value = other.Value;
        Length = other.Length;
        data = std::move(other.data);
        dynamic_handle = other.dynamic_handle;
        use_dynamic_alloc = other.use_dynamic_alloc;
        
        // Invalider l'autre
        other.dynamic_handle = nullptr;
        other.use_dynamic_alloc = false;
    }
    return *this;
}


tensor::~tensor() {
    if (use_dynamic_alloc && dynamic_handle) {
        auto& allocator = DynamicTensorAllocator::instance();
        allocator.freeTensor(
            static_cast<DynamicTensorAllocator::TensorHandle*>(dynamic_handle));
        dynamic_handle = nullptr;
    }
}

float* tensor::getData() {
    if (use_dynamic_alloc && dynamic_handle) {
        auto& allocator = DynamicTensorAllocator::instance();
        return allocator.getTensorData(
            static_cast<DynamicTensorAllocator::TensorHandle*>(dynamic_handle));
    }
    return data.data();
}

const float* tensor::getData() const {
    if (use_dynamic_alloc && dynamic_handle) {
        auto& allocator = DynamicTensorAllocator::instance();
        return allocator.getTensorData(
            static_cast<DynamicTensorAllocator::TensorHandle*>(
                const_cast<void*>(dynamic_handle)));
    }
    return data.data();
}

size_t tensor::getSize() const {
    if (use_dynamic_alloc && dynamic_handle) {
        auto handle = static_cast<DynamicTensorAllocator::TensorHandle*>(
            const_cast<void*>(dynamic_handle));
        return handle->size;
    }
    return data.size();
}

// ============================================================================
// TensorSystem (OpenCL)
// ============================================================================


const char *TensorSystem::weightKernelSource = R"CLC(
// Approximation rapide de tanh (Padé, 2× plus rapide que tanh natif)
inline float fast_tanh(float x) {
    x = fmin(fmax(x, -3.0f), 3.0f);  // Clipping réduit (tanh sature à ±3)
    const float x2 = x * x;
    return x * (27.0f + x2) / (27.0f + 9.0f * x2);
}

// Approximation rapide de sigmoid
inline float fast_sigmoid(float x) {
    x = fmin(fmax(x, -10.0f), 10.0f);
    // Approximation rationnelle: 1/(1+exp(-x)) ≈ 0.5 + x/(2*(1+|x|/2))
    const float abs_x = fabs(x);
    return 0.5f + 0.5f * x / (1.0f + abs_x * 0.5f);
}

// Constantes précalculées
#define INV_255 0.00392156862f    // 1/255
#define INV_1024 0.0009765625f     // 1/1024
#define HALF_PI 1.57079632679f     // π/2

__kernel void compute_weights(__global const float4* pos,
                              __global const uchar*  val,
                              __global const ushort* len,
                              __global ushort*       out)
{
    const uint i = get_global_id(0);

    // Charger données avec vectorisation native
    const float4 p = pos[i];
    
    // === COMPOSANTE SPATIALE (optimisée avec mad) ===
    const float r2 = mad(p.x, p.x, mad(p.y, p.y, p.z * p.z));  // FMA natif
    const float r = native_sqrt(r2);  // sqrt GPU rapide
    const float spatial_nl = fast_tanh(r * 0.5f);

    // === COMPOSANTE DYNAMIQUE (constantes précalculées) ===
    const float v01 = (float)val[i] * INV_255;
    const float l01 = (float)len[i] * INV_1024;
    const float vel_nl = fast_tanh(mad(v01, 2.0f, -1.0f));  // (v01-0.5)*2 = v01*2-1
    const float den_nl = fast_tanh(mad(clamp(l01, 0.0f, 1.0f), 2.0f, -1.0f));

    // === EMBEDDING CONTEXTUEL (sin/cos GPU natifs) ===
    const float w = p.w;
    
    // Utiliser native_sin/cos pour GPUs modernes (4× plus rapide)
    const float w_high = w * 0.50f;
    const float w_mid = w * 0.10f;
    const float w_low = w * 0.02f;
    
    const float s1 = native_sin(w_high);
    const float c1 = native_cos(w_high);
    const float s2 = native_sin(w_mid);
    const float c2 = native_cos(w_mid);
    const float s3 = native_sin(w_low);
    const float c3 = native_cos(w_low);

    // Combinaison avec FMA
    const float ctx_core = fast_tanh(w * 0.05f);
    const float ctx_feat = mad(0.50f, s1, mad(0.35f, c1, 
                           mad(0.30f, s2, mad(0.20f, c2, 
                           mad(0.15f, s3, 0.10f * c3)))));
    const float ctx_sum = clamp(mad(0.6f, ctx_feat, ctx_core), -1.0f, 1.0f);

    // === GATING MECHANISM ===
    const float gate = fast_sigmoid(2.2f * ctx_sum);

    const float w_spatial = mad(1.0f - gate, 0.95f, 0.05f);  // (1-gate)*0.95 + 0.05
    const float w_dyn = gate * 0.90f;

    // === COMBINAISON PONDÉRÉE ===
    const float dyn_component = mad(-0.7f, den_nl, vel_nl);  // vel - 0.7*den
    const float combined = mad(w_spatial, spatial_nl, w_dyn * dyn_component);

    // === NORMALISATION FINALE ===
    float s = fast_sigmoid(2.0f * combined);
    s = mad(0.98f, s, 0.01f);  // 0.98*s + 0.01

    // Quantification sur 16 bits avec arrondi
    const int iw = convert_int_sat(s * 65535.0f + 0.5f);  // Saturation automatique
    out[i] = (ushort)clamp(iw, 0, 65535);
}
)CLC";

// Helper: print OpenCL build log
static void printBuildLog(cl_program prog, cl_device_id device) {
    size_t log_size = 0;
    clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
    if (log_size == 0) return;
    std::string log(log_size, '\0');
    clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, log_size, &log[0], nullptr);
    std::cerr << "OpenCL build log:\n" << log << "\n";
}

TensorSystem::TensorSystem()
    : platform_id(nullptr), device_id(nullptr),
      context(nullptr), command_queue(nullptr),
      program(nullptr), kernel(nullptr), initialized(false)
{}

TensorSystem::~TensorSystem() {
    if (kernel) { clReleaseKernel(kernel); kernel = nullptr; }
    if (program) { clReleaseProgram(program); program = nullptr; }
    if (command_queue) { clReleaseCommandQueue(command_queue); command_queue = nullptr; }
    if (context) { clReleaseContext(context); context = nullptr; }
}

// Build OpenCL kernel from source (assumes context & device_id are valid)
bool TensorSystem::buildKernel(const char* kernelSource) {
    cl_int err = CL_SUCCESS;
    size_t src_len = std::strlen(kernelSource);
    program = clCreateProgramWithSource(context, 1, &kernelSource, &src_len, &err);
    if (err != CL_SUCCESS || !program) {
        std::cerr << "TensorSystem::buildKernel: clCreateProgramWithSource failed (err=" << err << ")\n";
        return false;
    }

    err = clBuildProgram(program, 1, &device_id, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "TensorSystem::buildKernel: clBuildProgram failed (err=" << err << ")\n";
        printBuildLog(program, device_id);
        clReleaseProgram(program);
        program = nullptr;
        return false;
    }

    kernel = clCreateKernel(program, "compute_weights", &err);
    if (err != CL_SUCCESS || !kernel) {
        std::cerr << "TensorSystem::buildKernel: clCreateKernel failed (err=" << err << ")\n";
        if (program) { clReleaseProgram(program); program = nullptr; }
        return false;
    }
    return true;
}

bool TensorSystem::initialize() {
    cl_int err = CL_SUCCESS;
    cl_uint num_platforms = 0;
    err = clGetPlatformIDs(0, nullptr, &num_platforms);
    if (err != CL_SUCCESS || num_platforms == 0) {
        std::cerr << "TensorSystem::initialize: No OpenCL platforms found — will fallback to CPU path\n";
        initialized = true;
        return true; // fallback allowed
    }

    std::vector<cl_platform_id> platforms(num_platforms);
    err = clGetPlatformIDs(num_platforms, platforms.data(), nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "TensorSystem::initialize: clGetPlatformIDs failed (err=" << err << ")\n";
        return false;
    }

    // Try to find a GPU device first, else CPU
    cl_platform_id chosen_platform = nullptr;
    cl_device_id chosen_device = nullptr;
    for (cl_uint i = 0; i < num_platforms; ++i) {
        cl_uint dev_count = 0;
        if (clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 0, nullptr, &dev_count) == CL_SUCCESS && dev_count > 0) {
            std::vector<cl_device_id> devs(dev_count);
            if (clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, dev_count, devs.data(), nullptr) == CL_SUCCESS) {
                chosen_platform = platforms[i];
                chosen_device = devs[0];
                break;
            }
        }
    }
    if (!chosen_device) {
        for (cl_uint i = 0; i < num_platforms; ++i) {
            cl_uint dev_count = 0;
            if (clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_CPU, 0, nullptr, &dev_count) == CL_SUCCESS && dev_count > 0) {
                std::vector<cl_device_id> devs(dev_count);
                if (clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_CPU, dev_count, devs.data(), nullptr) == CL_SUCCESS) {
                    chosen_platform = platforms[i];
                    chosen_device = devs[0];
                    break;
                }
            }
        }
    }

    if (!chosen_device) {
        std::cerr << "TensorSystem::initialize: No OpenCL device found — will fallback to CPU path\n";
        initialized = true;
        return true;
    }

    platform_id = chosen_platform;
    device_id = chosen_device;

    context = clCreateContext(nullptr, 1, &device_id, nullptr, nullptr, &err);
    if (err != CL_SUCCESS || !context) {
        std::cerr << "TensorSystem::initialize: clCreateContext failed (err=" << err << ")\n";
        // fallback allowed
        context = nullptr;
        initialized = true;
        return true;
    }

    // create command queue
    command_queue = clCreateCommandQueueWithProperties(context, device_id, nullptr, &err);
    if (err != CL_SUCCESS || !command_queue) {
        std::cerr << "TensorSystem::initialize: failed to create command queue (err=" << err << ")\n";
        if (context) { clReleaseContext(context); context = nullptr; }
        context = nullptr;
        initialized = true;
        return true;
    }

    // build kernel (best-effort)
    if (!buildKernel(weightKernelSource)) {
        std::cerr << "TensorSystem::initialize: buildKernel failed — releasing OpenCL resources and fallback to CPU\n";
        if (kernel) { clReleaseKernel(kernel); kernel = nullptr; }
        if (program) { clReleaseProgram(program); program = nullptr; }
        if (command_queue) { clReleaseCommandQueue(command_queue); command_queue = nullptr; }
        if (context) { clReleaseContext(context); context = nullptr; }
        initialized = true;
        return true;
    }

    initialized = true;
    return true;
}

bool TensorSystem::computeWeights(std::vector<tensor>& tensors) {
    if (!initialized) {
        std::cerr << "TensorSystem::computeWeights: system not initialized\n";
        return false;
    }
    const size_t n = tensors.size();
    if (n == 0) return true;

    // If OpenCL is available and kernel prepared, try GPU path
    if (context && command_queue && program && kernel) {
        cl_int err = CL_SUCCESS;

        // prepare host arrays (use portable host-side types)
        struct Float4 { float x, y, z, w; };
        std::vector<Float4> pos(n);
        std::vector<unsigned char> val(n);
        std::vector<unsigned short> len(n);
        std::vector<unsigned short> out(n);

        #pragma omp parallel for if(n > 2048) schedule(static)
        for (size_t i = 0; i < n; ++i) {
            pos[i].x = tensors[i].Pos.X;
            pos[i].y = tensors[i].Pos.Y;
            pos[i].z = tensors[i].Pos.Z;
            pos[i].w = tensors[i].Pos.W;
            val[i] = static_cast<unsigned char>(tensors[i].Value);
            len[i] = static_cast<unsigned short>(tensors[i].Length);
        }

        cl_mem posBuf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       sizeof(Float4) * n, pos.data(), &err);
        if (err != CL_SUCCESS) { std::cerr << "computeWeights: clCreateBuffer pos failed (err=" << err << ")\n"; return false; }

        cl_mem valBuf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       sizeof(unsigned char) * n, val.data(), &err);
        if (err != CL_SUCCESS) { clReleaseMemObject(posBuf); std::cerr << "computeWeights: clCreateBuffer val failed (err=" << err << ")\n"; return false; }

        cl_mem lenBuf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       sizeof(unsigned short) * n, len.data(), &err);
        if (err != CL_SUCCESS) { clReleaseMemObject(posBuf); clReleaseMemObject(valBuf); std::cerr << "computeWeights: clCreateBuffer len failed (err=" << err << ")\n"; return false; }

        cl_mem outBuf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(unsigned short) * n, nullptr, &err);
        if (err != CL_SUCCESS) { clReleaseMemObject(posBuf); clReleaseMemObject(valBuf); clReleaseMemObject(lenBuf); std::cerr << "computeWeights: clCreateBuffer out failed (err=" << err << ")\n"; return false; }

        // set args
        err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &posBuf);
        err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &valBuf);
        err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &lenBuf);
        err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &outBuf);
        if (err != CL_SUCCESS) {
            std::cerr << "computeWeights: clSetKernelArg failed (err=" << err << ")\n";
            clReleaseMemObject(posBuf); clReleaseMemObject(valBuf); clReleaseMemObject(lenBuf); clReleaseMemObject(outBuf);
            return false;
        }

        size_t global = n;
        err = clEnqueueNDRangeKernel(command_queue, kernel, 1, nullptr, &global, nullptr, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            std::cerr << "computeWeights: clEnqueueNDRangeKernel failed (err=" << err << ")\n";
            clReleaseMemObject(posBuf); clReleaseMemObject(valBuf); clReleaseMemObject(lenBuf); clReleaseMemObject(outBuf);
            return false;
        }

        err = clEnqueueReadBuffer(command_queue, outBuf, CL_TRUE, 0, sizeof(unsigned short) * n, out.data(), 0, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            std::cerr << "computeWeights: clEnqueueReadBuffer failed (err=" << err << ")\n";
            clReleaseMemObject(posBuf); clReleaseMemObject(valBuf); clReleaseMemObject(lenBuf); clReleaseMemObject(outBuf);
            return false;
        }

        // copy back avec OpenMP
        #pragma omp parallel for if(n > 1024)
        for (size_t i = 0; i < n; ++i) tensors[i].Weight = static_cast<uint16_t>(out[i]);

        clReleaseMemObject(posBuf);
        clReleaseMemObject(valBuf);
        clReleaseMemObject(lenBuf);
        clReleaseMemObject(outBuf);

        return true;
    }

    // CPU fallback: même math optimisée avec OpenMP
    #pragma omp parallel for schedule(dynamic, 256) if(n > 512)
    for (size_t i = 0; i < n; ++i) {
        const tensor &t = tensors[i];
        
        // Composante spatiale
        const float r2 = t.Pos.X*t.Pos.X + t.Pos.Y*t.Pos.Y + t.Pos.Z*t.Pos.Z;
        const float r = std::sqrt(r2);
        const float spatial_nl = std::tanh(std::min(std::max(r * 0.5f, -3.0f), 3.0f));
        
        // Composante dynamique
        const float v01 = static_cast<float>(t.Value) * (1.0f / 255.0f);
        const float l01 = static_cast<float>(t.Length) * (1.0f / 1024.0f);
        const float vel_nl = std::tanh((v01 * 2.0f - 1.0f));
        const float den_nl = std::tanh((std::min(std::max(l01, 0.0f), 1.0f) * 2.0f - 1.0f));

        // Embedding contextuel
        const float w = t.Pos.W;
        const float w_high = w * 0.50f, w_mid = w * 0.10f, w_low = w * 0.02f;
        const float s1 = std::sin(w_high), c1 = std::cos(w_high);
        const float s2 = std::sin(w_mid),  c2 = std::cos(w_mid);
        const float s3 = std::sin(w_low),  c3 = std::cos(w_low);

        const float ctx_core = std::tanh(w * 0.05f);
        const float ctx_feat = 0.50f*s1 + 0.35f*c1 + 0.30f*s2 + 0.20f*c2 + 0.15f*s3 + 0.10f*c3;
        const float ctx_sum = std::min(std::max(ctx_core + 0.6f * ctx_feat, -1.0f), 1.0f);

        // Gating
        const float gate = 1.0f / (1.0f + std::exp(-2.2f * ctx_sum));
        const float w_spatial = (1.0f - gate) * 0.95f + 0.05f;
        const float w_dyn = gate * 0.90f;
        
        // Combinaison
        const float combined = w_spatial * spatial_nl + w_dyn * (vel_nl - 0.7f*den_nl);
        
        // Normalisation finale
        float s = 1.0f / (1.0f + std::exp(-2.0f * combined));
        s = 0.98f * s + 0.01f;

        int iw = static_cast<int>(std::floor(s * 65535.0f + 0.5f));
        iw = std::max(0, std::min(iw, 65535));
        tensors[i].Weight = static_cast<uint16_t>(iw);
    }
    return true;
}

// ------------------------ SafeTensors writer/reader using json-glib ------------------------

static void write_le_u32(std::ofstream &f, uint32_t v) {
    unsigned char b[4];
    b[0] = (unsigned char)(v & 0xFF);
    b[1] = (unsigned char)((v >> 8) & 0xFF);
    b[2] = (unsigned char)((v >> 16) & 0xFF);
    b[3] = (unsigned char)((v >> 24) & 0xFF);
    f.write(reinterpret_cast<const char*>(b), 4);
}

static bool read_le_u32(std::ifstream &f, uint32_t &out) {
    unsigned char b[4];
    if (!f.read(reinterpret_cast<char*>(b), 4)) return false;
    out = (uint32_t)b[0] | ((uint32_t)b[1] << 8) | ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
    return true;
}

bool save_safetensors(const std::string &fname, const std::vector<std::string>& names,
                      const std::vector<size_t>& paramsCount, const std::vector<tensor>& params)
{
    if (names.size() != paramsCount.size()) {
        std::cerr << "save_safetensors: names/paramsCount mismatch\n";
        return false;
    }

    size_t totalCount = 0;
    for (size_t c : paramsCount) totalCount += c;
    if (totalCount > params.size()) {
        std::cerr << "save_safetensors: not enough params provided\n";
        return false;
    }

    // Construction du JSON avec nlohmann/json
    json root;
    json tensors;
    
    size_t offset_bytes = 0;
    for (size_t i = 0; i < names.size(); ++i) {
        json tensor_info;
        tensor_info["dtype"] = "u16";
        tensor_info["shape"] = {paramsCount[i]};
        tensor_info["offset"] = offset_bytes;
        tensors[names[i]] = tensor_info;
        offset_bytes += paramsCount[i] * 2;
    }
    
    root["tensors"] = tensors;
    std::string json_str = root.dump();

    // Écriture du fichier
    std::ofstream fout(fname, std::ios::binary);
    if (!fout) {
        std::cerr << "save_safetensors: cannot open " << fname << " for writing\n";
        return false;
    }

    const char magic[] = "safetensors";
    fout.write(magic, 10);
    write_le_u32(fout, static_cast<uint32_t>(json_str.size()));
    fout.write(json_str.data(), static_cast<std::streamsize>(json_str.size()));

    // Section données
    size_t pidx = 0;
    for (size_t i = 0; i < paramsCount.size(); ++i) {
        for (size_t k = 0; k < paramsCount[i]; ++k) {
            uint16_t w = 0;
            if (pidx < params.size()) w = params[pidx].Weight;
            unsigned char b0 = static_cast<unsigned char>(w & 0xFF);
            unsigned char b1 = static_cast<unsigned char>((w >> 8) & 0xFF);
            fout.put(static_cast<char>(b0));
            fout.put(static_cast<char>(b1));
            ++pidx;
        }
    }

    return true;
}

bool load_safetensors(const std::string &fname, std::vector<std::string>& names,
                      std::vector<size_t>& paramsCount, std::vector<tensor>& params)
{
    names.clear();
    paramsCount.clear();
    params.clear();

    std::ifstream fin(fname, std::ios::binary);
    if (!fin) {
        std::cerr << "load_safetensors: cannot open " << fname << "\n";
        return false;
    }

    char magic[11] = {0};
    fin.read(magic, 10);
    if (fin.gcount() != 10 || std::string(magic, 10) != "safetensors") {
        std::cerr << "load_safetensors: invalid magic\n";
        return false;
    }

    uint32_t jlen = 0;
    if (!read_le_u32(fin, jlen) || jlen == 0) {
        std::cerr << "load_safetensors: invalid json length\n";
        return false;
    }

    std::string json_str;
    json_str.resize(jlen);
    fin.read(&json_str[0], jlen);
    if (static_cast<uint32_t>(fin.gcount()) != jlen) {
        std::cerr << "load_safetensors: failed to read json block\n";
        return false;
    }

    // Parse JSON avec nlohmann/json
    json root;
    try {
        root = json::parse(json_str);
    } catch (const json::exception& e) {
        std::cerr << "load_safetensors: json parse error: " << e.what() << "\n";
        return false;
    }

    if (!root.contains("tensors") || !root["tensors"].is_object()) {
        std::cerr << "load_safetensors: missing or invalid tensors object\n";
        return false;
    }

    // Extraction des informations tenseurs
    const auto& tensors = root["tensors"];
    for (const auto& [name, info] : tensors.items()) {
        if (!info.contains("shape") || !info["shape"].is_array() || 
            !info.contains("dtype") || info["dtype"] != "u16") {
            continue;
        }
        names.push_back(name);
        paramsCount.push_back(info["shape"][0]);
    }

    // Lecture section données
    fin.seekg(0, std::ios::end);
    std::streamoff file_end = fin.tellg();
    std::streamoff data_start = 10 + 4 + static_cast<std::streamoff>(jlen);
    if (data_start > file_end) {
        std::cerr << "load_safetensors: invalid data start\n";
        return false;
    }

    std::streamoff data_size = file_end - data_start;
    fin.seekg(data_start, std::ios::beg);

    std::vector<unsigned char> data;
    data.resize(static_cast<size_t>(data_size));
    fin.read(reinterpret_cast<char*>(data.data()), data_size);
    size_t got = static_cast<size_t>(fin.gcount());
    if (got != static_cast<size_t>(data_size)) {
        std::cerr << "load_safetensors: failed to read data section\n";
        return false;
    }

    // Reconstruction des params
    params.clear();
    size_t byte_idx = 0;
    for (size_t i = 0; i < paramsCount.size(); ++i) {
        for (size_t k = 0; k < paramsCount[i]; ++k) {
            uint16_t w = 0;
            if (byte_idx + 1 < data.size()) {
                w = static_cast<uint16_t>(
                    (unsigned char)data[byte_idx] | 
                    ((unsigned char)data[byte_idx + 1] << 8)
                );
            }
            // Créer tensor in-place pour éviter copie
            params.emplace_back();
            tensor& t = params.back();
            t.Weight = w;
            t.Value = 0;
            t.Length = 0;
            t.Pos = Vector4F{0.0f,0.0f,0.0f,0.0f};
            byte_idx += 2;
        }
    }

    return true;
}