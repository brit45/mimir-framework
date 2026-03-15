#ifndef __VULKAN_COMPUTE_HPP__
#define __VULKAN_COMPUTE_HPP__

#include <vulkan/vulkan.h>
#include <vector>
#include <string>
#include <cstring>
#include <iostream>
#include <fstream>
#include <optional>
#include <filesystem>
#include <mutex>

namespace VulkanCompute {

namespace fs = std::filesystem;

static inline std::vector<uint32_t> read_spirv_u32(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    f.seekg(0, std::ios::end);
    const std::streamsize size = f.tellg();
    f.seekg(0, std::ios::beg);
    if (size <= 0 || (size % 4) != 0) return {};
    std::vector<uint32_t> out(static_cast<size_t>(size / 4));
    if (!f.read(reinterpret_cast<char*>(out.data()), size)) return {};
    return out;
}

static inline std::optional<std::string> find_shader_path_linear() {
    if (const char* env = std::getenv("MIMIR_VULKAN_LINEAR_SPV")) {
        if (env[0] != '\0' && fs::exists(env)) return std::string(env);
    }

    // Candidats relatifs au cwd (souvent la racine du repo)
    const char* candidates[] = {
        "./bin/shaders/linear_forward.comp.spv",
        "./shaders/linear_forward.comp.spv",
        "./build/shaders/linear_forward.comp.spv",
        "./build_static/shaders/linear_forward.comp.spv",
        "./build_sfml/shaders/linear_forward.comp.spv",
        "./shaders/linear_forward.comp.spv",
        "./bin/shaders/linear_forward.comp.spv",
    };
    for (const char* c : candidates) {
        if (fs::exists(c)) return std::string(c);
    }
    return std::nullopt;
}

static inline uint32_t findMemoryType(VkPhysicalDevice phys, uint32_t typeFilter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(phys, &memProperties);
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1u << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    return UINT32_MAX;
}

class ComputeEngine {
private:
    VkInstance instance;
    VkPhysicalDevice physicalDevice;
    VkDevice device;
    VkQueue computeQueue;
    uint32_t queueFamilyIndex;
    bool initialized;

    // Resources for Linear kernel
    bool linear_ready_ = false;
    VkDescriptorSetLayout linear_dsl_ = VK_NULL_HANDLE;
    VkPipelineLayout linear_pl_ = VK_NULL_HANDLE;
    VkPipeline linear_pipe_ = VK_NULL_HANDLE;
    VkDescriptorPool linear_dp_ = VK_NULL_HANDLE;
    VkCommandPool cmd_pool_ = VK_NULL_HANDLE;

    std::recursive_mutex linear_mutex_;
    
public:
    ComputeEngine() : instance(VK_NULL_HANDLE), physicalDevice(VK_NULL_HANDLE),
                      device(VK_NULL_HANDLE), computeQueue(VK_NULL_HANDLE),
                      queueFamilyIndex(0), initialized(false) {}
    
    ~ComputeEngine() {
        cleanup();
    }
    
    bool initialize() {
        // 1. Create Vulkan instance
        VkApplicationInfo appInfo = {};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "Mimir Compute";
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.pEngineName = "MimirEngine";
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion = VK_API_VERSION_1_2;
        
        VkInstanceCreateInfo createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &appInfo;
        
        if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
            std::cerr << "Failed to create Vulkan instance" << std::endl;
            return false;
        }
        
        // 2. Select physical device with compute capability
        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
        
        if (deviceCount == 0) {
            std::cerr << "No Vulkan devices found" << std::endl;
            return false;
        }
        
        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());
        
        // Pick first device with compute queue
        for (const auto& dev : devices) {
            uint32_t queueFamilyCount = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(dev, &queueFamilyCount, nullptr);
            
            std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
            vkGetPhysicalDeviceQueueFamilyProperties(dev, &queueFamilyCount, queueFamilies.data());
            
            for (uint32_t i = 0; i < queueFamilyCount; ++i) {
                if (queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                    physicalDevice = dev;
                    queueFamilyIndex = i;
                    break;
                }
            }
            
            if (physicalDevice != VK_NULL_HANDLE) break;
        }
        
        if (physicalDevice == VK_NULL_HANDLE) {
            std::cerr << "No compute-capable device found" << std::endl;
            return false;
        }
        
        // 3. Create logical device
        float queuePriority = 1.0f;
        VkDeviceQueueCreateInfo queueCreateInfo = {};
        queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueCreateInfo.queueFamilyIndex = queueFamilyIndex;
        queueCreateInfo.queueCount = 1;
        queueCreateInfo.pQueuePriorities = &queuePriority;
        
        VkDeviceCreateInfo deviceCreateInfo = {};
        deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        deviceCreateInfo.queueCreateInfoCount = 1;
        deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;
        
        if (vkCreateDevice(physicalDevice, &deviceCreateInfo, nullptr, &device) != VK_SUCCESS) {
            std::cerr << "Failed to create logical device" << std::endl;
            return false;
        }
        
        // 4. Get compute queue
        vkGetDeviceQueue(device, queueFamilyIndex, 0, &computeQueue);

        // Create command pool
        VkCommandPoolCreateInfo poolInfo = {};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.queueFamilyIndex = queueFamilyIndex;
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        if (vkCreateCommandPool(device, &poolInfo, nullptr, &cmd_pool_) != VK_SUCCESS) {
            std::cerr << "Failed to create Vulkan command pool" << std::endl;
            return false;
        }
        
        initialized = true;
        return true;
    }
    
    void cleanup() {
        cleanupLinearKernel();
        if (cmd_pool_ != VK_NULL_HANDLE) {
            vkDestroyCommandPool(device, cmd_pool_, nullptr);
            cmd_pool_ = VK_NULL_HANDLE;
        }
        if (device != VK_NULL_HANDLE) {
            vkDestroyDevice(device, nullptr);
            device = VK_NULL_HANDLE;
        }
        if (instance != VK_NULL_HANDLE) {
            vkDestroyInstance(instance, nullptr);
            instance = VK_NULL_HANDLE;
        }
        initialized = false;
    }
    
    bool isInitialized() const { return initialized; }
    VkDevice getDevice() const { return device; }
    VkPhysicalDevice getPhysicalDevice() const { return physicalDevice; }
    VkQueue getComputeQueue() const { return computeQueue; }
    uint32_t getQueueFamilyIndex() const { return queueFamilyIndex; }

    struct LinearDims {
        uint32_t batch;
        uint32_t in_f;
        uint32_t out_f;
    };

    bool linearForward(const float* input, const float* weights, const float* bias_or_null,
                       float* output, int batch, int in_f, int out_f) {
        std::lock_guard<std::recursive_mutex> lk(linear_mutex_);
        if (!initialized) return false;
        if (!input || !weights || !output) return false;
        if (batch <= 0 || in_f <= 0 || out_f <= 0) return false;
        if (!ensureLinearKernel()) return false;

        const size_t bytes_in = static_cast<size_t>(batch) * static_cast<size_t>(in_f) * sizeof(float);
        const size_t bytes_w  = static_cast<size_t>(out_f) * static_cast<size_t>(in_f) * sizeof(float);
        const size_t bytes_b  = static_cast<size_t>(out_f) * sizeof(float);
        const size_t bytes_o  = static_cast<size_t>(batch) * static_cast<size_t>(out_f) * sizeof(float);

        VkBuffer buf_in = VK_NULL_HANDLE, buf_w = VK_NULL_HANDLE, buf_b = VK_NULL_HANDLE, buf_o = VK_NULL_HANDLE;
        VkDeviceMemory mem_in = VK_NULL_HANDLE, mem_w = VK_NULL_HANDLE, mem_b = VK_NULL_HANDLE, mem_o = VK_NULL_HANDLE;

        auto make_buffer = [&](size_t size, VkBufferUsageFlags usage, VkBuffer& buf, VkDeviceMemory& mem) -> bool {
            VkBufferCreateInfo bi = {};
            bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            bi.size = size;
            bi.usage = usage;
            bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            if (vkCreateBuffer(device, &bi, nullptr, &buf) != VK_SUCCESS) return false;

            VkMemoryRequirements req;
            vkGetBufferMemoryRequirements(device, buf, &req);
            const uint32_t mt = findMemoryType(physicalDevice, req.memoryTypeBits,
                                               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            if (mt == UINT32_MAX) return false;

            VkMemoryAllocateInfo ai = {};
            ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            ai.allocationSize = req.size;
            ai.memoryTypeIndex = mt;
            if (vkAllocateMemory(device, &ai, nullptr, &mem) != VK_SUCCESS) return false;
            if (vkBindBufferMemory(device, buf, mem, 0) != VK_SUCCESS) return false;
            return true;
        };

        auto destroy_buf = [&](VkBuffer& b, VkDeviceMemory& m) {
            if (b != VK_NULL_HANDLE) { vkDestroyBuffer(device, b, nullptr); b = VK_NULL_HANDLE; }
            if (m != VK_NULL_HANDLE) { vkFreeMemory(device, m, nullptr); m = VK_NULL_HANDLE; }
        };

        if (!make_buffer(bytes_in, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, buf_in, mem_in)) { destroy_buf(buf_in, mem_in); return false; }
        if (!make_buffer(bytes_w,  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, buf_w,  mem_w))  { destroy_buf(buf_in, mem_in); destroy_buf(buf_w, mem_w); return false; }
        if (!make_buffer(bytes_b,  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, buf_b,  mem_b))  { destroy_buf(buf_in, mem_in); destroy_buf(buf_w, mem_w); destroy_buf(buf_b, mem_b); return false; }
        if (!make_buffer(bytes_o,  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, buf_o,  mem_o))  { destroy_buf(buf_in, mem_in); destroy_buf(buf_w, mem_w); destroy_buf(buf_b, mem_b); destroy_buf(buf_o, mem_o); return false; }

        auto upload = [&](VkDeviceMemory mem, const void* src, size_t sz) -> bool {
            void* mapped = nullptr;
            if (vkMapMemory(device, mem, 0, sz, 0, &mapped) != VK_SUCCESS) return false;
            std::memcpy(mapped, src, sz);
            vkUnmapMemory(device, mem);
            return true;
        };

        if (!upload(mem_in, input, bytes_in)) { destroy_buf(buf_in, mem_in); destroy_buf(buf_w, mem_w); destroy_buf(buf_b, mem_b); destroy_buf(buf_o, mem_o); return false; }
        if (!upload(mem_w,  weights, bytes_w)) { destroy_buf(buf_in, mem_in); destroy_buf(buf_w, mem_w); destroy_buf(buf_b, mem_b); destroy_buf(buf_o, mem_o); return false; }
        if (bias_or_null) {
            if (!upload(mem_b, bias_or_null, bytes_b)) { destroy_buf(buf_in, mem_in); destroy_buf(buf_w, mem_w); destroy_buf(buf_b, mem_b); destroy_buf(buf_o, mem_o); return false; }
        } else {
            std::vector<float> z(static_cast<size_t>(out_f), 0.0f);
            if (!upload(mem_b, z.data(), bytes_b)) { destroy_buf(buf_in, mem_in); destroy_buf(buf_w, mem_w); destroy_buf(buf_b, mem_b); destroy_buf(buf_o, mem_o); return false; }
        }

        // Descriptor set
        VkDescriptorSet ds = VK_NULL_HANDLE;
        if (!allocAndWriteLinearDescriptorSet(
            buf_in, bytes_in,
            buf_w,  bytes_w,
            buf_b,  bytes_b,
            buf_o,  bytes_o,
            ds)) {
            destroy_buf(buf_in, mem_in); destroy_buf(buf_w, mem_w); destroy_buf(buf_b, mem_b); destroy_buf(buf_o, mem_o);
            return false;
        }

        // Command buffer
        VkCommandBufferAllocateInfo cbai = {};
        cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cbai.commandPool = cmd_pool_;
        cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbai.commandBufferCount = 1;
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        if (vkAllocateCommandBuffers(device, &cbai, &cmd) != VK_SUCCESS) {
            destroy_buf(buf_in, mem_in); destroy_buf(buf_w, mem_w); destroy_buf(buf_b, mem_b); destroy_buf(buf_o, mem_o);
            return false;
        }

        VkCommandBufferBeginInfo bi = {};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (vkBeginCommandBuffer(cmd, &bi) != VK_SUCCESS) {
            vkFreeCommandBuffers(device, cmd_pool_, 1, &cmd);
            destroy_buf(buf_in, mem_in); destroy_buf(buf_w, mem_w); destroy_buf(buf_b, mem_b); destroy_buf(buf_o, mem_o);
            return false;
        }
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, linear_pipe_);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, linear_pl_, 0, 1, &ds, 0, nullptr);

        LinearDims dims{ static_cast<uint32_t>(batch), static_cast<uint32_t>(in_f), static_cast<uint32_t>(out_f) };
        vkCmdPushConstants(cmd, linear_pl_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(LinearDims), &dims);

        const uint32_t gx = (static_cast<uint32_t>(batch) + 15u) / 16u;
        const uint32_t gy = (static_cast<uint32_t>(out_f) + 15u) / 16u;
        vkCmdDispatch(cmd, gx, gy, 1);
        if (vkEndCommandBuffer(cmd) != VK_SUCCESS) {
            vkFreeCommandBuffers(device, cmd_pool_, 1, &cmd);
            destroy_buf(buf_in, mem_in); destroy_buf(buf_w, mem_w); destroy_buf(buf_b, mem_b); destroy_buf(buf_o, mem_o);
            return false;
        }

        VkFence fence = VK_NULL_HANDLE;
        VkFenceCreateInfo fci = {};
        fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        if (vkCreateFence(device, &fci, nullptr, &fence) != VK_SUCCESS) {
            vkFreeCommandBuffers(device, cmd_pool_, 1, &cmd);
            destroy_buf(buf_in, mem_in); destroy_buf(buf_w, mem_w); destroy_buf(buf_b, mem_b); destroy_buf(buf_o, mem_o);
            return false;
        }

        VkSubmitInfo si = {};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd;
        if (vkQueueSubmit(computeQueue, 1, &si, fence) != VK_SUCCESS) {
            vkDestroyFence(device, fence, nullptr);
            vkFreeCommandBuffers(device, cmd_pool_, 1, &cmd);
            destroy_buf(buf_in, mem_in); destroy_buf(buf_w, mem_w); destroy_buf(buf_b, mem_b); destroy_buf(buf_o, mem_o);
            return false;
        }
        vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);

        vkDestroyFence(device, fence, nullptr);
        vkFreeCommandBuffers(device, cmd_pool_, 1, &cmd);

        // Download output
        void* mapped = nullptr;
        if (vkMapMemory(device, mem_o, 0, bytes_o, 0, &mapped) != VK_SUCCESS) {
            destroy_buf(buf_in, mem_in); destroy_buf(buf_w, mem_w); destroy_buf(buf_b, mem_b); destroy_buf(buf_o, mem_o);
            return false;
        }
        std::memcpy(output, mapped, bytes_o);
        vkUnmapMemory(device, mem_o);

        destroy_buf(buf_in, mem_in);
        destroy_buf(buf_w,  mem_w);
        destroy_buf(buf_b,  mem_b);
        destroy_buf(buf_o,  mem_o);
        return true;
    }

private:
    void cleanupLinearKernel() {
        if (linear_dp_ != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(device, linear_dp_, nullptr);
            linear_dp_ = VK_NULL_HANDLE;
        }
        if (linear_pipe_ != VK_NULL_HANDLE) {
            vkDestroyPipeline(device, linear_pipe_, nullptr);
            linear_pipe_ = VK_NULL_HANDLE;
        }
        if (linear_pl_ != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(device, linear_pl_, nullptr);
            linear_pl_ = VK_NULL_HANDLE;
        }
        if (linear_dsl_ != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(device, linear_dsl_, nullptr);
            linear_dsl_ = VK_NULL_HANDLE;
        }
        linear_ready_ = false;
    }

    bool ensureLinearKernel() {
        std::lock_guard<std::recursive_mutex> lk(linear_mutex_);
        if (linear_ready_) return true;

        const auto shader_path = find_shader_path_linear();
        if (!shader_path.has_value()) {
            if (const char* v = std::getenv("MIMIR_ACCEL_VERBOSE")) {
                if (v[0] != '\0' && !(v[0] == '0' && v[1] == '\0')) {
                    std::cerr << "Vulkan linear shader not found. Set MIMIR_VULKAN_LINEAR_SPV or build shaders via glslangValidator.\n";
                }
            }
            return false;
        }

        std::vector<uint32_t> spirv = read_spirv_u32(*shader_path);
        if (spirv.empty()) {
            return false;
        }

        VkShaderModule shader = VK_NULL_HANDLE;
        VkShaderModuleCreateInfo smci = {};
        smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        smci.codeSize = spirv.size() * sizeof(uint32_t);
        smci.pCode = spirv.data();
        if (vkCreateShaderModule(device, &smci, nullptr, &shader) != VK_SUCCESS) {
            return false;
        }

        // Descriptor set layout (4 storage buffers)
        VkDescriptorSetLayoutBinding bindings[4] = {};
        for (uint32_t i = 0; i < 4; ++i) {
            bindings[i].binding = i;
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo dslci = {};
        dslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dslci.bindingCount = 4;
        dslci.pBindings = bindings;
        if (vkCreateDescriptorSetLayout(device, &dslci, nullptr, &linear_dsl_) != VK_SUCCESS) {
            vkDestroyShaderModule(device, shader, nullptr);
            return false;
        }

        VkPushConstantRange pcr = {};
        pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcr.offset = 0;
        pcr.size = sizeof(LinearDims);

        VkPipelineLayoutCreateInfo plci = {};
        plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount = 1;
        plci.pSetLayouts = &linear_dsl_;
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges = &pcr;
        if (vkCreatePipelineLayout(device, &plci, nullptr, &linear_pl_) != VK_SUCCESS) {
            vkDestroyShaderModule(device, shader, nullptr);
            return false;
        }

        VkPipelineShaderStageCreateInfo stage = {};
        stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = shader;
        stage.pName = "main";

        VkComputePipelineCreateInfo cpci = {};
        cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cpci.stage = stage;
        cpci.layout = linear_pl_;
        if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpci, nullptr, &linear_pipe_) != VK_SUCCESS) {
            vkDestroyShaderModule(device, shader, nullptr);
            return false;
        }
        vkDestroyShaderModule(device, shader, nullptr);

        // Descriptor pool
        VkDescriptorPoolSize ps = {};
        ps.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        ps.descriptorCount = 4;
        VkDescriptorPoolCreateInfo dpci = {};
        dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpci.maxSets = 1;
        dpci.poolSizeCount = 1;
        dpci.pPoolSizes = &ps;
        if (vkCreateDescriptorPool(device, &dpci, nullptr, &linear_dp_) != VK_SUCCESS) {
            return false;
        }

        linear_ready_ = true;
        return true;
    }

    bool allocAndWriteLinearDescriptorSet(
        VkBuffer inB, size_t inBytes,
        VkBuffer wB,  size_t wBytes,
        VkBuffer bB,  size_t bBytes,
        VkBuffer oB,  size_t oBytes,
        VkDescriptorSet& outDS
    ) {
        if (!linear_dp_ || !linear_dsl_) return false;

        // Important: on réutilise le même descriptor pool à chaque appel.
        // Sans reset/free, on épuise le pool après 1 allocation (maxSets=1) et
        // on retombe en fallback CPU après avoir déjà payé le coût des buffers.
        // Ici c'est safe car on attend la fence (pas d'utilisation en vol).
        vkResetDescriptorPool(device, linear_dp_, 0);

        VkDescriptorSetAllocateInfo ai = {};
        ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorPool = linear_dp_;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &linear_dsl_;
        if (vkAllocateDescriptorSets(device, &ai, &outDS) != VK_SUCCESS) {
            return false;
        }

        VkDescriptorBufferInfo infos[4] = {};
        infos[0].buffer = inB; infos[0].offset = 0; infos[0].range = static_cast<VkDeviceSize>(inBytes);
        infos[1].buffer = wB;  infos[1].offset = 0; infos[1].range = static_cast<VkDeviceSize>(wBytes);
        infos[2].buffer = bB;  infos[2].offset = 0; infos[2].range = static_cast<VkDeviceSize>(bBytes);
        infos[3].buffer = oB;  infos[3].offset = 0; infos[3].range = static_cast<VkDeviceSize>(oBytes);

        VkWriteDescriptorSet writes[4] = {};
        for (uint32_t i = 0; i < 4; ++i) {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = outDS;
            writes[i].dstBinding = i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &infos[i];
        }
        vkUpdateDescriptorSets(device, 4, writes, 0, nullptr);
        return true;
    }
};

// Buffer pour calculs GPU
class ComputeBuffer {
private:
    VkDevice device;
    VkBuffer buffer;
    VkDeviceMemory memory;
    size_t size;
    
public:
    ComputeBuffer(VkDevice dev) : device(dev), buffer(VK_NULL_HANDLE),
                                  memory(VK_NULL_HANDLE), size(0) {}
    
    ~ComputeBuffer() {
        cleanup();
    }
    
    bool allocate(VkPhysicalDevice physicalDevice, size_t byteSize, VkBufferUsageFlags usage) {
        size = byteSize;
        
        // Create buffer
        VkBufferCreateInfo bufferInfo = {};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = size;
        bufferInfo.usage = usage;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        
        if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
            return false;
        }
        
        // Get memory requirements
        VkMemoryRequirements memRequirements;
        vkGetBufferMemoryRequirements(device, buffer, &memRequirements);
        
        // Find suitable memory type
        VkPhysicalDeviceMemoryProperties memProperties;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);
        
        uint32_t memoryTypeIndex = UINT32_MAX;
        VkMemoryPropertyFlags properties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        
        for (uint32_t i = 0; i < memProperties.memoryTypeCount; ++i) {
            if ((memRequirements.memoryTypeBits & (1 << i)) &&
                (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                memoryTypeIndex = i;
                break;
            }
        }
        
        if (memoryTypeIndex == UINT32_MAX) {
            return false;
        }
        
        // Allocate memory
        VkMemoryAllocateInfo allocInfo = {};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = memoryTypeIndex;
        
        if (vkAllocateMemory(device, &allocInfo, nullptr, &memory) != VK_SUCCESS) {
            return false;
        }
        
        // Bind buffer to memory
        vkBindBufferMemory(device, buffer, memory, 0);
        
        return true;
    }
    
    void cleanup() {
        if (buffer != VK_NULL_HANDLE) {
            vkDestroyBuffer(device, buffer, nullptr);
            buffer = VK_NULL_HANDLE;
        }
        if (memory != VK_NULL_HANDLE) {
            vkFreeMemory(device, memory, nullptr);
            memory = VK_NULL_HANDLE;
        }
    }
    
    bool upload(const void* data, size_t offset = 0) {
        void* mapped;
        if (vkMapMemory(device, memory, offset, size, 0, &mapped) != VK_SUCCESS) {
            return false;
        }
        std::memcpy(mapped, data, size);
        vkUnmapMemory(device, memory);
        return true;
    }
    
    bool download(void* data, size_t offset = 0) {
        void* mapped;
        if (vkMapMemory(device, memory, offset, size, 0, &mapped) != VK_SUCCESS) {
            return false;
        }
        std::memcpy(data, mapped, size);
        vkUnmapMemory(device, memory);
        return true;
    }
    
    VkBuffer getBuffer() const { return buffer; }
    size_t getSize() const { return size; }
};

// Shader compute SPIR-V
class ComputeShader {
private:
    VkDevice device;
    VkShaderModule shaderModule;
    VkPipeline pipeline;
    VkPipelineLayout pipelineLayout;
    VkDescriptorSetLayout descriptorSetLayout;
    VkDescriptorPool descriptorPool;
    VkDescriptorSet descriptorSet;
    
public:
    ComputeShader(VkDevice dev) : device(dev), shaderModule(VK_NULL_HANDLE),
                                  pipeline(VK_NULL_HANDLE), pipelineLayout(VK_NULL_HANDLE),
                                  descriptorSetLayout(VK_NULL_HANDLE),
                                  descriptorPool(VK_NULL_HANDLE),
                                  descriptorSet(VK_NULL_HANDLE) {}
    
    ~ComputeShader() {
        cleanup();
    }
    
    bool loadFromSPIRV(const std::vector<uint32_t>& spirv) {
        VkShaderModuleCreateInfo createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        createInfo.codeSize = spirv.size() * sizeof(uint32_t);
        createInfo.pCode = spirv.data();
        
        return vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) == VK_SUCCESS;
    }
    
    void cleanup() {
        if (descriptorPool != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(device, descriptorPool, nullptr);
        }
        if (descriptorSetLayout != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
        }
        if (pipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(device, pipeline, nullptr);
        }
        if (pipelineLayout != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        }
        if (shaderModule != VK_NULL_HANDLE) {
            vkDestroyShaderModule(device, shaderModule, nullptr);
        }
    }
};

} // namespace VulkanCompute

#endif // __VULKAN_COMPUTE_HPP__
