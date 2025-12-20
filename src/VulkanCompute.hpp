#ifndef __VULKAN_COMPUTE_HPP__
#define __VULKAN_COMPUTE_HPP__

#include <vulkan/vulkan.h>
#include <vector>
#include <string>
#include <cstring>
#include <iostream>
#include <fstream>

namespace VulkanCompute {

class ComputeEngine {
private:
    VkInstance instance;
    VkPhysicalDevice physicalDevice;
    VkDevice device;
    VkQueue computeQueue;
    uint32_t queueFamilyIndex;
    bool initialized;
    
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
        
        initialized = true;
        std::cout << "✓ Vulkan Compute initialized" << std::endl;
        return true;
    }
    
    void cleanup() {
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
