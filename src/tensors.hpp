#ifndef __TENSORS_LIB_HPP__
#define __TENSORS_LIB_HPP__

#include <iostream>
#include <vector>
#include <cstdint>
#include <memory>

// Forward declaration
class DynamicTensorAllocator;

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
    
    // Support allocation dynamique (optionnel)
    void* dynamic_handle = nullptr;  // Pointeur vers TensorHandle
    bool use_dynamic_alloc = false;
    
    // Constructeur par défaut
    tensor() = default;
    
    // Constructeur avec taille (allocation classique)
    explicit tensor(size_t size) : data(size, 0.0f) {}
    
    // Constructeur avec données
    explicit tensor(const std::vector<float>& values) : data(values) {}
    
    // Constructeur avec allocation dynamique
    explicit tensor(size_t size, bool dynamic);
    
    // Copy constructor (désactivé pour éviter double-free)
    tensor(const tensor& other) = delete;
    
    // Move constructor
    tensor(tensor&& other) noexcept;
    
    // Copy assignment (désactivé pour éviter double-free)
    tensor& operator=(const tensor& other) = delete;
    
    // Move assignment
    tensor& operator=(tensor&& other) noexcept;
    
    // Destructeur
    ~tensor();
    
    // Accès aux données (transparent)
    float* getData();
    const float* getData() const;
    size_t getSize() const;
};

class TensorSystem {
public:
    TensorSystem();
    ~TensorSystem();

    // Prepare the public facade; computation is selected by RuntimeRouter.
    bool initialize();

    // Commit weights only after a runtime completes the whole calculation.
    bool computeWeights(std::vector<tensor>& tensors);

private:
    bool initialized = false;
};


#endif // __TENSORS_LIB_HPP__
