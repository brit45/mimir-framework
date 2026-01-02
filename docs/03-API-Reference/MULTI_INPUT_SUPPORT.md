# Multi-Input / Branch Support Implementation

## 📋 Overview

This document describes the complete implementation of multi-input and branch support in Mímir Framework v2.1, enabling operations like residual connections, tensor concatenation, matrix multiplication, and split operations.

## 🎯 Key Features

### 1. **TensorStore System**
- Named tensor routing via `std::unordered_map<std::string, std::vector<float>>`
- Default tensor name: `"x"` (main pipeline)
- Explicit tensor storage and retrieval with error handling

### 2. **Layer I/O Configuration**
- `Layer.inputs`: Vector of input tensor names (empty = `{"x"}` by default)
- `Layer.output`: Output tensor name (default: `"x"`)
- Lua API: `model.set_layer_io(layer_name, inputs_table, output_name)`

### 3. **Multi-Input Operations**
Implemented operations requiring multiple inputs:
- **Add**: Element-wise addition of 2 tensors
- **Multiply**: Element-wise multiplication of 2 tensors
- **Concat**: Concatenate N tensors along an axis
- **MatMul**: Matrix multiplication of 2 matrices
- **Split**: Split 1 tensor into N outputs (stored as `name_0`, `name_1`, ...)

## 🏗️ Architecture

### TensorStore Flow
```
Input → TensorStore["x"]
  ↓
Layer 1: inputs=["x"] → output="hidden"
  ↓
TensorStore["hidden"] = Layer1_output
  ↓
Layer 2: inputs=["x", "hidden"] → output="x"
  ↓
TensorStore["x"] = Layer2_output (Add of original x + hidden)
  ↓
Return TensorStore["x"]
```

### Code Structure

#### 1. **Model.hpp** (lines 410-433)
```cpp
class Model {
    std::unordered_map<std::string, std::vector<float>> tensor_store;
    
    const std::vector<float>& getTensor(const std::string& name) const;
    std::vector<float>& getTensorMutable(const std::string& name);
    void storeTensor(const std::string& name, const std::vector<float>& data);
    void storeTensor(const std::string& name, std::vector<float>&& data);
    std::vector<std::string> getAvailableTensors() const;
    void clearTensorStore();
    Layer* getLayerByName(const std::string& name);
};
```

#### 2. **Layers.hpp** (lines 183-186)
```cpp
struct Layer {
    std::vector<std::string> inputs;   // Input tensor names (empty = {"x"})
    std::string output = "x";          // Output tensor name
    std::vector<int> split_sizes;      // Split sizes (for Split operation)
    int split_axis = 0;                // Split axis
    // ... other fields
};
```

#### 3. **Model.cpp** - TensorStore Methods (lines 220-283)
```cpp
const std::vector<float>& Model::getTensor(const std::string& name) const {
    auto it = tensor_store.find(name);
    if (it == tensor_store.end()) {
        std::cerr << "❌ ERROR: Tensor '" << name << "' not found\n";
        std::cerr << "Available tensors: ";
        for (const auto& kv : tensor_store) {
            std::cerr << "'" << kv.first << "' ";
        }
        throw std::runtime_error("Tensor not found: " + name);
    }
    return it->second;
}
// + storeTensor(), getAvailableTensors(), clearTensorStore(), getLayerByName()
```

#### 4. **Model.cpp** - Forward Pass Integration (lines 1847-1950)
```cpp
std::vector<float> Model::forwardPass(...) {
    // Initialize TensorStore
    clearTensorStore();
    storeTensor("x", input);
    
    // Main loop
    for (size_t layer_idx = 0; layer_idx < layers.size(); ++layer_idx) {
        const Layer& layer = layers[layer_idx];
        
        // Retrieve inputs (multi-input support)
        std::vector<std::string> input_names = 
            layer.inputs.empty() ? std::vector<std::string>{"x"} : layer.inputs;
        
        std::vector<const std::vector<float>*> inputs;
        for (const auto& name : input_names) {
            inputs.push_back(&getTensor(name));
        }
        
        const std::vector<float>& x = *inputs[0];  // Backward compatibility
        
        // ... layer operations using inputs[] ...
        
        // Store output
        std::string output_name = layer.output.empty() ? "x" : layer.output;
        storeTensor(output_name, std::move(layer_output));
    }
    
    return getTensor("x");
}
```

#### 5. **LayerOps.hpp** - Split Implementation (lines 688-755)
```cpp
// Surcharge 1: Split égaux
inline std::vector<std::vector<float>> split_forward(
    const std::vector<float>& input,
    int num_splits,
    int axis = 0
) { /* ... */ }

// Surcharge 2: Split avec tailles explicites
inline std::vector<std::vector<float>> split_forward(
    const std::vector<float>& input,
    const std::vector<int>& split_sizes,
    int axis = 0
) {
    // Validate split_sizes sum == input.size()
    // Split input into N tensors according to split_sizes
}
```

## 📚 Usage Examples

### Example 1: Residual Connection (Skip Connection)
```lua
Mimir.Model.create("resnet_block")

-- Conv1: x → skip
model.push_layer("conv1", "Conv2d", 64*64*3*3)
model.set_layer_io("conv1", {"x"}, "skip")

-- Conv2: skip → out
model.push_layer("conv2", "Conv2d", 64*64*3*3)
model.set_layer_io("conv2", {"skip"}, "out")

-- Add: x + out → x
model.push_layer("add_residual", "Add", 0)
model.set_layer_io("add_residual", {"x", "out"}, "x")

model.allocate_params()
model.init_weights("he")
```

### Example 2: Concatenation (Feature Fusion)
```lua
Mimir.Model.create("feature_fusion")

-- Branch 1: x → feat_a
model.push_layer("conv_a", "Conv2d", 32*3*3*3)
model.set_layer_io("conv_a", {"x"}, "feat_a")

-- Branch 2: x → feat_b
model.push_layer("conv_b", "Conv2d", 32*3*3*3)
model.set_layer_io("conv_b", {"x"}, "feat_b")

-- Concat: [feat_a, feat_b] → x
model.push_layer("concat", "Concat", 0)
model.set_layer_io("concat", {"feat_a", "feat_b"}, "x")
```

### Example 3: Matrix Multiplication
```lua
Mimir.Model.create("matrix_ops")

-- Create matrix A (input → A)
model.push_layer("create_A", "Identity", 0)
model.set_layer_io("create_A", {"x"}, "A")

-- Create matrix B (input → B)
model.push_layer("create_B", "Identity", 0)
model.set_layer_io("create_B", {"x"}, "B")

-- MatMul: A × B → result
model.push_layer("matmul_AB", "MatMul", 0)
model.set_layer_io("matmul_AB", {"A", "B"}, "result")
-- Note: Layer must have in_features=M, out_features=K, embed_dim=N
```

### Example 4: Split Operation
```lua
Mimir.Model.create("split_demo")

-- Split: x → [x_0, x_1, x_2]
model.push_layer("split_layer", "Split", 0)
model.set_layer_io("split_layer", {"x"}, "branch")
-- This creates: branch_0, branch_1, branch_2 in TensorStore

-- Use split outputs
model.push_layer("conv_branch0", "Conv2d", 32*3*3*3)
model.set_layer_io("conv_branch0", {"branch_0"}, "out_0")

model.push_layer("conv_branch1", "Conv2d", 32*3*3*3)
model.set_layer_io("conv_branch1", {"branch_1"}, "out_1")
```

## 🔧 Implementation Details

### Error Handling
All operations include explicit error messages:
```cpp
if (inputs.size() < 2) {
    std::cerr << "⚠️  Add requires 2 inputs, got " << inputs.size() << "\n";
    std::cerr << "Available tensors: ";
    for (const auto& t : getAvailableTensors()) {
        std::cerr << "'" << t << "' ";
    }
    // Fallback: pass-through
    layer_output = x;
}
```

### Memory Safety
- All tensor operations use `DynamicTensorAllocator` and `MemoryGuard`
- TensorStore cleared at start of each forward pass
- Move semantics used when storing large tensors

### Backward Compatibility
- Empty `inputs` vector defaults to `{"x"}`
- Empty `output` string defaults to `"x"`
- Existing sequential models work unchanged

## 📊 Operation Summary

| Operation | Inputs | Output | Status | Notes |
|-----------|--------|--------|--------|-------|
| Add | 2 tensors | 1 tensor | ✅ Complete | Element-wise addition |
| Multiply | 2 tensors | 1 tensor | ✅ Complete | Element-wise multiplication |
| Concat | N tensors | 1 tensor | ✅ Complete | Concatenation along axis |
| MatMul | 2 matrices | 1 matrix | ✅ Complete | Requires M, K, N config |
| Split | 1 tensor | N tensors | ✅ Complete | Outputs: `name_0`, `name_1`, ... |

## 🧪 Testing

### Test Files
- `scripts/tests/test_api_simple.lua`: Basic API smoke test ✅
- `scripts/tests/test_branches.lua`: Comprehensive multi-input tests

### Running Tests
```bash
./bin/mimir --lua scripts/tests/test_api_simple.lua
./bin/mimir --lua scripts/tests/test_branches.lua
```

## 🔮 Future Work

### 1. Backward Pass Support
Gradient routing for multi-input operations:
- **Add**: Distribute gradient to both inputs
- **Concat**: Split gradient along concat axis
- **MatMul**: Compute A.T @ grad and grad @ B.T
- **Split**: Sum gradients from all outputs

### 2. Dynamic Graph Execution
- Topological sort for efficient execution order
- Cycle detection for invalid graphs
- Parallel execution of independent branches

### 3. Lua API Extensions
```lua
-- Query available tensors
local tensors = model.get_available_tensors()

-- Get specific tensor
local tensor = model.get_tensor("tensor_name")

-- Advanced configuration
model.set_layer_config("matmul_layer", {
    M = 128,
    K = 256,
    N = 512
})
```

## 📖 API Reference

### C++ API

#### TensorStore Methods
```cpp
// Retrieve tensor (throws if not found)
const std::vector<float>& getTensor(const std::string& name) const;

// Store tensor (copy)
void storeTensor(const std::string& name, const std::vector<float>& data);

// Store tensor (move)
void storeTensor(const std::string& name, std::vector<float>&& data);

// List available tensors
std::vector<std::string> getAvailableTensors() const;

// Clear all tensors
void clearTensorStore();

// Find layer by name
Layer* getLayerByName(const std::string& name);
```

### Lua API

#### Model Configuration
```lua
-- Configure layer inputs/outputs
model.set_layer_io(layer_name, inputs_table, output_name)

-- Example:
model.set_layer_io("add_layer", {"tensor1", "tensor2"}, "result")
```

## 📝 Notes

- **Default Behavior**: All layers without explicit `inputs`/`output` configuration use `"x"` as input and output
- **Tensor Names**: Use descriptive names for clarity (e.g., `"features"`, `"skip"`, `"query"`, `"key"`)
- **Split Outputs**: Automatically suffixed with `_0`, `_1`, etc.
- **Error Messages**: Always list available tensors when tensor not found
- **Performance**: TensorStore uses move semantics to minimize copies

## 🎓 Design Principles

1. **Simplicity**: No full DAG complexity, simple named routing
2. **Robustness**: Explicit errors, no silent fallbacks (except pass-through for safety)
3. **Maintainability**: Clear code structure, well-documented
4. **Memory Safety**: MemoryGuard integration, proper cleanup
5. **Backward Compatibility**: Existing code works unchanged

---

**Implementation Status**: ✅ **Complete** (2024)  
**Version**: Mímir Framework v2.1.0  
**Language**: C++17 + Lua 5.3
