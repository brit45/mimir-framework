# Tests Unitaires Mímir

Ce dossier contient les tests unitaires du framework Mímir utilisant **Google Test (gtest)** et CTest.

## Framework de test

Les tests utilisent **Google Test v1.14.0**, téléchargé automatiquement par CMake via `FetchContent`. Aucune installation manuelle n'est nécessaire.

## Structure des tests

- **test_tensors.cpp** - Tests pour tensors.hpp/cpp
  * Construction de tensors
  * Accès et modification des données
  * Vector4F
  * Copy semantics
  * TensorSystem (OpenCL)

- **test_tokenizer.cpp** - Tests pour Tokenizer.hpp/cpp
  * Construction et initialisation
  * Tokens spéciaux (<pad>, <unk>, <s>, </s>)
  * Ajout et tokenization
  * Décodage
  * Vocabulaire depuis texte
  * Sérialisation JSON
  * Padding et batch tokenization
  * Extraction de keywords

- **test_layers.cpp** - Tests pour Layers.hpp
  * Construction de layers
  * Activations (ReLU, ReLU6, Leaky ReLU, GELU, Sigmoid, Tanh, Swish, Softmax)
  * Weights et bias
  * Gradients
  * BatchNorm state
  * Configuration de layers

- **test_autograd.cpp** - Tests pour Autograd.hpp
  * Structure Gradients (add, get, zero)
  * Gradient clipping
  * MSE backward
  * ComputationGraph
  * Layer activations
  * Gradient accumulation

- **test_model.cpp** - Tests pour Model.hpp
  * LayerDesc
  * Optimizer (SGD, Adam, AdamW)
  * LR Decay strategies (None, Cosine, Linear, Exponential, Step)
  * Warmup
  * Weight decay (AdamW)

- **test_hardware.cpp** - Tests pour HardwareOpt.hpp
  * Détection des capacités CPU (AVX2, FMA, F16C, BMI2)
  * Détection du nombre de threads OpenMP
  * Configuration hardware

## Compilation

```bash
mkdir build && cd build
cmake -DBUILD_TESTS=ON ..
make -j$(nproc)
```

## Exécution

### Tous les tests avec CTest
```bash
cd build
ctest --output-on-failure
```

### Tests individuels (exécutables Google Test)
```bash
./test_tensors
./test_tokenizer
./test_layers
./test_autograd
./test_model
./test_hardware
```

### Tests avec verbosité Google Test
```bash
./test_tensors --gtest_verbose
./test_tokenizer --gtest_filter=TokenizerTest.*
```

### Tests avec CTest verbeux
```bash
ctest -V
```

### Tests par catégorie
```bash
ctest -R Tensor    # Tous les tests contenant "Tensor"
ctest -R Layer     # Tous les tests contenant "Layer"
ctest -R Optimizer # Tests d'optimisation
```

### Exécuter un test spécifique avec Google Test
```bash
./test_layers --gtest_filter=ActivationTest.ReLU
./test_tokenizer --gtest_filter="TokenizerTest.Add*"
```

## Résultats attendus

Chaque test utilise les macros Google Test :
- ✅ `[  PASSED  ]` si le test réussit
- ❌ `[  FAILED  ]` si le test échoue avec détails de l'assertion

Exemple de sortie Google Test :
```
[==========] Running 6 tests from 1 test suite.
[----------] Global test environment set-up.
[----------] 6 tests from TensorTest
[ RUN      ] TensorTest.Construction
[       OK ] TensorTest.Construction (0 ms)
[ RUN      ] TensorTest.DataAccess
[       OK ] TensorTest.DataAccess (0 ms)
[ RUN      ] TensorTest.DataModification
[       OK ] TensorTest.DataModification (0 ms)
[ RUN      ] TensorTest.Vector4F
[       OK ] TensorTest.Vector4F (0 ms)
[ RUN      ] TensorTest.CopySemantics
[       OK ] TensorTest.CopySemantics (0 ms)
[ RUN      Google Test et le header à tester :
```cpp
#include "../src/<module>.hpp"
#include <gtest/gtest.h>
```

3. Écrire des tests avec les macros Google Test :
```cpp
TEST(TestSuiteName, TestName) {
    // Setup
    MyClass obj;
    
    // Action
    obj.doSomething();
    
    // Assertions
    EXPECT_EQ(obj.getValue(), expected_value);
    ASSERT_NE(ptr, nullptr);  // Arrête le test si fail
}
```

4. Ajouter le test dans CMakeLists.txt :
```cmake
add_mimir_test(test_<m Google Test

```cpp
#include "../src/MyModule.hpp"
#include <gtest/gtest.h>

// Test fixture (optionnel, pour partager setup/teardown)
class MyModuleTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup avant chaque test
        obj = new MyClass();
    }
    
    void TearDown() override {
        // Cleanup après chaque test
        delete obj;
    }
    
    MyClass* obj;
};

// Test simple
TEST(MyModuleTest, BasicFunctionality) {
    MyClass obj;
    obj.doSomething();
    EXPECT_EQ(obj.getValue(), 42);
}

// Test avec fixture
TEST_F(MyModuleTest, UsingFixture) {
    // obj est disponible grâce au fixture
    obj->doSomething();
    EXPECT_TRUE(obj->isValid());
}

// Test paramétré (optionnel)
class ParameterizedTest : public ::testing::TestWithParam<int> {};

TEST_P(ParameterizedTest, WorksWithDifferentValues) {
    int param = GetParam();
    EXPECT_GE(param, 0);
}

INSTANTIATE_TEST_SUITE_P(Values, ParameterizedTest, 
                        ::testing::Values(1, 2, 3, 5, 8));
```

## Dépendances

Les tests utilisent :
- **Google Test v1.14.0** (téléchargé automatiquement)
- Standard Library C++17
- Headers Mímir (`src/*.hpp`)
- Bibliothèque `mimir_core` (linkée automatiquement)

Aucune installation manuelle de Google Test nécessaire

1. Créer un fichier `test_<module>.cpp` dans `tests/`
2. Inclure le header à tester : `#include "../src/<module>.hpp"`
3. Écrire des fonctions de test avec assertions
4. Ajouter le test dans CMakeLists.txt :
```cmake
if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_<module>.cpp")
    add_executable(test_<module> tests/test_<module>.cpp)
    target_link_libraries(test_<module> PRIVATE mimir_core)
    add_test(NAME <Module>Test COMMAND test_<module>)
endif()
```

## Structure d'un test

```cpp
#include "../src/MyModule.hpp"
#include <cassert>
#include <iostream>

void test_feature() {
    std::cout << "Test: Feature description... ";
    
    // Setup
    MyClass obj;
    
    // Action
    obj.doSomething();
    
    // Assertions
    assert(obj.getValue() == expected_value);
    
    std::cout << "PASSED" << std::endl;
}

int main() {
    std::cout << "=== Running MyModule Tests ===" << std::endl;
    test_feature();
    std::cout << "=== All Tests Passed ===" << std::endl;
    return 0;
}
```

## Dépendances

Les tests utilisent uniquement :
- Standard Library C++17
- Headers Mímir (`src/*.hpp`)
- Bibliothèque `mimir_core` (linkée automatiquement)

Pas de dépendances externes (Catch2, GoogleTest, etc.) pour simplicité.

## Intégration Continue

Les tests peuvent être intégrés dans un pipeline CI/CD :

```yaml
# .github/workflows/tests.yml
test:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v2
    - name: Build
      run: |
        mkdir build && cd build
        cmake -DBUILD_TESTS=ON ..
        make -j$(nproc)
    - name: Run tests
      run: cd build && ctest --output-on-failure
```

## Notes

- Les tests OpenCL (TensorSystem) passent même si OpenCL n'est pas disponible
- Les tests de hardware détectent automatiquement les capacités CPU
- Tous les tests utilisent `float_equals()` pour comparer les floats avec tolérance
- Les tests sont indépendants et peuvent être exécutés dans n'importe quel ordre
