#include "../src/Model.hpp"
#include "../src/Layers.hpp"
#include <iostream>
#include <cassert>

int main() {
    std::cout << "🧪 Test de fusion Layer (ancien LayerDesc)\n" << std::endl;
    
    // Test 1: Créer un modèle et ajouter des layers
    Model model;
    
    std::cout << "Test 1: Ajout de layers..." << std::endl;
    model.push("conv1", "Conv2d", 9216);
    model.push("bn1", "BatchNorm2d", 64);
    model.push("relu1", "ReLU", 0);
    model.push("pool1", "MaxPool2d", 0);
    
    size_t total = model.totalParamCount();
    std::cout << "  ✓ Total paramètres: " << total << std::endl;
    assert(total == 9216 + 64);
    
    // Test 2: Vérifier l'allocation
    std::cout << "\nTest 2: Allocation des paramètres..." << std::endl;
    model.allocateParams();
    std::cout << "  ✓ Paramètres alloués" << std::endl;
    
    // Test 3: Initialisation des poids
    std::cout << "\nTest 3: Initialisation Xavier..." << std::endl;
    model.initializeWeights("xavier", 42);
    std::cout << "  ✓ Poids initialisés" << std::endl;
    
    // Test 4: Vérifier la structure Layer directement
    std::cout << "\nTest 4: Vérification de la structure Layer..." << std::endl;
    Layer test_layer("test", "Dense", 1024);
    assert(test_layer.name == "test");
    assert(test_layer.type == "Dense");
    assert(test_layer.params_count == 1024);
    std::cout << "  ✓ Structure Layer fonctionnelle" << std::endl;
    
    // Test 5: Configuration avancée de Layer
    std::cout << "\nTest 5: Configuration avancée de Layer..." << std::endl;
    test_layer.in_features = 256;
    test_layer.out_features = 512;
    test_layer.use_bias = true;
    test_layer.activation = ActivationType::RELU;
    std::cout << "  ✓ Configuration: " << test_layer.in_features << " -> " 
              << test_layer.out_features << std::endl;
    
    // Test 6: Weights et gradients
    std::cout << "\nTest 6: Allocation weights/gradients..." << std::endl;
    test_layer.weights.resize(test_layer.in_features * test_layer.out_features);
    test_layer.bias.resize(test_layer.out_features);
    test_layer.grad_weights.resize(test_layer.weights.size());
    test_layer.grad_bias.resize(test_layer.bias.size());
    std::cout << "  ✓ Weights: " << test_layer.weights.size() << std::endl;
    std::cout << "  ✓ Bias: " << test_layer.bias.size() << std::endl;
    std::cout << "  ✓ Gradients alloués" << std::endl;
    
    // Test 7: Sauvegarde et chargement de la structure
    std::cout << "\nTest 7: Sauvegarde/chargement..." << std::endl;
    model.push("test_layer", "Dense", 512);
    
    std::filesystem::path test_path = "/tmp/test_layers.json";
    bool saved = model.saveLayersStructure(test_path);
    assert(saved);
    std::cout << "  ✓ Structure sauvegardée" << std::endl;
    
    Model model2;
    bool loaded = model2.loadLayersStructure(test_path);
    assert(loaded);
    assert(model2.totalParamCount() == model.totalParamCount());
    std::cout << "  ✓ Structure chargée (params: " << model2.totalParamCount() << ")" << std::endl;
    
    std::filesystem::remove(test_path);
    
    std::cout << "\n✅ Tous les tests réussis!" << std::endl;
    std::cout << "   Layer remplace complètement LayerDesc" << std::endl;
    
    return 0;
}
