// ============================================================================
// Test du Système de Blocage d'Allocation Mémoire
// ============================================================================

#include "MemoryGuard.hpp"
#include "AdvancedRAMManager.hpp"
#include "DynamicTensorAllocator.hpp"
#include <iostream>
#include <vector>
#include <thread>
#include <chrono>

void printSeparator(char c = '=') {
    std::cout << std::string(80, c) << std::endl;
}

void test_memory_guard_blocking() {
    printSeparator('=');
    std::cout << "TEST 1: MemoryGuard - Blocage Complet" << std::endl;
    printSeparator('=');
    
    auto& guard = MemoryGuard::instance();
    guard.reset();
    guard.setLimit(1ULL * 1024 * 1024 * 1024); // 1 GB
    
    std::cout << "\n📊 État initial:" << std::endl;
    guard.printStats();
    
    // Test allocation normale
    std::cout << "\n✅ Test 1a: Allocation normale (100 MB)" << std::endl;
    bool success = guard.requestAllocation(100 * 1024 * 1024, "test_normal");
    std::cout << "Résultat: " << (success ? "✓ Réussie" : "✗ Échouée") << std::endl;
    
    // Bloquer les allocations
    std::cout << "\n🔒 Activation du blocage..." << std::endl;
    guard.blockAllocations(true);
    
    // Tenter une allocation (devrait échouer)
    std::cout << "\n❌ Test 1b: Allocation avec blocage (50 MB)" << std::endl;
    success = guard.requestAllocation(50 * 1024 * 1024, "test_blocked");
    std::cout << "Résultat: " << (success ? "✗ Réussie (BUG!)" : "✓ Bloquée") << std::endl;
    
    // Débloquer
    std::cout << "\n🔓 Désactivation du blocage..." << std::endl;
    guard.blockAllocations(false);
    
    // Tenter à nouveau
    std::cout << "\n✅ Test 1c: Allocation après déblocage (50 MB)" << std::endl;
    success = guard.requestAllocation(50 * 1024 * 1024, "test_unblocked");
    std::cout << "Résultat: " << (success ? "✓ Réussie" : "✗ Échouée") << std::endl;
    
    std::cout << "\n📊 État final:" << std::endl;
    guard.printStats();
}

void test_freeze_mode() {
    printSeparator('=');
    std::cout << "TEST 2: MemoryGuard - Mode Freeze" << std::endl;
    printSeparator('=');
    
    auto& guard = MemoryGuard::instance();
    guard.reset();
    guard.setLimit(1ULL * 1024 * 1024 * 1024);
    
    // Allouer de la mémoire
    std::cout << "\n✅ Allocation initiale (200 MB)" << std::endl;
    guard.requestAllocation(200 * 1024 * 1024, "initial");
    
    // Activer freeze
    std::cout << "\n❄️  Activation du mode freeze..." << std::endl;
    guard.freezeAllocations(true);
    
    // Tenter allocation (devrait échouer)
    std::cout << "\n❌ Tentative d'allocation en mode freeze (100 MB)" << std::endl;
    bool success = guard.requestAllocation(100 * 1024 * 1024, "frozen");
    std::cout << "Résultat: " << (success ? "✗ Réussie (BUG!)" : "✓ Gelée") << std::endl;
    
    // Libération (devrait fonctionner)
    std::cout << "\n✅ Libération en mode freeze (50 MB)" << std::endl;
    guard.releaseAllocation(50 * 1024 * 1024);
    std::cout << "✓ Libération réussie" << std::endl;
    
    // Désactiver freeze
    std::cout << "\n☀️  Désactivation du mode freeze..." << std::endl;
    guard.freezeAllocations(false);
    
    // Nouvelle allocation
    std::cout << "\n✅ Allocation après freeze (100 MB)" << std::endl;
    success = guard.requestAllocation(100 * 1024 * 1024, "post_freeze");
    std::cout << "Résultat: " << (success ? "✓ Réussie" : "✗ Échouée") << std::endl;
    
    guard.printStats();
}

void test_temporary_blocking() {
    printSeparator('=');
    std::cout << "TEST 3: Blocage Temporaire avec Timeout" << std::endl;
    printSeparator('=');
    
    auto& guard = MemoryGuard::instance();
    guard.reset();
    guard.setLimit(1ULL * 1024 * 1024 * 1024);
    
    std::cout << "\n⏰ Blocage pour 2 secondes..." << std::endl;
    guard.blockTemporary(2000); // 2 secondes
    
    // Vérifier que c'est bloqué
    std::cout << "\n⏱️  Tentative immédiate (devrait être bloquée)" << std::endl;
    bool blocked = guard.isBlocked();
    std::cout << "État: " << (blocked ? "🔒 Bloqué" : "🔓 Débloqué") << std::endl;
    
    bool success = guard.requestAllocation(10 * 1024 * 1024, "immediate");
    std::cout << "Allocation: " << (success ? "✗ Réussie" : "✓ Bloquée") << std::endl;
    
    // Attendre 2.5 secondes
    std::cout << "\n⏳ Attente de 2.5 secondes..." << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(2500));
    
    // Devrait être débloqué maintenant
    std::cout << "\n⏱️  Tentative après timeout (devrait réussir)" << std::endl;
    blocked = guard.isBlocked();
    std::cout << "État: " << (blocked ? "🔒 Bloqué" : "🔓 Débloqué") << std::endl;
    
    success = guard.requestAllocation(10 * 1024 * 1024, "after_timeout");
    std::cout << "Allocation: " << (success ? "✓ Réussie" : "✗ Bloquée") << std::endl;
    
    guard.printStats();
}

void test_advanced_ram_manager() {
    printSeparator('=');
    std::cout << "TEST 4: AdvancedRAMManager - Blocage" << std::endl;
    printSeparator('=');
    
    auto& ram = AdvancedRAMManager::instance();
    
    AdvancedRAMManager::Config config;
    config.max_ram_bytes = 500 * 1024 * 1024; // 500 MB
    config.enable_compression = true;
    config.enable_async_loading = false;
    ram.configure(config);
    
    // Test allocation normale
    std::cout << "\n✅ Allocation normale" << std::endl;
    std::vector<uint8_t> data(1024 * 1024, 0x42); // 1 MB
    bool success = ram.allocate("test1", data, true);
    std::cout << "Résultat: " << (success ? "✓ Réussie" : "✗ Échouée") << std::endl;
    
    // Bloquer
    std::cout << "\n🔒 Activation du blocage..." << std::endl;
    ram.blockAllocations(true);
    
    // Tenter allocation
    std::cout << "\n❌ Tentative d'allocation bloquée" << std::endl;
    success = ram.allocate("test2", data, true);
    std::cout << "Résultat: " << (success ? "✗ Réussie (BUG!)" : "✓ Bloquée") << std::endl;
    
    // Débloquer
    std::cout << "\n🔓 Désactivation du blocage..." << std::endl;
    ram.blockAllocations(false);
    
    // Nouvelle allocation
    std::cout << "\n✅ Allocation après déblocage" << std::endl;
    success = ram.allocate("test3", data, true);
    std::cout << "Résultat: " << (success ? "✓ Réussie" : "✗ Échouée") << std::endl;
    
    ram.printStats();
}

void test_integration() {
    printSeparator('=');
    std::cout << "TEST 5: Intégration Complète" << std::endl;
    printSeparator('=');
    
    auto& guard = MemoryGuard::instance();
    auto& ram = AdvancedRAMManager::instance();
    auto& allocator = DynamicTensorAllocator::instance();
    
    std::cout << "\n🔧 Configuration des systèmes..." << std::endl;
    guard.reset();
    allocator.configure(1, true); // 1 GB
    
    std::cout << "\n✅ Allocation de tenseurs" << std::endl;
    auto* handle1 = allocator.allocateTensor(1000000, "tensor1"); // ~4 MB
    std::cout << "Tenseur 1: " << (handle1 ? "✓ Alloué" : "✗ Échec") << std::endl;
    
    auto* handle2 = allocator.allocateTensor(2000000, "tensor2"); // ~8 MB
    std::cout << "Tenseur 2: " << (handle2 ? "✓ Alloué" : "✗ Échec") << std::endl;
    
    // Bloquer au niveau MemoryGuard
    std::cout << "\n🔒 Blocage au niveau MemoryGuard..." << std::endl;
    guard.blockAllocations(true);
    
    // Tenter d'allouer un tenseur
    std::cout << "\n❌ Tentative d'allocation de tenseur (devrait échouer)" << std::endl;
    auto* handle3 = allocator.allocateTensor(3000000, "tensor3_blocked");
    std::cout << "Tenseur 3: " << (handle3 ? "✗ Alloué (BUG!)" : "✓ Bloqué") << std::endl;
    
    // Débloquer
    std::cout << "\n🔓 Déblocage..." << std::endl;
    guard.blockAllocations(false);
    
    // Nouvelle tentative
    std::cout << "\n✅ Allocation après déblocage" << std::endl;
    handle3 = allocator.allocateTensor(3000000, "tensor3_unblocked");
    std::cout << "Tenseur 3: " << (handle3 ? "✓ Alloué" : "✗ Échec") << std::endl;
    
    std::cout << "\n📊 Statistiques finales:" << std::endl;
    guard.printStats();
}

int main() {
    std::cout << R"(
╔════════════════════════════════════════════════════════════════════════════╗
║                 TEST DU SYSTÈME DE BLOCAGE MÉMOIRE                         ║
║                           Mímir Framework                                  ║
╚════════════════════════════════════════════════════════════════════════════╝
)" << std::endl;

    try {
        // Test 1: Blocage de base
        test_memory_guard_blocking();
        
        std::cout << "\n\n";
        
        // Test 2: Mode freeze
        test_freeze_mode();
        
        std::cout << "\n\n";
        
        // Test 3: Blocage temporaire
        test_temporary_blocking();
        
        std::cout << "\n\n";
        
        // Test 4: AdvancedRAMManager
        test_advanced_ram_manager();
        
        std::cout << "\n\n";
        
        // Test 5: Intégration
        test_integration();
        
        printSeparator('=');
        std::cout << "\n✨ TOUS LES TESTS RÉUSSIS!\n" << std::endl;
        printSeparator('=');
        
    } catch (const std::exception& e) {
        std::cerr << "\n❌ ERREUR: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
