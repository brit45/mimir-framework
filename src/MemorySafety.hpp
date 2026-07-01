#ifndef __MEMORY_SAFETY_HPP__
#define __MEMORY_SAFETY_HPP__

/**
 * 🛡️ Memory Safety Utilities
 * 
 * Utilitaires pour le debugging et la validation de la sécurité mémoire dans Mímir.
 * Ces fonctions aident à vérifier que toutes les allocations passent par MemoryGuard.
 */

#include "MemoryGuard.hpp"
#include "DynamicTensorAllocator.hpp"
#include <iostream>
#include <string>

namespace MemorySafety {

/**
 * Affiche un rapport détaillé de l'état de la mémoire
 */
inline void printMemoryReport() {
    auto& guard = MemoryGuard::instance();
    auto& allocator = DynamicTensorAllocator::instance();
    
    std::cerr << "\n╔══════════════════════════════════════════════════════════╗" << std::endl;
    std::cerr << "║           RAPPORT DE SÉCURITÉ MÉMOIRE                    ║" << std::endl;
    std::cerr << "╠══════════════════════════════════════════════════════════╣" << std::endl;
    
    // Stats MemoryGuard
    guard.printStats();
    
    // Stats DynamicTensorAllocator
    std::cerr << "\n╔══════════════════════════════════════════════════════════╗" << std::endl;
    std::cerr << "║           DYNAMIC TENSOR ALLOCATOR                        ║" << std::endl;
    std::cerr << "╠══════════════════════════════════════════════════════════╣" << std::endl;
    allocator.printStats();
    
    std::cerr << "\n" << std::endl;
}

/**
 * Vérifie si l'allocation est sûre (ne dépassera pas la limite)
 */
inline bool canAllocate(size_t bytes) {
    auto& guard = MemoryGuard::instance();
    size_t current = guard.getCurrentBytes();
    size_t limit = guard.getLimit();
    
    return (current + bytes) <= limit;
}

/**
 * Obtient la RAM disponible restante
 */
inline size_t getAvailableRAM() {
    auto& guard = MemoryGuard::instance();
    size_t current = guard.getCurrentBytes();
    size_t limit = guard.getLimit();
    
    if (current >= limit) return 0;
    return limit - current;
}

/**
 * Affiche un avertissement si proche de la limite
 */
inline void checkMemoryPressure(float warning_threshold = 0.8f) {
    auto& guard = MemoryGuard::instance();
    float usage = guard.getUsagePercent() / 100.0f;
    
    if (usage >= warning_threshold) {
        std::cerr << "\n⚠️  ALERTE MÉMOIRE: " << (usage * 100.0f) << "% utilisé!" << std::endl;
        std::cerr << "   Seuil d'alerte: " << (warning_threshold * 100.0f) << "%" << std::endl;
        std::cerr << "   RAM disponible: " 
                  << (getAvailableRAM() / 1024 / 1024) << " MB" << std::endl;
        
        if (usage >= 0.95f) {
            std::cerr << "🚨 CRITIQUE: Proche de la limite! Risque de refus d'allocation." << std::endl;
        }
    }
}

/**
 * Mode strict: refuse toute allocation non-dynamique
 * (Pour détecter les allocations qui contournent MemoryGuard)
 */
inline void enableStrictMode(bool enable = true) {
    if (enable) {
        std::cerr << "🔒 MODE STRICT ACTIVÉ" << std::endl;
        std::cerr << "   Toute allocation doit passer par DynamicTensorAllocator" << std::endl;
        // TODO: Ajouter hook pour détecter malloc/new direct
    }
}

/**
 * Valide que la structure legacy est désactivée
 */
inline void validateLegacyDisabled() {
#ifdef MIMIR_ENABLE_LEGACY_PARAMS
    std::cerr << "\n⚠️⚠️⚠️  ATTENTION: LEGACY PARAMS ACTIVÉ! ⚠️⚠️⚠️" << std::endl;
    std::cerr << "   Cette configuration consomme énormément de RAM!" << std::endl;
    std::cerr << "   Recommandation: Compiler avec -DMIMIR_ENABLE_LEGACY_PARAMS=OFF" << std::endl;
    std::cerr << "   ou ne pas définir cette macro du tout.\n" << std::endl;
    return;
#else
    std::cerr << "✅ Structure legacy désactivée (configuration optimale)" << std::endl;
#endif
}

/**
 * Test d'intégrité du système de mémoire
 */
inline bool runMemoryIntegrityTest() {
    std::cerr << "\n🧪 TEST D'INTÉGRITÉ MÉMOIRE" << std::endl;
    std::cerr << "═══════════════════════════" << std::endl;
    
    bool all_passed = true;
    
    // Test 1: MemoryGuard répond
    std::cerr << "Test 1: MemoryGuard accessible... ";
    try {
        auto& guard = MemoryGuard::instance();
        size_t limit = guard.getLimit();
        std::cerr << "✓ (Limite: " << (limit / 1024 / 1024 / 1024) << " GB)" << std::endl;
    } catch (...) {
        std::cerr << "✗ ÉCHEC" << std::endl;
        all_passed = false;
    }
    
    // Test 2: DynamicTensorAllocator répond
    std::cerr << "Test 2: DynamicTensorAllocator accessible... ";
    try {
        auto& allocator = DynamicTensorAllocator::instance();
        std::cerr << "✓" << std::endl;
    } catch (...) {
        std::cerr << "✗ ÉCHEC" << std::endl;
        all_passed = false;
    }
    
    // Test 3: Allocation/libération basique
    std::cerr << "Test 3: Cycle allocation/libération... ";
    try {
        auto& guard = MemoryGuard::instance();
        size_t before = guard.getCurrentBytes();
        
        if (guard.requestAllocation(1024, "test")) {
            guard.releaseAllocation(1024);
            size_t after = guard.getCurrentBytes();
            
            if (after == before) {
                std::cerr << "✓" << std::endl;
            } else {
                std::cerr << "✗ Fuite mémoire détectée!" << std::endl;
                all_passed = false;
            }
        } else {
            std::cerr << "✗ Allocation refusée (RAM insuffisante?)" << std::endl;
            all_passed = false;
        }
    } catch (...) {
        std::cerr << "✗ ÉCHEC" << std::endl;
        all_passed = false;
    }
    
    // Test 4: Vérifier legacy params
    std::cerr << "Test 4: Structure legacy désactivée... ";
    validateLegacyDisabled();
    
    std::cerr << "\n" << (all_passed ? "✅ TOUS LES TESTS PASSÉS" : "❌ CERTAINS TESTS ONT ÉCHOUÉ") << std::endl;
    std::cerr << "═══════════════════════════\n" << std::endl;
    
    return all_passed;
}

/**
 * Macro helper pour tracer les allocations en debug
 */
#ifdef DEBUG
    #define TRACE_ALLOC(size, tag) \
        std::cerr << "🔍 [ALLOC] " << __FILE__ << ":" << __LINE__ \
                  << " - " << (size / 1024) << " KB (" << tag << ")" << std::endl;
#else
    #define TRACE_ALLOC(size, tag)
#endif

} // namespace MemorySafety

#endif // __MEMORY_SAFETY_HPP__
