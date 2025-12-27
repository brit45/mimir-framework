// Implémentation unique de stb_image pour éviter les définitions multiples

// ⚠️ CRITIQUE: Router les allocations stb_image via MemoryGuard
// stb_image utilise malloc/realloc/free par défaut = contournement total de la limite RAM!
#include "MemoryGuard.hpp"
#include <cstdlib>
#include <iostream>

static void* stbi_malloc_wrapper(size_t size) {
    auto& guard = MemoryGuard::instance();
    if (!guard.requestAllocation(size, "stb_image")) {
        std::cerr << "❌ stb_image: MemoryGuard refuse allocation de " << (size / 1024) << " KB" << std::endl;
        return nullptr;
    }
    void* ptr = std::malloc(size);
    if (!ptr) {
        guard.releaseAllocation(size);
    }
    return ptr;
}

static void* stbi_realloc_wrapper(void* ptr, size_t new_size) {
    // Note: realloc() gère déjà le free de l'ancien bloc si nécessaire
    // On doit juste vérifier la nouvelle taille avec MemoryGuard
    auto& guard = MemoryGuard::instance();
    if (!guard.requestAllocation(new_size, "stb_image_realloc")) {
        std::cerr << "❌ stb_image: MemoryGuard refuse realloc de " << (new_size / 1024) << " KB" << std::endl;
        return nullptr;
    }
    void* new_ptr = std::realloc(ptr, new_size);
    if (!new_ptr) {
        guard.releaseAllocation(new_size);
    }
    return new_ptr;
}

static void stbi_free_wrapper(void* ptr) {
    if (ptr) {
        // Note: on ne peut pas connaître la taille exacte ici
        // MemoryGuard devra gérer ça via tracking interne
        std::free(ptr);
    }
}

#define STBI_MALLOC(sz)           stbi_malloc_wrapper(sz)
#define STBI_REALLOC(p, newsz)    stbi_realloc_wrapper(p, newsz)
#define STBI_FREE(p)              stbi_free_wrapper(p)

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
