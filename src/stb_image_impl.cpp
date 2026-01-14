// Implémentation unique de stb_image pour éviter les définitions multiples

// ⚠️ CRITIQUE: Router les allocations stb_image via MemoryGuard
// stb_image utilise malloc/realloc/free par défaut = contournement total de la limite RAM!
#include "MemoryGuard.hpp"
#include <cstdlib>
#include <iostream>

namespace {
// Stocke la taille (en bytes) juste avant le pointeur retourné à stb_image.
// Avantages: pas de mutex, pas de unordered_map, pas de warnings use-after-realloc.
static constexpr size_t STBI_HDR_SIZE = sizeof(size_t);

static inline void* stbi_base_ptr(void* user_ptr) {
    return user_ptr ? static_cast<void*>(static_cast<unsigned char*>(user_ptr) - STBI_HDR_SIZE) : nullptr;
}

static inline size_t stbi_user_size(void* user_ptr) {
    if (!user_ptr) return 0;
    void* base = stbi_base_ptr(user_ptr);
    return *static_cast<size_t*>(base);
}

static inline void stbi_set_user_size(void* user_ptr, size_t size) {
    void* base = stbi_base_ptr(user_ptr);
    *static_cast<size_t*>(base) = size;
}
} // namespace

static void* stbi_malloc_wrapper(size_t size) {
    auto& guard = MemoryGuard::instance();
    if (!guard.requestAllocation(size + STBI_HDR_SIZE, "stb_image")) {
        std::cerr << "❌ stb_image: MemoryGuard refuse allocation de " << (size / 1024) << " KB" << std::endl;
        return nullptr;
    }
    void* base = std::malloc(size + STBI_HDR_SIZE);
    if (!base) {
        guard.releaseAllocation(size + STBI_HDR_SIZE);
        return nullptr;
    }

    *static_cast<size_t*>(base) = size;
    return static_cast<void*>(static_cast<unsigned char*>(base) + STBI_HDR_SIZE);
}

static void* stbi_realloc_wrapper(void* ptr, size_t new_size) {
    auto& guard = MemoryGuard::instance();

    // realloc(NULL, n) == malloc(n)
    if (!ptr) {
        return stbi_malloc_wrapper(new_size);
    }

    const size_t old_size = stbi_user_size(ptr);

    // Ne demander au MemoryGuard que le delta (sinon on sur-compte et on "fuit" la limite).
    if (new_size > old_size) {
        const size_t delta = new_size - old_size;
        if (!guard.requestAllocation(delta, "stb_image_realloc")) {
            std::cerr << "❌ stb_image: MemoryGuard refuse realloc de " << (new_size / 1024) << " KB" << std::endl;
            return nullptr;
        }

        void* old_base = stbi_base_ptr(ptr);
        void* new_base = std::realloc(old_base, new_size + STBI_HDR_SIZE);
        if (!new_base) {
            guard.releaseAllocation(delta);
            return nullptr;
        }

        *static_cast<size_t*>(new_base) = new_size;
        return static_cast<void*>(static_cast<unsigned char*>(new_base) + STBI_HDR_SIZE);
    }

    // shrink ou taille identique: pas besoin de "request". On release le delta après succès.
    void* old_base = stbi_base_ptr(ptr);
    void* new_base = std::realloc(old_base, new_size + STBI_HDR_SIZE);
    if (!new_base) {
        // si realloc échoue, l'ancien bloc reste valide
        return nullptr;
    }

    if (old_size > new_size) {
        guard.releaseAllocation(old_size - new_size);
    }

    *static_cast<size_t*>(new_base) = new_size;
    return static_cast<void*>(static_cast<unsigned char*>(new_base) + STBI_HDR_SIZE);
}

static void stbi_free_wrapper(void* ptr) {
    if (!ptr) return;

    const size_t sz = stbi_user_size(ptr);
    MemoryGuard::instance().releaseAllocation(sz + STBI_HDR_SIZE);

    void* base = stbi_base_ptr(ptr);
    std::free(base);
}

#define STBI_MALLOC(sz)           stbi_malloc_wrapper(sz)
#define STBI_REALLOC(p, newsz)    stbi_realloc_wrapper(p, newsz)
#define STBI_FREE(p)              stbi_free_wrapper(p)

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
