# Installation / Build

Mímir est un projet C++17 buildable avec CMake.

## Dépendances

- Obligatoire : CMake, compilateur C++17, OpenMP
- Optionnel : Vulkan (si build configuré), SFML (si visualisation), OpenCL (si activé)

## Build (recommandé)

```bash
cmake -S . -B build
cmake --build build -j
```

## Exécuter les tests

Si le projet a été configuré avec CTest :

```bash
ctest --test-dir build
```

## Problèmes fréquents

- Si OpenMP n’est pas détecté : vérifier la toolchain (gcc/clang) et les flags CMake.
- Si Vulkan n’est pas dispo : le runtime doit rester fonctionnel en CPU-only.
