# Mímir Framework - Build Instructions

## Configuration rapide

```bash
# Créer un dossier de build
mkdir build && cd build

# Configuration avec CMake
cmake ..

# Compilation
make -j$(nproc)

# Installation (optionnel)
sudo make install
```

## Options de configuration

### Options principales

```bash
# Build en mode Release (optimisé)
cmake -DCMAKE_BUILD_TYPE=Release ..

# Build en mode Debug
cmake -DCMAKE_BUILD_TYPE=Debug ..

# Désactiver les optimisations SIMD
cmake -DENABLE_SIMD=OFF ..

# Désactiver OpenMP
cmake -DENABLE_OPENMP=OFF ..

# Désactiver Vulkan Compute
cmake -DENABLE_VULKAN=OFF ..

# Désactiver la compression LZ4
cmake -DENABLE_LZ4=OFF ..

# Ne pas compiler les tests
cmake -DBUILD_TESTS=OFF ..

# Ne pas compiler les exemples
cmake -DBUILD_EXAMPLES=OFF ..
```

### Configuration complète

```bash
cmake -DCMAKE_BUILD_TYPE=Release \
      -DENABLE_SIMD=ON \
      -DENABLE_OPENMP=ON \
      -DENABLE_VULKAN=ON \
      -DENABLE_LZ4=ON \
      -DBUILD_TESTS=ON \
      -DBUILD_EXAMPLES=ON \
      -DCMAKE_INSTALL_PREFIX=/usr/local \
      ..
```

## Dépendances requises

### Ubuntu/Debian

```bash
sudo apt update
sudo apt install -y \
    cmake \
    g++ \
    liblua5.3-dev \
    liblz4-dev \
    libvulkan-dev \
    vulkan-tools
```

### Fedora/RHEL

```bash
sudo dnf install -y \
    cmake \
    gcc-c++ \
    lua-devel \
    lz4-devel \
    vulkan-headers \
    vulkan-loader-devel
```

### Arch Linux

```bash
sudo pacman -S \
    cmake \
    gcc \
    lua \
    lz4 \
    vulkan-headers \
    vulkan-icd-loader
```

### macOS

```bash
brew install cmake lua lz4 vulkan-headers vulkan-loader
```

## Compilation

### Build standard

```bash
mkdir build
cd build
cmake ..
make -j$(nproc)
```

### Build avec Ninja (plus rapide)

```bash
mkdir build
cd build
cmake -G Ninja ..
ninja
```

### Build avec optimisations maximales

```bash
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_FLAGS="-O3 -march=native -flto" \
      ..
make -j$(nproc)
```

## Tests

```bash
cd build
ctest --output-on-failure
```

Ou exécuter individuellement :

```bash
./test_base_models
./test_vae
./benchmark_simd
```

## Installation

```bash
cd build
sudo make install
```

Installation dans un préfixe personnalisé :

```bash
cmake -DCMAKE_INSTALL_PREFIX=$HOME/.local ..
make install
```

## Utilisation

Après compilation, les binaires sont dans `bin/` :

```bash
# Exécuter Mímir avec un script Lua
./bin/mimir --lua scripts/train_llm.lua

# Démonstration des architectures
./bin/model_architectures_demo
```

## Nettoyage

```bash
# Nettoyer le build
cd build
make clean

# Supprimer complètement le dossier build
cd ..
rm -rf build
```

## Troubleshooting

### Lua non trouvée

```bash
# Spécifier le chemin manuellement
cmake -DLUA_INCLUDE_DIR=/usr/include/lua5.3 \
      -DLUA_LIBRARY=/usr/lib/x86_64-linux-gnu/liblua5.3.so \
      ..
```

### Vulkan non trouvé

```bash
# Installer les headers et loader Vulkan
sudo apt install libvulkan-dev vulkan-tools

# Ou désactiver Vulkan
cmake -DENABLE_VULKAN=OFF ..
```

### Erreur de compilation SIMD

```bash
# Vérifier le support CPU
cat /proc/cpuinfo | grep -E "avx2|fma|f16c|bmi2"

# Désactiver SIMD si le CPU ne supporte pas
cmake -DENABLE_SIMD=OFF ..
```

### OpenMP non trouvé

```bash
# Installer libomp
sudo apt install libomp-dev

# Ou désactiver OpenMP
cmake -DENABLE_OPENMP=OFF ..
```

## Configuration IDE

### Visual Studio Code

Installer l'extension CMake Tools, puis :

1. Ouvrir la commande `CMake: Configure`
2. Sélectionner le kit de compilation
3. Utiliser `CMake: Build` pour compiler

### CLion

CLion détecte automatiquement CMakeLists.txt. Configurer le profile de build dans Settings → Build, Execution, Deployment → CMake.

## Compilation croisée

### Pour ARM64

```bash
cmake -DCMAKE_TOOLCHAIN_FILE=toolchain-arm64.cmake ..
```

### Pour Windows (MinGW)

```bash
cmake -DCMAKE_TOOLCHAIN_FILE=/usr/share/mingw/toolchain-mingw64.cmake ..
```
