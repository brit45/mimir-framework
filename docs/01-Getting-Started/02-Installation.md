# Installation - Mímir

Guide d'installation complet pour toutes les plateformes.

---

## 📋 Prérequis

### Système d'Exploitation

- **Linux** : Ubuntu 20.04+, Debian 11+, Fedora 35+, Arch Linux
- **macOS** : 10.15 Catalina ou supérieur
- **Windows** : WSL2 avec Ubuntu 20.04+ (recommandé)

### Compilateur

- **GCC** : 9.0+ (recommandé: 11.0+)
- **Clang** : 10.0+
- **Support** : C++17 minimum

### Matériel

**CPU** :
- Support AVX2 recommandé (Intel Haswell 2013+, AMD Excavator 2015+)
- 4 cœurs minimum, 8+ recommandés

**RAM** :
- 8 GB minimum
- 16 GB+ recommandé pour entraînement de gros modèles

**GPU** (Optionnel) :
- Support OpenCL 1.2+
- AMD, NVIDIA, ou Intel GPU

**Stockage** :
- 500 MB pour sources et binaire
- 10+ GB pour datasets

---

## 🐧 Linux (Ubuntu/Debian)

### Installation Rapide

```bash
# Mettre à jour les paquets
sudo apt-get update

# Installer les dépendances
sudo apt-get install -y \
    build-essential \
    g++ \
    make \
    cmake \
    git \
    libopencl-dev \
    ocl-icd-opencl-dev \
    opencl-headers \
    libsfml-dev \
    liblua5.3-dev \
    pkg-config

# Vérifier GCC version (doit être ≥ 9.0)
g++ --version
```

### Installation OpenCL

#### Pour GPU NVIDIA

```bash
# Installer CUDA Toolkit (inclut OpenCL)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
sudo apt-get update
sudo apt-get install -y cuda

# Vérifier
nvidia-smi
clinfo
```

#### Pour GPU AMD

```bash
# ROCm pour AMD
sudo apt-get install -y rocm-opencl-dev

# Vérifier
clinfo
```

#### Pour CPU uniquement (Intel/AMD)

```bash
# PoCL (Portable OpenCL)
sudo apt-get install -y pocl-opencl-icd

# Vérifier
clinfo
```

---

## 🍎 macOS

### Installation via Homebrew

```bash
# Installer Homebrew (si pas déjà fait)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Installer les dépendances
brew install gcc make cmake git sfml lua

# OpenCL est fourni par macOS (Metal)
# Installer headers
brew install opencl-headers
```

### Configurer GCC

```bash
# macOS utilise Clang par défaut, forcer GCC
export CC=/usr/local/bin/gcc-11
export CXX=/usr/local/bin/g++-11

# Ajouter au ~/.zshrc ou ~/.bash_profile
echo 'export CC=/usr/local/bin/gcc-11' >> ~/.zshrc
echo 'export CXX=/usr/local/bin/g++-11' >> ~/.zshrc
source ~/.zshrc
```

---

## 🪟 Windows (WSL2)

### Installer WSL2

```powershell
# PowerShell en administrateur
wsl --install -d Ubuntu-20.04
wsl --set-default-version 2

# Redémarrer puis entrer dans WSL
wsl
```

### Suivre les instructions Linux

Une fois dans WSL, suivre les étapes [Linux (Ubuntu/Debian)](#-linux-ubuntudebian).

### Support GPU dans WSL2

```bash
# Pour NVIDIA GPU
# Installer driver Windows NVIDIA
# Dans WSL:
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-wsl-ubuntu-11-8-local_11.8.0-1_amd64.deb
sudo dpkg -i cuda-repo-wsl-ubuntu-11-8-local_11.8.0-1_amd64.deb
sudo cp /var/cuda-repo-wsl-ubuntu-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get install -y cuda

# Vérifier
nvidia-smi
```

---

## 🔧 Compilation de Mímir

### Cloner le dépôt

```bash
git clone https://github.com/votre-username/mimir.git
cd mimir
```

### Configuration du Makefile (optionnel)

Éditer `Makefile` si nécessaire :

```makefile
# Personnaliser les flags
FLAGS = -ffp-contract=fast \
        -funroll-loops \
        -funsafe-math-optimizations \
        -fno-trapping-math \
        -fno-math-errno

# Changer le compilateur si nécessaire
CXX = g++  # ou clang++
```

### Compiler

```bash
# Nettoyage
make clean

# Compilation complète
make

# Vérifier le binaire
ls -lh bin/unet
file bin/unet
```

**Sortie attendue :**

```
-rwxrwxr-x 1 user user 1.5M bin/unet
bin/unet: ELF 64-bit LSB executable, x86-64, dynamically linked
```

### Compilation sélective

```bash
# UNet uniquement
make bin/unet

# ConditionalVAE
make bin/train_conditional_vae

# Tests RAM Manager
make bin/test_ram_manager
make bin/test_advanced_ram
```

---

## ✅ Vérification de l'Installation

### Test 1 : Exécution basique

```bash
./bin/unet --help
```

**Sortie attendue :** Affichage de l'aide ou démarrage du programme.

### Test 2 : OpenCL

```bash
# Vérifier les devices OpenCL
clinfo

# Devrait afficher :
# - Plateforme(s) disponible(s)
# - Device(s) (CPU/GPU)
# - Version OpenCL
```

### Test 3 : Script Lua

```bash
cat > test_install.lua << 'EOF'
log("=== Test Installation Mímir ===")
log("Lua: OK")

tokenizer.create(1000)
log("Tokenizer: OK")

local config = {
    num_layers = 2,
    d_model = 128,
    num_heads = 4,
    vocab_size = 1000
}

model.create("encoder", config)
log("Model creation: OK")

local ok, params = model.build()
if ok then
    log(string.format("Model build: OK (%d params)", params))
else
    log("Model build: FAILED")
end

log("=== Installation Réussie! ===")
EOF

./bin/unet --script test_install.lua
```

### Test 4 : SIMD/AVX2

```bash
# Vérifier support AVX2
grep -q avx2 /proc/cpuinfo && echo "AVX2: ✓ Supporté" || echo "AVX2: ✗ Non supporté"

# Si AVX2 non supporté, recompiler sans -mavx2
# Éditer Makefile et enlever -mavx2
```

### Test 5 : Configuration JSON

```bash
# Vérifier que config.json est valide
cat config.json | python3 -m json.tool > /dev/null && echo "JSON: ✓ Valide" || echo "JSON: ✗ Invalide"
```

---

## 🐛 Résolution de Problèmes

### Erreur : `lua.h: No such file or directory`

**Solution Ubuntu/Debian :**

```bash
sudo apt-get install liblua5.3-dev

# Vérifier installation
dpkg -L liblua5.3-dev | grep lua.h
# Devrait afficher: /usr/include/lua5.3/lua.h
```

**Solution macOS :**

```bash
brew install lua@5.3
brew link lua@5.3
```

---

### Erreur : `CL/cl.h: No such file or directory`

**Solution Ubuntu/Debian :**

```bash
sudo apt-get install opencl-headers ocl-icd-opencl-dev
```

**Solution macOS :**

```bash
brew install opencl-headers

# Vérifier
ls /usr/local/include/CL/
```

---

### Erreur : `SFML/Graphics.hpp: No such file`

**Solution Ubuntu/Debian :**

```bash
sudo apt-get install libsfml-dev

# Vérifier
dpkg -L libsfml-dev | grep Graphics.hpp
```

**Solution macOS :**

```bash
brew install sfml
```

---

### Erreur : Linking failed (undefined references)

**Vérifier les bibliothèques :**

```bash
# Ubuntu/Debian
ldconfig -p | grep -E 'OpenCL|sfml|lua'

# macOS
brew list sfml lua
```

**Ajouter au Makefile si nécessaire :**

```makefile
LDFLAGS = -L/usr/local/lib
CXXFLAGS = -I/usr/local/include
```

---

### Segmentation Fault au démarrage

**Causes possibles :**
1. Dataset manquant
2. Config JSON invalide
3. Incompatibilité OpenCL

**Solutions :**

```bash
# 1. Vérifier config.json
cat config.json | python3 -m json.tool

# 2. Créer dataset test
mkdir -p datasets/text
echo "Test data" > datasets/text/test.txt

# 3. Désactiver OpenCL temporairement
# Éditer src/main.cpp et commenter l'initialisation OpenCL
```

---

### Performance lente

**Optimisations :**

```bash
# Vérifier compilation optimisée
grep "\-O3" Makefile  # Devrait être présent

# Vérifier AVX2
lscpu | grep avx2

# Activer tous les cœurs OpenMP
export OMP_NUM_THREADS=$(nproc)

# Recompiler avec optimisations agressives
make clean
make CXXFLAGS="-O3 -march=native -mavx2 -ffast-math"
```

---

## 🚀 Installation Avancée

### Build avec CMake (Alternatif)

```bash
# Créer CMakeLists.txt
cat > CMakeLists.txt << 'EOF'
cmake_minimum_required(VERSION 3.15)
project(mimir CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -mavx2 -fopenmp")

find_package(OpenCL REQUIRED)
find_package(SFML 2.5 COMPONENTS graphics window system REQUIRED)
find_package(Lua 5.3 REQUIRED)

file(GLOB SOURCES "src/*.cpp")
add_executable(mimir ${SOURCES})

target_include_directories(mimir PRIVATE 
    ${LUA_INCLUDE_DIR}
    ${OpenCL_INCLUDE_DIRS}
    src
)

target_link_libraries(mimir 
    ${OpenCL_LIBRARIES}
    sfml-graphics sfml-window sfml-system
    ${LUA_LIBRARIES}
    OpenMP::OpenMP_CXX
)
EOF

# Build
mkdir build && cd build
cmake ..
make -j$(nproc)
```

---

### Installation Système (Optionnel)

```bash
# Installer binaire
sudo cp bin/unet /usr/local/bin/mimir

# Installer docs
sudo mkdir -p /usr/local/share/doc/mimir
sudo cp -r docs/* /usr/local/share/doc/mimir/

# Installer scripts exemples
sudo mkdir -p /usr/local/share/mimir/scripts
sudo cp scripts/*.lua /usr/local/share/mimir/scripts/

# Utilisation
mimir --script /usr/local/share/mimir/scripts/example_encoder.lua
```

---

## 📊 Benchmarks Post-Installation

```bash
# Créer script benchmark
cat > benchmark.lua << 'EOF'
log("=== Benchmark Mímir ===")

local start = os.clock()

tokenizer.create(10000)
local config = {
    num_layers = 6,
    d_model = 512,
    num_heads = 8,
    vocab_size = 10000
}

model.create("encoder", config)
model.build()

local elapsed = os.clock() - start
log(string.format("Temps: %.2fs", elapsed))
log("=== Benchmark Terminé ===")
EOF

time ./bin/unet --script benchmark.lua
```

---

## 🔗 Ressources Supplémentaires

- **Documentation** : [INDEX](../00-INDEX.md)
- **Démarrage Rapide** : [Quick Start](01-Quick-Start.md)
- **Exemples** : `scripts/`
- **Support** : GitHub Issues

---

**Installation réussie !** 🎉 Vous pouvez maintenant utiliser Mímir.
