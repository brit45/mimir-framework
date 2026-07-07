#!/usr/bin/env bash
set -euo pipefail

# Mimir project bootstrap script (Linux Debian/Ubuntu)
# Usage:
#   ./config.sh
#   WORKROOT=/path/to/tensor-2 ./config.sh
#   ENABLE_VULKAN=0 ENABLE_OPENCL=0 ./config.sh
#   BUILD_TYPE=Debug ./config.sh

WORKROOT="${WORKROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
BUILD_DIR="${BUILD_DIR:-$WORKROOT/build}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
ENABLE_VULKAN="${ENABLE_VULKAN:-1}"
ENABLE_OPENCL="${ENABLE_OPENCL:-1}"
ENABLE_SFML="${ENABLE_SFML:-1}"
ENABLE_LZ4="${ENABLE_LZ4:-1}"
ENABLE_OPENMP="${ENABLE_OPENMP:-1}"
ENABLE_SIMD="${ENABLE_SIMD:-1}"

if [[ ! -f "$WORKROOT/CMakeLists.txt" ]]; then
  echo "[error] CMakeLists.txt introuvable dans WORKROOT=$WORKROOT"
  echo "        Passe WORKROOT vers la racine du projet tensor-2."
  exit 1
fi

if [[ "$OSTYPE" != linux* ]]; then
  echo "[error] Ce script cible Linux (Debian/Ubuntu). OSTYPE=$OSTYPE"
  exit 1
fi

if ! command -v apt-get >/dev/null 2>&1; then
  echo "[error] apt-get introuvable. Utilise le guide docs pour ton OS/distribution."
  exit 1
fi

SUDO=""
if [[ "${EUID:-$(id -u)}" -ne 0 ]]; then
  if command -v sudo >/dev/null 2>&1; then
    SUDO="sudo"
  else
    echo "[error] Exécute en root ou installe sudo."
    exit 1
  fi
fi

echo "[1/4] Installation des dépendances système..."
$SUDO apt-get update

REQUIRED_PACKAGES=(
  cmake
  build-essential
  lua5.3
  liblua5.3-dev
  libomp-dev
  pkg-config
)

OPTIONAL_PACKAGES=()
if [[ "$ENABLE_VULKAN" == "1" ]]; then
  OPTIONAL_PACKAGES+=(vulkan-tools libvulkan-dev glslang-tools)
fi
if [[ "$ENABLE_OPENCL" == "1" ]]; then
  OPTIONAL_PACKAGES+=(ocl-icd-opencl-dev clinfo)
fi
if [[ "$ENABLE_SFML" == "1" ]]; then
  OPTIONAL_PACKAGES+=(libsfml-dev)
fi
if [[ "$ENABLE_LZ4" == "1" ]]; then
  OPTIONAL_PACKAGES+=(liblz4-dev)
fi

$SUDO apt-get install -y "${REQUIRED_PACKAGES[@]}" "${OPTIONAL_PACKAGES[@]}"

echo "[2/4] Vérification des outils..."
cmake --version | head -n 1
c++ --version | head -n 1
lua -v || true
if [[ "$ENABLE_VULKAN" == "1" ]]; then
  command -v glslangValidator >/dev/null 2>&1 && glslangValidator --version | head -n 1 || true
fi
if [[ "$ENABLE_OPENCL" == "1" ]]; then
  command -v clinfo >/dev/null 2>&1 && clinfo | sed -n '1,20p' || true
fi

echo "[3/4] Configuration CMake..."
cmake -S "$WORKROOT" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
  -DENABLE_SIMD=$([[ "$ENABLE_SIMD" == "1" ]] && echo ON || echo OFF) \
  -DENABLE_OPENMP=$([[ "$ENABLE_OPENMP" == "1" ]] && echo ON || echo OFF) \
  -DENABLE_VULKAN=$([[ "$ENABLE_VULKAN" == "1" ]] && echo ON || echo OFF) \
  -DENABLE_OPENCL=$([[ "$ENABLE_OPENCL" == "1" ]] && echo ON || echo OFF) \
  -DENABLE_SFML=$([[ "$ENABLE_SFML" == "1" ]] && echo ON || echo OFF) \
  -DENABLE_LZ4=$([[ "$ENABLE_LZ4" == "1" ]] && echo ON || echo OFF)

echo "[4/4] Préparation terminée."
echo "Build prêt dans: $BUILD_DIR"
echo "Commande de compilation: cmake --build \"$BUILD_DIR\" -j\"$(nproc)\""
