#!/usr/bin/env bash
set -euo pipefail

WORKROOT="${WORKROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
BUILD_DIR="${BUILD_DIR:-$WORKROOT/build}"
BUILD_TYPE="${BUILD_TYPE:-Release}"

if [[ ! -f "$WORKROOT/CMakeLists.txt" ]]; then
  echo "[error] CMakeLists.txt introuvable dans WORKROOT=$WORKROOT"
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

declare -a FEATURE_KEYS=(
  ENABLE_SIMD
  ENABLE_OPENMP
  ENABLE_VULKAN
  ENABLE_OPENCL
  ENABLE_CUDA
  ENABLE_ROCM
  ENABLE_FFMPEG
  ENABLE_SFML
  ENABLE_LZ4
  ENABLE_SCRIPTING_REST
  ENABLE_SCRIPTING_JS
  ENABLE_SCRIPTING_CSHARP
  ENABLE_SCRIPTING_RUST
  MIMIR_ENABLE_TESTS
)

declare -A FEATURE_LABELS=(
  [ENABLE_SIMD]="SIMD (AVX2/FMA/F16C/BMI2)"
  [ENABLE_OPENMP]="OpenMP"
  [ENABLE_VULKAN]="Vulkan Compute"
  [ENABLE_OPENCL]="OpenCL Compute"
  [ENABLE_CUDA]="CUDA Compute"
  [ENABLE_ROCM]="ROCm Compute"
  [ENABLE_FFMPEG]="FFmpeg (audio/vidéo)"
  [ENABLE_SFML]="Visualizer (SFML)"
  [ENABLE_LZ4]="Compression LZ4"
  [ENABLE_SCRIPTING_REST]="Bridge REST (placeholder)"
  [ENABLE_SCRIPTING_JS]="Bridge JavaScript"
  [ENABLE_SCRIPTING_CSHARP]="Bridge C#"
  [ENABLE_SCRIPTING_RUST]="Bridge Rust"
  [MIMIR_ENABLE_TESTS]="Tests unitaires (CTest)"
)

declare -A FEATURE_DEFAULTS=(
  [ENABLE_SIMD]=1
  [ENABLE_OPENMP]=1
  [ENABLE_VULKAN]=1
  [ENABLE_OPENCL]=1
  [ENABLE_CUDA]=0
  [ENABLE_ROCM]=0
  [ENABLE_FFMPEG]=1
  [ENABLE_SFML]=1
  [ENABLE_LZ4]=1
  [ENABLE_SCRIPTING_REST]=0
  [ENABLE_SCRIPTING_JS]=1
  [ENABLE_SCRIPTING_CSHARP]=1
  [ENABLE_SCRIPTING_RUST]=1
  [MIMIR_ENABLE_TESTS]=0
)

is_interactive=0
if [[ -t 0 && -t 1 ]]; then
  is_interactive=1
fi
if [[ "${NON_INTERACTIVE:-0}" == "1" ]]; then
  is_interactive=0
fi

get_feature() {
  local key="$1"
  local env_val="${!key:-}"
  if [[ -n "$env_val" ]]; then
    case "$env_val" in
      1|ON|on|true|TRUE|yes|YES) echo 1 ;;
      0|OFF|off|false|FALSE|no|NO) echo 0 ;;
      *) echo "${FEATURE_DEFAULTS[$key]}" ;;
    esac
  else
    echo "${FEATURE_DEFAULTS[$key]}"
  fi
}

declare -A FEATURE_VALUES
for key in "${FEATURE_KEYS[@]}"; do
  FEATURE_VALUES[$key]="$(get_feature "$key")"
done

if [[ -n "${PRESET_MINIMAL:-}" ]]; then
  for key in "${FEATURE_KEYS[@]}"; do
    FEATURE_VALUES[$key]=0
  done
  FEATURE_VALUES[ENABLE_SIMD]=1
  FEATURE_VALUES[ENABLE_OPENMP]=1
fi

if [[ -n "${PRESET_FULL:-}" ]]; then
  for key in "${FEATURE_KEYS[@]}"; do
    FEATURE_VALUES[$key]=1
  done
fi

cmake_bool() {
  if [[ "$1" == "1" ]]; then
    echo ON
  else
    echo OFF
  fi
}

print_header() {
  echo
  echo "╭──────────────────────────────────────────────────────────────╮"
  echo "│ Mímir Configure                                              │"
  echo "│ Sélectionne les features à compiler (style utilitaire)      │"
  echo "╰──────────────────────────────────────────────────────────────╯"
  echo
}

interactive_configure_whiptail() {
  local whiptail_bin
  whiptail_bin="$(command -v whiptail || true)"
  if [[ -z "$whiptail_bin" ]]; then
    return 1
  fi

  local checklist_args=()
  for key in "${FEATURE_KEYS[@]}"; do
    local state="OFF"
    if [[ "${FEATURE_VALUES[$key]}" == "1" ]]; then
      state="ON"
    fi
    checklist_args+=("$key" "${FEATURE_LABELS[$key]}" "$state")
  done

  local selected
  if ! selected="$($whiptail_bin \
      --title "Mímir Configure" \
      --checklist "Flèches: naviguer | Espace: (dé)sélectionner | Tab: OK" \
      26 110 16 \
      "${checklist_args[@]}" \
      3>&1 1>&2 2>&3)"; then
    echo "Annulé."
    exit 0
  fi

  for key in "${FEATURE_KEYS[@]}"; do
    FEATURE_VALUES[$key]=0
  done

  for key in $selected; do
    key="${key//\"/}"
    if [[ -n "${FEATURE_VALUES[$key]+x}" ]]; then
      FEATURE_VALUES[$key]=1
    fi
  done

  local bt
  if ! bt="$($whiptail_bin \
      --title "Mímir Configure" \
      --menu "Choisis le Build type" \
      18 80 8 \
      "Release" "Optimisé" \
      "Debug" "Debug symboles" \
      "RelWithDebInfo" "Optimisé + debug" \
      "MinSizeRel" "Optimisé taille" \
      3>&1 1>&2 2>&3)"; then
    echo "Annulé."
    exit 0
  fi
  BUILD_TYPE="$bt"
  return 0
}

interactive_configure() {
  if interactive_configure_whiptail; then
    return 0
  fi

  echo "[warn] whiptail introuvable, fallback mode texte simple."
  print_header
  echo "Installe whiptail pour une sélection flèches + espace: sudo apt-get install -y whiptail"
  echo
  local i=1
  for key in "${FEATURE_KEYS[@]}"; do
    local mark="[ ]"
    if [[ "${FEATURE_VALUES[$key]}" == "1" ]]; then
      mark="[x]"
    fi
    printf " %2d) %s %-26s (%s)\n" "$i" "$mark" "${FEATURE_LABELS[$key]}" "$key"
    ((i++))
  done
  echo
  echo "Aucune modification interactive effectuée (fallback)."
}

if (( is_interactive == 1 )) && [[ "${AUTO_ACCEPT:-0}" != "1" ]]; then
  interactive_configure
fi

echo
printf "Configuration retenue:\n"
for key in "${FEATURE_KEYS[@]}"; do
  printf "  - %-24s : %s\n" "$key" "$(cmake_bool "${FEATURE_VALUES[$key]}")"
done
printf "  - %-24s : %s\n" "CMAKE_BUILD_TYPE" "$BUILD_TYPE"
printf "  - %-24s : %s\n" "BUILD_DIR" "$BUILD_DIR"

echo

echo "[1/4] Installation des dépendances système..."
$SUDO apt-get update

REQUIRED_PACKAGES=(
  cmake
  build-essential
  lua5.3
  liblua5.3-dev
  pkg-config
)

if [[ "${FEATURE_VALUES[ENABLE_OPENMP]}" == "1" ]]; then
  REQUIRED_PACKAGES+=(libomp-dev)
fi

OPTIONAL_PACKAGES=()
if [[ "${FEATURE_VALUES[ENABLE_VULKAN]}" == "1" ]]; then
  OPTIONAL_PACKAGES+=(vulkan-tools libvulkan-dev glslang-tools)
fi
if [[ "${FEATURE_VALUES[ENABLE_OPENCL]}" == "1" ]]; then
  OPTIONAL_PACKAGES+=(ocl-icd-opencl-dev clinfo)
fi
if [[ "${FEATURE_VALUES[ENABLE_FFMPEG]}" == "1" ]]; then
  OPTIONAL_PACKAGES+=(ffmpeg libavcodec-dev libavformat-dev libavutil-dev libswscale-dev libswresample-dev)
fi
if [[ "${FEATURE_VALUES[ENABLE_SFML]}" == "1" ]]; then
  OPTIONAL_PACKAGES+=(libsfml-dev)
fi
if [[ "${FEATURE_VALUES[ENABLE_LZ4]}" == "1" ]]; then
  OPTIONAL_PACKAGES+=(liblz4-dev)
fi

$SUDO apt-get install -y "${REQUIRED_PACKAGES[@]}" "${OPTIONAL_PACKAGES[@]}"

echo "[2/4] Vérification des outils..."
cmake --version | head -n 1
c++ --version | head -n 1
lua -v || true
if [[ "${FEATURE_VALUES[ENABLE_VULKAN]}" == "1" ]]; then
  command -v glslangValidator >/dev/null 2>&1 && glslangValidator --version | head -n 1 || true
fi
if [[ "${FEATURE_VALUES[ENABLE_OPENCL]}" == "1" ]]; then
  command -v clinfo >/dev/null 2>&1 && clinfo | sed -n '1,20p' || true
fi
if [[ "${FEATURE_VALUES[ENABLE_CUDA]}" == "1" ]]; then
  echo "[note] CUDA activé: installe CUDA Toolkit si nécessaire (non installé automatiquement ici)."
fi
if [[ "${FEATURE_VALUES[ENABLE_ROCM]}" == "1" ]]; then
  echo "[note] ROCm activé: installe ROCm/HIP si nécessaire (non installé automatiquement ici)."
fi

echo "[3/4] Configuration CMake..."

CMAKE_ARGS=(
  -S "$WORKROOT"
  -B "$BUILD_DIR"
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
)

for key in "${FEATURE_KEYS[@]}"; do
  CMAKE_ARGS+=("-D${key}=$(cmake_bool "${FEATURE_VALUES[$key]}")")
done

cmake "${CMAKE_ARGS[@]}"

echo "[4/4] Préparation terminée."
echo "Build prêt dans: $BUILD_DIR"
echo "Commande de compilation: cmake --build \"$BUILD_DIR\" -j\"$(nproc)\""
