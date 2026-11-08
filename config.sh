#!/usr/bin/env bash
set -eo pipefail

WORKROOT="${WORKROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
BUILD_DIR="${BUILD_DIR:-$WORKROOT/build}"
BUILD_TYPE="Release"
VIZ_BACKEND="${MIMIR_VIZ_BACKEND:-}"
SKIP_SYSTEM_DEPS="${SKIP_SYSTEM_DEPS:-0}"
CMAKE_GENERATOR=""
EXTRA_CMAKE_ARGS=()
FEATURE_OVERRIDE_KEYS=()
FEATURE_OVERRIDE_VALUES=()

set_feature_override() {
  local key="$1" value="$2" i
  for i in "${!FEATURE_OVERRIDE_KEYS[@]}"; do
    if [[ "${FEATURE_OVERRIDE_KEYS[$i]}" == "$key" ]]; then
      FEATURE_OVERRIDE_VALUES[$i]="$value"
      return 0
    fi
  done
  FEATURE_OVERRIDE_KEYS+=("$key")
  FEATURE_OVERRIDE_VALUES+=("$value")
}

to_upper() {
  printf '%s' "$1" | tr '[:lower:]' '[:upper:]'
}

detect_host_os() {
  local kernel
  kernel="$(uname -s 2>/dev/null || printf unknown)"
  case "$kernel" in
    Linux*) HOST_OS=linux ;;
    Darwin*) HOST_OS=macos ;;
    FreeBSD*) HOST_OS=freebsd ;;
    OpenBSD*) HOST_OS=openbsd ;;
    NetBSD*) HOST_OS=netbsd ;;
    MINGW*|MSYS*|CYGWIN*) HOST_OS=windows ;;
    *) HOST_OS=other ;;
  esac
}

detect_package_manager() {
  PACKAGE_MANAGER=none
  case "$HOST_OS" in
    macos)
      if command -v brew >/dev/null 2>&1; then PACKAGE_MANAGER=brew; fi ;;
    windows)
      case "${MSYSTEM:-}" in
        UCRT64|MINGW64|MINGW32|CLANG64|CLANGARM64)
          if command -v pacman >/dev/null 2>&1; then PACKAGE_MANAGER=msys2-pacman; fi ;;
      esac ;;
    freebsd)
      if command -v pkg >/dev/null 2>&1; then PACKAGE_MANAGER=pkg; fi ;;
    openbsd)
      if command -v pkg_add >/dev/null 2>&1; then PACKAGE_MANAGER=pkg_add; fi ;;
    netbsd)
      if command -v pkgin >/dev/null 2>&1; then PACKAGE_MANAGER=pkgin
      elif command -v pkg_add >/dev/null 2>&1; then PACKAGE_MANAGER=pkg_add; fi ;;
    linux)
      if command -v apt-get >/dev/null 2>&1; then PACKAGE_MANAGER=apt
      elif command -v dnf >/dev/null 2>&1; then PACKAGE_MANAGER=dnf
      elif command -v yum >/dev/null 2>&1; then PACKAGE_MANAGER=yum
      elif command -v pacman >/dev/null 2>&1; then PACKAGE_MANAGER=pacman
      elif command -v zypper >/dev/null 2>&1; then PACKAGE_MANAGER=zypper
      elif command -v apk >/dev/null 2>&1; then PACKAGE_MANAGER=apk
      elif command -v xbps-install >/dev/null 2>&1; then PACKAGE_MANAGER=xbps
      elif command -v eopkg >/dev/null 2>&1; then PACKAGE_MANAGER=eopkg
      elif command -v emerge >/dev/null 2>&1; then PACKAGE_MANAGER=emerge
      elif command -v brew >/dev/null 2>&1; then PACKAGE_MANAGER=brew
      fi ;;
  esac
}

usage() {
  cat <<'HELP'
Usage: ./config.sh [options]
  --gui BACKEND           SFML, QT (Qt 6), GTK (GTK 3), WEB ou NONE
  --viz-backend BACKEND   Alias de --gui
  --build-dir DIR         Répertoire CMake (défaut: build)
  --enable FEATURE       Activer une option CMake (ex.: MIMIR_ENABLE_TESTS)
  --disable FEATURE      Désactiver une option CMake (ex.: ENABLE_VULKAN)
  --ddp / --no-ddp       Compiler le support DDP via MPI (nécessite OpenMPI)
  --amp / --no-amp       Compiler le support AMP (autocast float16/bfloat16)
  --build-type TYPE      Release, Debug, RelWithDebInfo ou MinSizeRel
  --generator NAME       Générateur CMake (ex.: Ninja)
  --cmake-arg ARG        Argument CMake supplémentaire, répétable (cross compilation)
  --non-interactive      Utiliser les choix fournis sans ouvrir les menus
  --skip-system-deps     Ne pas installer les dépendances (elles sont déjà installées)
  -h, --help              Afficher cette aide

Sans --gui: MIMIR_VIZ_BACKEND, puis cache CMake, puis SFML sous Linux ou NONE ailleurs.
Qt/GTK/Web sont limités à Linux; SFML ou NONE sont disponibles sur les autres OS.
Installateurs détectés: apt, dnf/yum, pacman, zypper, apk, xbps, eopkg, emerge, Homebrew, BSD pkg/pkgin et MSYS2.
Si aucun installateur n'est reconnu, installe manuellement CMake, un compilateur C++ et Lua 5.3+, puis utilise --skip-system-deps.
Exemple portable sans interface: ./config.sh --gui NONE --non-interactive --skip-system-deps
HELP
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gui|--viz-backend|--build-dir)
      if [[ $# -lt 2 || -z "$2" || "$2" == --* ]]; then
        echo "[error] Valeur manquante pour $1" >&2; exit 1
      fi
      if [[ "$1" == "--build-dir" ]]; then BUILD_DIR="$2"; else VIZ_BACKEND="$2"; fi
      shift 2 ;;
    --ddp|--enable-ddp) set_feature_override ENABLE_MPI 1; shift ;;
    --no-ddp|--disable-ddp) set_feature_override ENABLE_MPI 0; shift ;;
    --amp|--enable-amp) set_feature_override ENABLE_AMP 1; shift ;;
    --no-amp|--disable-amp) set_feature_override ENABLE_AMP 0; shift ;;
    --enable|--disable|--build-type|--generator|--cmake-arg)
      if [[ $# -lt 2 || -z "$2" || "$2" == --* ]]; then
        echo "[error] Valeur manquante pour $1" >&2; exit 1
      fi
      case "$1" in
        --enable) set_feature_override "$2" 1 ;;
        --disable) set_feature_override "$2" 0 ;;
        --build-type) BUILD_TYPE="$2" ;;
        --generator) CMAKE_GENERATOR="$2" ;;
        --cmake-arg) EXTRA_CMAKE_ARGS+=("$2") ;;
      esac
      shift 2 ;;
    --non-interactive) NON_INTERACTIVE=1; shift ;;
    --skip-system-deps) SKIP_SYSTEM_DEPS=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[error] Option inconnue: $1" >&2; usage >&2; exit 1 ;;
  esac
done

detect_host_os
detect_package_manager

if [[ -z "$VIZ_BACKEND" && -f "$BUILD_DIR/CMakeCache.txt" ]]; then
  VIZ_BACKEND="$(sed -n 's/^MIMIR_VIZ_BACKEND:STRING=//p' "$BUILD_DIR/CMakeCache.txt")"
  VIZ_BACKEND="$(to_upper "$VIZ_BACKEND")"
  if [[ "$VIZ_BACKEND" == "SFML" ]]; then
    legacy_sfml_enabled="$(sed -n 's/^ENABLE_SFML:BOOL=//p' "$BUILD_DIR/CMakeCache.txt")"
    if [[ "$legacy_sfml_enabled" == "OFF" ]]; then VIZ_BACKEND="NONE"; fi
  fi
fi
if [[ -z "$VIZ_BACKEND" ]]; then
  case "$HOST_OS" in
    linux)
      if [[ "$PACKAGE_MANAGER" == "none" ]]; then VIZ_BACKEND=NONE
      else VIZ_BACKEND=SFML; fi ;;
    *) VIZ_BACKEND=NONE ;;
  esac
fi
VIZ_BACKEND="$(to_upper "$VIZ_BACKEND")"
case "$HOST_OS:$VIZ_BACKEND" in
  linux:SFML|linux:QT|linux:GTK|linux:WEB|linux:NONE|*:SFML|*:NONE) ;;
  *:QT|*:GTK|*:WEB)
    echo "[error] Le backend Viz $VIZ_BACKEND est actuellement limité à Linux par CMake; choisis SFML ou NONE." >&2
    exit 1 ;;
  *) echo "[error] Interface Viz inconnue: $VIZ_BACKEND (SFML/QT/GTK/WEB/NONE)" >&2; exit 1 ;;
esac

if [[ ! -f "$WORKROOT/CMakeLists.txt" ]]; then
  echo "[error] CMakeLists.txt introuvable dans WORKROOT=$WORKROOT"
  exit 1
fi

FEATURE_KEYS=(
  ENABLE_SIMD
  ENABLE_OPENMP
  ENABLE_VULKAN
  ENABLE_OPENCL
  MIMIR_AUTO_DETECT_GPU_BACKENDS
  ENABLE_CUDA
  ENABLE_ROCM
  ENABLE_MPI
  ENABLE_AMP
  ENABLE_FFMPEG
  ENABLE_LZ4
  ENABLE_SCRIPTING_REST
  ENABLE_SCRIPTING_JS
  ENABLE_SCRIPTING_CSHARP
  ENABLE_SCRIPTING_RUST
  MIMIR_ENABLE_TESTS
)

FEATURE_LABELS=(
  "SIMD auto (SSE2/AVX/AVX2/AVX-512/FMA/F16C)"
  "OpenMP"
  "Vulkan Compute"
  "OpenCL Compute"
  "Auto-détection GPU (CUDA/ROCm)"
  "CUDA Compute"
  "ROCm Compute"
  "Distributed Data Parallel (MPI/DDP)"
  "Mixed precision autocast (AMP)"
  "FFmpeg (audio/vidéo)"
  "Compression LZ4"
  "Bridge REST (placeholder)"
  "Bridge JavaScript"
  "Bridge C#"
  "Bridge Rust"
  "Tests unitaires (CTest)"
)

FEATURE_DEFAULTS=(
  1 1 1 1 0 0 0 0 1 1 1 0 0 0 0 0
)

case "$BUILD_TYPE" in
  Release|Debug|RelWithDebInfo|MinSizeRel) ;;
  *) echo "[error] Type de build invalide: $BUILD_TYPE" >&2; exit 1 ;;
esac
feature_index() {
  local key="$1" i
  for i in "${!FEATURE_KEYS[@]}"; do
    if [[ "${FEATURE_KEYS[$i]}" == "$key" ]]; then
      printf '%s' "$i"
      return 0
    fi
  done
  return 1
}

if [[ "$HOST_OS" == "openbsd" || "$HOST_OS" == "netbsd" || "$HOST_OS" == "other" ]]; then
  openmp_index="$(feature_index ENABLE_OPENMP 2>/dev/null || true)"
  if [[ -n "$openmp_index" ]]; then FEATURE_DEFAULTS[$openmp_index]=0; fi
fi

for key in "${FEATURE_OVERRIDE_KEYS[@]}"; do
  if ! feature_index "$key" >/dev/null; then
    echo "[error] Feature inconnue: $key" >&2; exit 1
  fi
done

is_interactive=0
if [[ -t 0 && -t 1 ]]; then
  is_interactive=1
fi
if [[ "${NON_INTERACTIVE:-0}" == "1" ]]; then
  is_interactive=0
fi

get_feature() {
  local key="$1" i index
  for i in "${!FEATURE_OVERRIDE_KEYS[@]}"; do
    if [[ "${FEATURE_OVERRIDE_KEYS[$i]}" == "$key" ]]; then
      printf '%s' "${FEATURE_OVERRIDE_VALUES[$i]}"
      return 0
    fi
  done
  index="$(feature_index "$key")" || return 1
  printf '%s' "${FEATURE_DEFAULTS[$index]}"
}

feature_value() {
  local index
  index="$(feature_index "$1")" || return 1
  printf '%s' "${FEATURE_VALUES[$index]}"
}

set_feature_value() {
  local index
  index="$(feature_index "$1")" || return 1
  FEATURE_VALUES[$index]="$2"
}

FEATURE_VALUES=()
for i in "${!FEATURE_KEYS[@]}"; do
  FEATURE_VALUES[$i]="$(get_feature "${FEATURE_KEYS[$i]}")"
done

cmake_bool() {
  if [[ "$1" == "1" ]]; then
    echo ON
  else
    echo OFF
  fi
}

generator_needs_ninja() {
  case "$CMAKE_GENERATOR" in
    *Ninja*) return 0 ;;
    *) return 1 ;;
  esac
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
  local i
  for i in "${!FEATURE_KEYS[@]}"; do
    local key="${FEATURE_KEYS[$i]}"
    local state="OFF"
    if [[ "${FEATURE_VALUES[$i]}" == "1" ]]; then
      state="ON"
    fi
    checklist_args+=("$key" "${FEATURE_LABELS[$i]}" "$state")
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

  for i in "${!FEATURE_KEYS[@]}"; do
    FEATURE_VALUES[$i]=0
  done

  for key in $selected; do
    key="${key//\"/}"
    if feature_index "$key" >/dev/null; then
      set_feature_value "$key" 1
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
  echo "Installe whiptail via le gestionnaire de paquets de ton système pour la sélection graphique."
  echo
  local i=1 index key
  for index in "${!FEATURE_KEYS[@]}"; do
    key="${FEATURE_KEYS[$index]}"
    local mark="[ ]"
    if [[ "${FEATURE_VALUES[$index]}" == "1" ]]; then
      mark="[x]"
    fi
    printf " %2d) %s %-36s (%s)\n" "$i" "$mark" "${FEATURE_LABELS[$index]}" "$key"
    i=$((i + 1))
  done
  echo
  echo "Aucune modification interactive effectuée (fallback)."
}

if [[ "$is_interactive" == "1" ]]; then
  interactive_configure
  if command -v whiptail >/dev/null 2>&1; then
    if [[ "$HOST_OS" == "linux" ]]; then
      VIZ_BACKEND="$(whiptail --title "Mímir Viz" --default-item "$VIZ_BACKEND" --menu "Interface Viz (choix exclusif à la compilation)" 18 80 5 \
        SFML "Fenêtre native SFML 3" QT "Fenêtre native Qt 6 + Cairo" GTK "Fenêtre native GTK 3 + Cairo" WEB "Navigateur local + Cairo" NONE "Sans GUI (entraînement headless)" 3>&1 1>&2 2>&3)" || exit 0
    else
      VIZ_BACKEND="$(whiptail --title "Mímir Viz" --default-item "$VIZ_BACKEND" --menu "Interface Viz (choix exclusif à la compilation)" 12 80 2 \
        SFML "Fenêtre native SFML 3" NONE "Sans GUI (entraînement headless)" 3>&1 1>&2 2>&3)" || exit 0
    fi
  else
    if [[ "$HOST_OS" == "linux" ]]; then
      read -r -p "Interface Viz : SFML 3 / QT 6 / GTK 3 / WEB / NONE [$VIZ_BACKEND]: " selected_viz
    else
      read -r -p "Interface Viz : SFML 3 / NONE [$VIZ_BACKEND]: " selected_viz
    fi
    VIZ_BACKEND="${selected_viz:-$VIZ_BACKEND}"
  fi
fi
VIZ_BACKEND="$(to_upper "$VIZ_BACKEND")"
case "$HOST_OS:$VIZ_BACKEND" in
  linux:SFML|linux:QT|linux:GTK|linux:WEB|linux:NONE|*:SFML|*:NONE) ;;
  *) echo "[error] Le backend Viz $VIZ_BACKEND n'est pas disponible sur $HOST_OS avec cette version de CMake." >&2; exit 1 ;;
esac
ENABLE_SFML_VALUE=0
if [[ "$VIZ_BACKEND" == "SFML" ]]; then ENABLE_SFML_VALUE=1; fi

detect_sfml_version() {
  if command -v pkg-config >/dev/null 2>&1; then
    if pkg-config --exists sfml-graphics; then
      pkg-config --modversion sfml-graphics 2>/dev/null || true
      return 0
    fi
    if pkg-config --exists sfml-all; then
      pkg-config --modversion sfml-all 2>/dev/null || true
      return 0
    fi
  fi
  echo ""
}

SFML_DETECTED_VERSION=""
SFML_MODE="disabled"
if [[ "$VIZ_BACKEND" == "SFML" ]]; then SFML_MODE="sfml3-required"; fi

echo
printf "Configuration retenue:\n"
for i in "${!FEATURE_KEYS[@]}"; do
  key="${FEATURE_KEYS[$i]}"
  printf "  - %-24s : %s\n" "$key" "$(cmake_bool "${FEATURE_VALUES[$i]}")"
done
if [[ -n "$SFML_DETECTED_VERSION" ]]; then
  printf "  - %-24s : %s\n" "SFML_DETECTED_VERSION" "$SFML_DETECTED_VERSION"
fi
printf "  - %-24s : %s\n" "MIMIR_VIZ_BACKEND" "$VIZ_BACKEND"
printf "  - %-24s : %s\n" "ENABLE_SFML" "$(cmake_bool "$ENABLE_SFML_VALUE")"
printf "  - %-24s : %s\n" "SFML_MODE" "$SFML_MODE"
printf "  - %-24s : %s\n" "CMAKE_BUILD_TYPE" "$BUILD_TYPE"
printf "  - %-24s : %s\n" "BUILD_DIR" "$BUILD_DIR"
printf "  - %-24s : %s\n" "SYSTEME" "$HOST_OS"
printf "  - %-24s : %s\n" "GESTIONNAIRE" "$PACKAGE_MANAGER"

echo

run_as_admin() {
  if [[ "$(id -u 2>/dev/null || printf 1)" == "0" ]]; then
    "$@"
  elif command -v sudo >/dev/null 2>&1; then
    sudo "$@"
  elif command -v doas >/dev/null 2>&1; then
    doas "$@"
  else
    echo "[error] L'installation système exige les privilèges administrateur; installe les dépendances manuellement ou utilise --skip-system-deps." >&2
    return 1
  fi
}

install_packages() {
  local package_group="$1"
  shift
  [[ $# -gt 0 ]] || return 0
  case "$PACKAGE_MANAGER" in
    apt) run_as_admin apt-get install -y --no-install-recommends "$@" ;;
    dnf) run_as_admin dnf install -y "$@" ;;
    yum) run_as_admin yum install -y "$@" ;;
    pacman) run_as_admin pacman -S --needed --noconfirm "$@" ;;
    msys2-pacman) pacman -S --needed --noconfirm "$@" ;;
    zypper) run_as_admin zypper --non-interactive install "$@" ;;
    apk) run_as_admin apk add "$@" ;;
    xbps) run_as_admin xbps-install -y "$@" ;;
    eopkg) run_as_admin eopkg install -y "$@" ;;
    emerge) run_as_admin emerge --ask=n "$@" ;;
    brew) brew install "$@" ;;
    pkg) run_as_admin pkg install -y "$@" ;;
    pkgin) run_as_admin pkgin -y install "$@" ;;
    pkg_add) run_as_admin pkg_add -I "$@" ;;
    *)
      echo "[warn] Aucun gestionnaire de paquets pris en charge; dépendances $package_group à installer manuellement."
      return 1 ;;
  esac
}

build_package_lists() {
  REQUIRED_PACKAGES=()
  OPTIONAL_PACKAGES=()
  local enable_openmp enable_mpi enable_vulkan enable_opencl enable_ffmpeg enable_lz4
  enable_openmp="$(feature_value ENABLE_OPENMP)"
  enable_mpi="$(feature_value ENABLE_MPI)"
  enable_vulkan="$(feature_value ENABLE_VULKAN)"
  enable_opencl="$(feature_value ENABLE_OPENCL)"
  enable_ffmpeg="$(feature_value ENABLE_FFMPEG)"
  enable_lz4="$(feature_value ENABLE_LZ4)"

  case "$PACKAGE_MANAGER" in
    apt)
      REQUIRED_PACKAGES+=(cmake build-essential lua5.3 liblua5.3-dev pkg-config)
      if [[ "$enable_openmp" == "1" ]]; then REQUIRED_PACKAGES+=(libomp-dev); fi
      if [[ "$enable_mpi" == "1" ]]; then REQUIRED_PACKAGES+=(openmpi-bin libopenmpi-dev); fi
      case "$VIZ_BACKEND" in
        QT) REQUIRED_PACKAGES+=(qt6-base-dev libcairo2-dev) ;;
        GTK) REQUIRED_PACKAGES+=(libgtk-3-dev libcairo2-dev) ;;
        WEB) REQUIRED_PACKAGES+=(libcairo2-dev) ;;
        SFML)
          if command -v apt-cache >/dev/null 2>&1 && apt-cache show libsfml3-dev >/dev/null 2>&1; then
            REQUIRED_PACKAGES+=(libsfml3-dev)
          elif command -v apt-cache >/dev/null 2>&1 &&
               sfml_version="$(apt-cache show libsfml-dev 2>/dev/null | sed -n 's/^Version: //p' | sed -n '1p')" &&
               [[ -n "$sfml_version" ]] && command -v dpkg >/dev/null 2>&1 &&
               dpkg --compare-versions "${sfml_version#*:}" ge 3.0; then
            REQUIRED_PACKAGES+=(libsfml-dev)
          else
            echo "[warn] SFML 3 n'est pas disponible dans les dépôts détectés; CMake vérifiera le paquet déjà installé."
          fi ;;
      esac
      if [[ "$enable_vulkan" == "1" ]]; then OPTIONAL_PACKAGES+=(vulkan-tools libvulkan-dev glslang-tools); fi
      if [[ "$enable_opencl" == "1" ]]; then OPTIONAL_PACKAGES+=(ocl-icd-opencl-dev clinfo); fi
      if [[ "$enable_ffmpeg" == "1" ]]; then OPTIONAL_PACKAGES+=(ffmpeg libavcodec-dev libavformat-dev libavutil-dev libswscale-dev libswresample-dev); fi
      if [[ "$enable_lz4" == "1" ]]; then OPTIONAL_PACKAGES+=(liblz4-dev); fi ;;
    dnf|yum)
      REQUIRED_PACKAGES+=(cmake gcc-c++ make lua lua-devel pkgconf-pkg-config)
      if [[ "$enable_mpi" == "1" ]]; then REQUIRED_PACKAGES+=(openmpi openmpi-devel); fi
      case "$VIZ_BACKEND" in
        SFML) REQUIRED_PACKAGES+=(SFML-devel) ;;
        QT) REQUIRED_PACKAGES+=(qt6-qtbase-devel) ;;
        GTK) REQUIRED_PACKAGES+=(gtk3-devel cairo-devel) ;;
        WEB) REQUIRED_PACKAGES+=(cairo-devel) ;;
      esac
      if [[ "$enable_vulkan" == "1" ]]; then OPTIONAL_PACKAGES+=(vulkan-headers vulkan-loader-devel vulkan-tools glslang); fi
      if [[ "$enable_opencl" == "1" ]]; then OPTIONAL_PACKAGES+=(ocl-icd-devel clinfo); fi
      if [[ "$enable_ffmpeg" == "1" ]]; then OPTIONAL_PACKAGES+=(ffmpeg ffmpeg-devel); fi
      if [[ "$enable_lz4" == "1" ]]; then OPTIONAL_PACKAGES+=(lz4-devel); fi ;;
    pacman)
      REQUIRED_PACKAGES+=(cmake base-devel lua pkgconf)
      if [[ "$enable_mpi" == "1" ]]; then REQUIRED_PACKAGES+=(openmpi); fi
      case "$VIZ_BACKEND" in
        SFML) REQUIRED_PACKAGES+=(sfml) ;;
        QT) REQUIRED_PACKAGES+=(qt6-base) ;;
        GTK) REQUIRED_PACKAGES+=(gtk3 cairo) ;;
        WEB) REQUIRED_PACKAGES+=(cairo) ;;
      esac
      if [[ "$enable_vulkan" == "1" ]]; then OPTIONAL_PACKAGES+=(vulkan-headers vulkan-icd-loader vulkan-tools glslang); fi
      if [[ "$enable_opencl" == "1" ]]; then OPTIONAL_PACKAGES+=(ocl-icd clinfo); fi
      if [[ "$enable_ffmpeg" == "1" ]]; then OPTIONAL_PACKAGES+=(ffmpeg); fi
      if [[ "$enable_lz4" == "1" ]]; then OPTIONAL_PACKAGES+=(lz4); fi ;;
    msys2-pacman)
      local mingw_prefix="${MINGW_PACKAGE_PREFIX:-}"
      if [[ -z "$mingw_prefix" ]]; then
        case "${MSYSTEM:-}" in
          UCRT64) mingw_prefix=mingw-w64-ucrt-x86_64 ;;
          MINGW64) mingw_prefix=mingw-w64-x86_64 ;;
          MINGW32) mingw_prefix=mingw-w64-i686 ;;
          CLANG64) mingw_prefix=mingw-w64-clang-x86_64 ;;
          CLANGARM64) mingw_prefix=mingw-w64-clang-aarch64 ;;
          *) echo "[error] MSYS2 détecté sans environnement MinGW connu (${MSYSTEM:-inconnu})." >&2; return 1 ;;
        esac
      fi
      REQUIRED_PACKAGES+=("$mingw_prefix-cmake" "$mingw_prefix-toolchain" "$mingw_prefix-lua53")
      if generator_needs_ninja; then REQUIRED_PACKAGES+=("$mingw_prefix-ninja"); fi
      if [[ "$enable_mpi" == "1" ]]; then REQUIRED_PACKAGES+=("$mingw_prefix-openmpi"); fi
      case "$VIZ_BACKEND" in
        SFML) REQUIRED_PACKAGES+=("$mingw_prefix-sfml") ;;
        QT) REQUIRED_PACKAGES+=("$mingw_prefix-qt6-base") ;;
        GTK) REQUIRED_PACKAGES+=("$mingw_prefix-gtk3" "$mingw_prefix-cairo") ;;
        WEB) REQUIRED_PACKAGES+=("$mingw_prefix-cairo") ;;
      esac
      if [[ "$enable_vulkan" == "1" ]]; then OPTIONAL_PACKAGES+=("$mingw_prefix-vulkan-headers" "$mingw_prefix-vulkan-loader" "$mingw_prefix-vulkan-tools" "$mingw_prefix-glslang"); fi
      if [[ "$enable_opencl" == "1" ]]; then OPTIONAL_PACKAGES+=("$mingw_prefix-ocl-icd" "$mingw_prefix-clinfo"); fi
      if [[ "$enable_ffmpeg" == "1" ]]; then OPTIONAL_PACKAGES+=("$mingw_prefix-ffmpeg"); fi
      if [[ "$enable_lz4" == "1" ]]; then OPTIONAL_PACKAGES+=("$mingw_prefix-lz4"); fi ;;
    zypper)
      REQUIRED_PACKAGES+=(cmake gcc-c++ make lua53 lua53-devel pkg-config)
      if [[ "$enable_mpi" == "1" ]]; then REQUIRED_PACKAGES+=(openmpi-devel); fi
      case "$VIZ_BACKEND" in
        SFML) REQUIRED_PACKAGES+=(SFML-devel) ;;
        QT) REQUIRED_PACKAGES+=(qt6-base-devel) ;;
        GTK) REQUIRED_PACKAGES+=(gtk3-devel cairo-devel) ;;
        WEB) REQUIRED_PACKAGES+=(cairo-devel) ;;
      esac
      if [[ "$enable_vulkan" == "1" ]]; then OPTIONAL_PACKAGES+=(vulkan-headers libvulkan1 vulkan-tools glslang); fi
      if [[ "$enable_opencl" == "1" ]]; then OPTIONAL_PACKAGES+=(ocl-icd clinfo); fi
      if [[ "$enable_ffmpeg" == "1" ]]; then OPTIONAL_PACKAGES+=(ffmpeg-4 libavformat-devel libavcodec-devel libavutil-devel libswscale-devel libswresample-devel); fi
      if [[ "$enable_lz4" == "1" ]]; then OPTIONAL_PACKAGES+=(lz4-devel); fi ;;
    apk)
      REQUIRED_PACKAGES+=(cmake build-base lua5.3 lua5.3-dev pkgconf)
      if [[ "$enable_openmp" == "1" ]]; then REQUIRED_PACKAGES+=(libgomp); fi
      if [[ "$enable_mpi" == "1" ]]; then REQUIRED_PACKAGES+=(openmpi openmpi-dev); fi
      case "$VIZ_BACKEND" in
        SFML) REQUIRED_PACKAGES+=(sfml-dev) ;;
        QT) REQUIRED_PACKAGES+=(qt6-qtbase-dev) ;;
        GTK) REQUIRED_PACKAGES+=(gtk+3.0-dev cairo-dev) ;;
        WEB) REQUIRED_PACKAGES+=(cairo-dev) ;;
      esac
      if [[ "$enable_vulkan" == "1" ]]; then OPTIONAL_PACKAGES+=(vulkan-headers vulkan-loader vulkan-tools glslang); fi
      if [[ "$enable_opencl" == "1" ]]; then OPTIONAL_PACKAGES+=(ocl-icd clinfo opencl-headers); fi
      if [[ "$enable_ffmpeg" == "1" ]]; then OPTIONAL_PACKAGES+=(ffmpeg-dev); fi
      if [[ "$enable_lz4" == "1" ]]; then OPTIONAL_PACKAGES+=(lz4-dev); fi ;;
    xbps)
      REQUIRED_PACKAGES+=(cmake gcc make lua53 lua53-devel pkg-config)
      if [[ "$enable_mpi" == "1" ]]; then REQUIRED_PACKAGES+=(openmpi-devel); fi
      case "$VIZ_BACKEND" in
        SFML) REQUIRED_PACKAGES+=(SFML-devel) ;;
        QT) REQUIRED_PACKAGES+=(qt6-base-devel) ;;
        GTK) REQUIRED_PACKAGES+=(gtk+3-devel cairo-devel) ;;
        WEB) REQUIRED_PACKAGES+=(cairo-devel) ;;
      esac
      if [[ "$enable_vulkan" == "1" ]]; then OPTIONAL_PACKAGES+=(vulkan-headers vulkan-loader vulkan-tools glslang); fi
      if [[ "$enable_opencl" == "1" ]]; then OPTIONAL_PACKAGES+=(ocl-icd clinfo); fi
      if [[ "$enable_ffmpeg" == "1" ]]; then OPTIONAL_PACKAGES+=(ffmpeg-devel); fi
      if [[ "$enable_lz4" == "1" ]]; then OPTIONAL_PACKAGES+=(lz4-devel); fi ;;
    eopkg)
      REQUIRED_PACKAGES+=(cmake gcc lua-devel pkg-config)
      if [[ "$enable_mpi" == "1" ]]; then REQUIRED_PACKAGES+=(openmpi-devel); fi
      case "$VIZ_BACKEND" in
        SFML) REQUIRED_PACKAGES+=(sfml-devel) ;;
        QT) REQUIRED_PACKAGES+=(qt6-base-devel) ;;
        GTK) REQUIRED_PACKAGES+=(gtk3-devel cairo-devel) ;;
        WEB) REQUIRED_PACKAGES+=(cairo-devel) ;;
      esac
      if [[ "$enable_vulkan" == "1" ]]; then OPTIONAL_PACKAGES+=(vulkan-headers vulkan-loader vulkan-tools glslang); fi
      if [[ "$enable_opencl" == "1" ]]; then OPTIONAL_PACKAGES+=(ocl-icd clinfo); fi
      if [[ "$enable_ffmpeg" == "1" ]]; then OPTIONAL_PACKAGES+=(ffmpeg-devel); fi
      if [[ "$enable_lz4" == "1" ]]; then OPTIONAL_PACKAGES+=(lz4-devel); fi ;;
    brew)
      REQUIRED_PACKAGES+=(cmake lua pkg-config)
      if generator_needs_ninja; then REQUIRED_PACKAGES+=(ninja); fi
      if [[ "$enable_openmp" == "1" ]]; then REQUIRED_PACKAGES+=(libomp); fi
      if [[ "$enable_mpi" == "1" ]]; then REQUIRED_PACKAGES+=(open-mpi); fi
      case "$VIZ_BACKEND" in
        SFML) REQUIRED_PACKAGES+=(sfml) ;;
        QT) REQUIRED_PACKAGES+=(qt) ;;
        GTK) REQUIRED_PACKAGES+=(gtk+3 cairo) ;;
        WEB) REQUIRED_PACKAGES+=(cairo) ;;
      esac
      if [[ "$enable_vulkan" == "1" ]]; then OPTIONAL_PACKAGES+=(vulkan-headers vulkan-loader vulkan-tools glslang); fi
      if [[ "$enable_opencl" == "1" ]]; then OPTIONAL_PACKAGES+=(opencl-headers); fi
      if [[ "$enable_ffmpeg" == "1" ]]; then OPTIONAL_PACKAGES+=(ffmpeg); fi
      if [[ "$enable_lz4" == "1" ]]; then OPTIONAL_PACKAGES+=(lz4); fi ;;
    pkg|pkgin|pkg_add)
      REQUIRED_PACKAGES+=(cmake lua53 pkgconf)
      if [[ "$CMAKE_GENERATOR" == "Ninja" ]]; then REQUIRED_PACKAGES+=(ninja); fi
      if [[ "$enable_openmp" == "1" ]]; then REQUIRED_PACKAGES+=(libomp); fi
      if [[ "$enable_mpi" == "1" ]]; then REQUIRED_PACKAGES+=(openmpi); fi
      case "$VIZ_BACKEND" in
        SFML) REQUIRED_PACKAGES+=(sfml3) ;;
        QT) REQUIRED_PACKAGES+=(qt6-base) ;;
        GTK) REQUIRED_PACKAGES+=(gtk3 cairo) ;;
        WEB) REQUIRED_PACKAGES+=(cairo) ;;
      esac
      if [[ "$enable_vulkan" == "1" ]]; then OPTIONAL_PACKAGES+=(vulkan-headers vulkan-loader vulkan-tools glslang); fi
      if [[ "$enable_opencl" == "1" ]]; then OPTIONAL_PACKAGES+=(ocl-icd clinfo); fi
      if [[ "$enable_ffmpeg" == "1" ]]; then OPTIONAL_PACKAGES+=(ffmpeg); fi
      if [[ "$enable_lz4" == "1" ]]; then OPTIONAL_PACKAGES+=(lz4); fi ;;
    emerge)
      REQUIRED_PACKAGES+=(dev-build/cmake sys-devel/gcc dev-lang/lua dev-build/pkgconf)
      if [[ "$enable_mpi" == "1" ]]; then REQUIRED_PACKAGES+=(sci-libs/openmpi); fi
      case "$VIZ_BACKEND" in
        SFML) REQUIRED_PACKAGES+=(media-libs/sfml) ;;
        QT) REQUIRED_PACKAGES+=(dev-qt/qtbase) ;;
        GTK) REQUIRED_PACKAGES+=(x11-libs/gtk+:3 x11-libs/cairo) ;;
        WEB) REQUIRED_PACKAGES+=(x11-libs/cairo) ;;
      esac
      if [[ "$enable_vulkan" == "1" ]]; then OPTIONAL_PACKAGES+=(dev-util/vulkan-headers media-libs/vulkan-loader dev-util/vulkan-tools dev-util/glslang); fi
      if [[ "$enable_opencl" == "1" ]]; then OPTIONAL_PACKAGES+=(dev-libs/ocl-icd dev-util/clinfo); fi
      if [[ "$enable_ffmpeg" == "1" ]]; then OPTIONAL_PACKAGES+=(media-video/ffmpeg); fi
      if [[ "$enable_lz4" == "1" ]]; then OPTIONAL_PACKAGES+=(app-arch/lz4); fi ;;
    *)
      echo "[warn] Pas de correspondance de paquets pour $PACKAGE_MANAGER; les dépendances doivent être installées manuellement."
      return 1 ;;
  esac
  if generator_needs_ninja && [[ "$PACKAGE_MANAGER" != "msys2-pacman" && "$PACKAGE_MANAGER" != "brew" && "$PACKAGE_MANAGER" != "pkg" && "$PACKAGE_MANAGER" != "pkgin" && "$PACKAGE_MANAGER" != "pkg_add" ]]; then
    case "$PACKAGE_MANAGER" in
      apt) REQUIRED_PACKAGES+=(ninja-build) ;;
      dnf|yum|zypper|pacman|apk|xbps|eopkg) REQUIRED_PACKAGES+=(ninja) ;;
    esac
  fi
}

if [[ "$SKIP_SYSTEM_DEPS" == "1" ]]; then
  echo "[1/4] Dépendances système ignorées (--skip-system-deps)."
else
  if [[ "$PACKAGE_MANAGER" == "none" ]]; then
    echo "[1/4] Aucun gestionnaire de paquets pris en charge sur $HOST_OS; dépendances à installer manuellement."
  else
    echo "[1/4] Installation des dépendances système via $PACKAGE_MANAGER..."
    build_package_lists
    if [[ "$PACKAGE_MANAGER" == "apt" ]]; then run_as_admin apt-get update; fi
    install_packages required "${REQUIRED_PACKAGES[@]}"
    if ! install_packages optional "${OPTIONAL_PACKAGES[@]}"; then
      echo "[warn] Certaines dépendances facultatives n'ont pas été installées; CMake désactivera les composants indisponibles."
    fi
  fi
fi

echo "[2/4] Vérification des outils..."
case "$VIZ_BACKEND" in
  QT) gui_modules=(cairo) ;;
  GTK) gui_modules=(gtk+-3.0 cairo) ;;
  WEB) gui_modules=(cairo) ;;
  *) gui_modules=() ;;
esac
for module in "${gui_modules[@]}"; do
  if ! command -v pkg-config >/dev/null 2>&1 || ! pkg-config --exists "$module"; then
    echo "[error] Dépendance GUI manquante pour $VIZ_BACKEND: $module. Relance sans --skip-system-deps ou installe son paquet de développement." >&2
    exit 1
  fi
  echo "[ok] GUI $VIZ_BACKEND: $module $(pkg-config --modversion "$module")"
done
if [[ "$VIZ_BACKEND" == "SFML" ]]; then
  SFML_DETECTED_VERSION="$(detect_sfml_version)"
  if [[ -n "$SFML_DETECTED_VERSION" ]]; then
    echo "[info] SFML $SFML_DETECTED_VERSION annoncé par pkg-config; CMake vérifiera que la version 3+ est trouvable."
  else
    echo "[note] Version SFML non détectable par pkg-config; CMake vérifiera SFML 3 via son paquet de configuration."
  fi
fi
if ! command -v cmake >/dev/null 2>&1; then
  echo "[error] CMake est introuvable; installe-le ou utilise un gestionnaire reconnu par config.sh." >&2
  exit 1
fi
if generator_needs_ninja && ! command -v ninja >/dev/null 2>&1; then
  echo "[error] Le générateur demandé ($CMAKE_GENERATOR) exige la commande ninja." >&2
  exit 1
fi
cmake --version | head -n 1
if command -v c++ >/dev/null 2>&1; then
  c++ --version | head -n 1 || true
elif command -v g++ >/dev/null 2>&1; then
  g++ --version | head -n 1 || true
elif command -v clang++ >/dev/null 2>&1; then
  clang++ --version | head -n 1 || true
elif command -v cl >/dev/null 2>&1; then
  cl 2>&1 | head -n 1 || true
else
  echo "[error] Aucun compilateur C++ détecté (c++, g++, clang++ ou cl)." >&2
  exit 1
fi
for lua_command in lua lua5.4 lua5.3; do
  if command -v "$lua_command" >/dev/null 2>&1; then
    "$lua_command" -v || true
    break
  fi
done
if [[ "$(feature_value ENABLE_VULKAN)" == "1" ]]; then
  command -v glslangValidator >/dev/null 2>&1 && glslangValidator --version | head -n 1 || true
fi
if [[ "$(feature_value ENABLE_OPENCL)" == "1" ]]; then
  # Match the AMD Rusticl activation used by run_mimir.sh. Preserve an explicit
  # environment override, including an empty value used to disable drivers.
  if [[ "$HOST_OS" == "linux" && -z "${RUSTICL_ENABLE+x}" ]]; then
    for vendor_file in /sys/class/drm/card*/device/vendor; do
      [[ -r "$vendor_file" ]] || continue
      if [[ "$(<"$vendor_file")" == "0x1002" ]]; then
        export RUSTICL_ENABLE=radeonsi
        echo "[note] Rusticl AMD: RUSTICL_ENABLE=radeonsi pour la détection OpenCL."
        break
      fi
    done
  fi
  command -v clinfo >/dev/null 2>&1 && clinfo --list || true
fi
if [[ "$(feature_value MIMIR_AUTO_DETECT_GPU_BACKENDS)" == "1" ]]; then
  echo "[note] Auto-détection GPU activée: CMake recherchera CUDA et ROCm/HIP."
elif [[ "$(feature_value ENABLE_CUDA)" == "1" ]]; then
  echo "[note] CUDA activé: installe CUDA Toolkit si nécessaire (non installé automatiquement ici)."
fi
if [[ "$(feature_value MIMIR_AUTO_DETECT_GPU_BACKENDS)" != "1" && "$(feature_value ENABLE_ROCM)" == "1" ]]; then
  echo "[note] ROCm activé: installe ROCm/HIP si nécessaire (non installé automatiquement ici)."
fi

echo "[3/4] Configuration CMake..."

CMAKE_ARGS=(
  -S "$WORKROOT"
  -B "$BUILD_DIR"
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
  -DMIMIR_VIZ_BACKEND="$VIZ_BACKEND"
  -DENABLE_SFML="$(cmake_bool "$ENABLE_SFML_VALUE")"
  -DBUILD_EXAMPLES=OFF
  -DMIMIR_ENABLE_LEGACY_PARAMS=OFF
)

for i in "${!FEATURE_KEYS[@]}"; do
  key="${FEATURE_KEYS[$i]}"
  CMAKE_ARGS+=("-D${key}=$(cmake_bool "${FEATURE_VALUES[$i]}")")
done

if [[ -n "$CMAKE_GENERATOR" ]]; then CMAKE_ARGS+=(-G "$CMAKE_GENERATOR"); fi
if [[ "$PACKAGE_MANAGER" == "brew" ]]; then
  brew_prefix="$(brew --prefix 2>/dev/null || true)"
  if [[ -n "$brew_prefix" ]]; then
    brew_prefixes="$brew_prefix"
    if [[ -n "${CMAKE_PREFIX_PATH:-}" ]]; then
      brew_prefixes="$brew_prefixes;$(printf '%s' "$CMAKE_PREFIX_PATH" | tr ':' ';')"
    fi
    CMAKE_ARGS+=("-DCMAKE_PREFIX_PATH=$brew_prefixes")
  fi
  if [[ "$(feature_value ENABLE_OPENMP)" == "1" ]]; then
    openmp_root="$(brew --prefix libomp 2>/dev/null || true)"
    if [[ -n "$openmp_root" ]]; then CMAKE_ARGS+=("-DOpenMP_ROOT=$openmp_root"); fi
  fi
fi
CMAKE_ARGS+=("${EXTRA_CMAKE_ARGS[@]}")
cmake "${CMAKE_ARGS[@]}"

echo "[4/4] Préparation terminée."
echo "Build prêt dans: $BUILD_DIR"
if command -v nproc >/dev/null 2>&1; then
  BUILD_JOBS="$(nproc)"
elif command -v sysctl >/dev/null 2>&1; then
  BUILD_JOBS="$(sysctl -n hw.ncpu 2>/dev/null || printf 1)"
elif command -v getconf >/dev/null 2>&1; then
  BUILD_JOBS="$(getconf _NPROCESSORS_ONLN 2>/dev/null || printf 1)"
else
  BUILD_JOBS=1
fi
echo "Commande de compilation: cmake --build \"$BUILD_DIR\" --parallel $BUILD_JOBS"
