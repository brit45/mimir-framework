# Installation et compilation

Mímir nécessite CMake 3.15+, un compilateur C++17 et Lua 5.3+ avec ses fichiers
de développement. Les options ci-dessous correspondent à `CMakeLists.txt`,
`config.sh` et `CMakePresets.json` du dépôt. La présence d'un chemin de build
pour une plateforme ne remplace pas une exécution de ses tests sur cette plateforme.

## Dépendances et fonctionnalités

| Dépendance | Usage | Option CMake |
| --- | --- | --- |
| CMake 3.15+, compilateur C++17, Lua 5.3+ | Compilation et scripting natif | Obligatoires |
| OpenMP | Parallélisme CPU | `ENABLE_OPENMP`, activé par défaut |
| SFML **3** | Interface native par défaut | `MIMIR_VIZ_BACKEND=SFML` |
| Qt / GTK / hôte Web | Autres interfaces | `MIMIR_VIZ_BACKEND=QT\|GTK\|WEB`, hôtes Linux actuels |
| Vulkan et glslangValidator | Calcul et compilation des shaders SPIR-V | `ENABLE_VULKAN`, activé par défaut |
| OpenCL | Calcul OpenCL | `ENABLE_OPENCL`, activé par défaut |
| CUDA Toolkit / ROCm | Calcul NVIDIA / AMD | `ENABLE_CUDA` / `ENABLE_ROCM`, désactivés par défaut |
| LZ4 | Compression disponible dans les chemins qui l'utilisent | `ENABLE_LZ4`, activé si trouvé |
| FFmpeg | Chargement audio et vidéo | `ENABLE_FFMPEG`, activé si trouvé |
| MPI et ses en-têtes C++ | Collectives DDP | `ENABLE_MPI`, désactivé par défaut |

Le nom du paquet SFML dépend du système et de ses dépôts : vérifiez qu'il
fournit la version **3**, exigée par `find_package(SFML 3 ...)`. Un paquet
SFML 2 ne satisfait pas cette dépendance. Le profil `NONE` évite toute GUI.
OpenMP est optionnel ; un build CPU sans OpenMP reste possible.

## Premier build CPU sans interface

Les commandes shell supposent un environnement POSIX et des dépendances déjà
installées. Sur Debian/Ubuntu, la base est par exemple :

```bash
sudo apt-get update
sudo apt-get install cmake build-essential lua5.3 liblua5.3-dev
```

Depuis la racine du dépôt :

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DMIMIR_VIZ_BACKEND=NONE \
  -DENABLE_VULKAN=OFF -DENABLE_OPENCL=OFF \
  -DENABLE_CUDA=OFF -DENABLE_ROCM=OFF \
  -DMIMIR_ENABLE_TESTS=ON
cmake --build build --parallel 2
./run_mimir.sh --help
./run_mimir.sh --lua scripts/tests/test_serialization_smoke.lua
```

Ce smoke test vérifie la création et la sérialisation d'un petit modèle. Il
ne valide pas toutes les architectures ni tous les backends.
`MIMIR_ENABLE_TESTS=ON` est l'option du projet qui crée les cibles CTest ;
`BUILD_TESTING=ON` seul ne les active pas.

```bash
ctest --test-dir build --output-on-failure
```

### Emplacement des exécutables

Le build principal `build/` produit `bin/mimir` dans le dépôt. Par défaut, un
build secondaire comme `build-cpu/` produit **`build-cpu/bin/mimir`**.
`MIMIR_RUNTIME_OUTPUT_DIRECTORY` permet de modifier cette destination ; les
générateurs multi-configuration peuvent ajouter `Release` ou `Debug`.

Pour utiliser un build secondaire avec le lanceur :

```bash
MIMIR_EXECUTABLE="$PWD/build-cpu/bin/mimir" ./run_mimir.sh --help
```

N'utilisez ce chemin qu'après avoir effectivement construit ce profil.

## Configuration assistée et portable

```bash
./config.sh --help
```

`config.sh` détecte l'OS et le gestionnaire de paquets, puis configure CMake.
Sans `--skip-system-deps`, il peut installer les dépendances et demander des
droits administrateur. Les chemins prévus couvrent Linux, macOS, plusieurs BSD
et MSYS2/MinGW ; chaque combinaison de système, SDK et compilateur doit être
validée séparément.

Exemple reproductible si les dépendances sont déjà présentes :

```bash
./config.sh --gui NONE --non-interactive --skip-system-deps \
  --build-dir build-cpu --generator Ninja \
  --enable MIMIR_ENABLE_TESTS --amp --no-ddp \
  --disable ENABLE_VULKAN --disable ENABLE_OPENCL
cmake --build build-cpu --parallel 2
```

| Option du script | Effet |
| --- | --- |
| `--gui SFML\|QT\|GTK\|WEB\|NONE` | Sélection de l'interface, indépendante du runtime de calcul |
| `--enable FEATURE` / `--disable FEATURE` | Valeur explicite d'une option CMake |
| `--amp` / `--no-amp` | Active / désactive `ENABLE_AMP` |
| `--ddp` / `--no-ddp` | Active / désactive `ENABLE_MPI` |
| `--build-dir CHEMIN`, `--generator Ninja` | Arbre de build et générateur |
| `--build-type Debug` | Configuration de débogage |
| `--cmake-arg ARG` | Argument CMake supplémentaire |
| `--skip-system-deps` | Saute l'installation des paquets système |
| `WORKROOT=/chemin/source` | Racine source explicite |

Les variables d'environnement `ENABLE_VULKAN=1` ou `ENABLE_OPENCL=0` ne
remplacent pas les flags `--enable` / `--disable` du script.

## Variantes par plateforme

### macOS

Avec Homebrew, installez CMake et Lua, puis configurez le profil headless
ci-dessus. OpenMP nécessite généralement `libomp` ; utilisez
`-DENABLE_OPENMP=OFF` si vous n'en disposez pas. Le dispatch SIMD est protégé
par l'architecture de compilation : un hôte ARM n'exécute pas des instructions
x86. La disponibilité Vulkan/OpenCL dépend des bibliothèques et pilotes de l'hôte.

### Windows natif

CMake possède des branches MSVC. Installez un compilateur C++17 et une
bibliothèque Lua compatible avec ce compilateur. Exemple PowerShell headless,
avec chemins Lua à adapter :

```powershell
cmake -S . -B build -DMIMIR_VIZ_BACKEND=NONE `
  -DENABLE_VULKAN=OFF -DENABLE_OPENCL=OFF `
  -DENABLE_SCRIPTING_JS=OFF `
  -DLUA_INCLUDE_DIR="C:/Lua/include" -DLUA_LIBRARY="C:/Lua/lib/lua53.lib"
cmake --build build --config Release --parallel 2
```

Le bridge JavaScript utilise des primitives POSIX et est désactivé par défaut
sous Windows. Les runtimes GPU ont leurs propres SDK et contraintes ; ne pas
assimiler une option CMake à une validation matérielle. `config.sh` et
`run_mimir.sh` sont des scripts Bash, utilisables depuis un environnement
adapté tel que MSYS2, pas des commandes PowerShell natives.

### Google Colab

Choisissez un environnement GPU, clonez le dépôt sous `/content/tensor-2` et
installez les dépendances de compilation du preset :

```python
!nvidia-smi
!apt-get update -qq
!apt-get install -y ninja-build build-essential pkg-config liblua5.3-dev liblz4-dev libavcodec-dev libavformat-dev libavutil-dev libswresample-dev libswscale-dev
%cd /content/tensor-2
!cmake --preset colab-cuda
!cmake --build --preset build-colab-cuda
!./build-colab/bin/mimir --version
```

Le preset active la détection automatique CUDA/ROCm et désactive la GUI,
Vulkan, OpenCL et les tests. Vérifiez le résumé CMake et le périphérique
sélectionné à l'exécution. L'auto-détection ne fournit pas un toolkit absent.
Les presets nécessitent une version de CMake qui prend en charge leur schéma,
supérieure au minimum du build manuel ; voir `CMakePresets.json`.

## Options CMake utiles

Ces valeurs sont des **arguments à ajouter à la commande `cmake -S ... -B ...`** :

| Argument | Effet |
| --- | --- |
| `-DENABLE_SIMD=OFF` | Désactive les chemins SIMD pour un diagnostic |
| `-DMIMIR_PORTABLE_BUILD=ON` | Profil portable, valeur par défaut |
| `-DENABLE_OPENMP=OFF` | Désactive le parallélisme OpenMP |
| `-DMIMIR_VIZ_BACKEND=NONE` | Compile sans interface graphique |
| `-DENABLE_CUDA=ON` | Demande CUDA ; toolkit requis |
| `-DENABLE_ROCM=ON` | Demande ROCm ; toolkit requis |
| `-DMIMIR_AUTO_DETECT_GPU_BACKENDS=ON` | Détecte les toolkits CUDA/ROCm disponibles |
| `-DENABLE_AMP=ON` | Compile le support d'autocast, activé par défaut |
| `-DENABLE_MPI=ON` | Compile DDP, bibliothèque MPI requise |
| `-DMIMIR_ENABLE_TESTS=ON` | Active les cibles de tests |
| `-DBUILD_EXAMPLES=ON` | Ajoute les exécutables d'exemple |
| `-DBUILD_MIMIR_STATIC=ON` | Tente un binaire Linux statique ; versions statiques des dépendances requises |

Le CPU choisit les kernels disponibles à l'exécution après détection des
capacités et du support des registres par l'OS : scalaire, SSE2, AVX/256,
AVX2, AVX-512, avec variantes FMA/F16C selon le chemin. Il n'est pas nécessaire
de désactiver tout SIMD parce qu'une machine ne possède pas AVX2. Voir
[les runtimes](../04-Architecture-Internals/21-GPU-Runtimes.md) pour les limites
et le diagnostic de la sélection.

## Dépannage

| Symptôme | Action |
| --- | --- |
| Lua introuvable | Installer ses en-têtes et bibliothèque de développement ; au besoin fournir `LUA_INCLUDE_DIR` et `LUA_LIBRARY` |
| SFML incompatible | Fournir SFML 3 ou sélectionner `MIMIR_VIZ_BACKEND=NONE` |
| glslangValidator absent | Installer l'outil ; le fetch est proposé si `MIMIR_FETCH_GLSLANG=ON`, sinon les shaders ne sont pas compilés |
| OpenMP indisponible | Installer la bibliothèque correspondante ou définir `ENABLE_OPENMP=OFF` |
| Build lent ou à court de mémoire | Réduire `--parallel` et lire la première erreur |
| Cache ou générateur incompatible | Utiliser un nouveau dossier de build et vérifier le chemin du binaire produit |
| Aucun test trouvé | Reconfigurer avec `MIMIR_ENABLE_TESTS=ON`, puis compiler les tests |

Avant un déploiement, conservez la version, le profil CMake, les dépendances,
la configuration et les résultats des tests du modèle et du backend utilisés.
La [validation de version](../06-Contributing/08-Release-Validation.md) décrit
les contrôles de livraison.

## Étapes suivantes

- [Première exécution](01-Quick-Start.md)
- [CLI et arguments des scripts](03-CLI.md)
- [Compilation, AMP et DDP](../08-Tuto/16-Tuto-Compilation-Portabilite-AMP-DDP.md)
