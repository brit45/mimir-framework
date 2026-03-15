# Backends hardware (CPU / Vulkan / OpenCL)

Mímir est **CPU-first**: le CPU reste le chemin de référence (justesse + compat), mais le runtime peut activer des accélérations optionnelles.

## Sources de vérité (C++)

- `src/HardwareOpt.hpp` (détection CPU)
- `src/Model.hpp` / `src/Model.cpp` (dispatch “use_hardware”)
- `src/VulkanCompute.hpp`
- `src/OpenCLCompute.hpp`
- `shaders/linear_forward.comp` (shader Vulkan)
- `CMakeLists.txt` (compilation SPIR-V best-effort)

## CPU (référence)

### Détection runtime

Le runtime expose des helpers `Model::hasAVX2()`, `hasFMA()`, `hasF16C()`, `hasBMI2()` (cf. `Model.hpp`).

Usage typique:

- choisir une implémentation SIMD vs scalaire
- activer/désactiver certains chemins “fast”

### Parallélisme

OpenMP est utilisé pour le parallélisme (si activé au build). La granularité dépend des ops (layers) et des implémentations.

## Accélération GPU: design général

Le modèle suit un principe simple:

- *si* un backend est disponible et initialisé, et que l’opération le supporte, on peut prendre un chemin accéléré.
- sinon fallback CPU.

Dans `Model.hpp`, plusieurs ops ont une signature “dispatchable” via `use_hardware` (ex: `computeLinear`, `computeConv2D`, `computeAttention`, etc.).

Important:

- “supporte” ne veut pas dire “implémenté”: certaines ops peuvent encore être CPU-only.

## Vulkan Compute (optionnel)

### Build (Vulkan)

- flag CMake: `ENABLE_VULKAN`
- dépendance: `find_package(Vulkan)`

### Compute engine (Vulkan)

Le backend Vulkan est implémenté dans `src/VulkanCompute.hpp` via `VulkanCompute::ComputeEngine`.

Init (simplifié):

- crée `VkInstance`
- choisit le premier device avec `VK_QUEUE_COMPUTE_BIT`
- crée `VkDevice` + récupère la queue compute
- crée un `VkCommandPool`

### Kernel actuellement ciblé (Vulkan): `linearForward`

Le fichier montre un pipeline dédié au kernel Linear:

- `ensureLinearKernel()` prépare pipeline/descriptor sets (voir plus bas dans le fichier)
- `linearForward(input, weights, bias, output, batch, in_f, out_f)`
  - alloue des buffers
  - upload input/weights/bias
  - dispatch compute
  - readback output

Point clé:

- l’allocation est **par appel** (buffers temporaires), donc on est dans une logique “fonctionnelle / simple” plutôt que “pooling agressif”.

### Shaders SPIR-V: compilation + lookup

Le shader attendu pour Linear est `linear_forward.comp.spv`.

Compilation (CMake, best-effort):

- `CMakeLists.txt` tente de trouver `glslangValidator`.
- si absent et `MIMIR_FETCH_GLSLANG=ON`: fetch de `KhronosGroup/glslang` et build local du validator.
- compilation:
  - source: `shaders/linear_forward.comp`
  - output: `${CMAKE_BINARY_DIR}/shaders/linear_forward.comp.spv`
- copie post-build best-effort à côté du binaire: `bin/shaders/linear_forward.comp.spv`

Lookup runtime:

- variable d’environnement: `MIMIR_VULKAN_LINEAR_SPV` (chemin direct)
- sinon candidats relatifs au cwd (ex: `./bin/shaders/linear_forward.comp.spv`, `./build/shaders/...`, etc.).

Conséquence:

- si le SPIR-V n’est pas trouvé, le runtime log et retombe sur CPU.

## OpenCL (optionnel)

### Build (OpenCL)

- flag CMake: `ENABLE_OPENCL`
- dépendance: librairie `OpenCL`

### Compute engine (OpenCL)

Le backend OpenCL est dans `src/OpenCLCompute.hpp` via `OpenCLCompute::ComputeEngine`.

Init (simplifié):

- détecte les plateformes
- choisit un device: GPU d’abord, sinon CPU
- crée `cl_context` + `cl_command_queue`
- compile un petit programme OpenCL embarqué (source string dans le header)

### Kernel actuellement ciblé (OpenCL): `linearForward`

API similaire à Vulkan:

- `linearForward(input, weights, bias_or_null, output, batch, in_f, out_f)`
- alloue des `cl_mem` par appel
- upload input/weights/bias
- `clEnqueueNDRangeKernel` sur une grille 2D `(batch, out_f)`
- readback output

Comme pour Vulkan:

- l’implémentation est volontairement simple (pas de pool/cache de buffers pour l’instant).

## API Lua (contrôle et introspection)

Selon le build, la couche Lua expose:

- `Mimir.Model.hardware_caps()` (capabilités CPU)
- `Mimir.Model.set_hardware(true/false)` (activer/désactiver certains chemins)

Les détails exacts des helpers et flags Lua sont définis dans `src/LuaScripting.cpp`.

## Notes et pièges

- La justesse doit être validée sur CPU d’abord: GPU est un “accélérateur”, pas le chemin de référence.
- La perf GPU peut être limitée par:
  - allocations temporaires par appel
  - transferts host↔device
  - disponibilité/compilation des shaders (Vulkan)
