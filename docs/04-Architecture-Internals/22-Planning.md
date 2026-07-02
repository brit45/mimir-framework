# Internals : Execution Planner (C++)

## Pour qui

Développeur avancé qui modifie le moteur C/C++.

## Objectif

Comprendre le fonctionnement interne exact des composants runtime.

## Avant de commencer

Connaître les bases C++ et la structure du dépôt.

## Résultat attendu

Tu peux modifier le code interne en limitant les régressions.

Cette page documente le système de planification statique du graphe d'exécution.

Source de vérité : `src/Planning/Planner.hpp`.

Intégration runtime : `src/Model.cpp` (forward) et `Model::static_plan_` dans `src/Model.hpp`.

---

## Rôle

Le `Planner` analyse la liste des layers **avant** l'exécution pour :

1. Calculer les durées de vie des tenseurs (quand chaque tenseur est créé/consommé en dernière fois)
2. Identifier les fusions de layers possibles (Conv2d+ReLU et fusions génériques)
3. Pré-calculer les tailles de buffers scratch nécessaires pour Conv2d

Ces informations permettent de :

- Allouer des activations de façon optimale (réutilisation mémoire)
- Éviter des passes intermédiaires (fusions)

---

## 1) Analyse des durées de vie (`TensorLifetime`)

```cpp
struct TensorLifetime {
    int first_def;          // Index du layer qui crée le tenseur
    int last_use;           // Index du layer qui le consomme en dernier
    std::string dtype;      // Type de données
    std::vector<int> shape; // Forme du tenseur
};

std::unordered_map<std::string, TensorLifetime>
analyze_tensor_lifetimes(const std::vector<Layer>& layers);
```

Utilisation : déterminer quels buffers peuvent être aliasés/réutilisés.

---

## 2) Fusions de layers (`FusionKind`)

```cpp
enum class FusionKind : uint8_t {
    NONE = 0,
    CONV2D_RELU = 1,
    GENERIC_ACTIVATION = 2,
    GENERIC_SPLIT = 3,
    GENERIC_CHUNK = 4,
    GENERIC_ACTIVATION_SPLIT = 5,
    GENERIC_ACTIVATION_CHUNK = 6,
    GENERIC_UNARY_SHAPE = 7,
};
```

Fusions actuellement implémentées (inférence uniquement, pas en training) :

- Conv2d + ReLU
- Producer + activation standalone (ReLU/SiLU/GELU/...)
- Producer + unary shape (`Flatten`, `Reshape`, `View`, `Transpose`, `Permute`, `Squeeze`, `Unsqueeze`, `Identity`)
- Producer + `Split`/`Chunk`
- Producer + activation + `Split`/`Chunk`

---

## 3) Plan d'exécution (`ExecutionPlan`)

```cpp
struct PlannedOp {
    int       layer_index;  // Index dans la liste des layers
    FusionKind fusion;      // Fusion applicable (ou NONE)
};

struct ExecutionPlan {
    std::vector<PlannedOp>  ops;
    std::vector<uint8_t>    skip_layer;
    std::vector<uint8_t>    fuse_relu_for_conv2d;
    std::vector<int>        fuse_activation_consumer;
    std::vector<int>        fuse_unary_consumer;
    std::vector<int>        fuse_split_consumer;
    std::vector<uint8_t>    fuse_split_kind; // 0=none, 1=Split, 2=Chunk
};

ExecutionPlan build_execution_plan_static(
    const std::vector<Layer>& layers,
    bool training);
```

`build_execution_plan_static(...)` est conservateur : en mode `training=true`, les fusions sont désactivées.

`skip_layer` marque les consommateurs fusionnés qui ne doivent plus être exécutés comme ops indépendantes dans la boucle de forward.

---

## 4) Scratchpad pour Conv2d (`Conv2dScratchPlan`)

```cpp
struct Conv2dScratchPlan {
    size_t wT_bytes;   // Taille du buffer transpose de poids W^T
    size_t xcol_bytes; // Taille du buffer im2col
    size_t c_bytes;    // Taille du buffer de sortie
};

Conv2dScratchPlan plan_conv2d_fastpath_scratch(
    const std::vector<Layer>& layers);
```

Retourne la taille maximale nécessaire sur l'ensemble des layers Conv2d du modèle. Permet de pré-allouer un scratchpad unique réutilisé sur tous les Conv2d, évitant les allocations dynamiques pendant le forward.

---

## 5) Intégration avec le runtime

Le plan est géré via un cache `static_plan_` dans `Model`.

Pendant le forward :

- `MIMIR_ENABLE_PLANNER` (défaut `1`) active la planification statique.
- `MIMIR_ENABLE_FUSION` (défaut `1`) active l'application des fusions issues du plan.
- `MIMIR_ENABLE_FUSION_TRAIN` (défaut `0`) autorise uniquement `Conv2d+ReLU` en mode `training=true` (opt-in). Les fusions génériques restent inférence-only.
- Le plan est (re)construit si nécessaire s'il n'a jamais été construit, si le mode `training` change, ou si le nombre d'ops planifiées ne correspond plus au nombre de layers.
- `MIMIR_PLANNER_DUMP=1` déclenche un dump des stats planner (durées de vie, fusions, scratch Conv2d).

Exemple prêt à copier/coller :

```bash
export MIMIR_ENABLE_PLANNER=1
export MIMIR_ENABLE_FUSION=1
export MIMIR_ENABLE_FUSION_TRAIN=1
export MIMIR_PLANNER_DUMP=1
```

Le forward utilise ensuite le plan pour :

- sauter les layers marqués `skip_layer`,
- appliquer les fusions au moment du traitement de l'output,
- garder un chemin d'exécution stable entre passes tant que les conditions ci-dessus ne changent pas.

En pratique, la reconstruction n'est pas liée à un appel `build()` explicite : elle est pilotée par les invariants vérifiés dans le forward.
