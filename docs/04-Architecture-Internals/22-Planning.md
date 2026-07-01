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

---

## Rôle

Le `Planner` analyse la liste des layers **avant** l'exécution pour :

1. Calculer les durées de vie des tenseurs (quand chaque tenseur est créé/consommé en dernière fois)
2. Identifier les fusions de layers possibles (ex: Conv2d + ReLU)
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
    NONE     = 0,
    CONV2D_RELU = 1   // Conv2d suivi d'un ReLU
};
```

Pour l'instant, une seule fusion est déclarée : `CONV2D_RELU`. L'enum est extensible pour d'autres fusions futures (Conv2d+BN, Linear+GELU, etc.).

---

## 3) Plan d'exécution (`ExecutionPlan`)

```cpp
struct PlannedOp {
    int       layer_index;  // Index dans la liste des layers
    FusionKind fusion;      // Fusion applicable (ou NONE)
};

struct ExecutionPlan {
    std::vector<PlannedOp>  ops;
    std::vector<uint8_t>    fuse_relu_for_conv2d;
    // fuse_relu_for_conv2d[i] = 1 si le layer i (Conv2d) peut fusionner ReLU suivant
};

ExecutionPlan build_execution_plan_static(
    const std::vector<Layer>& layers,
    bool training);
```

En mode `training=true`, certaines fusions sont désactivées (ex: ReLU doit rester séparable pour le backward).

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

Le plan est construit **une fois** lors de `Model::build()` et stocké dans le modèle. Le forward l'utilise pour :

- Décider si une fusion peut s'appliquer entre le layer courant et le suivant
- Pré-allouer les scratchpads Conv2d

Le plan n'est pas reconstruit si les layers changent après `build()`. Si vous modifiez dynamiquement le modèle, vous devez appeler `build()` à nouveau.
