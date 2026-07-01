# Internals — index étendu

## Pour qui

Développeur avancé qui modifie le moteur C/C++.

## Objectif

Comprendre le fonctionnement interne exact des composants runtime.

## Avant de commencer

Connaître les bases C++ et la structure du dépôt.

## Résultat attendu

Tu peux modifier le code interne en limitant les régressions.


Cette section regroupe la documentation **côté C++** (runtime, données, mémoire, perf) en restant alignée sur le code. Le style est volontairement proche de `10-Model-Class.md` : “source de vérité”, responsabilités, flux, invariants, pièges.

## Pages existantes (déjà présentes)

- `docs/04-Architecture-Internals/01-Engine-Overview.md`
- `docs/04-Architecture-Internals/02-Memory.md`
- `docs/04-Architecture-Internals/03-Hardware-Backends.md`
- `docs/04-Architecture-Internals/10-Model-Class.md`
- `docs/04-Architecture-Internals/11-Helpers.md`

## Nouvelles pages (extension)

### Cœur data / compute

- `docs/04-Architecture-Internals/12-Tensor-Storage.md` — le type `tensor` (storage), l’allocation dynamique, et les implications runtime.
- `docs/04-Architecture-Internals/13-Autograd-Gradients.md` — comment le backward est fait “best-effort”, ce qui est snapshoté, et comment les gradients sont stockés.
- `docs/04-Architecture-Internals/14-Layers-And-Ops.md` — `Layer`, `LayerType`, `LayerOps`, layouts des poids, conventions et gotchas.

### Runtime “outillage” (monitoring / UI)

- `docs/04-Architecture-Internals/04-Monitoring-Htop-Visualizer.md` — `HtopDisplay`, `Visualizer` (SFML) et `AsyncMonitor`.

### Mémoire “avancée” (best-effort)

- `docs/04-Architecture-Internals/05-AdvancedRAMManager.md` — `AdvancedRAMManager` (cache/compression/spill disque).

### Modules (première passe)

Les pages ci-dessous existent déjà et seront enrichies au fil des passes (plus de détails, invariants, exemples et cross-links).

- `docs/04-Architecture-Internals/15-Serialization-Internals.md`
- `docs/04-Architecture-Internals/16-Tokenizer-Encoder-Internals.md`
- `docs/04-Architecture-Internals/17-Lua-Bindings-Internals.md`
- `docs/04-Architecture-Internals/18-RuntimeAllocator-And-Scratchpads.md`
- `docs/04-Architecture-Internals/19-Models-Registry-And-Builders.md`
- `docs/04-Architecture-Internals/20-CLI-EntryPoints.md`

### GPU Runtimes (CUDA / ROCm)

- `docs/04-Architecture-Internals/21-GPU-Runtimes.md` — implémentation interne des fast-paths cuBLAS/rocBLAS (im2col, hybride Norm, Attention multi-SGEMM, DeviceBuf, pattern de dispatch).

### Execution Planner

- `docs/04-Architecture-Internals/22-Planning.md` — analyse des durées de vie des tenseurs, fusions Conv2d+ReLU, scratchpad pré-alloué.

## Convention

- Quand un comportement diverge entre “int path” (Embedding/ids) et “float path”, la doc l’indique explicitement.
- Quand une structure est **legacy** mais encore utilisée (ex: le type `tensor` historique), la doc explique pourquoi elle existe et ce qu’il ne faut pas faire.
