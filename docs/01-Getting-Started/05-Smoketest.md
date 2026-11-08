# Valider l’installation

Vérifier en 5 minutes que l'environnement est sain.

**Public concerné :** Débutant et contributeur.

> **Prérequis**
>
> Build effectué au moins une fois.

Ce smoketest vérifie que le build, la CLI et les chemins essentiels — Lua,
passe avant et sérialisation — fonctionnent sur votre machine.

## 1) Préparer les tests

```bash
cmake -S . -B build -DMIMIR_ENABLE_TESTS=ON
cmake --build build --parallel 2
```

## 2) Vérifier la CLI

```bash
./bin/mimir --help
```

## 3) Exécuter un exemple minimal (Lua)

```bash
./run_mimir.sh --lua scripts/examples/inspect_vae_conv.lua
```

## 4) Tester une surface API un peu plus large

### Registre

```bash
./bin/mimir --lua scripts/tools/inspect_architectures.lua -- -a
```

### Sérialisation

```bash
./bin/mimir --lua scripts/tests/test_serialization_smoke.lua
```

## 5) Vérifier les contrats natifs sans dataset

Ces tests utilisent des entrées synthétiques :

```bash
ctest --test-dir build --output-on-failure \
  -R '^ModelTest\.(VAEConvContract|VAEConvLogvar)$'
```

Le script `test_vae_conv_generate.lua` demande un checkpoint et une image :
il convient à une validation d'inférence préparée, pas à ce parcours sans données.

## 6) (Optionnel) Exécuter toute la suite CTest

Si votre build a généré des targets de test :

```bash
ctest --test-dir build --output-on-failure
```

## Notes

- Si vous voyez des refus MemoryGuard/OOM, lis `docs/02-User-Guide/09-Memory.md`.
- Les tests GPU peuvent être ignorés si leur backend ou matériel est absent.
  Un smoketest CPU ne valide pas les autres runtimes.
- Pour mesurer les performances, choisissez un workload dans le
  [guide des benchmarks](../05-Advanced/01-Performance.md) ; un benchmark n'est pas un test de démarrage.

## Étapes suivantes

- [Page précédente : Organisation du dépôt](04-Repo-Layout.md)
- [Index de la documentation](../00-INDEX.md)
- [Page suivante : Parcours par niveau de compétence](06-Learning-Paths.md)
