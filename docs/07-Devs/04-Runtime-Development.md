# Runtime - Modifier Et Ajouter Un Backend

## Pour qui

Développeur framework (C/C++/runtime/scripting).

## Objectif

Implémenter ou modifier des briques techniques sans casser le contrat global.

## Avant de commencer

Comprendre le registre d'architectures et les conventions I/O.

## Résultat attendu

Tu peux livrer des évolutions compatibles avec la base existante.


Ce chapitre explique comment modifier un runtime existant ou en ajouter un nouveau.

## Etat des runtimes de scripting

- Lua reste la reference fonctionnelle pour le scripting metier.
- Les runtimes JS, C# et Rust servent de couches de transition et de validation progressive.
- Ils ne doivent pas etre presents comme des remplacements complets de Lua tant que la parite n'est pas atteinte.

## 1. Contrat runtime

Le contrat de base est dans `src/runtimes/AbstractRuntime.hpp`.

Methodes centrales :

- `initialize(const RuntimeConfig&)`
- `shutdown()`
- `isInitialized()`
- `linearForward(...)`
- `forwardLayer(...)`

Et la config runtime est harmonisee par `RuntimeConfig::fromEnv(...)`.

## 2. Modifier un runtime existant

Exemple: optimiser CPU/CUDA/ROCM sur une op donnee.

Checklist :

1. Garder la signature du contrat intacte.
2. Ajouter des gardes sur shapes/dtypes.
3. Retourner `false` si la capa n'est pas supportee (pour fallback).
4. Instrumenter (verbose) sans bruit excessif.
5. Verifier par tests de non-regression.

## 3. Ajouter un nouveau runtime

Procedure type :

1. Creer un dossier backend dans `src/runtimes/<backend>/`.
2. Implementer une classe derivee de `AbstractRuntime`.
3. Brancher `initialize/shutdown/isInitialized`.
4. Implementer au minimum `linearForward` puis `forwardLayer`.
5. Ajouter la config env (`RuntimeConfig::fromEnv` conventions).
6. Integrer le backend au build (CMake) et au chemin d'initialisation runtime.

## 4. Conventions de qualite

- Correction d'abord, optimisation ensuite.
- Fallback explicite et deterministic.
- Pas d'hypothese cachee sur layout sans check.
- Messages d'erreur actionnables.

## 5. Tuning via variables d'environnement

Le pattern general est :

- kill switch backend,
- opt-in par type d'op,
- seuil minimal d'operations,
- index de device,
- mode verbose.

Objectif : activer progressivement les fast-paths sans regressions.

## 6. Matrice de validation minimale

1. Build debug + release.
2. Test unite op cible (exactness).
3. Test graph simple `forwardLayer`.
4. Test fallback quand op non supportee.
5. Test numerique (ecart tolere) vs CPU reference.

## 7. Erreurs frequentes

- retourner `true` alors que la sortie n'est pas complete,
- ecrire hors bornes sur buffers de sortie,
- melanger conventions NCHW/NHWC sans conversion,
- ignorer l'etat `isInitialized`.

## 8. Demo metier - fast-path sur Add avec fallback propre

Exemple conceptuel dans `forwardLayer(...)` :

```cpp
bool MyRuntime::forwardLayer(
	const std::vector<const std::vector<float>*>& inputs,
	std::vector<std::vector<float>>& outputs,
	const Layer& layer,
	bool training) {
	if (!isInitialized()) return false;
	if (layer.type != LayerType::Add) return false;
	if (inputs.size() != 2 || !inputs[0] || !inputs[1]) return false;

	const auto& a = *inputs[0];
	const auto& b = *inputs[1];
	if (a.size() != b.size()) return false;

	outputs.resize(1);
	outputs[0].resize(a.size());
	for (size_t i = 0; i < a.size(); ++i) outputs[0][i] = a[i] + b[i];
	return true;
}
```

Lecture metier :

1. Si le runtime ne sait pas faire, il retourne `false` sans casser le run.
2. Si les shapes ne matchent pas, il refuse proprement.
3. En succes, il produit une sortie complete et deterministic.

## 9. Demo metier - toggles runtime pour rollout progressif

```bash
export MIMIR_ACCEL_VERBOSE=1
export MIMIR_CUDA_LINEAR=1
export MIMIR_CUDA_LINEAR_MIN_OPS=1048576
./bin/mimir --lua scripts/benchmarks/benchmark.lua
```

But metier :

- activer un fast-path progressivement,
- observer son impact,
- garder une marche arriere immediate via variables d'environnement.

## 10. Definition de done pour un nouveau runtime

1. Build propre avec backend active/desactive.
2. Test de correction numerique vs reference CPU.
3. Test de fallback (retour `false`) sur op non supportee.
4. Logs exploitables en mode verbose.
5. Pas de regression sur scripts de smoke existants.
6. Si le runtime est un bridge de scripting non-Lua, documenter explicitement les APIs non encore supportees.
