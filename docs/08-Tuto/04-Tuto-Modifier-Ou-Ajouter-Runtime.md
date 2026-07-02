# Tuto - Modifier ou ajouter un runtime

## Pour qui

Profil avance qui touche au moteur d'execution C++.

## Objectif

Modifier un runtime existant ou en ajouter un nouveau sans casser le fallback.

## Avant de commencer

1. Lecture conseillee: [docs/07-Devs/04-Runtime-Development.md](../07-Devs/04-Runtime-Development.md).
2. Comprendre le contrat runtime.
3. Savoir compiler et tester rapidement.

## Résultat attendu

Le runtime respecte le contrat, retourne `false` quand non supporte, et n'introduit pas de regression.

## Etape 1 - Comprendre le contrat

Fichier de reference:
- `src/runtimes/AbstractRuntime.hpp`

Methodes critiques:
1. `initialize(...)`
2. `shutdown()`
3. `isInitialized()`
4. `linearForward(...)`
5. `forwardLayer(...)`

## Etape 2 - Modifier un runtime existant

Checklist simple:
1. garder la signature API,
2. verifier shapes et dtypes,
3. si non supporte: retourner `false`,
4. ajouter des logs utiles en mode verbose,
5. valider resultat numerique vs reference CPU.

## Etape 3 - Ajouter un nouveau runtime

Procedure pas-a-pas:
1. creer un dossier backend: `src/runtimes/<backend>/`,
2. creer une classe derivee de `AbstractRuntime`,
3. implementer init/shutdown/isInitialized,
4. implementer au minimum une op stable,
5. brancher la config runtime,
6. integrer au build CMake,
7. tester fallback + non-regression.

## Etape 4 - Test minimal obligatoire

1. Build debug + release.
2. Test op supportee (resultat attendu).
3. Test op non supportee (retour `false`).
4. Test run script standard sans crash.

Commande utile:

```bash
./bin/mimir --lua scripts/benchmarks/benchmark_official.lua -- --safe --iters 1
```

## Etape 5 - Definition de done

Ton travail est termine si:
1. les tests de base passent,
2. le fallback fonctionne,
3. les logs sont actionnables,
4. aucune regression visible sur scripts smoke.

## Exemple pratique

### Contexte

Tu ajoutes un fast path pour une operation simple. Si le runtime ne peut pas traiter le cas, il doit rendre la main proprement.

### Code commente

```cpp
bool MyRuntime::forwardLayer(
	const std::vector<const std::vector<float>*>& inputs,
	std::vector<std::vector<float>>& outputs,
	const Layer& layer,
	bool training) {
	// 1) Runtime non pret => fallback vers un autre runtime.
	if (!isInitialized()) return false;

	// 2) On ne traite que l'operation Add dans cet exemple.
	if (layer.type != LayerType::Add) return false;

	// 3) Validation defensive des entrees.
	if (inputs.size() != 2 || !inputs[0] || !inputs[1]) return false;
	const auto& a = *inputs[0];
	const auto& b = *inputs[1];
	if (a.size() != b.size()) return false;

	// 4) Calcul complet de la sortie.
	outputs.resize(1);
	outputs[0].resize(a.size());
	for (size_t i = 0; i < a.size(); ++i) {
		outputs[0][i] = a[i] + b[i];
	}

	// 5) true uniquement si la sortie est totalement produite.
	return true;
}
```

### Explication

1. `false` = je ne sais pas faire, laissez le fallback agir.
2. `true` = sortie complete et exploitable.
3. checks shapes/dtypes avant tout calcul.

### Test rapide

```bash
./bin/mimir --lua scripts/benchmarks/benchmark_official.lua -- --safe --iters 1
```

Verification attendue: pas de crash, et resultat numerique coherent par rapport au chemin de reference CPU.

## Suite

- Runtime dev: [docs/07-Devs/04-Runtime-Development.md](../07-Devs/04-Runtime-Development.md)
- Internals moteur: [docs/04-Architecture-Internals/01-Engine-Overview.md](../04-Architecture-Internals/01-Engine-Overview.md)
