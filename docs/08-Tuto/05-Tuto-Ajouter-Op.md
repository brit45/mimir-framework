# Tuto - Ajouter une OPs

## Pour qui

Developpeur avance qui veut ajouter une nouvelle operation (OP) au moteur.

## Objectif

Ajouter une OP proprement: definition, execution, integration, puis validation.

## Avant de commencer

1. Lecture conseillee:
- [docs/07-Devs/02-Building-Models-And-Layers.md](../07-Devs/02-Building-Models-And-Layers.md)
- [docs/04-Architecture-Internals/14-Layers-And-Ops.md](../04-Architecture-Internals/14-Layers-And-Ops.md)
2. Comprendre que `Mimir.Layers` est surtout un module de stubs pour des appels standalone (voir [docs/03-API-Reference/18-Layers-Module.md](../03-API-Reference/18-Layers-Module.md)).

## Résultat attendu

Ta nouvelle OP fonctionne dans le graphe modele (forward/backward si necessaire) et passe les tests de base.

## Etape 1 - Definir l'OP

Questions a fixer d'abord:
1. nom exact de l'OP,
2. entree(s)/sortie(s),
3. contraintes shape/dtype,
4. comportement numerique attendu.

Regle: ecrire ce contrat avant de coder.

## Etape 2 - Integrer au chemin des layers

Objectif technique:
1. ajouter le type d'operation dans les enums/structures de layer,
2. ajouter les parametres necessaires,
3. brancher l'execution dans le moteur (`forwardLayer` ou chemin equivalent),
4. gerer les erreurs shape/dtype proprement.

## Etape 3 - Ajouter backward (si OP entrainable)

Si l'OP participe a l'entrainement:
1. definir le gradient attendu,
2. implementer backward,
3. verifier gradient numerique sur un mini cas.

Si l'OP est inference-only, documenter clairement la limite.

## Etape 4 - Exposer dans scripts/config

1. ajouter la creation de cette OP dans le chemin de construction du modele,
2. ajouter un mini exemple script ou config,
3. verifier qu'un modele de demo peut l'executer.

## Etape 5 - Validation minimale

1. test shape valide -> resultat correct,
2. test shape invalide -> erreur claire,
3. test precision numerique vs reference,
4. test integration dans un mini modele,
5. test non-regression sur un script existant.

## Erreurs frequentes

1. Oublier un check shape avant calcul.
2. Ecrire une sortie partielle puis retourner succes.
3. Ne pas gerer le dtype correctement.
4. Ajouter l'OP sans doc ni exemple.

## Exemple pratique

### Contexte

Tu veux ajouter une OP simple et verifiable rapidement avant d'ajouter des cas plus complexes.

### Code commente

Exemple conceptuel pour une OP `Scale` (sortie = entree * facteur):

```cpp
// 1) Nouveau type d'operation (enum).
enum class LayerType {
	// ...
	Scale
};

// 2) Parametre minimal dans la couche.
struct Layer {
	LayerType type;
	float scale = 1.0f; // facteur multiplicatif pour l'OP Scale
	// ...
};

// 3) Execution forward: checks puis calcul.
bool runScale(const std::vector<float>& in, std::vector<float>& out, float factor) {
	if (in.empty()) return false; // check simple (exemple)
	out.resize(in.size());
	for (size_t i = 0; i < in.size(); ++i) {
		out[i] = in[i] * factor;
	}
	return true;
}

// 4) Branch dans forwardLayer (pseudo-code).
if (layer.type == LayerType::Scale) {
	return runScale(inputTensor, outputTensor, layer.scale);
}
```

### Explication

1. contrat simple (entree, sortie, parametre),
2. chemin execution explicite,
3. base claire pour ajouter ensuite backward + tests.

### Test rapide

```bash
./bin/mimir --lua scripts/templates/template_new_model.lua
```

Verification attendue: le modele de demo passe le forward sans erreur de shape/dtype apres integration de l'OP.

## Suite

- Layers/Ops internals: [docs/04-Architecture-Internals/14-Layers-And-Ops.md](../04-Architecture-Internals/14-Layers-And-Ops.md)
- Dev model wiring: [docs/07-Devs/02-Building-Models-And-Layers.md](../07-Devs/02-Building-Models-And-Layers.md)
