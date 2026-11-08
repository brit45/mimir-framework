# Installer Mímir et rendre un lancement reproductible

Ce tutoriel mène d'un premier lancement à une configuration JSON réutilisable.
Il s'adresse aux débutants, puis propose des exercices intermédiaires et avancés.
Le résultat est un petit modèle du registre exécuté sur CPU, sans dataset.

L'architecture de ce modèle est définie dans une classe C++ héritant de `Model`.
Lua et JSON sélectionnent ses paramètres et pilotent son exécution. MPK est
l'autre voie de description d'architecture, expérimentale et encore incomplète
pour la production ; ce tutoriel ne l'utilise pas.
Le [contrat d'architecture](../07-Devs/08-Architecture-Contract.md) précise
la séparation entre topologie C++ et orchestration du run.

## Prérequis et résultats

Travaillez depuis la racine du dépôt. Les commandes shell ci-dessous supposent
Bash ; sous Windows, utilisez un environnement MSYS2 adapté. Les petits fichiers
d'exercice sont placés dans `/tmp/mimir_tutoriels`. Adaptez ce chemin à votre
répertoire temporaire si nécessaire, en conservant les mêmes chemins entre étapes.

| Étape | Vous obtenez | Contrôle |
| --- | --- | --- |
| Installation | Un binaire et un profil de build identifiables | Aide du binaire |
| Registre | Une architecture native disponible | Liste `basic_mlp` |
| Script | Une sortie de deux nombres | Assertion sur la taille |
| JSON | Le même lancement avec sa configuration | Même seed et mêmes dimensions |

## 1. Choisir un profil de compilation

Commencez par consulter les options, sans installer ni configurer :

```bash
./config.sh --help
```

`config.sh` détecte Linux, macOS, plusieurs BSD et les environnements
MSYS2/MinGW. Il connaît plusieurs gestionnaires de paquets ; cela ne démontre
pas que chaque combinaison d'OS, de compilateur, de pilote et de GUI soit testée.
`NONE` évite la dépendance à une interface graphique. Qt, GTK et Web ciblent
Linux dans le script actuel ; SFML et `NONE` sont proposés ailleurs.

La recette suivante est une configuration à exécuter sur votre poste, après
installation des dépendances C++17, CMake, Ninja et Lua de développement.
Elle utilise un dossier distinct pour préserver un autre cache CMake :

```bash
./config.sh --gui NONE --build-dir build-tutoriels \
  --generator Ninja --non-interactive --skip-system-deps \
  --enable MIMIR_ENABLE_TESTS \
  --disable ENABLE_VULKAN --disable ENABLE_OPENCL --no-ddp
cmake --build build-tutoriels --parallel 2
```

`--skip-system-deps` signifie que les dépendances sont déjà installées. Sans
cette option, le script peut lancer l'installateur du système et demander les
droits nécessaires. Pour une compilation déjà fonctionnelle, conservez son
profil et passez directement à l'étape suivante. Ne reconfigurez pas un build
GPU uniquement pour suivre ce tutoriel CPU.

Consultez l'[installation](../01-Getting-Started/02-Installation.md) pour les
variantes par plateforme. La compilation et l'exécution sur les autres OS
restent à valider sur le matériel et la chaîne d'outils concernés.

Le build secondaire produit par défaut `build-tutoriels/bin/mimir`. Pour suivre
la suite avec ce profil, sélectionnez-le dans le shell courant :

```bash
export MIMIR_EXECUTABLE="$PWD/build-tutoriels/bin/mimir"
```

Un build habituel sous `build/` produit `bin/mimir` et ne demande pas cet export.
`MIMIR_RUNTIME_OUTPUT_DIRECTORY` peut modifier ces destinations.

## 2. Vérifier le binaire et le registre

```bash
./run_mimir.sh --help
./run_mimir.sh --lua scripts/tools/inspect_architectures.lua -- --show-archs
```

Vérifiez la version affichée et la présence de `basic_mlp`. Le registre associe
ce nom à `BasicMLPModel`, déclaré dans `src/Models/MLP/BasicMLPModel.hpp`.
Un nom de modèle reconnu ne garantit pas qu'un workflow métier soit complet :
les limites des familles figurent dans le [tutoriel des modèles](13-Tuto-Familles-De-Modeles.md).

## 3. Écrire un script de pilotage

Créez `/tmp/mimir_tutoriels/inspect.lua` avec ce contenu ; créez son dossier
au préalable. Il ne définit aucun graphe : il demande une architecture existante.

```lua
local architecture = CONF.model.architecture
local cfg = assert(Mimir.Architectures.default_config(architecture))
for key, value in pairs(CONF.model) do
  if key ~= "architecture" then cfg[key] = value end
end
assert(Mimir.Model.create(architecture, cfg))
assert(Mimir.Model.set_hardware(false))
assert(Mimir.Model.allocate_params())
assert(Mimir.Model.init_weights("xavier", CONF.seed))
local output = assert(Mimir.Model.forward({0.1, 0.2, 0.3, 0.4}, false))
assert(#output == 2)
print("CONFIG_TUTORIEL_OK", Mimir.Model.total_params(), output[1], output[2])
```

`create` construit le modèle natif. `allocate_params` réserve les poids ;
`init_weights` les initialise. Un chargement de checkpoint remplacerait cette
dernière étape. `set_hardware(false)` choisit ici le chemin CPU du modèle.

## 4. Enregistrer la configuration du run

Créez `/tmp/mimir_tutoriels/config.json` :

```json
{
  "env": {
    "OMP_NUM_THREADS": "1",
    "MIMIR_DISABLE_CUDA": "1",
    "MIMIR_DISABLE_ROCM": "1",
    "MIMIR_DISABLE_VULKAN": "1",
    "MIMIR_DISABLE_OPENCL": "1"
  },
  "lua": {
    "scripts": ["/tmp/mimir_tutoriels/inspect.lua"]
  },
  "model": {
    "architecture": "basic_mlp",
    "input_dim": 4,
    "hidden_dim": 8,
    "hidden_layers": 1,
    "output_dim": 2,
    "dropout": 0.0
  },
  "seed": 1337
}
```

Puis lancez :

```bash
./run_mimir.sh --conf /tmp/mimir_tutoriels/config.json
```

Le résultat attendu contient `CONFIG_TUTORIEL_OK` et deux sorties numériques.
Relancez sans modifier les fichiers pour comparer les valeurs dans le même
environnement. La reproductibilité sur un autre backend se vérifie séparément.

Le binaire injecte `CONF`, `CONF_PATH` et `CONF_DIR`. Plusieurs entrées
`lua.scripts` s'exécutent dans des VM Lua distinctes : ne leur faites pas
partager un modèle via une variable globale. Utilisez un checkpoint entre étapes.

## 5. Passer à des paramètres et à des tâches

Le premier `--` sépare les options du binaire de celles du script :

```bash
./run_mimir.sh --lua scripts/training/train_causal_lm.lua -- --help
```

Un script lit ses options avec `scripts/modules/args.lua` et la table standard
`arg`. Une valeur JSON n'a d'effet que si le script ou le moteur la consomme.
Pour plusieurs tâches et les overrides, poursuivez avec le
[guide de configuration](../02-User-Guide/08-Config-Driven-Scripting.md).

## Exercices par niveau

| Niveau | Exercice | Réussite observable |
| --- | --- | --- |
| Débutant | Changer uniquement `hidden_dim` de 8 à 12 | Le nombre de paramètres change, la sortie garde deux valeurs |
| Intermédiaire | Changer la seed, puis la remettre à 1337 | Les poids initiaux changent puis retrouvent leur lancement de référence |
| Avancé | Conserver config, version, profil CMake et journal pour deux machines | Vous savez expliquer les différences de build et de backend |

Changer un paramètre supporté fait construire une variante par la classe C++.
Ajouter une branche ou un nouvel opérateur au graphe demande une modification
de cette classe et ses tests ; ne contournez pas cette règle dans le script Lua.

## Dépannage

| Symptôme | Vérification utile |
| --- | --- |
| Dépendance introuvable | Le paquet de développement et le compilateur correspondent-ils au même environnement ? |
| Mauvais générateur CMake | Utilisez un nouveau dossier de build plutôt que mélanger Makefiles et Ninja |
| `CONF` absent | Le script doit être lancé avec `--conf` pour cet exemple |
| Architecture inconnue | Consultez le registre du binaire réellement exécuté |
| Fenêtre ou périphérique inattendu | Vérifiez la section `env` et le profil de compilation |

## Étapes suivantes

- [Préparer des données et un tokenizer](11-Tuto-Donnees-Tokenizer.md).
- [Entraîner, valider, reprendre et générer](12-Tuto-Entrainement-Validation-Reprise-Inference.md).
- [Index des tutoriels](00-INDEX.md).
