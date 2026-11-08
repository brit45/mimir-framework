# Automatiser un run, exporter ses résultats et contribuer

Ce tutoriel relie un script Lua, sa configuration, ses fichiers de sortie et
ses vérifications. Il utilise un modèle natif enregistré et aucune donnée
externe. La dernière partie montre comment préparer une contribution du moteur.

| Niveau | Résultat |
| --- | --- |
| Débutant | Exécuter le même script avec un argument différent. |
| Intermédiaire | Sauvegarder, recharger et comparer un modèle. |
| Expert | Vérifier le contrat d'un bridge ou une extension C++ en CI. |

## 1. Respecter la frontière architecture / orchestration

Le graphe DL se définit en C/C++ dans une classe dérivée de `Model`, puis
s'enregistre dans le registre. Les scripts choisissent cette architecture,
appliquent ses paramètres et pilotent ses méthodes. Ils ne réécrivent pas sa
topologie avec des appels de création de couches.
Le [contrat d'architecture](../07-Devs/08-Architecture-Contract.md) formalise
cette frontière pour tous les tutoriels.

MPK est l'autre voie de description d'une architecture, encore expérimentale
et pas pleinement exploitable en production. Un package MPK n'est pas un
checkpoint de poids et ne doit pas remplacer un workflow natif validé sans
vérification explicite. Voir [l'atelier MPK](09-Tuto-Creer-MPK.md).

## 2. Écrire un script de run autonome

Sur POSIX, enregistrer dans `/tmp/mimir_exporter.lua` :

```lua
local Args = dofile(ROOTWORK .. "/scripts/modules/args.lua")
local FS = dofile(ROOTWORK .. "/scripts/modules/fs.lua")
local opts = Args.parse(arg) or {}
local destination = Args.get_str(opts, "out", "runs/tutoriels/export")
assert(FS.mkdir_p(destination))

local configuration = {
  input_dim = 4, hidden_dim = 8, output_dim = 3,
  hidden_layers = 1, dropout = 0, dtype = "float32",
}
if type(CONF) == "table" and type(CONF.model) == "table" then
  for key, value in pairs(CONF.model) do configuration[key] = value end
end
assert(Mimir.Model.create("basic_mlp", configuration))
assert(Mimir.Model.set_hardware(false))
assert(Mimir.Model.allocate_params())
assert(Mimir.Model.init_weights("xavier", 51))
local input = {0.2, -0.1, 0.7, 0.0}
local reference = assert(Mimir.Model.forward(input, false))

local path = FS.join(destination, "model.safetensors")
local options = {
  save_tokenizer = false, save_encoder = false, save_optimizer = false,
}
assert(Mimir.Serialization.save(path, "safetensors", options))
assert(Mimir.Serialization.load(path, "safetensors", {
  strict_mode = true, load_tokenizer = false,
  load_encoder = false, load_optimizer = false,
}))
local restored = assert(Mimir.Model.forward(input, false))
assert(#reference == #restored)
for i, value in ipairs(reference) do assert(restored[i] == value) end
assert(Mimir.Serialization.save_enhanced_debug(
  FS.join(destination, "model.debug.json"), {
    include_gradients = false, include_activations = false,
    max_values_per_tensor = 8,
  }))
print("EXPORT_ROUNDTRIP_OK", path)
```

Depuis la racine du dépôt :

```bash
./run_mimir.sh --lua /tmp/mimir_exporter.lua -- \
  --out runs/tutoriels/export --no-viz --no-htop
```

Résultat attendu : `EXPORT_ROUNDTRIP_OK`, le fichier SafeTensors et le JSON
d'inspection. Le script compare les sorties du même petit modèle avant et
après rechargement. Il ne démontre pas une reprise d'optimiseur : aucun état
d'entraînement n'a été enregistré dans cet atelier d'inférence.

L'argument `--out` est transmis par la table globale `arg`. `Args` est un module
Lua local, pas `Mimir.Args`. `ROOTWORK` permet de retrouver les modules du dépôt.
Les opérations de fichiers utilisent `FS`, plutôt que des commandes `mkdir`
ou `ls` spécifiques à un OS dans le script applicatif.

## 3. Passer la configuration par JSON

Le mode `--conf` injecte `CONF`, `CONF_PATH` et `CONF_DIR` dans le script.
Il peut aussi définir des tâches nommées et une section `env`. La structure
exacte du lanceur et des chemins figure dans le
[guide configuration](../02-User-Guide/08-Config-Driven-Scripting.md).

Exercice : conserver les dimensions d'entrée et de sortie de l'exemple et
déplacer `hidden_dim` dans `CONF.model`. Relancer vers un nouveau dossier et
vérifier le changement de nombre de paramètres. Ne pas utiliser une modification
de topologie comme override d'une reprise stricte de checkpoint existant.

Les modules de pipeline coordonnent création depuis le registre, allocation,
entraînement et export. Leur utilisation ne transforme pas le script de run
en définition d'architecture. Le [tutoriel registre et checkpoint](08-Tuto-Registre-Pipeline-Checkpoint.md)
présente ce parcours avec les conventions de modules du dépôt.

## 4. Choisir le bon artefact

| Artefact | Usage | À ne pas en déduire |
| --- | --- | --- |
| RawFolder | Checkpoint natif avec fichiers et métadonnées associés. | Un dossier incomplet n'est pas une reprise valide. |
| SafeTensors | Poids typés et métadonnées selon le writer Mímir. | Des poids externes seuls ne reconstruisent pas forcément le graphe. |
| DebugJSON | Inspection de configuration, statistiques et échantillons. | Il ne remplace pas automatiquement un checkpoint de reprise. |
| MPK | Architecture/configuration expérimentale. | Il ne contient pas les poids appris. |
| CSV / rapport HTML | Historique et analyse de métriques. | Ce ne sont ni des paramètres ni une validation recalculée. |

Pour reprendre un entraînement, préserver aussi l'optimiseur, sa progression,
le tokenizer et les états spécialisés du modèle. Le
[contrat de sérialisation](../03-API-Reference/16-Serialization.md) distingue les
options `load_optimizer`, `strict_mode`, `load_layer_metadata` et les dtypes.

## 5. Changer de bridge sans supposer la parité

Lua est la référence complète. Le bridge JavaScript expose des callbacks
natifs par Node.js, avec ses contraintes de transport et de plateforme.
Les bridges Rust et C# restent incomplets ; vérifier chaque API nécessaire
avant de déplacer un workflow. Les noms globaux attendus viennent de
`ScriptingContext`, pas d'une convention inventée par le script.

Le [contrat entre langages](../07-Devs/05-Scripting-System-Contract.md) et
la [référence JavaScript](../03-API-Reference/21-JavaScript-API.md) précisent les
limites. Le bridge REST natif n'est pas une API disponible à activer en supposant
qu'un serveur sera créé. Le serveur Web de Viz est un autre service, local et
destiné à l'affichage.

## 6. Préparer une contribution vérifiable

Commencer par classer le changement :

| Changement | Source de vérité | Vérification ciblée |
| --- | --- | --- |
| Modèle | Classe C++ dérivée de `Model` et registre. | Formes, forward/backward, checkpoint et script de run. |
| Opération | Type de layer et kernels des runtimes. | Valeurs, gradients, domaines, refus atomiques et backend choisi. |
| Bridge | Enregistrement des callbacks et contexte partagé. | Noms globaux, arguments, erreurs et aller-retour des valeurs. |
| Interface | Publication de snapshots et backend de rendu. | Test sans affichage puis test graphique réel si disponible. |
| Documentation | API et commandes réellement présentes. | Liens, exemples et limites explicites. |

Avant de proposer le changement :

```bash
git diff --check
python3 scripts/tools/verify_docs.py
bash scripts/tools/verify_api_sync.sh
ctest --test-dir build --output-on-failure -R '^LuaTest\.ApiBasics$'
```

Cette recette suppose les cibles de test construites. Une liste CTest n'est
pas une exécution ; un test ignoré faute de SDK ou de device n'est pas une
preuve matérielle. Ajouter ensuite les tests correspondant au code modifié.
Le [guide d'ajout d'opération](05-Tuto-Ajouter-Op.md) montre comment suivre le
type jusqu'au routeur et aux kernels.

## 7. Exercices et dépannage

| Niveau | Exercice | Résultat à contrôler |
| --- | --- | --- |
| Débutant | Changer `--out`. | Les fichiers sont écrits dans le dossier choisi. |
| Intermédiaire | Exporter une seconde fois après chargement. | Formes, dtype et prédictions sont préservés. |
| Expert | Ajouter une option publique à un run natif. | Config, aide, bindings éventuels, tests et documentation concordent. |

Une erreur de forme au rechargement peut indiquer une architecture incompatible,
pas un fichier corrompu. Une API absente d'un autre bridge exige de vérifier
sa couverture ; modifier les noms globaux au hasard masque le problème.

## Étapes suivantes

- [Ajouter un modèle C++](02-Tuto-Ajouter-Modele.md)
- [Modifier un runtime](04-Tuto-Modifier-Ou-Ajouter-Runtime.md)
- [Index des tutoriels](00-INDEX.md)
