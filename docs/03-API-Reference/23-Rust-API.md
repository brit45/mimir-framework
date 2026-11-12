# Bridge Rust

Le bridge Rust exécute les callbacks natifs de référence Lua **immédiatement**,
par le même transport synchrone que JavaScript. Il ne rejoue plus de commandes
après la fin du script et ne renvoie plus de compteurs de paramètres en cache.
Les namespaces et leurs méthodes sont découverts dans le registre natif à chaque
exécution, y compris les closures `Layers` et les builders `Database`.

```sh
./run_mimir.sh --rust scripts/examples/bridge_smoke.rs
./run_mimir.sh --rust mon_script.rs -- argument1 argument2
```

`rustc` est requis dans `PATH` ; `rust-script`, Cargo et des dépendances réseau
ne sont pas nécessaires à l'exécution. La compilation du script est incluse
dans la latence de lancement. Les programmes exécutés doivent être de confiance,
comme les scripts Lua/JS : ce mécanisme n'est pas une sandbox.

## Appels, arguments et erreurs

Un fichier peut contenir des instructions, automatiquement enveloppées dans
`fn main() -> mimir::Result<()>`, ou un programme complet avec `fn main`.
Le module `mimir` et ses namespaces sont injectés par le bridge.

```rust
let cfg = Architectures::default_config(&["basic_mlp".into()])?;
Mimir::Model::create(&["basic_mlp".into(), cfg])?;
Model::allocate_params(&[])?;
Model::init_weights(&["he".into(), 42.into()])?;
println!("{:?}", Model::total_params(&[])?);

// Une API future reste accessible sans modifier de catalogue Rust.
let layers = mimir::call("Model.get_layers", &[])?;
assert!(layers.as_array().is_some());
```

Chaque méthode prend `&[Value]`, conserve l'ordre des arguments Lua et renvoie
`Result<Value>`. Utiliser `Value::Null` pour sauter une position optionnelle.
Les anciennes signatures du bootstrap Rust partiel (`Model::create(name, json)`)
sont remplacées par cette convention uniforme ; les exemples ont été migrés.

- Aucun retour devient `Value::Null` ; un retour devient une valeur ; plusieurs
  retours deviennent `Value::Array`.
- Une erreur native levée ou un tuple `nil, message` / `false, message` devient
  `Err(Error)`. `?` la propage immédiatement ; `.is_err()` permet de la tester.
- `mimir::results("Architectures.default_config", &args)` conserve tous les
  retours bruts, y compris les tuples d'erreur. Une closure retournée par un
  builder s'invoque avec `value.call(&args)` ou `value.results(&args)`.
- `Value::object`, `Value::array`, `.into()`, les accesseurs `as_*`, l'indexation
  et `Value::parse` / `stringify` permettent de construire et lire les données.
- `mimir::floats(&[1.0, 0.0])` préserve le type flottant, y compris pour des
  valeurs entières. Les indices métier (`Dataset::get(&[1.into()])`) restent
  ceux de Lua ; les listes de résultats Rust sont indexées à partir de zéro.
- Les chaînes UTF-8, échappements et caractères Unicode sont conservés. Les
  entiers Rust sont sérialisés sans passer par un `double` ; le moteur conserve
  les limites numériques de Lua. Les valeurs non finies et la profondeur
  supérieure à 64 sont rejetées.

## Configuration et contexte

`mimir::globals()` donne les variables système, dont `Mimir`, `arg`, `CONF`,
`CONF_PATH`, `CONF_DIR` et `ROOTWORK`. Les fonctions `arg()`, `conf()`,
`conf_path()` et `conf_dir()` offrent un accès direct. `arg()[0]` est le chemin
logique du script ; les arguments utilisateur commencent en position 1.
Les aliases `model`, `architectures`, `tokenizer`, `dataset`, `htop`, `viz` sont
des modules Rust ; `Memory`, `MemoryGuard` et `Allocator` sont également exposés.

Les modes `rust.script`, `rust.scripts`, `run.rust` et `tasks`/`--run` injectent
la configuration complète selon les mêmes conventions que JS et Lua :

```json
{"rust":{"scripts":[{"script":"scripts/examples/bridge_smoke.rs","args":["exemple"]}]}}
```

Le modèle, tokenizer, dataset et moniteur persistent entre les exécutions d'un
même contexte Rust. Les variables du programme Rust ne persistent pas : chaque
exécution compile et lance un nouveau processus. Le contexte Lua préexistant
est restauré, même après une erreur. Les exécutions JS et Rust partagent un
verrou ; les accès concurrents directs au singleton Lua restent hors contrat.

## Transport et limites

Le transport POSIX utilise un socket dédié, distinct de stdin/stdout/stderr.
Sur Windows, configurer `ENABLE_SCRIPTING_RUST=OFF` et
`ENABLE_SCRIPTING_JS=OFF` ; ces bridges sont désactivés par défaut pour cette
cible. Une activation explicite est rejetée à la configuration. Lua et C#
restent compilables, notamment dans la CI de compilation croisée MinGW.
Les arguments ne passent pas par un shell et l'environnement du parent n'est
pas modifié. Les fichiers temporaires sont créés dans un dossier privé et
supprimés après l'exécution, y compris après erreur. Un programme complet garde
la résolution de ses modules et `include!` relativement au fichier source ; les
fragments d'instructions sont compilés depuis le dossier temporaire.

La parité couvre les API natives de pilotage, pas la bibliothèque standard Lua,
les modules `.lua`, les userdata ni des callbacks utilisateurs Rust injectés
dans Lua. Les grandes données ont un coût de copie JSON. Le bridge n'est pas
une interface tensorielle sans copie et ne prend pas en charge les crates
externes déclarées par un script Cargo autonome.

## Validation et IDE

`BridgeTest.RustParity` couvre les erreurs, retours multiples, closures, contexte,
config/arguments Unicode, forward/backward/optimiseur, égalité avec le moteur,
checkpoint, tokenizer, dataset RGB/texte, JSON volumineux et modules relatifs.
`BridgeTest.RustConf` couvre les entrées CLI et la persistance entre scripts.
Ces tests ne valident pas le rendu graphique de toutes les interfaces Viz.

Les helpers sont dans [ide/rust](../../ide/rust/Cargo.toml) ; voir
[les instructions par éditeur](../../ide/README.md).

## Étapes suivantes

- [Index de la documentation](../00-INDEX.md)
- [Bridge JavaScript](21-JavaScript-API.md)
- [Contrat de scripting](00-Scripting-Contract.md)
