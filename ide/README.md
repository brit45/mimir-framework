# Helpers IDE Mímir

Tous les stubs et helpers sont centralisés ici. Ils décrivent l'API native de
pilotage ; ils ne remplacent pas le moteur. Ne pas exécuter le stub Lua avec
`dofile` : cela remplacerait les fonctions natives par des fonctions vides.

| Langage | Aide | Serveur / éditeurs |
|---|---|---|
| Lua | `lua/mimir-api.lua` : EmmyLua, signatures, types et documentation | LuaLS, VS Code/VSCodium, Neovim, Emacs, Sublime ; EmmyLua pour JetBrains |
| JavaScript | `javascript/mimir.d.ts` : namespaces, arguments, primitives, retours bruts | TypeScript Language Service, VS Code/VSCodium, WebStorm, Neovim, Emacs, Sublime |
| Rust | `rust/` : crate std-only, `Value`, `Result`, méthodes des namespaces | rust-analyzer, RustRover/IntelliJ Rust, VS Code/VSCodium, Neovim, Emacs |
| MPK | `vscode-mpk/` : extension, schéma et snippets existants | VS Code/VSCodium |

Les fichiers `.luarc.json`, `.vscode/settings.json` et `jsconfig.json` à la racine
référencent ces dossiers. Les recommandations VS Code sont dans
`.vscode/extensions.json`. Les presets versionnés sont sous `vscode/` : copier
les clés utiles dans `.vscode/` si ce dossier est absent ou ignoré par Git.
Les paramètres existants de l'utilisateur sont conservés.

## Installer dans un éditeur

- **VS Code / VSCodium** : ouvrir la racine du dépôt, installer les extensions
  recommandées. LuaLS et JavaScript découvrent les fichiers de configuration ;
  rust-analyzer charge `ide/rust/Cargo.toml`. Pour l'extension MPK, voir
  [son installation](vscode-mpk/README.md).
- **JetBrains** : ajouter `ide/lua` aux bibliothèques EmmyLua ; le service
  TypeScript utilise les déclarations incluses par `jsconfig.json`. Pour Rust,
  attacher `ide/rust/Cargo.toml`. Aucun plugin JetBrains propriétaire n'est fourni.
- **Neovim 0.11+** : installer les trois serveurs de langage puis charger
  `neovim/mimir.lua`. Ce fichier utilise les API LSP intégrées.
- **Emacs / Eglot** : installer les serveurs et les modes de langage, charger
  `emacs/mimir.el`, puis lancer `M-x eglot` dans le projet.
- **Sublime Text / LSP** : ouvrir `sublime/mimir.sublime-project` avec les clients
  Lua, TypeScript et Rust installés. La découverte Lua/JS utilise les fichiers
  de configuration à la racine. Pour Rust, ouvrir aussi le projet Cargo sous
  `ide/rust` si le client ne découvre pas les crates imbriquées.

Les configurations sont fournies et contrôlées syntaxiquement ; l'interface de
chaque éditeur n'est pas testée automatiquement.

## Scripts Rust et analyse statique

Le CLI accepte des instructions Rust ou un fichier complet avec `fn main` et
injecte le module `mimir`. Les namespaces s'utilisent par exemple ainsi :

```rust
let cfg = Architectures::default_config(&["basic_mlp".into()])?;
Model::create(&["basic_mlp".into(), cfg])?;
```

Les arguments sont une liste de `Value`, les résultats un `Result<Value>`.
`Value::Null` permet de sauter un argument optionnel. La méthode générique
`mimir::call("Model.forward", &args)` expose aussi les futures fonctions du
registre. Les fermetures des builders possèdent `.call()` et `.results()`.

Un fragment d'instructions seul n'est pas une unité de compilation pour
rust-analyzer. Utiliser le [fichier complet d'exemple](rust/example.rs) pour
l'analyse statique, ou créer un projet Cargo dépendant de `mimir-ide` par chemin :

```toml
[dependencies]
mimir-ide = { path = "/chemin/vers/mimir/ide/rust" }
```

Dans ce projet, `use mimir_ide as mimir;` permet de vérifier le code ; son
exécution native doit passer par le bridge Mímir. La crate ne télécharge aucune
dépendance. L'exemple contient un module local pour l'IDE : pour le lancer par le
CLI, retirer ce module local, puisque le CLI injecte déjà `mimir`.

## Régénération et contrôle

Depuis la racine, avec un binaire à jour :

```sh
python3 scripts/tools/generate_ide_helpers.py
python3 scripts/tools/generate_ide_helpers.py --check
bash scripts/tools/verify_api_sync.sh
cargo check --offline --all-targets --manifest-path ide/rust/Cargo.toml
```

Le générateur interroge le registre Lua réel, y compris les closures dynamiques
`Layers`. Il conserve les annotations Lua écrites à la main, ajoute les fonctions
manquantes et génère les déclarations JS, les méthodes Rust et
`api-manifest.json`. Les champs de configuration extensibles et certains retours
JS conservent un type ouvert ; les déclarations ne prétendent pas décrire une
structure plus stricte que l'API native.
