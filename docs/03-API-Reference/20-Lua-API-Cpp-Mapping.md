# Mapping API Lua → C/C++ (bindings) → sous-systèmes

Cette page est un **sommaire** “appel Lua ↔ binding C++ ↔ à quoi ça fait référence”.

- Source de vérité : `src/LuaScripting.cpp` (fonction `LuaScripting::registerAPI()`).
- Les fonctions C++ listées ci-dessous sont les **bindings** (souvent `LuaScripting::lua_*`).
- La colonne “Référence interne” pointe les **classes / fichiers** réellement utilisés derrière.

> Note “legacy” :
>
> - `Mimir.Checkpoint.*` est **déprécié** et n’est qu’un alias vers `Mimir.Serialization.*`.
> - `Mimir.Model.save/load` utilise un ancien chemin (méthodes `Model::*Checkpoint/*Load*`).
>   Pour les formats modernes, privilégiez `Mimir.Serialization.save/load/detect_format`.


## Vue rapide (modules)

|Module Lua|Rôle|Exemple d’usage (script)|Référence interne|
|---|---|---|---|
|`Mimir.Model`|Création/build + forward/backward + training helpers|créer → allouer → init → forward|`Model`, `ModelArchitectures`, `Optimizer`, `Gradients`, `Tokenizer`, `Encoder`|
|`Mimir.Architectures`|Registry des architectures|lister + demander config défaut|`src/Models/Registry/ModelArchitectures.*`|
|`Mimir.Serialization`|Save/load checkpoints + formats|save/load en `raw_folder` ou `safetensors`|`src/Serialization/Serialization.*`|
|`Mimir.Tokenizer`|Tokenisation/BPE + vocab + analyse|create + tokenize + detokenize|`src/Tokenizer.*`|
|`Mimir.Dataset`|Chargement d’un dataset + séquences|load + get + prepare_sequences|`loadDataset` (`src/Helpers.hpp`), `DatasetItem`|
|`Mimir.Memory`|Stats/config mémoire (best-effort)|config + stats/usage|`AdvancedRAMManager`|
|`Mimir.Guard` / `Mimir.MemoryGuard`|Limite stricte / compteurs|setLimit + getStats|`MemoryGuard`|
|`Mimir.Allocator`|Allocateur dynamique de tenseurs|configure au début du script|`DynamicTensorAllocator`|
|`Mimir.Htop`|Monitoring console|create + update + render|`HtopDisplay`|
|`Mimir.Viz` / `Mimir.visualiser`|Visualisation SFML + async monitor|create + initialize + loop events|`Visualizer`, `AsyncMonitor`, “viz taps” dans `Model`|
|`Mimir.Layers`|Ops “layer” (actuellement stubs)|(à éviter)|renvoie “Non implémenté”|
|`Mimir.NeuroPulse`|Texte → rendu (audio/lumière)|render + params|`NeuroPulseModel`|

---

## `Mimir.Model`

### Exemple (script) — `Mimir.Model`

```lua
-- 0) (recommandé) config mémoire au tout début
Mimir.Allocator.configure({ max_ram_gb = 8 })
Mimir.MemoryGuard.setLimit(8 * 1024 * 1024 * 1024)

-- 1) choisir une architecture + config
local cfg = Mimir.Architectures.default_config("transformer")

-- 2) créer le modèle
assert(Mimir.Model.create("transformer", cfg))

-- 3) allouer + init
local ok_alloc, nparams_or_err = Mimir.Model.allocate_params()
assert(ok_alloc, nparams_or_err)
assert(Mimir.Model.init_weights("he", 123))

-- 4) forward (tokens ou floats)
local y_or_nil, err = Mimir.Model.forward({ 1, 2, 3, 4 }, false)
if not y_or_nil then
  log("forward error:", err)
end
```

|Appel Lua|Effet (ce que ça fait)|Binding C++|Référence interne|Notes|
|---|---|---|---|---|
|`Mimir.Model.create(name, cfg?)`|Crée un modèle via le registre (config défaut si absente)|`LuaScripting::lua_createModel`|`ModelArchitectures::defaultConfig/create`, `LuaContext`, `sync_model_tokenizer_encoder_from_context`||
|`Mimir.Model.build()`|Reconstruit le modèle depuis `ctx.modelType/modelConfig`|`LuaScripting::lua_buildModel`|`ModelArchitectures::create`, `sync_model_tokenizer_encoder_from_context`|Compat/legacy : préférez `create(name, cfg)` direct.|
|`Mimir.Model.train(epochs, lr)`|Entraîne (selon le type de modèle) + gère optimizer/scheduler/interruption|`LuaScripting::lua_trainModel`|`Optimizer`, `Model::getSerializedOptimizer/setSerializedOptimizer`, `DatasetItem`, `Mimir::Serialization::save_checkpoint`|Chemin “legacy” peut être non supporté selon modèle.|
|`Mimir.Model.infer(input)`|Inférence texte (tokenize→encode→forward→decode)|`LuaScripting::lua_inferModel`|`Tokenizer`, `Encoder`, `Model::forward/eval`|Historique : contient un fallback si tokenizer absent.|
|`Mimir.Model.save(path)`|Sauvegarde modèle (ancien format checkpoint)|`LuaScripting::lua_saveModel`|`Model::saveCheckpoint`, `Tokenizer`, `MagicToken`|**Legacy** : préférez `Mimir.Serialization.save`.|
|`Mimir.Model.load(path)`|Charge modèle depuis dossier (cherche safetensors)|`LuaScripting::lua_loadModel`|`Model::tryLoadExistingModel`, `Tokenizer`, `Encoder`, `MagicToken`, `sync_model_tokenizer_encoder_from_context`|**Legacy** : préférez `Mimir.Serialization.load`.|
|`Mimir.Model.allocate_params()`|Alloue les paramètres/poids du modèle|`LuaScripting::lua_allocateParams`|`Model::allocateParams`, `Model::totalParamCount`|Renvoie `(ok, nparams)` ou `(false, err)`.|
|`Mimir.Model.init_weights(method="he", seed=0)`|Initialise les poids (He/Xavier/…)|`LuaScripting::lua_initWeights`|`Model::initializeWeights`||
|`Mimir.Model.total_params()`|Donne le nombre de paramètres|`LuaScripting::lua_totalParams`|`Model::totalParamCount`||
|`Mimir.Model.push_layer(name, type, params_count)`|Ajoute un layer au graph (API bas niveau)|`LuaScripting::lua_pushLayer`|`Model::push`|Utilisé pour construire manuellement.|
|`Mimir.Model.set_layer_io(layer, inputs, output?)`|Fixe les entrées (noms) et l’output (nom) d’un layer|`LuaScripting::lua_setLayerIO`|`Model::getLayerByName`, `Layer::inputs/output`||
|`Mimir.Model.forward(input, training=true)`|Forward pass; accepte liste int/float ou map `{name->table}`|`LuaScripting::lua_forwardPass`|`Model::forwardPass` / `Model::forwardPassNamed`, viz taps (`Model::*VizTaps*`), `Visualizer::BlockFrame`|Retourne `table` ou `(nil, err)`.|
|`Mimir.Model.forward_prompt_image_seed(text_vec, image_vec, seed, training=false)`|Forward spécialisé (texte encodé + image + seed)|`LuaScripting::lua_forwardPromptImageSeed`|`Model::forwardPromptImageSeed`, viz taps|Utile diffusion/t2i.|
|`Mimir.Model.encode_prompt(prompt)` / `encodePrompt`|Prompt texte → vecteur embedding (via tokenizer+encoder)|`LuaScripting::lua_encodePrompt`|`Tokenizer`, `Encoder`, config modèle (ex: `PonyXLDDPMModel::getConfig`)|Respecte `tokenizer_frozen` si présent.|
|`Mimir.Model.backward(loss_grad)`|Backprop depuis un gradient de loss|`LuaScripting::lua_backwardPass`|`Model::backwardPass` → `Gradients`||
|`Mimir.Model.zero_grads()`|Zéro les gradients|`LuaScripting::lua_zeroGradients`|`Model::zeroGradients`||
|`Mimir.Model.get_gradients()`|Retourne les gradients (table)|`LuaScripting::lua_getGradients`|`Model::getGradients`||
|`Mimir.Model.optimizer_step(lr, opt_type="adamw")`|Applique un step optimizer (helper)|`LuaScripting::lua_optimizerStep`|`Model::optimizerStep`, `Optimizer`|Pour contrôle fin, utilisez `set_optimizer` + `train`.|
|`Mimir.Model.get_optimizer()`|Lit optimizer depuis modèle (ou reconstruit depuis `modelConfig`)|`LuaScripting::lua_getOptimizer`|`Model::getSerializedOptimizer` + `ctx.modelConfig`|Best-effort.|
|`Mimir.Model.set_optimizer(opts)`|Met à jour config optimizer persistante + (si modèle) état sérialisé|`LuaScripting::lua_setOptimizer`|`ctx.modelConfig` + `Model::setSerializedOptimizer`|Supporte `reset_state=true`.|
|`Mimir.Model.reset_optimizer_state()`|Réinitialise moments/step (garde hyperparams)|`LuaScripting::lua_resetOptimizerState`|`Model::setSerializedOptimizer`||
|`Mimir.Model.set_hardware(enable)`|Active/désactive accélération matérielle (global)|`LuaScripting::lua_setHardwareAccel`|`Model::setHardwareAcceleration`||
|`Mimir.Model.hardware_caps()`|Retourne les capacités CPU + flags GPU compilés|`LuaScripting::lua_getHardwareCaps`|`Model::hasAVX2/hasFMA/hasF16C/hasBMI2`, `ENABLE_VULKAN/ENABLE_OPENCL`||

---

## `Mimir.Architectures`

### Exemple (script) — `Mimir.Architectures`

```lua
local names = Mimir.Architectures.available()
log("architectures:", #names)

local cfg = Mimir.Architectures.default_config(names[1])
log("first arch:", names[1])
```

|Appel Lua|Effet|Binding C++|Référence interne|Notes|
|---|---|---|---|---|
|`Mimir.Architectures.available()`|Liste les architectures disponibles|`LuaScripting::lua_archAvailable`|`ModelArchitectures::available` (`src/Models/Registry/ModelArchitectures.*`)||
|`Mimir.Architectures.default_config(name)`|Renvoie la config par défaut de l’architecture|`LuaScripting::lua_archDefaultConfig`|`ModelArchitectures::defaultConfig`||

---

## `Mimir.Serialization`

### Exemple (script) — `Mimir.Serialization`

```lua
-- Sauvegarde moderne (recommandé)
local ok, err = Mimir.Serialization.save("./checkpoint_out", "raw_folder", {

  save_optimizer = true,
  save_tokenizer = true,
  save_encoder = true,
})
assert(ok, err)

local fmt = Mimir.Serialization.detect_format("./checkpoint_out")
log("format:", fmt)

assert(Mimir.Serialization.load("./checkpoint_out", "auto"))
```

|Appel Lua|Effet|Binding C++|Référence interne|Notes|
|---|---|---|---|---|
|`Mimir.Serialization.save(path, format?, opts?)`|Sauvegarde un checkpoint (formats multiples)|`LuaScripting::lua_saveCheckpoint`|`Mimir::Serialization::save_checkpoint` (`src/Serialization/Serialization.*`)|`format`: `safetensors`, `raw_folder`, `debug_json`, `auto`, …|
|`Mimir.Serialization.load(path, format?, opts?)`|Charge un checkpoint + resync tokenizer/encoder en contexte|`LuaScripting::lua_loadCheckpoint`|`Mimir::Serialization::load_checkpoint` + `sync_context_tokenizer_encoder_from_model`||
|`Mimir.Serialization.detect_format(path)`|Détecte le format d’un checkpoint|`LuaScripting::lua_detectFormat`|`Mimir::Serialization::detect_format`|Typiquement avant un `load(..., "auto")`.|
|`Mimir.Serialization.save_enhanced_debug(path, opts?)`|Dump debug enrichi (inspection)|`LuaScripting::lua_saveEnhancedDebugJson`|debug JSON||

### `Mimir.Checkpoint` (déprécié)

|Appel Lua|Effet|Binding C++|Référence interne|Notes|
|---|---|---|---|---|
|`Mimir.Checkpoint.save(...)`|Alias legacy → `Mimir.Serialization.save`|`LuaScripting::lua_saveCheckpoint`|Identique à `Mimir.Serialization.save`|Déprécié.|
|`Mimir.Checkpoint.load(...)`|Alias legacy → `Mimir.Serialization.load`|`LuaScripting::lua_loadCheckpoint`|Identique à `Mimir.Serialization.load`|Déprécié.|

---

## `Mimir.Tokenizer`

### Exemple (script) — `Mimir.Tokenizer`

```lua
assert(Mimir.Tokenizer.create(50000))

local ids = Mimir.Tokenizer.tokenize("salut le monde")
log("tokens:", #ids)

local text = Mimir.Tokenizer.detokenize(ids)
log("roundtrip:", text)
```

|Appel Lua|Effet|Binding C++|Référence interne|Notes|
|---|---|---|---|---|
|`Mimir.Tokenizer.create(max_vocab)`|Crée un tokenizer (vocab max) dans le contexte|`LuaScripting::lua_createTokenizer`|`Tokenizer(max_vocab)` (`src/Tokenizer.*`)||
|`Mimir.Tokenizer.tokenize(text)`|Tokenise un texte (BPE) → `TokenIds`|`LuaScripting::lua_tokenize`|`Tokenizer::tokenizeBPE`||
|`Mimir.Tokenizer.detokenize(tokens)`|Reconstruit du texte depuis des tokens|`LuaScripting::lua_detokenize`|`Tokenizer::decode`||
|`Mimir.Tokenizer.vocab_size()`|Renvoie la taille du vocab|`LuaScripting::lua_getVocabSize`|`Tokenizer::getVocabSize`||
|`Mimir.Tokenizer.save(path)`|Sauvegarde tokenizer en JSON|`LuaScripting::lua_saveTokenizer`|`Tokenizer::to_json` + IO||
|`Mimir.Tokenizer.load(path)`|Charge tokenizer depuis JSON|`LuaScripting::lua_loadTokenizer`|`Tokenizer::from_json` + IO||
|`Mimir.Tokenizer.add_token(tok)`|Ajoute un token au vocab, renvoie l’id|`LuaScripting::lua_addToken`|`Tokenizer::addToken`||
|`Mimir.Tokenizer.ensure_vocab_from_text(text)`|Ajoute au vocab ce qu’il faut pour couvrir un texte|`LuaScripting::lua_ensureVocabFromText`|`Tokenizer::ensureVocabFromText`||
|`Mimir.Tokenizer.tokenize_ensure(text)`|Tokenise en étendant le vocab si nécessaire|`LuaScripting::lua_tokenizeEnsure`|`Tokenizer::tokenizeEnsure`||
|`Mimir.Tokenizer.pad_id()`|Id du token PAD|`LuaScripting::lua_getPadId`|`Tokenizer::getPadId`||
|`Mimir.Tokenizer.unk_id()`|Id du token UNK|`LuaScripting::lua_getUnkId`|`Tokenizer::getUnkId`||
|`Mimir.Tokenizer.seq_id()`|Id du token SEQ|`LuaScripting::lua_getSeqId`|`Tokenizer::getSeqId`||
|`Mimir.Tokenizer.mod_id()`|Id du token MOD|`LuaScripting::lua_getModId`|`Tokenizer::getModId`||
|`Mimir.Tokenizer.mag_id()`|Id du token MAG|`LuaScripting::lua_getMagId`|`Tokenizer::getMagId`||
|`Mimir.Tokenizer.get_token_by_id(id)`|Token string correspondant à un id|`LuaScripting::lua_getTokenById`|`Tokenizer::getTokenById`||
|`Mimir.Tokenizer.learn_bpe(corpus, merges=1000)`|Apprend BPE sur un corpus|`LuaScripting::lua_learnBPEFromCorpus`|`Tokenizer::learnBPEFromCorpus`||
|`Mimir.Tokenizer.tokenize_bpe(text)`|Tokenise explicitement via BPE|`LuaScripting::lua_tokenizeBPE`|`Tokenizer::tokenizeBPE`||
|`Mimir.Tokenizer.set_max_length(n)`|Fixe une longueur max interne (si utilisée)|`LuaScripting::lua_setMaxSequenceLength`|`Tokenizer::setMaxSequenceLength`||
|`Mimir.Tokenizer.pad_sequence(tokens, target_len?)`|Pad/truncate une séquence|`LuaScripting::lua_padSequence`|`Tokenizer::padSequence`||
|`Mimir.Tokenizer.batch_tokenize(texts, max_len=512)`|Tokenise un batch de textes|`LuaScripting::lua_batchTokenize`|`Tokenizer::batchTokenize`||
|`Mimir.Tokenizer.print_stats()`|Affiche des stats vocab|`LuaScripting::lua_printVocabStats`|`Tokenizer::printVocabStats`||
|`Mimir.Tokenizer.get_frequencies(text)`|Renvoie map token→fréquence|`LuaScripting::lua_getTokenFrequencies`|`Tokenizer::getTokenFrequencies`||
|`Mimir.Tokenizer.analyze_text(text)`|Analyse simple (entities/modifiers/actions/…)|`LuaScripting::lua_analyzeText`|`Tokenizer::analyzeText`||
|`Mimir.Tokenizer.extract_keywords(text, topN=5)`|Extrait `topN` keywords|`LuaScripting::lua_extractKeywords`|`Tokenizer::extractKeywords`||

---

## `Mimir.Dataset`

### Exemple (script) — `Mimir.Dataset`

```lua
local ok, n_or_err = Mimir.Dataset.load("./dataset_2")
assert(ok, n_or_err)

local item = Mimir.Dataset.get(1)
log("first item text_file:", item.text_file)

-- Prépare des séquences tokenisées (utile pour chemins legacy)
local ok_seq, nseq_or_err = Mimir.Dataset.prepare_sequences(128)
assert(ok_seq, nseq_or_err)
```

|Appel Lua|Effet|Binding C++|Référence interne|Notes|
|---|---|---|---|---|
|`Mimir.Dataset.load(dir)`|Charge la liste d’items dataset depuis un dossier|`LuaScripting::lua_loadDataset`|`loadDataset(dir)` (`src/Helpers.hpp`), `DatasetItem`|Remplit `ctx.currentDataset`.|
|`Mimir.Dataset.get(i)`|Renvoie un item (chemins + metadata)|`LuaScripting::lua_getDataset`|`ctx.currentDataset[i]`|Retourne `text` si déjà chargé en mémoire.|
|`Mimir.Dataset.prepare_sequences(seq_len)`|Tokenise + pad/trunc le texte en séquences|`LuaScripting::lua_prepareSequences`|`DatasetItem::loadText`, `Tokenizer::tokenize`, pad/trunc|Remplit `ctx.currentSequences`.|

---

## `Mimir.Memory`

> Les bindings mémoire sont “best-effort” et dépendent de l’implémentation côté C++.

### Exemple (script) — `Mimir.Memory`

```lua
Mimir.Memory.config({ max_ram_gb = 8 })
Mimir.Memory.print_stats()

local usage = Mimir.Memory.get_usage()
log("ram usage:", usage)
```

|Appel Lua|Effet|Binding C++|Référence interne|Notes|
|---|---|---|---|---|
|`Mimir.Memory.config(tbl)`|Configure le gestionnaire RAM (seuils/limites/options)|`LuaScripting::lua_memoryConfig`|`AdvancedRAMManager`||
|`Mimir.Memory.get_stats()` / `getStats()`|Renvoie des stats mémoire (table)|`LuaScripting::lua_memoryGetStats`|`AdvancedRAMManager`||
|`Mimir.Memory.print_stats()` / `printStats()`|Affiche les stats mémoire|`LuaScripting::lua_memoryPrintStats`|`AdvancedRAMManager`||
|`Mimir.Memory.clear()`|Reset/clear l’état mémoire (selon implémentation)|`LuaScripting::lua_memoryClear`|`AdvancedRAMManager`||
|`Mimir.Memory.get_usage()` / `getUsage()`|Renvoie l’usage courant|`LuaScripting::lua_memoryGetUsage`|`AdvancedRAMManager`||
|`Mimir.Memory.set_limit(bytes)` / `setLimit()`|Fixe une limite (octets)|`LuaScripting::lua_memorySetLimit`|`AdvancedRAMManager`||

---

## `Mimir.Guard` et `Mimir.MemoryGuard`

### Exemple (script) — `Mimir.Guard` / `Mimir.MemoryGuard`

```lua
Mimir.Guard.set_limit(8 * 1024 * 1024 * 1024)
Mimir.Guard.print_stats()

log("guard usage:", Mimir.MemoryGuard.getCurrentUsage())
log("guard peak:", Mimir.MemoryGuard.getPeakUsage())
```

|Appel Lua|Effet|Binding C++|Référence interne|Notes|
|---|---|---|---|---|
|`Mimir.Guard.set_limit(bytes)` / `setLimit()`|Impose une limite stricte (enforcement)|`LuaScripting::lua_guardSetLimit`|`MemoryGuard`||
|`Mimir.Guard.get_stats()` / `getStats()`|Renvoie stats/compteurs du guard|`LuaScripting::lua_guardGetStats`|`MemoryGuard`||
|`Mimir.Guard.print_stats()` / `printStats()`|Affiche stats/compteurs|`LuaScripting::lua_guardPrintStats`|`MemoryGuard`||
|`Mimir.Guard.reset()`|Remet les compteurs à zéro|`LuaScripting::lua_guardReset`|`MemoryGuard`||
|`Mimir.MemoryGuard.getCurrentUsage()`|Renvoie l’usage courant|`LuaScripting::lua_memoryguardGetCurrentUsage`|`MemoryGuard`||
|`Mimir.MemoryGuard.getPeakUsage()`|Renvoie le pic d’usage|`LuaScripting::lua_memoryguardGetPeakUsage`|`MemoryGuard`||
|`Mimir.MemoryGuard.getLimit()`|Renvoie la limite|`LuaScripting::lua_memoryguardGetLimit`|`MemoryGuard`||
|`Mimir.MemoryGuard.getStats()` / `printStats()` / `reset()`|Alias modernes vers les fonctions `Mimir.Guard.*`|`LuaScripting::lua_guardGetStats` / `lua_guardPrintStats` / `lua_guardReset`|`MemoryGuard`|API plus “OO”.|

---

## `Mimir.Allocator`

### Exemple (script) — `Mimir.Allocator`

```lua
Mimir.Allocator.configure({ max_ram_gb = 8 })
Mimir.Allocator.print_stats()
```

|Appel Lua|Effet|Binding C++|Référence interne|Notes|
|---|---|---|---|---|
|`Mimir.Allocator.configure(opts)`|Configure l’allocateur dynamique|`LuaScripting::lua_allocatorConfigure`|`DynamicTensorAllocator`||
|`Mimir.Allocator.get_stats()` / `getStats()`|Renvoie les stats de l’allocateur|`LuaScripting::lua_allocatorGetStats`|`DynamicTensorAllocator`||
|`Mimir.Allocator.print_stats()` / `printStats()`|Affiche les stats de l’allocateur|`LuaScripting::lua_allocatorPrintStats`|`DynamicTensorAllocator`||

---

## `Mimir.Htop`

### Exemple (script) — `Mimir.Htop`

```lua
Mimir.Htop.create()
Mimir.Htop.enable(true)
Mimir.Htop.update({ step = 1, loss = 0.123 })
Mimir.Htop.render()
```

|Appel Lua|Effet|Binding C++|Référence interne|Notes|
|---|---|---|---|---|
|`Mimir.Htop.create()`|Crée l’objet monitor|`LuaScripting::lua_htopCreate`|`HtopDisplay`||
|`Mimir.Htop.update(tbl)`|Met à jour des métriques|`LuaScripting::lua_htopUpdate`|`HtopDisplay`||
|`Mimir.Htop.render()`|Rend l’affichage dans la console|`LuaScripting::lua_htopRender`|`HtopDisplay`||
|`Mimir.Htop.clear()`|Efface l’affichage|`LuaScripting::lua_htopClear`|`HtopDisplay`||
|`Mimir.Htop.enable(bool)`|Active/désactive le rendu|`LuaScripting::lua_htopEnable`|`HtopDisplay`||

---

## `Mimir.Viz` / `Mimir.visualiser`

### Exemple (script) — `Mimir.Viz`

```lua
Mimir.Viz.create()
Mimir.Viz.set_enabled(true)
Mimir.Viz.initialize()

while Mimir.Viz.is_open() do
  Mimir.Viz.process_events()
  Mimir.Viz.update_metrics({ step = 1, loss = 0.123 })
  Mimir.Viz.update()
end
```

|Appel Lua|Effet|Binding C++|Référence interne|Notes|
|---|---|---|---|---|
|`Mimir.Viz.create()`|Crée le monitor async + visualiser|`LuaScripting::lua_vizCreate`|`AsyncMonitor`, `Visualizer`||
|`Mimir.Viz.initialize()`|Initialise la fenêtre SFML|`LuaScripting::lua_vizInitialize`|`AsyncMonitor` / SFML||
|`Mimir.Viz.is_open()`|Indique si la fenêtre est ouverte|`LuaScripting::lua_vizIsOpen`|`Visualizer`||
|`Mimir.Viz.process_events()`|Traite les events (input/close)|`LuaScripting::lua_vizProcessEvents`|`Visualizer`||
|`Mimir.Viz.update()`|Tick/update du monitor|`LuaScripting::lua_vizUpdate`|`AsyncMonitor`||
|`Mimir.Viz.add_image(label, w, h, pixels)`|Ajoute une image à afficher|`LuaScripting::lua_vizAddImage`|`Visualizer`||
|`Mimir.Viz.update_metrics(tbl)`|Met à jour les métriques UI|`LuaScripting::lua_vizUpdateMetrics`|`Visualizer`||
|`Mimir.Viz.add_loss_point(step, loss)`|Ajoute un point à la courbe de loss|`LuaScripting::lua_vizAddLossPoint`|`Visualizer`||
|`Mimir.Viz.clear()`|Clear l’affichage|`LuaScripting::lua_vizClear`|`Visualizer`||
|`Mimir.Viz.set_enabled(bool)`|Active/désactive la viz (arrêt du monitor si `false`)|`LuaScripting::lua_vizSetEnabled`|`AsyncMonitor::stop`|Pas de `setEnabled` côté viz → stop monitor si `false`.|
|`Mimir.Viz.save_loss_history(path)`|Sauvegarde l’historique de loss|`LuaScripting::lua_vizSaveLossHistory`|`Visualizer::saveLossHistory`|Dump CSV/JSON selon implémentation.|

---

## `Mimir.Layers` (ops)

|Appel Lua|Effet|Binding C++|Référence interne|Notes|
|---|---|---|---|---|
|`Mimir.Layers.conv2d(...)`|Stub : renvoie “Non implémenté”|`LuaScripting::lua_computeConv2D`|—||
|`Mimir.Layers.linear(...)`|Stub : renvoie “Non implémenté”|`LuaScripting::lua_computeLinear`|—||
|`Mimir.Layers.maxpool2d(...)`|Stub : renvoie “Non implémenté”|`LuaScripting::lua_computeMaxPool2D`|—||
|`Mimir.Layers.avgpool2d(...)`|Stub : renvoie “Non implémenté”|`LuaScripting::lua_computeAvgPool2D`|—||
|`Mimir.Layers.activation(...)`|Stub : renvoie “Non implémenté”|`LuaScripting::lua_computeActivation`|—||
|`Mimir.Layers.batchnorm(...)`|Stub : renvoie “Non implémenté”|`LuaScripting::lua_computeBatchNorm`|—||
|`Mimir.Layers.layernorm(...)`|Stub : renvoie “Non implémenté”|`LuaScripting::lua_computeLayerNorm`|—||
|`Mimir.Layers.attention(...)`|Stub : renvoie “Non implémenté”|`LuaScripting::lua_computeAttention`|—||

---

## `Mimir.NeuroPulse`

### Exemple (script) — `Mimir.NeuroPulse`

```lua
local out = Mimir.NeuroPulse.render("salut")
log("neuropulse out:", out)

local p = Mimir.NeuroPulse.params()
write_json("./neuropulse_params.json", p)
```

|Appel Lua|Effet|Binding C++|Référence interne|Notes|
|---|---|---|---|---|
|`Mimir.NeuroPulse.render(text, opts?)`|Rend/convertit un texte en sortie NeuroPulse|`LuaScripting::lua_neuropulseRender`|`NeuroPulseModel`||
|`Mimir.NeuroPulse.params()`|Expose paramètres/support|`LuaScripting::lua_neuropulseParams`|`NeuroPulseModel`||
