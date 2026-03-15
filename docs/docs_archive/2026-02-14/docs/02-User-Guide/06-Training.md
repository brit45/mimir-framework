# Training (Entraînement)

Cette section décrit le support actuel de l’entraînement dans le framework **Mímir**.

⚠️ **Important — état actuel**  
Le support de l’entraînement est **en cours de développement**.  
La boucle d’entraînement, la gestion des paramètres et l’allocation des gradients sont fonctionnelles,  
mais **toutes les opérations ne disposent pas encore d’un backward complet**.

Mímir privilégie actuellement :

- la **stabilité du runtime CPU-first**
- la **cohérence structurelle des modèles**
- la **gestion mémoire explicite**

avant la généralisation de l’autograd.

---

## Ce qui est actuellement supporté

- Boucle d’entraînement (`model.train`)
- Allocation des gradients par blocs de poids
- Initialisation des poids (He / Xavier)
- Calcul de pertes simples (ex: MSE, Cross-Entropy basique)
- Mise à jour des paramètres (optimisation simple)

Ces mécanismes permettent :

- des tests d’apprentissage,
- des démonstrations fonctionnelles,
- des validations de pipeline.

---

## Limitations actuelles

Les opérations suivantes peuvent être :

- **structurelles uniquement**
- ou disposer d’un forward sans backward complet

Selon la version :

- certaines couches avancées (attention complète, diffusion, etc.)
- certaines normalisations complexes

Ces limitations sont **connues et assumées**.  
Elles seront levées progressivement à mesure que les kernels CPU sont consolidés.

---

## Philosophie

Mímir ne cherche pas à fournir immédiatement :

- un moteur d’entraînement universel,
- ni un remplacement de frameworks GPU-first.

L’objectif est de construire un moteur :

- **compréhensible**
- **auditable**
- **contrôlable**
- **optimisé CPU**

avant d’élargir le support complet de l’autograd.

---

## Exemple minimal

```lua
-- API actuelle (v2.3):
--   Mimir.Model.train(epochs, learning_rate)
Mimir.Model.train(10, 1e-3)
```

---

## Squelette recommandé (script d'entraînement)

Ce squelette correspond à l'API réellement exposée par `src/LuaScripting.cpp`.

```lua
-- 0) Allocator / limites mémoire (recommandé)
Mimir.Allocator.configure({ max_ram_gb = 10.0, enable_compression = true })

-- 1) Dataset
local ok_ds, ds_err = Mimir.Dataset.load("dataset")
if not ok_ds then error(ds_err) end

local max_seq_len = 256
local ok_prep, prep_err = Mimir.Dataset.prepare_sequences(max_seq_len)
if not ok_prep then error(prep_err) end

-- 2) Modèle
local cfg, err = Mimir.Architectures.default_config("transformer")
if not cfg then error(err) end
cfg.max_seq_len = max_seq_len

local ok_create, create_err = Mimir.Model.create("transformer", cfg)
if not ok_create then error(create_err) end

-- build() = rebuild compat (ne remplace pas allocate/init)
local ok_build, build_err = Mimir.Model.build()
if not ok_build then error(build_err) end

local ok_alloc, alloc_info = Mimir.Model.allocate_params()
if not ok_alloc then error(alloc_info) end

local ok_init, init_err = Mimir.Model.init_weights("xavier", 42)
if ok_init == false then error(init_err) end

-- 3) Entraînement
local epochs = 10
local lr = 1e-4
local ok_train, train_err = Mimir.Model.train(epochs, lr)
if ok_train == false then error(train_err) end

-- 4) Sauvegarde (recommandé v2.3+)
Mimir.Serialization.save("checkpoints/run.safetensors", "safetensors")
```
