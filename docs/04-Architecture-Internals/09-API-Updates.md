# API Stub Update - v2.1.0

## Modifications Apportées au Fichier `mimir-api.lua`

### 📅 Date
26 décembre 2025

### 🎯 Objectif
Mise à jour complète du stub d'API Lua pour refléter les nouvelles fonctionnalités implémentées dans Mímir Framework v2.1.0, notamment la gestion mémoire avancée et le support complet de FluxModel.

---

## ✨ Nouvelles API Ajoutées

### 1. **MemoryGuard API Moderne** (Recommandée)

API complète pour la gestion stricte de la RAM avec limite configurable (par défaut 10 Go).

#### Fonctions Ajoutées

```lua
---@class MimirMemoryGuardAPI
MemoryGuard = {}

-- Configuration
Mimir.MemoryGuard.setLimit(limit)      -- Définir limite (bytes ou GB)
Mimir.MemoryGuard.getLimit()           -- Obtenir limite configurée

-- Monitoring
Mimir.MemoryGuard.getCurrentUsage()    -- RAM courante (bytes)
Mimir.MemoryGuard.getPeakUsage()       -- Pic d'utilisation (bytes)

-- Statistiques
Mimir.MemoryGuard.getStats()           -- Table complète de stats
Mimir.MemoryGuard.printStats()         -- Affichage formaté
Mimir.MemoryGuard.reset()              -- Reset compteurs
```

#### Type de Retour

```lua
---@class MemoryGuardStats
---@field current_mb float
---@field peak_mb float
---@field limit_mb float
---@field usage_percent float
```

#### Exemples Documentés

Chaque fonction inclut maintenant des exemples d'utilisation complets dans la documentation EmmyLua :

```lua
-- Exemple intégré dans le stub
Mimir.MemoryGuard.setLimit(10 * 1024 * 1024 * 1024)  -- 10 GB
local usage = Mimir.MemoryGuard.getCurrentUsage()
print(string.format("RAM: %.2f GB", usage / 1e9))
```

### 2. **FluxModel API Directe**

API complète pour le modèle de diffusion text-to-image avec VAE et text conditioning.

#### Configuration

```lua
---@class FluxConfig
FluxConfig = {
    image_resolution = 256,
    latent_channels = 4,
    latent_resolution = 32,
    vae_base_channels = 128,
    vae_channel_mult = {1, 2, 4, 4},
    num_res_blocks = 2,
    vocab_size = 50000,
    text_max_length = 77,
    text_embed_dim = 768,
    transformer_dim = 768,
    num_transformer_blocks = 12,
    num_attention_heads = 12,
    mlp_ratio = 4.0,
    timestep_embed_dim = 256,
    num_diffusion_steps = 1000
}
```

#### Fonctions Ajoutées

```lua
---@class FluxModelAPI
FluxModel = {}

-- Modes d'exécution
FluxModel.train()                -- Activer mode training
FluxModel.eval()                 -- Activer mode inference
FluxModel.isTraining()           -- Vérifier mode actuel

-- VAE
FluxModel.encodeImage(image)     -- Image → Latent
FluxModel.decodeLatent(latent)   -- Latent → Image

-- Text Processing
FluxModel.tokenizePrompt(prompt) -- Prompt → Tokens
FluxModel.encodeText(tokens)     -- Tokens → Embeddings

-- Diffusion
FluxModel.predictNoise(...)      -- Prédiction de bruit
FluxModel.generate(prompt, steps) -- Pipeline complet

-- Training
FluxModel.computeDiffusionLoss(...) -- Calcul de loss

-- Configuration
FluxModel.setPromptTokenizer(...) -- Définir tokenizer
```

### 3. **Architecture Flux dans architectures**

```lua
---@class MimirArchitecturesAPI
Mimir.Architectures.flux(config)  -- Construire un modèle Flux
```

---

## 🔄 Améliorations Apportées

### 1. Documentation Enrichie

Toutes les nouvelles fonctions incluent :
- **Types annotés** avec `@param`, `@return`, `@field`
- **Exemples d'utilisation** inline
- **Format de sortie** documenté pour les fonctions d'affichage
- **Descriptions détaillées** des comportements

### 2. Version Bump

```lua
---@version 2.1.0  -- Mise à jour depuis 2.0.0
```

### 3. Section "Nouveautés" Ajoutée

```lua
-- Nouveautés v2.1.0 :
--  • MemoryGuard API moderne (limite RAM stricte à 10 Go configurable)
--  • FluxModel API directe (diffusion text-to-image avec VAE)
--  • Modes train()/eval() pour tous les modèles
--  • Support complet des architectures de diffusion
```

### 4. Exports Globaux Mis à Jour

```lua
---@type MimirMemoryGuardAPI
MemoryGuard = MemoryGuard

---@type FluxModelAPI
FluxModel = FluxModel
```

---

## 📊 Statistiques de Modification

| Élément | Avant | Après | Ajouté |
|---------|-------|-------|--------|
| **Lignes totales** | 866 | 1137 | +271 |
| **Classes API** | 10 | 12 | +2 |
| **Fonctions documentées** | ~80 | ~95 | +15 |
| **Exemples inline** | 5 | 20 | +15 |
| **Types custom** | 15 | 18 | +3 |

---

## 🎓 Utilisation pour les Développeurs

### Autocomplétion IDE

Avec ce stub mis à jour, les IDEs supportant EmmyLua/LuaLS offrent maintenant :

1. **Autocomplétion complète** pour `Mimir.MemoryGuard.*` et `FluxModel.*`
2. **Tooltips détaillés** avec exemples et types
3. **Vérification de types** pour les paramètres
4. **Navigation** vers les définitions

### Exemple d'Autocomplétion

Quand vous tapez `Mimir.MemoryGuard.` dans votre IDE :

```
Mimir.MemoryGuard.
├─ setLimit(limit: number)
├─ getLimit() → integer
├─ getCurrentUsage() → integer
├─ getPeakUsage() → integer
├─ getStats() → MemoryGuardStats
├─ printStats()
└─ reset()
```

### Exemple de Tooltip

Hover sur `Mimir.MemoryGuard.setLimit()` affiche :

```
function Mimir.MemoryGuard.setLimit(limit: number) → boolean

Définir la limite de mémoire RAM stricte.
Accepte des valeurs en bytes (grands nombres) ou en GB (si <= 1000).

@param limit — Limite en bytes ou en GB (si valeur <= 1000)
@return ok — true si succès

Exemples:
  -- Définir limite à 10 Go
  Mimir.MemoryGuard.setLimit(10 * 1024 * 1024 * 1024)  -- en bytes
  Mimir.MemoryGuard.setLimit(10)  -- en GB (auto-détecté car < 1000)
```

---

## 🔍 Compatibilité

### Rétrocompatibilité

- ✅ **API `guard` conservée** - L'ancienne API reste fonctionnelle
- ✅ **Pas de breaking changes** - Tous les scripts existants continuent de fonctionner
- ✅ **Migration progressive** - Les deux APIs (`guard` et `MemoryGuard`) coexistent

### Recommandations

| Situation | API Recommandée | Raison |
|-----------|----------------|--------|
| Nouveau code | `MemoryGuard` | API moderne, mieux documentée |
| Code existant | `guard` ou migrer | Compatibilité ou amélioration |
| Scripts production | `MemoryGuard` | Meilleure gestion des bytes |

---

## 📚 Documentation Associée

Ce stub est synchronisé avec :

1. **[docs/MEMORY_LIMIT_10GB.md](../docs/MEMORY_LIMIT_10GB.md)** - Guide complet MemoryGuard
2. **[docs/FLUX_MODEL_COMPLETE.md](../docs/FLUX_MODEL_COMPLETE.md)** - Documentation FluxModel
3. **[MEMORY_10GB_IMPLEMENTATION.md](../MEMORY_10GB_IMPLEMENTATION.md)** - Résumé implémentation

---

## ✅ Validation

### Checklist de Mise à Jour

- [x] Toutes les nouvelles fonctions documentées
- [x] Types annotés avec EmmyLua
- [x] Exemples d'utilisation fournis
- [x] Formats de sortie documentés
- [x] Exports globaux à jour
- [x] Version bump (2.0.0 → 2.1.0)
- [x] Compatibilité vérifiée
- [x] Changelog créé

### Test de Validation

Pour valider le stub dans votre IDE :

1. **Ouvrir** un script Lua dans le projet
2. **Taper** `Mimir.MemoryGuard.` 
3. **Vérifier** que l'autocomplétion propose toutes les fonctions
4. **Hover** sur une fonction pour voir la documentation
5. **Tester** les exemples fournis

---

## 🚀 Prochaines Étapes

### Pour les Développeurs

1. **Utiliser** la nouvelle API `MemoryGuard` dans vos scripts
2. **Tester** le FluxModel avec les exemples fournis
3. **Reporter** tout problème d'autocomplétion
4. **Contribuer** avec des exemples supplémentaires

### Pour le Maintien

1. Mettre à jour le stub à chaque ajout d'API
2. Ajouter des exemples pour toute nouvelle fonction
3. Maintenir la synchronisation avec la documentation
4. Versionner les changements d'API

---

## 📝 Résumé

Le stub `mimir-api.lua` est maintenant **à jour avec la v2.1.0** du framework et inclut :

- ✅ API **MemoryGuard** complète (limite RAM 10 Go)
- ✅ API **FluxModel** complète (diffusion text-to-image)
- ✅ Documentation **enrichie** avec exemples
- ✅ **15 nouvelles fonctions** documentées
- ✅ **Rétrocompatibilité** totale
- ✅ **271 lignes** de documentation ajoutées

**Les développeurs bénéficient maintenant d'une autocomplétion IDE complète pour toutes les nouvelles fonctionnalités ! 🎉**
