# Visualizer: Tips Et Nouvelles Features

## Pour qui

Developpeur framework (C++/runtime/monitoring/UI SFML).

## Objectif

Savoir brancher correctement les tips (viz taps), et ajouter/configurer/implementer
une nouvelle feature Viz sans casser le run.

## Avant de commencer

Comprendre les chapitres Dev 01, 04 et 05.

## Resultat attendu

Tu peux:

1. ajouter un nouveau tip visuel dans le flux training,
2. etendre l'UI Visualizer (rendu, raccourcis, options),
3. garder un comportement stable et backward-compatible.

## 1. Pipeline Viz dans Mimir

Chemin de donnees (vue simplifiee):

1. `Model` calcule le forward.
2. `Model` produit des `VizFrame` (tips) quand `viz_taps_enabled_` est actif.
3. `AsyncMonitor` recupere les frames et les passe au `Visualizer`.
4. `Visualizer::setLayerBlockImages(...)` convertit en `ImageData`.
5. `Visualizer` cree/rebuild les textures SFML puis rend les panels.

Fichiers de reference:

- `src/Model.cpp` (generation des viz taps)
- `src/AsyncMonitor.hpp` (transport async des metriques + frames)
- `src/Visualizer.hpp`
- `src/Visualizer.cpp`

## 2. Comment brancher un tip

### 2.1 Produire la frame cote modele

Dans la section viz taps du forward (`src/Model.cpp`):

1. construire un `VizFrame` valide,
2. remplir `pixels`, `w`, `h`, `channels`, `label`,
3. optionnel: `pixels_real` (version naturelle pour mode non-heatmap),
4. pousser avec `addVizTapFrame(std::move(vf))`.

Champs minimaux:

- `w > 0`, `h > 0`
- `channels` dans `{1,3,4}`
- `pixels.size() == w*h*channels`
- `label` stable et explicite

### 2.2 Conventions de label (important)

Le label pilote le tri/couleur/section UI.
Utiliser une forme hiérarchique stable, par exemple:

`<model>/blocks/<path>/<type>`

Exemples:

- `vae_conv/blocks/dec/out/Conv2d`
- `vae_conv/latent/mu`
- `vae_conv/err/resdiff_abs`

Bonnes pratiques:

- garder des noms predictibles,
- eviter les labels ambigus,
- ne pas reutiliser un label pour un contenu semantiquement different.

## 3. Ajouter une feature Viz (rendu/comportement)

### 3.1 Choisir le bon niveau

- Generation de donnees: `Model.cpp`
- Transport async: `AsyncMonitor.hpp`
- Rendu/interactions: `Visualizer.cpp/.hpp`

Regle cle:

- Une feature purement UI doit rester dans `Visualizer`.
- Ne pas deplacer inutilement de logique UI dans `Model`.

### 3.2 Ajouter un nouvel etat UI

1. ajouter l'etat dans `Visualizer.hpp` (bool, enum, valeur),
2. initialiser dans le constructeur `Visualizer::Visualizer(...)`,
3. brancher la persistance dans `serializeUILayout()` / `applyUILayout()`,
4. afficher l'etat (badge/texte) dans `drawPanelChrome(...)` si utile.

### 3.3 Ajouter un raccourci clavier

Dans `Visualizer::processEvents()`:

1. intercepter la touche,
2. modifier l'etat,
3. appeler le rebuild adequat (`rebuildLayerBlockTextures()` ou `rebuildAllTextures()`),
4. sauvegarder le layout (`saveUILayoutToLast()`) si l'etat est persistant,
5. documenter la touche dans `renderHelpOverlay()`.

### 3.4 Mettre a jour le rendu

Selon la feature:

- creation/traitement pixels: `setLayerBlockImages(...)`
- upload textures: `createLayerBlockTexture(...)`, `createImageTexture(...)`
- rendu panel: `renderLayerBlocks()`, `renderTrainingProgress()`, etc.

## 4. Configurer une nouvelle feature

### 4.1 Runtime config (JSON modelConfig)

Si la feature depend du modele/training:

- ajouter une cle de config dans le chemin de lecture approprie,
- conserver un default stable,
- ne pas casser les anciens configs.

### 4.2 UI config (`viz_ui_settings.json`)

Si la feature est pure UI:

- persister via `serializeUILayout()`,
- relire via `applyUILayout()`,
- valider les bornes (`clamp`) a la lecture.

## 5. Checklist de securite (anti-regressions)

1. Ne jamais interrompre le forward pour un besoin Viz.
2. Filtrer une preview dans le chemin Viz, pas dans l'execution du layer.
3. Verifier les dimensions/canaux avant upload texture.
4. Eviter les interpretations de layout implicites non verifiees (CHW/HWC).
5. Pour toute interaction clavier, verifier l'effet immediat sans attendre la step suivante.
6. Compiler et verifier les erreurs avant merge.

## 6. Recette concrete: ajouter une feature de palette

Exemple de sequence complete:

1. Ajouter l'etat (enum palette) dans `Visualizer.hpp`.
2. Ajouter la conversion palette -> couleur dans `Visualizer.cpp`.
3. Appliquer la palette dans `setLayerBlockImages(...)` / rebuild.
4. Ajouter la touche (`K`) dans `processEvents()`.
5. Persister la palette dans le layout UI.
6. Ajouter un badge `PAL ...` dans le header du panel.
7. Documenter dans l'overlay aide et dans la doc utilisateur.

## 7. Validation rapide avant commit

Commandes conseillees:

```bash
cmake --build build --target mimir --parallel
```

Verification manuelle:

1. la Viz s'ouvre et reste fluide,
2. les tips cibles apparaissent dans la bonne section,
3. les toggles clavier ont un effet immediat,
4. la progression training reste coherent (global + batch),
5. pas d'artefact de texture evident (bande, repetition, couleurs incoherentes).
