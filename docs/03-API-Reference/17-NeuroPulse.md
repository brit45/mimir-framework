# API : `Mimir.NeuroPulse`

## Pour qui

Développeur et utilisateur intermédiaire/avancé.

## Objectif

Trouver rapidement le contrat API réel et les paramètres utilisables.

## Avant de commencer

Connaître les commandes de base de Mímir.

## Résultat attendu

Tu peux appeler l'API sans ambiguïté de signature ou de comportement.


NeuroPulse est un module “texte → paramètres neuro/audio/lumière”, avec rendu optionnel.

Source : `src/scriptings/Lua/luaScripting/LuaScripting.cpp`, `src/Models/NeuroPulse.*`.

## `params(text: string, opts?: table) -> table`

Retourne une table de métadonnées/paramètres (sha256, band, fréquences, etc.).

## `render(text: string, opts?: table) -> (bool, meta|err)`

Rend des fichiers (WAV / CSV lumière) et retourne un `meta` JSON.

Options (exemples) :

- `duration_s`, `sample_rate`
- `carrier_hz`, `binaural_hz`
- `safe_light`, `light_fps`, `light_hz`
- `out_wav`, `out_light_csv`
- `organic_nn`, `nn_*`

Voir `scripts/examples/neuropulse_demo.lua`.
