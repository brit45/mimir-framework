# API : `Mimir.Layers` (ops)

`Mimir.Layers` expose des opérations “standalone” (hors graphe) pour des tests rapides.

Statut (important) : dans la version actuelle, **toutes ces fonctions sont des stubs** côté C++ et renvoient :

- `(false, "Non implémenté - utilisez model.forward() à la place")`

Source de vérité : `src/LuaScripting.cpp` (section “Layer Operations API”).

## Exemple (script) — tester le statut

```lua
local ok, err = Mimir.Layers.linear({})
if not ok then
  log("Layers.linear:", err)
  log("Astuce: utilisez un modèle + Mimir.Model.forward()")
end
```

## Référence (appel + effet)

|Appel Lua|Effet|Retour|Notes|
|---|---|---|---|
|`Mimir.Layers.conv2d(...)`|Conv2D standalone (hors graphe)|`(false, err)`|Stub (non implémenté).|
|`Mimir.Layers.linear(...)`|Linear/GEMM standalone (hors graphe)|`(false, err)`|Stub (non implémenté).|
|`Mimir.Layers.maxpool2d(...)`|MaxPool2D standalone (hors graphe)|`(false, err)`|Stub (non implémenté).|
|`Mimir.Layers.avgpool2d(...)`|AvgPool2D standalone (hors graphe)|`(false, err)`|Stub (non implémenté).|
|`Mimir.Layers.activation(...)`|Activation standalone (hors graphe)|`(false, err)`|Stub (non implémenté).|
|`Mimir.Layers.batchnorm(...)`|BatchNorm standalone (hors graphe)|`(false, err)`|Stub (non implémenté).|
|`Mimir.Layers.layernorm(...)`|LayerNorm standalone (hors graphe)|`(false, err)`|Stub (non implémenté).|
|`Mimir.Layers.attention(...)`|Attention standalone (hors graphe)|`(false, err)`|Stub (non implémenté).|

## Alternative recommandée

Pour tester des ops de manière réaliste, le chemin supporté est :

1) `Mimir.Architectures.default_config(...)`
2) `Mimir.Model.create(...)`
3) `Mimir.Model.allocate_params()` + `Mimir.Model.init_weights(...)` (ou `Mimir.Serialization.load(...)`)
4) `Mimir.Model.forward(...)`
