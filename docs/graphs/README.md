# Schémas de la documentation

Les 29 fichiers SVG numérotés illustrent les pages actives du framework. Ils
sont des schémas conceptuels relus contre les sources, pas des captures de
fenêtre ni des dumps d’un modèle avec une configuration implicite.

Les images conservent leurs noms pour préserver les liens des pages. Elles
utilisent une typographie commune, des retours à la ligne sans troncature,
des branches de contrôle explicites et des métadonnées de provenance.

## Lire un schéma dans son contexte

- Les architectures natives sont des classes C++ dérivées de `Model` ;
  Lua et JSON pilotent leur configuration et leur exécution.
- MPK est une alternative expérimentale, pas entièrement exploitable en production.
- `CPU-first` désigne la référence et le repli, pas une priorité CPU imposée
  avant tous les autres runtimes.
- Les blocs optionnels dépendent de la configuration de la classe native.
- Les familles de modèles ne sont pas des mesures de performance ou de qualité.
  Le schéma de comparaison ne présente pas de nombre de paramètres sans config.
- `sd3_5` est présenté comme un squelette/placeholder, pas un pipeline SD3.5 complet.

Chaque SVG possède un titre, une description accessible et des métadonnées
contenant ses sources et les pages auxquelles il se rapporte. Les noms des
sources sont également visibles sous l’image. Les couleurs regroupent des
responsabilités ; les badges textuels précisent les statuts expérimentaux ou
de diagnostic sans dépendre uniquement de la couleur.

## Régénérer les images

Python 3.10 ou plus récent suffit ; le générateur ne dépend d’aucun paquet tiers :

```bash
python3 scripts/docs/generate_diagrams.py --bundle
python3 scripts/docs/generate_diagrams.py --check --bundle
python3 scripts/tools/verify_docs.py
```

Les contenus éditables sont séparés de la mise en page :

- [Architecture, registre et publication](../../scripts/docs/diagrams_architecture.json)
- [Runtime, mémoire et cycle de vie](../../scripts/docs/diagrams_runtime.json)
- [Modèles et comparaison](../../scripts/docs/diagrams_models.json)
- [Générateur SVG](../../scripts/docs/generate_diagrams.py)

Modifiez le manifeste correspondant, vérifiez ses affirmations dans les sources
indiquées, puis régénérez les SVG. `--check` vérifie les références locales, les
noms et positions des blocs, les connexions et l’identité des fichiers générés.
Il ne remplace pas la relecture du contenu ni le contrôle visuel des flèches.

Pour prévisualiser les images ailleurs sans modifier celles de la documentation :

```bash
python3 scripts/docs/generate_diagrams.py --output-dir /tmp/mimir-doc-diagrams
```

L’option `--bundle` produit aussi l’archive locale
`mimir_framework_svg_pack_v3_corrected.zip` à partir des mêmes contenus. Cette
sortie générée est ignorée par Git. Les logos, badges de CI, rapports issus d’un
run particulier et images des archives historiques sont distincts de ces schémas.

## Repères

| Sujet | Images |
| --- | --- |
| Architecture et responsabilités | `00`, `01`, `02`, `04`, `06`, `16`, `24`, `26` |
| Registre et familles de couches | `05`, `13` |
| Cycle de vie, entraînement et inférence | `03`, `11`, `12` |
| Exécution, mémoire et matériel | `07`, `08`, `14`, `15`, `27` |
| Checkpoints, inspection et export | `09`, `10`, `25` |
| Structures des modèles et comparaison | `17` à `23` |
| Contribution et publication de la documentation | `28` |

Voir le [contrat des architectures](../07-Devs/08-Architecture-Contract.md) et
les [parcours de tutoriels](../08-Tuto/00-INDEX.md).

## Étapes suivantes

- [Vue d’ensemble du moteur](../04-Architecture-Internals/01-Engine-Overview.md)
- [Exemple de rapport d’un modèle identifié](analyze-model-report.md)
- [Index de la documentation](../00-INDEX.md)
