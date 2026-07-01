# Style editorial de la documentation Mimir

## Objectif

Unifier le ton et la structure des pages pour qu'un lecteur trouve vite:

1. si la page le concerne,
2. ce qu'il va accomplir,
3. ce qu'il doit preparer,
4. ce qu'il doit obtenir a la fin.

## Structure minimale obligatoire (toutes les pages)

Chaque page commence par:

- `## Pour qui`
- `## Objectif`
- `## Avant de commencer`
- `## Résultat attendu`

## Regle de lisibilite

- Phrases courtes.
- Une idee principale par paragraphe.
- Preferer des listes d'etapes aux gros blocs de texte.
- Exemples commandes directement executables.

## Regle debutant (01-Getting-Started, 02-User-Guide)

- Vocabulaire non jargonne autant que possible.
- Expliquer les acronymes la premiere fois.
- Donner au moins un chemin "copier-coller" qui marche.
- Toujours indiquer quoi faire si la commande echoue.

## Regle technique (03+)

- Precision d'abord: signatures, prerequis, conventions.
- Pointer vers les fichiers source de verite (C/C++/Lua).
- Signaler explicitement les APIs legacy/depreciees.

## Convention I/O du framework

- Entree par defaut: `__input__`
- Entree texte dediee: `text_ids`
- Sortie principale: `x`

## Politique legacy

- Une API depreciee doit etre marquee "depreciee/obsolete" dans la page qui la cite.
- Les sections debutantes ne doivent pas recommander une API depreciee.
