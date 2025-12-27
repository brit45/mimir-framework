# Pipeline API

La Pipeline API fournit une couche d’orchestration permettant de :
- simplifier l’utilisation du framework,
- enchaîner build / train / infer / save,
- standardiser les workflows expérimentaux.

---

## Statut actuel

⚠️ **La Pipeline API est expérimentale.**

Elle est conçue pour :
- les démonstrations,
- les benchmarks,
- le prototypage rapide.

Certaines implémentations internes peuvent :
- évoluer,
- être refactorées,
- changer de comportement entre versions mineures.

---

## Objectif

La Pipeline API vise à :
- réduire le code Lua nécessaire,
- fournir une interface cohérente,
- préparer une API plus stable à long terme.

Elle **ne remplace pas** l’API bas niveau `model.*`.

---

## Recommandation

Pour un contrôle maximal :
- utiliser directement `model.create`, `model.build`, `model.train`, etc.

La Pipeline API est idéale pour :
- scripts rapides,
- exemples,
- tests automatisés.

---

## Évolutions prévues

- Normalisation des retours (`true / false, error`)
- Réduction du code dupliqué
- Stabilisation progressive

Voir : `06-Contributing/05-Roadmap.md`
