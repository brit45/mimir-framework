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
model.train{
    epochs = 10,
    batch_size = 16,
    learning_rate = 1e-3
}
```
