# Modèle causal decoder-only

Mímir fournit deux architectures texte distinctes :

- `transformer`, un Transformer avec embedding interne recevant des IDs de
  tokens dans `__input__`, avec des vecteurs flottants `mag` et `mod` de
  dimension `d_model`, puis une sortie globale de `output_dim` valeurs ;
- `causal_lm`, un modèle de langage decoder-only recevant des identifiants de
  tokens et produisant `seq_len * vocab_size` logits.

Pour entraîner un LLM causal, utilisez `causal_lm`.

## Architecture `causal_lm`

Le builder natif se trouve dans `src/Models/NLP/CausalLMModel.cpp`. Il assemble :

- embeddings de tokens ;
- RMSNorm pré-attention et pré-MLP ;
- attention causale avec RoPE et GQA (`num_kv_heads`) ;
- branche locale RMSNorm → convolution causale 1D par canal → SiLU → projection résiduelle ;
- bloc SwiGLU avec deux projections et une couche native `SwiGLU` à deux entrées ;
- connexions résiduelles ;
- RMSNorm final ;
- tête LM partageant les poids de l'embedding.

Configuration principale :

| Clé | Rôle |
| --- | --- |
| `vocab_size` | Capacité demandée ; après préparation, le graphe utilise le vocabulaire effectivement composé |
| `seq_len` | Longueur du contexte d'entraînement |
| `d_model` | Dimension cachée |
| `num_layers` | Nombre de blocs decoder |
| `num_heads` | Nombre de têtes de requête |
| `num_kv_heads` | Nombre de têtes clé/valeur ; doit diviser `num_heads` |
| `mlp_hidden` | Largeur intermédiaire SwiGLU |
| `causal_arch_version` | Version du graphe : 2 sur les nouveaux modèles, 1 pour les checkpoints historiques |
| `local_conv_kernel` | Taille du noyau causal par canal, 3 par défaut ; 0 désactive la branche |
| `local_conv_dilation` | Espacement entre les positions du noyau, 1 par défaut |
| `fused_swiglu` | `true` : couche native ; `false` : SiLU puis Multiply, pour comparaison ou reprise historique |
| `padding_idx` | Identifiant PAD |
| `norm_eps` | Epsilon RMSNorm |
| `rope_theta` | Base des fréquences RoPE |
| `dtype` | Dtype du modèle |

La configuration fournie est `configs/causal_lm.json` et le script
d'entraînement est `scripts/training/train_causal_lm.lua`.

## Lancer l'entraînement

Le corpus est un fichier texte local, choisi par `--corpus` ou par
`dataset.text_path` dans la configuration. Aucun téléchargement automatique
n'est effectué. Choisissez aussi les sorties pour isoler chaque run :

```bash
./run_mimir.sh --lua scripts/training/train_causal_lm.lua -- \
  --corpus /chemin/corpus.txt \
  --tokenizer checkpoints/causal_demo/tokenizer.json \
  --checkpoint-dir checkpoints/causal_demo \
  --steps 1000 \
  --lr 3e-4
```

Les options sont analysées par `scripts/modules/args.lua`. Les raccourcis
principaux sont `--vocab-size`, `--seq-len`, `--d-model`, `--layers`,
`--heads`, `--kv-heads`, `--mlp-hidden`, `--dtype`, `--steps`, `--lr`,
`--local-conv-kernel`, `--local-conv-dilation`, `--fused-swiglu`,
`--optimizer`, `--amp`, `--amp-dtype`, `--ddp`, `--save-every` et
`--checkpoint-dir`.

Les overrides structurés ont la priorité finale :

```bash
./run_mimir.sh --lua scripts/training/train_causal_lm.lua -- \
  --corpus /chemin/vers/corpus.txt \
  --no-amp --no-ddp \
  --override model.num_layers=8 \
  --override model.num_kv_heads=2
```

AMP et DDP restent désactivés par défaut. L'autocast CPU couvre les couches
Linear; DDP exige un build avec MPI (`-DENABLE_MPI=ON`).

## Tokenizer

Au démarrage, le script délègue la préparation à
`CausalLMModel::prepareTrainingCorpus`, avant l'allocation des paramètres :

1. charge le tokenizer indiqué s'il existe ;
2. ajoute les balises de contrôle nécessaires et vérifie la couverture du
   corpus, y compris les contextes enrichis ;
3. apprend le BPE si le tokenizer est absent ou produit encore des `<UNK>` ;
4. sauvegarde le vocabulaire et les règles BPE ensemble, puis vérifie les
   tokens spéciaux, le PAD et les identifiants du corpus.

Un tokenizer chargé qui couvre déjà ces textes est réutilisé sans deuxième
étude BPE, même si `--bpe-merges` a changé. Un JSON illisible ou des IDs spéciaux
invalides provoquent une erreur explicite. Hors reprise, un vocabulaire qui
dépasse la capacité demandée peut être recomposé depuis le corpus.

Le vocabulaire ne peut pas dépasser la capacité demandée. Avant l'allocation,
le modèle reconstruit son graphe avec la taille du vocabulaire effectif et
conserve la capacité dans `tokenizer_capacity`. Un mot absent est représenté
par ses sous-mots BPE ; il n'est pas nécessaire d'ajouter chaque mot entier.
Si le corpus contient encore des `<UNK>` après composition, la préparation
échoue. Le nombre maximal de merges à apprendre se règle avec
`tokenizer.bpe_merges` ou `--bpe-merges` (défaut : 1000).

Le corpus doit produire plus de `seq_len` tokens.

Pour les conversations structurées avec `Système:`, `Utilisateur:` et
`Assistante:`, les questions et l'historique restent visibles comme contexte,
mais la perte est calculée uniquement sur la réponse de l'assistante et son
token de fin. Les dialogues à plusieurs tours prennent le dernier tour de
l'assistante comme cible. Les corpus texte sans ces marqueurs gardent le mode
classique où tous les tokens sont des cibles.

Les fichiers de corpus locaux sont des données d'entraînement, pas des modèles
préentraînés. Leur présence ne garantit ni la justesse des réponses, ni une
capacité de raisonnement fiable. Évaluez ces propriétés sur des exemples
indépendants après entraînement.

### Séquences et balises de contrôle

La préparation native encadre les exemples avec `<BOS>` et `<EOS>` et normalise
les marqueurs de dialogue vers `<SYSTEM>`, `<USER>` et `<ASSISTANT>`.
`<CONTEXT>`, `</CONTEXT>`, `<REFLECT>` et `<FINAL>` sont aussi des tokens
spéciaux atomiques.
Ils séparent les contenus appris et pilotent le format ; leur nom ne crée pas
à lui seul un processus de réflexion.

`seq_len`, également exposé par `--train-window`, reste la taille des fenêtres
de calcul d'entraînement. Les documents longs ne deviennent pas un seul
forward sans limite : la boucle échantillonne des fenêtres dans le flux encadré
et applique son masque de cibles. L'inférence adapte sa longueur au prompt
complet et au budget de génération ; son coût mémoire reste limité par les
ressources disponibles, notamment pour l'attention quadratique.

## Boucle Lua bas niveau

```lua
local cfg, cfg_err = Mimir.Architectures.default_config("causal_lm")
assert(cfg, cfg_err)
cfg.vocab_size = 256
cfg.seq_len = 16
cfg.d_model = 32
cfg.num_layers = 1
cfg.num_heads = 4
cfg.num_kv_heads = 2
cfg.mlp_hidden = 64

assert(Mimir.Model.create("causal_lm", cfg))
assert(Mimir.Model.allocate_params())
assert(Mimir.Model.init_weights("xavier", 42))

local ids = {}
for i = 1, cfg.seq_len do ids[i] = (i - 1) % cfg.vocab_size end
local logits, err = Mimir.Model.forward({ __input__ = ids }, true)
assert(logits, err)
assert(#logits == cfg.seq_len * cfg.vocab_size)
```

Ce bloc illustre seulement le forward bas niveau. Le script officiel appelle
`Mimir.Model.causal_lm_prepare_corpus(...)`, puis
`Mimir.Model.causal_lm_train(...)`. Le dataset, le masque des cibles, la
cross-entropy, le backward, l'optimiseur, la validation et la mémoire de
conversation sont implémentés dans la classe C++ `CausalLMModel`.

## Validation et reprise

La validation est activée par défaut toutes les 100 étapes, sur au plus huit
items, avec un holdout de 10 %. Les options communes permettent notamment
`--validate-every-steps`, `--validate-items`, `--validate-holdout-frac` et
`--validate-holdout-items`. Les conversations sont séparées par exemple ; avec
un seul texte ou un seul exemple exploitable, la préparation tente une
séparation des tokens cibles. Le champ `validation_source` indique si le
résultat provient d'un holdout indépendant ou des données d'entraînement.
La validation utilise des forwards sans backward ni mise à jour des poids,
et publie cross-entropy, perplexité et précision du prochain token.

```bash
./run_mimir.sh --lua scripts/training/train_causal_lm.lua -- \
  --corpus /chemin/corpus.txt \
  --checkpoint-dir checkpoints/causal_demo --resume true --steps 2000
```

`--steps` désigne le total visé, y compris les étapes déjà effectuées. Le
résolveur cherche les RawFolder `step-*`/`step_*`, `final` et la racine ;
`--resume-from /chemin/checkpoint` sélectionne un checkpoint précis.
La reprise restaure la topologie, le tokenizer, l'état optimiseur, la seed et
la partition de validation. Elle refuse un vocabulaire modifié, un BPE à
réapprendre ou des options incompatibles. DDP exige aussi le même nombre de
processus. Un nouveau corpus ou une nouvelle topologie demande un autre run.

## Générer du texte

L'inférence autoregressive réexécute le forward sur le contexte à chaque token
(pas encore de KV-cache). Le script reconstruit la topologie sérialisée du
checkpoint RawFolder ou SafeTensors natif (`--format safetensors`). Pour un
RawFolder, il utilise son tokenizer sauvegardé par défaut :

```bash
./run_mimir.sh --lua scripts/inferences/causal_lm_generate.lua -- \
  --conf configs/causal_lm.json \
  --checkpoint checkpoints/causal_demo/final \
  --prompt "Explique les réseaux de neurones" \
  --max-new-tokens 100 --temperature 0.8 --top-k 40 --top-p 0.95
```

Pour une conversation au clavier, lancez le mode interactif :

```bash
./run_mimir.sh --lua scripts/inferences/causal_lm_generate.lua -- \
  --conf configs/causal_lm.json \
  --checkpoint checkpoints/causal_demo/final \
  --memory-file checkpoints/causal_demo/conversation_memory.json \
  --interactive --max-new-tokens 96
```

Le chat conserve les tours en JSON et les réinjecte dans le contexte complet,
sans les tronquer à la fenêtre d'entraînement. Par défaut le chemin dépend de
`training.checkpoint_dir` dans la configuration de lancement, avec un repli
historique vers `checkpoints/lina_chat` ; fournissez `--memory-file` pour le
lier explicitement au run. `/memory` affiche le nombre de tours,
`/reset` efface la mémoire et `--no-memory` désactive le stockage. Pour choisir
un autre emplacement, passez `--memory-file /chemin/vers/memoire.json`. Le
fichier contient le texte des conversations en clair : évitez d'y saisir des
informations sensibles.

Le rôle et les consignes viennent du corpus et de `--system-prompt` (ou
`chat.system_prompt` dans la configuration), sans personnage imposé par la
classe. En conversation, le modèle reçoit ces consignes et l'historique, puis
s'arrête sur EOS, l'ID d'arrêt choisi ou le budget de tokens. Pour transmettre
un prompt multiligne complet, utilisez `--prompt-file /chemin/prompt.txt`.

`--temperature 0` active le décodage greedy. `--stop-id` ajoute un identifiant
d'arrêt ; EOS arrête également la génération. Le modèle fournit une architecture causale
decoder-only courante ; cela ne reproduit ni les poids, ni les données, ni les
capacités propriétaires de Claude Opus. La qualité dépend du corpus, du
tokenizer, de la taille du modèle et de l'entraînement réalisé.

## Mélange local causal et SwiGLU natif

Chaque nouveau bloc applique après l’attention une convolution depthwise sur les
tokens, avec une disposition `[T,C]`. Pour un canal `c`, son résultat est
`y[t,c] = somme_i w[c,i] * x[t - i*dilation,c]` ; les positions négatives valent
zéro. Le token courant est inclus et aucun token futur n’est lu. Une projection
réinjecte ce signal par une connexion résiduelle avant le bottleneck latent.
Le nombre de paramètres du filtre est `d_model * local_conv_kernel`, indépendant
de la longueur du prompt. La branche ajoute aussi une RMSNorm et une projection
de `d_model` vers `d_model`.

Ce choix vise l’apprentissage de motifs locaux entre tokens. Il s’inspire de
l’usage de convolutions depthwise dans [Primer](https://arxiv.org/abs/2109.08668),
avec ici une branche résiduelle distincte de l’attention. Ce n’est pas une
reproduction de cette architecture ; son effet sur la qualité se mesure avec la
validation et le corpus du modèle.

La couche native `SwiGLU` reçoit les deux projections `gate` et `up` et calcule
directement `SiLU(gate) * up`. Elle remplace les deux opérations précédentes sans
ajouter de paramètres, et fournit un gradient pour chacune des entrées. Le
contrat est celui des [variantes GLU](https://arxiv.org/abs/2002.05202). Le chemin
composite reste disponible avec `--no-fused-swiglu`.

Les nouvelles opérations ont des kernels CPU et Vulkan pour le forward et le
backward. La séquence est déduite du buffer à chaque appel ; `seq_len` reste une
fenêtre d’entraînement, sans plafond supplémentaire pour l’inférence. Les tips
Viz montrent `/local_conv`, `/local_residual` et `/swiglu` ; les métriques du modèle
publient aussi le noyau, la dilatation et le choix de SwiGLU.

Pour un nouveau run, ajouter par exemple à la commande d’entraînement :

```bash
--local-conv-kernel 3 --local-conv-dilation 1 --fused-swiglu \
--resume false --checkpoint-dir checkpoints/causal_local_v2
```

Les checkpoints historiques sans ces champs sont reconstruits avec
`local_conv_kernel=0` et `fused_swiglu=false`. Une reprise conserve ces choix et
refuse une CLI ou un override incompatible. Le script d’inférence reconstruit
la topologie depuis le checkpoint, même si la configuration de lancement décrit
un autre modèle. Il utilise par défaut le tokenizer du checkpoint RawFolder.

## Bottleneck latent causal

Le modèle `causal_lm` inclut par défaut un bottleneck latent contextuel dans
chaque bloc. Après l’attention causale, chaque position est comprimée dans un
code appris (`d_model → latent_dim → d_model`), puis cette mise à jour est
réinjectée avant le bloc SwiGLU. Le code reçoit donc le contexte causal déjà
assimilé par l’attention, sans accès aux tokens futurs, et son apprentissage
est guidé par la perte de prédiction du prochain token.

Sans `latent_dim` explicite, sa dimension vaut `min(128, d_model / 2)` (au
moins 1). `--latent-dim 0` désactive la branche; une valeur positive choisit
sa largeur. Ce chemin ajoute une représentation utile à la composition du
contexte, mais ne garantit pas à lui seul une meilleure compréhension ou un
raisonnement fiable : ces qualités dépendent aussi des données et de
l’évaluation.

Une configuration de départ utilisant le corpus français et un latent de 128
dimensions est disponible dans `configs/causal_lm_francais_mix.json`. Elle
réduit aussi `mlp_hidden` à 512 pour compenser une partie des paramètres du
chemin latent.

Quand la Viz est active, le modèle publie des repères sur les embeddings, les
sorties d’attention causale, les codes latents, leur réinjection résiduelle,
les sorties des blocs et les logits. Une vignette texte affiche le contexte
fourni, les propositions top-1 par position, leur accord avec le token suivant
visible et la marge entre les deux logits les plus élevés. La couleur montre
cette marge; elle ne représente pas une probabilité. La configuration française
limite la capture aux couches repérées et réserve 48 vues afin de garder les
étapes latentes visibles dans l’interface.

```bash
./run_mimir.sh --conf configs/causal_lm_francais_mix.json
```

## AMP et entraînement distribué

Les scripts `train_causal_lm.lua` et `train_t5.lua` acceptent `--amp` et
`--amp-dtype bfloat16` ou `float16`. Ce chemin exige un build avec
`-DENABLE_AMP=ON`. L'autocast actuellement disponible arrondit
les entrées et poids des couches Linear en précision réduite, avec accumulation
et gradients en `float32`. Ces couches sont exécutées par le runtime CPU; les
autres opérations restent en `float32`. Cela active le chemin mixte, sans
promettre d'accélération GPU ni de réduction mémoire sur les tenseurs du graphe.

DDP utilise les collectives MPI. Configurez et compilez Mímir avec MPI, puis
lancez plusieurs processus avec le même fichier de configuration :

```bash
cmake -S . -B build -DENABLE_MPI=ON
cmake --build build -j
mpirun -n 4 ./run_mimir.sh --conf configs/causal_lm_francais_mix.json \
  --override training.ddp=true
```

Le lanceur peut aussi préparer MPI et répartir les threads OpenMP par rang :

```bash
./run_mimir.sh --mpi-ranks 2 --lua scripts/training/train_causal_lm.lua -- \
  --conf configs/causal_lm_francais_mix.json --ddp
```

`--mpi-ranks` (alias `--mpi` et `--ddp-ranks`) lance les processus ; `--ddp`
active séparément les collectives du modèle. Le lanceur refuse un MPI imbriqué.

Le runtime diffuse les poids du processus 0 au premier forward et moyenne les
gradients avant chaque mise à jour. Les scripts ci-dessus échantillonnent des
fenêtres différentes par rang et réservent l'écriture du tokenizer et des
checkpoints au processus 0. Sans build MPI, `--ddp` échoue avec un message
indiquant l'option CMake requise.

## Limites actuelles

- La boucle native entraîne par fenêtres échantillonnées, sans batching.
- Pas encore de KV-cache : chaque token généré recalcule tout le contexte.
- `Mimir.Model.infer()` reste un chemin historique ; utiliser
  `causal_lm_generate.lua` pour générer avec `causal_lm`.

## Étapes suivantes

- [Tokenizer et encodeur](07-Tokenizer-Encoder.md)
- [Entraînement](04-Training.md)
- [API `Mimir.Model`](../03-API-Reference/10-Model.md)
