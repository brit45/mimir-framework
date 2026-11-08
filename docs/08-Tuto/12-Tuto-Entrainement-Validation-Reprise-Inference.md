# Entraîner, valider, reprendre et générer

Ce tutoriel exécute un entraînement causal minuscule jusqu'au checkpoint,
le reprend puis génère quelques tokens. Il permet d'apprendre le cycle complet
avant de consacrer du temps à un corpus réel. Le résultat attendu est un pipeline
cohérent, pas un modèle francophone compétent après quelques pas.

Le graphe est celui de `CausalLMModel`, classe C++ dérivée de `Model`.
Le script fourni configure et appelle l'entraînement natif ; il ne construit
pas de branches DL en Lua. MPK reste une alternative expérimentale non pleinement
exploitable en production. Voir le [contrat d'architecture](../07-Devs/08-Architecture-Contract.md).

## Prérequis et progression

Préparez le corpus synthétique du [tutoriel des données](11-Tuto-Donnees-Tokenizer.md).
Le fichier `/tmp/mimir_tutoriels/corpus.txt` doit exister et contenir les douze
dialogues de cet exercice. Le binaire doit être compilé ; aucun GPU, dataset
externe ou checkpoint préentraîné n'est nécessaire.

| Niveau | Objectif | Résultat observable |
| --- | --- | --- |
| Débutant | Terminer quatre pas sur CPU | Loss finie et checkpoint `final` |
| Intermédiaire | Séparer validation et entraînement, puis reprendre | Step global conservé et porté à 6 |
| Avancé | Évaluer sur des documents nouveaux et vérifier la reprise | Config, tokenizer, optimiseur et partition cohérents |

Les commandes utilisent Bash et le dossier temporaire des tutoriels précédents.
Pour recommencer depuis des poids neufs, choisissez un **nouveau** dossier de
checkpoint. Pour poursuivre des poids existants, utilisez explicitement la reprise.

## 1. Lire les options du script utilisé

```bash
./run_mimir.sh --lua scripts/training/train_causal_lm.lua -- --help
./run_mimir.sh --lua scripts/inferences/causal_lm_generate.lua -- --help
```

`--seq-len` règle ici la fenêtre d'entraînement. La génération ajuste la séquence
au prompt complet et au budget de sortie ; cela ne rend pas le coût d'attention
ou la mémoire illimités. Les documents sont cadrés par `<BOS>` et `<EOS>` dans
la préparation native, et les balises de rôle peuvent organiser une conversation.

## 2. Exécuter un premier entraînement court

Le vocabulaire, la largeur, la profondeur et le latent sont volontairement petits.
La préparation du corpus et le BPE ont lieu avant l'allocation des poids.

```bash
env OMP_NUM_THREADS=1 MIMIR_DISABLE_CUDA=1 MIMIR_DISABLE_ROCM=1 \
  MIMIR_DISABLE_VULKAN=1 MIMIR_DISABLE_OPENCL=1 \
  ./run_mimir.sh --lua scripts/training/train_causal_lm.lua -- \
  --corpus /tmp/mimir_tutoriels/corpus.txt \
  --tokenizer /tmp/mimir_tutoriels/causal/tokenizer.json \
  --checkpoint-dir /tmp/mimir_tutoriels/causal \
  --vocab-size 512 --bpe-merges 40 --seq-len 32 \
  --d-model 16 --layers 1 --heads 2 --kv-heads 1 --mlp-hidden 32 \
  --latent-dim 8 --dtype float32 \
  --steps 4 --lr 0.0003 --optimizer adamw --seed 1337 \
  --log-every 1 --save-every 2 \
  --validate-every-steps 2 --validate-items 2 --validate-holdout-items 2 \
  --no-amp --no-ddp --no-htop --no-viz
```

Contrôlez les lignes de préparation : vocabulaire utilisé, merges BPE et nombre
d'items de validation. Le vocabulaire peut être inférieur à la capacité 512.
Un tokenizer déjà compatible est réutilisé ; une nouvelle étude n'est pas une
étape à provoquer systématiquement à chaque lancement.

Ce lancement fonctionne sans interface et affiche les métriques dans le journal.
Il ne promet pas d'export CSV. La variante avec moniteur et CSV est décrite dans
le [tutoriel d'observation](14-Tuto-Viz-Metriques-Diagnostic.md).

À la fin, le dossier contient notamment :

```text
/tmp/mimir_tutoriels/causal/
  tokenizer.json
  step-0000002/
  step-0000004/
  final/
    model/architecture.json
    model/training.json
    tokenizer/tokenizer.json
```

Cet aperçu ne liste pas tous les fichiers du format RawFolder : les poids et
l'état d'optimiseur font aussi partie du checkpoint. Conservez le dossier entier.

## 3. Lire la loss et la validation

| Mesure | Ce qu'elle vérifie | Ce qu'elle ne prouve pas |
| --- | --- | --- |
| Cross-entropy / loss | La probabilité attribuée au token cible | Une réponse correcte au niveau du sens |
| Perplexité | Une transformation exponentielle de la loss, plafonnée dans la métrique exposée | Une comparaison valide entre tokenizers différents |
| Accord top-1 | La proportion de tokens cibles gagnants | La qualité d'une conversation autonome |
| Gradient et clipping | La présence et l'amplitude des mises à jour | L'absence de surapprentissage |
| Loss de validation | La prédiction sur la partition de validation sélectionnée | La généralisation à tous les domaines |

La validation utilise des forwards d'évaluation, sans pas d'optimiseur sur ces
items. Les deux items holdout sont séparés de la sélection d'entraînement.
Gardez cette partition stable pendant les comparaisons. Vérifiez l'annonce de
la source : une évaluation sur des items d'entraînement ne constitue pas un holdout.

Dans Viz, le « top-1 proposé après chaque position » observe les prédictions
conditionnées par le contexte fourni. Ce n'est pas une longue réponse générée
token après token à partir de ses propres sorties.

## 4. Reprendre jusqu'au step global 6

```bash
env OMP_NUM_THREADS=1 MIMIR_DISABLE_CUDA=1 MIMIR_DISABLE_ROCM=1 \
  MIMIR_DISABLE_VULKAN=1 MIMIR_DISABLE_OPENCL=1 \
  ./run_mimir.sh --lua scripts/training/train_causal_lm.lua -- \
  --corpus /tmp/mimir_tutoriels/corpus.txt \
  --checkpoint-dir /tmp/mimir_tutoriels/causal \
  --resume --steps 6 --bpe-merges 40 --log-every 1 --save-every 2 \
  --no-amp --no-ddp --no-htop --no-viz
```

Pour ce script, `--steps 6` est la **cible globale** : une reprise au step 4
effectue deux pas supplémentaires. Le checkpoint détermine la topologie, les IDs
du tokenizer, l'état de l'optimiseur et les paramètres de partition sauvegardés.
Ne réinitialisez pas les poids après le chargement. Un override de dimensions
incompatible doit être refusé, pas interprété comme une reprise valable.

Vous pouvez vérifier le step sauvegardé :

```bash
python3 - <<'PY'
import json
from pathlib import Path
path = Path('/tmp/mimir_tutoriels/causal/final/model/architecture.json')
config = json.loads(path.read_text())['model_config']
assert config['training_state']['global_step'] == 6
print('REPRISE_TUTORIEL_OK', config['training_state']['global_step'])
PY
```

## 5. Générer à partir du checkpoint

```bash
env OMP_NUM_THREADS=1 MIMIR_DISABLE_CUDA=1 MIMIR_DISABLE_ROCM=1 \
  MIMIR_DISABLE_VULKAN=1 MIMIR_DISABLE_OPENCL=1 \
  ./run_mimir.sh --lua scripts/inferences/causal_lm_generate.lua -- \
  --checkpoint /tmp/mimir_tutoriels/causal/final \
  --prompt "Que regarde le chat ?" --max-new-tokens 8 \
  --system-prompt "Tu expliques avec des phrases simples." \
  --temperature 0.8 --top-k 8 --seed 1337 \
  --no-memory --no-htop --no-viz
```

Le script reconstruit le modèle natif depuis les métadonnées du checkpoint et
utilise son tokenizer. Huit tokens peuvent produire une sortie incohérente ou
très courte : c'est attendu pour ce smoke test. Pour un chat interactif, consulter
les options `--interactive` et `--memory-file` du même script ; la mémoire de
conversation est un contexte sérialisé, pas une mise à jour des poids.

Le prompt et la consigne système reprennent ici le vocabulaire des dialogues
préparés. Le BPE doit aussi couvrir les gabarits et repères ajoutés par le modèle,
dont leurs valeurs numériques, pas seulement les mots visibles du prompt.
Si la génération signale `<UNK>`,
vérifiez cette couverture. Ne réapprenez pas le tokenizer d'un checkpoint chargé
pour contourner l'erreur ; préparez un corpus couvrant le contexte dans un nouveau run.

## Exercices et dépannage

| Niveau | Travail | Critère de réussite |
| --- | --- | --- |
| Débutant | Repérer le step, la loss et le chemin du checkpoint | Vous pouvez décrire chaque fichier utile sans les déplacer séparément |
| Intermédiaire | Comparer deux LR dans deux dossiers neufs avec la même seed | Les courbes et la validation appartiennent au bon run |
| Avancé | Ajouter un corpus externe de test, absent du BPE d'évaluation et de l'entraînement | Vous expliquez la provenance des données et les limites de la mesure |

Si la validation est vide, vérifiez le nombre de documents et les options holdout.
Si les IDs ne correspondent plus, rechargez le tokenizer du checkpoint plutôt que
le régénérer. Si la reprise annonce zéro pas supplémentaire, la cible `--steps`
est déjà atteinte. Une loss non finie demande un diagnostic des entrées, du LR,
des gradients et du backend avant d'allonger le run.

## Étapes suivantes

- [Choisir une famille de modèles](13-Tuto-Familles-De-Modeles.md).
- [Guide causal](../02-User-Guide/12-Transformer-GPT.md), [checkpoints](../02-User-Guide/08-Checkpoints.md) et [reprise générale](../03-API-Reference/16-Serialization.md).
- [Index des tutoriels](00-INDEX.md).
