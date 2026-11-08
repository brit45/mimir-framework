# Préparer les données et comprendre le tokenizer

Ce tutoriel prépare un petit corpus synthétique, apprend un BPE et vérifie sa
sauvegarde. Il progresse de l'inspection de fichiers à la gestion des identifiants
de tokens dans un entraînement reproductible. Aucun corpus externe n'est requis.

Les exemples Lua manipulent des données et des assets. Ils ne définissent pas
d'architecture : les graphes natifs sont écrits en C/C++ dans une classe dérivée
de `Model`. Le format MPK reste une alternative expérimentale, incomplète pour
un usage de production.
La règle complète est décrite dans le [contrat d'architecture](../07-Devs/08-Architecture-Contract.md).

## Prérequis et résultat

Le binaire Mímir doit fonctionner. Exécutez les commandes depuis la racine du
dépôt, avec Python 3 pour créer la fixture. Les exemples utilisent le dossier
temporaire `/tmp/mimir_tutoriels`, comme le [premier tutoriel](10-Tuto-Installation-Configuration.md).

Vous obtiendrez un corpus de douze documents, un petit dataset texte et un
tokenizer JSON dont les identifiants survivent au rechargement.

## 1. Créer des données contrôlables

```bash
python3 - <<'PY'
from pathlib import Path
root = Path('/tmp/mimir_tutoriels')
data = root / 'dataset'
data.mkdir(parents=True, exist_ok=True)
documents = [
    'Le chat regarde le jardin. La lumière entre par la fenêtre.',
    'Une plante utilise la lumière. Ses racines absorbent de l eau.',
    'Deux plus trois font cinq. On vérifie le résultat par une addition.',
    'Une question claire donne le contexte et ce que nous cherchons.',
    'Le vent déplace les nuages. Le ciel devient plus clair ce matin.',
    'La bibliothèque contient des livres. Chaque livre porte un titre.',
    'Une mesure doit annoncer son unité. Une longueur se mesure en mètres.',
    'Une hypothèse peut être testée. Les observations peuvent la contredire.',
    'Un rectangle possède quatre côtés. Ses angles sont droits.',
    'Pour comparer deux idées, il faut préciser le critère utilisé.',
    'Le sommeil aide le repos. Une routine organise les heures de la journée.',
    'Une conversation demande une écoute attentive et des réponses liées.',
]
questions = [
    'Que regarde le chat ?',
    'Comment vit une plante ?',
    'Combien font deux plus trois ?',
    'Que contient une question claire ?',
    'Que fait le vent ?',
    'Que trouve-t-on à la bibliothèque ?',
    'Comment exprimer une mesure ?',
    'Comment examiner une hypothèse ?',
    'Comment décrire un rectangle ?',
    'Comment comparer deux idées ?',
    'À quoi sert une routine de sommeil ?',
    'Que demande une conversation ?',
]
dialogues = [
    'Système: Tu expliques avec des phrases simples.\n'
    f'Utilisateur: {question}\nAssistante: {answer}'
    for question, answer in zip(questions, documents)
]
(root / 'corpus.txt').write_text('\n\n'.join(dialogues), encoding='utf-8')
for index, text in enumerate(documents):
    (data / f'{index:03}.txt').write_text(text, encoding='utf-8')
print(root / 'corpus.txt', len(documents), 'documents')
PY
```

Le fichier unique contient douze dialogues et servira à `causal_lm`. Le dossier
`dataset` garde les douze textes bruts pour apprendre l'API générale du loader.
N'ajoutez pas les JSON de configuration ou les logs
dans ce dossier : ils pourraient être reconnus comme des données textuelles.

Douze documents suffisent pour vérifier un chemin de traitement. Ils ne suffisent
pas à apprendre une langue ou à évaluer des connaissances générales.

## 2. Inspecter le loader sans décoder d'images

Enregistrez ce script dans `/tmp/mimir_tutoriels/dataset.lua` :

```lua
local ok, count = Mimir.Dataset.load("/tmp/mimir_tutoriels/dataset", 16, 16, 1)
assert(ok, count)
assert(count == 12)
for index = 1, count do
  local item = assert(Mimir.Dataset.get(index, false))
  assert(type(item.text) == "string" and #item.text > 0)
  print(index, item.text_file, item.text)
  assert(Mimir.Dataset.release(index))
end
print("DATASET_TUTORIEL_OK")
```

```bash
./run_mimir.sh --lua /tmp/mimir_tutoriels/dataset.lua -- --no-htop --no-viz
```

Le résultat attendu contient douze items et `DATASET_TUTORIEL_OK`.
`get(i, false)` conserve le texte et les métadonnées sans demander le décodage
des pixels. `release(i)` libère les buffers lazy de l'item.

## 3. Relier les modalités correctement

Le loader groupe les fichiers par **chemin relatif sans extension** :

```text
dataset/
  scene/001.txt
  scene/001.png
  scene/001.wav
  scene/001.mp4
```

Ces fichiers décrivent un item multimodal. `autre/001.png` désigne un autre item.
L'argument `min_modalities` de `Dataset.load` filtre le nombre de modalités
nécessaires ; il vaut 1 dans notre exemple texte.

| Modalité | Ce qu'il faut vérifier avant l'entraînement |
| --- | --- |
| Texte | Encodage, documents vides, frontières, qualité et doublons |
| Image | Canaux, taille, convention de disposition et correspondance caption/image |
| Audio | Décodeur disponible, fréquence, canaux et convention de l'entrée du modèle |
| Vidéo | Décodeur disponible, cadence, échantillonnage des frames et budget mémoire |

La reconnaissance d'une extension ne prouve pas qu'un modèle accepte la modalité.
FFmpeg est optionnel pour les chemins média qui l'utilisent. Un chemin de fichier
ou des octets compressés ne sont pas un tenseur de pixels ou de samples audio.

L'[atelier d'annotation](../../annotator/README.md) produit des sidecars JSON
contenant captions, entités, boîtes, masques et relations. Pour un sidecar
structuré reconnu, `captions.primary` est la description principale ; une
annotation invalide n'est pas remplacée silencieusement par un ancien `.txt`.
Les formats et priorités sont détaillés dans le [guide des données](../02-User-Guide/03-Data.md).

## 4. Étudier le BPE et conserver les identifiants

Enregistrez `/tmp/mimir_tutoriels/tokenizer.lua` :

```lua
local texts = {
  "Une plante utilise la lumière.",
  "La lumière entre dans le jardin.",
  "Une question précise apporte du contexte.",
}
assert(Mimir.Tokenizer.create(512))
assert(Mimir.Tokenizer.learn_bpe(texts, 40))
local before = assert(Mimir.Tokenizer.tokenize_bpe(texts[1]))
assert(#before > 0)
assert(Mimir.Tokenizer.save("/tmp/mimir_tutoriels/tokenizer.json"))
assert(Mimir.Tokenizer.load("/tmp/mimir_tutoriels/tokenizer.json"))
local after = assert(Mimir.Tokenizer.tokenize_bpe(texts[1]))
assert(#before == #after)
for index = 1, #before do assert(before[index] == after[index]) end
print("TOKENIZER_TUTORIEL_OK", Mimir.Tokenizer.vocab_size(),
      Mimir.Tokenizer.bpe_merge_count(), #after)
```

```bash
./run_mimir.sh --lua /tmp/mimir_tutoriels/tokenizer.lua -- --no-htop --no-viz
```

Dans ce binding, `learn_bpe` reçoit une **table de textes**, pas directement un
chemin de corpus. La capacité 512 est une limite de vocabulaire ; 40 est un
budget de merges, pas le nombre final de mots connus. Un mot peut être représenté
par plusieurs sous-mots. La détokenisation peut normaliser la forme du texte :
le test vérifie les identifiants, pas une identité des octets du texte d'origine.

## 5. Relier tokenizer, corpus et checkpoint causal

`CausalLMModel` possède sa préparation native de corpus et son tokenizer.
Le [tutoriel d'entraînement](12-Tuto-Entrainement-Validation-Reprise-Inference.md)
l'utilise ; ne lui imposez pas le tokenizer miniature de l'étape précédente.

La préparation des dialogues étudie aussi leurs gabarits natifs : balises de rôle,
consignes ajoutées et repères de contexte. Un BPE appris uniquement sur de petits
paragraphes bruts peut ne pas couvrir les caractères ou les sous-mots de ces
gabarits à la génération. C'est pourquoi le corpus causal de cet exercice utilise
des dialogues reconnus, tandis que l'exercice du loader reste sur du texte brut.

La préparation compose le vocabulaire avant allocation des embeddings. Un fichier
compatible déjà étudié est réutilisé ; une reprise emploie le tokenizer du
checkpoint. Modifier ses identifiants après apprentissage désynchroniserait les
lignes d'embedding et la tête de prédiction.

Les séquences causales sont cadrées par `<BOS>` et `<EOS>`. Les balises natives
de rôle et contexte comprennent `<SYSTEM>`, `<USER>`, `<ASSISTANT>`, `<CONTEXT>`,
`</CONTEXT>`, `<REFLECT>` et `<FINAL>`. Leur présence structure le texte ; elle
ne crée pas à elle seule une capacité de raisonnement. Ces douze conversations
servent à vérifier le traitement, pas à mesurer la qualité d'un assistant.

## Exercices par niveau

| Niveau | Exercice | Vérification |
| --- | --- | --- |
| Débutant | Ajouter un treizième document dans une copie de la fixture | Le compteur du loader change ; adaptez l'assertion attendue |
| Intermédiaire | Comparer 20 et 40 merges dans deux tokenizers neufs | Notez vocabulaire, merges et longueur de la même phrase |
| Avancé | Préparer une partition train/validation sans doublons de documents | Un contenu presque identique ne se retrouve pas dans les deux partitions |

## Dépannage et suite

Un vocabulaire trop petit peut laisser des `<UNK>` ; augmenter uniquement le
nombre de merges ne garantit pas la couverture. Si un checkpoint refuse un
tokenizer, comparez capacité, IDs spéciaux et vocabulaire sauvegardé plutôt
que réapprendre le BPE après chargement des poids.

- [Entraîner et reprendre](12-Tuto-Entrainement-Validation-Reprise-Inference.md).
- [API Tokenizer](../03-API-Reference/12-Tokenizer.md) et [API Dataset](../03-API-Reference/13-Dataset.md).
- [Index des tutoriels](00-INDEX.md).
