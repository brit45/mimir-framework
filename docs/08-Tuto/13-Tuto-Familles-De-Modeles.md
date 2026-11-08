# Choisir et vérifier une famille de modèles

Ce tutoriel apprend à choisir un modèle à partir de son entrée, de sa sortie et
de son objectif. Trois ateliers courts portent sur un MLP, un encodeur T5 et un
VAE convolutionnel ; un parcours avancé explique les prérequis de Lumen.

Une architecture native est écrite en C/C++ dans une classe héritant de `Model`.
Les scripts Lua appellent son registre, paramètrent son exécution et inspectent
ses sorties. Le format MPK est une alternative expérimentale, dont la couverture
reste incomplète pour la production. Voir le [contrat d'architecture](../07-Devs/08-Architecture-Contract.md).

## Prérequis et choix initial

Le binaire doit être compilé. Les trois ateliers ne demandent ni dataset externe
ni poids téléchargés. Le premier écrit un checkpoint sous `/tmp/mimir_tutoriels` ;
créez ce dossier comme dans le [tutoriel des données](11-Tuto-Donnees-Tokenizer.md).
Les commandes sont lancées depuis la racine du dépôt.

| Besoin | Point de départ | Question à vérifier |
| --- | --- | --- |
| Vecteur vers vecteur, régression | `basic_mlp` | Les dimensions de cible correspondent-elles à la sortie ? |
| Prédiction du prochain token | `causal_lm` | Corpus, masque de cible et tokenizer sont-ils cohérents ? |
| Représentation bidirectionnelle du texte | `t5` | Utilisez-vous l'encodeur ou sa tête de débruitage ? |
| Compression et reconstruction de tokens | `vae_text` | Reconstruction, KL et contenu latent sont-ils évalués séparément ? |
| Compression et reconstruction d'images | `vae_conv` | Les conventions image, latent et décodeur correspondent-elles ? |
| Vision discriminative | `resnet`, `vit`, `mobilenet` | Le script d'entraînement fournit-il labels et loss compatibles ? |
| Génération d'images conditionnée | `lumen_diffusion` | VAE, conditionnement texte, tokenizer et données sont-ils compatibles ? |

`transformer`, les CNN, les VAE et les modèles de diffusion ne partagent pas un
format universel de données ou une loss interchangeable. Un modèle inscrit au
registre n'est pas nécessairement un pipeline métier complet.

## Atelier 1 — Débutant : MLP, sortie et checkpoint

Créez `/tmp/mimir_tutoriels/mlp.lua` :

```lua
local cfg = assert(Mimir.Architectures.default_config("basic_mlp"))
cfg.input_dim, cfg.hidden_dim, cfg.output_dim = 4, 8, 2
cfg.hidden_layers, cfg.dropout = 1, 0.0
assert(Mimir.Model.create("basic_mlp", cfg))
assert(Mimir.Model.set_hardware(false))
assert(Mimir.Model.allocate_params())
assert(Mimir.Model.init_weights("xavier", 1337))
local input = {0.1, 0.2, 0.3, 0.4}
local reference = assert(Mimir.Model.forward(input, false))
assert(#reference == 2)
local path = "/tmp/mimir_tutoriels/mlp.safetensors"
local options = {save_tokenizer=false, save_encoder=false, save_optimizer=false}
assert(Mimir.Serialization.save(path, "safetensors", options))

assert(Mimir.Model.create("basic_mlp", cfg))
assert(Mimir.Model.set_hardware(false))
assert(Mimir.Model.allocate_params())
assert(Mimir.Serialization.load(path, "safetensors", {
  load_tokenizer=false, load_encoder=false, load_optimizer=false,
  strict_mode=true,
}))
local restored = assert(Mimir.Model.forward(input, false))
for index = 1, #reference do
  assert(math.abs(reference[index] - restored[index]) < 1e-5)
end
print("MLP_FAMILLES_OK", Mimir.Model.total_params())
```

```bash
env OMP_NUM_THREADS=1 MIMIR_DISABLE_CUDA=1 MIMIR_DISABLE_ROCM=1 \
  MIMIR_DISABLE_VULKAN=1 MIMIR_DISABLE_OPENCL=1 \
  ./run_mimir.sh --lua /tmp/mimir_tutoriels/mlp.lua -- --no-htop --no-viz
```

Le marqueur `MLP_FAMILLES_OK` confirme l'aller-retour sur cette configuration.
Les poids ne sont pas réinitialisés après leur chargement. Ce checkpoint ne
contient pas d'état d'optimiseur : il convient à cette vérification d'inférence,
pas à une preuve de reprise exacte d'entraînement.

La topologie est construite dans `BasicMLPModel::buildFromConfig`. Modifier
`hidden_dim` sélectionne une variante supportée par cette classe ; le script
n'ajoute pas de couches et ne change pas leurs connexions.

## Atelier 2 — Intermédiaire : distinguer T5 et causal

Exécutez le smoke natif fourni :

```bash
env OMP_NUM_THREADS=1 MIMIR_DISABLE_CUDA=1 MIMIR_DISABLE_ROCM=1 \
  MIMIR_DISABLE_VULKAN=1 MIMIR_DISABLE_OPENCL=1 \
  ./run_mimir.sh --lua scripts/tests/test_t5.lua -- --no-htop --no-viz
```

Le marqueur attendu est `T5_SMOKE_OK`. L'exemple utilise des IDs synthétiques
sous l'entrée nommée `text_ids`, vérifie logits, backward et mise à jour des
poids, puis la forme de sortie en mode encodeur.

Le T5 du framework est un encodeur bidirectionnel avec une tête de débruitage
par tokens. Le décodeur T5 text-to-text original et l'import de poids Hugging Face
ne sont pas fournis par cette architecture. L'encodeur ne doit donc pas être
présenté comme un assistant conversationnel déjà entraîné.

Pour comparer : le [tutoriel causal](12-Tuto-Entrainement-Validation-Reprise-Inference.md)
apprend à prédire le prochain token puis génère de manière autorégressive.
`vae_text` reconstruit du texte via un latent et une pénalité KL ; son guide
documente aussi des objectifs de contenu. Les noms de têtes auxiliaires ne
constituent pas une mesure de compréhension sémantique.

**Exercice :** dessinez les formes d'entrée et de sortie de chaque famille.
Expliquez pourquoi des logits `[T,V]` ne sont ni un texte décodé ni les
représentations `[T,D]` d'un encodeur.

## Atelier 3 — Intermédiaire : inspecter VAEConv

```bash
env OMP_NUM_THREADS=1 MIMIR_DISABLE_CUDA=1 MIMIR_DISABLE_ROCM=1 \
  MIMIR_DISABLE_VULKAN=1 MIMIR_DISABLE_OPENCL=1 \
  ./run_mimir.sh --lua scripts/examples/inspect_vae_conv.lua -- --no-htop --no-viz
```

Ce script construit une petite configuration native, alloue et initialise ses
poids puis affiche les couches et le prior appris. Il ne lit pas de dataset et
ne démontre pas la qualité d'une reconstruction d'image.

L'entrée image est un vecteur HWC, adapté au graphe convolutionnel interne.
La sortie d'entraînement rassemble reconstruction, `mu` et `logvar` ; l'inférence
sélectionne la reconstruction. Le latent spatial possède ses propres dimensions
et sa convention CHW. Ne passez pas une image aplatie à la place d'un latent
simplement parce que leurs nombres de valeurs sont voisins.

Le mode `--conv-mode 3d` ou `4d` sélectionne les couches correspondantes dans
la classe native. Pour des images 2D, les axes supplémentaires sont de taille 1 :
cela n'invente ni une vidéo ni un volume. Ce choix appartient à la topologie et
doit correspondre au checkpoint à la reprise.

Pour continuer avec un checkpoint réel, consultez l'aide de l'outil :

```bash
./run_mimir.sh --lua scripts/inferences/infer_vae_conv.lua -- --help
```

L'encodage attend une image PPM, le décodage un latent RAW float32 suivant la
convention du modèle. Les procédures complètes figurent dans le
[guide VAEConv](../02-User-Guide/14-VAEConv.md) et l'[atelier sans dataset](07-Tuto-VAEConv-Sans-Dataset.md).

## Parcours avancé — Assembler les prérequis de Lumen

Avant un entraînement Lumen, établissez cette fiche de compatibilité :

| Élément | Vérification à conserver dans le run |
| --- | --- |
| VAEConv entraîné | Résolution, canaux, taille latente et checkpoint complet |
| Dataset image/texte | Descriptions liées aux bonnes images, annotations valides |
| Conditionnement texte | Mode `vae_text`, `t5` ou `transformer` explicite |
| Encodeur texte préentraîné | Dimensions et vocabulaire compatibles avec le mode choisi |
| Tokenizer | Identifiants correspondant aux embeddings importés |
| Reprise | Topologie et sous-modèles gelés cohérents avec le checkpoint |

```bash
./run_mimir.sh --lua scripts/training/train_lumen_diffusion.lua -- --help
```

Le script sélectionne notamment `--text-conditioning`, `--vae-checkpoint`,
`--text-vae-checkpoint` ou `--text-t5-checkpoint`. Les conditionnements VAEText
et T5 sont exclusifs. Leurs encodeurs importés restent gelés ; la projection de
conditionnement du modèle est entraînable. Le tokenizer de l'encodeur importé
ne doit pas être agrandi librement.

Cette étape exige de vrais checkpoints compatibles ; l'aide seule n'exécute
pas un entraînement de diffusion. Suivez le [guide diffusion](../02-User-Guide/13-Diffusion.md)
et le [guide T5 pour Lumen](../02-User-Guide/16-T5.md).

## Exercices avancés et dépannage

Choisissez une même tâche et justifiez la famille, les données, la cible et la
mesure de validation. Décrivez ensuite un test qui pourrait invalider votre choix.
Pour une nouvelle topologie, modifiez ou créez la classe C++ dérivée de `Model`
avec son registre et ses tests ; ne l'assemblez pas dans Lua ou dans JSON.

Si une entrée nommée manque, vérifiez le contrat du modèle plutôt que renommer
arbitrairement les tenseurs. Si les poids ne se chargent pas, comparez la classe,
la topologie, les shapes, le dtype et le tokenizer. Un fichier SafeTensors externe
ne fournit pas à lui seul la sémantique d'un graphe DL.

## Étapes suivantes

- [Entraînement et reprise](12-Tuto-Entrainement-Validation-Reprise-Inference.md).
- [Ajouter une architecture native](02-Tuto-Ajouter-Modele.md).
- [Index des tutoriels](00-INDEX.md).
