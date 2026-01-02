# Tokenization

**Version:** 2.3.0  
**API:** Conforme à mimir-api.lua  

Guide complet de la tokenization et gestion du vocabulaire dans Mímir Framework.

> 💡 **Syntaxe Recommandée:** Utiliser `Mimir.Tokenizer.*` pour bénéficier de l'autocompletion IDE.

---

## 📋 Table des Matières

- [Introduction](#introduction)
- [Création de Tokenizer](#création-de-tokenizer)
- [Vocabulaire](#vocabulaire)
- [Encodage et Décodage](#encodage-et-décodage)
- [Byte Pair Encoding (BPE)](#byte-pair-encoding-bpe)
- [Tokens Spéciaux](#tokens-spéciaux)
- [Best Practices](#best-practices)
- [Exemples Complets](#exemples-complets)

---

## 🎯 Introduction

### Qu'est-ce que la Tokenization ?

**Tokenization** = conversion texte ↔ nombres pour les réseaux de neurones.

```
Texte brut : "Hello, world!"
     ↓ Tokenize
Tokens : ["Hello", ",", "world", "!"]
     ↓ Encode
IDs : [5234, 6, 2874, 8]
     ↓ Embed
Vecteurs : [[0.1, 0.3, ...], [0.5, 0.2, ...], ...]
```

### Pourquoi Tokenizer ?

Les réseaux de neurones ne peuvent pas traiter directement du texte. La tokenization :

1. **Convertit** texte → nombres
2. **Gère** le vocabulaire (mots connus)
3. **Encode** séquences pour le modèle
4. **Décode** prédictions → texte lisible

---

## 🏗️ Création de Tokenizer

### Tokenizer Vide

```lua
-- Créer tokenizer avec taille vocabulaire
Mimir.Tokenizer.create(5000)

-- Vérifier création
local size = Mimir.Tokenizer.vocab_size()
print("Vocabulaire:", size)  -- 5000
```

### Charger Vocabulaire Existant

```lua
-- Créer et charger depuis fichier
Mimir.Tokenizer.create(50000)
Mimir.Tokenizer.load("Mimir.Tokenizer.json")

-- Vérifier taille
local vocab_size = Mimir.Tokenizer.vocab_size()
print("Vocabulaire:", vocab_size, "tokens")
-- Vocabulaire: 50000 tokens
```

### Format Vocabulaire JSON

```json
{
  "vocab": {
    "<PAD>": 0,
    "<UNK>": 1,
    "<BOS>": 2,
    "<EOS>": 3,
    "the": 4,
    "a": 5,
    "hello": 6,
    "world": 7,
    ...
  },
  "config": {
    "vocab_size": 50000,
    "model_type": "bpe",
    "unk_token": "<UNK>",
    "pad_token": "<PAD>",
    "bos_token": "<BOS>",
    "eos_token": "<EOS>"
  }
}
```

---

## 📚 Vocabulaire

### Informations

```lua
-- Taille vocabulaire
local vocab_size = Mimir.Tokenizer.vocab_size()
print("Taille vocabulaire:", vocab_size)
```

### Token ↔ ID

```lua
-- Tokenisation d'un mot
local token_ids = Mimir.Tokenizer.tokenize("hello")
print("IDs de 'hello':", table.concat(token_ids, ", "))

-- Détokenisation
local text = Mimir.Tokenizer.detokenize({6, 7, 8})
print("Texte:", text)
```

### Vérifier Vocabulaire

```lua
-- Vérifier taille
local size = Mimir.Tokenizer.vocab_size()
print("Taille:", size)
```

### Ajouter Tokens

```lua
-- Ajouter un token au vocabulaire
Mimir.Tokenizer.add_token("newword")

-- Ajouter plusieurs
Mimir.Tokenizer.addTokens(tokenizer, {"token1", "token2", "token3"})

-- Vérifier
print("Nouveau vocab size:", Mimir.Tokenizer.getVocabSize(tokenizer))
```

### Sauvegarder Vocabulaire

```lua
-- Sauvegarder tokenizer
Mimir.Tokenizer.save("my_tokenizer.json")
```

---

## 🔄 Encodage et Décodage

### Tokenization (Texte → IDs)

```lua
-- Tokenizer une phrase
local text = "The quick brown fox jumps"
local token_ids = Mimir.Tokenizer.tokenize(text)

-- Résultat : table d'entiers
print("Token IDs:", table.concat(token_ids, ", "))
-- Token IDs: 4, 2341, 3456, 7823, 9012
```

### Détokenization (IDs → Texte)

```lua
-- Détokenizer IDs en texte
local token_ids = {4, 2341, 3456, 7823, 9012}
local text = Mimir.Tokenizer.detokenize(token_ids)

print("Texte décodé:", text)
-- Texte décodé: The quick brown fox jumps
```

### Exemple Complet

```lua
-- Cycle complet tokenization
local text = "Hello, world!"

-- Tokenize
local token_ids = Mimir.Tokenizer.tokenize(text)
print("IDs:", table.concat(token_ids, ", "))

-- Detokenize
local reconstructed = Mimir.Tokenizer.detokenize(token_ids)
print("Texte:", reconstructed)  -- "Hello, world!"
```

---

## 🔤 Byte Pair Encoding (BPE)

### Qu'est-ce que BPE ?

**BPE** = algorithme de sous-mots pour gérer vocabulaire ouvert.

```
Mot inconnu : "running"
Sans BPE : [<UNK>]
Avec BPE : ["run", "##ning"] → [2341, 4523]
```

**Avantages** :
- Gère mots OOV (out-of-vocabulary)
- Vocabulaire compact
- Partage sous-mots communs

### Entraîner BPE

```lua
-- Créer tokenizer
Mimir.Tokenizer.create(50000)

-- Apprendre BPE sur corpus
Mimir.Tokenizer.learn_bpe(corpus_text, 50000)

-- Sauvegarder
Mimir.Tokenizer.save("tokenizer_bpe.json")
```

### Utiliser BPE

```lua
-- Charger tokenizer BPE
Mimir.Tokenizer.create(50000)
Mimir.Tokenizer.load("tokenizer_bpe.json")

-- Tokenizer avec BPE (automatique)
local text = "running quickly"
local ids = Mimir.Tokenizer.tokenize_bpe(text)

print("BPE IDs:", table.concat(ids, ", "))
-- BPE IDs: 2341, 4523, 7812, 8934
-- Correspond à: ["run", "##ning", "quick", "##ly"]
```

### Décoder BPE

```lua
-- Détokenizer fusionne automatiquement les sous-mots
local ids = {2341, 4523, 7812, 8934}
local text = Mimir.Tokenizer.detokenize(ids)

print("Texte:", text)
-- Texte: running quickly
```

### BPE vs Word-Level

| Critère | Word-Level | BPE |
|---------|-----------|-----|
| **Vocab Size** | 100K+ | 30K-50K |
| **OOV Handling** | <UNK> | Sous-mots |
| **Compositionality** | Non | Oui |
| **Speed** | Rapide | Moyen |
| **Usage** | Petits corpus | Production NLP |

---

## 🏷️ Tokens Spéciaux

### Types Courants

```lua
-- Configuration tokens spéciaux
Mimir.Tokenizer.setSpecialTokens(tokenizer, {
    pad_token = "<PAD>",   -- Padding
    unk_token = "<UNK>",   -- Unknown word
    bos_token = "<BOS>",   -- Begin of sequence
    eos_token = "<EOS>",   -- End of sequence
    cls_token = "<CLS>",   -- Classification (BERT)
    sep_token = "<SEP>",   -- Separator (BERT)
    mask_token = "<MASK>"  -- Masked LM (BERT)
})
```

### Obtenir IDs Spéciaux

```lua
-- ID du PAD token
local pad_id = Mimir.Tokenizer.getPadTokenId(tokenizer)

-- ID du UNK token
local unk_id = Mimir.Tokenizer.getUnkTokenId(tokenizer)

-- ID du BOS token
local bos_id = Mimir.Tokenizer.getBosTokenId(tokenizer)

-- ID du EOS token
local eos_id = Mimir.Tokenizer.getEosTokenId(tokenizer)
```

### Usage en Encodage

```lua
-- Ajouter BOS/EOS automatiquement
local ids = Mimir.Tokenizer.encode(tokenizer, "Hello world", {
    add_bos = true,
    add_eos = true
})

-- ids[1] = <BOS>
-- ids[2..n-1] = tokens réels
-- ids[n] = <EOS>
```

### Padding

```lua
-- Padder séquence à longueur fixe
local ids = Mimir.Tokenizer.encode(tokenizer, "Short text", {
    max_length = 128,
    padding = "max_length",
    truncation = true
})

-- Si len(ids) < 128 : ajouter <PAD>
-- Si len(ids) > 128 : tronquer
```

---

## 💡 Best Practices

### 1. Vocabulaire Cohérent

```lua
-- ✅ BON : Même vocabulaire train/inference
-- Training
local tokenizer = Mimir.Tokenizer.create()
Mimir.Tokenizer.trainBPE(tokenizer, "train.txt", 50000, "vocab.json")

-- Inference (plus tard)
local tokenizer = Mimir.Tokenizer.create()
Mimir.Tokenizer.loadVocab(tokenizer, "vocab.json")  -- Même vocab
```

### 2. Normalisation Texte

```lua
-- Normaliser avant encodage
function preprocess(text)
    text = text:lower()                  -- Minuscules
    text = text:gsub("%s+", " ")         -- Espaces multiples → 1
    text = text:gsub("^%s+", "")         -- Trim début
    text = text:gsub("%s+$", "")         -- Trim fin
    return text
end

local text = "  Hello   World!  "
local normalized = preprocess(text)
local ids = Mimir.Tokenizer.encode(tokenizer, normalized)
```

### 3. Gestion OOV

```lua
-- Vérifier tokens inconnus
local text = "Hello unknownword world"
local tokens = Mimir.Tokenizer.tokenize(tokenizer, text)

for i, token in ipairs(tokens) do
    if not Mimir.Tokenizer.hasToken(tokenizer, token) then
        print("Token inconnu:", token)
        -- Gérer : ajouter au vocab, remplacer par <UNK>, etc.
    end
end
```

### 4. Batch Encoding

```lua
-- Encoder plusieurs séquences
local texts = {
    "First sentence",
    "Second sentence",
    "Third sentence"
}

local all_ids = {}
for i, text in ipairs(texts) do
    all_ids[i] = Mimir.Tokenizer.encode(tokenizer, text)
end

-- Padder au même longueur
local max_len = 0
for _, ids in ipairs(all_ids) do
    max_len = math.max(max_len, #ids)
end

local pad_id = Mimir.Tokenizer.getPadTokenId(tokenizer)
for i, ids in ipairs(all_ids) do
    while #ids < max_len do
        table.insert(ids, pad_id)
    end
end

-- Maintenant : all_ids[1..3] ont même longueur
```

### 5. Sauvegarder Config

```lua
-- Sauvegarder tokenizer complet (vocab + config)
Mimir.Tokenizer.save(tokenizer, "tokenizer_full.json")

-- Recharger tout
local tokenizer2 = Mimir.Tokenizer.create()
Mimir.Tokenizer.load(tokenizer2, "tokenizer_full.json")
```

---

## 📚 Exemples Complets

### Exemple 1 : Tokenizer Simple

```lua
-- Créer et configurer
local tokenizer = Mimir.Tokenizer.create()

-- Définir vocabulaire manuel
local words = {"<PAD>", "<UNK>", "hello", "world", "foo", "bar"}
for i, word in ipairs(words) do
    Mimir.Tokenizer.addToken(tokenizer, word)
end

-- Configurer tokens spéciaux
Mimir.Tokenizer.setSpecialTokens(tokenizer, {
    pad_token = "<PAD>",
    unk_token = "<UNK>"
})

-- Encoder
local text = "hello world foo unknown"
local ids = Mimir.Tokenizer.encode(tokenizer, text)
print("IDs:", table.concat(ids, ", "))
-- IDs: 2, 3, 4, 1  (unknown → <UNK> = 1)

-- Décoder
local decoded = Mimir.Tokenizer.decode(tokenizer, ids)
print("Décodé:", decoded)
-- Décodé: hello world foo <UNK>
```

### Exemple 2 : BPE pour NLP

```lua
-- 1. Préparer corpus
local corpus = [[
The quick brown fox jumps over the lazy dog.
A journey of a thousand miles begins with a single step.
To be or not to be, that is the question.
]]

-- Sauvegarder corpus
local file = io.open("corpus.txt", "w")
file:write(corpus)
file:close()

-- 2. Entraîner BPE
local tokenizer = Mimir.Tokenizer.create()
Mimir.Tokenizer.trainBPE(tokenizer, "corpus.txt", 1000, "vocab_bpe.json")

-- 3. Utiliser
Mimir.Tokenizer.loadVocab(tokenizer, "vocab_bpe.json")

local test_text = "The fox jumps quickly"
local ids = Mimir.Tokenizer.encodeBPE(tokenizer, test_text)
local decoded = Mimir.Tokenizer.decodeBPE(tokenizer, ids)

print("Original:", test_text)
print("IDs:", table.concat(ids, ", "))
print("Décodé:", decoded)
```

### Exemple 3 : Dataset Tokenization

```lua
-- Tokenizer
local tokenizer = Mimir.Tokenizer.create()
Mimir.Tokenizer.loadVocab(tokenizer, "vocab.json")

-- Dataset brut
local raw_data = {
    {text = "I love this movie", label = 1},  -- Positive
    {text = "This film is terrible", label = 0},  -- Negative
    {text = "Great acting and story", label = 1}
}

-- Tokenizer tout le dataset
local tokenized_data = {}
for i, item in ipairs(raw_data) do
    local ids = Mimir.Tokenizer.encode(tokenizer, item.text, {
        max_length = 32,
        padding = "max_length",
        truncation = true
    })
    
    tokenized_data[i] = {
        input = {ids},  -- [1, seq_len]
        target = {{item.label, 1 - item.label}}  -- [1, 2] one-hot
    }
end

-- Sauvegarder dataset tokenizé
local json = require("json")
local file = io.open("dataset_tokenized.json", "w")
file:write(json.encode(tokenized_data))
file:close()

-- Charger pour entraînement
local dataset = Mimir.Dataset.loadFromJson("dataset_tokenized.json")
Mimir.Model.train(model, dataset, 10)
```

### Exemple 4 : Génération avec Tokenizer

```lua
-- Modèle de génération
local tokenizer = Mimir.Tokenizer.create()
Mimir.Tokenizer.loadVocab(tokenizer, "vocab.json")

local model = Mimir.Model.load("generator.json")

-- Fonction de génération
function generate_text(prompt, max_length)
    -- Encoder prompt
    local input_ids = Mimir.Tokenizer.encode(tokenizer, prompt, {
        add_bos = true
    })
    
    local eos_id = Mimir.Tokenizer.getEosTokenId(tokenizer)
    
    -- Générer token par token
    for i = 1, max_length do
        -- Forward
        local output = Mimir.Model.forward(model, {input_ids})
        
        -- Prendre dernier token (greedy)
        local next_token = argmax(output[1][#output[1]])
        
        -- Ajouter
        table.insert(input_ids, next_token)
        
        -- Stop si EOS
        if next_token == eos_id then
            break
        end
    end
    
    -- Décoder
    local generated_text = Mimir.Tokenizer.decode(tokenizer, input_ids, {
        skip_special_tokens = true
    })
    
    return generated_text
end

-- Utiliser
local prompt = "Once upon a time"
local story = generate_text(prompt, 100)
print(story)
```

---

## 🎯 Prochaines Étapes

- [Data Management](05-Data-Management.md) - Préparer datasets
- [Training](06-Training.md) - Entraîner modèles NLP
- [Inference](07-Inference.md) - Génération de texte

---

**Questions ?** Consultez [API Reference](../03-API-Reference/04-Tokenizer-API.md) pour toutes les fonctions.
