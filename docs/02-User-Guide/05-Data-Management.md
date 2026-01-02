# Gestion des Données

> **⚠️ AVERTISSEMENT**  
> Cette documentation peut contenir des inexactitudes sur l'API Mimir.Dataset.  
> **Référez-vous aux scripts dans `scripts/` pour des exemples vérifiés.**  

Guide de préparation et gestion des datasets dans Mímir Framework.

---

## 📋 Table des Matières

- [Structure Dataset](#structure-dataset)
- [Chargement](#chargement)
- [Préparation](#préparation)
- [Augmentation](#augmentation)
- [Batching](#batching)

---

## 📦 Structure Dataset

### Format Basique

```lua
local dataset = {
    {
        input = {{1.0, 2.0, 3.0}},  -- [1, 3]
        target = {{1.0, 0.0}}       -- [1, 2]
    },
    {
        input = {{4.0, 5.0, 6.0}},
        target = {{0.0, 1.0}}
    }
}
```

### Format JSON

```json
[
    {
        "input": [[1.0, 2.0, 3.0]],
        "target": [[1.0, 0.0]]
    },
    {
        "input": [[4.0, 5.0, 6.0]],
        "target": [[0.0, 1.0]]
    }
]
```

---

## 📂 Chargement

### Depuis JSON

```lua
local dataset = Mimir.Dataset.loadFromJson("data.json")
print("Dataset size:", #dataset)
```

### Depuis Texte

```lua
-- Pour NLP
local tokenizer = Mimir.Tokenizer.create()
Mimir.Tokenizer.loadVocab(tokenizer, "vocab.json")

local dataset = Mimir.Dataset.loadText("corpus.txt", tokenizer)
-- Crée paires (input, target) pour language modeling
```

### Création Manuelle

```lua
local dataset = {}
for i = 1, 1000 do
    local input = generate_input()
    local target = compute_target(input)
    table.insert(dataset, {input = input, target = target})
end
```

---

## 🔧 Préparation

### Normalisation

```lua
-- Normaliser inputs [0, 1]
function normalize_dataset(dataset, min_val, max_val)
    for i, item in ipairs(dataset) do
        for j, row in ipairs(item.input) do
            for k, val in ipairs(row) do
                item.input[j][k] = (val - min_val) / (max_val - min_val)
            end
        end
    end
    return dataset
end

-- Usage
local dataset = Mimir.Dataset.loadFromJson("raw_data.json")
dataset = normalize_dataset(dataset, 0, 255)  -- Images 0-255 → 0-1
```

### Split Train/Val/Test

```lua
function split_dataset(dataset, train_ratio, val_ratio)
    local n = #dataset
    local train_size = math.floor(n * train_ratio)
    local val_size = math.floor(n * val_ratio)
    
    local train, val, test = {}, {}, {}
    
    for i = 1, train_size do
        table.insert(train, dataset[i])
    end
    for i = train_size + 1, train_size + val_size do
        table.insert(val, dataset[i])
    end
    for i = train_size + val_size + 1, n do
        table.insert(test, dataset[i])
    end
    
    return train, val, test
end

-- Usage
local dataset = Mimir.Dataset.loadFromJson("all_data.json")
local train, val, test = split_dataset(dataset, 0.8, 0.1)
print("Train:", #train, "Val:", #val, "Test:", #test)
```

### Shuffle

```lua
function shuffle_dataset(dataset)
    for i = #dataset, 2, -1 do
        local j = math.random(1, i)
        dataset[i], dataset[j] = dataset[j], dataset[i]
    end
    return dataset
end

-- Usage
local dataset = Mimir.Dataset.loadFromJson("data.json")
dataset = shuffle_dataset(dataset)
```

---

## 🎨 Augmentation

### Images

```lua
function augment_image(image)
    -- Flip horizontal (50% chance)
    if math.random() > 0.5 then
        image = flip_horizontal(image)
    end
    
    -- Rotation aléatoire (-10° to +10°)
    local angle = (math.random() - 0.5) * 20
    image = rotate(image, angle)
    
    -- Brightness
    local brightness = 0.8 + math.random() * 0.4  -- 0.8-1.2
    image = adjust_brightness(image, brightness)
    
    return image
end

-- Appliquer
for i, item in ipairs(dataset) do
    item.input = augment_image(item.input)
end
```

### Texte

```lua
function augment_text(text, tokenizer)
    -- Synonym replacement
    text = replace_with_synonyms(text, 0.1)
    
    -- Random deletion (10% mots)
    text = random_deletion(text, 0.1)
    
    -- Random swap
    text = random_swap(text, 2)
    
    return text
end
```

---

## 📊 Batching

### Créer Batches

```lua
function create_batches(dataset, batch_size)
    local batches = {}
    for i = 1, #dataset, batch_size do
        local batch = {}
        for j = i, math.min(i + batch_size - 1, #dataset) do
            table.insert(batch, dataset[j])
        end
        table.insert(batches, batch)
    end
    return batches
end

-- Usage
local dataset = Mimir.Dataset.loadFromJson("data.json")
local batches = create_batches(dataset, 32)
print("Nombre de batches:", #batches)

-- Entraîner par batch
for epoch = 1, 10 do
    for i, batch in ipairs(batches) do
        model.trainBatch(model, batch)
    end
end
```

### Padding Dynamique

```lua
function pad_batch(batch, pad_id)
    -- Trouver max longueur
    local max_len = 0
    for i, item in ipairs(batch) do
        max_len = math.max(max_len, #item.input[1])
    end
    
    -- Padder tous
    for i, item in ipairs(batch) do
        while #item.input[1] < max_len do
            table.insert(item.input[1], pad_id)
        end
    end
    
    return batch
end
```

---

## 📚 Exemples Complets

### Dataset MNIST-like

```lua
-- Charger images brutes
function load_mnist_dataset(images_file, labels_file)
    local images = load_binary_images(images_file)  -- [N, 28, 28]
    local labels = load_binary_labels(labels_file)  -- [N]
    
    local dataset = {}
    for i = 1, #images do
        -- Flatten image
        local input = {}
        for j = 1, 28 do
            for k = 1, 28 do
                table.insert(input, images[i][j][k] / 255.0)  -- Normalize
            end
        end
        
        -- One-hot label
        local target = {}
        for j = 1, 10 do
            table.insert(target, j == labels[i] + 1 and 1.0 or 0.0)
        end
        
        table.insert(dataset, {
            input = {input},     -- [1, 784]
            target = {target}    -- [1, 10]
        })
    end
    
    return dataset
end

-- Utiliser
local dataset = load_mnist_dataset("train-images.idx", "train-labels.idx")
dataset = shuffle_dataset(dataset)
local train, val = split_dataset(dataset, 0.9, 0.1)

Mimir.Model.train(model, train, 10)
```

### Dataset NLP

```lua
-- Préparer dataset texte
function prepare_nlp_dataset(texts, labels, tokenizer, max_length)
    local dataset = {}
    
    for i, text in ipairs(texts) do
        -- Tokenize
        local ids = Mimir.Tokenizer.encode(tokenizer, text, {
            max_length = max_length,
            padding = "max_length",
            truncation = true
        })
        
        -- One-hot label
        local target = {}
        for j = 1, num_classes do
            table.insert(target, j == labels[i] and 1.0 or 0.0)
        end
        
        table.insert(dataset, {
            input = {ids},
            target = {target}
        })
    end
    
    return dataset
end

-- Usage
local tokenizer = Mimir.Tokenizer.create()
Mimir.Tokenizer.loadVocab(tokenizer, "vocab.json")

local texts = load_texts("reviews.txt")
local labels = load_labels("labels.txt")

local dataset = prepare_nlp_dataset(texts, labels, tokenizer, 128)
Mimir.Model.train(model, dataset, 20)
```

---

## 🎯 Prochaines Étapes

- [Training](06-Training.md) - Entraîner avec ces datasets
- [Tokenization](04-Tokenization.md) - Préparation NLP
- [API Reference](../03-API-Reference/05-Dataset-API.md)

---

**Questions ?** Consultez [INDEX](../00-INDEX.md).
