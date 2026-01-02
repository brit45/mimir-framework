# Inférence

> **⚠️ AVERTISSEMENT IMPORTANT**  
> Les exemples de génération de texte sont théoriques.  
> API réelle : `Mimir.Model.infer(input)` ou `Mimir.Model.forward(input)`.  
> **Référez-vous à `scripts/example_gpt.lua` pour des exemples corrects.**  

Guide d'utilisation de modèles entraînés pour la génération et prédiction.

---

## 📋 Table des Matières

- [Forward Pass](#forward-pass)
- [Génération de Texte](#génération-de-texte)
- [Prédiction d'Images](#prédiction-dimages)
- [Batch Inference](#batch-inference)
- [Optimisation](#optimisation)

---

## ⚡ Forward Pass

### Simple Prédiction

```lua
-- Charger modèle
local model = Mimir.Model.load("trained_model.json")

-- Mode eval (désactive dropout)
model.setMode(model, "eval")

-- Input
local input = {{1.0, 2.0, 3.0, 4.0}}  -- [1, 4]

-- Forward
local output = Mimir.Model.forward(model, input)
-- output = [[0.1, 0.3, 0.6]] → [1, 3]

-- Interpréter
local predicted_class = argmax(output[1])
print("Prédiction:", predicted_class)
```

### Avec Preprocessing

```lua
function predict(model, raw_input)
    -- Preprocessing
    local normalized = normalize(raw_input)
    local input_tensor = {normalized}
    
    -- Forward
    local output = Mimir.Model.forward(model, input_tensor)
    
    -- Postprocessing
    local probabilities = softmax(output[1])
    local prediction = argmax(probabilities)
    
    return prediction, probabilities
end

-- Usage
local image = load_image("test.png")
local class_id, probs = predict(model, image)
print("Classe:", class_id, "Confiance:", probs[class_id])
```

---

## 📝 Génération de Texte

### Greedy Decoding

```lua
function generate_greedy(model, tokenizer, prompt, max_length)
    -- Encoder prompt
    local input_ids = Mimir.Tokenizer.encode(tokenizer, prompt)
    local eos_id = Mimir.Tokenizer.getEosTokenId(tokenizer)
    
    for i = 1, max_length do
        -- Forward
        local output = Mimir.Model.forward(model, {input_ids})
        
        -- Prendre token avec plus haute probabilité
        local logits = output[1][#output[1]]
        local next_token = argmax(logits)
        
        -- Ajouter
        table.insert(input_ids, next_token)
        
        -- Stop si EOS
        if next_token == eos_id then
            break
        end
    end
    
    -- Décoder
    return Mimir.Tokenizer.decode(tokenizer, input_ids)
end

-- Usage
local model = Mimir.Model.load("gpt_model.json")
local tokenizer = Mimir.Tokenizer.create()
Mimir.Tokenizer.loadVocab(tokenizer, "vocab.json")

local prompt = "Once upon a time"
local story = generate_greedy(model, tokenizer, prompt, 100)
print(story)
```

### Sampling

```lua
function generate_with_temperature(model, tokenizer, prompt, max_length, temperature)
    local input_ids = Mimir.Tokenizer.encode(tokenizer, prompt)
    local eos_id = Mimir.Tokenizer.getEosTokenId(tokenizer)
    
    for i = 1, max_length do
        local output = Mimir.Model.forward(model, {input_ids})
        local logits = output[1][#output[1]]
        
        -- Appliquer température
        for j = 1, #logits do
            logits[j] = logits[j] / temperature
        end
        
        -- Softmax + sampling
        local probs = softmax(logits)
        local next_token = sample_from_distribution(probs)
        
        table.insert(input_ids, next_token)
        
        if next_token == eos_id then
            break
        end
    end
    
    return Mimir.Tokenizer.decode(tokenizer, input_ids)
end

-- Temperature :
-- < 1.0 : Plus déterministe
-- = 1.0 : Distribution normale
-- > 1.0 : Plus aléatoire

local text1 = generate_with_temperature(model, tokenizer, prompt, 50, 0.7)  -- Conservative
local text2 = generate_with_temperature(model, tokenizer, prompt, 50, 1.2)  -- Creative
```

### Top-K Sampling

```lua
function generate_topk(model, tokenizer, prompt, max_length, k)
    local input_ids = Mimir.Tokenizer.encode(tokenizer, prompt)
    local eos_id = Mimir.Tokenizer.getEosTokenId(tokenizer)
    
    for i = 1, max_length do
        local output = Mimir.Model.forward(model, {input_ids})
        local logits = output[1][#output[1]]
        
        -- Garder top-K
        local top_k_indices, top_k_probs = get_top_k(logits, k)
        
        -- Renormaliser et sampler
        local probs = softmax(top_k_probs)
        local sampled_idx = sample_from_distribution(probs)
        local next_token = top_k_indices[sampled_idx]
        
        table.insert(input_ids, next_token)
        
        if next_token == eos_id then
            break
        end
    end
    
    return Mimir.Tokenizer.decode(tokenizer, input_ids)
end

-- k = 5 : Très conservateur
-- k = 50 : Équilibré
-- k = 100 : Plus créatif
```

### Beam Search

```lua
function beam_search(model, tokenizer, prompt, max_length, beam_width)
    local eos_id = Mimir.Tokenizer.getEosTokenId(tokenizer)
    local start_ids = Mimir.Tokenizer.encode(tokenizer, prompt)
    
    -- Initialiser beams
    local beams = {{ids = start_ids, score = 0.0}}
    
    for step = 1, max_length do
        local candidates = {}
        
        for _, beam in ipairs(beams) do
            if beam.ids[#beam.ids] == eos_id then
                table.insert(candidates, beam)
            else
                -- Générer suite
                local output = Mimir.Model.forward(model, {beam.ids})
                local logits = output[1][#output[1]]
                local probs = softmax(logits)
                
                -- Top-K candidats
                local top_k_indices, top_k_probs = get_top_k(probs, beam_width)
                
                for i = 1, beam_width do
                    local new_ids = copy(beam.ids)
                    table.insert(new_ids, top_k_indices[i])
                    
                    local new_score = beam.score + math.log(top_k_probs[i])
                    
                    table.insert(candidates, {ids = new_ids, score = new_score})
                end
            end
        end
        
        -- Garder top beam_width
        table.sort(candidates, function(a, b) return a.score > b.score end)
        beams = {}
        for i = 1, math.min(beam_width, #candidates) do
            table.insert(beams, candidates[i])
        end
    end
    
    -- Retourner meilleur beam
    local best_beam = beams[1]
    return Mimir.Tokenizer.decode(tokenizer, best_beam.ids)
end

-- beam_width = 3-5 : Bon compromis qualité/vitesse
```

---

## 🖼️ Prédiction d'Images

### Classification

```lua
function classify_image(model, image_path, class_names)
    -- Charger et preprocesser
    local image = load_image(image_path)
    image = resize(image, 224, 224)
    image = normalize(image, {0.485, 0.456, 0.406}, {0.229, 0.224, 0.225})
    
    -- Forward
    model.setMode(model, "eval")
    local output = Mimir.Model.forward(model, {image})
    
    -- Top-5 prédictions
    local probs = softmax(output[1])
    local top5_indices, top5_probs = get_top_k(probs, 5)
    
    print("Top 5 prédictions:")
    for i = 1, 5 do
        local class_id = top5_indices[i]
        local confidence = top5_probs[i]
        print(string.format("%d. %s: %.2f%%", i, class_names[class_id], confidence * 100))
    end
    
    return top5_indices[1], top5_probs[1]
end

-- Usage
local model = Mimir.Model.load("resnet50.json")
local classes = load_imagenet_classes("imagenet_classes.txt")
classify_image(model, "cat.jpg", classes)
```

### Segmentation

```lua
function segment_image(model, image_path, num_classes)
    local image = load_image(image_path)
    local h, w = get_dimensions(image)
    
    -- Forward
    local output = Mimir.Model.forward(model, {image})
    -- output = [1, num_classes, h, w]
    
    -- Argmax par pixel
    local segmentation_map = {}
    for i = 1, h do
        segmentation_map[i] = {}
        for j = 1, w do
            local pixel_logits = {}
            for c = 1, num_classes do
                pixel_logits[c] = output[1][c][i][j]
            end
            segmentation_map[i][j] = argmax(pixel_logits)
        end
    end
    
    return segmentation_map
end

-- Visualiser
local seg_map = segment_image(unet_model, "medical_scan.png", 2)
save_segmentation(seg_map, "output_mask.png")
```

---

## 📦 Batch Inference

### Parallélisation

```lua
function batch_predict(model, inputs, batch_size)
    local results = {}
    
    for i = 1, #inputs, batch_size do
        -- Créer batch
        local batch = {}
        for j = i, math.min(i + batch_size - 1, #inputs) do
            table.insert(batch, inputs[j])
        end
        
        -- Padder au même longueur
        batch = pad_batch(batch)
        
        -- Forward
        local outputs = Mimir.Model.forward(model, batch)
        
        -- Collecter
        for k, output in ipairs(outputs) do
            table.insert(results, output)
        end
    end
    
    return results
end

-- Usage
local test_images = load_all_images("test/*.jpg")
local predictions = batch_predict(model, test_images, 32)
```

---

## ⚡ Optimisation

### Caching

```lua
-- Cache KV pour Transformers
local kv_cache = {}

function forward_with_cache(model, input_ids, cache)
    -- Utiliser cache si disponible
    if cache.keys and cache.values then
        -- Forward uniquement sur nouveau token
        local output = model.forwardIncremental(model, input_ids, cache)
        return output
    else
        -- Forward complet
        local output = Mimir.Model.forward(model, input_ids)
        cache.keys, cache.values = model.getKVCache(model)
        return output
    end
end

-- 3-5x plus rapide pour génération autoregressive
```

### Quantization

```lua
-- Quantize modèle en INT8
local quantized_model = model.quantize(model, "int8")

-- Plus petit (4x), plus rapide (2-3x)
-- Légère perte précision (<1%)
```

### Mode Inference

```lua
-- Optimisations inference
model.setMode(model, "eval")  -- Désactive dropout, batchnorm training

-- Optimisations SIMD
model.enableSIMD(model, true)

-- Optimisations mémoire
model.setInferenceMode(model, true)  -- Pas de gradients
```

---

## 🎯 Exemples Complets

### Chatbot Simple

```lua
local model = Mimir.Model.load("chatbot.json")
local tokenizer = Mimir.Tokenizer.create()
Mimir.Tokenizer.loadVocab(tokenizer, "vocab.json")

local conversation_history = ""

while true do
    io.write("You: ")
    local user_input = io.read()
    
    if user_input == "quit" then
        break
    end
    
    -- Ajouter à historique
    conversation_history = conversation_history .. "User: " .. user_input .. "\n"
    
    -- Générer réponse
    local response = generate_with_temperature(
        model, tokenizer, conversation_history .. "Bot:", 50, 0.8
    )
    
    -- Extraire uniquement réponse bot
    response = response:match("Bot: (.+)")
    
    conversation_history = conversation_history .. "Bot: " .. response .. "\n"
    
    print("Bot:", response)
end
```

### Analyseur de Sentiment

```lua
function analyze_sentiment(model, tokenizer, text)
    -- Preprocesser
    text = text:lower()
    local input_ids = Mimir.Tokenizer.encode(tokenizer, text, {
        max_length = 128,
        padding = "max_length",
        truncation = true
    })
    
    -- Prédire
    local output = Mimir.Model.forward(model, {input_ids})
    local probs = softmax(output[1])
    
    -- Interpréter
    local sentiments = {"Négatif", "Neutre", "Positif"}
    local predicted_sentiment = argmax(probs)
    local confidence = probs[predicted_sentiment]
    
    return sentiments[predicted_sentiment], confidence
end

-- Test
local texts = {
    "Ce film est incroyable!",
    "Je n'ai pas aimé du tout.",
    "C'était correct, sans plus."
}

for i, text in ipairs(texts) do
    local sentiment, conf = analyze_sentiment(model, tokenizer, text)
    print(string.format("%s → %s (%.1f%%)", text, sentiment, conf * 100))
end
```

---

## 🎯 Prochaines Étapes

- [Save/Load](08-Save-Load.md) - Sauvegarder résultats
- [Advanced](../05-Advanced/) - Optimisation avancée
- [Deployment](../05-Advanced/07-Deployment.md) - Production

---

**Questions ?** Consultez [INDEX](../00-INDEX.md).
