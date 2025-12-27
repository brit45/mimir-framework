# Monitoring

> **⚠️ AVERTISSEMENT**  
> Les API htop et viz peuvent être partiellement incorrectes.  
> Fonctions réelles : `htop.create()`, `viz.initialize()`, etc.  
> **Référez-vous aux scripts dans `scripts/` pour usage vérifié.**  
> Voir [VERIFICATION_REPORT.md](../VERIFICATION_REPORT.md) pour les détails.

Guide de surveillance et visualisation de l'entraînement dans Mímir Framework.

---

## 📋 Table des Matières

- [Htop Display](#htop-display)
- [Visualizer SFML](#visualizer-sfml)
- [Métriques Custom](#métriques-custom)
- [AsyncMonitor](#asyncmonitor)
- [Logging](#logging)

---

## 📊 Htop Display

### Démarrage

```lua
-- Démarrer monitoring terminal
htop.start()

-- Configurer refresh rate (ms)
htop.setRefreshRate(500)  -- Update tous les 500ms

-- Entraîner (htop affiche en temps réel)
model.train(model, dataset, epochs)

-- Arrêter
htop.stop()
```

### Affichage

```
┌─────────────────────────────────────────────────┐
│          Mímir Framework - Training             │
├─────────────────────────────────────────────────┤
│ Epoch: 15/50                                    │
│ Batch: 234/1000                                 │
│ Loss: 0.3421                                    │
│ Learning Rate: 0.001                            │
├─────────────────────────────────────────────────┤
│ Memory Usage:                                   │
│  Used: 2.3 GB / 8.0 GB [████████░░░░░] 29%    │
│  Tensors: 145                                   │
│  Peak: 3.1 GB                                   │
├─────────────────────────────────────────────────┤
│ Performance:                                    │
│  Speed: 234 samples/sec                         │
│  Time elapsed: 00:23:45                         │
│  ETA: 01:12:30                                  │
└─────────────────────────────────────────────────┘
```

### Configuration Avancée

```lua
-- Personnaliser affichage
htop.setDisplayMode("compact")  -- compact, full, minimal

-- Activer/désactiver sections
htop.showMemory(true)
htop.showPerformance(true)
htop.showProgress(true)

-- Couleurs
htop.setColorScheme("dark")  -- dark, light, monochrome
```

---

## 🎨 Visualizer SFML

### Fenêtre de Base

```lua
-- Créer fenêtre
viz.createWindow("Training Metrics", 800, 600)

-- Boucle d'entraînement
for epoch = 1, epochs do
    local loss = train_one_epoch(model, dataset)
    
    -- Afficher métriques
    viz.clear()
    viz.plotLine("Loss", epoch, loss)
    viz.render()
end

-- Fermer
viz.closeWindow()
```

### Graphiques Multiples

```lua
viz.createWindow("Dashboard", 1200, 800)

local train_losses = {}
local val_losses = {}
local accuracies = {}

for epoch = 1, epochs do
    -- Entraîner
    local train_loss = train_one_epoch(model, train_data)
    local val_loss = evaluate_loss(model, val_data)
    local accuracy = evaluate_accuracy(model, val_data)
    
    table.insert(train_losses, train_loss)
    table.insert(val_losses, val_loss)
    table.insert(accuracies, accuracy)
    
    -- Visualiser
    viz.clear()
    
    -- Plot 1: Loss
    viz.subplot(2, 2, 1)
    viz.plotLines({
        {name = "Train", data = train_losses, color = "blue"},
        {name = "Val", data = val_losses, color = "red"}
    })
    viz.title("Loss over Epochs")
    
    -- Plot 2: Accuracy
    viz.subplot(2, 2, 2)
    viz.plotLine("Accuracy", accuracies, "green")
    viz.title("Validation Accuracy")
    
    -- Plot 3: Learning Rate
    viz.subplot(2, 2, 3)
    local lr = model.getLearningRate(model)
    viz.plotScalar("LR", lr)
    
    -- Plot 4: Gradient Norms
    viz.subplot(2, 2, 4)
    local grad_norm = model.getGradientNorm(model)
    viz.plotLine("Grad Norm", epoch, grad_norm)
    
    viz.render()
end

viz.closeWindow()
```

### Visualisation Poids

```lua
-- Visualiser distribution poids
viz.createWindow("Weight Distribution", 800, 600)

local weights = model.getWeights(model)

for layer_name, layer_weights in pairs(weights) do
    -- Histogramme
    viz.histogram(layer_weights, layer_name)
end

viz.render()
viz.waitKey()  -- Attendre appui touche
viz.closeWindow()
```

### Heatmaps

```lua
-- Visualiser attention weights
viz.createWindow("Attention", 800, 800)

local attention = model.getAttentionWeights(model)
-- attention = [num_heads, seq_len, seq_len]

for head = 1, num_heads do
    viz.subplot(4, 4, head)
    viz.heatmap(attention[head])
    viz.title("Head " .. head)
end

viz.render()
```

---

## 📈 Métriques Custom

### Tracker Personnalisé

```lua
local MetricsTracker = {}
MetricsTracker.__index = MetricsTracker

function MetricsTracker.new()
    local self = setmetatable({}, MetricsTracker)
    self.metrics = {}
    return self
end

function MetricsTracker:add(name, value)
    if not self.metrics[name] then
        self.metrics[name] = {}
    end
    table.insert(self.metrics[name], value)
end

function MetricsTracker:get(name)
    return self.metrics[name] or {}
end

function MetricsTracker:average(name)
    local values = self:get(name)
    if #values == 0 then return 0 end
    
    local sum = 0
    for i, v in ipairs(values) do
        sum = sum + v
    end
    return sum / #values
end

function MetricsTracker:save(path)
    local json = require("json")
    local file = io.open(path, "w")
    file:write(json.encode(self.metrics))
    file:close()
end

-- Usage
local tracker = MetricsTracker.new()

for epoch = 1, epochs do
    local loss = train_one_epoch(model, dataset)
    local acc = evaluate(model, val_data)
    
    tracker:add("loss", loss)
    tracker:add("accuracy", acc)
    
    print(string.format("Epoch %d - Loss: %.4f, Acc: %.2f%%", 
        epoch, loss, acc * 100))
end

tracker:save("metrics.json")
```

### Callbacks

```lua
-- Système de callbacks
local Callbacks = {}

function Callbacks.on_epoch_begin(epoch)
    print("Starting epoch", epoch)
end

function Callbacks.on_epoch_end(epoch, metrics)
    print("Epoch", epoch, "completed")
    print("Metrics:", json.encode(metrics))
    
    -- Sauvegarder checkpoint
    if epoch % 10 == 0 then
        model.save(model, "checkpoint_" .. epoch .. ".json")
    end
end

function Callbacks.on_batch_end(batch, loss)
    if batch % 100 == 0 then
        print("Batch", batch, "- Loss:", loss)
    end
end

-- Intégrer dans entraînement
function train_with_callbacks(model, dataset, epochs, callbacks)
    for epoch = 1, epochs do
        callbacks.on_epoch_begin(epoch)
        
        local epoch_loss = 0
        for batch, item in ipairs(dataset) do
            local loss = train_batch(model, item)
            epoch_loss = epoch_loss + loss
            
            callbacks.on_batch_end(batch, loss)
        end
        
        local metrics = {
            loss = epoch_loss / #dataset,
            accuracy = evaluate(model, val_data)
        }
        
        callbacks.on_epoch_end(epoch, metrics)
    end
end
```

---

## 🔄 AsyncMonitor

### Monitoring Asynchrone

```lua
-- Démarrer monitoring thread
mimir.async.startMonitor()

-- Enregistrer métriques (thread-safe)
for epoch = 1, epochs do
    local loss = train_one_epoch(model, dataset)
    
    -- Publier métrique (non-bloquant)
    mimir.async.publishMetric("epoch", epoch)
    mimir.async.publishMetric("loss", loss)
end

-- Récupérer statistiques
local stats = mimir.async.getStats()
print("Average loss:", stats.loss.mean)
print("Peak memory:", stats.memory.peak)

-- Arrêter
mimir.async.stopMonitor()
```

### Alertes

```lua
-- Configurer alertes
mimir.async.setAlert("loss", {
    condition = "greater_than",
    threshold = 10.0,
    callback = function(value)
        print("⚠️  WARNING: Loss exploding:", value)
        -- Sauvegarder avant crash
        model.save(model, "emergency_backup.json")
    end
})

mimir.async.setAlert("memory", {
    condition = "greater_than",
    threshold = 7.5 * 1024 * 1024 * 1024,  -- 7.5 GB
    callback = function(usage)
        print("⚠️  WARNING: High memory usage:", usage / 1e9, "GB")
        memory.clearCache()
    end
})
```

---

## 📝 Logging

### Logger Simple

```lua
local Logger = {}
Logger.__index = Logger

function Logger.new(path)
    local self = setmetatable({}, Logger)
    self.file = io.open(path, "w")
    self.start_time = os.time()
    
    -- Header
    self.file:write("timestamp,epoch,batch,loss,accuracy,lr,memory\n")
    self.file:flush()
    
    return self
end

function Logger:log(epoch, batch, loss, accuracy, lr, memory)
    local elapsed = os.time() - self.start_time
    self.file:write(string.format("%d,%d,%d,%.6f,%.4f,%.6f,%d\n",
        elapsed, epoch, batch, loss, accuracy, lr, memory))
    self.file:flush()
end

function Logger:close()
    self.file:close()
end

-- Usage
local logger = Logger.new("training.csv")

for epoch = 1, epochs do
    for batch, item in ipairs(dataset) do
        local loss = train_batch(model, item)
        local acc = evaluate_batch(model, item)
        local lr = model.getLearningRate(model)
        local mem = memory.getUsage()
        
        logger:log(epoch, batch, loss, acc, lr, mem)
    end
end

logger:close()
```

### TensorBoard-like

```lua
-- Export format compatible TensorBoard
function export_to_tensorboard(metrics, output_dir)
    os.execute("mkdir -p " .. output_dir)
    
    for metric_name, values in pairs(metrics) do
        local path = output_dir .. "/" .. metric_name .. ".txt"
        local file = io.open(path, "w")
        
        for i, value in ipairs(values) do
            file:write(string.format("%d\t%.6f\n", i, value))
        end
        
        file:close()
    end
end

-- Usage
local metrics = {
    train_loss = {},
    val_loss = {},
    accuracy = {}
}

for epoch = 1, epochs do
    table.insert(metrics.train_loss, train_one_epoch(model, train_data))
    table.insert(metrics.val_loss, evaluate_loss(model, val_data))
    table.insert(metrics.accuracy, evaluate_accuracy(model, val_data))
end

export_to_tensorboard(metrics, "tensorboard_logs")
```

---

## 📚 Exemple Complet

```lua
-- Setup monitoring complet
htop.start()
viz.createWindow("Training Dashboard", 1200, 800)
mimir.async.startMonitor()

local logger = Logger.new("training.csv")
local tracker = MetricsTracker.new()

-- Alertes
mimir.async.setAlert("loss", {
    condition = "greater_than",
    threshold = 10.0,
    callback = function() print("Loss alert!") end
})

-- Entraînement
for epoch = 1, epochs do
    print("Epoch", epoch)
    
    -- Train
    local train_loss = 0
    for i, item in ipairs(train_data) do
        local loss = train_batch(model, item)
        train_loss = train_loss + loss
        
        logger:log(epoch, i, loss, 0, lr, memory)
    end
    train_loss = train_loss / #train_data
    
    -- Evaluate
    local val_loss = evaluate_loss(model, val_data)
    local accuracy = evaluate_accuracy(model, val_data)
    
    -- Track
    tracker:add("train_loss", train_loss)
    tracker:add("val_loss", val_loss)
    tracker:add("accuracy", accuracy)
    
    -- Visualize
    viz.clear()
    viz.subplot(2, 2, 1)
    viz.plotLines({
        {name = "Train", data = tracker:get("train_loss")},
        {name = "Val", data = tracker:get("val_loss")}
    })
    viz.render()
end

-- Cleanup
logger:close()
tracker:save("final_metrics.json")
htop.stop()
viz.closeWindow()
mimir.async.stopMonitor()
```

---

## 🎯 Prochaines Étapes

- [Advanced](../05-Advanced/) - Techniques avancées
- [Debugging](../05-Advanced/05-Debugging.md) - Débug approfondi
- [API Reference](../03-API-Reference/09-Htop-API.md)

---

**Questions ?** Consultez [INDEX](../00-INDEX.md).
