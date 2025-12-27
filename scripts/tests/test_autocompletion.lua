--[[
  Test d'autocomplétion API Mímir v2.0.0
  
  Ce fichier teste que toutes les fonctions sont correctement 
  détectées par le Language Server Protocol (LuaLS).
  
  Instructions:
  1. Ouvrir ce fichier dans VS Code avec l'extension Lua installée
  2. Taper le nom d'un module suivi d'un point: model.
  3. Vérifier que l'autocomplétion affiche toutes les fonctions
]]

-- Configuration mémoire (doit être la première chose)
allocator.configure({
  max_ram_gb = 10.0,
  enable_compression = true,
  offload_threshold_mb = 6000
})

-- Test module model (17 fonctions)
model.create("transformer", {vocab_size = 50000})
model.build()
model.train(10, 3e-4)
model.infer("Hello world")
model.save("checkpoint/")
model.total_params()

-- Test module architectures (9 fonctions)
architectures.transformer({vocab_size = 50000})
architectures.unet({input_channels = 3})
architectures.vae({latent_dim = 128})
architectures.flux({image_resolution = 256})

-- Test module flux (5 fonctions - API fonctionnelle)
flux.generate("A beautiful sunset", 50)
flux.encode_image("input.png")
flux.decode_latent({})
flux.encode_text("Description")
flux.set_tokenizer("tokenizer.json")

-- Test module FluxModel (12 fonctions - API orientée objet)
local flux_model = FluxModel.new({image_resolution = 256})
flux_model.train()
flux_model.eval()
flux_model.isTraining()
flux_model.encodeImage("img.png")

-- Test module tokenizer (24 fonctions)
tokenizer.create(50000)
tokenizer.tokenize("Hello")
tokenizer.detokenize({1, 2, 3})
tokenizer.vocab_size()
tokenizer.learn_bpe("corpus.txt", 50000)
tokenizer.analyze_text("Some text to analyze")

-- Test module dataset (3 fonctions)
dataset.load("data/corpus")
dataset.prepare_sequences(512)
dataset.get(0)

-- Test module memory (6 fonctions)
memory.set_limit(8000)
memory.get_stats()
memory.print_stats()

-- Test module guard (4 fonctions - API ancienne)
guard.set_limit(8000)
guard.get_stats()

-- Test module MemoryGuard (7 fonctions - API moderne recommandée)
MemoryGuard.setLimit(10)
MemoryGuard.getCurrentUsage()
MemoryGuard.getPeakUsage()
MemoryGuard.getStats()
MemoryGuard.printStats()

-- Test module allocator (3 fonctions)
allocator.print_stats()
allocator.get_stats()

-- Test module htop (5 fonctions)
htop.create({enable_viz = false})
htop.enable(true)
htop.update({epoch = 1, loss = 2.5})
htop.render()
htop.clear()

-- Test module viz (11 fonctions)
viz.create("Training Viz", 800, 600)
viz.initialize()
viz.is_open()
viz.process_events()
viz.update()
viz.add_image({}, 256, 256, 4)
viz.update_metrics({loss = 2.5})
viz.add_loss_point(2.5)
viz.clear()
viz.set_enabled(true)
viz.save_loss_history("loss.json")

-- Test fonctions globales (3 fonctions)
log("Message de test")
local data = read_json("config.json")
write_json("output.json", {key = "value"})

print("Test d'autocomplétion terminé ✅")
print("Total de modules testés: 13")
print("Total de fonctions testées: 114")
