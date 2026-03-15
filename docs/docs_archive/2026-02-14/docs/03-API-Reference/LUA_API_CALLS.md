# Appels API Lua – Mímir Framework

Source: `mimir-api.lua` (stub EmmyLua synchronisé avec `src/LuaScripting.cpp`).

> Généré automatiquement le 14 janvier 2026.

## Mimir.Model (18)

- `Mimir.Model.allocate_params(...)`
- `Mimir.Model.backward(...)`
- `Mimir.Model.build(...)`
- `Mimir.Model.create(...)`
- `Mimir.Model.forward(...)`
- `Mimir.Model.get_gradients(...)`
- `Mimir.Model.hardware_caps(...)`
- `Mimir.Model.infer(...)`
- `Mimir.Model.init_weights(...)`
- `Mimir.Model.load(...)`
- `Mimir.Model.optimizer_step(...)`
- `Mimir.Model.push_layer(...)`
- `Mimir.Model.save(...)`
- `Mimir.Model.set_hardware(...)`
- `Mimir.Model.set_layer_io(...)`
- `Mimir.Model.total_params(...)`
- `Mimir.Model.train(...)`
- `Mimir.Model.zero_grads(...)`

## Mimir.Architectures (2)

- `Mimir.Architectures.available(...)`
- `Mimir.Architectures.default_config(...)`

## Mimir.Layers (8)

- `Mimir.Layers.activation(...)`
- `Mimir.Layers.attention(...)`
- `Mimir.Layers.avgpool2d(...)`
- `Mimir.Layers.batchnorm(...)`
- `Mimir.Layers.conv2d(...)`
- `Mimir.Layers.layernorm(...)`
- `Mimir.Layers.linear(...)`
- `Mimir.Layers.maxpool2d(...)`

## Mimir.Tokenizer (24)

- `Mimir.Tokenizer.add_token(...)`
- `Mimir.Tokenizer.analyze_text(...)`
- `Mimir.Tokenizer.batch_tokenize(...)`
- `Mimir.Tokenizer.create(...)`
- `Mimir.Tokenizer.detokenize(...)`
- `Mimir.Tokenizer.ensure_vocab_from_text(...)`
- `Mimir.Tokenizer.extract_keywords(...)`
- `Mimir.Tokenizer.get_frequencies(...)`
- `Mimir.Tokenizer.get_token_by_id(...)`
- `Mimir.Tokenizer.learn_bpe(...)`
- `Mimir.Tokenizer.load(...)`
- `Mimir.Tokenizer.mag_id(...)`
- `Mimir.Tokenizer.mod_id(...)`
- `Mimir.Tokenizer.pad_id(...)`
- `Mimir.Tokenizer.pad_sequence(...)`
- `Mimir.Tokenizer.print_stats(...)`
- `Mimir.Tokenizer.save(...)`
- `Mimir.Tokenizer.seq_id(...)`
- `Mimir.Tokenizer.set_max_length(...)`
- `Mimir.Tokenizer.tokenize(...)`
- `Mimir.Tokenizer.tokenize_bpe(...)`
- `Mimir.Tokenizer.tokenize_ensure(...)`
- `Mimir.Tokenizer.unk_id(...)`
- `Mimir.Tokenizer.vocab_size(...)`

## Mimir.Dataset (3)

- `Mimir.Dataset.get(...)`
- `Mimir.Dataset.load(...)`
- `Mimir.Dataset.prepare_sequences(...)`

## Mimir.Memory (6)

- `Mimir.Memory.clear(...)`
- `Mimir.Memory.config(...)`
- `Mimir.Memory.get_stats(...)`
- `Mimir.Memory.get_usage(...)`
- `Mimir.Memory.print_stats(...)`
- `Mimir.Memory.set_limit(...)`

## Mimir.Guard (4)

- `Mimir.Guard.get_stats(...)`
- `Mimir.Guard.print_stats(...)`
- `Mimir.Guard.reset(...)`
- `Mimir.Guard.set_limit(...)`

## Mimir.MemoryGuard (7)

- `Mimir.MemoryGuard.getCurrentUsage(...)`
- `Mimir.MemoryGuard.getLimit(...)`
- `Mimir.MemoryGuard.getPeakUsage(...)`
- `Mimir.MemoryGuard.getStats(...)`
- `Mimir.MemoryGuard.printStats(...)`
- `Mimir.MemoryGuard.reset(...)`
- `Mimir.MemoryGuard.setLimit(...)`

## Mimir.Allocator (3)

- `Mimir.Allocator.configure(...)`
- `Mimir.Allocator.get_stats(...)`
- `Mimir.Allocator.print_stats(...)`

## Mimir.Htop (5)

- `Mimir.Htop.clear(...)`
- `Mimir.Htop.create(...)`
- `Mimir.Htop.enable(...)`
- `Mimir.Htop.render(...)`
- `Mimir.Htop.update(...)`

## Mimir.Viz (11)

- `Mimir.Viz.add_image(...)`
- `Mimir.Viz.add_loss_point(...)`
- `Mimir.Viz.clear(...)`
- `Mimir.Viz.create(...)`
- `Mimir.Viz.initialize(...)`
- `Mimir.Viz.is_open(...)`
- `Mimir.Viz.process_events(...)`
- `Mimir.Viz.save_loss_history(...)`
- `Mimir.Viz.set_enabled(...)`
- `Mimir.Viz.update(...)`
- `Mimir.Viz.update_metrics(...)`

## Mimir.Serialization (4)

- `Mimir.Serialization.detect_format(...)`
- `Mimir.Serialization.load(...)`
- `Mimir.Serialization.save(...)`
- `Mimir.Serialization.save_enhanced_debug(...)`

