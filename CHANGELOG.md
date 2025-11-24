# Changelog

All notable changes to Mímir Framework will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Git Flow workflow configuration
- CONTRIBUTING.md with development guidelines
- Dual licensing (GPL-2.0 + Commercial)
- COMMERCIAL_LICENSE.md with pricing tiers

## [1.0.0] - 2025-11-24

### Added
- Core framework with Model base class
- Tensor system with autograd support
- 3 optimizers: SGD, Adam, AdamW
- 5 LR decay strategies (Cosine, Step, Exponential, Linear, None)
- Complete Lua scripting API
  - `model.create()`, `model.build()`, `model.train()`, `model.infer()`
  - `model.save()`, `model.load()`
  - `tokenizer.*` operations
  - `dataset.load()`, `dataset.prepare_sequences()`
- BPE Tokenizer with text understanding
- Encoder with special embeddings (SEQ, MOD, MAG)
- SafeTensors format for model serialization
- CLI with `--script` and `--config` support
- Interactive Lua REPL mode
- SIMD AVX2 optimizations (matmul, GELU)
- OpenCL 3.0 GPU support
- OpenMP multi-threading
- Advanced RAM Manager with LRU cache and compression
- Visualizer with SFML
- HtopDisplay for RAM monitoring
- Complete documentation (80 KB)
  - README.md, QUICKSTART.md, ARCHITECTURE.md
  - API_LUA.md, API_CPP.md
  - INSTALLATION.md, ROADMAP.md, STATUS.md
- Example Lua scripts
  - example_training.lua
  - example_gpt.lua
  - example_encoder.lua, example_unet.lua, example_vit.lua
- Makefile with optimization flags
- Configuration system via JSON

### Technical Details
- Language: C++17
- Dependencies: OpenCL, SFML, Lua 5.3, nlohmann/json
- Platforms: Linux (primary)

### Performance
- SIMD AVX2 for vectorized operations
- OpenCL for GPU acceleration
- Multi-threading with OpenMP
- Memory optimization with uint16 quantization
- Lazy loading for datasets

### Architecture
- Model base class with virtual hooks
- Multimodal support via MagicTokens
- Atomic checkpoint saving
- DatasetMemoryManager singleton
- LuaContext for scripting state

---

## Version History

- **1.0.0** (2025-11-24) - Initial release

## Upgrade Guide

### From Pre-release to 1.0.0

This is the first stable release. No upgrade needed.

## Future Releases

See [ROADMAP.md](docs/ROADMAP.md) for planned features.

---

[Unreleased]: https://github.com/mimir-framework/mimir/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/mimir-framework/mimir/releases/tag/v1.0.0
