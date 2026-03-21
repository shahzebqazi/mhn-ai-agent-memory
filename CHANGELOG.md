# Changelog

## 0.3.0 (2026-03-21)

### Added
- Repulsive attention (contrastive learning) via `RepulsiveMHN` subclass
- `HopfieldMemory(repulsive=True)` toggle for opt-in contrastive mode
- `store_negative(fact)` to mark patterns the network should avoid
- `diagnose(query)` method for agents to measure convergence and decide
  whether repulsive mode is beneficial for their workload
- Independent `beta_neg` and `clamp_radius` parameters
- `EnergyComponents` for energy term logging
- Benchmarks showing 17x convergence speedup under stressed conditions

### Changed
- `HopfieldMemory.save()`/`load()` now round-trip repulsive state
- Version bump to 0.3.0

## 0.2.0 (2026-03-21)

### Added
- Pluggable encoder interface with four implementations:
  RandomIndexEncoder, TFIDFEncoder, SentenceTransformerEncoder, OpenAIEncoder
- Adaptive beta for per-query inverse temperature tuning
- Contradiction detection via similarity + structural heuristics
- Multi-hop chained retrieval
- Tiered hot/cold storage (exact Hopfield + FAISS/numpy ANN)
- Configurable presets: small, medium, large, massive
- Capacity-aware warnings when patterns exceed 50% of dimension
- `query_with_confidence()` method
- Serialization with `save()` and `load()`

### Changed
- ModernHopfieldNetwork uses adaptive beta by default
- HopfieldMemory accepts pluggable Encoder via constructor

## 0.1.0 (2026-03-21)

### Added
- Initial implementation of Modern Hopfield Network (Ramsauer et al., 2021)
- HopfieldMemory with random-index text encoding
- Basic store/retrieve/query API
- Energy function computation
- JSON serialization
