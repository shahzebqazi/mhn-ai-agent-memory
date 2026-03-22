# Changelog

## 1.0.0 (2026-03-21)

### Fixed
- **Save/load sentinel duplication**: `load()` no longer creates a duplicate sentinel
  pattern; `save()` now persists sentinel metadata with backward-compatible legacy
  inference (checks both `facts[0]==""` and `patterns[0]` all-zero)
- **Cross-process encoder determinism**: `RandomIndexEncoder` and `TFIDFEncoder`
  fallback use `hashlib.sha256` instead of Python's per-process-salted `hash()`
- **Energy/dynamics consistency**: `energy()` and `RepulsiveMHN.energy_components()`
  now use the state-dependent beta when `adaptive_beta` is enabled, matching
  `retrieve()` dynamics; accepts optional `beta` parameter for explicit control
- **Match predicate alignment**: `is_match` and `has_match` now incorporate all
  three documented signals (similarity, gap, sentinel weight) with consistent
  `>=` threshold
- **OpenAI encoder dimension**: Accepts optional `dim` override; uses known-model
  lookup instead of fragile substring heuristic

### Added
- 9 new tests (43 total): save/load roundtrip variants, encoder determinism,
  softmax weight properties, convex combination norm bound, pattern/fact count
  invariant after load
- MCP server (`mcp-server/`) for Cursor, Claude Code, and other MCP-compatible
  AI coding agents
- `llms.txt` and `llms-full.txt` for AI agent discoverability
- `CITATION.cff` for academic citation
- GitHub Pages blog post at `docs/`

### Changed
- Encoder ABC docstring clarifies empty text returns the zero vector
- `test_capacity_with_noisy_queries` explicitly uses `adaptive_beta=False`
  for theory-aligned capacity measurement
- Development Status classifier bumped to Beta
- Version bump to 1.0.0

## 0.4.0 (2026-03-21)

### Added
- **"No match" detection** -- the network can now tell agents when nothing in memory matches a query
- `query_or_none(question)` returns None instead of a forced answer when nothing matches
- `has_match(question)` boolean check for match existence
- `match_quality(question)` returns all three detection signals for agent decision-making:
  - `max_similarity` -- raw dot product before softmax (most reliable signal)
  - `gap` -- attention weight separation between top two patterns
  - `sentinel_weight` -- how much attention goes to the null sentinel
- Sentinel pattern (zero vector) automatically stored on init to anchor the "no match" signal
- Sentinel excluded from all query results and fact counts
- 7 new tests for match detection behavior

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
