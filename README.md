<div align="center">

# MHN AI Agent Memory

### Associative Memory for AI Agents Using Modern Hopfield Networks

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-43%20passing-brightgreen.svg)](#development)
[![arXiv](https://img.shields.io/badge/arXiv-2008.02217-b31b1b.svg)](https://arxiv.org/abs/2008.02217)

**Give your AI agents real memory.**
**Not a database with an LLM wrapper. Actual associative memory backed by mathematics.**

[Install](#install) &#8226; [Quick Start](#30-second-example) &#8226; [Features](#features-at-a-glance) &#8226; [How It Works](#how-it-works-plain-english) &#8226; [API Reference](#project-structure)

</div>

---

<div align="center">

```
          ┌─────────────────────────────────────────────┐
          │                                             │
          │   xi_new = X @ softmax( beta * X^T @ xi )  │
          │                                             │
          │   One equation. One matrix multiply.        │
          │   Deterministic. Microseconds. Free.        │
          │                                             │
          └─────────────────────────────────────────────┘
```

</div>

---

## What is this?

When an AI agent needs to "remember" something today, the standard approach is:

1. Store text in a database
2. Call an LLM to reason about it (costs money, takes seconds, non-deterministic)
3. Hope the LLM doesn't hallucinate about what it stored

This library replaces that pipeline with the [Modern Hopfield Network](https://arxiv.org/abs/2008.02217) update rule -- the same mathematical structure as transformer attention, but exposed as an explicit, controllable memory.

| | Traditional (LLM + DB) | This Library |
|---|---|---|
| **Retrieval** | LLM API call | One matrix multiply |
| **Latency** | Seconds | Microseconds |
| **Cost** | Per-token | Zero after storage |
| **Determinism** | Non-deterministic | Deterministic |
| **Capacity** | Depends on embedding quality | Exponential in dimension (proven) |

**Store a fact.** It becomes a pattern in an energy landscape.
**Query with a partial cue.** The network relaxes to the nearest stored pattern.
**Get the answer.** Confidence score included.

---

## Install

```bash
pip install mhn-ai-agent-memory
```

Optional extras for better quality and scale:

```bash
pip install mhn-ai-agent-memory[semantic]   # sentence-transformers (~80MB model)
pip install mhn-ai-agent-memory[openai]     # OpenAI embedding API
pip install mhn-ai-agent-memory[scale]      # FAISS for million-scale storage
pip install mhn-ai-agent-memory[all]        # everything
```

---

## 30-Second Example

```python
from hopfield_memory import HopfieldMemory

mem = HopfieldMemory()

mem.store("Alice is a mathematician who studies topology")
mem.store("Bob is a painter who works with oil on canvas")
mem.store("Carol is a physicist researching quantum entanglement")

fact, confidence = mem.query_with_confidence("topology math")
print(fact)        # "Alice is a mathematician who studies topology"
print(confidence)  # 0.9999
```

No API keys. No database. No config files. The memory is a numpy array.

---

## Features at a Glance

### Pluggable Encoders

Swap how text becomes vectors. Start simple, upgrade when you need to.

```python
from hopfield_memory import HopfieldMemory, SentenceTransformerEncoder

mem = HopfieldMemory(encoder=SentenceTransformerEncoder())
```

| Encoder | Quality | Dependencies |
|---|---|---|
| `RandomIndexEncoder` | Basic (exact word match) | numpy only |
| `TFIDFEncoder` | Medium | scikit-learn |
| `SentenceTransformerEncoder` | High | sentence-transformers |
| `OpenAIEncoder` | Highest | openai SDK + API key |

### Contradiction Detection

Catch conflicting facts before they corrupt your memory.

```python
from hopfield_memory import check_and_store, ContradictionDetector

detector = ContradictionDetector()
idx, conflict = check_and_store(mem, "The capital of France is Lyon", detector=detector)
if conflict.has_conflict:
    print(conflict.explanation)
```

### Multi-Hop Retrieval

Chain queries to follow reasoning across related facts.

```python
from hopfield_memory import chain_query

mem.store("Alice lives in France")
mem.store("The capital of France is Paris")

chain_query(mem, "capital of Alice's country", max_hops=3)
# -> ["Alice lives in France", "The capital of France is Paris"]
```

### Scale from 10 to 10 Million Facts

```python
from hopfield_memory import small_memory, large_memory, massive_memory

mem = small_memory()    # ~100 facts, for tools and plugins
mem = large_memory()    # ~100k facts, tiered hot/cold storage
mem = massive_memory()  # millions, FAISS-backed cold store
```

The tiered system keeps your most-accessed memories in an exact Hopfield network (microsecond retrieval) and archives the rest to an approximate nearest-neighbor index on disk.

### Repulsive Attention (Opt-In)

Up to 17x faster convergence for multi-step retrieval by adding contrastive "hills" to the energy landscape.

```python
mem = HopfieldMemory(repulsive=True)
mem.store("Alice is a mathematician")
mem.store_negative("Known confusable pattern to avoid")

diag = mem.diagnose("topology math")
print(diag["recommendation"])  # agents decide at runtime
```

### "Nothing Matches" Detection

AI agents need to know when a query has no relevant memory -- not just pick the least-bad option. This library solves it with three independent signals combined via a sentinel pattern.

```python
from hopfield_memory import HopfieldMemory

mem = HopfieldMemory()
mem.store("The Eiffel Tower is in Paris")
mem.store("Mount Fuji is in Japan")

# Returns the fact when there's a match
result = mem.query_or_none("Eiffel Tower Paris")
print(result)  # "The Eiffel Tower is in Paris"

# Returns None when nothing matches
result = mem.query_or_none("basketball playoffs score")
print(result)  # None

# For more detail, inspect the match signals
mq = mem.match_quality("basketball playoffs score")
print(mq["max_similarity"])   # ~0.14 (low -- no real word overlap)
print(mq["is_match"])         # False
```

Under the hood, the network stores a zero-vector sentinel pattern. Three signals are combined:
- **max_similarity** -- raw dot product before softmax (the primary signal)
- **gap** -- attention weight separation between top patterns
- **sentinel_weight** -- how much attention goes to the "nothing" anchor

---

## How It Works (Plain English)

1. **You store a fact.** The text is encoded into a vector and added to a matrix of stored patterns. This matrix IS the memory.

2. **You query with a cue.** The network computes similarity between the cue and every stored pattern, then uses a softmax to sharply concentrate attention on the best match.

3. **You get a result.** The output is the stored pattern the network "attracted" to -- the nearest memory. The softmax weight is your confidence score.

> **Key insight:** This has the same mathematical structure as a single step of transformer attention (with tied keys and values). This library exposes that operation directly as a memory system, without wrapping it in an LLM.

---

## How It Compares

| | This Library | Honcho | Zep | MemGPT/Letta | Vector DB + LLM |
|---|---|---|---|---|---|
| **Architecture** | Hopfield energy landscape | LLM reasoning + DB | Embedding + temporal graph | LLM self-editing context | ANN index + LLM |
| **Retrieval latency** | ~10 us (numpy matmul) | Seconds (LLM call) | ~ms (vector search) | Seconds (LLM call) | ~ms (ANN) + seconds (LLM) |
| **Cost per query** | Zero | LLM token cost | Zero (self-hosted) | LLM token cost | LLM token cost |
| **Deterministic** | Yes | No | Partially | No | No |
| **"No match" detection** | Built-in (sentinel) | Via LLM judgment | No | Via LLM judgment | No |
| **Capacity theory** | Exponential in dim (proven) | Unbounded (DB) | Unbounded (DB) | Context window | Unbounded (DB) |
| **Dependencies** | numpy | Python + LLM API + DB | Python + DB | Python + LLM API | Python + vector DB + LLM API |
| **MCP server** | Included | Cursor/Claude plugins | No | No | Custom |
| **Best for** | Fast, deterministic agent memory | Personalized long-term user models | Session history | Autonomous context management | Semantic search over documents |

> This library is not a replacement for Honcho, Zep, or MemGPT -- they solve different problems.
> Use this when you need fast, deterministic, cost-free associative recall.
> Use them when you need LLM-powered reasoning about memory, user modeling, or unbounded storage.

---

## Limitations

This section exists because honest documentation matters more than marketing.

- **Default encoder is bag-of-words.** "dog" and "canine" get zero similarity without `[semantic]` extra.
- **Contradiction detection is heuristic.** Works best with simple factual statements.
- **Multi-hop is retrieval chaining, not logical inference.** It finds related facts, not derived conclusions.
- **Confidence is relative, not absolute.** Softmax always sums to 1, so `query_with_confidence()` always reports high confidence. Use `query_or_none()` or `has_match()` to detect non-matches.
- **Adaptive beta is a heuristic.** The convergence proof assumes fixed inverse temperature.
- **Exponential capacity has conditions.** Requires patterns with sufficient separation in high dimension.

---

## Project Structure

```
mhn-ai-agent-memory/
  pyproject.toml              # Package config
  llms.txt                    # AI agent discoverability
  CITATION.cff                # Academic citation metadata
  src/hopfield_memory/
    network.py                # ModernHopfieldNetwork (the math)
    memory.py                 # HopfieldMemory (the user API)
    repulsive.py              # RepulsiveMHN (contrastive attention)
    encoders.py               # 4 text encoders
    contradiction.py          # Conflict detection
    multihop.py               # Chained retrieval
    tiered.py                 # Hot/cold storage for scale
    presets.py                # small/medium/large/massive factories
  mcp-server/                 # MCP server for Cursor, Claude Code, etc.
  tests/                      # 43 tests
  examples/                   # Runnable demos
  benchmarks/                 # A/B: baseline vs repulsive
  docs/                       # GitHub Pages blog post
```

---

## Development

```bash
git clone https://github.com/shahzebqazi/mhn-ai-agent-memory.git
cd mhn-ai-agent-memory
pip install -e ".[dev]"
pytest
```

---

## References

- Ramsauer et al., [Hopfield Networks is All You Need](https://arxiv.org/abs/2008.02217), ICLR 2021
- Krotov & Hopfield, [Dense Associative Memory for Pattern Recognition](https://arxiv.org/abs/1606.01164), NeurIPS 2016
- Comparison baseline: [Honcho](https://github.com/plastic-labs/honcho) (LLM-augmented database approach)

---

<div align="center">

**MIT License** &#8226; Built by [@shahzebqazi](https://github.com/shahzebqazi)

</div>
