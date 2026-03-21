# MHN AI Agent Memory

**Give your AI agents real memory. Not a database with an LLM wrapper. Actual associative memory backed by mathematics.**

This library implements [Modern Hopfield Networks](https://arxiv.org/abs/2008.02217) (Ramsauer et al., 2021) as a practical memory system for AI agents. Store facts, retrieve them by meaning, and get deterministic results in microseconds -- no API calls, no tokens burned, no database required.

---

## What is this?

When an AI agent needs to "remember" something today, the standard approach is:

1. Store text in a database
2. Call an LLM to reason about it (costs money, takes seconds, non-deterministic)
3. Hope the LLM doesn't hallucinate about what it stored

This library replaces that entire pipeline with one equation:

```
xi_new = X @ softmax(beta * X^T @ xi)
```

That is the Modern Hopfield update rule. It has the same structure as a single step of transformer attention with tied keys and values -- the same core operation inside every LLM -- but exposed as an explicit, controllable memory you can store facts in and query directly.

**Store a fact.** It becomes a pattern in an energy landscape.
**Query with a partial cue.** The network relaxes to the nearest stored pattern.
**Get the answer.** One matrix multiply. Deterministic. Microseconds. Free.

---

## Install

```bash
pip install mhn-ai-agent-memory
```

Want better text understanding? Add a semantic encoder:

```bash
pip install mhn-ai-agent-memory[semantic]   # sentence-transformers (~80MB model)
pip install mhn-ai-agent-memory[all]        # everything including FAISS for scale
```

---

## 30-Second Example

```python
from hopfield_memory import HopfieldMemory

# Create a memory
mem = HopfieldMemory()

# Store some facts
mem.store("Alice is a mathematician who studies topology")
mem.store("Bob is a painter who works with oil on canvas")
mem.store("Carol is a physicist researching quantum entanglement")

# Query with a partial cue
fact, confidence = mem.query_with_confidence("topology math")
print(fact)        # "Alice is a mathematician who studies topology"
print(confidence)  # 0.9999
```

That's it. No API keys. No database. No config files. The memory is a numpy array.

---

## Why not just use a vector database?

Vector databases (pgvector, Pinecone, Weaviate) store embeddings and retrieve by cosine similarity. That is half of what a Hopfield network does. The other half -- the softmax sharpening via inverse temperature beta -- is what gives you:

- **High confidence scores** instead of a flat ranked list
- **Content-addressable recall** where partial cues reconstruct the full pattern
- **Mathematically characterized capacity** -- exponential in dimension under the conditions of Ramsauer et al. (2021), not unconditional
- **Energy-based convergence** guarantees (the network provably settles)

A vector database gives you "these 10 results are somewhat similar." A Hopfield network gives you "this is the answer, confidence 0.9999."

---

## Features at a Glance

### Pluggable Encoders

Swap how text becomes vectors. Start simple, upgrade when you need to.

```python
from hopfield_memory import HopfieldMemory, SentenceTransformerEncoder

# High-quality semantic encoding ("dog" and "canine" will match)
mem = HopfieldMemory(encoder=SentenceTransformerEncoder())
```

| Encoder | Quality | Needs |
|---|---|---|
| RandomIndexEncoder | Basic (exact word match) | Nothing (default) |
| TFIDFEncoder | Medium | scikit-learn |
| SentenceTransformerEncoder | High | sentence-transformers |
| OpenAIEncoder | Highest | OpenAI API key |

### Contradiction Detection

Catch conflicting facts before they corrupt your memory.

```python
from hopfield_memory import check_and_store, ContradictionDetector

detector = ContradictionDetector()
idx, conflict = check_and_store(mem, "The capital of France is Lyon", detector=detector)
if conflict.has_conflict:
    print(conflict.explanation)  # Shows the conflicting fact
```

### Multi-Hop Reasoning

Chain queries to follow reasoning across related facts.

```python
from hopfield_memory import chain_query

mem.store("Alice lives in France")
mem.store("The capital of France is Paris")

chain_query(mem, "capital of Alice's country", max_hops=3)
# -> ["Alice lives in France", "The capital of France is Paris"]
```

### Scale from 10 to 10 Million Facts

Pick a preset that matches your use case.

```python
from hopfield_memory import small_memory, large_memory, massive_memory

mem = small_memory()    # ~100 facts, for tools and plugins
mem = large_memory()    # ~100k facts, tiered hot/cold storage
mem = massive_memory()  # millions, FAISS-backed cold store
```

The tiered system keeps your most-accessed memories in an exact Hopfield network (microsecond retrieval) and archives the rest to an approximate nearest-neighbor index on disk.

### Repulsive Attention (Opt-In)

For workloads that use multi-step retrieval, repulsive attention provides up to 17x faster convergence by adding "hills" to the energy landscape between attractors.

```python
from hopfield_memory import HopfieldMemory

mem = HopfieldMemory(repulsive=True)
mem.store("Alice is a mathematician")
mem.store("Bob is a painter")

# Mark a known confusable blend as something to avoid
mem.store_negative("Someone who paints mathematical diagrams")

# Agents can check if repulsive mode is helping their workload
diag = mem.diagnose("topology math")
print(diag["recommendation"])  # "fast convergence; repulsive mode not needed"
print(diag["steps"])           # number of iterations to converge
```

The `diagnose()` method lets agents decide at runtime whether repulsive mode is worth the overhead. If convergence is already fast, it recommends turning it off. If the network is slow to settle, it recommends keeping it on.

---

## How It Works (Plain English)

1. **You store a fact.** The text is encoded into a vector and added to a matrix of stored patterns. This matrix IS the memory -- there is no separate database.

2. **You query with a cue.** The cue is encoded into a vector. The network computes how similar this cue is to every stored pattern, then uses a softmax to sharply concentrate attention on the best match.

3. **You get a result.** The output is the stored pattern that the network "attracted" to -- the nearest memory. The softmax weight is your confidence score.

The key insight: this has the same mathematical structure as a single step of transformer attention (with tied keys and values). This library exposes that operation directly as a memory system, without wrapping it in an LLM.

---

## Limitations

- **Default encoder is bag-of-words.** Without `sentence-transformers` installed, the RandomIndexEncoder only matches exact words. "dog" and "canine" will get zero similarity. Install the `[semantic]` extra for real semantic understanding.
- **Contradiction detection is heuristic.** The structural check (first 4 tokens as "subject") is brittle for complex sentences. It works best with simple factual statements and a semantic encoder.
- **Multi-hop is retrieval chaining, not reasoning.** `chain_query` augments the query with retrieved text and re-queries. It does not perform logical inference -- it finds related facts by shifting the energy landscape.
- **Confidence is relative, not absolute.** Softmax must sum to 1, so the network always gives some pattern high confidence -- even for a completely irrelevant query. It tells you "which stored fact is the best match" not "whether any stored fact matches." Agents should compare confidence across queries, not threshold a single query's confidence.
- **Adaptive beta is not proven.** The convergence guarantee from Ramsauer et al. assumes fixed inverse temperature. Adaptive beta is a practical heuristic that improves confidence empirically but has no formal proof.
- **Exponential capacity has conditions.** The theoretical exponential storage bound requires patterns with sufficient separation in high dimension. Densely clustered or low-dimensional patterns will hit capacity limits sooner.

---

## Project Structure

```
mhn-ai-agent-memory/
  pyproject.toml              # Package config, install with pip
  src/hopfield_memory/        # The library
    network.py                # ModernHopfieldNetwork (the math)
    memory.py                 # HopfieldMemory (the user API)
    encoders.py               # 4 text encoders at different quality levels
    contradiction.py          # Conflict detection
    multihop.py               # Chained retrieval
    tiered.py                 # Hot/cold storage for scale
    presets.py                # small/medium/large/massive factories
  tests/                      # pytest suite (27 tests)
  examples/                   # Runnable demos
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

## License

MIT
