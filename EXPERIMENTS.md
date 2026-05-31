# Experiments and benchmarks

Reproducible numbers for **mhn-ai-agent-memory** (Modern Hopfield associative memory).  
Implements the update rule from Ramsauer et al., [Hopfield Networks is All You Need](https://arxiv.org/abs/2008.02217) (arXiv:2008.02217) — this repo is an **implementation and stress-test**, not a new paper.

**Environment:** macOS, Python 3.x, `numpy` only (core package). Re-run locally to verify.

---

## Test suite

```bash
cd /path/to/mhn-ai-agent-memory
python3 -m pytest tests/ -q
```

| Metric | Result (2026-05-30) |
|--------|---------------------|
| Tests | **43 passed** |
| Runtime | ~0.6 s |

---

## Retrieval latency (single-step, in-memory)

Script: inline one-off; equivalent to `ModernHopfieldNetwork.retrieve(..., num_steps=1)`.

| Setting | Result |
|---------|--------|
| dim=256, N=50 stored patterns | **~16 µs** per retrieve (1000-run median, warm cache) |

This is the raw Hopfield attention step on CPU — no LLM, no database round-trip.

---

## Benchmark v2: baseline vs repulsive MHN

**Run:**

```bash
python3 benchmarks/repulsive_benchmark_v2.py
```

**Design:** Stress regimes where a plain Modern Hopfield network is expected to struggle (low dimension vs pattern count, confusable pairs, empirical negative patterns from baseline failures). Compares `ModernHopfieldNetwork` vs `RepulsiveMHN`.

**Parameters:** `seed=42`, `beta+=8.0`, `beta-=6.0`, `clamp_R=1.5`

### Results snapshot (2026-05-30)

| Regime | Baseline vs repulsive |
|--------|------------------------|
| Low-dim capacity | No avg gain in this run |
| Confusable pairs | No avg gain in this run |
| Empirical negatives | No avg gain in this run |
| Noisy retrieval (SNR sweep) | No avg gain in this run |
| **Convergence (dim=48, N=30, 100 trials)** | **17.1×** fewer median steps (137 → 8) |

**Verdict from script:** Mixed results — repulsive helps in some regimes; situational opt-in, not a universal win.

Full console output is the source of truth; paste your machine’s `SUMMARY` block when citing.

---

## What this is / is not

| Is | Is not |
|----|--------|
| Tested Python library (43 tests) | Peer-reviewed publication by author |
| Reproducible benchmarks in `benchmarks/` | AV perception or LiDAR stack |
| Toy / research memory for agents | Production vector DB replacement |

---

## Citation

If you use this code, cite the **original Hopfield-is-All-You-Need paper** (see `CITATION.cff`) and link this repository for the implementation.
