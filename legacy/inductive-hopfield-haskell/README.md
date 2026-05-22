# Inductive Hopfield Networks in Haskell

**Goal:** Build classical Hopfield networks inductively from n=2, measuring
storage capacity and interference at every step, until the classical regime's
limitations are empirically obvious. Then pivot to modern Hopfield networks
and build toward a compact associative-memory API.

**Status:** Scaffold only; no executable code yet.

---

## Why Start at n=2

A Hopfield network with 2 neurons can store exactly 1 bipolar pattern (and its
complement). By implementing each step from n=2 upward and recording what changes
in the weight matrix, update rule, energy function, and storage capacity, the
inductive structure of the classical Hopfield equations becomes visible in the
code itself.

The point is not to reach large N quickly. It is to see the pattern clearly
enough to write a compact closed-form expression for an arbitrary-N classical
Hopfield network, then to demonstrate empirically why that closed form hits
a hard capacity ceiling around 0.138N stored patterns (Amit et al., 1985).

---

## Directory Layout

```
hopfield-networks/
├── README.md                           # This file (human-facing)
├── prompt.md                           # AI agent operating contract
├── index.html                          # Three.js energy-landscape visualization
├── inductive-hopfield.cabal            # Package definition
├── cabal.project                       # Cabal project config
├── app/
│   └── Main.hs                         # CLI entry point for experiments
├── src/
│   └── Hopfield/
│       ├── Classical/
│       │   ├── Types.hs                # Network, Pattern, WeightMatrix types
│       │   ├── Hebbian.hs              # Hebbian imprinting (W = sum xi*xi^T)
│       │   ├── Update.hs               # Asynchronous state update (sign rule)
│       │   ├── Energy.hs               # E = -0.5 * s^T * W * s
│       │   └── Capacity.hs             # Storage/recall measurement utilities
│       ├── Inductive/
│       │   └── Step.hs                 # n -> n+1 growth logic and bookkeeping
│       └── Modern/
│           └── Placeholder.hs          # Stub for post-pivot modern Hopfield work
├── test/
│   └── Hopfield/
│       ├── ClassicalSpec.hs            # Unit + property tests for classical ops
│       ├── InductiveSpec.hs            # Tests for inductive step consistency
│       └── CapacitySpec.hs             # Capacity and interference measurement tests
└── experiments/
    ├── README.md                       # How to read and record experiment results
    └── runs/                           # Per-N result logs (agents append here)
```

---

## What Counts as a Successful Inductive Step

For each move from n to n+1, the implementing agent must:

1. Extend (or generalize) the current implementation to handle the new network size.
2. Write or extend tests covering the new size.
3. Run recall experiments: store increasing numbers of random bipolar patterns,
   measure how many can be recalled without error.
4. Record: N, patterns attempted, patterns recalled, failure mode (none,
   interference, spurious attractor, oscillation).
5. Summarize what changed in the weight matrix / update / energy from the
   previous step.
6. Commit before moving to n+2.

---

## When to Stop Scaling Classical

The classical phase ends when agents observe all of the following:

- Storage capacity is empirically confirmed near the ~0.138N ceiling.
- Adding neurons buys incremental memory but no new structural insight.
- The code for each n+1 step is largely mechanical (same pattern, bigger matrix).
- Recall failures are dominated by cross-talk interference, not implementation bugs.

At that point, freeze the classical baseline and begin the modern Hopfield branch
under `src/Hopfield/Modern/`.

---

## Modern Hopfield Phase

After the classical pivot, the goal shifts:

- Implement the continuous Hopfield update rule: x_new = softmax(beta * X^T * xi).
- Build an explicit storage/retrieval API (not markdown scratch files).
- Measure exponential storage capacity empirically.
- Test whether AI agents can use the Haskell Hopfield API as associative memory
  for small structured facts, and record where it works and where it fails.

The first modern milestone is a minimal working API that stores toy patterns and
retrieves them reproducibly under test.

---

## References

- Hopfield, J.J. (1982). Neural networks and physical systems with emergent
  collective computational abilities. PNAS 79(8): 2554-2558.
- Amit, D.J., Gutfreund, H., Sompolinsky, H. (1985). Storing infinite numbers
  of patterns in a spin-glass model of neural networks. PRL 55(14): 1530-1533.
- Ramsauer, H. et al. (2021). Hopfield Networks is All You Need. ICLR.
- Krotov, D. & Hopfield, J.J. (2016). Dense Associative Memory for Pattern
  Recognition. NeurIPS.
