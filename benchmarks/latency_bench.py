#!/usr/bin/env python3
"""Micro-benchmark: single-step Hopfield retrieve latency (CPU, in-memory)."""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hopfield_memory.network import ModernHopfieldNetwork


def main() -> None:
    rng = np.random.default_rng(42)
    dim, n = 256, 50
    patterns = []
    for _ in range(n):
        v = rng.standard_normal(dim)
        patterns.append(v / (np.linalg.norm(v) + 1e-12))

    net = ModernHopfieldNetwork(dim=dim, beta=8.0, adaptive_beta=False)
    for p in patterns:
        net.store(p)

    query = patterns[0]
    for _ in range(100):
        net.retrieve(query, num_steps=1)

    runs = 1000
    t0 = time.perf_counter()
    for _ in range(runs):
        net.retrieve(query, num_steps=1)
    us_per_op = (time.perf_counter() - t0) / runs * 1e6

    print(f"dim={dim} N={n} patterns")
    print(f"single-step retrieve: {us_per_op:.1f} µs/op ({runs} runs after warmup)")


if __name__ == "__main__":
    main()
