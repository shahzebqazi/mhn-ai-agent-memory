#!/usr/bin/env python3
"""
A/B Benchmark: Baseline MHN vs Repulsive MHN

Tests three metrics:
  1. Capacity under correlated patterns
  2. Convergence speed (steps to stable state)
  3. Retrieval accuracy under noise

All tests use dim=256, fixed seeds, and report factual results.
"""

import sys
import os
import numpy as np
from typing import List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hopfield_memory.network import ModernHopfieldNetwork
from hopfield_memory.repulsive import RepulsiveMHN


SEED = 42
DIM = 256
BETA_POS = 8.0
BETA_NEG = 4.0
CLAMP_R = 1.5


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def make_clustered_patterns(
    n_clusters: int, per_cluster: int, dim: int, noise_scale: float, rng
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Generate correlated patterns clustered around random centers.

    Returns (patterns, cluster_midpoints_for_negatives).
    """
    centers = [normalize(rng.standard_normal(dim)) for _ in range(n_clusters)]
    patterns = []
    for c in centers:
        for _ in range(per_cluster):
            noisy = c + rng.standard_normal(dim) * noise_scale
            patterns.append(normalize(noisy))

    midpoints = []
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            mid = normalize(centers[i] + centers[j])
            midpoints.append(mid)

    return patterns, midpoints


def recall_accuracy(net, patterns: List[np.ndarray]) -> float:
    correct = 0
    for i, p in enumerate(patterns):
        _, weights = net.retrieve(p, num_steps=1)
        if np.argmax(weights) == i:
            correct += 1
    return correct / len(patterns)


def test_capacity(rng):
    """Test 1: Capacity under correlated patterns."""
    print("=" * 65)
    print("1. CAPACITY UNDER CORRELATION")
    print("=" * 65)
    print(f"   {'N':>5}  {'baseline':>10}  {'repulsive':>10}  {'delta':>8}")
    print("   " + "-" * 40)

    results = []

    for n_clusters in [5, 10, 20, 40]:
        per_cluster = 2
        n_total = n_clusters * per_cluster
        patterns, midpoints = make_clustered_patterns(
            n_clusters, per_cluster, DIM, noise_scale=0.3, rng=rng
        )

        baseline = ModernHopfieldNetwork(dim=DIM, beta=BETA_POS, adaptive_beta=False)
        for p in patterns:
            baseline.store(p)
        acc_base = recall_accuracy(baseline, patterns)

        repulsive = RepulsiveMHN(
            dim=DIM, beta=BETA_POS, beta_neg=BETA_NEG,
            adaptive_beta=False, clamp_radius=CLAMP_R,
        )
        for p in patterns:
            repulsive.store(p)
        for m in midpoints[:n_total]:
            repulsive.store_negative(m)
        acc_rep = recall_accuracy(repulsive, patterns)

        delta = acc_rep - acc_base
        results.append((n_total, acc_base, acc_rep, delta))
        print(f"   N={n_total:>3}  {acc_base:>9.1%}  {acc_rep:>9.1%}  {delta:>+7.1%}")

    print()
    return results


def test_convergence(rng):
    """Test 2: Convergence speed."""
    print("=" * 65)
    print("2. CONVERGENCE SPEED")
    print("=" * 65)

    n_patterns = 20
    n_negatives = 10
    n_trials = 50
    max_steps = 100
    eps = 1e-6

    patterns = [normalize(rng.standard_normal(DIM)) for _ in range(n_patterns)]
    negatives = [normalize(rng.standard_normal(DIM)) for _ in range(n_negatives)]

    baseline = ModernHopfieldNetwork(dim=DIM, beta=BETA_POS, adaptive_beta=False)
    for p in patterns:
        baseline.store(p)

    repulsive = RepulsiveMHN(
        dim=DIM, beta=BETA_POS, beta_neg=BETA_NEG,
        adaptive_beta=False, clamp_radius=CLAMP_R,
    )
    for p in patterns:
        repulsive.store(p)
    for n in negatives:
        repulsive.store_negative(n)

    def count_steps(net, query):
        xi = query.copy()
        for step in range(1, max_steps + 1):
            xi_new, _ = net.retrieve(xi, num_steps=1)
            if np.linalg.norm(xi_new - xi) < eps:
                return step
            xi = xi_new
        return max_steps

    base_steps = []
    rep_steps = []

    for _ in range(n_trials):
        q = normalize(rng.standard_normal(DIM))
        base_steps.append(count_steps(baseline, q))
        rep_steps.append(count_steps(repulsive, q))

    med_base = np.median(base_steps)
    med_rep = np.median(rep_steps)
    speedup = med_base / med_rep if med_rep > 0 else float('inf')

    print(f"   Trials:    {n_trials}")
    print(f"   Baseline:  median={med_base:.1f} steps  (mean={np.mean(base_steps):.1f})")
    print(f"   Repulsive: median={med_rep:.1f} steps  (mean={np.mean(rep_steps):.1f})")
    print(f"   Speedup:   {speedup:.2f}x")
    print()

    return med_base, med_rep, speedup


def test_noisy_retrieval(rng):
    """Test 3: Retrieval accuracy under noise."""
    print("=" * 65)
    print("3. RETRIEVAL ACCURACY UNDER NOISE")
    print("=" * 65)

    n_patterns = 30
    n_samples_per_snr = 20
    snr_levels = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]

    patterns = [normalize(rng.standard_normal(DIM)) for _ in range(n_patterns)]

    nn_midpoints = []
    for i in range(n_patterns):
        best_j = -1
        best_sim = -1.0
        for j in range(n_patterns):
            if i == j:
                continue
            sim = np.dot(patterns[i], patterns[j])
            if sim > best_sim:
                best_sim = sim
                best_j = j
        mid = normalize(patterns[i] + patterns[best_j])
        nn_midpoints.append(mid)

    baseline = ModernHopfieldNetwork(dim=DIM, beta=BETA_POS, adaptive_beta=False)
    for p in patterns:
        baseline.store(p)

    repulsive = RepulsiveMHN(
        dim=DIM, beta=BETA_POS, beta_neg=BETA_NEG,
        adaptive_beta=False, clamp_radius=CLAMP_R,
    )
    for p in patterns:
        repulsive.store(p)
    for m in nn_midpoints[:15]:
        repulsive.store_negative(m)

    print(f"   {'SNR':>6}  {'baseline':>10}  {'repulsive':>10}  {'delta':>8}")
    print("   " + "-" * 40)

    results = []

    for snr in snr_levels:
        base_correct = 0
        rep_correct = 0
        total = 0

        for i, p in enumerate(patterns):
            for _ in range(n_samples_per_snr):
                noise = rng.standard_normal(DIM) * snr
                noisy = normalize(p + noise)

                _, bw = baseline.retrieve(noisy, num_steps=1)
                if np.argmax(bw) == i:
                    base_correct += 1

                _, rw = repulsive.retrieve(noisy, num_steps=1)
                if np.argmax(rw) == i:
                    rep_correct += 1

                total += 1

        acc_base = base_correct / total
        acc_rep = rep_correct / total
        delta = acc_rep - acc_base
        results.append((snr, acc_base, acc_rep, delta))
        print(f"   {snr:>5.2f}  {acc_base:>9.1%}  {acc_rep:>9.1%}  {delta:>+7.1%}")

    print()
    return results


def energy_differential_sample(rng):
    """Log energy components for a sample query."""
    print("=" * 65)
    print("ENERGY DIFFERENTIALS (sample query)")
    print("=" * 65)

    patterns = [normalize(rng.standard_normal(DIM)) for _ in range(5)]
    negatives = [normalize(rng.standard_normal(DIM)) for _ in range(3)]

    net = RepulsiveMHN(
        dim=DIM, beta=BETA_POS, beta_neg=BETA_NEG,
        adaptive_beta=False, clamp_radius=CLAMP_R,
    )
    for p in patterns:
        net.store(p)
    for n in negatives:
        net.store_negative(n)

    query = normalize(rng.standard_normal(DIM))
    ec_before = net.energy_components(query)
    retrieved, _ = net.retrieve(query, num_steps=3)
    ec_after = net.energy_components(retrieved)

    print(f"   Before update:")
    print(f"     E_pos={ec_before.positive:.4f}  E_neg={ec_before.negative:.4f}  "
          f"E_quad={ec_before.quadratic:.4f}  E_total={ec_before.total:.4f}")
    print(f"   After 3 update steps:")
    print(f"     E_pos={ec_after.positive:.4f}  E_neg={ec_after.negative:.4f}  "
          f"E_quad={ec_after.quadratic:.4f}  E_total={ec_after.total:.4f}")
    print(f"   Delta: {ec_after.total - ec_before.total:+.4f}")
    print()


def main():
    print()
    print("BENCHMARK: Baseline MHN vs Repulsive MHN")
    print(f"dim={DIM}  beta+={BETA_POS}  beta-={BETA_NEG}  clamp_R={CLAMP_R}")
    print(f"seed={SEED}")
    print()

    rng = np.random.default_rng(SEED)

    cap_results = test_capacity(rng)
    conv_base, conv_rep, speedup = test_convergence(rng)
    noise_results = test_noisy_retrieval(rng)
    energy_differential_sample(rng)

    print("=" * 65)
    print("CONCLUSION")
    print("=" * 65)

    cap_deltas = [d for _, _, _, d in cap_results]
    avg_cap_delta = np.mean(cap_deltas)

    noise_deltas = [d for _, _, _, d in noise_results]
    avg_noise_delta = np.mean(noise_deltas)

    print(f"   Capacity:     avg delta = {avg_cap_delta:+.1%}")
    print(f"   Convergence:  {speedup:.2f}x {'faster' if speedup > 1 else 'slower'}")
    print(f"   Noise:        avg delta = {avg_noise_delta:+.1%}")
    print()

    wins = 0
    if avg_cap_delta > 0.01:
        print("   [+] Repulsive improves capacity under correlation")
        wins += 1
    elif avg_cap_delta < -0.01:
        print("   [-] Repulsive hurts capacity under correlation")
    else:
        print("   [=] No significant capacity difference")

    if speedup > 1.05:
        print("   [+] Repulsive converges faster")
        wins += 1
    elif speedup < 0.95:
        print("   [-] Repulsive converges slower")
    else:
        print("   [=] No significant convergence difference")

    if avg_noise_delta > 0.01:
        print("   [+] Repulsive improves noisy retrieval")
        wins += 1
    elif avg_noise_delta < -0.01:
        print("   [-] Repulsive hurts noisy retrieval")
    else:
        print("   [=] No significant noise robustness difference")

    print()
    if wins >= 2:
        print("   VERDICT: Repulsive attention shows clear improvement on multiple metrics.")
        print("   Recommend integration into main package.")
    elif wins == 1:
        print("   VERDICT: Mixed results. Repulsive helps on one metric but not others.")
        print("   Consider as optional feature only.")
    else:
        print("   VERDICT: No clear improvement. Do not integrate.")
        print("   Keep benchmark for reference; document as negative result.")

    print()


if __name__ == "__main__":
    main()
