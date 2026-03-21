#!/usr/bin/env python3
"""
A/B Benchmark v2: Baseline MHN vs Repulsive MHN

v1 used dim=256 where random patterns are nearly orthogonal and the
baseline never fails. This version tests three regimes where the
baseline SHOULD fail, giving the repulsive term something to fix:

  1. Low dimension relative to pattern count (d < 2N)
  2. Highly structured/correlated patterns with known confusable pairs
  3. Negative patterns placed at empirically observed failure points
     (where the baseline actually retrieves wrong answers)
"""

import sys
import os
import numpy as np
from typing import List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hopfield_memory.network import ModernHopfieldNetwork
from hopfield_memory.repulsive import RepulsiveMHN


SEED = 42
BETA_POS = 8.0
BETA_NEG = 6.0
CLAMP_R = 1.5


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def recall_accuracy(net, patterns: List[np.ndarray], num_steps=1) -> float:
    correct = 0
    for i, p in enumerate(patterns):
        _, weights = net.retrieve(p, num_steps=num_steps)
        if np.argmax(weights) == i:
            correct += 1
    return correct / len(patterns)


def find_baseline_failures(baseline, patterns):
    """Run baseline and find which queries retrieve the wrong pattern.

    Returns list of (query_index, retrieved_state) for failures.
    These retrieved states are empirical spurious attractors.
    """
    failures = []
    for i, p in enumerate(patterns):
        retrieved, weights = baseline.retrieve(p, num_steps=1)
        if np.argmax(weights) != i:
            failures.append((i, normalize(retrieved)))
    return failures


def make_confusable_patterns(n: int, dim: int, similarity: float, rng):
    """Generate n patterns where consecutive pairs have high cosine similarity.

    Pattern 2k and 2k+1 share a cosine similarity of approximately `similarity`.
    This creates known confusable pairs.
    """
    patterns = []
    for k in range(n // 2):
        base = normalize(rng.standard_normal(dim))
        perturb = rng.standard_normal(dim)
        perturb -= np.dot(perturb, base) * base
        perturb = normalize(perturb)

        blend = similarity
        p1 = normalize(base)
        p2 = normalize(blend * base + np.sqrt(1 - blend**2) * perturb)
        patterns.append(p1)
        patterns.append(p2)

    if n % 2 == 1:
        patterns.append(normalize(rng.standard_normal(dim)))

    return patterns


# ─────────────────────────────────────────────────────────────────
# TEST 1: Low dimension relative to pattern count
# ─────────────────────────────────────────────────────────────────

def test_low_dimension(rng):
    """Stress the baseline by using d < 2N."""
    print("=" * 70)
    print("1. LOW DIMENSION (d < 2N) -- where metastable blends form")
    print("=" * 70)
    print(f"   {'dim':>5} {'N':>5}  {'d/N':>5}  {'baseline':>10}  {'repulsive':>10}  {'delta':>8}")
    print("   " + "-" * 55)

    results = []

    for dim, n_patterns in [(16, 12), (16, 16), (32, 24), (32, 32), (32, 48), (64, 48), (64, 64), (64, 96)]:
        patterns = [normalize(rng.standard_normal(dim)) for _ in range(n_patterns)]

        baseline = ModernHopfieldNetwork(dim=dim, beta=BETA_POS, adaptive_beta=False)
        for p in patterns:
            baseline.store(p)

        acc_base = recall_accuracy(baseline, patterns)

        failures = find_baseline_failures(baseline, patterns)

        repulsive = RepulsiveMHN(
            dim=dim, beta=BETA_POS, beta_neg=BETA_NEG,
            adaptive_beta=False, clamp_radius=CLAMP_R,
        )
        for p in patterns:
            repulsive.store(p)

        for _, spurious_state in failures:
            repulsive.store_negative(spurious_state)

        if not failures:
            for i in range(min(n_patterns, n_patterns // 3)):
                j = (i + 1) % n_patterns
                mid = normalize(patterns[i] + patterns[j])
                repulsive.store_negative(mid)

        acc_rep = recall_accuracy(repulsive, patterns)
        delta = acc_rep - acc_base
        ratio = dim / n_patterns

        results.append((dim, n_patterns, ratio, acc_base, acc_rep, delta))
        print(f"   {dim:>5} {n_patterns:>5}  {ratio:>5.2f}  {acc_base:>9.1%}  {acc_rep:>9.1%}  {delta:>+7.1%}")

    print()
    return results


# ─────────────────────────────────────────────────────────────────
# TEST 2: Highly structured patterns with known confusable pairs
# ─────────────────────────────────────────────────────────────────

def test_confusable_pairs(rng):
    """Patterns are designed to be confusable: consecutive pairs have high cosine similarity."""
    print("=" * 70)
    print("2. CONFUSABLE PAIRS -- structured patterns with known overlap")
    print("=" * 70)
    print(f"   {'sim':>5} {'N':>5} {'dim':>5}  {'baseline':>10}  {'repulsive':>10}  {'delta':>8}")
    print("   " + "-" * 55)

    results = []
    dim = 64

    for similarity in [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]:
        n_patterns = 20
        patterns = make_confusable_patterns(n_patterns, dim, similarity, rng)

        baseline = ModernHopfieldNetwork(dim=dim, beta=BETA_POS, adaptive_beta=False)
        for p in patterns:
            baseline.store(p)
        acc_base = recall_accuracy(baseline, patterns)

        repulsive = RepulsiveMHN(
            dim=dim, beta=BETA_POS, beta_neg=BETA_NEG,
            adaptive_beta=False, clamp_radius=CLAMP_R,
        )
        for p in patterns:
            repulsive.store(p)

        for k in range(n_patterns // 2):
            mid = normalize(patterns[2*k] + patterns[2*k + 1])
            repulsive.store_negative(mid)

        acc_rep = recall_accuracy(repulsive, patterns)
        delta = acc_rep - acc_base

        results.append((similarity, n_patterns, dim, acc_base, acc_rep, delta))
        print(f"   {similarity:>5.2f} {n_patterns:>5} {dim:>5}  {acc_base:>9.1%}  {acc_rep:>9.1%}  {delta:>+7.1%}")

    print()
    return results


# ─────────────────────────────────────────────────────────────────
# TEST 3: Empirically placed negatives (at actual failure points)
# ─────────────────────────────────────────────────────────────────

def test_empirical_negatives(rng):
    """Find where baseline fails, place negatives there, re-test."""
    print("=" * 70)
    print("3. EMPIRICAL NEGATIVES -- placed at baseline failure points")
    print("=" * 70)

    results = []

    for dim, n_patterns in [(32, 30), (48, 40), (64, 50)]:
        patterns = make_confusable_patterns(n_patterns, dim, similarity=0.8, rng=rng)

        baseline = ModernHopfieldNetwork(dim=dim, beta=BETA_POS, adaptive_beta=False)
        for p in patterns:
            baseline.store(p)

        acc_base = recall_accuracy(baseline, patterns)
        failures = find_baseline_failures(baseline, patterns)

        repulsive = RepulsiveMHN(
            dim=dim, beta=BETA_POS, beta_neg=BETA_NEG,
            adaptive_beta=False, clamp_radius=CLAMP_R,
        )
        for p in patterns:
            repulsive.store(p)

        for _, spurious_state in failures:
            repulsive.store_negative(spurious_state)

        for k in range(n_patterns // 2):
            mid = normalize(patterns[2*k] + patterns[2*k + 1])
            repulsive.store_negative(mid)

        acc_rep = recall_accuracy(repulsive, patterns)
        delta = acc_rep - acc_base

        results.append((dim, n_patterns, len(failures), acc_base, acc_rep, delta))
        print(f"   dim={dim:>3} N={n_patterns:>3}  failures={len(failures):>3}  "
              f"baseline={acc_base:>6.1%}  repulsive={acc_rep:>6.1%}  delta={delta:>+6.1%}")

    print()
    return results


# ─────────────────────────────────────────────────────────────────
# TEST 4: Noisy retrieval under stress conditions
# ─────────────────────────────────────────────────────────────────

def test_noisy_retrieval_stressed(rng):
    """Noisy queries in low-dim with confusable patterns."""
    print("=" * 70)
    print("4. NOISY RETRIEVAL -- low dim, confusable patterns")
    print("=" * 70)

    dim = 48
    n_patterns = 30
    n_samples = 20
    snr_levels = [0.05, 0.1, 0.2, 0.3, 0.5]

    patterns = make_confusable_patterns(n_patterns, dim, similarity=0.75, rng=rng)

    baseline = ModernHopfieldNetwork(dim=dim, beta=BETA_POS, adaptive_beta=False)
    for p in patterns:
        baseline.store(p)

    failures = find_baseline_failures(baseline, patterns)

    repulsive = RepulsiveMHN(
        dim=dim, beta=BETA_POS, beta_neg=BETA_NEG,
        adaptive_beta=False, clamp_radius=CLAMP_R,
    )
    for p in patterns:
        repulsive.store(p)
    for _, spurious in failures:
        repulsive.store_negative(spurious)
    for k in range(n_patterns // 2):
        mid = normalize(patterns[2*k] + patterns[2*k + 1])
        repulsive.store_negative(mid)

    print(f"   dim={dim} N={n_patterns} confusable_pairs={n_patterns//2} negatives={repulsive.num_negative_patterns}")
    print(f"   {'SNR':>6}  {'baseline':>10}  {'repulsive':>10}  {'delta':>8}")
    print("   " + "-" * 40)

    results = []
    for snr in snr_levels:
        base_correct = 0
        rep_correct = 0
        total = 0

        for i, p in enumerate(patterns):
            for _ in range(n_samples):
                noise = rng.standard_normal(dim) * snr
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


# ─────────────────────────────────────────────────────────────────
# TEST 5: Convergence in stressed regime
# ─────────────────────────────────────────────────────────────────

def test_convergence_stressed(rng):
    """Convergence speed in low-dim stressed regime."""
    print("=" * 70)
    print("5. CONVERGENCE -- low dim stressed regime")
    print("=" * 70)

    dim = 48
    n_patterns = 30
    n_trials = 100
    max_steps = 200
    eps = 1e-6

    patterns = make_confusable_patterns(n_patterns, dim, similarity=0.75, rng=rng)

    baseline = ModernHopfieldNetwork(dim=dim, beta=BETA_POS, adaptive_beta=False)
    for p in patterns:
        baseline.store(p)

    failures = find_baseline_failures(baseline, patterns)

    repulsive = RepulsiveMHN(
        dim=dim, beta=BETA_POS, beta_neg=BETA_NEG,
        adaptive_beta=False, clamp_radius=CLAMP_R,
    )
    for p in patterns:
        repulsive.store(p)
    for _, spurious in failures:
        repulsive.store_negative(spurious)
    for k in range(n_patterns // 2):
        mid = normalize(patterns[2*k] + patterns[2*k + 1])
        repulsive.store_negative(mid)

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
        q = normalize(rng.standard_normal(dim))
        base_steps.append(count_steps(baseline, q))
        rep_steps.append(count_steps(repulsive, q))

    med_base = np.median(base_steps)
    med_rep = np.median(rep_steps)
    speedup = med_base / med_rep if med_rep > 0 else float('inf')

    print(f"   dim={dim} N={n_patterns} trials={n_trials}")
    print(f"   Baseline:  median={med_base:.1f}  mean={np.mean(base_steps):.1f}")
    print(f"   Repulsive: median={med_rep:.1f}  mean={np.mean(rep_steps):.1f}")
    print(f"   Speedup:   {speedup:.2f}x")
    print()

    return med_base, med_rep, speedup


def energy_sample(rng):
    """Energy component logging."""
    print("=" * 70)
    print("ENERGY DIFFERENTIALS (sample, dim=48)")
    print("=" * 70)

    dim = 48
    patterns = make_confusable_patterns(10, dim, similarity=0.8, rng=rng)

    baseline_net = ModernHopfieldNetwork(dim=dim, beta=BETA_POS, adaptive_beta=False)
    for p in patterns:
        baseline_net.store(p)

    failures = find_baseline_failures(baseline_net, patterns)

    net = RepulsiveMHN(dim=dim, beta=BETA_POS, beta_neg=BETA_NEG, adaptive_beta=False, clamp_radius=CLAMP_R)
    for p in patterns:
        net.store(p)
    for _, sp in failures:
        net.store_negative(sp)
    for k in range(len(patterns) // 2):
        net.store_negative(normalize(patterns[2*k] + patterns[2*k+1]))

    query = normalize(rng.standard_normal(dim))
    ec_before = net.energy_components(query)
    retrieved, _ = net.retrieve(query, num_steps=5)
    ec_after = net.energy_components(retrieved)

    print(f"   Before:  E_pos={ec_before.positive:+.4f}  E_neg={ec_before.negative:+.4f}  "
          f"E_quad={ec_before.quadratic:.4f}  total={ec_before.total:+.4f}")
    print(f"   After:   E_pos={ec_after.positive:+.4f}  E_neg={ec_after.negative:+.4f}  "
          f"E_quad={ec_after.quadratic:.4f}  total={ec_after.total:+.4f}")
    print(f"   Delta:   {ec_after.total - ec_before.total:+.4f}")
    print()


def main():
    print()
    print("=" * 70)
    print("BENCHMARK v2: Baseline MHN vs Repulsive MHN")
    print("Testing under stressed conditions where baseline should fail")
    print(f"beta+={BETA_POS}  beta-={BETA_NEG}  clamp_R={CLAMP_R}  seed={SEED}")
    print("=" * 70)
    print()

    rng = np.random.default_rng(SEED)

    r1 = test_low_dimension(rng)
    r2 = test_confusable_pairs(rng)
    r3 = test_empirical_negatives(rng)
    r4 = test_noisy_retrieval_stressed(rng)
    conv_base, conv_rep, speedup = test_convergence_stressed(rng)
    energy_sample(rng)

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    low_dim_deltas = [d for _, _, _, _, _, d in r1]
    confuse_deltas = [d for _, _, _, _, _, d in r2]
    empirical_deltas = [d for _, _, _, _, _, d in r3]
    noise_deltas = [d for _, _, _, d in r4]

    metrics = {
        "Low-dim capacity": np.mean(low_dim_deltas),
        "Confusable pairs": np.mean(confuse_deltas),
        "Empirical negatives": np.mean(empirical_deltas),
        "Noisy retrieval": np.mean(noise_deltas),
    }

    wins = 0
    for name, delta in metrics.items():
        tag = "[+]" if delta > 0.01 else ("[-]" if delta < -0.01 else "[=]")
        if delta > 0.01:
            wins += 1
        print(f"   {tag} {name}: avg delta = {delta:+.1%}")

    conv_tag = "[+]" if speedup > 1.05 else ("[-]" if speedup < 0.95 else "[=]")
    if speedup > 1.05:
        wins += 1
    print(f"   {conv_tag} Convergence: {speedup:.2f}x")

    print()
    if wins >= 3:
        print("   VERDICT: Repulsive attention shows clear improvement under stress.")
        print("   Recommend integration as optional feature.")
    elif wins >= 1:
        print("   VERDICT: Mixed results. Repulsive helps in some regimes.")
        print("   Consider as situational opt-in.")
    else:
        print("   VERDICT: No improvement even under stress. Negative result confirmed.")

    print()


if __name__ == "__main__":
    main()
