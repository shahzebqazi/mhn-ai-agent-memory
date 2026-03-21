"""Unit tests for RepulsiveMHN."""

import numpy as np
from hopfield_memory.repulsive import RepulsiveMHN, EnergyComponents


def _random_patterns(n, dim, rng):
    patterns = []
    for _ in range(n):
        p = rng.standard_normal(dim)
        p /= np.linalg.norm(p)
        patterns.append(p)
    return patterns


class TestRepulsiveStoreRetrieve:
    def test_retrieves_correct_positive_under_noise(self):
        """Query with noisy version of a stored positive, verify correct recall."""
        dim = 128
        rng = np.random.default_rng(42)
        net = RepulsiveMHN(dim=dim, beta=8.0, beta_neg=4.0, clamp_radius=1.0)

        positives = _random_patterns(5, dim, rng)
        negatives = _random_patterns(3, dim, rng)

        for p in positives:
            net.store(p)
        for n in negatives:
            net.store_negative(n)

        for i, p in enumerate(positives):
            noise = rng.standard_normal(dim) * 0.2
            noisy = p + noise
            noisy /= np.linalg.norm(noisy)
            _, weights = net.retrieve(noisy)
            assert np.argmax(weights) == i, (
                f"Pattern {i}: noisy query returned wrong index {np.argmax(weights)}"
            )

    def test_negative_patterns_counted(self):
        net = RepulsiveMHN(dim=64, beta=8.0, beta_neg=4.0)
        rng = np.random.default_rng(1)

        for p in _random_patterns(3, 64, rng):
            net.store(p)
        for p in _random_patterns(2, 64, rng):
            net.store_negative(p)

        assert net.num_patterns == 3
        assert net.num_negative_patterns == 2


class TestEnergyComponents:
    def test_energy_has_three_terms(self):
        dim = 64
        rng = np.random.default_rng(7)
        net = RepulsiveMHN(dim=dim, beta=5.0, beta_neg=3.0)

        for p in _random_patterns(3, dim, rng):
            net.store(p)
        for p in _random_patterns(2, dim, rng):
            net.store_negative(p)

        state = rng.standard_normal(dim)
        state /= np.linalg.norm(state)
        ec = net.energy_components(state)

        assert isinstance(ec, EnergyComponents)
        assert ec.positive < 0, "Positive energy term should be negative (attractive)"
        assert ec.negative > 0, "Negative energy term should be positive (repulsive)"
        assert ec.quadratic > 0, "Quadratic term should be positive"
        assert abs(ec.total - (ec.positive + ec.negative + ec.quadratic)) < 1e-10

    def test_without_negatives_repulsive_term_is_zero(self):
        dim = 64
        rng = np.random.default_rng(99)
        net = RepulsiveMHN(dim=dim, beta=5.0, beta_neg=3.0)

        for p in _random_patterns(4, dim, rng):
            net.store(p)

        state = rng.standard_normal(dim)
        state /= np.linalg.norm(state)
        ec = net.energy_components(state)

        assert ec.negative == 0.0


def test_energy_decreases_fixed_beta():
    """Energy should decrease (or stay flat) after update with fixed beta.

    Uses a tolerance of 0.1 because the contrastive energy landscape does
    not have the same strict monotonic guarantee as the standard MHN.
    """
    dim = 128
    rng = np.random.default_rng(42)
    net = RepulsiveMHN(dim=dim, beta=8.0, beta_neg=4.0, adaptive_beta=False, clamp_radius=2.0)

    for p in _random_patterns(5, dim, rng):
        net.store(p)
    for p in _random_patterns(3, dim, rng):
        net.store_negative(p)

    query = rng.standard_normal(dim)
    query /= np.linalg.norm(query)

    e_before = net.energy(query)
    retrieved, _ = net.retrieve(query, num_steps=1)
    e_after = net.energy(retrieved)

    assert e_after <= e_before + 0.1, (
        f"Energy increased beyond tolerance: {e_before:.4f} -> {e_after:.4f}"
    )


def test_norm_clamping():
    """State norm must not exceed clamp_radius after update."""
    dim = 64
    rng = np.random.default_rng(11)
    radius = 0.5
    net = RepulsiveMHN(dim=dim, beta=10.0, beta_neg=8.0, clamp_radius=radius)

    for p in _random_patterns(3, dim, rng):
        net.store(p)
    for p in _random_patterns(2, dim, rng):
        net.store_negative(p)

    query = rng.standard_normal(dim) * 5.0
    retrieved, _ = net.retrieve(query, num_steps=5)
    assert np.linalg.norm(retrieved) <= radius + 1e-10


def test_fallback_without_negatives_noisy():
    """Without negatives, RepulsiveMHN should recall noisy queries correctly."""
    dim = 128
    rng = np.random.default_rng(55)
    net = RepulsiveMHN(dim=dim, beta=8.0, beta_neg=4.0, adaptive_beta=False, clamp_radius=10.0)

    patterns = _random_patterns(5, dim, rng)
    for p in patterns:
        net.store(p)

    for i, p in enumerate(patterns):
        noise = rng.standard_normal(dim) * 0.15
        noisy = p + noise
        noisy /= np.linalg.norm(noisy)
        _, weights = net.retrieve(noisy)
        assert np.argmax(weights) == i
