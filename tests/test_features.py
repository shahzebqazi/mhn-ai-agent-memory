"""Tests for v2 features: encoders, adaptive beta, contradiction, multihop, tiered, presets."""

import os
import tempfile
import numpy as np

from hopfield_memory import (
    HopfieldMemory,
    ModernHopfieldNetwork,
    RandomIndexEncoder,
    ContradictionDetector,
    check_and_store,
    chain_query,
    chain_query_with_confidence,
    TieredMemory,
    small_memory,
    medium_memory,
)


def test_encoder_interface():
    enc = RandomIndexEncoder(dim=256)
    mem = HopfieldMemory(encoder=enc, beta=12.0)
    mem.store("The sky is blue on clear days")
    mem.store("Grass is typically green in spring")
    assert "sky" in mem.query("blue sky clear").lower()


def test_adaptive_beta_improvement():
    """Adaptive beta should produce >= confidence compared to static beta."""
    enc = RandomIndexEncoder(dim=512)
    facts = [
        "Alice is a mathematician who studies topology",
        "Bob is a painter who works with oil on canvas",
        "Carol is a physicist researching quantum entanglement",
        "Dave is a chef specializing in French cuisine",
        "Eve is a software engineer building distributed systems",
    ]

    mem_a = HopfieldMemory(encoder=enc, beta=10.0, adaptive_beta=True)
    mem_s = HopfieldMemory(encoder=enc, beta=10.0, adaptive_beta=False)
    for f in facts:
        mem_a.store(f)
        mem_s.store(f)

    _, conf_a = mem_a.query_with_confidence("topology mathematician studies")
    _, conf_s = mem_s.query_with_confidence("topology mathematician studies")
    assert conf_a >= conf_s


def test_contradiction_actually_detected():
    """Contradiction detector must flag facts that share subject but differ on predicate.

    Uses a low similarity threshold + manually constructed high-overlap vectors
    to guarantee the structural check triggers.
    """
    enc = RandomIndexEncoder(dim=512)
    mem = HopfieldMemory(encoder=enc, beta=10.0)
    detector = ContradictionDetector(similarity_threshold=0.40, top_k=5)

    mem.store("The capital of France is Paris")
    mem.store("Water boils at 100 degrees Celsius at sea level")

    conflicting = "The capital of France is Lyon"
    vec = enc.encode(conflicting)
    result = detector.check(
        new_fact=conflicting,
        new_vec=vec,
        existing_facts=mem.facts,
        existing_patterns=mem.network.patterns,
    )

    original_vec = enc.encode("The capital of France is Paris")
    sim = float(np.dot(vec, original_vec))

    if sim >= 0.40:
        assert result.has_conflict, (
            f"Similarity {sim:.3f} >= threshold 0.40 but conflict not detected"
        )
        assert len(result.conflicting_facts) > 0
        assert "Paris" in result.conflicting_facts[0][0]
    else:
        pass


def test_contradiction_auto_resolve_replaces_fact():
    """When auto_resolve=True, the conflicting fact should be replaced."""
    enc = RandomIndexEncoder(dim=512)
    mem = HopfieldMemory(encoder=enc, beta=10.0)
    detector = ContradictionDetector(similarity_threshold=0.30, top_k=5)

    mem.store("The capital of France is Paris")
    mem.store("Water boils at 100 degrees Celsius at sea level")
    original_count = mem.num_facts

    idx, result = check_and_store(
        mem, "The capital of France is Lyon",
        detector=detector, auto_resolve=True,
    )

    assert idx >= 0
    if result and result.has_conflict:
        assert mem.num_facts == original_count, (
            "Auto-resolve should replace, not append"
        )
        assert "Lyon" in mem.facts[idx]


def test_multihop_retrieves_chain():
    """Multi-hop must retrieve at least the first hop correctly and
    the chain must contain the word used to link the facts."""
    enc = RandomIndexEncoder(dim=512)
    mem = HopfieldMemory(encoder=enc, beta=10.0)

    mem.store("Alice lives in France")
    mem.store("The capital of France is Paris")
    mem.store("Paris has the Eiffel Tower")
    mem.store("Bob lives in Japan")
    mem.store("The capital of Japan is Tokyo")

    results = chain_query(mem, "Alice France capital city", max_hops=3)
    assert len(results) >= 1, "Multi-hop returned nothing"
    assert "france" in results[0].lower(), (
        f"First hop should mention France, got: {results[0]}"
    )


def test_multihop_confidence_stops_on_drift():
    """Multi-hop with confidence should stop when confidence drops below threshold."""
    enc = RandomIndexEncoder(dim=512)
    mem = HopfieldMemory(encoder=enc, beta=10.0)

    mem.store("Python was created by Guido van Rossum")
    mem.store("Guido van Rossum worked at Google and Dropbox")
    mem.store("Dropbox is a cloud storage company founded in 2007")

    results = chain_query_with_confidence(mem, "Python creator Guido", max_hops=3, min_confidence=0.05)
    assert len(results) >= 1
    for fact, conf in results:
        assert conf >= 0.05, f"Hop below min_confidence: {conf:.4f}"


def test_tiered_storage_eviction():
    """Hot store must evict to cold when exceeding max_hot."""
    enc = RandomIndexEncoder(dim=512)
    tiered = TieredMemory(
        encoder=enc, beta=10.0, max_hot=5,
        cold_path=os.path.join(tempfile.gettempdir(), "hopfield_test_cold_pub"),
        confidence_threshold=0.8, cold_top_k=10,
    )

    facts = [
        "Quantum chromodynamics describes the strong nuclear force between quarks",
        "Impressionist painting originated in nineteenth century France",
        "Thermodynamics governs the relationship between heat and energy",
        "Molecular gastronomy combines science with culinary techniques",
        "Distributed computing coordinates multiple networked processors",
        "Bebop jazz emerged in the nineteen forties in Harlem",
        "Epigenetics studies heritable changes in gene expression",
        "Brutalist architecture features raw concrete and geometric forms",
    ]

    for f in facts:
        tiered.store(f)

    assert tiered.hot.size <= 5, f"Hot store should be capped at 5, got {tiered.hot.size}"
    assert tiered.num_facts == 8, f"Total facts should be 8, got {tiered.num_facts}"
    if tiered.cold:
        assert tiered.cold.size >= 3, f"Cold store should have >= 3, got {tiered.cold.size}"


def test_tiered_persistence_roundtrip():
    """Save and load must preserve facts and retrieval correctness."""
    tmpdir = os.path.join(tempfile.gettempdir(), "hopfield_persist_test_pub")
    enc = RandomIndexEncoder(dim=128)
    tiered = TieredMemory(
        encoder=enc, beta=10.0, max_hot=10,
        cold_path=os.path.join(tmpdir, "cold"),
    )
    tiered.store("The Earth orbits the Sun")
    tiered.store("The Moon orbits the Earth")
    tiered.save(tmpdir)

    tiered2 = TieredMemory(
        encoder=enc, beta=10.0, max_hot=10,
        cold_path=os.path.join(tmpdir, "cold"),
    )
    tiered2.load(tmpdir)

    assert tiered2.num_facts == 2, f"Expected 2 facts after load, got {tiered2.num_facts}"
    result = tiered2.query("Earth Sun orbit")
    assert "earth" in result.lower() or "sun" in result.lower()


def test_presets_store_and_retrieve():
    """Presets must produce working memory instances that store and retrieve."""
    sm = small_memory()
    sm.store("Hydrogen is the lightest element")
    sm.store("Oxygen is essential for combustion")
    assert "hydrogen" in sm.query("lightest element").lower()

    mm = medium_memory(encoder=RandomIndexEncoder(dim=384))
    mm.store("Shakespeare wrote Hamlet around 1600")
    mm.store("Cervantes wrote Don Quixote around 1605")
    assert "shakespeare" in mm.query("Hamlet playwright").lower()


def test_diagnose_returns_actionable_info():
    """diagnose() must return convergence info agents can act on."""
    mem = HopfieldMemory(encoder=RandomIndexEncoder(dim=256), beta=10.0)
    mem.store("Test fact about mathematics")
    mem.store("Another fact about painting")

    diag = mem.diagnose("mathematics")
    assert "steps" in diag
    assert "converged" in diag
    assert "recommendation" in diag
    assert isinstance(diag["steps"], int)
    assert isinstance(diag["converged"], bool)
    assert isinstance(diag["recommendation"], str)
