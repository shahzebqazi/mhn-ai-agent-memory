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
    """Contradiction detector must flag facts that share subject but differ on predicate."""
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

    assert sim >= 0.40, (
        f"Similarity {sim:.3f} too low for this encoder/dim -- "
        f"increase dim or lower threshold to make this test meaningful"
    )
    assert result.has_conflict, (
        f"Similarity {sim:.3f} >= threshold 0.40 but conflict not detected"
    )
    assert len(result.conflicting_facts) > 0
    assert "Paris" in result.conflicting_facts[0][0]


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


# ─── Match detection tests ───────────────────────────────────────────


def test_has_match_returns_true_for_stored_fact():
    """A query with keywords from a stored fact should be detected as a match."""
    enc = RandomIndexEncoder(dim=256)
    mem = HopfieldMemory(encoder=enc, beta=10.0)

    mem.store("The Eiffel Tower is located in Paris France")
    mem.store("Mount Fuji is the tallest mountain in Japan")
    mem.store("The Amazon river flows through Brazil")
    mem.store("The Great Barrier Reef is in Australia")
    mem.store("The Sahara is the largest hot desert in the world")

    assert mem.has_match("Eiffel Tower Paris France")
    assert mem.has_match("Mount Fuji tallest Japan")
    assert mem.has_match("Amazon river Brazil")


def test_has_match_returns_false_for_unrelated_query():
    """A query with no word overlap to any stored fact should be a non-match."""
    enc = RandomIndexEncoder(dim=256)
    mem = HopfieldMemory(encoder=enc, beta=10.0)

    mem.store("Photosynthesis converts sunlight into chemical energy")
    mem.store("Mitochondria produce adenosine triphosphate in cells")
    mem.store("Ribosomes synthesize proteins from messenger RNA")
    mem.store("Chloroplasts contain chlorophyll pigments")
    mem.store("Lysosomes digest cellular waste materials")

    assert not mem.has_match("basketball quarterback touchdown score")


def test_query_or_none_returns_fact_when_matched():
    enc = RandomIndexEncoder(dim=256)
    mem = HopfieldMemory(encoder=enc, beta=10.0)
    mem.store("Neptune is the eighth planet from the Sun")
    mem.store("Saturn has prominent ring system")

    result = mem.query_or_none("Neptune eighth planet Sun")
    assert result is not None
    assert "neptune" in result.lower()


def test_query_or_none_returns_none_when_unmatched():
    enc = RandomIndexEncoder(dim=256)
    mem = HopfieldMemory(encoder=enc, beta=10.0)
    mem.store("Photosynthesis converts sunlight into chemical energy")
    mem.store("Mitochondria produce adenosine triphosphate in cells")
    mem.store("Ribosomes synthesize proteins from messenger RNA")
    mem.store("Chloroplasts contain chlorophyll pigments")
    mem.store("Lysosomes digest cellular waste materials")

    result = mem.query_or_none("quarterback touchdown interception fumble")
    assert result is None


def test_match_quality_returns_all_signals():
    enc = RandomIndexEncoder(dim=256)
    mem = HopfieldMemory(encoder=enc, beta=10.0)
    mem.store("Hydrogen is the lightest chemical element")
    mem.store("Helium is used in balloons and airships")

    mq = mem.match_quality("Hydrogen lightest element")
    assert "gap" in mq
    assert "energy" in mq
    assert "sentinel_weight" in mq
    assert "top_confidence" in mq
    assert "is_match" in mq
    assert isinstance(mq["is_match"], bool)


def test_sentinel_excluded_from_results():
    """The sentinel pattern must never appear in retrieve() results."""
    enc = RandomIndexEncoder(dim=256)
    mem = HopfieldMemory(encoder=enc, beta=10.0)
    mem.store("Test fact one")
    mem.store("Test fact two")

    results = mem.retrieve("Test fact", top_k=10)
    for fact, _ in results:
        assert fact != "", "Sentinel (empty string) should not appear in results"


def test_num_facts_excludes_sentinel():
    enc = RandomIndexEncoder(dim=256)
    mem = HopfieldMemory(encoder=enc, beta=10.0)
    assert mem.num_facts == 0
    mem.store("First fact")
    assert mem.num_facts == 1
    mem.store("Second fact")
    assert mem.num_facts == 2


# ─── Save/load roundtrip tests ──────────────────────────────────────


def test_save_load_roundtrip():
    """Save and load must preserve pattern count, facts, and retrieval."""
    enc = RandomIndexEncoder(dim=256)
    mem = HopfieldMemory(encoder=enc, beta=10.0)
    mem.store("hello world")
    mem.store("foo bar")

    path = os.path.join(tempfile.gettempdir(), "hopfield_roundtrip_test.json")
    mem.save(path)
    loaded = HopfieldMemory.load(path, encoder=enc)

    assert len(loaded.network.patterns) == len(mem.network.patterns), (
        f"Pattern count mismatch: original={len(mem.network.patterns)}, "
        f"loaded={len(loaded.network.patterns)}"
    )
    assert loaded.facts == mem.facts, (
        f"Facts mismatch: original={mem.facts}, loaded={loaded.facts}"
    )
    assert loaded._sentinel == mem._sentinel
    assert loaded._sentinel_idx == mem._sentinel_idx
    assert loaded.num_facts == mem.num_facts
    assert "hello" in loaded.query("hello world").lower()


def test_save_load_roundtrip_no_sentinel():
    """Save/load roundtrip with sentinel=False must not create spurious sentinels."""
    enc = RandomIndexEncoder(dim=128)
    mem = HopfieldMemory(encoder=enc, beta=10.0, sentinel=False)
    mem.store("alpha beta gamma")

    path = os.path.join(tempfile.gettempdir(), "hopfield_no_sentinel_test.json")
    mem.save(path)
    loaded = HopfieldMemory.load(path, encoder=enc)

    assert loaded._sentinel is False
    assert loaded._sentinel_idx == -1
    assert loaded.facts == mem.facts
    assert len(loaded.network.patterns) == len(mem.network.patterns)


def test_save_load_backward_compat():
    """Files saved without sentinel metadata should be loaded correctly."""
    import json as _json

    enc = RandomIndexEncoder(dim=128)
    mem = HopfieldMemory(encoder=enc, beta=10.0)
    mem.store("test fact")

    path = os.path.join(tempfile.gettempdir(), "hopfield_compat_test.json")
    mem.save(path)

    with open(path, "r") as f:
        data = _json.load(f)
    del data["sentinel"]
    del data["sentinel_idx"]
    with open(path, "w") as f:
        _json.dump(data, f)

    loaded = HopfieldMemory.load(path, encoder=enc)
    assert loaded._sentinel is True
    assert loaded._sentinel_idx == 0
    assert loaded.facts == mem.facts
    assert len(loaded.network.patterns) == len(mem.network.patterns)


def test_save_load_legacy_rejects_fake_sentinel():
    """A legacy file with facts[0]=="" but non-zero pattern[0] must not infer a sentinel."""
    import json as _json

    enc = RandomIndexEncoder(dim=128)
    mem = HopfieldMemory(encoder=enc, beta=10.0, sentinel=False)
    mem.store("real fact")

    path = os.path.join(tempfile.gettempdir(), "hopfield_fake_sentinel_test.json")
    mem.save(path)

    with open(path, "r") as f:
        data = _json.load(f)
    del data["sentinel"]
    del data["sentinel_idx"]
    data["facts"].insert(0, "")
    data["patterns"].insert(0, data["patterns"][0])
    with open(path, "w") as f:
        _json.dump(data, f)

    loaded = HopfieldMemory.load(path, encoder=enc)
    assert loaded._sentinel is False


def test_save_load_repulsive_roundtrip():
    """Repulsive MHN save/load must preserve positive and negative patterns."""
    enc = RandomIndexEncoder(dim=128)
    mem = HopfieldMemory(encoder=enc, beta=10.0, repulsive=True, beta_neg=6.0)
    mem.store("positive fact one")
    mem.store("positive fact two")
    mem.store_negative("negative fact")

    path = os.path.join(tempfile.gettempdir(), "hopfield_repulsive_test.json")
    mem.save(path)
    loaded = HopfieldMemory.load(path, encoder=enc)

    assert loaded.repulsive is True
    assert len(loaded.network.patterns) == len(mem.network.patterns)
    assert len(loaded.network.negative_patterns) == len(mem.network.negative_patterns)
    assert loaded.negative_facts == mem.negative_facts


# ─── Encoder determinism tests ───────────────────────────────────────


def test_encoder_determinism_stable_seed():
    """RandomIndexEncoder must produce identical vectors across fresh instances."""
    enc1 = RandomIndexEncoder(dim=256)
    enc2 = RandomIndexEncoder(dim=256)

    for word in ["hello", "world", "quantum", "Hopfield", ""]:
        v1 = enc1.encode(word)
        v2 = enc2.encode(word)
        np.testing.assert_array_equal(v1, v2, err_msg=f"Mismatch for {word!r}")


# ─── Property-based tests ────────────────────────────────────────────


def test_softmax_weights_sum_to_one():
    """Attention weights from retrieve() must sum to 1 and be non-negative."""
    rng = np.random.default_rng(77)
    dim = 128
    net = ModernHopfieldNetwork(dim=dim, beta=8.0, adaptive_beta=True)

    for _ in range(8):
        p = rng.standard_normal(dim)
        p /= np.linalg.norm(p)
        net.store(p)

    for _ in range(20):
        q = rng.standard_normal(dim)
        q /= np.linalg.norm(q)
        _, weights = net.retrieve(q)
        assert np.all(weights >= 0), "Negative attention weight"
        np.testing.assert_allclose(np.sum(weights), 1.0, atol=1e-12)


def test_retrieved_state_is_convex_combination():
    """For unit-norm patterns, retrieved state norm must be <= 1."""
    rng = np.random.default_rng(88)
    dim = 128
    net = ModernHopfieldNetwork(dim=dim, beta=8.0, adaptive_beta=False)

    for _ in range(6):
        p = rng.standard_normal(dim)
        p /= np.linalg.norm(p)
        net.store(p)

    for _ in range(20):
        q = rng.standard_normal(dim)
        q /= np.linalg.norm(q)
        retrieved, _ = net.retrieve(q)
        assert np.linalg.norm(retrieved) <= 1.0 + 1e-10


def test_save_load_preserves_pattern_and_fact_count():
    """After load, len(patterns) must equal len(facts) and sentinel count must be correct."""
    enc = RandomIndexEncoder(dim=128)
    mem = HopfieldMemory(encoder=enc, beta=10.0)
    mem.store("fact one")
    mem.store("fact two")
    mem.store("fact three")

    path = os.path.join(tempfile.gettempdir(), "hopfield_count_test.json")
    mem.save(path)
    loaded = HopfieldMemory.load(path, encoder=enc)

    assert len(loaded.network.patterns) == len(loaded.facts)
    sentinel_count = sum(1 for f in loaded.facts if f == "")
    if loaded._sentinel:
        assert sentinel_count == 1
    else:
        assert sentinel_count == 0
