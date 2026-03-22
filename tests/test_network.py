"""Core network tests: mathematical properties + genuine memory behavior."""

import numpy as np
from hopfield_memory import HopfieldMemory, ModernHopfieldNetwork, RandomIndexEncoder


def _mem(dim=512, beta=10.0):
    return HopfieldMemory(encoder=RandomIndexEncoder(dim=dim), beta=beta)


class TestStoreAndRetrieve:
    """Store facts, query with keyword subsets, verify correct recall."""

    FACTS = [
        "Alice is a mathematician who studies topology and algebraic geometry",
        "Bob is a painter who works with oil on canvas landscapes",
        "Carol is a physicist researching quantum entanglement at CERN",
        "Dave is a chef specializing in French cuisine and pastry",
        "Eve is a software engineer building distributed systems in Rust",
        "Frank is a musician who plays jazz piano at clubs in New York",
        "Grace is a biologist studying CRISPR gene editing techniques",
        "Hank is an architect designing sustainable buildings with bamboo",
    ]

    QUERIES = [
        ("topology algebraic geometry", "Alice"),
        ("painting oil canvas", "Bob"),
        ("quantum physics entanglement", "Carol"),
        ("French cuisine pastry chef", "Dave"),
        ("software engineer distributed Rust", "Eve"),
        ("jazz piano musician", "Frank"),
        ("biology CRISPR gene editing", "Grace"),
        ("architect sustainable bamboo buildings", "Hank"),
    ]

    def test_all_eight_facts_recalled(self):
        mem = _mem()
        for f in self.FACTS:
            mem.store(f)
        for query_text, expected_name in self.QUERIES:
            result = mem.query(query_text)
            assert expected_name.lower() in result.lower(), (
                f"query={query_text!r} expected={expected_name!r} got={result!r}"
            )


class TestPartialCueRetrieval:
    """Query with fewer keywords than the stored fact contains."""

    def test_keyword_queries(self):
        mem = _mem(beta=12.0)
        mem.store("The capital of France is Paris")
        mem.store("The speed of light is approximately 300000 kilometers per second")
        mem.store("Water boils at 100 degrees Celsius at sea level")
        mem.store("Python was created by Guido van Rossum in 1991")
        mem.store("The mitochondria is the powerhouse of the cell")

        for query, fragment in [
            ("France Paris capital", "France"),
            ("speed light kilometers", "light"),
            ("water boils degrees", "Water boils"),
            ("Python Guido created", "Python"),
            ("mitochondria cell", "mitochondria"),
        ]:
            result = mem.query(query)
            assert fragment.lower() in result.lower()


class TestNoisyRetrieval:
    """Query with corrupted/noisy versions of stored patterns.

    This is the core test for associative memory: given a degraded cue,
    can the network reconstruct the correct stored pattern?
    """

    def test_noisy_vector_recall(self):
        dim = 256
        rng = np.random.default_rng(42)
        net = ModernHopfieldNetwork(dim=dim, beta=10.0, adaptive_beta=False)

        patterns = []
        for _ in range(10):
            p = rng.standard_normal(dim)
            p /= np.linalg.norm(p)
            net.store(p)
            patterns.append(p)

        for i, p in enumerate(patterns):
            noise = rng.standard_normal(dim) * 0.2
            noisy = p + noise
            noisy /= np.linalg.norm(noisy)

            _, weights = net.retrieve(noisy, num_steps=1)
            assert np.argmax(weights) == i, (
                f"Pattern {i}: noisy query retrieved wrong pattern (got {np.argmax(weights)})"
            )

    def test_noisy_text_recall(self):
        """Query with only 2 of 5+ keywords, verify correct fact returned."""
        mem = _mem(dim=512, beta=10.0)
        mem.store("The Amazon river flows through Brazil and Peru in South America")
        mem.store("Mount Everest is the tallest mountain on Earth at 8849 meters")
        mem.store("The Great Wall of China was built over many centuries by multiple dynasties")

        assert "amazon" in mem.query("Amazon Brazil").lower()
        assert "everest" in mem.query("tallest mountain").lower()
        assert "wall" in mem.query("China dynasties").lower()


class TestQueryNotStored:
    """Verify behavior when querying for something not in memory."""

    def test_empty_memory(self):
        mem = _mem()
        result = mem.query("anything at all")
        assert result == "[No facts stored]"

    def test_relevant_query_retrieves_correct_fact(self):
        """A relevant query must retrieve the matching fact, not a random one.

        Note: Softmax confidence is relative, not absolute. The network cannot
        detect "no match" -- it always picks the closest pattern. This is a
        documented limitation, not a bug.
        """
        mem = _mem(dim=256, beta=10.0)
        mem.store("The chemical formula for water is H2O")
        mem.store("Pythagoras developed the theorem about right triangles")
        mem.store("Shakespeare wrote Hamlet in approximately 1600")
        mem.store("The speed of light is 299792458 meters per second")
        mem.store("DNA carries genetic information in living organisms")

        assert "water" in mem.query("water H2O chemical formula").lower()
        assert "pythagoras" in mem.query("theorem right triangles").lower()
        assert "shakespeare" in mem.query("Hamlet wrote approximately").lower()


def test_confidence_separation():
    """Top result should have much higher weight than the runner-up."""
    mem = _mem()
    mem.store("Haskell is a purely functional programming language with lazy evaluation")
    mem.store("Rust is a systems programming language focused on memory safety")
    mem.store("The Eiffel Tower was built in 1889 for the World Fair in Paris")

    results = mem.retrieve("Haskell functional lazy", top_k=3)
    top_weight = results[0][1]
    second_weight = results[1][1]
    assert top_weight > 0.5
    assert top_weight > 2 * second_weight


def test_energy_decreases():
    """Energy must decrease under fixed beta (matches the convergence proof)."""
    dim = 128
    net = ModernHopfieldNetwork(dim=dim, beta=5.0, adaptive_beta=False)
    rng = np.random.default_rng(42)

    for _ in range(5):
        p = rng.standard_normal(dim)
        p /= np.linalg.norm(p)
        net.store(p)

    query = rng.standard_normal(dim)
    query /= np.linalg.norm(query)

    e_before = net.energy(query)
    retrieved, _ = net.retrieve(query, num_steps=1)
    e_after = net.energy(retrieved)

    assert e_after <= e_before + 1e-10


def test_energy_decreases_adaptive():
    """With adaptive beta, energy() now uses the state-dependent beta.

    Since the effective beta varies with state, this is not a strict
    Lyapunov guarantee. We test the fixed-beta energy surface (passing
    base_beta explicitly) to verify the underlying energy still roughly
    decreases even under adaptive dynamics.
    """
    dim = 128
    net = ModernHopfieldNetwork(dim=dim, beta=5.0, adaptive_beta=True)
    rng = np.random.default_rng(42)

    for _ in range(5):
        p = rng.standard_normal(dim)
        p /= np.linalg.norm(p)
        net.store(p)

    query = rng.standard_normal(dim)
    query /= np.linalg.norm(query)

    e_before = net.energy(query, beta=net.base_beta)
    retrieved, _ = net.retrieve(query, num_steps=1)
    e_after = net.energy(retrieved, beta=net.base_beta)

    assert e_after <= e_before + 0.5


def test_capacity_with_noisy_queries():
    """Store N patterns, query with noisy versions, verify all recall correctly.

    Uses adaptive_beta=False for a theory-aligned capacity test under
    fixed inverse temperature.
    """
    dim = 256
    rng = np.random.default_rng(123)

    for n_patterns in [5, 10, 20, 50]:
        net = ModernHopfieldNetwork(dim=dim, beta=10.0, adaptive_beta=False)
        patterns = []
        for _ in range(n_patterns):
            p = rng.standard_normal(dim)
            p /= np.linalg.norm(p)
            net.store(p)
            patterns.append(p)

        correct = 0
        for i, p in enumerate(patterns):
            noise = rng.standard_normal(dim) * 0.15
            noisy = p + noise
            noisy /= np.linalg.norm(noisy)
            _, weights = net.retrieve(noisy, num_steps=1)
            if np.argmax(weights) == i:
                correct += 1

        assert correct == n_patterns, (
            f"N={n_patterns}: {correct}/{n_patterns} recalled under noise"
        )
