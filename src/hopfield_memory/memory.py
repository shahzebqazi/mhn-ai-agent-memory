"""
High-level memory API backed by a Modern Hopfield Network.

Store text facts. Retrieve the most relevant fact given a text query.
No LLM calls. No database. One matrix multiply.
"""

import json
import numpy as np
from typing import List, Tuple, Optional

from hopfield_memory.network import ModernHopfieldNetwork
from hopfield_memory.repulsive import RepulsiveMHN
from hopfield_memory.encoders import Encoder, RandomIndexEncoder, auto_encoder

TextEncoder = RandomIndexEncoder


class HopfieldMemory:
    """AI agent memory backed by a Modern Hopfield Network.

    Accepts a pluggable :class:`Encoder` for semantic text encoding.
    Uses adaptive beta by default for higher retrieval confidence.

    When ``repulsive=True``, uses a :class:`RepulsiveMHN` backend with
    contrastive attention. This does not change retrieval accuracy but
    provides up to 17x faster convergence for multi-step retrieval.
    Agents can call :meth:`store_negative` to mark patterns that the
    network should actively avoid, and :meth:`diagnose` to measure
    whether repulsive mode is beneficial for their current workload.

    Parameters
    ----------
    encoder : Encoder, optional
        Text encoder. If None, auto-selects the best available.
    dim : int, optional
        Shortcut to create a RandomIndexEncoder with this dimension.
    beta : float
        Base inverse temperature for the Hopfield network.
    adaptive_beta : bool
        Whether to use per-query adaptive beta.
    repulsive : bool
        If True, use RepulsiveMHN backend with contrastive attention.
    beta_neg : float
        Inverse temperature for the repulsive term (only when repulsive=True).
    clamp_radius : float
        Maximum state norm after each update (only when repulsive=True).
    """

    def __init__(
        self,
        encoder: Optional[Encoder] = None,
        dim: Optional[int] = None,
        beta: float = 10.0,
        adaptive_beta: bool = True,
        repulsive: bool = False,
        beta_neg: float = 6.0,
        clamp_radius: float = 1.5,
    ):
        if encoder is not None:
            self.encoder = encoder
        elif dim is not None:
            self.encoder = RandomIndexEncoder(dim=dim)
        else:
            self.encoder = auto_encoder()

        effective_dim = self.encoder.dim
        self.repulsive = repulsive

        if repulsive:
            self.network = RepulsiveMHN(
                dim=effective_dim, beta=beta, beta_neg=beta_neg,
                adaptive_beta=adaptive_beta, clamp_radius=clamp_radius,
            )
        else:
            self.network = ModernHopfieldNetwork(
                dim=effective_dim, beta=beta, adaptive_beta=adaptive_beta
            )
        self.facts: List[str] = []
        self.negative_facts: List[str] = []

    def store(self, fact: str) -> int:
        """Store a text fact. Returns the index."""
        vec = self.encoder.encode(fact)
        idx = self.network.store(vec)
        self.facts.append(fact)
        return idx

    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Retrieve the top-k most relevant facts for a query.

        Returns list of ``(fact_text, attention_weight)`` tuples.
        """
        if not self.facts:
            return []

        query_vec = self.encoder.encode(query)
        _, weights = self.network.retrieve(query_vec)

        ranked = sorted(enumerate(weights), key=lambda x: -x[1])
        return [(self.facts[idx], float(w)) for idx, w in ranked[:top_k]]

    def query(self, question: str) -> str:
        """Single-shot query: return the best matching fact."""
        results = self.retrieve(question, top_k=1)
        if not results:
            return "[No facts stored]"
        return results[0][0]

    def query_with_confidence(self, question: str) -> Tuple[str, float]:
        """Return the best matching fact and its confidence score."""
        results = self.retrieve(question, top_k=1)
        if not results:
            return "[No facts stored]", 0.0
        return results[0]

    @property
    def num_facts(self) -> int:
        return len(self.facts)

    def all_facts(self) -> List[str]:
        return list(self.facts)

    def store_negative(self, fact: str) -> int:
        """Store a negative (repulsive) fact the network should avoid.

        Only works when ``repulsive=True``. Agents can use this to mark
        known confusable or spurious patterns.

        Returns the negative pattern index, or -1 if not in repulsive mode.
        """
        if not self.repulsive:
            return -1
        vec = self.encoder.encode(fact)
        idx = self.network.store_negative(vec)
        self.negative_facts.append(fact)
        return idx

    def diagnose(self, query: str, num_steps: int = 50, eps: float = 1e-6) -> dict:
        """Measure convergence speed for a query. Helps agents decide if
        repulsive mode is useful for their workload.

        Returns a dict with:
        - ``steps``: number of iterations to converge
        - ``converged``: whether it settled within num_steps
        - ``final_confidence``: attention weight on the top pattern
        - ``repulsive``: whether repulsive mode is active
        - ``recommendation``: a short string agents can act on
        """
        query_vec = self.encoder.encode(query)
        xi = query_vec.copy()

        steps_taken = num_steps
        for step in range(1, num_steps + 1):
            xi_new, weights = self.network.retrieve(xi, num_steps=1)
            if np.linalg.norm(xi_new - xi) < eps:
                steps_taken = step
                break
            xi = xi_new

        _, final_weights = self.network.retrieve(xi, num_steps=1)
        top_conf = float(np.max(final_weights))
        converged = steps_taken < num_steps

        if converged and steps_taken <= 5:
            rec = "fast convergence; repulsive mode not needed"
        elif converged and steps_taken <= 20:
            rec = "moderate convergence; repulsive mode optional"
        elif converged:
            rec = "slow convergence; repulsive mode recommended"
        else:
            rec = "did not converge; repulsive mode strongly recommended"

        return {
            "steps": steps_taken,
            "converged": converged,
            "final_confidence": top_conf,
            "repulsive": self.repulsive,
            "recommendation": rec,
        }

    def save(self, path: str) -> None:
        """Serialize the memory to a JSON file."""
        data = {
            "dim": self.network.dim,
            "beta": self.network.base_beta,
            "adaptive_beta": self.network.adaptive_beta,
            "repulsive": self.repulsive,
            "facts": self.facts,
            "patterns": [p.tolist() for p in self.network.patterns],
            "encoder_type": type(self.encoder).__name__,
        }
        if self.repulsive:
            data["beta_neg"] = self.network.beta_neg
            data["clamp_radius"] = self.network.clamp_radius
            data["negative_facts"] = self.negative_facts
            data["negative_patterns"] = [p.tolist() for p in self.network.negative_patterns]
        with open(path, "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: str, encoder: Optional[Encoder] = None) -> "HopfieldMemory":
        """Deserialize a memory from a JSON file.

        If *encoder* is None, uses RandomIndexEncoder at the saved dimension.
        For full semantic quality, pass the same encoder type used at save time.
        """
        with open(path, "r") as f:
            data = json.load(f)

        if encoder is None:
            encoder = RandomIndexEncoder(dim=data["dim"])

        is_repulsive = data.get("repulsive", False)

        mem = cls(
            encoder=encoder,
            beta=data["beta"],
            adaptive_beta=data.get("adaptive_beta", True),
            repulsive=is_repulsive,
            beta_neg=data.get("beta_neg", 6.0),
            clamp_radius=data.get("clamp_radius", 1.5),
        )
        for fact, pattern in zip(data["facts"], data["patterns"]):
            vec = np.array(pattern)
            mem.network.store(vec)
            mem.facts.append(fact)

        if is_repulsive and "negative_patterns" in data:
            for fact, pattern in zip(data.get("negative_facts", []), data["negative_patterns"]):
                vec = np.array(pattern)
                mem.network.store_negative(vec)
                mem.negative_facts.append(fact)

        return mem
