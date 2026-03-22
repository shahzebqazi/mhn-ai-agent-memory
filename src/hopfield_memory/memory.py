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
        sentinel: bool = True,
    ):
        if encoder is not None:
            self.encoder = encoder
        elif dim is not None:
            self.encoder = RandomIndexEncoder(dim=dim)
        else:
            self.encoder = auto_encoder()

        effective_dim = self.encoder.dim
        self.repulsive = repulsive
        self._sentinel = sentinel
        self._sentinel_idx: int = -1

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

        if sentinel:
            null_vec = np.zeros(effective_dim)
            self._sentinel_idx = self.network.store(null_vec)
            self.facts.append("")

    def store(self, fact: str) -> int:
        """Store a text fact. Returns the index."""
        vec = self.encoder.encode(fact)
        idx = self.network.store(vec)
        self.facts.append(fact)
        return idx

    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Retrieve the top-k most relevant facts for a query.

        Returns list of ``(fact_text, attention_weight)`` tuples.
        The sentinel pattern (if enabled) is excluded from results.
        """
        if not self.facts:
            return []

        query_vec = self.encoder.encode(query)
        _, weights = self.network.retrieve(query_vec)

        ranked = sorted(enumerate(weights), key=lambda x: -x[1])
        results = []
        for idx, w in ranked:
            if idx == self._sentinel_idx:
                continue
            if not self.facts[idx]:
                continue
            results.append((self.facts[idx], float(w)))
            if len(results) >= top_k:
                break
        return results

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
        return sum(1 for f in self.facts if f)

    def all_facts(self) -> List[str]:
        return [f for f in self.facts if f]

    def match_quality(self, query: str) -> dict:
        """Assess how well a query matches any stored fact.

        Combines three independent signals:

        1. **max_similarity** -- highest raw dot product between the query
           vector and any stored pattern (before softmax). This is the most
           reliable signal: high means shared content, low means no overlap.
        2. **gap** -- difference between the top two attention weights.
           Large gap = strong match. Small gap = ambiguous or no match.
        3. **sentinel_weight** -- attention weight on the null sentinel pattern.
           High sentinel weight = query is far from all real patterns.

        Returns a dict with all signals plus a boolean ``is_match``.
        """
        if not self.facts or self.num_facts == 0:
            return {
                "max_similarity": 0.0,
                "gap": 0.0,
                "energy": 0.0,
                "sentinel_weight": 1.0,
                "top_confidence": 0.0,
                "is_match": False,
            }

        query_vec = self.encoder.encode(query)
        retrieved, weights = self.network.retrieve(query_vec)

        X = self.network._pattern_matrix()
        raw_sims = X.T @ query_vec
        real_sims = [float(raw_sims[i]) for i in range(len(self.facts))
                     if i != self._sentinel_idx and self.facts[i]]
        max_sim = max(real_sims) if real_sims else 0.0

        sorted_w = np.sort(weights)[::-1]
        gap = float(sorted_w[0] - sorted_w[1]) if len(sorted_w) > 1 else float(sorted_w[0])

        energy = float(self.network.energy(retrieved))

        sentinel_w = float(weights[self._sentinel_idx]) if self._sentinel_idx >= 0 else 0.0

        is_match = (
            max_sim >= 0.25
            and gap > 0.01
            and (self._sentinel_idx < 0 or sentinel_w < 0.5)
        )

        return {
            "max_similarity": max_sim,
            "gap": gap,
            "energy": energy,
            "sentinel_weight": sentinel_w,
            "top_confidence": float(np.max(weights)),
            "is_match": is_match,
        }

    def has_match(self, query: str, min_similarity: float = 0.25) -> bool:
        """Return True if the query meaningfully matches a stored fact.

        Uses the maximum raw dot product between the query vector and
        stored patterns as the primary signal (equals cosine similarity
        when patterns and query are unit-norm, as with RandomIndexEncoder).

        Also checks that the attention gap is above a floor and that the
        sentinel pattern does not dominate the softmax.

        Parameters
        ----------
        query : str
            The query text.
        min_similarity : float
            Minimum dot product to any stored pattern. Default 0.25 works
            well with the RandomIndexEncoder for 5+ stored facts.
        """
        mq = self.match_quality(query)
        return (
            mq["max_similarity"] >= min_similarity
            and mq["gap"] > 0.01
            and (self._sentinel_idx < 0 or mq["sentinel_weight"] < 0.5)
        )

    def query_or_none(self, question: str, min_similarity: float = 0.25) -> Optional[str]:
        """Return the best matching fact, or None if nothing matches.

        This is the primary method for agents that need to distinguish
        "I found a relevant memory" from "nothing in memory is relevant."
        """
        if not self.has_match(question, min_similarity=min_similarity):
            return None
        return self.query(question)

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
            "sentinel": self._sentinel,
            "sentinel_idx": self._sentinel_idx,
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
            sentinel=False,
        )
        for fact, pattern in zip(data["facts"], data["patterns"]):
            vec = np.array(pattern)
            mem.network.store(vec)
            mem.facts.append(fact)

        def _infer_legacy_sentinel(d):
            if not d.get("facts") or d["facts"][0] != "":
                return False
            if not d.get("patterns") or any(v != 0.0 for v in d["patterns"][0]):
                return False
            return True

        has_sentinel = data.get("sentinel", _infer_legacy_sentinel(data))
        mem._sentinel = has_sentinel
        mem._sentinel_idx = int(data.get("sentinel_idx",
                                         0 if has_sentinel else -1))

        if is_repulsive and "negative_patterns" in data:
            for fact, pattern in zip(data.get("negative_facts", []), data["negative_patterns"]):
                vec = np.array(pattern)
                mem.network.store_negative(vec)
                mem.negative_facts.append(fact)

        return mem
