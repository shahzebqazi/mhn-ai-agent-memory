"""
Tiered hot/cold storage for Hopfield Memory.

HotStore:  exact Modern Hopfield retrieval, patterns in RAM (~10k patterns)
ColdStore: approximate nearest-neighbor on disk (FAISS or numpy fallback)
TieredMemory: unified API routing queries through hot then cold as needed
"""

import os
import json
import warnings
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

from hopfield_memory.encoders import Encoder, auto_encoder
from hopfield_memory.network import ModernHopfieldNetwork


@dataclass
class RetrievalResult:
    """A single retrieval result with source metadata."""
    fact: str
    confidence: float
    source: str
    index: int


class HotStore:
    """In-memory exact Hopfield retrieval."""

    def __init__(self, dim: int, beta: float = 10.0, adaptive_beta: bool = True):
        self.network = ModernHopfieldNetwork(dim=dim, beta=beta, adaptive_beta=adaptive_beta)
        self.facts: List[str] = []
        self.access_counts: List[int] = []

    @property
    def size(self) -> int:
        return len(self.facts)

    def store(self, fact: str, vec: np.ndarray) -> int:
        idx = self.network.store(vec)
        self.facts.append(fact)
        self.access_counts.append(0)
        return idx

    def retrieve(self, query_vec: np.ndarray, top_k: int = 3) -> List[RetrievalResult]:
        if not self.facts:
            return []
        _, weights = self.network.retrieve(query_vec)
        ranked = np.argsort(weights)[::-1][:top_k]
        results = []
        for idx in ranked:
            self.access_counts[idx] += 1
            results.append(RetrievalResult(
                fact=self.facts[idx], confidence=float(weights[idx]),
                source="hot", index=int(idx),
            ))
        return results

    def evict_least_accessed(self, count: int) -> List[Tuple[str, np.ndarray]]:
        if count >= self.size:
            count = max(0, self.size - 1)
        indices = np.argsort(self.access_counts)[:count]
        indices = sorted(indices, reverse=True)
        evicted = []
        for idx in indices:
            evicted.append((self.facts[idx], self.network.patterns[idx].copy()))
            del self.facts[idx]
            del self.network.patterns[idx]
            del self.access_counts[idx]
        self.network._X = None
        return evicted

    def save(self, path: str) -> None:
        if self.network.patterns:
            patterns = np.column_stack(self.network.patterns)
        else:
            patterns = np.zeros((self.network.dim, 0))
        np.savez_compressed(path, patterns=patterns, access_counts=np.array(self.access_counts))

    def load_patterns(self, path: str, facts: List[str]) -> None:
        data = np.load(path)
        patterns = data["patterns"]
        access_counts = data.get("access_counts", np.zeros(patterns.shape[1]))
        self.facts = list(facts)
        self.access_counts = access_counts.tolist()
        self.network.patterns = [patterns[:, i] for i in range(patterns.shape[1])]
        self.network._X = None


class ColdStore:
    """Approximate nearest-neighbor store on disk (FAISS or numpy fallback)."""

    def __init__(self, dim: int, path: str):
        self.dim = dim
        self.path = path
        self._use_faiss = False
        self._index = None
        self._patterns: List[np.ndarray] = []
        self.facts: List[str] = []
        try:
            import faiss
            self._index = faiss.IndexFlatIP(dim)
            self._use_faiss = True
        except ImportError:
            pass

    @property
    def size(self) -> int:
        return len(self.facts)

    def store(self, fact: str, vec: np.ndarray) -> int:
        vec_2d = vec.reshape(1, -1).astype(np.float32)
        if self._use_faiss:
            self._index.add(vec_2d)
        self._patterns.append(vec.copy())
        self.facts.append(fact)
        return len(self.facts) - 1

    def store_batch(self, items: List[Tuple[str, np.ndarray]]) -> None:
        for fact, vec in items:
            self.store(fact, vec)

    def search(self, query_vec: np.ndarray, top_k: int = 10) -> List[Tuple[int, float]]:
        if not self.facts:
            return []
        if self._use_faiss:
            query_2d = query_vec.reshape(1, -1).astype(np.float32)
            k = min(top_k, len(self.facts))
            distances, indices = self._index.search(query_2d, k)
            return [(int(indices[0][i]), float(distances[0][i])) for i in range(k)]
        X = np.column_stack(self._patterns)
        similarities = X.T @ query_vec
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [(int(idx), float(similarities[idx])) for idx in top_indices]

    def get_patterns(self, indices: List[int]) -> List[np.ndarray]:
        return [self._patterns[i] for i in indices if i < len(self._patterns)]

    def save(self) -> None:
        meta_path = self.path + ".meta.json"
        patterns_path = self.path + ".patterns.npy"
        with open(meta_path, "w") as f:
            json.dump({"facts": self.facts, "dim": self.dim}, f)
        if self._patterns:
            np.save(patterns_path, np.column_stack(self._patterns))
        if self._use_faiss:
            import faiss
            faiss.write_index(self._index, self.path + ".faiss")

    def load(self) -> None:
        meta_path = self.path + ".meta.json"
        patterns_path = self.path + ".patterns.npy"
        if not os.path.exists(meta_path):
            return
        with open(meta_path, "r") as f:
            meta = json.load(f)
        self.facts = meta["facts"]
        if os.path.exists(patterns_path):
            data = np.load(patterns_path)
            self._patterns = [data[:, i] for i in range(data.shape[1])]
        if self._use_faiss:
            faiss_path = self.path + ".faiss"
            if os.path.exists(faiss_path):
                import faiss
                self._index = faiss.read_index(faiss_path)
            elif self._patterns:
                import faiss
                self._index = faiss.IndexFlatIP(self.dim)
                vecs = np.vstack(self._patterns).astype(np.float32)
                self._index.add(vecs)


class TieredMemory:
    """Unified memory with hot (exact Hopfield) and cold (ANN) tiers."""

    def __init__(
        self,
        encoder: Optional[Encoder] = None,
        beta: float = 10.0,
        adaptive_beta: bool = True,
        max_hot: int = 5000,
        cold_path: Optional[str] = None,
        confidence_threshold: float = 0.5,
        cold_top_k: int = 20,
    ):
        self.encoder = encoder or auto_encoder()
        dim = self.encoder.dim
        self.max_hot = max_hot
        self.confidence_threshold = confidence_threshold
        self.cold_top_k = cold_top_k
        self.hot = HotStore(dim=dim, beta=beta, adaptive_beta=adaptive_beta)
        self.cold: Optional[ColdStore] = None
        if cold_path:
            self.cold = ColdStore(dim=dim, path=cold_path)

    def store(self, fact: str) -> int:
        vec = self.encoder.encode(fact)
        idx = self.hot.store(fact, vec)
        if self.hot.size > self.max_hot:
            self._evict_to_cold(self.hot.size - self.max_hot)
        return idx

    def _evict_to_cold(self, count: int) -> None:
        if self.cold is None:
            warnings.warn(
                f"Hot store exceeded max_hot ({self.max_hot}) but no cold store configured.",
                stacklevel=3,
            )
            return
        evicted = self.hot.evict_least_accessed(count)
        self.cold.store_batch(evicted)

    def retrieve(self, query: str, top_k: int = 3) -> List[RetrievalResult]:
        query_vec = self.encoder.encode(query)
        hot_results = self.hot.retrieve(query_vec, top_k=top_k)
        top_confidence = hot_results[0].confidence if hot_results else 0.0
        if top_confidence >= self.confidence_threshold or self.cold is None:
            return hot_results[:top_k]
        cold_candidates = self.cold.search(query_vec, top_k=self.cold_top_k)
        if not cold_candidates:
            return hot_results[:top_k]
        cold_indices = [idx for idx, _ in cold_candidates]
        cold_patterns = self.cold.get_patterns(cold_indices)
        temp_net = ModernHopfieldNetwork(
            dim=self.encoder.dim,
            beta=self.hot.network.base_beta,
            adaptive_beta=self.hot.network.adaptive_beta,
        )
        temp_facts = []
        for i, ci in enumerate(cold_indices):
            if i < len(cold_patterns):
                temp_net.store(cold_patterns[i])
                temp_facts.append(self.cold.facts[ci])
        if temp_net.num_patterns == 0:
            return hot_results[:top_k]
        _, cold_weights = temp_net.retrieve(query_vec)
        cold_results = []
        for i, w in enumerate(cold_weights):
            cold_results.append(RetrievalResult(
                fact=temp_facts[i], confidence=float(w), source="cold",
                index=cold_indices[i] if i < len(cold_indices) else -1,
            ))
        all_results = hot_results + cold_results
        all_results.sort(key=lambda r: -r.confidence)
        return all_results[:top_k]

    def query(self, question: str) -> str:
        results = self.retrieve(question, top_k=1)
        if not results:
            return "[No facts stored]"
        return results[0].fact

    @property
    def num_facts(self) -> int:
        total = self.hot.size
        if self.cold:
            total += self.cold.size
        return total

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "facts.json"), "w") as f:
            json.dump({"hot_facts": self.hot.facts}, f)
        self.hot.save(os.path.join(directory, "hot.npz"))
        if self.cold:
            self.cold.path = os.path.join(directory, "cold")
            self.cold.save()

    def load(self, directory: str) -> None:
        facts_path = os.path.join(directory, "facts.json")
        if os.path.exists(facts_path):
            with open(facts_path, "r") as f:
                data = json.load(f)
            hot_facts = data.get("hot_facts", [])
        else:
            hot_facts = []
        hot_path = os.path.join(directory, "hot.npz")
        if os.path.exists(hot_path):
            self.hot.load_patterns(hot_path, hot_facts)
        if self.cold:
            self.cold.path = os.path.join(directory, "cold")
            self.cold.load()
