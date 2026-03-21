"""
Contradiction detection for Hopfield Memory.

Detects when a newly stored fact conflicts with an existing one by
combining vector similarity (from the Hopfield network itself) with
structural token-overlap heuristics. No LLM calls needed.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from hopfield_memory.encoders import Encoder


@dataclass
class ConflictResult:
    """Result of a contradiction check."""
    has_conflict: bool
    new_fact: str
    conflicting_facts: List[Tuple[str, float]] = field(default_factory=list)
    explanation: str = ""


def _tokenize(text: str) -> List[str]:
    return text.lower().replace(",", " ").replace(".", " ").replace("?", " ").split()


def _overlap_ratio(tokens_a: List[str], tokens_b: List[str]) -> float:
    if not tokens_a or not tokens_b:
        return 0.0
    set_a = set(tokens_a)
    set_b = set(tokens_b)
    intersection = set_a & set_b
    return len(intersection) / min(len(set_a), len(set_b))


def _structural_conflict(fact_a: str, fact_b: str, subject_len: int = 5) -> bool:
    """Check if two facts share a subject but diverge on the predicate/object.

    Uses subject_len=5 to include linking verbs (e.g. "is") in the subject
    portion, so "The capital of France is Paris" vs "...is Lyon" correctly
    flags the differing noun as a conflict.
    """
    tokens_a = _tokenize(fact_a)
    tokens_b = _tokenize(fact_b)

    subj_a = tokens_a[:subject_len]
    subj_b = tokens_b[:subject_len]
    pred_a = tokens_a[subject_len:]
    pred_b = tokens_b[subject_len:]

    subject_overlap = _overlap_ratio(subj_a, subj_b)
    predicate_overlap = _overlap_ratio(pred_a, pred_b) if (pred_a and pred_b) else 1.0

    return subject_overlap > 0.6 and predicate_overlap < 0.4


class ContradictionDetector:
    """Detect contradictions when storing new facts.

    Uses cosine similarity from the Hopfield network patterns to find
    candidates, then applies structural heuristics to distinguish
    updates/contradictions from related-but-compatible facts.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.80,
        top_k: int = 3,
    ):
        self.similarity_threshold = similarity_threshold
        self.top_k = top_k

    def check(
        self,
        new_fact: str,
        new_vec: np.ndarray,
        existing_facts: List[str],
        existing_patterns: List[np.ndarray],
    ) -> ConflictResult:
        """Check if new_fact contradicts any existing fact."""
        if not existing_patterns:
            return ConflictResult(has_conflict=False, new_fact=new_fact)

        X = np.column_stack(existing_patterns)
        similarities = X.T @ new_vec

        top_indices = np.argsort(similarities)[::-1][:self.top_k]

        conflicts = []
        for idx in top_indices:
            sim = float(similarities[idx])
            if sim < self.similarity_threshold:
                continue

            old_fact = existing_facts[idx]
            if _structural_conflict(old_fact, new_fact):
                conflicts.append((old_fact, sim))

        if not conflicts:
            return ConflictResult(has_conflict=False, new_fact=new_fact)

        explanations = []
        for old_fact, sim in conflicts:
            explanations.append(f"  similarity={sim:.3f}: {old_fact!r}")

        return ConflictResult(
            has_conflict=True,
            new_fact=new_fact,
            conflicting_facts=conflicts,
            explanation=(
                f"New fact {new_fact!r} may contradict:\n"
                + "\n".join(explanations)
            ),
        )


def check_and_store(
    memory,
    fact: str,
    detector: Optional[ContradictionDetector] = None,
    auto_resolve: bool = False,
) -> Tuple[int, Optional[ConflictResult]]:
    """Store a fact with optional contradiction checking.

    If *auto_resolve* is True and a contradiction is found, the old
    conflicting fact's pattern is replaced with the new one.

    Returns ``(index, conflict_result_or_None)``.
    """
    if detector is None:
        detector = ContradictionDetector()

    vec = memory.encoder.encode(fact)
    result = detector.check(
        new_fact=fact,
        new_vec=vec,
        existing_facts=memory.facts,
        existing_patterns=memory.network.patterns,
    )

    if result.has_conflict and not auto_resolve:
        return -1, result

    if result.has_conflict and auto_resolve:
        for old_fact, _ in result.conflicting_facts:
            try:
                old_idx = memory.facts.index(old_fact)
                memory.network.patterns[old_idx] = vec.copy()
                memory.network._X = None
                memory.facts[old_idx] = fact
                return old_idx, result
            except ValueError:
                continue

    idx = memory.store(fact)
    return idx, result
