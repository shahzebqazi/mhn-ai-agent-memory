"""
Multi-hop retrieval for Hopfield Memory.

Chains multiple single-hop queries by augmenting the query with retrieved
context. Each hop shifts the energy landscape so related facts become
the new nearest attractor.

Example:
    Store: "Alice lives in France"
    Store: "The capital of France is Paris"
    chain_query(mem, "capital of the country where Alice lives")
    -> ["Alice lives in France", "The capital of France is Paris"]
"""

from typing import List, Tuple


def chain_query(
    memory,
    question: str,
    max_hops: int = 3,
) -> List[str]:
    """Retrieve facts by iteratively augmenting the query.

    After each hop, the retrieved fact is appended to the query so that
    the next Hopfield retrieval is biased toward facts related to the
    full chain of retrieved context.

    Returns the list of retrieved facts in hop order.
    """
    retrieved: List[str] = []
    current_query = question

    for _ in range(max_hops):
        fact = memory.query(current_query)
        if fact in retrieved or fact == "[No facts stored]":
            break
        retrieved.append(fact)
        current_query = question + " " + " ".join(retrieved)

    return retrieved


def chain_query_with_confidence(
    memory,
    question: str,
    max_hops: int = 3,
    min_confidence: float = 0.1,
) -> List[Tuple[str, float]]:
    """Like chain_query but returns (fact, confidence) tuples.

    Stops early if confidence drops below min_confidence, indicating
    the query has drifted too far from any stored attractor.
    """
    retrieved: List[Tuple[str, float]] = []
    seen_facts: set = set()
    current_query = question

    for _ in range(max_hops):
        results = memory.retrieve(current_query, top_k=1)
        if not results:
            break

        fact, confidence = results[0]
        if fact in seen_facts:
            break
        if confidence < min_confidence:
            break

        retrieved.append((fact, confidence))
        seen_facts.add(fact)
        current_query = question + " " + " ".join(f for f, _ in retrieved)

    return retrieved
