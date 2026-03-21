#!/usr/bin/env python3
"""Basic usage example for hopfield-memory."""

from hopfield_memory import HopfieldMemory, RandomIndexEncoder

mem = HopfieldMemory(encoder=RandomIndexEncoder(dim=512), beta=10.0)

mem.store("Alice is a mathematician who studies topology and algebraic geometry")
mem.store("Bob is a painter who works with oil on canvas landscapes")
mem.store("Carol is a physicist researching quantum entanglement at CERN")

fact, confidence = mem.query_with_confidence("topology algebraic geometry")
print(f"Query:      'topology algebraic geometry'")
print(f"Retrieved:  {fact}")
print(f"Confidence: {confidence:.4f}")
print()

for q in ["painting oil canvas", "quantum physics entanglement"]:
    print(f"Query: {q!r}  ->  {mem.query(q)}")
