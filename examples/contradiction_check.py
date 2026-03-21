#!/usr/bin/env python3
"""Contradiction detection demo."""

from hopfield_memory import HopfieldMemory, RandomIndexEncoder, ContradictionDetector, check_and_store

enc = RandomIndexEncoder(dim=512)
mem = HopfieldMemory(encoder=enc, beta=10.0)
detector = ContradictionDetector(similarity_threshold=0.60)

mem.store("The capital of France is Paris")
mem.store("Water boils at 100 degrees Celsius")

print("Stored 2 facts. Now storing a potentially conflicting fact...")
print()

idx, conflict = check_and_store(
    mem, "The capital of France is Lyon",
    detector=detector, auto_resolve=False,
)

if conflict and conflict.has_conflict:
    print("Contradiction detected!")
    print(conflict.explanation)
else:
    print(f"No contradiction detected (stored at index {idx})")
    print("(With a semantic encoder like sentence-transformers, this would likely be caught)")
