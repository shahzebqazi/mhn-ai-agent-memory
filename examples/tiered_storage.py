#!/usr/bin/env python3
"""Tiered storage demo: hot/cold with eviction."""

import os
import tempfile
from hopfield_memory import TieredMemory, RandomIndexEncoder

enc = RandomIndexEncoder(dim=256)
mem = TieredMemory(
    encoder=enc, beta=10.0, max_hot=3,
    cold_path=os.path.join(tempfile.gettempdir(), "hopfield_demo_cold"),
)

facts = [
    "The speed of light is 299792458 meters per second",
    "Water freezes at zero degrees Celsius",
    "The Earth is the third planet from the Sun",
    "DNA carries genetic information in all living organisms",
    "Gravity accelerates objects at 9.8 meters per second squared",
]

for f in facts:
    mem.store(f)

print(f"Total facts: {mem.num_facts}")
print(f"Hot store:   {mem.hot.size}")
print(f"Cold store:  {mem.cold.size if mem.cold else 0}")
print()
print(f"Query 'speed light meters' -> {mem.query('speed light meters')}")
print(f"Query 'DNA genetic'        -> {mem.query('DNA genetic')}")
