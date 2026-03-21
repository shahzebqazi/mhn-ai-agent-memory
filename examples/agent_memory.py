#!/usr/bin/env python3
"""Cross-process agent memory demo.

1. Stores facts including one target memory, serializes to disk.
2. Loads the serialized network and attempts recall with partial cues.
"""

import os
import tempfile
from hopfield_memory import HopfieldMemory, RandomIndexEncoder

MEMORY_FILE = os.path.join(tempfile.gettempdir(), "hopfield_agent_demo.json")

def store_phase():
    mem = HopfieldMemory(encoder=RandomIndexEncoder(dim=512), beta=12.0)
    mem.store("The boiling point of liquid nitrogen is minus 196 degrees Celsius")
    mem.store("Johannes Gutenberg invented the movable type printing press around 1440")
    mem.store("The seventeenth moon of Neptune is named Hippocamp and was discovered in 2013 by Mark Showalter using Hubble images")
    mem.save(MEMORY_FILE)
    print(f"Stored {mem.num_facts} facts -> {MEMORY_FILE}")

def recall_phase():
    mem = HopfieldMemory.load(MEMORY_FILE)
    print(f"Loaded {mem.num_facts} facts")
    result = mem.query("Neptune moon Hippocamp Showalter 2013")
    print(f"Query:    'Neptune moon Hippocamp Showalter 2013'")
    print(f"Recalled: {result}")

if __name__ == "__main__":
    store_phase()
    recall_phase()
