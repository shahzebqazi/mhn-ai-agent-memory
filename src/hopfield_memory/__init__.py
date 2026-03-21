"""
hopfield-memory: AI agent memory using Modern Hopfield Networks.

.. code-block:: python

    from hopfield_memory import HopfieldMemory

    mem = HopfieldMemory()
    mem.store("Alice is a mathematician")
    result = mem.query("math topology")
"""

__version__ = "0.3.0"

from hopfield_memory.network import ModernHopfieldNetwork
from hopfield_memory.repulsive import RepulsiveMHN, EnergyComponents
from hopfield_memory.memory import HopfieldMemory, TextEncoder
from hopfield_memory.encoders import (
    Encoder,
    RandomIndexEncoder,
    TFIDFEncoder,
    SentenceTransformerEncoder,
    OpenAIEncoder,
    auto_encoder,
)
from hopfield_memory.contradiction import (
    ContradictionDetector,
    ConflictResult,
    check_and_store,
)
from hopfield_memory.multihop import chain_query, chain_query_with_confidence
from hopfield_memory.tiered import TieredMemory, HotStore, ColdStore, RetrievalResult
from hopfield_memory.presets import small_memory, medium_memory, large_memory, massive_memory

__all__ = [
    "ModernHopfieldNetwork",
    "RepulsiveMHN",
    "EnergyComponents",
    "HopfieldMemory",
    "TextEncoder",
    "Encoder",
    "RandomIndexEncoder",
    "TFIDFEncoder",
    "SentenceTransformerEncoder",
    "OpenAIEncoder",
    "auto_encoder",
    "ContradictionDetector",
    "ConflictResult",
    "check_and_store",
    "chain_query",
    "chain_query_with_confidence",
    "TieredMemory",
    "HotStore",
    "ColdStore",
    "RetrievalResult",
    "small_memory",
    "medium_memory",
    "large_memory",
    "massive_memory",
]
