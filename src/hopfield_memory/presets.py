"""
Configurable presets for Hopfield Memory.

Factory functions that select appropriate dim, beta, hot capacity, and cold
store settings for different use cases.
"""

import os
import tempfile
from typing import Optional

from hopfield_memory.encoders import Encoder, RandomIndexEncoder, auto_encoder
from hopfield_memory.memory import HopfieldMemory
from hopfield_memory.tiered import TieredMemory


def small_memory(encoder: Optional[Encoder] = None) -> HopfieldMemory:
    """For tools, plugins, single-task agents. ~100 facts."""
    enc = encoder or RandomIndexEncoder(dim=256)
    return HopfieldMemory(encoder=enc, beta=12.0, adaptive_beta=True)


def medium_memory(encoder: Optional[Encoder] = None) -> HopfieldMemory:
    """For conversational agents, session memory. ~10k facts."""
    enc = encoder or auto_encoder(preferred_dim=384)
    return HopfieldMemory(encoder=enc, beta=10.0, adaptive_beta=True)


def large_memory(
    encoder: Optional[Encoder] = None,
    cold_path: Optional[str] = None,
) -> TieredMemory:
    """For knowledge bases, long-running agents. ~100k facts."""
    enc = encoder or auto_encoder(preferred_dim=512)
    path = cold_path or os.path.join(tempfile.gettempdir(), "hopfield_cold_large")
    return TieredMemory(
        encoder=enc, beta=8.0, adaptive_beta=True,
        max_hot=10000, cold_path=path, confidence_threshold=0.4, cold_top_k=30,
    )


def massive_memory(
    encoder: Optional[Encoder] = None,
    cold_path: Optional[str] = None,
) -> TieredMemory:
    """For millions of facts. FAISS-backed cold store recommended."""
    enc = encoder or auto_encoder(preferred_dim=768)
    path = cold_path or os.path.join(tempfile.gettempdir(), "hopfield_cold_massive")
    return TieredMemory(
        encoder=enc, beta=6.0, adaptive_beta=True,
        max_hot=10000, cold_path=path, confidence_threshold=0.3, cold_top_k=50,
    )
