import os
import sys
from pathlib import Path
from typing import Any, List, Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from mcp.server.fastmcp import FastMCP

from hopfield_memory import (
    HopfieldMemory,
    OpenAIEncoder,
    RandomIndexEncoder,
    SentenceTransformerEncoder,
    TFIDFEncoder,
    auto_encoder,
)

mcp = FastMCP("hopfield-memory")


def _env_bool(key: str, default: bool) -> bool:
    raw = os.environ.get(key)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _make_encoder() -> Any:
    kind = os.environ.get("HOPFIELD_ENCODER", "random").strip().lower()
    dim = int(os.environ.get("HOPFIELD_DIM", "512"))
    if kind in ("", "random", "random_index", "rix"):
        return RandomIndexEncoder(dim=dim)
    if kind == "auto":
        return auto_encoder(preferred_dim=dim)
    if kind in ("sentence_transformer", "st", "minilm"):
        model = os.environ.get("HOPFIELD_ST_MODEL", "all-MiniLM-L6-v2")
        return SentenceTransformerEncoder(model_name=model)
    if kind == "tfidf":
        return TFIDFEncoder(dim=dim)
    if kind == "openai":
        return OpenAIEncoder()
    raise ValueError(f"Unknown HOPFIELD_ENCODER: {kind!r}")


def _build_memory() -> HopfieldMemory:
    beta = float(os.environ.get("HOPFIELD_BETA", "10.0"))
    repulsive = _env_bool("HOPFIELD_REPULSIVE", False)
    return HopfieldMemory(encoder=_make_encoder(), beta=beta, repulsive=repulsive)


_memory: HopfieldMemory = _build_memory()


@mcp.tool()
def store(fact: str) -> int:
    return _memory.store(fact)


@mcp.tool()
def retrieve(query: str, top_k: int = 3) -> List[List[Any]]:
    pairs = _memory.retrieve(query, top_k=top_k)
    return [[fact, weight] for fact, weight in pairs]


@mcp.tool()
def query(question: str) -> str:
    return _memory.query(question)


@mcp.tool()
def query_or_none(question: str, min_similarity: float = 0.25) -> Optional[str]:
    return _memory.query_or_none(question, min_similarity=min_similarity)


@mcp.tool()
def store_negative(fact: str) -> int:
    return _memory.store_negative(fact)


@mcp.tool()
def match_quality(query: str) -> dict:
    return _memory.match_quality(query)


@mcp.tool()
def save(path: str) -> dict:
    _memory.save(path)
    return {"path": path, "num_facts": _memory.num_facts}


@mcp.tool()
def load(path: str) -> dict:
    global _memory
    _memory = HopfieldMemory.load(path, encoder=_make_encoder())
    return {"path": path, "num_facts": _memory.num_facts}


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
