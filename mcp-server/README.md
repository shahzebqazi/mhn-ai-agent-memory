# Hopfield Memory MCP server

Python [Model Context Protocol](https://modelcontextprotocol.io/) server that exposes [HopfieldMemory](../src/hopfield_memory/memory.py) to agents (store, retrieve, query, repulsive patterns, persistence).

## Requirements

- Python 3.10+ (required by the `mcp` package)
- The parent package `mhn-ai-agent-memory` checked out beside this folder (this server adds `../src` to `sys.path` at startup)

Optional encoders (set `HOPFIELD_ENCODER`):

- `sentence_transformer` — `pip install -e ..[semantic]` from the repo root
- `tfidf` — `pip install -e ..[tfidf]`
- `openai` — `pip install -e ..[openai]` and `OPENAI_API_KEY`

## Installation

From this directory:

```bash
cd mcp-server
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .
```

`numpy` is pulled in as a dependency; HopfieldMemory itself only needs the parent repo on `sys.path` (handled by `server.py`).

## Environment variables

| Variable | Default | Meaning |
|----------|---------|---------|
| `HOPFIELD_DIM` | `512` | Dimension for `RandomIndexEncoder` (when `HOPFIELD_ENCODER=random`) |
| `HOPFIELD_BETA` | `10.0` | Hopfield inverse temperature |
| `HOPFIELD_REPULSIVE` | `false` | Enable repulsive / contrastive backend (`true` / `1` / `yes`) |
| `HOPFIELD_ENCODER` | `random` | `random`, `auto`, `sentence_transformer`, `tfidf`, `openai` |
| `HOPFIELD_ST_MODEL` | `all-MiniLM-L6-v2` | SentenceTransformer model name when using `sentence_transformer` |

## Cursor

Add a server entry to your MCP config (user-level `~/.cursor/mcp.json` or project `.cursor/mcp.json`):

```json
{
  "mcpServers": {
    "hopfield-memory": {
      "command": "/absolute/path/to/mhn-ai-agent-memory/mcp-server/.venv/bin/python",
      "args": ["/absolute/path/to/mhn-ai-agent-memory/mcp-server/server.py"],
      "env": {
        "HOPFIELD_DIM": "512",
        "HOPFIELD_BETA": "10.0",
        "HOPFIELD_REPULSIVE": "false"
      }
    }
  }
}
```

Use the same Python interpreter that has `mcp` installed (the venv shown above, or another environment where you ran `pip install -e .` in `mcp-server`).

## Claude Code

In Claude Code, register an MCP server with the same **command**, **args**, and **env** as in the Cursor example (Claude Desktop / Claude Code MCP UI: “stdio” server with `python` and path to `server.py`). Paths must be absolute on your machine.

Example `claude_desktop_config.json` (macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "hopfield-memory": {
      "command": "/absolute/path/to/mhn-ai-agent-memory/mcp-server/.venv/bin/python",
      "args": ["/absolute/path/to/mhn-ai-agent-memory/mcp-server/server.py"]
    }
  }
}
```

## Tools

| Tool | Description |
|------|-------------|
| `store` | `fact: str` → stored pattern index |
| `retrieve` | `query: str`, `top_k: int = 3` → `[[fact, weight], ...]` |
| `query` | `question: str` → best-matching fact string |
| `query_or_none` | `question: str`, `min_similarity: float = 0.25` → fact or null |
| `store_negative` | `fact: str` → index (or `-1` if not in repulsive mode) |
| `match_quality` | `query: str` → signal dict (`max_similarity`, `gap`, `is_match`, …) |
| `save` | `path: str` → `{"path", "num_facts"}` |
| `load` | `path: str` → reloads global memory; `{"path", "num_facts"}` |

## Example agent flow

1. **`store`** — “User’s favorite color is teal.”
2. **`retrieve`** with `query`: “what color do they like?” and `top_k`: `3` — ranked facts with attention weights.
3. **`query_or_none`** with the same question — returns the fact or `null` if similarity is below `min_similarity`.
4. **`save`** to `/tmp/agent_memory.json` before shutdown; on next session **`load`** that path to restore.

Run manually (stdio MCP transport):

```bash
source .venv/bin/activate
python server.py
```

Or via the installed console script:

```bash
hopfield-memory-mcp
```
