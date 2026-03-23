# Hopfield Memory MCP server

This Python [Model Context Protocol](https://modelcontextprotocol.io/) server exposes [HopfieldMemory](../src/hopfield_memory/memory.py) to agents. It supports storage, retrieval, repulsive patterns, and persistence.

## Requirements

- Python 3.10+ (required by the `mcp` package)
- The parent package `mhn-ai-agent-memory` checked out beside this folder (the server adds `../src` to `sys.path` at startup)

Optional encoders, selected with `HOPFIELD_ENCODER`:

- `sentence_transformer` — `pip install -e ..[semantic]` from the repo root
- `tfidf` — `pip install -e ..[tfidf]`
- `openai` — `pip install -e ..[openai]` and `OPENAI_API_KEY`

## Installation

From `mcp-server/`:

```bash
cd mcp-server
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .
```

`numpy` is installed as a dependency. `server.py` handles the parent repo import path automatically.

## Environment variables

| Variable | Default | Meaning |
|----------|---------|---------|
| `HOPFIELD_DIM` | `512` | Dimension for `RandomIndexEncoder` (when `HOPFIELD_ENCODER=random`) |
| `HOPFIELD_BETA` | `10.0` | Hopfield inverse temperature |
| `HOPFIELD_REPULSIVE` | `false` | Enable repulsive / contrastive backend (`true` / `1` / `yes`) |
| `HOPFIELD_ENCODER` | `random` | `random`, `auto`, `sentence_transformer`, `tfidf`, `openai` |
| `HOPFIELD_ST_MODEL` | `all-MiniLM-L6-v2` | SentenceTransformer model name when using `sentence_transformer` |
| `HOPFIELD_STATE_PATH` | *(unset)* | If set, load this file on server startup when it exists (shared disk-backed working memory for multiple agents / sessions) |
| `HOPFIELD_AUTO_SAVE` | `false` | If `true`/`1`/`yes`, persist to `HOPFIELD_STATE_PATH` after each `store` / `store_negative` (requires `HOPFIELD_STATE_PATH`) |

## Cursor

Add a server entry to your MCP config, either in `~/.cursor/mcp.json` or the project-level `.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "hopfield-memory": {
      "command": "/absolute/path/to/mhn-ai-agent-memory/mcp-server/.venv/bin/python",
      "args": ["/absolute/path/to/mhn-ai-agent-memory/mcp-server/server.py"],
      "env": {
        "HOPFIELD_DIM": "512",
        "HOPFIELD_BETA": "10.0",
        "HOPFIELD_REPULSIVE": "false",
        "HOPFIELD_STATE_PATH": "/absolute/path/to/your/project/.mhn/working-memory.json",
        "HOPFIELD_AUTO_SAVE": "true"
      }
    }
  }
}
```

Use the same Python interpreter where you installed `mcp`, whether that is the venv above or another environment where you ran `pip install -e .` inside `mcp-server`.

## Claude Code

In Claude Code, register an MCP server with the same **command**, **args**, and **env** as in the Cursor example. In the Claude Desktop or Claude Code MCP UI, that means a `stdio` server using `python` and the path to `server.py`. Paths must be absolute on your machine.

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
| `list_facts` | → all stored fact strings (browse / audit shared working memory) |
| `working_memory_status` | → `state_path`, `state_path_exists`, `auto_save`, `num_facts`, `encoder` |

## Example agent flow

1. **`store`** — store "User’s favorite color is teal."
2. **`retrieve`** — query "what color do they like?" with `top_k=3` to inspect ranked matches and attention weights.
3. **`query_or_none`** — ask the same question when you want either one fact or `null`.
4. **`save`** — write to `/tmp/agent_memory.json` before shutdown, then **`load`** that path in the next session.

**Shared project working memory:** set `HOPFIELD_STATE_PATH` to a JSON file inside the repo, for example `.mhn/working-memory.json`, and set `HOPFIELD_AUTO_SAVE=true`. Any Cursor agent using the same MCP config will read and write the same file, so the memory stays with the project instead of one chat. Use **`list_facts`** to inspect stored facts and **`working_memory_status`** to confirm the active path and encoder.

Run manually (stdio MCP transport):

```bash
source .venv/bin/activate
python server.py
```

Or via the installed console script:

```bash
hopfield-memory-mcp
```
