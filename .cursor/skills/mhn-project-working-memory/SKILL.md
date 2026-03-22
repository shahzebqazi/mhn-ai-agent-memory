---
name: mhn-project-working-memory
description: Uses Modern Hopfield (MHN) MCP tools for on-disk project working memory shared across Cursor agents and sessions. Use when the user wants agent memory, working memory, a local knowledge base, facts persisted in the repo, or handoff between agents without re-explaining context.
---

# MHN project working memory (Cursor)

## Idea

Associative memory lives in a **JSON file on disk** (not in chat). Any agent with the same MCP config reads and updates the same store, so memory is **swappable between agents** and survives new conversations.

## Setup (once per machine)

1. Create the MCP server venv: see `mcp-server/README.md` (`pip install -e .` inside `mcp-server/`).
2. Copy `.cursor/mcp.json.example` to `.cursor/mcp.json` in this repo (or merge the `hopfield-memory` entry into user-level MCP config).
3. Replace placeholder paths with **absolute** paths to `mcp-server/.venv/bin/python` and `mcp-server/server.py`, or use `${workspaceFolder}` if your Cursor build expands it.
4. Set `HOPFIELD_STATE_PATH` to a path **inside the project** (default example: `.mhn/working-memory.json`). Enable `HOPFIELD_AUTO_SAVE=true` so `store` / `store_negative` persist immediately.
5. Restart MCP / Cursor so the server picks up env vars.

Use `sentence_transformer` (or better encoders) for semantic recall; `random` is fine only for exact-token overlap demos.

## Agent workflow

1. **`working_memory_status`** — Confirm `state_path`, `auto_save`, and `num_facts` before relying on memory.
2. **`list_facts`** — Treat as a cheap “table scan” of the local knowledge base (all stored strings).
3. **`retrieve`** — Ranked facts with weights for a cue (like a fuzzy DB query).
4. **`query_or_none`** — Prefer when the agent must distinguish “found in memory” vs “nothing relevant” (set `min_similarity` if needed).
5. **`store`** — Persist decisions, constraints, file paths, API shapes, user prefs, open questions. Keep facts **short and atomic** when possible.
6. **`save` / `load`** — Optional explicit snapshots to other paths (e.g. backup or experiment branch).

## Rules

- Do **not** store secrets (tokens, keys); memory files may be copied or logged.
- After changing `HOPFIELD_ENCODER`, treat old state files as incompatible unless you know encoders match.
- Prefer **`query_or_none`** over **`query`** when hallucinating a wrong “best fact” is worse than admitting no match.

## Marketplace plugin

Install the local Cursor plugin from `.cursor-plugin/` (see that folder’s `plugin.json`) to ship this skill with the repo; submit the same tree to the Cursor plugin marketplace when ready.
