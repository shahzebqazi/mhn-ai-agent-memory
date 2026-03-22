---
name: hopfield-mcp-setup
description: Install the Hopfield MCP server venv and wire Cursor MCP + shared working-memory file for this repo
---

# Hopfield MCP + shared working memory

## 1. Server venv (from repo root)

```bash
cd mcp-server
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

For semantic encoding (recommended when `HOPFIELD_ENCODER` is `sentence_transformer`, as in `.cursor/mcp.json.example`):

```bash
# From repo root, install extras into the **mcp-server venv** (not system Python)
./mcp-server/.venv/bin/pip install -e ".[semantic]"
```

If you have not created the venv yet, run step 1 first, then the command above.

## 2. Cursor MCP config

Copy `.cursor/mcp.json.example` to `.cursor/mcp.json` at the **repository root**.

Edit:

- If your Cursor build does not expand `${workspaceFolder}`, replace those segments with **absolute** paths to this repo.
- `command` → absolute path to `mcp-server/.venv/bin/python` (or your interpreter with `mcp` installed).
- `args` → absolute path to `mcp-server/server.py`.
- `HOPFIELD_STATE_PATH` → absolute path to `.mhn/working-memory.json` under this repo (parent dir is gitignored).
- Set `HOPFIELD_AUTO_SAVE` to `true` for automatic persistence after each `store`.
- Set `HOPFIELD_ENCODER` to `sentence_transformer` when `mhn-ai-agent-memory[semantic]` is installed.

Reload MCP servers or restart Cursor.

## 3. Verify

In chat, rely on the **mhn-project-working-memory** skill and ask the agent to call MCP tool **`working_memory_status`**. You should see your `state_path`, `auto_save`, and `num_facts`.

## 4. Optional local plugin install

To load the bundled skill/command from disk: in Cursor, add a **local plugin** pointing at the `.cursor-plugin` directory in this repository (see Cursor docs for “local plugin” / marketplace dev workflow).
