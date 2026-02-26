This repo uses the `strands` framework to build an OpenSearch semantic search solution architect agent. The agent collects user requirements and gives recommendations for index types.

There are two ways to use the agent: as a **standalone interactive CLI** or via an **MCP server** that any MCP-compatible client can drive.

## Standalone Agent

Start the interactive orchestrator in a terminal:

```bash
python opensearch_orchestrator/orchestrator.py
```

The orchestrator guides you through sample collection, requirements gathering, solution planning, and execution — all in one interactive session.

## MCP Server (Cursor, Claude Desktop, etc.)

The MCP server exposes the same orchestrator workflow as a set of phase tools. A client LLM drives the conversation with the user and calls the tools in order.

### Prerequisites

Install [uv](https://docs.astral.sh/uv/) (one-time, no sudo needed):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Running manually

```bash
uv run opensearch_orchestrator/mcp_server.py
```

`uv` reads the inline script metadata in `opensearch_orchestrator/mcp_server.py` and auto-installs dependencies into a cached virtual environment.

### Running from PyPI (`uvx`)

After publishing to PyPI, run the MCP server without cloning the repo:

```bash
uvx opensearch-orchestrator@latest
```

If you install via `pip`, you can also run:

```bash
opensearch-orchestrator
```

Important: this command starts a stdio MCP server (JSON-RPC), not an interactive CLI. It should be launched by an MCP client such as Cursor, Claude Desktop, or MCP Inspector. If you want an interactive terminal workflow, run:

```bash
python opensearch_orchestrator/orchestrator.py
```

### MCP workflow tools

The server exposes high-level phase tools that mirror the standalone orchestrator workflow:

| Tool | Phase | Description |
|------|-------|-------------|
| `load_sample` | 1 | Load a sample document (built-in, file, URL, index, or paste) |
| `set_preferences` | 2 | Set budget, performance, query pattern, deployment preferences |
| `start_planning` | 3 | Start the planning agent; returns initial architecture proposal |
| `refine_plan` | 3 | Send user feedback to refine the proposal |
| `finalize_plan` | 3 | Finalize the plan when the user confirms |
| `execute_plan` | 4 | Execute the plan (create index, models, pipelines, UI) |
| `retry_execution` | 4 | Resume from a failed execution step |
| `cleanup_verification` | Post | Remove test documents on user request |

Low-level domain tools (`create_index`, `submit_sample_doc`, etc.) are also exposed for advanced use.

### Cursor integration

1. Add the following to `.cursor/mcp.json` in your workspace (adjust `cwd` to the repo path):

```json
{
  "mcpServers": {
    "opensearch-orchestrator": {
      "command": "uv",
      "args": ["run", "opensearch_orchestrator/mcp_server.py"],
      "cwd": "/path/to/agent-poc"
    }
  }
}
```

2. Reload the Cursor window (`Cmd+Shift+P` → "Developer: Reload Window"), then enable the server in **Cursor Settings → MCP**.

3. A Cursor rule at `.cursor/rules/opensearch-workflow.mdc` auto-activates when you ask about OpenSearch solution design and teaches the LLM the tool sequence.

If Cursor cannot find `uv` on its PATH, use the absolute path (e.g. `~/.local/bin/uv`).

### Claude Desktop integration

1. Copy `claude_desktop_config.example.json` to your Claude Desktop config directory:

   - **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
   - **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

2. Edit the `cwd` path to point to this repo.

3. Restart Claude Desktop. The `opensearch_workflow` prompt is available in the prompt picker and describes the full tool sequence.

### Generic MCP clients

Any MCP-compatible client can connect via stdio and discover tools with `tools/list`. The `opensearch_workflow` prompt (available via `prompts/list`) describes the workflow. Tool docstrings also include prerequisite hints.

### Without uv

If you prefer not to install `uv`, install dependencies manually and use Python directly:

```bash
pip install mcp opensearch-py
```

```json
{
  "mcpServers": {
    "opensearch-orchestrator": {
      "command": "python3",
      "args": ["opensearch_orchestrator/mcp_server.py"],
      "cwd": "/path/to/agent-poc"
    }
  }
}
```

## Release checklist

Build and validate before publishing:

```bash
uv run pytest -q
uv build
python -m zipfile -l dist/*.whl
python -c "import opensearch_orchestrator.mcp_server as m; print(hasattr(m, 'main'))"
uvx --from dist/*.whl opensearch-orchestrator

# Upload to PyPI (needs a PyPI account + API token)
uv publish --token pypi-YOUR-TOKEN
```

Then publish to TestPyPI for smoke tests, followed by PyPI.
