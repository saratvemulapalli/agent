This repo uses the `strands` framework to build an OpenSearch semantic search solution architect agent. The agent collects user requirements and gives recommendations for index types.

## Standalone Agent

Start the interactive orchestrator:

```bash
python orchestrator.py
```

## MCP Server

The same domain tools are also available as an MCP server so any MCP-compatible client (Cursor, Claude Desktop, etc.) can use them.

### Prerequisites

Install [uv](https://docs.astral.sh/uv/) (one-time, no sudo needed):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Running manually

```bash
uv run mcp_server.py
```

`uv` reads the inline script metadata in `mcp_server.py` and auto-installs `mcp` and `opensearch-py` into a cached virtual environment.

### Cursor integration

Add the following to `.cursor/mcp.json` in your workspace (adjust `cwd` to the repo path):

```json
{
  "mcpServers": {
    "opensearch-agent": {
      "command": "uv",
      "args": ["run", "mcp_server.py"],
      "cwd": "/path/to/agent-poc"
    }
  }
}
```

If Cursor cannot find `uv` on its PATH, use the absolute path (e.g. `~/.local/bin/uv`).

After saving, reload the Cursor window (`Cmd+Shift+P` → "Developer: Reload Window"), then enable the server in **Cursor Settings → MCP**.

### Without uv

If you prefer not to install `uv`, install dependencies manually and use Python directly:

```bash
pip install mcp opensearch-py
```

```json
{
  "mcpServers": {
    "opensearch-agent": {
      "command": "python3",
      "args": ["mcp_server.py"],
      "cwd": "/path/to/agent-poc"
    }
  }
}
```
