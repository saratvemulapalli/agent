# OpenSearch Launchpad

An MCP-powered assistant that guides you from initial requirements to a running OpenSearch search setup. It collects a sample document, gathers preferences, plans a search architecture, and executes the plan — creating indices, ML models, ingest pipelines, and a local search UI — with optional deployment to Amazon OpenSearch Service or Serverless.

The package is built from **`opensearch_orchestrator`** and is distributed for:

| Platform | Location | Purpose |
|----------|----------|---------|
| **Kiro** | `kiro/opensearch-launchpad/` | Kiro Power (workflow + MCP config) |
| **Claude** | `.claude/skills/` → `../skills` | Claude Agent Skill (symlink) |
| **Cursor** | `cursor/` (template); plugin at `cursor/plugins/opensearch-launchpad/` | Cursor plugin (Agent Skill only; no MCP) |

See [cursor/README.md](cursor/README.md) and [cursor/plugins/opensearch-launchpad/README.md](cursor/plugins/opensearch-launchpad/README.md) for the Cursor plugin layout and installation.

---

## Kiro Power (Primary Integration)

OpenSearch Launchpad is packaged as a **Kiro Power**. Install it in Kiro by adding https://github.com/opensearch-project/opensearch-launchpad/tree/main/kiro/opensearch-launchpad as a power source. Kiro reads `POWER.md` for workflow instructions and calls MCP tools exposed by the server.

The `mcp.json` at the repo root runs `uvx opensearch-launchpad@latest` — no local clone required.

---

## Agent Skills (Claude Code, Cursor, Kiro)

OpenSearch Launchpad is also available as an **Agent Skill** that works across
multiple IDEs. The skill lives in `skills/opensearch-launchpad/` with symlinks
for each IDE's discovery path:

```
skills/opensearch-launchpad/              # Source of truth
.claude/skills -> ../skills               # Claude Code
.cursor/skills -> ../skills               # Cursor (project-level)
.kiro/skills   -> ../skills               # Kiro
cursor/plugins/opensearch-launchpad/skills -> ../../../skills   # Cursor plugin
```

### Claude Code

1. Clone the repo (the `.claude/skills` symlink is checked in):
   ```bash
   git clone https://github.com/opensearch-project/opensearch-launchpad.git
   cd opensearch-launchpad
   ```

2. Start Claude Code — the skill is auto-discovered from `.claude/skills/opensearch-launchpad/SKILL.md`.

The Agent Skill guides the IDE agent through the full workflow using scripts in
`skills/opensearch-launchpad/scripts/` — no custom MCP server needed. The IDE agent
runs the scripts directly (start OpenSearch, create indices, deploy models, etc.).

For AWS deployment (Phase 5), add these MCP servers to `.mcp.json`:
```json
{
  "mcpServers": {
    "opensearch-mcp-server": {
      "command": "uvx",
      "args": ["opensearch-mcp-server-py@latest"]
    },
    "aws-knowledge-mcp-server": {
      "command": "uvx",
      "args": ["fastmcp", "run", "https://knowledge-mcp.global.api.aws"]
    }
  }
}
```

### Cursor

1. Clone the repo (the `.cursor/skills` symlink is checked in):
   ```bash
   git clone https://github.com/opensearch-project/opensearch-launchpad.git
   cd opensearch-launchpad
   ```

2. Open the project in Cursor — the skill is discovered from `.cursor/skills/opensearch-launchpad/SKILL.md`.

Same as Claude Code: the agent follows the skill instructions and runs scripts
directly. **No MCP server needed** for the core workflow. For AWS deployment (Phase 5), add the same optional MCP servers to `.cursor/mcp.json` as in the Claude section above.

### Kiro

Kiro users can install the **Kiro Power** (see above) for the released, production
experience, which uses the `opensearch-launchpad` MCP server. The `.kiro/skills`
symlink is also available for Agent Skills support as Kiro adopts the standard.

---

## Standalone CLI (Local Development)

Start the interactive orchestrator in a terminal:

```bash
python opensearch_orchestrator/orchestrator.py
```

The orchestrator guides you through sample collection, requirements gathering, solution planning, and execution — all in one interactive session.

---

## MCP Server

The MCP server exposes the orchestrator workflow as a set of phase tools. Any MCP-compatible client (Claude Desktop, MCP Inspector, etc.) can drive the conversation.

### Prerequisites

Install [uv](https://docs.astral.sh/uv/) (one-time, no sudo needed):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Running from PyPI

```bash
uvx opensearch-launchpad@latest
```

If installed via `pip`:

```bash
opensearch-launchpad
```

> This starts a stdio MCP server (JSON-RPC), not an interactive CLI. Launch it from an MCP client. For an interactive terminal session, use `python opensearch_orchestrator/orchestrator.py` instead.

### Running locally (dev)

```bash
uv run opensearch_orchestrator/mcp_server.py
```

`uv` reads inline script metadata and auto-installs dependencies into a cached virtual environment.

### Claude Desktop integration

1. Add to `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "opensearch-launchpad": {
      "command": "uvx",
      "args": ["opensearch-launchpad@latest"]
    }
  }
}
```

2. Restart Claude Desktop. The `opensearch_workflow` prompt is available in the prompt picker and describes the full tool sequence.

### Generic MCP clients

Any MCP-compatible client can connect via stdio and discover tools with `tools/list`. The `opensearch_workflow` prompt (available via `prompts/list`) describes the workflow. Tool docstrings also include prerequisite hints.

### Without uv

Install dependencies manually and point to the server script:

```bash
pip install mcp opensearch-py
```

```json
{
  "mcpServers": {
    "opensearch-launchpad": {
      "command": "python3",
      "args": ["opensearch_orchestrator/mcp_server.py"],
      "cwd": "/path/to/agent"
    }
  }
}
```

---

## MCP Workflow Tools

The server exposes high-level phase tools:

| Tool | Phase | Description |
|------|-------|-------------|
| `load_sample` | 1 | Load a sample document (built-in IMDB, local file, URL, index, or paste) |
| `set_preferences` | 2 | Set budget, performance, query pattern, deployment preferences |
| `start_planning` | 3 | Start the planning agent; returns initial architecture proposal |
| `refine_plan` | 3 | Send user feedback to refine the proposal |
| `finalize_plan` | 3 | Finalize the plan when the user confirms |
| `set_plan_from_planning_complete` | 3 | Parse/store a `<planning_complete>` planner response |
| `execute_plan` | 4 | Return worker bootstrap payload for execution |
| `set_execution_from_execution_report` | 4 | Parse/store `<execution_report>` and update retry state |
| `retry_execution` | 4 | Return resume bootstrap payload from last failed step |
| `prepare_aws_deployment` | 5 | Return deployment target and steering files for AWS |
| `connect_search_ui_to_endpoint` | 5 | Switch Search UI to query an AWS OpenSearch endpoint after deployment |
| `cleanup` | Post | Remove test documents on user request |

The following execution/knowledge tools are also exposed:
`create_index`, `create_and_attach_pipeline`, `create_bedrock_embedding_model`,
`create_local_pretrained_model`, `apply_capability_driven_verification`,
`launch_search_ui`, `set_search_ui_suggestions`, `read_knowledge_base`,
`read_dense_vector_models`, `read_sparse_vector_models`, `search_opensearch_org`.

Advanced tools are hidden by default; set `OPENSEARCH_MCP_ENABLE_ADVANCED_TOOLS=true` to expose them.

### Localhost index auth (`source_type="localhost_index"`)

| Mode | Behavior |
|------|----------|
| `"default"` | Username `admin`, password `myStrongPassword123!` |
| `"none"` | No authentication |
| `"custom"` | Requires `localhost_auth_username` + `localhost_auth_password` |

Local Docker auto-bootstrap uses `admin` and reads the password from `OPENSEARCH_PASSWORD` (falls back to `myStrongPassword123!`).

### Planner backend in MCP mode

- Planning uses client sampling (client LLM only — no server-side Bedrock in MCP mode).
- If the client does not support `sampling/createMessage`, `start_planning` returns `manual_planning_required=true` with `manual_planner_system_prompt` and `manual_planner_initial_input`. Run planner turns with your LLM and call `set_plan_from_planning_complete(planner_response)`.

---

## Release Checklist

```bash
# 1) Bump version in both files to the same value, e.g. 0.10.1
#    - pyproject.toml: [project].version
#    - opensearch_orchestrator/__init__.py: __version__

# Optional sanity check:
python -c "import tomllib; p=tomllib.load(open('pyproject.toml','rb')); import opensearch_orchestrator as pkg; print('pyproject=', p['project']['version'], 'package=', pkg.__version__)"

# 2) All tests must pass
uv run pytest -q

# 3) Build and verify artifacts
uv build
for whl in dist/*.whl; do python -m zipfile -l "$whl"; done
python -c "import opensearch_orchestrator.mcp_server as m; print(hasattr(m, 'main'))"

# Smoke-test the wheel
VERSION="$(python -c "import tomllib; print(tomllib.load(open('pyproject.toml','rb'))['project']['version'])")"
WHEEL_PATH="$(ls dist/opensearch_launchpad-${VERSION}-*.whl 2>/dev/null || ls dist/opensearch_orchestrator-${VERSION}-*.whl)"
uvx --from "$WHEEL_PATH" opensearch-launchpad

# 4) Publish to PyPI
uv publish --token pypi-YOUR-TOKEN
```
