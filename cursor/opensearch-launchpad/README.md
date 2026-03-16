# OpenSearch Launchpad — Cursor Plugin

This directory is a **Cursor plugin** (manifest + assets) for [OpenSearch Launchpad](https://github.com/opensearch-project/opensearch-launchpad): an MCP server and skill that guide you from requirements to a running OpenSearch search setup (indices, ML models, ingest pipelines, search UI, optional AWS deployment).

It follows Cursor’s plugin structure from [Plugins reference](https://cursor.com/docs/reference/plugins).

---

## Prerequisites

- **Python 3.11+** and [uv](https://docs.astral.sh/uv/getting-started/installation/) (or `uvx` on PATH)
- **Docker** (for local OpenSearch and search UI)
- **For AWS deployment (Phase 5):** AWS credentials and optional AWS/OpenSearch MCP servers

---

## Installation

### Option 1: Use as a Cursor plugin (recommended)

This plugin bundles:

- `.mcp.json` (MCP server definition)
- `skills/` (Agent Skill + references)
- `.cursor-plugin/plugin.json` (required plugin manifest)

To install it in Cursor, add this plugin repo/folder via your Cursor plugin workflow, or copy this folder into a plugin collection repo. Cursor will discover:

- MCP servers from `.mcp.json`
- Skills from `skills/*/SKILL.md`

### Option 2: Manual project config (works without plugins)

Cursor also supports project-level MCP config at `.cursor/mcp.json` (or global `~/.cursor/mcp.json`). You can copy the `mcpServers` entry from this plugin’s `.mcp.json` into `.cursor/mcp.json`:

Cursor → Settings → **Tools & MCP** → Add new MCP server:

- **Name:** `opensearch-launchpad`
- **Command:** `uvx`
- **Args:** `opensearch-launchpad@latest`
- **Env:** `FASTMCP_LOG_LEVEL=ERROR` (optional)

If `uvx` is not on PATH when Cursor runs, use the bootstrap command in the *Troubleshooting* section below.

### Add the Agent Skill (only needed for manual config)

So Cursor’s agent knows **when** and **how** to use the tools, add the skill:

- **From this plugin:** Copy `skills/opensearch-search-builder/` into your project’s `.cursor/skills/` (create `.cursor/skills/` if needed).

  ```bash
  mkdir -p .cursor/skills
  cp -r cursor/opensearch-launchpad/skills/opensearch-search-builder .cursor/skills/
  ```

- **Shared skill (same repo):** This repo also ships the skill under `.claude/skills/opensearch-search-builder/`. If your Cursor setup reads from `.claude/skills/`, no extra copy is needed; ensure the MCP config above is in `.cursor/mcp.json`.

Restart Cursor (or reload MCP) so the server and skill are picked up.

---

## What you get

- **MCP server:** `opensearch-launchpad` (PyPI package `opensearch-launchpad`, built from `opensearch_orchestrator` in this repo).
- **Skill:** `opensearch-search-builder` — workflow (sample → preferences → plan → execute → optional AWS) and tool usage so the agent follows phases and rules (e.g. one preference question per message, never skip Phase 1).

Same package and workflow are used for **Kiro** (power in `kiro/opensearch-launchpad/`) and **Claude** (skill in `.claude/skills/opensearch-search-builder/`); this folder is the **Cursor** counterpart for plugin/marketplace and project setup.

---

## Troubleshooting

### `uvx` not found (e.g. spawn ENOENT)

If Cursor runs without your shell PATH, use a bootstrap command in `.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "opensearch-launchpad": {
      "command": "bash",
      "args": [
        "--noprofile",
        "--norc",
        "-c",
        "set -euo pipefail; PATH=\"$HOME/.local/bin:$HOME/.cargo/bin:/opt/homebrew/bin:/usr/local/bin:$PATH\"; exec uvx opensearch-launchpad@latest"
      ],
      "env": {
        "FASTMCP_LOG_LEVEL": "ERROR"
      }
    }
  }
}
```

Adjust `PATH` if your `uv`/`uvx` lives elsewhere (e.g. `which uvx`).

### Docker not found

Ensure Docker is on the same PATH Cursor uses. If needed, add Docker’s directory (e.g. `/usr/local/bin`) to `env.PATH` in the MCP server config.

---

## Marketplace / distribution

To submit this plugin to the Cursor marketplace, ensure this folder contains:

- `.cursor-plugin/plugin.json` (required)
- `.mcp.json` (MCP server definition)
- `skills/opensearch-search-builder/SKILL.md` (+ references)

The package name on PyPI is **`opensearch-launchpad`**; the server implementation is shared with Kiro and Claude integrations.
