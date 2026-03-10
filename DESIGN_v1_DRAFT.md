# OpenSearch Launchpad — Architecture & Design (v1 Draft)

> **Status**: Draft for review. Not intended for commit.
> Copy to quip.amazon.com for team feedback.

## 1. Overview

OpenSearch Launchpad is an MCP-powered assistant that guides developers from initial
requirements to a running OpenSearch search setup. It collects a sample document,
gathers preferences (budget, performance, query pattern), plans a search architecture,
executes the plan (indices, ML models, ingest pipelines, search UI), and optionally
deploys to Amazon OpenSearch Service or Serverless.

The system is designed to work across multiple agentic IDEs — Kiro, Cursor, Claude Code,
VS Code (Copilot), and others — by leveraging the IDE's own agent rather than bundling
a custom LLM. Domain knowledge and procedures live in **steering files** (delivered as
**Agent Skills** per the open standard at agentskills.io). **MCP tools** provide
execution capabilities and workflow predictability. As agents mature, the orchestration
layer thins and steering files + external MCP servers become sufficient on their own.

---

## 2. Prerequisites

### All Phases (Local Development)

| Requirement | Version / Notes |
|-------------|----------------|
| **Python** | 3.11+ |
| **uv** | Package runner ([install](https://docs.astral.sh/uv/getting-started/installation/)) |
| **Docker** | Installed and running ([download](https://docs.docker.com/get-docker/)) |
| **opensearch-launchpad MCP server** | `uvx opensearch-launchpad@latest` (from PyPI) |

### Phase 5 Only (AWS Deployment)

| Requirement | Notes |
|-------------|-------|
| **AWS CLI** | With credentials configured (`aws configure` or env vars) |
| **IAM permissions** | OpenSearch Service, IAM, Bedrock (for semantic/agentic search) |
| **awslabs.aws-api-mcp-server** | `uvx awslabs.aws-api-mcp-server@latest` |
| **aws-docs MCP server** | `uvx awslabs.aws-documentation-mcp-server@latest` |
| **opensearch-mcp-server** | `uvx opensearch-mcp-server-py@latest` |

All Phase 5 MCP servers must be added to the IDE's MCP configuration before starting
AWS deployment. See §5.5 for per-IDE MCP config locations.

---

## 3. Tenets

These tenets guide all architectural decisions. When in conflict, earlier tenets
take priority.

### T1. Leverage the IDE agent — never bundle your own LLM

The IDE's agent (Kiro, Claude Code, Cursor, etc.) is the orchestrator. We provide
knowledge and tools; the IDE provides reasoning. This eliminates model dependencies,
reduces cost, and lets each IDE use its best model.

### T2. Keep steering files under 500 lines

Any file loaded into an agent's context window — SKILL.md, steering files, rules
files — must stay under 500 lines. Longer files dilute attention and reduce
adherence across all platforms (Claude Code recommends < 200 lines, Cursor recommends
< 500 lines). When a file grows beyond this limit, split it using progressive
disclosure: metadata in the skill, details in reference files or MCP tool responses.

### T3. Progressive disclosure over monolithic context

Agents should load only what they need, when they need it. At startup, only skill
names and descriptions are in context (~100 tokens each). Full instructions load on
activation (< 5000 tokens recommended). Detailed references, knowledge bases, and
step-by-step procedures load on demand via MCP tool calls or reference file reads.

### T4. One domain per file

Each skill, steering file, or rule file covers exactly one concern. A file that
covers "AWS serverless provisioning" should not also cover "search UI design tokens."
This keeps files focused, testable, and within the 500-line limit.

### T5. Steering files override where external knowledge falls short

External MCP servers (AWS docs, AWS API, OpenSearch MCP) are the primary knowledge
source — they provide up-to-date, authoritative documentation and APIs. The model's
training data serves as a secondary source, filling gaps with general understanding.
Steering files sit on top as an override layer for project-specific context that
neither external servers nor the model can know: our specific workflow, tool
sequencing, conventions, and edge cases. As external MCP servers cover more ground,
steering files should shrink to only what genuinely needs overriding.

### T6. Orchestration tools add predictability — design them to thin over time

Orchestration MCP tools (`prepare_aws_deployment`, `set_preferences`, etc.) exist to
keep the agent on the right track: which steering file to read, which phase we're in,
what to ask next. They compensate for current agent limitations — agents sometimes
skip steps, lose their place, or drift in long multi-phase workflows. The
orchestration layer should be as thin as possible and is expected to shrink as agents
improve. The end state is: steering files + execution tools + external MCP servers —
with orchestration becoming optional or removed entirely. Design decisions should
favor putting knowledge in steering files (durable) over encoding it in orchestration
tools (transitional).

### T7. Write once, run on any IDE

All domain knowledge lives in steering files delivered as Agent Skills (the open
standard). IDE-specific integration is a thin adapter layer. Adding support for a new
IDE should require only a configuration file and possibly a skill-to-native mapping —
never forking the core knowledge or procedures.

---

## 4. Current Architecture (v0.x — Kiro-only)

### 4.1 How It Works Today

```
User <-> Kiro Agent <-> MCP Protocol <-> mcp_server.py <-> OrchestratorEngine
              |                                |
         reads POWER.md                   exposes tools
         reads steering/*.md              manages state
```

- **POWER.md** (17.5K chars, ~350 lines): Kiro reads this at session start. Contains
  the full workflow description, all tool documentation, rules, and phase instructions.
- **Steering files** (800-18K chars each): Loaded by Kiro based on `inclusion: auto`
  descriptions. Contain step-by-step AWS deployment procedures, UI design tokens, etc.
- **MCP tools**: Stateful phase tools (`load_sample`, `set_preferences`,
  `start_planning`, `execute_plan`, etc.) plus low-level OpenSearch operations.
- **OrchestratorEngine**: Transport-agnostic state machine with phases:
  `COLLECT_SAMPLE -> GATHER_INFO -> EXEC_FAILED -> DONE`.
- **Knowledge files**: Markdown guides (semantic search, dense/sparse vector models,
  agentic search) read via `read_knowledge_base` tool calls.

### 4.2 Workflow Phases

| Phase | Tools | Description |
|-------|-------|-------------|
| 1. Collect Sample | `load_sample` | Load sample doc (IMDB, file, URL, index, paste) |
| 2. Preferences | `set_preferences` | Budget, performance, query pattern, deployment |
| 3. Plan | `start_planning`, `refine_plan`, `finalize_plan` | Architecture proposal |
| 4. Execute | `execute_plan`, `retry_execution` | Create index, models, pipelines, UI |
| 4.5. Evaluate | `start_evaluation` | Optional search quality evaluation |
| 5. Deploy | `prepare_aws_deployment` | Optional AWS deployment |


---

## 5. Proposed Architecture (v1.0 — Multi-IDE via Agent Skills)

### 5.1 Why Agent Skills?

Agent Skills (agentskills.io) is an open standard originally developed by Anthropic
and now supported by 25+ tools:

- **Kiro**, **Cursor**, **Claude Code**, **VS Code (Copilot)**
- **JetBrains Junie**, **Gemini CLI**, **OpenAI Codex**, **Roo Code**
- **Goose**, **Amp**, **OpenHands**, **Databricks**, and more

Kiro supports Agent Skills natively (`.kiro/skills/`) alongside its existing Powers
and steering file systems. Skills in Kiro use the same progressive disclosure model
and can coexist with Powers for MCP integrations.

A skill is a directory with a `SKILL.md` file containing YAML frontmatter (name,
description) and markdown instructions. Skills support progressive disclosure:

1. **Discovery** (~100 tokens): name + description loaded at startup
2. **Activation** (< 5000 tokens): full SKILL.md loaded when task matches
3. **Execution** (on demand): reference files, scripts, assets loaded as needed

This maps perfectly to our needs: the workflow overview loads on activation, and
phase-specific details load on demand via reference files or MCP tool calls.

### 5.2 High-Level Architecture

```
User <-> IDE Agent <-> Agent Skills / Steering Files (knowledge + procedures)
              |                |
              |          references/ loaded on demand
              |
              +------> MCP Protocol <-> mcp_server.py <-> OrchestratorEngine
                                              |
                                   orchestration tools (predictability)
                                   execution tools (real operations)
                                              |
                                   external MCP servers
                                   (AWS API, OpenSearch MCP, AWS docs)
```

**Layer 1 — Steering Files (knowledge — the durable layer)**
- Skills discovered at startup via name + description (~100 tokens each)
- Full SKILL.md loads on activation (< 500 lines)
- Reference files (procedures, domain guides) load on demand
- This is where all procedural knowledge lives: how to provision, how to deploy,
  how to configure search architectures
- Steering files persist and improve independently of the orchestration layer

**Layer 2 — Orchestration Tools (predictability — the shrinking layer)**
- Keep the agent on track in multi-phase workflows
- Route to the right steering file at the right time
- Track workflow state (which phase, what's been completed)
- Expected to thin over time as agents improve

**Layer 3 — Execution Tools + External MCP Servers (operations — the permanent layer)**
- Execution tools: `create_index`, `load_sample`, `launch_search_ui` — do real work
- External MCP servers: AWS API, OpenSearch MCP, AWS docs — the actual infrastructure
  surface for cloud deployment
- These persist regardless of how smart agents become

### 5.2.1 Architecture Trajectory

```
Today:      steering files + orchestration tools + execution tools + external MCP servers
            (orchestration needed because agents drift in multi-phase workflows)

Near-term:  steering files + lighter orchestration + execution tools + external MCP servers
            (agents improve; orchestration shrinks to critical guardrails only)

Future:     steering files + execution tools + external MCP servers
            (agents reliably follow steering files; orchestration optional or removed)
```

The orchestration layer is a compensator for current agent limitations, not a permanent
architectural fixture. Design decisions should favor putting knowledge in steering
files (durable) over encoding it in orchestration tools (transitional).

### 5.3 Skill Structure

One skill, one workflow. The current monolithic POWER.md becomes a single Agent Skill
with reference files for progressive disclosure:

```
skills/opensearch-launchpad/
    SKILL.md                             # < 300 lines: rules, tool overview, workflow
    references/
        phase1-collect-sample.md         # Phase 1 procedures (< 500 lines each)
        phase2-preferences.md            # Phase 2 procedures
        phase3-planning.md               # Phase 3 procedures
        phase4-execution.md              # Phase 4 procedures
        phase5-aws-deployment.md         # Phase 5 overview + routing
        serverless-provision.md          # AWS serverless provisioning steps
        serverless-deploy.md             # AWS serverless search deployment
        domain-provision.md              # AWS domain provisioning steps
        domain-deploy.md                 # AWS domain search deployment
        domain-agentic.md               # AWS agentic search setup
        aws-reference.md                 # Cost, security, HA reference
```

Users install one skill. The agent activates it when the task matches, and loads
reference files on demand as the workflow progresses through phases. AWS deployment
(Phase 5) is just another phase — not a separate skill.

**Key properties:**
- Single skill, single install — no user confusion about which to choose
- `SKILL.md` stays under 500 lines (tenet T2); covers all 5 phases at overview level
- Reference files loaded on demand per phase (tenet T3)
- Each reference file covers one domain (tenet T4)
- Same skill works across Kiro, Claude Code, Cursor, VS Code, JetBrains, etc. (tenet T7)

### 5.4 SKILL.md Frontmatter

Following the agentskills.io specification:

```yaml
---
name: opensearch-launchpad
description: >
  Build OpenSearch search applications with guided architecture planning.
  Collects sample documents, gathers preferences, plans search architecture,
  and executes setup with indices, ML models, pipelines, and search UI.
  Use when the user wants to build a search app, set up OpenSearch, or
  design search architecture.
metadata:
  author: opensearch-project
  version: "1.0"
---
```

| Field | Constraint | Our usage |
|-------|-----------|-----------|
| `name` | Max 64 chars, lowercase + hyphens | `opensearch-launchpad` |
| `description` | Max 1024 chars | Workflow summary + trigger keywords |
| `compatibility` | Max 500 chars (optional) | `Requires Python 3.11+, uv, Docker` |

### 5.5 MCP Server Configuration Across IDEs

The Agent Skills spec does not include a mechanism for bundling MCP server
configuration. Each IDE handles MCP config differently:

| IDE | MCP Config Location |
|-----|-------------------|
| **Kiro** | Power `mcp.json` or `.kiro/settings/mcp.json` |
| **Claude Code** | `claude_desktop_config.json` or `--mcp` flag |
| **Cursor** | `.cursor/mcp.json` |
| **VS Code** | `.vscode/mcp.json` |

For Kiro, we retain a minimal Power (`kiro/opensearch-launchpad/`) containing only
`mcp.json` — no `POWER.md`, no steering files. All knowledge and procedures live in
Agent Skills. Kiro's docs note that "for MCP integrations, powers are usually a
better fit" — but the Power's role here is strictly MCP config delivery, not knowledge.

For other IDEs, the SKILL.md body includes setup instructions telling the user how
to configure the MCP server for their IDE. Long-term, if Agent Skills adds an MCP
config mechanism or IDEs converge on a standard location, even the minimal Kiro Power
can be removed.

### 5.6 IDE Integration Matrix

| IDE | Skills Location | Skill Discovery | MCP Config |
|-----|----------------|-----------------|------------|
| **Kiro** | `.kiro/skills/` | Auto (description match) | Power `mcp.json` |
| **Claude Code** | `.claude/skills/` | Auto or `/opensearch-launchpad` | `claude_desktop_config.json` |
| **Cursor** | `.claude/skills/` | Auto on keyword match | `.cursor/mcp.json` |
| **VS Code Copilot** | Agent Skills standard | Auto | `.vscode/mcp.json` |
| **JetBrains Junie** | Agent Skills standard | Auto | IDE MCP settings |
| **Gemini CLI** | Agent Skills standard | Auto | CLI config |

All IDEs share the same MCP server (`opensearch-launchpad` on PyPI) and the same
Agent Skills content. The only difference is the directory convention for skill
discovery (`.kiro/skills/` vs `.claude/skills/`), which can be resolved with
symlinks or by placing skills in a shared location.


### 5.7 Steering Files vs Orchestration Tools vs Execution Tools

| Concern | Where it lives | Why |
|---------|---------------|-----|
| Procedures (how to provision, deploy, configure) | Steering files / references | Durable knowledge; persists as orchestration thins |
| Domain expertise (search architecture, model selection) | Steering files / references | Durable knowledge |
| Behavioral rules (one question per message, etc.) | Steering file (SKILL.md) | Shapes agent reasoning across all phases |
| Workflow routing (which file to read next) | Orchestration tools | Compensates for agents losing track |
| Phase tracking (where we are) | Orchestration tools | Compensates for agents losing state |
| Conditional logic (skip questions based on data) | Orchestration tools | Agents can't reliably evaluate conditions in prose |
| Real operations (create index, start UI) | Execution tools | Actual work; always needed |
| AWS infrastructure (provision, configure) | External MCP servers | AWS API, OpenSearch MCP, AWS docs |

**Key insight:** The middle column (orchestration tools) is the one expected to shrink.
The outer columns (steering files and execution tools/external MCP servers) are permanent.

---

## 6. Directory Structure (Target)

```
opensearch-launchpad/
    skills/
        opensearch-launchpad/               # Single skill (shared by all IDEs)
            SKILL.md                        # < 300 lines: rules, tool overview, workflow
            references/
                phase1-collect-sample.md    # Phase 1 procedures (< 500 lines each)
                phase2-preferences.md       # Phase 2 procedures
                phase3-planning.md          # Phase 3 procedures
                phase4-execution.md         # Phase 4 procedures
                phase5-aws-deployment.md    # Phase 5 overview + routing
                serverless-provision.md     # AWS serverless provisioning steps
                serverless-deploy.md        # AWS serverless search deployment
                domain-provision.md         # AWS domain provisioning steps
                domain-deploy.md            # AWS domain search deployment
                domain-agentic.md           # AWS agentic search setup
                aws-reference.md            # Cost, security, HA reference
    .kiro/
        skills/ -> ../skills                # Symlink (Kiro skill discovery)
    .claude/
        skills/ -> ../skills                # Symlink (Claude Code / Cursor discovery)
    kiro/
        opensearch-launchpad/
            mcp.json                        # MCP server config only (no POWER.md)
    opensearch_orchestrator/
        mcp_server.py                       # MCP server (shared by all IDEs)
        orchestrator_engine.py              # State machine + orchestration routing
        planning_session.py
        solution_planning_assistant.py
        worker.py
        tools.py
        opensearch_ops_tools.py
        shared.py
        handler.py
        knowledge/                          # Domain knowledge read via MCP tools
            opensearch_semantic_search_guide.md
            dense_vector_models.md
            sparse_vector_models.md
            agentic_search_guide.md
        sample_data/
        ui/
    tests/
    pyproject.toml
```

The `skills/` directory at the repo root is the single source of truth. IDE-specific
directories (`.kiro/skills/`, `.claude/skills/`) symlink to it, so all IDEs read the
same skill content with zero duplication.

---

## 7. Technical Stack

| Concern | Technology |
|---------|-----------|
| Agent orchestration | IDE-native agent (no bundled LLM) |
| Skill format | Agent Skills open standard (agentskills.io) |
| MCP server | `fastmcp` (via `mcp` package) |
| State machine | `OrchestratorEngine` in `orchestrator_engine.py` |
| OpenSearch client | `opensearch-py` |
| Package manager | `uv` / `uvx` |
| Distribution | PyPI (`opensearch-launchpad`) |
| IDE integration | Agent Skills (universal); Kiro Power for `mcp.json` only |

---

## 8. FAQ

### Why Agent Skills over custom per-IDE adapters?

Agent Skills is supported by 25+ tools including all our targets: Kiro, Cursor,
Claude Code, VS Code, JetBrains, Gemini CLI, and more. Writing one SKILL.md gives us
all of these without maintaining separate `.cursorrules`, `CLAUDE.md`, and `POWER.md`
files with duplicated content. Kiro supports Agent Skills natively (`.kiro/skills/`),
so the only Kiro-specific artifact is `mcp.json` for MCP server configuration.

### Why keep procedures in steering files instead of MCP tool responses?

Steering files are the durable knowledge layer. Procedures (how to provision AWS,
how to configure search pipelines) are domain knowledge — they should live where
knowledge lives, not be encoded into tool implementations. MCP orchestration tools
route the agent to the right steering file at the right time, but they don't own the
content. This separation matters because:

1. Steering files are human-readable, auditable, and editable without code changes.
2. Steering files work across IDEs without tool-specific coupling.
3. As agents improve at following multi-file instructions, the orchestration tools
   can thin out without losing knowledge — it's all still in the steering files.
4. External MCP servers (AWS API, OpenSearch MCP, AWS docs) already provide the
   execution surface. Steering files tell the agent how to use them.

### Why have orchestration tools at all?

Agents today lose track of where they are in multi-phase workflows. They skip steps,
batch questions that should be asked one at a time, or read the wrong steering file.
Orchestration tools (`prepare_aws_deployment`, `set_preferences`, etc.) add
predictability: they track state and route the agent to the right file at the right
time. This is a compensator for current limitations, not a permanent architectural
layer. As agents improve, these tools should be thinned or removed.

### Why not keep using a custom planner/worker LLM?

The original architecture used Strands agents with Bedrock Claude for planning and
execution. This added a model dependency, increased cost, and couldn't leverage
improvements in the IDE's own model. By delegating to the IDE agent and providing
knowledge via skills + MCP, we get better results with less complexity. Strands-based
agents remain available as fallbacks for standalone CLI mode.

### Why keep a minimal Kiro Power?

Agent Skills has no mechanism for MCP server configuration. Kiro delivers MCP config
via Powers (`mcp.json`). We retain a minimal Power containing only `mcp.json` — no
`POWER.md`, no steering files. All knowledge lives in Agent Skills. If Agent Skills
adds MCP config support or Kiro adds another way to configure MCP servers, the Power
can be removed entirely.

### Why not use sub-agents?

Sub-agents (spawning a child agent with its own context and system prompt) are useful
when a task requires deep, isolated reasoning that would pollute the main agent's
context — e.g., "analyze these inputs and produce a structured result." The original
Strands-based planner and evaluator were effectively sub-agents.

We don't use sub-agents in the current design because:

1. **The IDE agent is capable enough.** With the right steering files and reference
   material loaded on demand, the IDE agent can handle planning, evaluation, and
   execution without needing a separate reasoning context.
2. **Sub-agents add complexity.** Each sub-agent needs its own system prompt, tool
   access, and result serialization. This is more code to maintain and debug.
3. **Sub-agents conflict with T1.** Spawning a sub-agent means either bundling a model
   (violates T1) or using client sampling, which not all IDEs support reliably.
4. **Steering files scale better.** Rather than isolating reasoning in a sub-agent, we
   load focused reference files on demand (T3). The agent gets the knowledge it needs
   for the current phase without carrying the full workflow in context.

If a future phase proves too complex for the IDE agent with steering files alone (e.g.,
a planning step that needs to reason over very large inputs), sub-agents can be
reconsidered — but the burden of proof is on the sub-agent approach to justify the
added complexity.

### Why 500 lines as the limit?

- [Claude Code](https://docs.anthropic.com/en/docs/claude-code/memory#claudemd-files) recommends < 200 lines for CLAUDE.md files
- [Cursor](https://docs.cursor.com/context/rules) recommends < 500 lines for rules files
- [Agent Skills spec](https://agentskills.io/specification) recommends < 500 lines for SKILL.md
- Kiro has no published limit, but the same principles apply

500 lines is the common upper bound. In practice, the primary instruction file
(SKILL.md) should target 200-300 lines, with 500 as the absolute ceiling.

---

## 9. Open Questions

1. **Skill directory symlinks**: Will all IDEs correctly follow symlinks from
   `.kiro/skills/` and `.claude/skills/` to the shared `skills/` directory? Need
   to test on all target platforms.

2. **Cursor MCP discovery**: How does Cursor discover MCP servers referenced by Agent
   Skills? Is `mcp.json` auto-detected or does the user configure it separately?

3. **Reference file granularity**: Are the current reference file splits right, or
   should some be merged? E.g., should `serverless-provision.md` and
   `serverless-deploy.md` be one file if they're always used together?

4. **Orchestration thinning timeline**: How quickly are agents improving at following
   multi-phase steering files without orchestration help? We should benchmark
   periodically: run the workflow with orchestration tools disabled and measure
   how often the agent drifts. When success rate is high enough, start removing
   orchestration.

5. **MCP config in Agent Skills**: Will the Agent Skills spec add a mechanism for
   declaring MCP server dependencies (e.g., an `mcp.json` in `assets/`)? This would
   eliminate the need for the minimal Kiro Power and per-IDE MCP config instructions.
