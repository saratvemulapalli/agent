---
name: "search-builder"
displayName: "OpenSearch Solution Architect"
description: "An OpenSearch Solution Architect that guides you from requirements to a running search setup. Collects sample documents, gathers preferences, plans architecture, and executes the solution with indices, models, pipelines, and a search UI."
keywords: ["opensearch", "search", "semantic search", "vector search", "hybrid search", "RAG", "embeddings", "knn", "neural search", "BM25", "index", "search architecture"]
author: "OpenSearch"
---

# Onboarding

## Prerequisites

1. **Python 3.10+** and `uv` installed ([Install uv](https://docs.astral.sh/uv/getting-started/installation/))
2. **OpenSearch cluster** running locally (Docker) or remotely for the execution phase
   - Local quickstart: `docker run -d -p 9200:9200 -e "discovery.type=single-node" -e "DISABLE_SECURITY_PLUGIN=true" opensearchproject/opensearch:latest`
3. **AWS credentials configured**
   - `aws configure` or `~/.aws/credentials`
   - Required IAM permissions: `bedrock:*` for the power to access LLM

## Configuration

After installing this power, update the MCP server configuration with your AWS profile and region (if using Bedrock):

1. Open Kiro Settings → MCP Servers (or edit `~/.kiro/settings/mcp.json`)
2. Find the `opensearch-orchestrator` entry
3. Update the `env` section:

```json
"env": {
  "AWS_PROFILE": "your-profile-name",  // ← Change to your AWS profile
  "AWS_REGION": "us-west-2",           // ← Change to your region
  "FASTMCP_LOG_LEVEL": "ERROR"
}
```

**Default:** Uses `default` AWS profile and `us-east-1` region.

## Quick Test

After configuration, try: *"I want to build a semantic search app with 10M docs"*

---

# OpenSearch Solution Architect

An MCP-powered assistant that guides you from requirements to a running OpenSearch search setup.

## Overview

This power provides an OpenSearch Solution Architect workflow. It collects a sample document, gathers your preferences (budget, performance, query pattern), plans an architecture using an AI planner agent, and executes the plan to create indices, models, pipelines, and a search UI.

## Workflow Phases

### Phase 1: Collect Sample Document (mandatory first step)
- Call `load_sample(source_type, source_value)`.
  - source_type: "builtin_imdb" | "local_file" | "url" | "localhost_index" | "paste"
  - source_value: file path, URL, index name, or pasted JSON content (empty string for builtin_imdb)
- The result includes `inferred_text_fields` and `text_search_required`.
- A sample document is required before any planning or execution.

### Phase 2: Gather Preferences
- Before calling `set_preferences`, collect each preference from the user one at a time.
- For each preference, ask a single question and wait for the user's answer before asking the next.
- Preference values:
  - **Budget**: "flexible" (no cost constraints) or "cost-sensitive" (minimize resource usage)
  - **Performance priority**: "speed-first", "balanced", or "accuracy-first"
  - **Query pattern**: "mostly-exact" (BM25), "balanced", or "mostly-semantic" (neural)
  - **Deployment preference** (only if query pattern is "mostly-semantic"): "opensearch-node", "sagemaker-endpoint", or "external-embedding-api"
- After collecting all answers, call `set_preferences(budget, performance, query_pattern, deployment_preference)` with the values.

### Phase 3: Plan
- Call `start_planning()` to get an initial architecture proposal from the planner agent.
- Present the proposal to the user **verbatim** (do not summarize it away).
- If the user has feedback or questions, call `refine_plan(user_feedback)`. Repeat as needed.
- When the user confirms, call `finalize_plan()`.
  This returns {solution, search_capabilities, keynote}.

### Phase 4: Execute
- Call `execute_plan()` to create the index, models, pipelines, and launch the search UI.
- If execution fails, the user can fix the issue (e.g., restart Docker) and you call `retry_execution()`.

### Post-Execution
- After successful `execute_plan()`/`retry_execution()`, explicitly tell the user how to access the UI using the `ui_access` URLs returned by the tool result.
- `cleanup_verification()` removes test/verification documents when the user explicitly asks.

## Available Tools

### High-Level Workflow Tools
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

### Knowledge Tools
| Tool | Description |
|------|-------------|
| `read_knowledge_base` | Read the OpenSearch Semantic Search Guide |
| `read_dense_vector_models` | Read the Dense Vector Models Guide |
| `read_sparse_vector_models` | Read the Sparse Vector Models Guide |
| `search_opensearch_org` | Search opensearch.org documentation |

### Operation Tools (advanced)
| Tool | Description |
|------|-------------|
| `create_index` | Create an OpenSearch index with configuration |
| `create_bedrock_embedding_model` | Register a Bedrock embedding model |
| `create_local_pretrained_model` | Deploy a local pretrained model |
| `create_and_attach_pipeline` | Create and attach ingest/search pipelines |
| `index_doc` | Index a document |
| `delete_doc` | Delete a document |

## Rules
- Never skip Phase 1. A sample document is mandatory before planning.
- Do not generate the solution plan yourself; always delegate to the planner tools.
- Show the planner's proposal text to the user verbatim; do not summarize it away.
- For preference questions, ask one question at a time and wait for the user's answer before proceeding.
- Do not ask redundant clarification questions for items already inferred from the sample data.

## Prerequisites
- See the [Onboarding](#onboarding) section at the top of this document.
