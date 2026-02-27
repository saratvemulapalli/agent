---
name: "opensearch-search-builder"
displayName: "Build a POC search application with OpenSearch"
description: "Accelerate proof-of-concept search applications with guided, end-to-end architecture planning. Ingests sample documents, captures preferences, designs the solution architecture, and provisions indices, ML models, ingest pipelines, and a search UI."
keywords: ["opensearch", "search", "semantic search", "vector search", "hybrid search", "RAG", "embeddings", "knn", "neural search", "BM25", "index", "search architecture", "docker", "local", "aws", "serverless", "aoss"]
author: "AWS"
---

# Onboarding

## Prerequisites

1. **Python 3.10+** and `uv` installed ([Install uv](https://docs.astral.sh/uv/getting-started/installation/))
2. **Docker** installed and running ([Download Docker](https://docs.docker.com/get-docker/))

## Quick Test

After configuration, try: *"I want to build a semantic search app with 10M docs"*

---

# Overview

An MCP-powered assistant that guides you from requirements to a running OpenSearch search setup.

This power provides an OpenSearch Search Solution building workflow. It collects a sample document, gathers your preferences (budget, performance, query pattern), plans an architecture using an AI planner agent, and executes the plan to create indices, models, pipelines, and a search UI.

## Workflow Phases

### Phase 1: Collect Sample Document (mandatory first step)
- Call `load_sample(source_type, source_value)`.
  - source_type: "builtin_imdb" | "local_file" | "url" | "localhost_index" | "paste"
  - source_value: file path, URL, index name, or pasted JSON content (empty string for builtin_imdb)
- The result includes `inferred_text_fields` and `text_search_required`.
- A sample document is required before any planning or execution.

### Phase 2: Gather Preferences
- Ask one preference question at a time, in this order.
- Present each question as a numbered list and ask the user to reply with the number of their choice.

  **Budget:**
  1. Flexible
  2. Cost-sensitive

  **Performance priority:**
  1. Speed-first
  2. Balanced
  3. Accuracy-first

  **Query pattern:**
  1. Mostly-exact (e.g. "Carmencita 1894")
  2. Mostly-semantic (e.g. "early silent films about dancers")
  3. Balanced (mix of both)

- If query pattern is balanced or mostly-semantic, ask deployment preference as a separate follow-up question:

  **Deployment preference:**
  1. OpenSearch node
  2. SageMaker endpoint
  3. External embedding API

- Call `set_preferences(budget, performance, query_pattern, deployment_preference)` with the values.

### Phase 3: Plan
- Call `start_planning()` to get the initial architecture proposal.
- If `start_planning()` returns `manual_planning_required=true`, use
  `manual_planner_system_prompt` + `manual_planner_initial_input` to run planner turns with the client LLM.
- Present the proposal to the user **verbatim** (do not summarize it away).
- If the user has feedback, refine the proposal (with tools when available, otherwise directly with the client LLM) and repeat.
- When the user confirms:
  - tool-driven path: call `finalize_plan()` and use {solution, search_capabilities, keynote}
  - manual path: call `set_plan_from_planning_complete(planner_response)` with the finalized planner output
- After plan finalization, write a manifest file to the customer's local directory documenting the recommended search strategy, including architecture decisions, model choices, index configuration, and pipeline setup.

### Phase 4: Execute
- Call `execute_plan()` to create the index, models, pipelines, and launch the UI.
- If execution fails, the user can fix the issue (e.g., restart Docker) and you
  call `retry_execution()`.

### Phase 5: Deploy to AWS OpenSearch (optional)
- After successful local execution, offer to deploy the search strategy to AWS OpenSearch.
- Use AWS MCP tools (from the aws-api-mcp-server) to provision OpenSearch Serverless or managed OpenSearch Service.
- Follow the AWS deployment steering file for detailed provisioning steps.
- Migrate the local configuration (indices, models, pipelines) to AWS.
- Configure AWS-specific settings (VPC, security, IAM roles, etc.).
- Provide the user with AWS endpoint URLs and access instructions.

### Post-Execution
- After successful `execute_plan()`/`retry_execution()`, explicitly tell the user
  how to access the UI using the `ui_access` URLs returned by the tool result.
- After Phase 5 AWS deployment, provide AWS endpoint URLs and configuration details.
- `cleanup_verification()` removes test documents when the user explicitly asks.

## Available Tools

### High-Level Workflow Tools
| Tool | Phase | Description |
|------|-------|-------------|
| `load_sample` | 1 | Load a sample document (built-in, file, URL, index, or paste) |
| `set_preferences` | 2 | Set budget, performance, query pattern, deployment preferences |
| `start_planning` | 3 | Start planning; may return `manual_planning_required` with planner prompt/input |
| `refine_plan` | 3 | Send user feedback to refine the proposal |
| `finalize_plan` | 3 | Finalize the plan when the user confirms |
| `set_plan_from_planning_complete` | 3 | Parse/store finalized planner output for manual planning mode |
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
- **CRITICAL**: You MUST ask exactly ONE preference question per message. Do NOT batch multiple preference questions together. Wait for the user's answer before asking the next question.
- Never skip Phase 1. A sample document is mandatory before planning.
- Prefer planner tools for plan generation.
- If `manual_planning_required=true`, use the returned planner prompt/input and persist via `set_plan_from_planning_complete(...)`.
- Show the planner's proposal text to the user verbatim; do not summarize it away.
- For preference questions, present numbered lists. Accept either a number or a free-text answer.
- Do not ask redundant clarification questions for items already inferred from the sample data.
- Phase 5 (AWS deployment) is optional and should only be offered after successful Phase 4 execution.

## Prerequisites
- See the [Onboarding](#onboarding) section at the top of this document.
