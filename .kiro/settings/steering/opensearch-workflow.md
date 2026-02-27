---
description: "OpenSearch solution architect workflow. Triggers for: build search app, semantic search, vector search, hybrid search, OpenSearch, index setup, search architecture, large-scale search, RAG, retrieval, embeddings, knn, neural search, document search, search relevance, BM25, dense vector, sparse vector, sagemaker, bedrock"
inclusion: always
---

## OpenSearch Search Builder — MCP Workflow

You have access to the `opensearch-orchestrator` MCP server.
Follow these phases in order using its tools:

### Phase 1: Collect Sample Document
- Ask the user how they want to provide a sample document. Present as a numbered list:

  **How would you like to provide a sample document?**
  1. Use built-in IMDB dataset
  2. Load from a local file
  3. Load from a URL
  4. Load from a localhost OpenSearch index
  5. Paste JSON directly

- Based on the user's choice, call `load_sample(source_type, source_value)`:
  - source_type: `builtin_imdb`, `local_file`, `url`, `localhost_index`, or `paste`
  - source_value: file path, URL, index name, or JSON content (empty string for builtin_imdb)
- The result includes `inferred_text_fields` and `text_search_required`.

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
- Call `set_preferences(budget, performance, query_pattern, deployment_preference)`.

### Phase 3: Plan
- Call `start_planning()` to get the initial architecture proposal.
- If `start_planning()` returns `manual_planning_required=true`, use
  `manual_planner_system_prompt` + `manual_planner_initial_input` to run planner turns with the client LLM.
- Present the proposal to the user.
- If the user has feedback, refine the proposal (with tools when available, otherwise directly with the client LLM) and repeat.
- When the user confirms:
  - tool-driven path: call `finalize_plan()` and use {solution, search_capabilities, keynote}
  - manual path: call `set_plan_from_planning_complete(planner_response)` with the finalized planner output

### Phase 4: Execute
- Call `execute_plan()` to create the index, models, pipelines, and launch the UI.
- If execution fails, the user can fix the issue (e.g., restart Docker) and you
  call `retry_execution()`.

### Post-Execution
- After successful `execute_plan()`/`retry_execution()`, explicitly tell the user
  how to access the UI using the `ui_access` URLs returned by the tool result.
- `cleanup_verification()` removes test documents when the user explicitly asks.

### Rules
- **CRITICAL**: You MUST ask exactly ONE preference question per message. Do NOT batch multiple preference questions together. Wait for the user's answer before asking the next question.
- Never skip Phase 1. A sample document is mandatory before planning.
- Prefer planner tools for plan generation.
- If `manual_planning_required=true`, use the returned planner prompt/input and persist via `set_plan_from_planning_complete(...)`.
- Show the planner's proposal text to the user verbatim; do not summarize it away.
- For preference questions, present numbered lists. Accept either a number or a free-text answer.
- Do not ask redundant clarification questions for items already inferred from the sample data.
