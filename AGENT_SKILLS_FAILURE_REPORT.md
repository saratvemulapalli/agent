# Agent Skills Failure Report: OpenSearch Search Builder

**Date:** 2026-03-09
**Context:** Converting Kiro Power (with custom MCP server) to Agent Skill for Claude Code
**Skill:** opensearch-search-builder (`opensearch-launchpad` MCP server)
**Test scenario:** Build a sparse vector search app on IMDB dataset, deploy to AWS OpenSearch Serverless

---

## Failure 1: Client Sampling (`sampling/createMessage`) Not Supported

### Affected Phases
- Phase 3 (Planning)
- Phase 4 (Execution)
- Phase 4.5 (Evaluation)

### Error
```json
{
  "error": "Planning failed in client mode.",
  "details": ["client-sampling planner failed: Method not found"],
  "manual_planning_required": true
}
```

### Root Cause
The MCP server uses `sampling/createMessage` to drive sub-agent conversations (planner, worker, evaluator). Claude Code does not support this MCP protocol method. The same applies to most MCP clients (Cursor, Windsurf, VS Code, etc.).

### Impact
Every phase that relies on sub-agent orchestration falls back to manual mode. The client LLM (Claude) must:
1. Read the returned system prompt and initial input
2. Generate the sub-agent's response (planning proposal, execution steps, evaluation scores)
3. Construct structured XML payloads (`<planning_complete>`, `<execution_report>`, `<evaluation_complete>`)
4. Commit them via `set_plan_from_planning_complete`, `set_execution_from_execution_report`, `set_evaluation_from_evaluation_complete`

This is error-prone because the client LLM must correctly follow the sub-agent's system prompt, produce valid XML in the exact required format, and handle all edge cases the sub-agent would normally handle.

### Recommendation
- **Make manual mode the primary path.** Most MCP clients don't support `sampling/createMessage`. The manual fallback should not feel like a fallback — it should be the designed-for experience.
- **Simplify the manual payloads.** Instead of requiring the client LLM to produce complex XML blocks, consider accepting structured JSON parameters directly (e.g., `finalize_plan(retrieval_method="sparse_vector", model_id="...", capabilities=[...])` instead of requiring `<planning_complete>` XML).
- **Provide explicit examples** in the manual fallback response showing what a valid payload looks like for common scenarios.

---

## Failure 2: State Machine Not Advancing After Manual Commits

### Affected Phase
- Phase 5 (AWS Deployment)

### Error
```json
{
  "error": "AWS deployment requires successful local execution (Phase 4). Current phase: GATHER_INFO"
}
```

### Root Cause
After manually executing all plan steps (model creation, index creation, pipeline setup, verification, UI launch) and calling the individual MCP tools directly, the MCP server's internal state machine still showed `GATHER_INFO`. The state only advances when `set_execution_from_execution_report` is called with a valid `<execution_report>` block.

### Impact
The client LLM successfully completed all execution steps but forgot to commit the execution report. This blocked `prepare_aws_deployment()` because it requires Phase 4 to be complete.

### Why This Happened
In manual execution mode, the MCP server returns a `worker_system_prompt` and `worker_initial_input` expecting the client LLM to follow the worker agent's full protocol — including producing the `<execution_report>` block at the end. But the client LLM executed steps one-by-one using individual MCP tools and never produced the report, because it was acting as an orchestrator, not as the worker agent.

### Recommendation
- **Auto-advance state on tool success.** If the client calls `apply_capability_driven_verification` and `launch_search_ui` successfully, the state should advance to execution-complete without requiring an explicit report commit.
- **Or: track step completion internally.** The MCP server already knows which tools were called and whether they succeeded. It could infer execution status from tool call history.
- **At minimum:** Surface a clear warning in tool responses when the state hasn't been committed. For example, after `launch_search_ui` succeeds, the response could say: "Note: Call `set_execution_from_execution_report(...)` to finalize execution state."

---

## Failure 3: `prepare_aws_deployment()` Returns Wrong Strategy

### Affected Phase
- Phase 5 (AWS Deployment)

### Observed Behavior
```json
{
  "deployment_target": "serverless",
  "search_strategy": "dense_vector"
}
```

### Expected Behavior
```json
{
  "deployment_target": "serverless",
  "search_strategy": "sparse_vector"
}
```

### Root Cause
The plan committed via `set_plan_from_planning_complete` clearly specified "Retrieval Method: Sparse Vector", but `prepare_aws_deployment()` returned `search_strategy: "dense_vector"`. The strategy normalization logic either:
1. Defaults to `dense_vector` when it can't parse the plan text
2. Doesn't correctly extract the retrieval method from the manually committed plan
3. Uses a different field or mapping than what the manual commit path provides

### Impact
- The steering file routing becomes incorrect (Dense Vector Path vs Neural Sparse Path)
- The state file template includes wrong defaults
- The client LLM may follow the wrong deployment guide if it trusts the returned strategy

### Recommendation
- **Add integration tests** for the manual plan commit → prepare deployment path, specifically for each retrieval method (BM25, dense, sparse, hybrid, agentic).
- **Return the strategy as stored** rather than re-inferring it. The plan text is already parsed and stored — use that.
- **Validate on commit.** When `set_plan_from_planning_complete` is called, echo back the parsed retrieval method so the client can verify it was understood correctly.

---

## Failure 4: Required AWS MCP Servers Silently Fail

### Affected Phase
- Phase 5 (AWS Deployment)

### Configuration
```json
{
  "awslabs.aws-api-mcp-server": {
    "command": "uvx",
    "args": ["awslabs.aws-api-mcp-server@latest"]
  },
  "opensearch-mcp-server": {
    "command": "uvx",
    "args": ["opensearch-mcp-server@latest"]
  }
}
```

### Observed Behavior
Both servers were configured in `.mcp.json` but no tools from either server appeared in the tool registry. `ToolSearch` queries for `+awslabs` and `+opensearch-mcp` returned "No matching deferred tools found". No error was surfaced to the user or the agent.

### Root Cause
Likely one of:
1. The packages failed to install via `uvx` (dependency conflicts, network issues)
2. The packages exist but expose tools under different naming conventions
3. The MCP server processes started but crashed during initialization
4. Claude Code silently drops MCP servers that fail to connect

### Impact
The entire AWS deployment had to be done via AWS CLI (`aws` and `awscurl`) as a workaround. This works but bypasses the structured MCP workflow the skill was designed around.

### Recommendation
- **Document exact package names and versions** that are known to work. Include a verification step in the skill (e.g., "Run `uvx awslabs.aws-api-mcp-server@latest --help` to verify installation").
- **Provide a health-check tool** in the opensearch-launchpad MCP server that validates whether required companion MCP servers are reachable (e.g., `check_mcp_dependencies()`).
- **Document the AWS CLI fallback path** as a first-class alternative, since MCP server availability is not guaranteed across IDE clients.
- **Consider bundling AWS operations** into the opensearch-launchpad MCP server itself (using boto3) rather than depending on external MCP servers. This eliminates the multi-server dependency.

---

## Failure 5: AOSS-Specific API Constraints Not Handled

### Affected Phase
- Phase 5 (AWS Deployment — document ingestion and search)

### Sub-issues

#### 5a: Index Creation Method
**What happened:** I tried to create the index by PUTting mappings to the collection endpoint (`awscurl PUT https://<endpoint>/imdb-movies`).
**Correct approach:** Use `aws opensearchserverless create-index --id <collection-id> --index-name <name> --index-schema '<mappings>'` (the control plane API).
**Why it matters:** The data plane PUT may work for basic indices, but `create-index` API is required for features like `semantic_enrichment`.

#### 5b: Document ID Not Supported
**What happened:** I tried `POST /<index>/_doc/tt0000001` (specifying document ID).
**Error:** Silent failure — the request appeared to succeed but no document was indexed.
**Correct approach:** AOSS only supports `POST /<index>/_doc` (auto-generated IDs). Document-by-ID operations (`PUT /<index>/_doc/<id>`, `GET /<index>/_doc/<id>`, `DELETE /<index>/_doc/<id>`) are not supported.

#### 5c: Refresh Latency Not Accounted For
**What happened:** Searched immediately after indexing and got 0 results. Assumed the index was broken.
**Root cause:** AOSS SEARCH collections have a ~10 second refresh latency. Documents are not searchable immediately after ingestion.
**Correct approach:** Wait at least 10 seconds after ingestion before searching. Also verify the ingest response shows `"result": "created"`.

#### 5d: Shard Count Shows 0
**What happened:** Search responses showed `"_shards": {"total": 0, "successful": 0}` which looked like an error.
**Root cause:** This is normal AOSS behavior. AOSS manages shards internally and doesn't expose shard counts in the same way as self-managed OpenSearch.

### Recommendation
- **Add an AOSS constraints section** to the knowledge base and steering files covering:
  - No document-by-ID operations
  - Use control plane API for index creation (especially for semantic_enrichment)
  - 10s refresh latency for SEARCH collections, 30s for VECTORSEARCH
  - Shard metadata differences
  - No `_cat` API support
  - No `GET /` (info) endpoint
- **Add AOSS-specific examples** to the steering files showing correct `awscurl` commands or `aws opensearchserverless` CLI commands.
- **Consider an AOSS validation tool** in the MCP server that checks common mistakes before execution (e.g., "You're using doc-by-ID which isn't supported on AOSS").

---

## Failure 6: Automatic Semantic Enrichment Not Used Despite Being Documented

### Affected Phase
- Phase 5 (AWS Deployment — index creation)

### What Happened
When deploying sparse vector to AOSS, I initially attempted to:
1. Create an index with `rank_features` fields manually
2. Set up a sparse encoding model on AOSS
3. Create an ingest pipeline with a `sparse_encoding` processor

This replicated the local OpenSearch pattern, ignoring AOSS's built-in semantic enrichment.

### What Should Have Happened
The skill's own steering file (`aws-serverless-02-deploy-search.md`, lines 28-59) clearly documents the Neural Sparse Path:

```json
{
  "mappings": {
    "properties": {
      "<text-field>": {
        "type": "text",
        "semantic_enrichment": {
          "status": "ENABLED",
          "language_options": "english"
        }
      }
    }
  }
}
```

With explicit notes:
- "System automatically deploys sparse model, creates ingest/search pipelines"
- "Standard match queries are automatically rewritten to neural sparse queries"
- "No manual model or pipeline management required"

### Why It Was Missed

1. **Strategy mismatch:** `prepare_aws_deployment()` returned `search_strategy: "dense_vector"` (Failure 3), so the "Route by Strategy" section in the steering file pointed to the Dense Vector Path, not the Neural Sparse Path.
2. **Late file reading:** I didn't read the steering files until after the user asked about semantic enrichment. The skill workflow should have directed me to read them immediately after `prepare_aws_deployment()`.
3. **No proactive routing:** The steering file contains three paths (Neural Sparse, Dense Vector, BM25) but the MCP server doesn't tell the client which path to follow — it relies on the client reading the full file and selecting the right section based on the (incorrectly returned) strategy.

### Recommendation
- **Fix the strategy return** (Failure 3) — this is the primary blocker.
- **Return the specific steering section**, not the full file. Instead of returning `steering_files: ["serverless-02-deploy-search.md"]`, return the relevant section content directly (e.g., the Neural Sparse Path instructions).
- **Or: provide a deployment execution tool** that takes the strategy and handles the AOSS-specific setup internally (similar to how `execute_plan()` works for local deployment). This would eliminate the need for the client to parse steering files.

---

## Failure 7: `connect_search_ui_to_endpoint` Fails for AOSS

### Affected Phase
- Post-deployment (connecting Search Builder UI to AWS)

### Error
```
"Error: Could not connect to xnx7tny5kttjwybcwct2.us-west-2.aoss.amazonaws.com:443.
Verify the endpoint is active and credentials are correct.
Search UI remains connected to the previous endpoint."
```

### Root Cause
The `_can_connect()` function in `opensearch_ops_tools.py` (line 667) uses this connectivity check chain:

1. `client.info()` → **Fails** (AOSS returns 404 on `GET /`)
2. `client.cat.indices(format="json")` → **Fails** (AOSS doesn't support `_cat` APIs)
3. Returns `connected=false`

Even after patching to add a third fallback (`client.search(index="*", body={"size": 0})`), the connection still failed. The likely deeper issue is that `boto3.Session()` at line 1166 creates a session without explicit credentials — it relies on the default credential chain. Even though AWS env vars are set in `.mcp.json`, they may not propagate correctly to the boto3 session in all cases (e.g., if the MCP server process inherits a different environment).

### Code Path
```python
# opensearch_ops_tools.py:1161-1174
if aws_region and aws_service:
    session = boto3.Session()                    # No explicit credentials
    credentials = session.get_credentials()       # May return None or stale creds
    auth = AWSV4SignerAuth(credentials, aws_region, aws_service)
    kwargs["http_auth"] = auth
    kwargs["connection_class"] = RequestsHttpConnection
```

### Recommendation

**Fix `_can_connect` for AOSS:**
```python
def _can_connect(opensearch_client: OpenSearch) -> tuple[bool, bool]:
    try:
        opensearch_client.info()
        return True, False
    except Exception as e:
        lowered = normalize_text(e).lower()
        if "404" in lowered or "notfounderror" in lowered:
            # AOSS returns 404 on GET /. Try a search as connectivity check.
            try:
                opensearch_client.search(index="*", body={"size": 0})
                return True, False
            except Exception as search_e:
                search_lowered = normalize_text(search_e).lower()
                # Any HTTP response (even 403) confirms network connectivity.
                if any(code in search_lowered for code in ["403", "forbidden", "security_exception"]):
                    return True, False
        auth_failure = any(token in lowered for token in _AUTH_FAILURE_TOKENS)
        return False, auth_failure
```

**Fix credential propagation:**
```python
# Accept explicit credentials or pass env vars to boto3.Session
if aws_region and aws_service:
    session = boto3.Session(
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        aws_session_token=os.environ.get("AWS_SESSION_TOKEN"),
        region_name=aws_region,
    )
```

**Or: accept credentials as tool parameters:**
```python
def connect_search_ui_to_endpoint(
    endpoint: str,
    ...
    aws_access_key_id: str = "",
    aws_secret_access_key: str = "",
    aws_session_token: str = "",
):
```

---

## Failure 8: MCP Server Cannot Be Restarted Programmatically

### Affected Phase
- Debugging (multiple occurrences)

### What Happened
After updating `.mcp.json` (adding AWS credentials) and after patching source code (`_can_connect`), the changes required an MCP server restart. Claude Code does not provide a tool to restart MCP servers, so each time I had to ask the user to manually restart or reload.

### Impact
- Slowed down the debugging cycle significantly
- Each code fix required a manual user action before it could be tested
- The user had to run `/mcp` or restart Claude Code multiple times

### Recommendation
- This is a Claude Code platform limitation, not an Agent Skill issue.
- **However:** The skill could minimize restarts by:
  - Making credential configuration runtime (via tool parameters) rather than environment-based
  - Avoiding code changes that require server restarts — instead, make behavior configurable via tool parameters or state

---

## Failure 9: AWS Documentation MCP Server Not Used Despite Skill Instructions

### Affected Phase
- Phase 5 (AWS Deployment — all steps)

### What the Skill Says
From `SKILL.md`, Phase 5:
> **Required additional MCP servers for AWS deployment:**
> - `awslabs.aws-api-mcp-server`
> - `opensearch-mcp-server`
> - `awslabs.aws-documentation-mcp-server` (recommended)

### What Happened
When configuring MCP servers for AWS deployment, I only added `awslabs.aws-api-mcp-server` and `opensearch-mcp-server` to `.mcp.json`. I completely skipped `awslabs.aws-documentation-mcp-server` because it was marked "(recommended)" rather than "required".

After the first two MCP servers silently failed to connect (Failure 4), I abandoned the MCP server approach entirely and fell back to AWS CLI — without even attempting to add the documentation server.

### Impact
The AWS documentation MCP server would have provided access to up-to-date AOSS documentation, which would have prevented or mitigated multiple downstream failures:

| Downstream Failure | What AWS Docs Would Have Provided |
|---|---|
| **Failure 5a:** Wrong index creation method | AOSS `CreateIndex` API documentation showing the control plane approach |
| **Failure 5b:** Document-by-ID not supported | AOSS supported operations reference listing the restriction |
| **Failure 5c:** Refresh latency not known | AOSS collection types documentation specifying 10s/30s refresh |
| **Failure 6:** Semantic enrichment missed | Semantic enrichment feature documentation and examples |

In total, at least 4 of the user corrections during AWS deployment could have been avoided if the documentation MCP server had been configured and consulted.

### Why This Matters for Agent Skills
This is a **skill instruction adherence failure** — the skill clearly listed the server and its purpose, but the agent (Claude) made a judgment call to skip it based on the "(recommended)" qualifier. The skill's instructions were correct; the agent didn't follow them.

This also reveals a broader pattern: when one dependency fails (Failure 4), the agent may prematurely abandon the entire dependency category rather than troubleshooting or trying alternatives.

### Recommendation
- **Strengthen the language** in the skill instructions. Instead of "(recommended)", use language like "Required for AOSS deployment" or "Required unless using AWS CLI fallback". Agents tend to skip items marked as optional.
- **Make the docs server a hard prerequisite** for Phase 5. The `prepare_aws_deployment()` response could include a pre-flight check: "Verify these MCP servers are connected before proceeding: [list]".
- **Add explicit consultation instructions** in the steering files. For example, before index creation: "Use `awslabs.aws-documentation-mcp-server` to look up the latest AOSS CreateIndex API reference and supported operations."
- **Alternatively: embed critical AOSS constraints directly** in the steering files or knowledge base, so the agent doesn't need an external documentation lookup for essential information. The steering files already partially do this but miss key constraints (doc-by-ID, refresh latency, `_cat` API unsupported).

---

## Summary Table

| # | Failure | Severity | Category |
|---|---------|----------|----------|
| 1 | `sampling/createMessage` not supported | **High** | MCP Protocol Compatibility |
| 2 | State machine doesn't advance after manual tool calls | **High** | State Management |
| 3 | `prepare_aws_deployment()` returns wrong strategy | **High** | State Management |
| 4 | AWS MCP servers silently fail to connect | **Medium** | Dependency Management |
| 5 | AOSS API constraints not handled (doc-by-ID, refresh, etc.) | **Medium** | Knowledge Gap |
| 6 | Semantic enrichment not picked up despite being in steering files | **High** | Routing / Strategy |
| 7 | `connect_search_ui_to_endpoint` fails for AOSS | **High** | Connectivity / Auth |
| 8 | MCP server can't be restarted programmatically | **Low** | Platform Limitation |
| 9 | AWS docs MCP server not used despite skill instructions | **Medium** | Skill Instruction Adherence |

### Priority Fix Order
1. **Failures 1 + 2:** Make manual mode first-class with auto-advancing state
2. **Failure 3 + 6:** Fix strategy persistence and steering file routing
3. **Failure 7:** Fix AOSS connectivity check and credential propagation
4. **Failure 5 + 9:** Add AOSS constraints to knowledge base; strengthen docs server requirement
5. **Failure 4:** Bundle AWS operations or document CLI fallback
