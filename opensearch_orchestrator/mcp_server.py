# /// script
# dependencies = ["mcp", "opensearch-py", "strands-agents"]
# ///

"""MCP server exposing the OpenSearch orchestrator workflow as phase tools.

Clients (Cursor, Claude Desktop, generic MCP) call these tools in order:
  load_sample -> set_preferences -> start_planning -> refine_plan/finalize_plan -> execute_plan

Low-level domain tools are also exposed for advanced use.
"""

from __future__ import annotations

if __package__ in {None, ""}:
    from pathlib import Path
    import sys

    _SCRIPT_EXECUTION_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
    if _SCRIPT_EXECUTION_PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _SCRIPT_EXECUTION_PROJECT_ROOT)

import errno
import json
import sys

import anyio
from mcp.server.fastmcp import FastMCP

from opensearch_orchestrator.orchestrator import (
    SessionState,
    _infer_semantic_text_fields,
    _capture_sample_from_result,
    _build_budget_preference_note,
    _build_performance_preference_note,
    _build_semantic_query_pattern_preference_note,
    _build_model_deployment_preference_note,
    _build_prefix_wildcard_requirement_note,
    _build_text_search_use_case_note,
    _build_hybrid_weight_profile_note,
    _augment_worker_context_with_source,
    _run_worker_agent_with_state,
    _is_semantic_dominant_query_pattern,
    _reset_session_state,
    _clear_orchestrator_sample_state,
    _BUDGET_OPTION_FLEXIBLE,
    _BUDGET_OPTION_COST_SENSITIVE,
    _PERFORMANCE_OPTION_SPEED,
    _PERFORMANCE_OPTION_BALANCED,
    _PERFORMANCE_OPTION_ACCURACY,
    _QUERY_PATTERN_OPTION_MOSTLY_EXACT,
    _QUERY_PATTERN_OPTION_BALANCED,
    _QUERY_PATTERN_OPTION_MOSTLY_SEMANTIC,
    _MODEL_DEPLOYMENT_OPTION_OPENSEARCH_NODE,
    _MODEL_DEPLOYMENT_OPTION_SAGEMAKER_ENDPOINT,
    _MODEL_DEPLOYMENT_OPTION_EXTERNAL_EMBEDDING_API,
    _HYBRID_WEIGHT_OPTION_SEMANTIC,
    _HYBRID_WEIGHT_OPTION_BALANCED,
    _HYBRID_WEIGHT_OPTION_LEXICAL,
    _DEFAULT_QUERY_FEATURES_NOTE,
    _MODEL_DEPLOYMENT_SCOPE_NOTE,
    _PERFORMANCE_PRIORITY_SCOPE_NOTE,
    _SAMPLE_FINAL_TRUTH_NOTE,
    _SEMANTIC_EXPANSION_EXPLANATION_NOTE,
    _NATURAL_LANGUAGE_CONCEPT_SEARCH_NOTE,
    _MAPPING_CLARITY_FEEDBACK_NOTE,
    _DEFAULT_SPECIFIC_USE_CASES_NOTE,
    _DEFAULT_QUERY_SUPPORT_SCOPE_NOTE,
    _DEFAULT_DASHBOARD_REQUIREMENT_NOTE,
    _DEFAULT_REALTIME_REQUIREMENT_NOTE,
    _DEFAULT_CUSTOM_REQUIREMENTS_NOTE,
    _RESUME_WORKER_MARKER,
)
from opensearch_orchestrator.planning_session import PlanningSession
from opensearch_orchestrator.scripts.shared import Phase, get_last_worker_run_state
from opensearch_orchestrator.scripts.tools import (
    BUILTIN_IMDB_SAMPLE_PATH,
    submit_sample_doc,
    submit_sample_doc_from_local_file,
    submit_sample_doc_from_localhost_index,
    submit_sample_doc_from_url,
    get_sample_docs_for_verification,
    read_knowledge_base,
    read_dense_vector_models,
    read_sparse_vector_models,
    search_opensearch_org,
)
from opensearch_orchestrator.scripts.opensearch_ops_tools import (
    SEARCH_UI_HOST,
    SEARCH_UI_PORT,
    create_index,
    create_and_attach_pipeline,
    create_bedrock_embedding_model,
    create_local_pretrained_model,
    index_doc,
    index_verification_docs,
    delete_doc,
    cleanup_verification_docs,
    apply_capability_driven_verification,
    preview_capability_driven_verification,
    launch_search_ui,
    cleanup_ui_server,
    set_search_ui_suggestions,
)
from opensearch_orchestrator.worker import worker_agent as worker_agent_impl

# -------------------------------------------------------------------------
# Workflow prompt (shared by MCP prompt and Cursor rule)
# -------------------------------------------------------------------------

WORKFLOW_PROMPT = """\
You are an OpenSearch Solution Architect assistant.
Use the opensearch-orchestrator MCP tools to guide the user from requirements to a running OpenSearch setup.

## Workflow Phases

### Phase 1: Collect Sample Document (mandatory first step)
- Call `load_sample(source_type, source_value)`.
  - source_type: "builtin_imdb" | "local_file" | "url" | "localhost_index" | "paste"
  - source_value: file path, URL, index name, or pasted JSON content (empty string for builtin_imdb)
- The result includes `inferred_text_fields` and `text_search_required`.
- A sample document is required before any planning or execution.

### Phase 2: Gather Preferences
- Ask the user about:
  - **Budget**: flexible or cost-sensitive
  - **Performance priority**: speed-first, balanced, or accuracy-first
  - **Query pattern**: mostly-exact, balanced, or mostly-semantic
  - **Deployment preference** (only if query pattern is mostly-semantic): opensearch-node, sagemaker-endpoint, or external-embedding-api
- Use fixed-option format for each question, not free-text.
- Call `set_preferences(budget, performance, query_pattern, deployment_preference)` with the collected values.

### Phase 3: Plan
- Call `start_planning()` to get an initial architecture proposal from the planner agent.
- Present the proposal to the user verbatim (do not summarize it away).
- If the user has feedback or questions, call `refine_plan(user_feedback)`. Repeat as needed.
- When the user confirms, call `finalize_plan()`.
  This returns {solution, search_capabilities, keynote}.

### Phase 4: Execute
- Call `execute_plan()` to create the index, models, pipelines, and launch the search UI.
- If execution fails, the user can fix the issue (e.g., restart Docker) and you call `retry_execution()`.

### Post-Execution
- After successful `execute_plan()`/`retry_execution()`, explicitly tell the user
  how to access the UI using the `ui_access` URLs returned by the tool result.
- `cleanup_verification()` removes test/verification documents when the user explicitly asks.

## Rules
- Never skip Phase 1. A sample document is mandatory before planning.
- Do not generate the solution plan yourself; always delegate to the planner tools.
- Show the planner's proposal text to the user verbatim; do not summarize it away.
- For preference questions, use fixed-option format, not free-text.
- Do not ask redundant clarification questions for items already inferred from the sample data.
"""

# -------------------------------------------------------------------------
# Session state (single session per stdio connection)
# -------------------------------------------------------------------------


class OrchestratorSession:
    def __init__(self) -> None:
        self.state = SessionState()
        self.phase = Phase.COLLECT_SAMPLE
        self.planning: PlanningSession | None = None
        self.plan_result: dict | None = None

    def reset(self) -> None:
        _reset_session_state(self.state)
        self.phase = Phase.COLLECT_SAMPLE
        self.planning = None
        self.plan_result = None


_session = OrchestratorSession()

# -------------------------------------------------------------------------
# MCP server
# -------------------------------------------------------------------------

mcp = FastMCP("OpenSearch Orchestrator", json_response=True)

# -------------------------------------------------------------------------
# Phase tools
# -------------------------------------------------------------------------

VALID_SOURCE_TYPES = {"builtin_imdb", "local_file", "url", "localhost_index", "paste"}
VALID_BUDGET = {_BUDGET_OPTION_FLEXIBLE, _BUDGET_OPTION_COST_SENSITIVE}
VALID_PERFORMANCE = {_PERFORMANCE_OPTION_SPEED, _PERFORMANCE_OPTION_BALANCED, _PERFORMANCE_OPTION_ACCURACY}
VALID_QUERY_PATTERN = {_QUERY_PATTERN_OPTION_MOSTLY_EXACT, _QUERY_PATTERN_OPTION_BALANCED, _QUERY_PATTERN_OPTION_MOSTLY_SEMANTIC}
VALID_DEPLOYMENT = {
    _MODEL_DEPLOYMENT_OPTION_OPENSEARCH_NODE,
    _MODEL_DEPLOYMENT_OPTION_SAGEMAKER_ENDPOINT,
    _MODEL_DEPLOYMENT_OPTION_EXTERNAL_EMBEDDING_API,
}
QUERY_PATTERN_TO_HYBRID_WEIGHT = {
    _QUERY_PATTERN_OPTION_MOSTLY_EXACT: _HYBRID_WEIGHT_OPTION_LEXICAL,
    _QUERY_PATTERN_OPTION_BALANCED: _HYBRID_WEIGHT_OPTION_BALANCED,
    _QUERY_PATTERN_OPTION_MOSTLY_SEMANTIC: _HYBRID_WEIGHT_OPTION_SEMANTIC,
}


def _build_ui_access_payload() -> dict[str, object]:
    public_host = "localhost" if SEARCH_UI_HOST in {"0.0.0.0", "::"} else SEARCH_UI_HOST
    urls: list[str] = []
    for host in (public_host, "127.0.0.1", "localhost"):
        url = f"http://{host}:{SEARCH_UI_PORT}"
        if url not in urls:
            urls.append(url)
    return {
        "primary_url": urls[0],
        "alternate_urls": urls[1:],
    }


@mcp.tool()
def load_sample(source_type: str, source_value: str = "") -> dict:
    """Load a sample document for OpenSearch solution design.
    This MUST be called first before any planning or execution.

    Args:
        source_type: One of "builtin_imdb", "local_file", "url",
                     "localhost_index", or "paste".
        source_value: File path, URL, index name, or pasted JSON content.
                      Use empty string for builtin_imdb.

    Returns:
        dict with sample_doc, inferred_text_fields, text_search_required,
        and status message.
    """
    if source_type not in VALID_SOURCE_TYPES:
        return {"error": f"Invalid source_type '{source_type}'. Must be one of: {sorted(VALID_SOURCE_TYPES)}"}

    state = _session.state
    _clear_orchestrator_sample_state(state)

    if source_type == "builtin_imdb":
        result = submit_sample_doc_from_local_file(BUILTIN_IMDB_SAMPLE_PATH)
    elif source_type == "local_file":
        if not source_value:
            return {"error": "source_value is required for local_file source_type (provide a file path)."}
        result = submit_sample_doc_from_local_file(source_value)
    elif source_type == "url":
        if not source_value:
            return {"error": "source_value is required for url source_type (provide a URL)."}
        result = submit_sample_doc_from_url(source_value)
    elif source_type == "localhost_index":
        result = submit_sample_doc_from_localhost_index(source_value)
    else:  # paste
        if not source_value:
            return {"error": "source_value is required for paste source_type (provide JSON content)."}
        result = submit_sample_doc(source_value)

    loaded = _capture_sample_from_result(state, result)

    if not loaded:
        if isinstance(result, str) and result.startswith("Error:"):
            return {"error": result}
        return {"error": f"Failed to load sample document. Raw result: {result}"}

    parsed_result = json.loads(result)
    sample_payload = parsed_result["sample_doc"]
    state.inferred_semantic_text_fields = _infer_semantic_text_fields(sample_payload)
    state.inferred_text_search_required = bool(state.inferred_semantic_text_fields)

    source_is_localhost = bool(parsed_result.get("source_localhost_index"))
    if source_is_localhost:
        state.source_index_name = str(parsed_result.get("source_index_name", "")).strip() or None
        raw_doc_count = parsed_result.get("source_index_doc_count")
        if isinstance(raw_doc_count, int) and not isinstance(raw_doc_count, bool):
            state.source_index_doc_count = max(0, raw_doc_count)

    _session.phase = Phase.GATHER_INFO

    return {
        "status": parsed_result.get("status", "Sample loaded."),
        "sample_doc": sample_payload,
        "inferred_text_fields": state.inferred_semantic_text_fields,
        "text_search_required": state.inferred_text_search_required,
        "source_index_name": state.source_index_name,
        "source_index_doc_count": state.source_index_doc_count,
    }


@mcp.tool()
def set_preferences(
    budget: str = "flexible",
    performance: str = "balanced",
    query_pattern: str = "balanced",
    deployment_preference: str = "",
) -> dict:
    """Set user preferences for budget, performance, query pattern, and deployment.
    Call this after load_sample and before start_planning.

    Args:
        budget: "flexible" or "cost-sensitive".
        performance: "speed-first", "balanced", or "accuracy-first".
        query_pattern: "mostly-exact", "balanced", or "mostly-semantic".
        deployment_preference: "opensearch-node", "sagemaker-endpoint", or
            "external-embedding-api". Only needed when query_pattern is
            "mostly-semantic". Defaults to "opensearch-node".

    Returns:
        dict confirming stored preferences and generated context notes.
    """
    state = _session.state
    if state.sample_doc_json is None:
        return {"error": "No sample document loaded. Call load_sample first."}

    budget_val = budget if budget in VALID_BUDGET else _BUDGET_OPTION_FLEXIBLE
    perf_val = performance if performance in VALID_PERFORMANCE else _PERFORMANCE_OPTION_BALANCED
    qp_val = query_pattern if query_pattern in VALID_QUERY_PATTERN else _QUERY_PATTERN_OPTION_BALANCED
    hw_val = QUERY_PATTERN_TO_HYBRID_WEIGHT.get(qp_val, _HYBRID_WEIGHT_OPTION_BALANCED)

    state.budget_preference = budget_val
    state.performance_priority = perf_val
    state.hybrid_weight_profile = hw_val

    if state.inferred_text_search_required:
        state.prefix_wildcard_enabled = False

    if _is_semantic_dominant_query_pattern(hw_val) and state.inferred_text_search_required:
        dep_val = deployment_preference if deployment_preference in VALID_DEPLOYMENT else _MODEL_DEPLOYMENT_OPTION_OPENSEARCH_NODE
        state.model_deployment_preference = dep_val
    else:
        state.model_deployment_preference = None

    return {
        "budget": state.budget_preference,
        "performance": state.performance_priority,
        "query_pattern": qp_val,
        "hybrid_weight_profile": state.hybrid_weight_profile,
        "deployment_preference": state.model_deployment_preference,
        "context_notes": _build_context_notes(state),
    }


@mcp.tool()
def start_planning(additional_context: str = "") -> dict:
    """Start the solution planning phase. Returns the planner's initial proposal.
    Call this after set_preferences.

    Args:
        additional_context: Optional extra context to include.

    Returns:
        dict with response text, is_complete flag, and result (if complete).
    """
    state = _session.state
    if state.sample_doc_json is None:
        return {"error": "No sample loaded. Call load_sample first."}

    context = _build_planning_context(state, additional_context)
    _session.planning = PlanningSession()
    result = _session.planning.start(context)

    if result.get("is_complete") and result.get("result"):
        _session.plan_result = result["result"]

    return result


@mcp.tool()
def refine_plan(user_feedback: str) -> dict:
    """Send user feedback to the planner and get a refined proposal.
    Call after start_planning. Repeat as needed.

    Args:
        user_feedback: User's feedback, questions, or change requests.

    Returns:
        dict with response text, is_complete flag, and result (if complete).
    """
    if _session.planning is None:
        return {"error": "No planning session active. Call start_planning first."}

    result = _session.planning.send(user_feedback)

    if result.get("is_complete") and result.get("result"):
        _session.plan_result = result["result"]

    return result


@mcp.tool()
def finalize_plan() -> dict:
    """Force the planner to finalize and return the structured plan.
    Call when the user confirms they are satisfied with the proposal.

    Returns:
        dict with solution, search_capabilities, and keynote.
    """
    if _session.planning is None:
        return {"error": "No planning session active. Call start_planning first."}

    result = _session.planning.finalize()

    if result.get("is_complete") and result.get("result"):
        _session.plan_result = result["result"]

    return result


@mcp.tool()
def execute_plan(additional_context: str = "") -> dict:
    """Execute the finalized solution plan (create index, models, pipelines, UI).
    Call after finalize_plan.

    Args:
        additional_context: Optional extra instructions for the worker.

    Returns:
        dict with worker execution report and UI access URLs.
    """
    if _session.plan_result is None:
        return {"error": "No finalized plan available. Complete the planning phase first."}

    plan = _session.plan_result
    solution = plan.get("solution", "")
    capabilities = plan.get("search_capabilities", "")
    keynote = plan.get("keynote", "")

    worker_context = f"Solution:\n{solution}\n\nSearch Capabilities:\n{capabilities}\n\nKeynote:\n{keynote}"
    if additional_context:
        worker_context += f"\n\n{additional_context}"

    worker_result = _run_worker_agent_with_state(_session.state, worker_context)
    _session.phase = Phase.DONE

    return {
        "execution_report": worker_result,
        "ui_access": _build_ui_access_payload(),
    }


@mcp.tool()
def retry_execution() -> dict:
    """Retry execution from the last failed step.
    Call after execute_plan fails and the user has fixed the issue.

    Returns:
        dict with worker execution report and UI access URLs.
    """
    worker_state = get_last_worker_run_state()
    recovery_context = str(worker_state.get("context", "")).strip() if isinstance(worker_state, dict) else ""

    if not recovery_context:
        return {"error": "No checkpoint context available. Run execute_plan first."}

    worker_result = _run_worker_agent_with_state(
        _session.state,
        f"{_RESUME_WORKER_MARKER}\n{recovery_context}",
    )

    latest_state = get_last_worker_run_state()
    latest_status = str(latest_state.get("status", "")).lower()
    _session.phase = Phase.DONE if latest_status == "success" else Phase.EXEC_FAILED

    return {
        "execution_report": worker_result,
        "ui_access": _build_ui_access_payload(),
    }


@mcp.tool()
def cleanup_verification() -> str:
    """Remove verification/test documents from the OpenSearch index.
    Call only when the user explicitly asks for cleanup.

    Returns:
        str: Cleanup result message.
    """
    return cleanup_verification_docs()


# -------------------------------------------------------------------------
# Context building helpers
# -------------------------------------------------------------------------


def _build_context_notes(state: SessionState) -> str:
    """Build the full requirement notes context block from session state."""
    notes: list[str] = [
        _DEFAULT_QUERY_FEATURES_NOTE,
        _build_text_search_use_case_note(
            state.inferred_text_search_required,
            state.inferred_semantic_text_fields,
        ),
        _MODEL_DEPLOYMENT_SCOPE_NOTE,
        _PERFORMANCE_PRIORITY_SCOPE_NOTE,
        _SAMPLE_FINAL_TRUTH_NOTE,
        _SEMANTIC_EXPANSION_EXPLANATION_NOTE,
        _MAPPING_CLARITY_FEEDBACK_NOTE,
        _DEFAULT_SPECIFIC_USE_CASES_NOTE,
        _DEFAULT_QUERY_SUPPORT_SCOPE_NOTE,
        _DEFAULT_DASHBOARD_REQUIREMENT_NOTE,
        _DEFAULT_REALTIME_REQUIREMENT_NOTE,
        _DEFAULT_CUSTOM_REQUIREMENTS_NOTE,
    ]

    if (
        state.inferred_text_search_required
        and _is_semantic_dominant_query_pattern(state.hybrid_weight_profile)
    ):
        notes.append(_NATURAL_LANGUAGE_CONCEPT_SEARCH_NOTE)

    if state.budget_preference:
        notes.append(_build_budget_preference_note(state.budget_preference))
    if state.performance_priority:
        notes.append(_build_performance_preference_note(state.performance_priority))
    if state.hybrid_weight_profile:
        notes.append(_build_semantic_query_pattern_preference_note(state.hybrid_weight_profile))
    if state.prefix_wildcard_enabled is not None:
        notes.append(_build_prefix_wildcard_requirement_note(state.prefix_wildcard_enabled))
    if state.model_deployment_preference:
        notes.append(_build_model_deployment_preference_note(state.model_deployment_preference))
    if state.hybrid_weight_profile:
        notes.append(_build_hybrid_weight_profile_note(state.hybrid_weight_profile))

    return "\n".join(notes)


def _build_planning_context(state: SessionState, additional_context: str = "") -> str:
    """Build the full context string for the planning agent."""
    parts: list[str] = []

    if state.sample_doc_json:
        parts.append(f"Sample document: {state.sample_doc_json}")

    if state.source_index_name:
        parts.append(
            f"Execution policy: source is localhost OpenSearch index "
            f"'{state.source_index_name}' (system-enforced, not user-stated); "
            "if target index already exists during setup, "
            "do NOT recreate it (replace_if_exists=false). "
            "Use a different target index name."
        )
        if isinstance(state.source_index_doc_count, int):
            parts.append(
                "Requirements note: exact current document count already measured "
                f"from OpenSearch count API: {state.source_index_doc_count:,}. "
                "Do NOT ask current-count or growth-projection questions. "
                "Assume this is representative sample data and ingestion will continue."
            )

    parts.append(_build_context_notes(state))

    if additional_context:
        parts.append(additional_context)

    return "\n\n".join(parts)


# -------------------------------------------------------------------------
# MCP prompt (for Claude Desktop and generic MCP clients)
# -------------------------------------------------------------------------

@mcp.prompt()
def opensearch_workflow() -> str:
    """OpenSearch Solution Architect workflow guide.

    Select this prompt to learn how to use the opensearch-orchestrator
    tools for designing and deploying an OpenSearch search solution.
    """
    return WORKFLOW_PROMPT


# -------------------------------------------------------------------------
# Low-level domain tools (kept for advanced / direct-access clients)
# -------------------------------------------------------------------------

mcp.tool()(submit_sample_doc)
mcp.tool()(submit_sample_doc_from_local_file)
mcp.tool()(submit_sample_doc_from_url)
mcp.tool()(get_sample_docs_for_verification)
mcp.tool()(read_knowledge_base)
mcp.tool()(read_dense_vector_models)
mcp.tool()(read_sparse_vector_models)
mcp.tool()(search_opensearch_org)

mcp.tool()(create_index)
mcp.tool()(create_and_attach_pipeline)
mcp.tool()(create_bedrock_embedding_model)
mcp.tool()(create_local_pretrained_model)
mcp.tool()(index_doc)
mcp.tool()(index_verification_docs)
mcp.tool()(delete_doc)
mcp.tool()(apply_capability_driven_verification)
mcp.tool()(preview_capability_driven_verification)
mcp.tool()(launch_search_ui)
mcp.tool()(cleanup_ui_server)
mcp.tool()(set_search_ui_suggestions)


def _flatten_exception_leaves(exc: BaseException) -> list[BaseException]:
    if isinstance(exc, BaseExceptionGroup):
        leaves: list[BaseException] = []
        for nested in exc.exceptions:
            leaves.extend(_flatten_exception_leaves(nested))
        return leaves
    return [exc]


def _is_expected_stdio_disconnect(exc: BaseException) -> bool:
    leaves = _flatten_exception_leaves(exc)
    if not leaves:
        return False

    expected_types = (
        anyio.BrokenResourceError,
        anyio.ClosedResourceError,
        BrokenPipeError,
        EOFError,
    )

    for leaf in leaves:
        if isinstance(leaf, expected_types):
            continue
        if isinstance(leaf, OSError) and leaf.errno in {errno.EPIPE, errno.EBADF}:
            continue
        return False
    return True


def main() -> None:
    """Entry point for the MCP server (used by both `uv run` and PyPI console_scripts)."""
    if sys.stdin.isatty():
        print(
            "This MCP server uses JSON-RPC over stdio and must be launched by an MCP client "
            "(Cursor/Claude Desktop/Inspector)."
        )
        print("For an interactive local workflow, run: python opensearch_orchestrator/orchestrator.py")
        raise SystemExit(0)
    # IDE MCP integrations commonly run stdio servers as child processes (for this repo:
    # clients like Cursor launches `uv run opensearch_orchestrator/mcp_server.py` from `.cursor/mcp.json`).
    # Reconnect-like events (window reload/restart, MCP toggle, cancel/disconnect/re-init)
    # close and reopen the stdio pipe. When that pipe closes, this process should exit
    # cleanly; the client starts a new process for the new connection.
    # In practice: reconnect == restart (new PID, fresh in-memory session state).
    try:
        mcp.run(transport="stdio")
    except BaseException as exc:
        if _is_expected_stdio_disconnect(exc):
            raise SystemExit(0)
        raise


if __name__ == "__main__":
    main()
