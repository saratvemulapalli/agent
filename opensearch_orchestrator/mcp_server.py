# /// script
# dependencies = ["anyio", "mcp", "opensearch-py", "pandas>=2.3.3", "pyarrow>=23.0.1", "strands-agents"]
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
import os
import re
import sys

import anyio
from mcp import types as mcp_types
from mcp.server.fastmcp import Context, FastMCP

from opensearch_orchestrator.orchestrator import create_transport_agnostic_engine
from opensearch_orchestrator.planning_session import PlanningSession
from opensearch_orchestrator.solution_planning_assistant import (
    SYSTEM_PROMPT as PLANNER_SYSTEM_PROMPT,
)
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
    preview_cap_driven_verification,
    launch_search_ui,
    cleanup_ui_server,
    set_search_ui_suggestions,
)

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
- Ask one preference question at a time, in this order:
  - **Budget**: flexible or cost-sensitive
  - **Performance priority**: speed-first, balanced, or accuracy-first
  - **Query pattern**: mostly-exact (like "Carmencita 1894"), mostly-semantic
    (like "early silent films about dancers"), or balanced (mix of both)
- Use the client user-input UI for each question (fixed options only, not free-text).
- If query pattern is balanced or mostly-semantic, ask **Deployment preference** as a separate follow-up question:
  opensearch-node, sagemaker-endpoint, or external-embedding-api (also via user-input UI).
- Call `set_preferences(budget, performance, query_pattern, deployment_preference)` with the collected values.

### Phase 3: Plan
- Call `start_planning()` to get an initial architecture proposal from the planner agent.
- If `start_planning()` returns `manual_planning_required=true`, use the client LLM to draft
  planner turns using the returned `manual_planner_system_prompt` and
  `manual_planner_initial_input`, then call `set_plan_from_planning_complete(...)`
  once the user confirms.
- Otherwise, present the proposal to the user verbatim (do not summarize it away).
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
- Prefer planner tools for plan generation.
- If manual planning is required (client sampling unavailable), generate the plan with the
  client LLM using the provided planner prompt/input and persist it with
  `set_plan_from_planning_complete(...)` before execution.
- Show the planner's proposal text to the user verbatim; do not summarize it away.
- For preference questions, ask one question per turn and use user-input UI fixed options, not free-text.
- Do not ask redundant clarification questions for items already inferred from the sample data.
"""

# -------------------------------------------------------------------------
# Shared workflow engine (single session per stdio connection)
# -------------------------------------------------------------------------

_engine = create_transport_agnostic_engine()

# -------------------------------------------------------------------------
# MCP server
# -------------------------------------------------------------------------

mcp = FastMCP("OpenSearch Orchestrator", json_response=True)

# -------------------------------------------------------------------------
# Phase tools
# -------------------------------------------------------------------------

PLANNER_MODE_ENV = "OPENSEARCH_MCP_PLANNER_MODE"
PLANNER_MODE_CLIENT = "client"
PLANNER_MODE_SERVER = "server"
_PLANNING_COMPLETE_PATTERN = re.compile(
    r"<planning_complete>(.*?)</planning_complete>",
    re.DOTALL | re.IGNORECASE,
)
_SOLUTION_PATTERN = re.compile(r"<solution>(.*?)</solution>", re.DOTALL | re.IGNORECASE)
_CAPABILITIES_PATTERN = re.compile(
    r"<search_capabilities>(.*?)</search_capabilities>",
    re.DOTALL | re.IGNORECASE,
)
_KEYNOTE_PATTERN = re.compile(r"<keynote>(.*?)</keynote>", re.DOTALL | re.IGNORECASE)


def _resolve_planner_mode() -> str:
    raw = str(os.getenv(PLANNER_MODE_ENV, PLANNER_MODE_CLIENT)).strip().lower()
    if raw in {PLANNER_MODE_CLIENT, PLANNER_MODE_SERVER}:
        return raw
    return PLANNER_MODE_CLIENT


def _is_method_not_found_error(exc: Exception) -> bool:
    message = str(exc or "").lower()
    return "method not found" in message


def _build_current_planning_context(additional_context: str = "") -> str:
    build_fn = getattr(_engine, "_build_planning_context", None)
    state = getattr(_engine, "state", None)
    if not callable(build_fn) or state is None:
        return str(additional_context or "")
    return str(build_fn(state, additional_context))


def _build_manual_planner_bootstrap(additional_context: str = "") -> dict[str, str]:
    """Build bootstrap prompts for manual client-LLM planning fallback.

    MCP client-mode usage flow:
    1. Set `OPENSEARCH_MCP_PLANNER_MODE=client`.
    2. Call `load_sample(...)`, then `set_preferences(...)`, then `start_planning()`.
    3. If `start_planning()` returns `manual_planning_required=true`, run planner turns
       in the client LLM using:
       - system message: `manual_planner_system_prompt`
       - first user message: `manual_planner_initial_input`
       - follow-up user feedback turns until the user confirms the plan
    4. Commit the confirmed plan via
       `set_plan_from_planning_complete(planner_response)`, where
       `planner_response` includes:
       `<planning_complete><solution>...</solution><search_capabilities>...</search_capabilities><keynote>...</keynote></planning_complete>`
    5. Continue with `execute_plan()` (and `retry_execution()` if needed).
    """
    planning_context = _build_current_planning_context(additional_context)
    parser = PlanningSession(agent=lambda _prompt: "")
    return {
        "manual_planner_system_prompt": PLANNER_SYSTEM_PROMPT,
        "manual_planner_initial_input": parser._build_initial_input(planning_context),
    }


def _parse_planning_complete_response(response_text: str) -> dict[str, str] | dict[str, object]:
    text = str(response_text or "")
    match = _PLANNING_COMPLETE_PATTERN.search(text)
    if match is None:
        return {
            "error": "No <planning_complete> block found.",
            "details": [
                "Provide the planner output containing <planning_complete>...</planning_complete>.",
            ],
        }

    content = match.group(1)
    solution_match = _SOLUTION_PATTERN.search(content)
    capabilities_match = _CAPABILITIES_PATTERN.search(content)
    keynote_match = _KEYNOTE_PATTERN.search(content)
    solution = solution_match.group(1).strip() if solution_match else ""
    search_capabilities = capabilities_match.group(1).strip() if capabilities_match else ""
    keynote = keynote_match.group(1).strip() if keynote_match else ""
    if not solution:
        return {
            "error": "Invalid <planning_complete> block.",
            "details": ["<solution> is required."],
        }
    return {
        "solution": solution,
        "search_capabilities": search_capabilities,
        "keynote": keynote,
    }


def _normalize_manual_plan(
    *,
    solution: str,
    search_capabilities: str,
    keynote: str,
    additional_context: str = "",
) -> dict[str, object]:
    planning_context = _build_current_planning_context(additional_context)
    parser = PlanningSession(agent=lambda _prompt: "")
    parser._initial_context = planning_context
    parser._confirmation_received = True

    wrapped = (
        "<planning_complete>\n"
        "<solution>\n"
        f"{str(solution or '').strip()}\n"
        "</solution>\n"
        "<search_capabilities>\n"
        f"{str(search_capabilities or '').strip()}\n"
        "</search_capabilities>\n"
        "<keynote>\n"
        f"{str(keynote or '').strip()}\n"
        "</keynote>\n"
        "</planning_complete>"
    )
    match = _PLANNING_COMPLETE_PATTERN.search(wrapped)
    if match is None:
        return {
            "error": "Failed to parse manual plan.",
            "details": ["Unable to construct <planning_complete> block for validation."],
        }

    retry_feedback = parser._try_extract_result(match)
    if retry_feedback is not None:
        return {
            "error": "Manual plan failed planner validation.",
            "details": [retry_feedback],
            "hint": (
                "Regenerate the planner output using the same planner prompt/initial input "
                "and submit a corrected <planning_complete> block."
            ),
        }
    result = parser._result
    if not isinstance(result, dict):
        return {
            "error": "Manual plan failed planner validation.",
            "details": ["No normalized planner result was produced."],
        }
    return result


def _sampling_content_to_text(content: object) -> str:
    if isinstance(content, mcp_types.TextContent):
        return str(content.text or "")
    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, mcp_types.TextContent):
                text_parts.append(str(item.text or ""))
        if text_parts:
            return "\n".join(part for part in text_parts if part)
    return str(content or "")


class _ClientSamplingPlannerAgent:
    """Planner callable that delegates generation to MCP client sampling."""

    def __init__(self, ctx: Context) -> None:
        self._session = ctx.session
        self._messages: list[mcp_types.SamplingMessage] = []

    def reset(self) -> None:
        self._messages = []

    async def __call__(self, prompt: str) -> str:
        prompt_text = str(prompt or "").strip()
        if prompt_text:
            self._messages.append(
                mcp_types.SamplingMessage(
                    role="user",
                    content=mcp_types.TextContent(type="text", text=prompt_text),
                )
            )

        result = await self._session.create_message(
            messages=self._messages,
            max_tokens=4000,
            system_prompt=PLANNER_SYSTEM_PROMPT,
        )
        assistant_text = _sampling_content_to_text(result.content)
        self._messages.append(
            mcp_types.SamplingMessage(
                role="assistant",
                content=mcp_types.TextContent(type="text", text=assistant_text),
            )
        )
        return assistant_text


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
    return _engine.load_sample(source_type=source_type, source_value=source_value)


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
            "external-embedding-api". Used when query_pattern is
            "balanced" or "mostly-semantic". Defaults to "opensearch-node".

    Returns:
        dict confirming stored preferences and generated context notes.
    """
    return _engine.set_preferences(
        budget=budget,
        performance=performance,
        query_pattern=query_pattern,
        deployment_preference=deployment_preference,
    )


@mcp.tool()
async def start_planning(additional_context: str = "", ctx: Context | None = None) -> dict:
    """Start the solution planning phase. Returns the planner's initial proposal.
    Call this after set_preferences.

    Args:
        additional_context: Optional extra context to include.

    Returns:
        dict with response text, is_complete flag, and result (if complete).
    """
    planner_mode = _resolve_planner_mode()
    if planner_mode == PLANNER_MODE_CLIENT:
        if ctx is None:
            return {
                "error": "Planning failed in client mode.",
                "details": ["MCP context is unavailable for client sampling."],
                "hint": (
                    "Call start_planning via an MCP client session, "
                    f"or set `{PLANNER_MODE_ENV}={PLANNER_MODE_SERVER}`."
                ),
            }
        try:
            result = await _engine.start_planning(
                additional_context=additional_context,
                planning_agent=_ClientSamplingPlannerAgent(ctx),
            )
            result["planner_backend"] = "client_sampling"
            return result
        except Exception as exc:
            if _is_method_not_found_error(exc):
                bootstrap = _build_manual_planner_bootstrap(additional_context)
                return {
                    "error": "Planning failed in client mode.",
                    "details": [f"client-sampling planner failed: {exc}"],
                    "planner_backend": "client_manual",
                    "manual_planning_required": True,
                    "hint": (
                        "The MCP client does not support `sampling/createMessage`. "
                        "Use the returned manual planner prompt/input to generate planner turns "
                        "with the client LLM, then call `set_plan_from_planning_complete(...)` "
                        "after user confirmation."
                    ),
                    **bootstrap,
                }
            return {
                "error": "Planning failed in client mode.",
                "details": [f"client-sampling planner failed: {exc}"],
                "hint": f"Set `{PLANNER_MODE_ENV}={PLANNER_MODE_SERVER}` to use Bedrock planner.",
            }

    if planner_mode == PLANNER_MODE_SERVER:
        try:
            result = await _engine.start_planning(
                additional_context=additional_context,
            )
            result["planner_backend"] = "server_bedrock"
            return result
        except Exception as exc:
            return {
                "error": "Planning failed in server mode.",
                "details": [f"server planner failed: {exc}"],
            }

    return {
        "error": "Failed to start planning.",
        "details": [f"Unsupported planner mode: {planner_mode!r}"],
        "hint": (
            f"Set `{PLANNER_MODE_ENV}` to '{PLANNER_MODE_CLIENT}' or '{PLANNER_MODE_SERVER}'."
        ),
    }


@mcp.tool()
async def refine_plan(user_feedback: str) -> dict:
    """Send user feedback to the planner and get a refined proposal.
    Call after start_planning. Repeat as needed.

    Args:
        user_feedback: User's feedback, questions, or change requests.

    Returns:
        dict with response text, is_complete flag, and result (if complete).
    """
    return await _engine.refine_plan(user_feedback)


@mcp.tool()
async def finalize_plan() -> dict:
    """Force the planner to finalize and return the structured plan.
    Call when the user confirms they are satisfied with the proposal.

    Returns:
        dict with solution, search_capabilities, and keynote.
    """
    return await _engine.finalize_plan()


@mcp.tool()
def set_plan(solution: str, search_capabilities: str = "", keynote: str = "") -> dict:
    """Store a client-authored finalized plan for execution after planner validation.
    Call this when the MCP client cannot run `start_planning` via client sampling
    and the client LLM authored the proposal directly.

    Args:
        solution: Finalized architecture plan text.
        search_capabilities: Search capability section text.
        keynote: Key assumptions and caveats.

    Returns:
        dict with status and stored normalized plan.
    """
    normalized = _normalize_manual_plan(
        solution=solution,
        search_capabilities=search_capabilities,
        keynote=keynote,
    )
    if "error" in normalized:
        return normalized
    return _engine.set_plan(
        solution=str(normalized.get("solution", "")),
        search_capabilities=str(normalized.get("search_capabilities", "")),
        keynote=str(normalized.get("keynote", "")),
    )


@mcp.tool()
def set_plan_from_planning_complete(planner_response: str, additional_context: str = "") -> dict:
    """Parse and store planner output from a `<planning_complete>` block.
    Preferred manual-mode commit path when `manual_planning_required=true`.

    Args:
        planner_response: Full planner response text containing `<planning_complete>`.
        additional_context: Optional context to include for normalization/validation.

    Returns:
        dict with status and stored normalized plan, or validation feedback.
    """
    parsed = _parse_planning_complete_response(planner_response)
    if "error" in parsed:
        return parsed
    normalized = _normalize_manual_plan(
        solution=str(parsed.get("solution", "")),
        search_capabilities=str(parsed.get("search_capabilities", "")),
        keynote=str(parsed.get("keynote", "")),
        additional_context=additional_context,
    )
    if "error" in normalized:
        return normalized
    return _engine.set_plan(
        solution=str(normalized.get("solution", "")),
        search_capabilities=str(normalized.get("search_capabilities", "")),
        keynote=str(normalized.get("keynote", "")),
    )


@mcp.tool()
async def execute_plan(additional_context: str = "") -> dict:
    """Execute the finalized solution plan (create index, models, pipelines, UI).
    Call after finalize_plan, set_plan, or set_plan_from_planning_complete.

    Args:
        additional_context: Optional extra instructions for the worker.

    Returns:
        dict with worker execution report and UI access URLs.
    """
    result = await _engine.execute_plan(
        additional_context=additional_context,
    )
    if "error" in result:
        return result
    return {
        "execution_report": result["execution_report"],
        "ui_access": _build_ui_access_payload(),
    }


@mcp.tool()
async def retry_execution() -> dict:
    """Retry execution from the last failed step.
    Call after execute_plan fails and the user has fixed the issue.

    Returns:
        dict with worker execution report and UI access URLs.
    """
    result = await _engine.retry_execution()
    if "error" in result:
        return result
    return {
        "execution_report": result["execution_report"],
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
mcp.tool()(preview_cap_driven_verification)
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
