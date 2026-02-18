import asyncio

from strands import Agent
from strands.models import BedrockModel
from scripts.handler import ThinkingCallbackHandler
from scripts.tools import (
    submit_sample_doc,
    get_sample_doc,
    submit_sample_doc_from_local_file,
    submit_sample_doc_from_url,
    clear_sample_doc,
)
from scripts.shared import (
    Phase,
    read_multiline_input,
    check_and_clear_execution_flag,
    looks_like_new_request,
    looks_like_cancel,
    looks_like_cleanup_request,
    looks_like_worker_retry,
    looks_like_url_message,
    looks_like_local_path_message,
    clear_last_worker_context,
    clear_last_worker_run_state,
    get_last_worker_run_state,
)
from scripts.opensearch_ops_tools import cleanup_verification_docs
from solution_planning_assistant import solution_planning_assistant, reset_planner_agent
from worker import worker_agent


# -------------------------------------------------------------------------
# System Prompt
# -------------------------------------------------------------------------

SYSTEM_PROMPT = """
You are an intelligent Orchestrator Agent for an OpenSearch Solution Architect system.

Your goal is to guide the user from initial requirements to a finalized, executed solution.

### Workflow Phases

1.  **Collect Sample Document (Mandatory First Step)**:
    *   A sample document is required before any planning or execution.
    *   **Pre-loaded samples (highest priority)**: If the user message contains a
        "System note" stating a sample document has already been loaded, trust it.
        Call `get_sample_doc` immediately and proceed to Phase 2. Do NOT ask the
        user to paste content or re-upload — the sample is already stored.
    *   If the user pastes sample content, call `submit_sample_doc`.
    *   If no sample is available, ask the user to paste one sample document.
    *   Do not skip this step.

2.  **Clarify Requirements**:
    *   Based on your analysis of the sample doc, engage the user only **once** to gather REMAINING critical information.
    *   **Infer First**: If sample was loaded from URL/local source, use inferred metadata from tool output before asking questions.
    *   **Document Size**: Infer approximate size from source/profile when possible. Ask follow-up to confirm whether provided file/folder is the full corpus or only a sample, and expected future growth.
    *   **Languages**: Infer likely language/script from sample data first. Ask whether future/new datasets will include additional languages and whether cross-lingual search is required.
    *   **Budget/Cost**: Is there a strict budget? Is cost-effective search required?
    *   **Latency Requirements**: What is the target P99 latency?
    *   **Latency-Accuracy Trade-off**: What is the desired trade-off between latency and accuracy?
    *   **Hybrid Weight Preference (only if hybrid is being considered)**: Ask one explicit choice for lexical-vs-semantic emphasis: `semantic-heavy`, `balanced`, or `lexical-heavy`.
    *   **Model Deployment**: SageMaker GPU endpoint, embedding API service, OpenSearch Node deployment, custom model deployment, etc.?
    *   **Special Requirements**: Any special requirements? (e.g., prefix queries, wildcard support, etc.)
    
    Only prompt the user **once** for these details. 
    Even there are missing information, do not repeatedly request in separate turns. Do proper assumptions from sample data and provided information.

3.  **Proposal & Refinement (Handover)**:
    *   Once the required information is gathered, call `solution_planning_assistant` with the collected context.
    *   **IMPORTANT**: The `solution_planning_assistant` will take over the conversation. It will handle the proposal presentation, answering user questions, and refining the plan until the user is satisfied.
    *   You (Orchestrator) will *wait* for this tool to complete.
    *   When hybrid retrieval is in scope, include a normalized context line passed to planner:
        *   `Hybrid Weight Profile: semantic-heavy|balanced|lexical-heavy`
        *   Use `balanced` if user did not provide a clear preference.
    *   The tool will return a structured result containing:
        *   `SOLUTION`: The final technical plan.
        *   `SEARCH_CAPABILITIES`: Canonical capability bullets for downstream verification/suggestion flow.
        *   `KEYNOTE`: A summary of the refinement conversation.

4.  **Execution (Final Step)**:
    *   When `solution_planning_assistant` returns `SOLUTION`, `SEARCH_CAPABILITIES`, and `KEYNOTE`:
    *   Call `worker_agent` immediately with a combined context that includes at least `SOLUTION` and `SEARCH_CAPABILITIES` (you may include `KEYNOTE` as well).
    *   You may mention `KEYNOTE` points to the user as confirmation their preferences were heard, but the primary action is to trigger the worker with planner-authored capabilities preserved.
    *   Confirm completion to the user.

### Important Rules

*   **Delegation**: Do NOT generate the plan yourself. Do NOT answer technical questions yourself. Always delegate to `solution_planning_assistant` for the planning and Q&A phase.
*   **State Awareness**: The `solution_planning_assistant` is interactive. Once you call it, trust it to handle the refinement loop.
*   **Worker Call**: You MUST call `worker_agent` immediately after the planning phase completes.
*   **Sample Doc Gate**: A sample document must exist before clarification/planning. When the message says a sample is pre-loaded, proceed directly — do not re-collect.
*   **No Redundant Questions**: Do not ask users to restate values already inferred from source profile/sample data. Only ask confirmation or forward-looking deltas.
*   **Producer-Driven Typing**: Reinforce strict schema typing in planning/execution context: map `boolean` only when producer sends native booleans; if producer sends string flags like `0`/`1`, map as `keyword`.
*   **Persona**: You are the interface; be helpful, polite, and professional.
"""


# -------------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------------

MODEL_ID = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"


# -------------------------------------------------------------------------
# Agent Factory
# -------------------------------------------------------------------------

def _create_orchestrator_agent() -> Agent:
    """Build a fresh orchestrator agent (no conversation history)."""
    model = BedrockModel(
        model_id=MODEL_ID,
        max_tokens=4000,
        additional_request_fields={
            "thinking": {
                "type": "enabled",
                "budget_tokens": 1024,
            }
        },
    )
    return Agent(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        tools=[
            submit_sample_doc,
            get_sample_doc,
            solution_planning_assistant,
            worker_agent,
        ],
        callback_handler=ThinkingCallbackHandler(),
    )


# -------------------------------------------------------------------------
# State Management
# -------------------------------------------------------------------------

def _reset_all_state() -> tuple[Phase, Agent]:
    """Full reset: sample doc, planner session, and orchestrator agent."""
    clear_sample_doc()
    clear_last_worker_context()
    clear_last_worker_run_state()
    reset_planner_agent()
    agent = _create_orchestrator_agent()
    return Phase.COLLECT_SAMPLE, agent


# -------------------------------------------------------------------------
# Main Loop
# -------------------------------------------------------------------------

async def main():
    """Orchestrator main loop with explicit phase tracking and intent routing."""

    print(f"Initializing Orchestrator Agent with model: {MODEL_ID}...")

    try:
        agent = _create_orchestrator_agent()
    except Exception as e:
        print(f"Failed to initialize orchestrator: {e}")
        return

    phase = Phase.COLLECT_SAMPLE

    print("Orchestrator ready. Type 'exit' or 'quit' to stop.")
    print("-" * 50)

    while True:
        try:
            user_input = read_multiline_input()

            if not user_input:
                continue

            if user_input.lower() in ("exit", "quit"):
                print("Goodbye!")
                break

            # ── Intent routing (hard-coded, before LLM) ────────────

            if phase in (Phase.DONE, Phase.EXEC_FAILED) and looks_like_cleanup_request(user_input):
                cleanup_result = cleanup_verification_docs()
                print(f"Orchestrator: {cleanup_result}\n")
                continue

            # In failed-execution phase, default to resume-from-checkpoint for any
            # non-new-request/non-cancel input (e.g., "Docker is ready").
            if (
                phase == Phase.EXEC_FAILED
                and not looks_like_new_request(user_input)
                and not looks_like_cancel(user_input)
            ):
                worker_state = get_last_worker_run_state()
                recovery_context = (
                    str(worker_state.get("context", "")).strip()
                    if isinstance(worker_state, dict)
                    else ""
                )
                if not recovery_context:
                    print(
                        "Orchestrator: Cannot resume worker execution because no checkpoint context is available. "
                        "Run a full execution first.\n"
                    )
                    continue

                if looks_like_worker_retry(user_input):
                    retry_reason = "explicit retry request"
                else:
                    retry_reason = "auto-resume from failed phase"

                retry_result = worker_agent(
                    f"[RESUME_WORKER_FROM_FAILED_STEP]\n{recovery_context}"
                )
                print(f"Orchestrator: ({retry_reason}) {retry_result}\n")
                check_and_clear_execution_flag()
                latest_state = get_last_worker_run_state()
                latest_status = str(latest_state.get("status", "")).lower()
                phase = Phase.DONE if latest_status == "success" else Phase.EXEC_FAILED
                continue

            if looks_like_new_request(user_input):
                phase, agent = _reset_all_state()
                user_input = (
                    "The user started a new request. Ignore all previous context.\n"
                    f"New request: {user_input}\n"
                    "Please start by collecting a sample document."
                )

            elif looks_like_cancel(user_input) and phase != Phase.COLLECT_SAMPLE:
                phase, agent = _reset_all_state()
                print("Orchestrator: Request cancelled. Feel free to start a new request.\n")
                continue

            elif phase == Phase.DONE:
                # Any input after a completed flow starts a fresh cycle.
                phase, agent = _reset_all_state()
                user_input = (
                    "The previous workflow is not active. The user has a new request.\n"
                    f"New request: {user_input}\n"
                    "Please start by collecting a sample document."
                )

            # ── Phase-specific pre-processing ───────────────────────

            if phase in (Phase.COLLECT_SAMPLE, Phase.GATHER_INFO):
                sample_state = get_sample_doc()
                if sample_state == "MISSING_SAMPLE_DOC":
                    load_result = None
                    source_label = None

                    if looks_like_url_message(user_input):
                        load_result = submit_sample_doc_from_url(user_input)
                        source_label = "URL"
                    elif looks_like_local_path_message(user_input):
                        load_result = submit_sample_doc_from_local_file(user_input)
                        source_label = "local file/folder path"

                    if load_result and load_result.startswith("Sample document loaded from"):
                        phase = Phase.GATHER_INFO
                        if source_label == "local file/folder path":
                            user_input = (
                                "[SYSTEM PRE-PROCESSING RESULT]\n"
                                "A sample document has already been loaded from the user's local source.\n"
                                f"Load result: {load_result}\n"
                                "ACTION REQUIRED: Call `get_sample_doc` immediately, then proceed to "
                                "Phase 2 (requirements gathering). DO NOT ask the user to paste sample content.\n\n"
                                "[USER MESSAGE]\n"
                                f"{user_input}"
                            )
                        else:
                            user_input = (
                                "[SYSTEM PRE-PROCESSING RESULT]\n"
                                f"A sample document has already been loaded from the user's {source_label}.\n"
                                f"Load result: {load_result}\n"
                                "ACTION REQUIRED: Call `get_sample_doc` immediately, then proceed to "
                                "Phase 2 (requirements gathering). DO NOT ask the user to paste sample content.\n\n"
                                "[USER MESSAGE]\n"
                                f"{user_input}"
                            )
                    elif load_result and load_result.startswith("Error:"):
                        source = source_label or "source"
                        user_input = (
                            f"{user_input}\n\n"
                            f"System note: Automatic sample loading from user's {source} failed.\n"
                            f"Failure reason: {load_result}\n"
                            "System instruction: Briefly tell the user this exact failure reason, "
                            "then ask them to provide a corrected accessible path/URL or paste one sample document."
                        )
                elif phase == Phase.COLLECT_SAMPLE:
                    # Sample doc already present (e.g. agent loaded it in an earlier turn).
                    phase = Phase.GATHER_INFO

            # ── Agent turn ──────────────────────────────────────────

            print("Orchestrator: ", end="", flush=True)

            stream = agent.stream_async(user_input)
            async for event in stream:
                pass

            print()

            # ── Post-turn phase transitions ─────────────────────────

            if phase == Phase.COLLECT_SAMPLE and get_sample_doc() != "MISSING_SAMPLE_DOC":
                phase = Phase.GATHER_INFO

            if check_and_clear_execution_flag():
                worker_state = get_last_worker_run_state()
                worker_status = str(worker_state.get("status", "")).lower()
                phase = Phase.DONE if worker_status == "success" else Phase.EXEC_FAILED

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")


if __name__ == "__main__":
    asyncio.run(main())
