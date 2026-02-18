import re

from strands import Agent, tool
from strands.models import BedrockModel
from scripts.handler import ThinkingCallbackHandler
from scripts.tools import read_knowledge_base, read_dense_vector_models, read_sparse_vector_models, search_opensearch_org
from scripts.shared import (
    read_multiline_input,
    looks_like_new_request,
    looks_like_execution_intent,
)
from worker import SAMPLE_CONTEXT

# -------------------------------------------------------------------------
# System Prompt
# -------------------------------------------------------------------------

SYSTEM_PROMPT = """
# OpenSearch Semantic Search Expert Assistant

You are an expert OpenSearch search architect and solution consultant.
Your goal is to collaborate with the user to design the best OpenSearch retrieval strategy (BM25 / dense / sparse / hybrid).

## Your Responsibilities

1.  **Analyze & Propose**: Based on the initial context, analyze the requirements and propose a technical solution using `read_knowledge_base` and other tools.
2.  **Consult & Refine**: 
    *   Present your proposal to the user. Ask if the solution is acceptable.
    *   **Interact with the user**: Answer their questions, explain technical details/trade-offs, and adjust the plan based on their feedback.
    *   You act as the expert consultant. If the user asks specific questions (e.g., "Why not IVF?", "What is HNSW?"), use your knowledge base to answer them accurately.
3.  **Finalize**:
    *   Once the user is satisfied and explicitly confirms the plan (e.g., "Yes, let's go", "Looks good"), you MUST output the final result in the specified XML format.

## Tools & Knowledge
*   Use `read_knowledge_base`, `read_dense_vector_models`, `read_sparse_vector_models` as your primary source of truth.
*   Use `search_opensearch_org` when you need latest public OpenSearch documentation updates from opensearch.org.
*   Do not fabricate benchmarks or capabilities not present in the tools.

## Constraints
1.  **NO Cost Estimation**: Do not provide any cost estimates or pricing details.
2.  **NO Implementation Details**: Do not provide specific index settings, mappings, or query DSL (JSON bodies). Focus on architectural decisions and model selection.
3.  **Attribution Accuracy**:
    *   Distinguish explicitly between requirements that are user-stated and requirements that are inferred by the assistant.
    *   If a requirement is inferred, label it as inferred (for example: "Inferred from your 'all search types' response: prefix matching").
    *   Do not claim inferred items as explicit user statements.
    *   Do not use phrasing like "you mentioned X" unless the user explicitly said X.

## Output Format

### During the Conversation
*   Communicate naturally with the user.
*   Provide analysis, answers, and updated proposals.
*   When asking follow-up questions about inferred requirements, say they are inferred and ask for confirmation.

### When Plan is Confirmed (Final Step)
When the user confirms the plan, you must output a special block wrapped in `<planning_complete>` tags.
This block acts as the signal to the orchestrator to proceed.

Structure:

<planning_complete>
    <solution>
        - Retrieval Method (e.g., Hybrid with BM25 + kNN)
        - Hybrid Weight Profile: semantic-heavy|balanced|lexical-heavy (required when retrieval is hybrid lexical+semantic)
        - Algorithm/Engine (e.g., HNSW, faiss, lucene)
        - Model Deployment (e.g., SageMaker, Embedding API)
        - Specific Model IDs (e.g., "amazon.titan-embed-text-v2")
    </solution>
    <search_capabilities>
        - Exact: ...
        - Semantic: ...
        - Structured: ...
        - Combined: ...
        - Autocomplete: ...
        - Fuzzy: ...
    </search_capabilities>
    <keynote>
        (A brief summary of the conversation for the orchestrator)
        - What were the user's main concerns?
        - Any specific preferences revealed during refinement (e.g., "User prioritized low latency over cost")?
        - Key decisions made.
        - BE BRIEF AND CONCISE.
    </keynote>
</planning_complete>

## Important Rules
*   **Do not** output `<planning_complete>` until the user has explicitly confirmed the plan.
*   If the user has questions, answer them first.
*   Only use the `<planning_complete>` tag at the very end.
*   If the user clearly starts a new request/topic, stop refining the old plan and treat the latest request as a new conversation context.
*   If the user explicitly asks to proceed to setup/implementation, treat that as confirmation and finalize with `<planning_complete>`.
*   In `<search_capabilities>`, include only applicable capabilities.
*   Every capability bullet in `<search_capabilities>` MUST start with one canonical prefix exactly: `Exact:`, `Semantic:`, `Structured:`, `Combined:`, `Autocomplete:`, or `Fuzzy:`.
*   Do not use non-canonical prefixes in `<search_capabilities>`.
*   If the selected retrieval method is hybrid lexical+semantic (for example BM25+dense), include a plain-text line in `<solution>` exactly in this form:
    *   `Hybrid Weight Profile: semantic-heavy|balanced|lexical-heavy`
*   For non-lexical hybrid combinations (for example dense+sparse), do not invent a lexical-vs-semantic profile unless user explicitly asked for one.
"""

model_id = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"

model = BedrockModel(
    model_id=model_id,
    max_tokens=16000,
    additional_request_fields={
        "thinking": {
            "type": "enabled",
            "budget_tokens": 4000,
        }
    }
)

_CANONICAL_CAPABILITY_PREFIX = re.compile(
    r"^[-*]\s*(Exact|Semantic|Structured|Combined|Autocomplete|Fuzzy)\s*:",
    re.IGNORECASE,
)


def _extract_canonical_capability_ids(capability_block: str) -> list[str]:
    """Extract canonical capability IDs from <search_capabilities> block."""
    ids: list[str] = []
    seen: set[str] = set()
    for raw_line in capability_block.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = _CANONICAL_CAPABILITY_PREFIX.match(line)
        if not match:
            # Ignore non-bullet text in section, but reject non-canonical bullet lines.
            if line.startswith("-") or line.startswith("*"):
                return []
            continue
        capability_id = match.group(1).lower()
        if capability_id in seen:
            continue
        seen.add(capability_id)
        ids.append(capability_id)
    return ids


def _create_planner_agent() -> Agent:
    return Agent(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        tools=[read_knowledge_base, read_dense_vector_models, read_sparse_vector_models, search_opensearch_org],
        callback_handler=ThinkingCallbackHandler(output_color="\033[94m"),  # Blue output
    )


def reset_planner_agent() -> None:
    """Reset the planner agent (public — called by orchestrator on new_request)."""
    global agent
    agent = _create_planner_agent()


# Initialize the internal agent for planning loop
agent = _create_planner_agent()

# -------------------------------------------------------------------------
# Worker Execution
# -------------------------------------------------------------------------

@tool
def solution_planning_assistant(context: str) -> dict:
    """Act as a semantic search expert assistant to provide technical recommendations based on user context.
    This tool initiates an interactive session with the user to refine the plan.

    Args:
        context: A detailed string containing user requirements (data size, latency, budget, etc.), preferences, and any other relevant context for decision making.

    Returns:
        dict: A comprehensive technical recommendation report and conversation summary
        (Solution + Search Capabilities + Keynote).
    """
    print(f"\033[91m[solution_planning_assistant] Input context: {context}\033[0m")

    try:
        # Initial prompt to the internal agent
        current_input = f"Here is the user context: {context}. Please provide a technical recommendation."
        finalization_retry_count = 0
        finalization_retry_limit = 2
        # Interaction Loop
        while True:
            # Get response from the planner agent
            response = agent(current_input)
            response_text = str(response)

            # Check for completion tags
            match = re.search(r'<planning_complete>(.*?)</planning_complete>', response_text, re.DOTALL)
            if match:
                content = match.group(1)
                
                # Extract solution and keynote
                solution_match = re.search(r'<solution>(.*?)</solution>', content, re.DOTALL)
                capabilities_match = re.search(r'<search_capabilities>(.*?)</search_capabilities>', content, re.DOTALL)
                keynote_match = re.search(r'<keynote>(.*?)</keynote>', content, re.DOTALL)
                
                solution = solution_match.group(1).strip() if solution_match else "No solution parsed."
                search_capabilities = capabilities_match.group(1).strip() if capabilities_match else ""
                keynote = keynote_match.group(1).strip() if keynote_match else "No keynote provided."
                capability_ids = _extract_canonical_capability_ids(search_capabilities)

                if search_capabilities and capability_ids:
                    # Return a structured string for the Orchestrator to parse
                    return {
                        "solution": solution,
                        "search_capabilities": search_capabilities,
                        "keynote": keynote
                    }

                if finalization_retry_count < finalization_retry_limit:
                    finalization_retry_count += 1
                    current_input = (
                        "Your previous <planning_complete> output is invalid because <search_capabilities> is missing or not canonical.\n"
                        "Regenerate the full <planning_complete> block now.\n"
                        "Requirements:\n"
                        "- Include <solution>, <search_capabilities>, and <keynote>.\n"
                        "- If retrieval method is hybrid lexical+semantic, include in <solution> exactly one line in this format: "
                        "'Hybrid Weight Profile: semantic-heavy|balanced|lexical-heavy'.\n"
                        "- In <search_capabilities>, include only applicable capability bullets.\n"
                        "- Every capability bullet MUST start with one canonical prefix: "
                        "Exact:, Semantic:, Structured:, Combined:, Autocomplete:, or Fuzzy:.\n"
                        "- Do not use any other capability bullet prefixes.\n"
                    )
                    continue
                
                return {
                    "solution": "INVALID_PLANNING_COMPLETE",
                    "search_capabilities": "",
                    "keynote": (
                        "Planner failed to produce canonical <search_capabilities> after retries. "
                        "Expected canonical prefixes: Exact/Semantic/Structured/Combined/Autocomplete/Fuzzy."
                    ),
                }

            # Get explicit user feedback; never fabricate a user confirmation.
            while True:
                try:
                    user_input = read_multiline_input()
                except KeyboardInterrupt:
                    return {
                        "solution": "CANCELLED",
                        "search_capabilities": "",
                        "keynote": "User cancelled planning."
                    }

                if user_input:
                    if looks_like_new_request(user_input):
                        reset_planner_agent()
                        current_input = (
                            "The user started a new request. Ignore previous planning context and treat this as a new conversation.\n"
                            f"New request: {user_input}\n"
                            "Please provide a technical recommendation."
                        )
                        break

                    if looks_like_execution_intent(user_input):
                        current_input = (
                            "The user asked to proceed with setup/implementation. Treat this as confirmation.\n"
                            f"User message: {user_input}\n"
                            "If the plan is acceptable, finalize now using <planning_complete>."
                        )
                        break

                    current_input = user_input
                    break

                print("Please enter feedback or an explicit confirmation to continue.")

    except Exception as e:
        raise e


SAMPLE_CONTEXT = """
User Requirements:
- Document count: 10 million documents
- Language: English (monolingual)
- Content type: Text documents (sample: "The quick brown fox jumps over the lazy dog. This is a sample document for testing search capabilities.")
- Budget: No budget limitation
- Priority: BEST SEARCH RELEVANCE (accuracy is the top priority)
- Latency requirements: Not specified (reasonable latency acceptable given focus on accuracy)
- Model deployment: Not specified (can use any deployment method)
- Special requirements: None specified

Key Focus: Maximum search accuracy and relevance for English text semantic search at 10M document scale.
"""

if __name__ == "__main__":
    # Test run
    sample_context = "I have 10 million documents, mostly English. Low latency is critical (<50ms). Budget is flexible. Preference for managed services."
    result = solution_planning_assistant(SAMPLE_CONTEXT)
    print(result)
