import os
import sys
import re

from strands import Agent, tool
from strands.models import BedrockModel
from scripts.handler import ThinkingCallbackHandler
from scripts.tools import read_knowledge_base, read_dense_vector_models, read_sparse_vector_models, search_opensearch_org
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

## Output Format

### During the Conversation
*   Communicate naturally with the user.
*   Provide analysis, answers, and updated proposals.

### When Plan is Confirmed (Final Step)
When the user confirms the plan, you must output a special block wrapped in `<planning_complete>` tags.
This block acts as the signal to the orchestrator to proceed.

Structure:

<planning_complete>
    <solution>
        - Retrieval Method (e.g., Hybrid with BM25 + kNN)
        - Algorithm/Engine (e.g., HNSW, faiss, lucene)
        - Model Deployment (e.g., SageMaker, Embedding API)
        - Specific Model IDs (e.g., "amazon.titan-embed-text-v2")
    </solution>
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

# Initialize the internal agent for the planning loop
agent = Agent(
    model=model, 
    system_prompt=SYSTEM_PROMPT,
    tools=[read_knowledge_base, read_dense_vector_models, read_sparse_vector_models, search_opensearch_org],
    callback_handler=ThinkingCallbackHandler(output_color="\033[94m") # Blue output
)

# -------------------------------------------------------------------------
# Worker Execution
# -------------------------------------------------------------------------

def _read_multiline_input() -> str:
    """Read user input until an empty line is entered."""
    print("\nYou:")
    lines = []
    while True:
        line = input()
        if line == "":
            break
        lines.append(line)
    return "\n".join(lines).strip()


@tool
def solution_planning_assistant(context: str) -> dict:
    """Act as a semantic search expert assistant to provide technical recommendations based on user context.
    This tool initiates an interactive session with the user to refine the plan.

    Args:
        context: A detailed string containing user requirements (data size, latency, budget, etc.), preferences, and any other relevant context for decision making.

    Returns:
        dict: A comprehensive technical recommendation report and conversation summary (Solution + Keynote).
    """
    print(f"\033[91m[solution_planning_assistant] Input context: {context}\033[0m")

    try:
        # Initial prompt to the internal agent
        current_input = f"Here is the user context: {context}. Please provide a technical recommendation."
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
                keynote_match = re.search(r'<keynote>(.*?)</keynote>', content, re.DOTALL)
                
                solution = solution_match.group(1).strip() if solution_match else "No solution parsed."
                keynote = keynote_match.group(1).strip() if keynote_match else "No keynote provided."
                
                # Return a structured string for the Orchestrator to parse
                return {
                    "solution": solution,
                    "keynote": keynote
                }

            # Get explicit user feedback; never fabricate a user confirmation.
            while True:
                try:
                    user_input = _read_multiline_input()
                except KeyboardInterrupt:
                    return {
                        "solution": "CANCELLED",
                        "keynote": "User cancelled planning."
                    }

                if user_input:
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
