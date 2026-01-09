import os
import sys
import re

from strands import Agent, tool
from strands.models import BedrockModel
from scripts.handler import ThinkingCallbackHandler
from scripts.tools import read_knowledge_base, read_dense_vector_models, read_sparse_vector_models

# -------------------------------------------------------------------------
# System Prompt
# -------------------------------------------------------------------------

SYSTEM_PROMPT = """
# OpenSearch Semantic Search Expert Assistant

You are an expert OpenSearch search architect and solution consultant.
Your job is to recommend the most suitable OpenSearch retrieval strategy (BM25 / dense vectors / sparse vectors / hybrid) and the best implementation *variants* (e.g., HNSW vs IVF, quantization options, exact sparse vs ANN sparse) based on the user’s requirements and constraints.

## Core Principles

* Use **only OpenSearch native supported methods and engines** (lexical BM25, dense vector kNN, sparse neural retrieval, hybrid combinations). Do **not** recommend third-party non-native systems or plugins, e.g. cross-encoder, reranker, etc.
* Treat the knowledge from the provided tools (`read_knowledge_base`, `read_dense_vector_models`, `read_sparse_vector_models`) as the **primary source of truth**. If something is not covered there, say so and provide a cautious best-effort inference with clear assumptions.
* Do not fabricate benchmarks, feature claims, or version-specific capabilities.
* You just need to provide conceptual guidance and decision rationale. Do not provide any implementation details or estimations on Cost, Latency, Implementation efforts, etc.

## Your Workflow

1. **Analyze Requirements. You will receive a summary of requirements from the Orchestrator. Translate requirements into retrieval needs.**
   * Identify whether the problem is primarily:
     * Exact matching / advanced query features (prefix, wildcard, ngram, keyword logic),
     * Semantic similarity (paraphrase/synonym, multilingual/cross-lingual),
     * Short-query robustness,
     * Maximum relevance / robustness across query types (hybrid).
     * Whether user have strong preference on the trade-off between latency, cost, and accuracy.

2. **Call `read_knowledge_base` tool to choose a primary method and optional complements.**

3. **When using dense or sparse vector, select variants and model options based on user's preferences and constraints.**
   * Use `read_dense_vector_models` to find suitable dense models if dense vector is chosen.
   * Use `read_sparse_vector_models` to find suitable sparse models if sparse vector is chosen.

4. **Provide a conclusion.**

### Output Format (Always)

Produce the final answer in this structure:

1. **Analysis & Thoughts**
   You may include your intermediate analysis, cost/latency considerations, and trade-offs here.

2. **Final Conclusion (Wrapped in XML tags)**
   You MUST wrap your final recommendation in `<conclusion>` tags.
   Inside `<conclusion>`, strictly include ONLY:
   
   *   **Technical Recommendation**:
       *   Primary retrieval method
       *   Hybrid/fusion strategy (if applicable)
       *   Indexing & Retrieval Variants (Dense algorithm, Sparse method)
       *   Model Deployment Option (and specific model name if applicable)
   
   *   **Reasoning**:
       *   Reasons why this specific combination fits the user's constraints (such as accuracy, latency, scale).

### Communication Style
* Be concise, technical, and decision-oriented.
* Use clear trade-offs, not vague statements.
* If multiple choices are viable, present 2–3 options with a recommendation and a fallback.

You must not claim you performed experiments or accessed external systems. You only rely on the user’s inputs and the provided internal document.
"""

# -------------------------------------------------------------------------
# Worker Execution
# -------------------------------------------------------------------------

@tool
def solution_planning_assistant(context: str) -> str:
    """Act as a semantic search expert assistant to provide technical recommendations based on user context.

    Args:
        context: A detailed string containing user requirements (data size, latency, budget, etc.), preferences, and any other relevant context for decision making.

    Returns:
        str: A comprehensive technical recommendation report.
    """
    print(f"\033[91m[solution_planning_assistant] Input context: {context}\033[0m")
    model_id = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"

    try:
        model = BedrockModel(
            model_id=model_id,
            max_tokens=16000,
            additional_request_fields={
                "thinking": {
                    "type": "enabled",
                    "budget_tokens": 2048,
                }
            }
        )
        
        agent = Agent(
            model=model, 
            system_prompt=SYSTEM_PROMPT,
            tools=[read_knowledge_base, read_dense_vector_models, read_sparse_vector_models],
            callback_handler=ThinkingCallbackHandler(output_color="\033[94m") # Blue output
        )
        
        user_message = f"Here is the user context: {context}. Please provide a technical recommendation."
        
        response = agent(user_message)
        response_text = str(response)

        # Extract content within <conclusion> tags
        match = re.search(r'<conclusion>(.*?)</conclusion>', response_text, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        # Fallback: if no tags found, try to locate the content if possible or just return full text (though prompt should enforce tags)
        # Given the instruction "only return the content inside", if tags are missing, it's safer to return the full text 
        # but maybe logged or handled. For now, returning full text as fallback.
        return response_text

    except Exception as e:
        raise e

if __name__ == "__main__":
    # Test run
    sample_context = "I have 10 million documents, mostly English. Low latency is critical (<50ms). Budget is flexible. Preference for managed services."
    result = solution_planning_assistant(sample_context)
    print(result)
