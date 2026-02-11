import os
import sys
import re

from strands import Agent, tool
from strands.models import BedrockModel
from scripts.handler import ThinkingCallbackHandler
from scripts.tools import read_knowledge_base, read_dense_vector_models, read_sparse_vector_models, search_opensearch_org

# -------------------------------------------------------------------------
# System Prompt
# -------------------------------------------------------------------------

SYSTEM_PROMPT = """
# OpenSearch QA Assistant

You are an OpenSearch QA assistant.
Your goal is to answer questions related to OpenSearch semantic search.

## Responsibilities

1. **Understand the Question**
2. **Consult the Knowledge Base**: Use `read_knowledge_base`, `read_dense_vector_models`, or `read_sparse_vector_models` to check for specific details based on the topic.
3. **Use Live Docs Search When Needed**: Use `search_opensearch_org` for the latest public docs on opensearch.org.
4. **Provide the Answer**: Offer a brief answer based on available tool evidence.

## Output
Provide a brief response answering the question. And explain the reasoning behind the answer.

## Principles
* Be brief, concise, and clear.
* Use the knowledge base to ground your answers in the specific guide provided.
* Do not mention the knowledge base in your answer.
"""

model_id = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
model = BedrockModel(
    model_id=model_id,
    max_tokens=8000,
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
    tools=[read_knowledge_base, read_dense_vector_models, read_sparse_vector_models, search_opensearch_org],
    callback_handler=ThinkingCallbackHandler(output_color="\033[96m") # Cyan output for follow-up
)

# -------------------------------------------------------------------------
# Tool Execution
# -------------------------------------------------------------------------

@tool
def opensearch_qa_assistant(query: str) -> str:
    """Execute the OpenSearch QA assistant to answer questions related to OpenSearch semantic search based on the knowledge base.

    Args:
        query: The question to answer.

    Returns:
        str: The answer to the question.
    """
    print(f"\033[91m[opensearch_qa_assistant] Input query: {query}\033[0m")
        
    response = agent(query)
    return str(response)

if __name__ == "__main__":
    # Test run
    sample_query = "For dense vector search, what are the best models to use?"
    result = opensearch_qa_assistant(sample_query)
    sample_query = "How does doc-only mode work in sparse neural search? Explain the mechanism and why it's faster and cheaper than bi-encoder mode."
    result = opensearch_qa_assistant(sample_query)
    print(result)
