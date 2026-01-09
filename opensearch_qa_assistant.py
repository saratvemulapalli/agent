import os
import sys
import re

from strands import Agent, tool
from strands.models import BedrockModel
from scripts.handler import ThinkingCallbackHandler
from scripts.tools import read_knowledge_base

# -------------------------------------------------------------------------
# System Prompt
# -------------------------------------------------------------------------

SYSTEM_PROMPT = """
# OpenSearch QA Assistant

You are an OpenSearch QA assistant.
Your goal is to answer questions related to OpenSearch semantic search.

## Responsibilities

1. **Understand the Question**
2. **Consult the Knowledge Base**: Use `read_knowledge_base` to check for specific details.
3. **Provide the Answer**: Offer a brief answer based on the knowledge base.

## Output
Provide a brief response answering the question. And explain the reasoning behind the answer.

## Principles
* Be brief, concise, and clear.
* Use the knowledge base to ground your answers in the specific guide provided.
* Do not mention the knowledge base in your answer.
"""

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
    model_id = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"

    try:
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
            tools=[read_knowledge_base],
            callback_handler=ThinkingCallbackHandler(output_color="\033[96m") # Cyan output for follow-up
        )
        
        user_message = f"User Question: {query}"
        
        response = agent(user_message)
        return str(response)

    except Exception as e:
        return f"Error in opensearch qa assistant: {e}"

if __name__ == "__main__":
    # Test run
    sample_query = "For dense vector search, what are the best models to use?"
    sample_query = "How does doc-only mode work in sparse neural search? Explain the mechanism and why it's faster and cheaper than bi-encoder mode."
    result = opensearch_qa_assistant(sample_query)
    print(result)
