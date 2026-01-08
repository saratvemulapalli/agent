import os
import sys
import asyncio

from strands import Agent, tool
from strands.models import BedrockModel
from scripts.handler import ThinkingCallbackHandler
from semantic_search_expert_assistant import semantic_search_expert_assistant

# -------------------------------------------------------------------------
# Tool Definitions
# -------------------------------------------------------------------------

# semantic_search_expert_assistant is already a tool, so we can use it directly.

# -------------------------------------------------------------------------
# System Prompt
# -------------------------------------------------------------------------

SYSTEM_PROMPT = """
You are an intelligent Orchestrator Agent for an OpenSearch Solution Architect system.

Your primary goal is to **clarify the user's requirements** for their search application and then **delegate the technical selection** to a specialist worker agent.

### Your Responsibilities

1.  **Clarify Requirements**: Engage the user to gather the following critical information if not already provided:
    *   **Data Size**: How many documents? Total size?
    *   **Languages**: What languages are the documents in?
    *   **Budget/Cost**: Is there a strict budget? Infrastructure constraints?
    *   **Latency Requirements**: What is the target P99 latency?
    *   **Accuracy/Relevance**: What type of matching is needed? (Exact, semantic, hybrid?)
    *   **Update Frequency**: How often is data indexed/updated?
    *   **Filters/Security**: Are there complex filters or document-level security?
    *   **Deployment**: AWS Managed OpenSearch, self-hosted, custom model deployment, etc.?

2.  **Call the Expert**: Once you have sufficient information (or if the user insists on proceeding with what they have), call the `semantic_search_expert_assistant` tool with the summarized requirements.

3.  **Present Results**: Present the report returned by the expert to the user.

### Workflow

*   **Step 1**: Check if user input contains enough requirements.
*   **Step 2**: If not enough info, ask clarifying questions (up to 3-5 questions at a time or sequentially, depending on flow).
*   **Step 3**: If enough info, invoke `semantic_search_expert_assistant(context="...")`.
*   **Step 4**: Output the expert's response to the user.

### Important Notes

*   Do NOT attempt to make the technical recommendation yourself. Always use the tool.
*   You are the interface; be helpful and polite.
*   If the user asks non-technical questions (e.g., "What is this system?"), answer them directly.
"""

# -------------------------------------------------------------------------
# Orchestrator Execution
# -------------------------------------------------------------------------

async def main():
    """
    Main loop for the Orchestrator Agent.
    """
    
    model_id = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
    
    print(f"Initializing Orchestrator Agent with model: {model_id}...")
    
    try:
        model = BedrockModel(
            model_id=model_id,
            max_tokens=4000, 
            additional_request_fields={
                "thinking": {
                    "type": "enabled",
                    "budget_tokens": 1024,
                }
            }
        )
        
        agent = Agent(
            model=model, 
            system_prompt=SYSTEM_PROMPT,
            tools=[semantic_search_expert_assistant],
            callback_handler=ThinkingCallbackHandler()
        )
        
    except Exception as e:
        print(f"Failed to initialize orchestrator: {e}")
        return

    print("Orchestrator ready. Type 'exit' or 'quit' to stop.")
    print("-" * 50)

    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ['exit', 'quit']:
                print("Goodbye!")
                break
            
            print("Orchestrator: ", end="", flush=True)
            
            # Stream the response
            stream = agent.stream_async(user_input)
            
            async for event in stream:
                pass
            
            print() 
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(main())
