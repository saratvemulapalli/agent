import os
import sys
import asyncio

from strands import Agent, tool
from strands.models import BedrockModel
from scripts.handler import ThinkingCallbackHandler
from scripts.tools import get_sample_doc
from solution_planning_assistant import solution_planning_assistant
# from opensearch_qa_assistant import opensearch_qa_assistant # No longer used in main flow
from worker import worker_agent


# -------------------------------------------------------------------------
# System Prompt
# -------------------------------------------------------------------------

SYSTEM_PROMPT = """
You are an intelligent Orchestrator Agent for an OpenSearch Solution Architect system.

Your goal is to guide the user from initial requirements to a finalized, executed solution.

### Workflow Phases

1.  **Data Analysis (First Step)**:
    *   call `get_sample_doc` to retrieve a sample document from the user.
    *   Analyze the returned sample to understand the data structure, potential language, and content type.

2.  **Clarify Requirements**:
    *   Based on your analysis of the sample doc, engage the user only **once** to gather REMAINING critical information.
    *   **Document Size**: How many documents?
    *   **Languages**: What languages are the corpus? Mono-lingual or multi-lingual? Is cross-lingual search required?
    *   **Budget/Cost**: Is there a strict budget? Is cost-effective search required?
    *   **Latency Requirements**: What is the target P99 latency?
    *   **Latency-Accuracy Trade-off**: What is the desired trade-off between latency and accuracy?
    *   **Model Deployment**: SageMaker GPU endpoint, embedding API service, OpenSearch Node deployment, custom model deployment, etc.?
    *   **Special Requirements**: Any special requirements? (e.g., prefix queries, wildcard support, etc.)
    
    Only prompt the user **once** for these details. 
    Even there are missing information, do not repeatedly request in separate turns. Do proper assumptions from sample data and provided information.

3.  **Proposal & Refinement (Handover)**:
    *   Once the required information is gathered, call `solution_planning_assistant` with the collected context.
    *   **IMPORTANT**: The `solution_planning_assistant` will take over the conversation. It will handle the proposal presentation, answering user questions, and refining the plan until the user is satisfied.
    *   You (Orchestrator) will *wait* for this tool to complete.
    *   The tool will return a structured result containing:
        *   `SOLUTION`: The final technical plan.
        *   `KEYNOTE`: A summary of the refinement conversation.

4.  **Execution (Final Step)**:
    *   When `solution_planning_assistant` returns the `SOLUTION` and `KEYNOTE`:
    *   Call `worker_agent` immediately with the `SOLUTION`.
    *   You may mention the `KEYNOTE` points to the user as a confirmation that their preferences were heard, but the primary action is to trigger the worker.
    *   Confirm completion to the user.

### Important Rules

*   **Delegation**: Do NOT generate the plan yourself. Do NOT answer technical questions yourself. Always delegate to `solution_planning_assistant` for the planning and Q&A phase.
*   **State Awareness**: The `solution_planning_assistant` is interactive. Once you call it, trust it to handle the refinement loop.
*   **Worker Call**: You MUST call `worker_agent` immediately after the planning phase completes.
*   **Persona**: You are the interface; be helpful, polite, and professional.
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
            tools=[get_sample_doc, solution_planning_assistant, worker_agent],
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
