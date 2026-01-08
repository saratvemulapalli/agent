import os
import sys
import asyncio

from strands import Agent, tool
from strands.models import BedrockModel
from scripts.handler import ThinkingCallbackHandler

@tool(name="read_knowledge_base", description="Read the OpenSearch Semantic Search Guide to retrieve detailed information about search methods (BM25, Dense Vector, Sparse Vector, Hybrid), algorithms (HNSW, IVF, Disk-based), cost profiles, and deployment options.")
def read_knowledge_base() -> str:
    try:
        with open("opensearch_semantic_search_guide.md", "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error reading knowledge base: {e}"

async def main():
    """
    Sample Strands Agent using Amazon Bedrock Claude 4.5 model with streaming output.
    """
    
    model_id = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
    
    # 1. Read the system instruction
    instruction_file_path = "prompt.md"
    with open(instruction_file_path, "r", encoding="utf-8") as f:
        instruction_content = f.read()

    # 2. Define the System Instruction
    system_prompt = instruction_content

    print(f"Initializing Strands Agent with model: {model_id} (Bedrock)...")
    
    try:
        # Initialize the model with reasoning enabled
        # We use additional_request_fields to pass the 'thinking' parameter
        # budget_tokens determines how much effort the model spends on reasoning
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
        
        # Initialize the agent with the system_prompt
        # callback_handler=None disables the default PrintingCallbackHandler 
        agent = Agent(
            model=model, 
            system_prompt=system_prompt,
            tools=[read_knowledge_base],
            callback_handler=ThinkingCallbackHandler()
        )
        
    except Exception as e:
        print(f"Failed to initialize agent: {e}")
        return

    print("Agent ready. Type 'exit' or 'quit' to stop.")
    print("-" * 50)

    while True:
        try:
            # Get user input
            # Note: input() is blocking, but acceptable for this simple CLI
            user_input = input("\nYou: ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ['exit', 'quit']:
                print("Goodbye!")
                break
            
            print("Agent: ", end="", flush=True)
            
            # Stream the response
            # using stream_async for streaming output
            stream = agent.stream_async(user_input)
            
            async for event in stream:
                pass
            
            print() # Print a newline at the end of the response
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(main())

