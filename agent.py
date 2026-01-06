import os
import sys
import asyncio

# Check if strands is installed
try:
    from strands import Agent
    from strands.models import BedrockModel
except ImportError:
    print("Error: 'strands-agents' package is not installed.")
    print("Please install it using: pip install strands-agents strands-agents-tools")
    sys.exit(1)

async def main():
    """
    Sample Strands Agent using Amazon Bedrock Claude 4.5 model with streaming output.
    """
    
    model_id = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
    
    # 1. Read the external knowledge base file
    kb_file_path = "opensearch_semantic_search_guide.md"
    instruction_file_path = "prompt.md"
    with open(instruction_file_path, "r", encoding="utf-8") as f:
        instruction_content = f.read()
    with open(kb_file_path, "r", encoding="utf-8") as f:
        knowledge_content = f.read()

    # 2. Define the System Instruction and inject Knowledge
    system_prompt = instruction_content + "\n\n" + knowledge_content

    print(f"Initializing Strands Agent with model: {model_id} (Bedrock)...")
    
    try:
        # Initialize the model with reasoning enabled
        # We use additional_request_fields to pass the 'thinking' parameter
        # budget_tokens determines how much effort the model spends on reasoning
        model = BedrockModel(
            model_id=model_id,
            additional_request_fields={
                "thinking": {
                    "type": "enabled",
                    "budget_tokens": 4096  # Set your desired budget here
                }
            }
        )
        
        # Initialize the agent with the system_prompt
        # callback_handler=None disables the default PrintingCallbackHandler 
        agent = Agent(
            model=model, 
            system_prompt=system_prompt
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

