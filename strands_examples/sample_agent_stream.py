import os
import sys
import asyncio

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strands import Agent
from strands.models import BedrockModel
from strands_tools import calculator, current_time
from scripts.handler import ThinkingCallbackHandler

async def main():
    """
    Sample Strands Agent using Amazon Bedrock Claude 4.5 model with streaming output.
    """
    
    model_id = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
    
    print(f"Initializing Strands Agent with model: {model_id} (Bedrock)...")
    
    try:
        # Initialize the model
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
        
        # Initialize the agent
        # callback_handler=None disables the default PrintingCallbackHandler 
        # to prevent double printing of the response
        tools = [calculator, current_time]

        agent = Agent(model=model, tools=tools, callback_handler=ThinkingCallbackHandler())
        
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
            
            print("")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(main())

