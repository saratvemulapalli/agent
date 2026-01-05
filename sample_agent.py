import os
import sys

# Check if strands is installed
try:
    from strands import Agent
    from strands.models import BedrockModel
except ImportError:
    print("Error: 'strands-agents' package is not installed.")
    print("Please install it using: pip install strands-agents strands-agents-tools")
    sys.exit(1)

def main():
    """
    Sample Strands Agent using Amazon Bedrock Claude 4.5 model with synchronous output.
    """
    
    model_id = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
    
    print(f"Initializing Strands Agent with model: {model_id} (Bedrock)...")
    
    try:
        # Initialize the model
        model = BedrockModel(model_id=model_id)
        
        # Initialize the agent
        # callback_handler=None disables the default PrintingCallbackHandler 
        # to prevent double printing of the response
        agent = Agent(model=model, callback_handler=None)
        
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
            
            # Non-streaming synchronous response
            # Calling the agent instance directly invokes the synchronous __call__ method
            response = agent(user_input)
            print(response)
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()

