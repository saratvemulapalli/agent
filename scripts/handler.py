from strands import Agent

class ThinkingCallbackHandler:
    def __init__(self):
        self.tool_count = 0
        self.previous_tool_use = None
    
    def __call__(self, **kwargs):
        reasoning_text = kwargs.get("reasoningText")
        data = kwargs.get("data")
        complete = kwargs.get("complete")
        current_tool_use = kwargs.get("current_tool_use", {})
        
        # 1. Handle Thinking/Reasoning
        if reasoning_text:
            # Print reasoning in gray
            print(f"\033[90m{reasoning_text}\033[0m", end="", flush=True)
        
        # 2. Handle Text Data
        if data:
            print(data, end="" if not complete else "\n", flush=True)

        # 3. Handle Tool Use (Restored functionality)
        if current_tool_use and current_tool_use.get("name"):
            # Check if this is a new tool call or continuation of the same one
            if self.previous_tool_use != current_tool_use:
                self.previous_tool_use = current_tool_use
                self.tool_count += 1
                tool_name = current_tool_use.get("name", "Unknown tool")
                print(f"\nTool #{self.tool_count}: {tool_name}.")
        
        # 4. Handle Completion
        if complete and data:
            print("\n")
