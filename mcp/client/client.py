import asyncio
from typing import Optional
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from dotenv import load_dotenv
import os
from openai import OpenAI
load_dotenv()  # load environment variables from .env

class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")  # Ensure DeepSeek API key is 
        deepseek_api_base_url = "https://api.deepseek.com"  # Default DeepSeek API URL
        self.llm_client = OpenAI(
            api_key=deepseek_api_key,
            base_url=deepseek_api_base_url
        )
        self.llm_model = "deepseek-chat"

    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server
        
        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")
            
        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )
        
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        
        await self.session.initialize()
        
        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])

    async def process_query(self, query: str) -> str:
        """Process a query using LLM client and available tools"""
        messages = [
            {
                "role": "user",
                "content": query
            }
        ]

        response = await self.session.list_tools()
      
        available_tools = [{ 
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema
            }
        } for tool in response.tools]
        # Initial LLM API call
        response = self.llm_client.chat.completions.create(
            model=self.llm_model,
            max_tokens=1000,
            messages=messages,
            tools=available_tools
        )

        final_text = []

        for choice in response.choices:
            content = choice.message
            if content.tool_calls:
                for tool_call in content.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = tool_call.function.arguments
                    # Ensure tool_args is a dict
                    if isinstance(tool_args, str):
                        import json
                        tool_args = json.loads(tool_args)
                    # Execute tool call
                    result = await self.session.call_tool(tool_name, tool_args)
                    final_text.append(f"[Calling tool {tool_name} with args {tool_args}]")
                    # Continue conversation with tool results
                    messages.append({
                        "role": "user",
                        "content": result.content
                    })
                    # Get next response from LLM
                    response = self.llm_client.chat.completions.create(
                        model= self.llm_model,
                        max_tokens=1000,
                        messages=messages,
                    )
                    final_text.append(response.choices[0].message.content)
            else:
                final_text.append(content.content)

        return "\n".join(final_text)

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")
        
        while True:
            try:
                query = input("\nQuery: ").strip()
                
                if query.lower() == 'quit':
                    break
                    
                response = await self.process_query(query)
                print("\n" + response)
                    
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"\nError: {str(e)}")
    
    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()

async def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script>")
        sys.exit(1)
        
    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    import sys
    asyncio.run(main())