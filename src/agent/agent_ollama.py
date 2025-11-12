import os, sys
from pprint import pp

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import asyncio
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_ollama import ChatOllama
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from agent.system_prompt import SYSTEM_PROMPT
from agent.utils import parent_dir
from environment import OLLAMA_HOST, OLLAMA_MODEL, OLLAMA_MODEL_TEMPERATURE, RECURSION_LIMIT

MCP_SERVER_FILE_NAME = "mcp_server.py"


class DaMiOllama:
    """DaMi, the DAta MIning agent!"""

    def __init__(self):
        """Initialize the data analysis agent with Ollama model and MCP tools"""
        self.conversation_history = []
        self.tools = []
        self.agent = None
        self.ollama_host = None
        self.ollama_model = None
        self.ollama_model_temperature = None

        self._setup_llm()
        self._print_agent_information()

    def _setup_llm(self):

        self.ollama_host = OLLAMA_HOST
        if not self.ollama_host:
            raise ValueError(
                "OLLAMA_HOST must be specified in the .env file. See the instructions of the assignment."
            )

        self.ollama_model = OLLAMA_MODEL
        if not self.ollama_model:
            raise ValueError(
                "OLLAMA_MODEL must be specified in the .env file. See the instructions of the assignment."
            )

        self.ollama_model_temperature = OLLAMA_MODEL_TEMPERATURE

        self.llm = ChatOllama(
            name="DaMi",  # DAta MIning and warehouses Course :)
            model=self.ollama_model,
            temperature=self.ollama_model_temperature,
            reasoning=True,  # this works only for selected models: https://ollama.com/search?c=thinking
        )

    def _print_agent_information(self):
        print("🤖  Data Analysis Agent initializing...")
        print(f"📡  Ollama Host: {self.ollama_host}")
        print(f"🧠  Model: {self.ollama_model}")
        print(f"🌡️  Model Temperature: {self.ollama_model_temperature}")

    @property
    def model(self) -> ChatOllama:
        """Get the LLM model instance."""
        return self.llm

    async def setup_mcp_tools(self):
        """Set up MCP tools using the proper adapter."""
        try:
            # Get the absolute path to our main.py MCP server
            current_dir = os.path.dirname(os.path.abspath(__file__))
            server_path = os.path.join(current_dir, "main.py")
            server_path = os.path.join(parent_dir, MCP_SERVER_FILE_NAME)

            # Create server parameters for our MCP server
            server_params = StdioServerParameters(
                command="python",
                args=[server_path],
            )

            print(f"🔧 Connecting to MCP server: {server_path}")

            # Store the connection context managers for later cleanup
            self._stdio_client = stdio_client(server_params)
            self.read, self.write = await self._stdio_client.__aenter__()

            self._session = ClientSession(self.read, self.write)
            self.session = await self._session.__aenter__()

            # Initialize the connection
            await self.session.initialize()

            # Load MCP tools using the adapter
            self.tools = await load_mcp_tools(self.session)

            print(f"🛠️  Loaded {len(self.tools)} MCP tools")

            # Create the agent with Ollama, MCP tools, and system prompt
            self.agent = create_agent(
                model=self.llm,
                tools=self.tools,
                system_prompt=SYSTEM_PROMPT,
            ).with_config({"recursion_limit": RECURSION_LIMIT})

            print("✅ Agent setup complete!")
            return True

        except Exception as e:
            import traceback

            print(f"❌ Failed to setup MCP tools: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            return False

    async def cleanup(self):
        """Clean up MCP connections."""
        try:
            if hasattr(self, "_session"):
                await self._session.__aexit__(None, None, None)
            if hasattr(self, "_stdio_client"):
                await self._stdio_client.__aexit__(None, None, None)
        except Exception as e:
            print(f"⚠️  Warning during cleanup: {e}")

    async def process_message(self, user_input: str) -> str:
        """Process a user message and return the agent's response."""
        if not self.agent:
            return """❌ Agent not properly initialized. Please make sure that the 
        Ollama server is up and running. Also, please ensure that you have specified
        the correct Ollama host (ngrok) address in the .env file. Every time you run 
        ngrok, a different, random IP address is generated."""

        try:
            # Add user message to conversation history
            self.conversation_history.append(HumanMessage(content=user_input))

            # Use the agent to process the message with full conversation history
            response = await self.agent.ainvoke({"messages": self.conversation_history})

            # Extract the response content
            if response and "messages" in response:
                # Get the assistant's response (last message)
                last_message = response["messages"][-1]
                for item in response.items():
                    pp(item)
                print()
                assistant_response = last_message.content

                # Add assistant's response to conversation history
                self.conversation_history.append(AIMessage(content=assistant_response))

                return assistant_response
            else:
                return "❌ No response from agent"

        except Exception as e:
            import traceback

            error_msg = f"Error processing message: {str(e)}\nTraceback: {traceback.format_exc()}"
            print(f"❌ {error_msg}")
            return error_msg

    def clear_conversation_history(self):
        """Clear the conversation history."""
        self.conversation_history = []
        print("🧹 Conversation history cleared!")

    def get_conversation_stats(self) -> dict:
        """Get conversation statistics."""
        return {
            "total_messages": len(self.conversation_history),
            "user_messages": len(
                [
                    msg
                    for msg in self.conversation_history
                    if isinstance(msg, HumanMessage)
                ]
            ),
            "assistant_messages": len(
                [msg for msg in self.conversation_history if isinstance(msg, AIMessage)]
            ),
        }

    async def run_multi_turn_conversation(self):
        """Run the agent in interactive mode (multi-turn conversation)."""
        try:
            # Setup MCP tools first
            if not await self.setup_mcp_tools():
                print("❌ Failed to initialize MCP tools. Exiting.")
                return

            print("\n💬 Starting interactive conversation...")
            print("You can ask me about data analysis, clustering, or visualization!")
            print(
                "Example: 'What tables are available?' or 'Help me analyze customer data'"
            )
            print("\nSpecial commands:")
            print("  - 'clear' or 'reset' - Clear conversation history")
            print("  - 'stats' - Show conversation statistics")
            print("  - 'quit' or 'exit' - End the conversation")
            print()

            while True:
                try:
                    # Get user input
                    user_input = input("👤 You: ").strip()

                    # Check for special commands
                    if user_input.lower() in ["quit", "exit", "bye"]:
                        print("👋 Goodbye!")
                        exit(0)
                    elif user_input.lower() in ["clear", "reset"]:
                        self.clear_conversation_history()
                        continue
                    elif user_input.lower() == "stats":
                        stats = self.get_conversation_stats()
                        print(
                            f"📊 Conversation Stats: {stats['total_messages']} total messages ({stats['user_messages']} from you, {stats['assistant_messages']} from assistant)"
                        )
                        continue

                    if not user_input:
                        continue

                    # Process the message
                    print("🤖 Assistant: ", end="", flush=True)
                    response = await self.process_message(user_input)
                    print(response)
                    print()  # Add blank line for readability

                except KeyboardInterrupt:
                    print("\n👋 Goodbye!")
                    break
                except Exception as e:
                    print(f"❌ Error: {e}")
                    continue
        finally:
            # Clean up connections
            await self.cleanup()


async def main():
    """Main entry point."""
    try:
        # Create and run the agent
        agent = DaMiOllama()
        await agent.run_multi_turn_conversation()

    except Exception as e:
        print(f"❌ Failed to start agent: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
