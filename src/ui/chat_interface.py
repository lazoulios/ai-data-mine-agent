import streamlit as st
import asyncio
import os
import re
import sys
from typing import List
import threading
import queue

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from ui.css_styling import INLINE_CSS_STYLING
from agent.agent_ollama import DaMiOllama
from environment import MODEL_RESPONSE_TIMEOUT

# Page config
st.set_page_config(
    page_title="DaMi, Your Intelligent Data Mining Companion!",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Simple inline CSS for basic styling
st.markdown(INLINE_CSS_STYLING, unsafe_allow_html=True)


def extract_file_paths(text: str, extensions: List[str]):
    """Extract file paths with specific extensions from text."""
    # Create pattern for the given extensions
    ext_pattern = "|".join(extensions)
    patterns = [
        rf"([^\s]+\.(?:{ext_pattern}))",  # Basic file paths
        rf"`([^`]+\.(?:{ext_pattern}))`",  # Paths in backticks
        rf'"([^"]+\.(?:{ext_pattern}))"',  # Paths in double quotes
        rf"'([^']+\.(?:{ext_pattern}))'",  # Paths in single quotes
    ]

    file_paths = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        file_paths.extend(matches)

    # Keep only existing files without duplicates
    existing_paths = []
    for path in file_paths:
        if os.path.exists(path) and path not in existing_paths:
            existing_paths.append(path)

    return existing_paths


def extract_image_paths(text: str):
    """Extract image paths from text."""
    return extract_file_paths(text, ["png", "jpg", "jpeg", "gif", "bmp", "webp"])


def extract_csv_paths(text: str):
    """Extract CSV file paths from text."""
    return extract_file_paths(text, ["csv"])


def extract_json_paths(text: str):
    """Extract JSON file paths from text."""
    return extract_file_paths(text, ["json"])


def display_csv_preview(file_path: str, max_rows: int = 5):
    """Display a preview of a CSV file."""
    try:
        import pandas as pd

        df = pd.read_csv(file_path)

        st.markdown(f"**📄 CSV Preview: {os.path.basename(file_path)}**")
        st.markdown(f"*Shape: {df.shape[0]} rows × {df.shape[1]} columns*")

        # Show first few rows
        if len(df) > max_rows:
            st.dataframe(df.head(max_rows))
            st.caption(f"Showing first {max_rows} of {len(df)} rows")
        else:
            st.dataframe(df)

    except Exception as e:
        st.error(f"Could not preview CSV {file_path}: {str(e)}")


def display_json_preview(file_path: str, max_items: int = 5):
    """Display a preview of a JSON file."""
    try:
        import json

        with open(file_path, "r") as f:
            data = json.load(f)

        st.markdown(f"**📄 JSON Preview: {os.path.basename(file_path)}**")

        if isinstance(data, dict):
            st.markdown(f"*Object with {len(data)} keys*")
            # Show first few key-value pairs
            preview_data = dict(list(data.items())[:max_items])
            st.json(preview_data)
            if len(data) > max_items:
                st.caption(f"Showing first {max_items} of {len(data)} keys")

        elif isinstance(data, list):
            st.markdown(f"*Array with {len(data)} items*")
            # Show first few items
            preview_data = data[:max_items]
            st.json(preview_data)
            if len(data) > max_items:
                st.caption(f"Showing first {max_items} of {len(data)} items")
        else:
            # Simple value
            st.json(data)

    except Exception as e:
        st.error(f"Could not preview JSON {file_path}: {str(e)}")


def display_tool_calls_popup(tool_calls, message_id):
    """Display tool calls in a popup dialog."""
    if not tool_calls:
        st.info("No tool calls were made for this response.")
        return

    st.markdown("### 🔧 Tool Execution Details")
    st.markdown("*Here are the tools the agent used to generate this response:*")

    # Group tool calls and responses
    call_pairs = {}
    standalone_calls = []

    for item in tool_calls:
        if item.get("type") == "tool_call":
            call_id = item.get("id", "")
            if call_id:
                call_pairs[call_id] = {"call": item, "response": None}
            else:
                standalone_calls.append(item)
        elif item.get("type") == "tool_response":
            call_id = item.get("tool_call_id", "")
            if call_id in call_pairs:
                call_pairs[call_id]["response"] = item
            else:
                standalone_calls.append(item)

    # Display paired calls and responses
    for i, (call_id, pair) in enumerate(call_pairs.items(), 1):
        call = pair["call"]
        response = pair["response"]

        tool_name = call.get("name", "Unknown")
        with st.expander(f"🛠️ Tool {i}: {tool_name}", expanded=True):
            # Tool Call section
            st.markdown("#### 📤 Tool Call")
            col1, col2 = st.columns([1, 2])

            with col1:
                st.markdown("**Tool Name:**")
                st.code(tool_name)

                st.markdown("**Call ID:**")
                st.code(call_id)

            with col2:
                st.markdown("**Arguments:**")
                args = call.get("args", {})
                if args:
                    st.json(args)
                else:
                    st.code("No arguments")

            # Tool Response section
            if response:
                st.markdown("#### 📥 Tool Response")
                content = response.get("content", "")

                # Try to parse as JSON for better display
                try:
                    import json

                    parsed_content = json.loads(content)
                    st.json(parsed_content)
                except:
                    # If not JSON, display as code
                    if len(content) > 500:
                        st.text_area(
                            "Response Content", content, height=200, disabled=True
                        )
                    else:
                        st.code(content)
            else:
                st.info("No response captured for this tool call")

    # Display standalone items
    for i, item in enumerate(standalone_calls, len(call_pairs) + 1):
        item_type = item.get("type", "unknown")
        name = item.get("name", "Unknown")

        with st.expander(f"🔧 {item_type.title()} {i}: {name}", expanded=False):
            st.json(item)


def display_message(message_data, is_user: bool = False, message_id: int = None):
    """Display a message using Streamlit's default styling."""
    # Handle both old string format and new dict format
    if isinstance(message_data, str):
        content = message_data
        tool_calls = []
    else:
        content = message_data.get("content", str(message_data))
        tool_calls = message_data.get("tool_calls", [])

    prefix = "👤 **You:**" if is_user else "🤖 **DaMi:**"

    # Create a container for the message header with tool calls button
    if not is_user and tool_calls:
        col1, col2 = st.columns([10, 1])
        with col1:
            st.markdown(f"{prefix}")
        with col2:
            # Question mark button for tool calls
            if st.button(
                "🔎", key=f"tool_calls_{message_id}", help="See behind the scenes"
            ):
                st.session_state[f"show_tool_calls_{message_id}"] = (
                    not st.session_state.get(f"show_tool_calls_{message_id}", False)
                )
    else:
        st.markdown(f"{prefix}")

    

    # Show tool calls as expandable section if requested
    if (
        not is_user
        and tool_calls
        and st.session_state.get(f"show_tool_calls_{message_id}", False)
    ):
        with st.expander("🔎 Behind the scenes", expanded=True):
            display_tool_calls_popup(tool_calls, message_id)
            
    # Display the message content
    st.markdown(content)

    if is_user or not content:
        st.divider()
        return

    # If the message is from DaMi and the content is not empty, show file previews (if any)

    # Show images (if any)
    image_paths = extract_image_paths(content)
    if image_paths:
        for image_path in image_paths:
            try:
                if os.path.exists(image_path):
                    st.image(
                        image_path,
                        caption=f"📊 {os.path.basename(image_path)}",
                        width=600,
                    )
                else:
                    st.warning(f"Image file not found: {image_path}")
            except Exception as e:
                st.error(f"Could not display image {image_path}: {str(e)}")

    # Show CSV previews (if any)
    csv_paths = extract_csv_paths(content)
    if csv_paths:
        for csv_path in csv_paths:
            if os.path.exists(csv_path):
                display_csv_preview(csv_path)
            else:
                st.warning(f"CSV file not found: {csv_path}")

    # Show JSON previews (if any)
    json_paths = extract_json_paths(content)
    if json_paths:
        for json_path in json_paths:
            if os.path.exists(json_path):
                display_json_preview(json_path)
            else:
                st.warning(f"JSON file not found: {json_path}")

    st.divider()


class AsyncAgentManager:
    """Manages the agent in a separate thread with its own event loop."""

    def __init__(self):
        self.agent = None
        self.loop = None
        self.thread = None
        self.request_queue = queue.Queue()
        self.response_queue = queue.Queue()
        self.shutdown_event = threading.Event()

    def start(self):
        """Start the async agent manager in a separate thread."""
        if self.thread is None or not self.thread.is_alive():
            self.thread = threading.Thread(target=self._run_async_loop, daemon=True)
            self.thread.start()

    def _run_async_loop(self):
        """Run the async event loop in the separate thread."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        try:
            self.loop.run_until_complete(self._async_worker())
        finally:
            self.loop.close()

    async def _async_worker(self):
        """Worker that handles async operations."""
        while not self.shutdown_event.is_set():
            try:
                # Check for requests with timeout
                try:
                    request = self.request_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                request_type, data, response_id = request

                try:
                    if request_type == "initialize":
                        result = await self._initialize_agent()
                        self.response_queue.put((response_id, "success", result))
                    elif request_type == "process_message":
                        result = await self._process_message(data)
                        self.response_queue.put((response_id, "success", result))
                    elif request_type == "cleanup":
                        await self._cleanup_agent()
                        self.response_queue.put((response_id, "success", None))
                    elif request_type == "shutdown":
                        break
                except Exception as e:
                    self.response_queue.put((response_id, "error", str(e)))

            except Exception as e:
                print(f"Error in async worker: {e}")

        # Final cleanup
        if self.agent:
            try:
                await self.agent.cleanup()
            except Exception as e:
                print(f"Error during final cleanup: {e}")

    async def _initialize_agent(self):
        """Initialize the agent."""
        if self.agent is None:
            self.agent = DaMiOllama()
            success = await self.agent.setup_mcp_tools()

            # Add our custom method to capture full responses
            self.agent.process_message_with_details = (
                self._create_detailed_process_method(self.agent)
            )

            if success and hasattr(self.agent, "tools") and self.agent.tools:
                tools_data = []
                for tool in self.agent.tools:
                    tools_data.append(
                        {"name": tool.name, "description": tool.description}
                    )
                return {"success": True, "tools": tools_data}
            else:
                self.agent = None
                return {"success": False, "tools": []}
        return {"success": True, "tools": []}

    async def _process_message(self, message):
        """Process a message with the agent and return full response data."""
        if self.agent:
            # We need to modify the agent to return both content and full response
            # For now, let's call the existing method and try to extract tool calls
            response_content = await self.agent.process_message_with_details(message)
            return response_content
        return {
            "content": "❌ Agent not initialized",
            "full_response": None,
            "tool_calls": [],
        }

    def _create_detailed_process_method(self, agent):
        """Create a method that captures full response details."""

        async def process_message_with_details(user_input: str):
            """Process a user message and return detailed response including tool calls."""
            if not agent.agent:
                return {
                    "content": "❌ Agent not properly initialized",
                    "tool_calls": [],
                    "full_response": None,
                }

            try:
                from langchain_core.messages import HumanMessage, AIMessage

                # Add user message to conversation history
                agent.conversation_history.append(HumanMessage(content=user_input))

                # Use the agent to process the message with full conversation history
                response = await agent.agent.ainvoke(
                    {"messages": agent.conversation_history}
                )

                # Print the full response for debugging (like in the original CLI)
                from pprint import pp

                print("\n" + "=" * 50)
                print("FULL RESPONSE DEBUG:")
                for item in response.items():
                    pp(item)
                print("=" * 50 + "\n")

                # Extract tool calls and tool responses from all messages in the response
                tool_calls = []
                if response and "messages" in response:
                    for msg in response["messages"]:
                        print(f"Processing message type: {type(msg).__name__}")
                        print(f"Message attributes: {dir(msg)}")

                        # Extract tool calls from AI messages
                        if hasattr(msg, "tool_calls") and msg.tool_calls:
                            print(f"Found {len(msg.tool_calls)} tool calls")
                            for tool_call in msg.tool_calls:
                                call_id = tool_call.get("id", "")
                                # Only include tool calls with valid call IDs
                                if call_id and call_id.strip():
                                    tool_calls.append(
                                        {
                                            "type": "tool_call",
                                            "name": tool_call.get("name", "Unknown"),
                                            "args": tool_call.get("args", {}),
                                            "id": call_id,
                                            "call_type": tool_call.get(
                                                "type", "tool_call"
                                            ),
                                        }
                                    )
                                else:
                                    print(
                                        f"Skipping tool call with empty ID: {tool_call}"
                                    )

                        # Extract tool responses from ToolMessage - check different possible attributes
                        msg_type = type(msg).__name__
                        if msg_type == "ToolMessage" or hasattr(msg, "tool_call_id"):
                            tool_call_id = getattr(msg, "tool_call_id", "")
                            # Only include tool responses with valid call IDs
                            if tool_call_id and tool_call_id.strip():
                                print(f"Found ToolMessage: {msg}")
                                tool_calls.append(
                                    {
                                        "type": "tool_response",
                                        "name": getattr(msg, "name", "Unknown"),
                                        "content": getattr(msg, "content", ""),
                                        "tool_call_id": tool_call_id,
                                        "id": getattr(msg, "id", ""),
                                    }
                                )
                            else:
                                print(
                                    f"Skipping tool response with empty tool_call_id: {msg}"
                                )
                        elif hasattr(msg, "name") and hasattr(msg, "content"):
                            # Alternative check for tool messages
                            tool_call_id = getattr(msg, "tool_call_id", "")
                            if tool_call_id and tool_call_id.strip():
                                print(f"Found potential tool response: {msg}")
                                tool_calls.append(
                                    {
                                        "type": "tool_response",
                                        "name": msg.name,
                                        "content": msg.content,
                                        "tool_call_id": tool_call_id,
                                        "id": getattr(msg, "id", ""),
                                    }
                                )
                            else:
                                print(
                                    f"Skipping potential tool response with empty tool_call_id: {msg}"
                                )

                # Print extracted tool calls for debugging
                print(f"\nExtracted {len(tool_calls)} tool calls:")
                for i, tc in enumerate(tool_calls):
                    print(f"  {i+1}. {tc}")
                print()

                # Extract the final response content
                assistant_response = ""
                if response and "messages" in response:
                    # Get the assistant's response (last message)
                    last_message = response["messages"][-1]
                    assistant_response = last_message.content

                    # Add assistant's response to conversation history
                    agent.conversation_history.append(
                        AIMessage(content=assistant_response)
                    )

                return {
                    "content": assistant_response,
                    "tool_calls": tool_calls,
                    "full_response": response,
                }

            except Exception as e:
                import traceback

                error_msg = f"Error processing message: {str(e)}\nTraceback: {traceback.format_exc()}"
                print(f"❌ {error_msg}")
                return {"content": error_msg, "tool_calls": [], "full_response": None}

        return process_message_with_details

    async def _cleanup_agent(self):
        """Clean up the agent."""
        if self.agent:
            await self.agent.cleanup()
            self.agent = None

    def _make_request(self, request_type, data=None):
        """Make a request to the async worker and wait for response."""
        import uuid

        response_id = str(uuid.uuid4())

        self.request_queue.put((request_type, data, response_id))

        # Wait for response
        while True:
            try:
                resp_id, status, result = self.response_queue.get(
                    timeout=MODEL_RESPONSE_TIMEOUT
                )
                if resp_id == response_id:
                    if status == "error":
                        raise Exception(result)
                    return result
            except queue.Empty:
                raise Exception("Request timed out")

    def initialize_agent(self):
        """Initialize agent synchronously."""
        self.start()
        return self._make_request("initialize")

    def process_message(self, message):
        """Process message synchronously."""
        return self._make_request("process_message", message)

    def cleanup(self):
        """Cleanup synchronously."""
        try:
            self._make_request("cleanup")
        except:
            pass

    def shutdown(self):
        """Shutdown the manager."""
        self.shutdown_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)


def get_agent_manager():
    """Get or create the agent manager."""
    if "agent_manager" not in st.session_state:
        st.session_state["agent_manager"] = AsyncAgentManager()
    return st.session_state["agent_manager"]


def get_agent_or_else_initialize():
    """Initialize agent and store in session state."""
    if "agent_initialized" not in st.session_state:
        try:
            manager = get_agent_manager()
            result = manager.initialize_agent()
            if result["success"]:
                st.session_state["agent_initialized"] = True
                st.session_state["tools_info"] = result["tools"]
            else:
                st.session_state["agent_initialized"] = False
        except Exception as e:
            print(f"Error initializing agent: {e}")
            st.session_state["agent_initialized"] = False

    return st.session_state.get("agent_initialized", False)


def get_agent_response(message: str):
    """Get response from agent."""
    if not get_agent_or_else_initialize():
        return {
            "content": "❌ Failed to setup the agent and its tools",
            "tool_calls": [],
            "full_response": None,
        }

    try:
        manager = get_agent_manager()
        response = manager.process_message(message)
        return response
    except Exception as e:
        return {
            "content": f"❌ Failed to get a response from the model: {e}",
            "tool_calls": [],
            "full_response": None,
        }


def get_tools_info():
    """Get tools information for the sidebar."""
    if "tools_info" not in st.session_state:
        # Initialize agent to get tools info
        get_agent_or_else_initialize()

    return st.session_state.get("tools_info", [])


def main():
    st.title("🤖 DaMi, the DAta MIning agent!")
    st.markdown("*Combine the power of Agentic AI with your data expertise to skyrocket your productivity*")

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "generating" not in st.session_state:
        st.session_state.generating = False

    # Create a container for the conversation area
    conversation_container = st.container()

    with conversation_container:
        # Display conversation
        for i, (role, content) in enumerate(st.session_state.messages):
            # Skip the generating placeholder - we'll handle it separately
            if role == "agent" and content == "GENERATING":
                continue
            display_message(content, is_user=(role == "user"), message_id=i)

        # Process agent response if we're generating - with spinner in conversation area
        if st.session_state.generating:
            # Get the user's last message
            user_messages = [
                msg for role, msg in st.session_state.messages if role == "user"
            ]
            if user_messages:
                last_user_message = user_messages[-1]

                # Use the original beautiful animated spinner RIGHT HERE in conversation
                with st.spinner("🤖 DaMi is working..."):
                    # Get agent response
                    response = get_agent_response(last_user_message)

                # Add the agent response (now includes tool calls)
                st.session_state.messages.append(("agent", response))
                st.session_state.generating = False

                # Rerun to show the response
                st.rerun()

    # Input form - use key to reset form after submission
    form_key = f"message_form_{len(st.session_state.messages)}"
    with st.form(form_key, clear_on_submit=True):
        user_input = st.text_area(
            "Your message:",
            placeholder="Ask me about data analysis, clustering, or visualization...",
            help="Type your message and press Ctrl+Enter or click Send",
            key=f"input_{len(st.session_state.messages)}",
        )

        col1, col2, col3 = st.columns([1, 1, 4])

        with col1:
            send_button = st.form_submit_button(
                "Send", type="primary", disabled=st.session_state.generating
            )

        with col2:
            clear_button = st.form_submit_button(
                "Clear Chat", disabled=st.session_state.generating
            )

    # Handle clear
    if clear_button:
        # Clean up agent if it exists
        if "agent_manager" in st.session_state:
            try:
                st.session_state["agent_manager"].cleanup()
                st.session_state["agent_manager"].shutdown()
            except Exception as e:
                print(f"Error cleaning up agent: {e}")

        # Clear session state
        st.session_state.messages = []
        st.session_state.generating = False
        if "agent_manager" in st.session_state:
            del st.session_state["agent_manager"]
        if "agent_initialized" in st.session_state:
            del st.session_state["agent_initialized"]
        if "tools_info" in st.session_state:
            del st.session_state["tools_info"]
        st.rerun()

    # Handle send
    if send_button and user_input.strip() and not st.session_state.generating:
        # Add user message immediately
        st.session_state.messages.append(("user", user_input.strip()))
        st.session_state.generating = True

        # Rerun to show user message and start processing
        st.rerun()

    # Sidebar with available tools
    with st.sidebar:
        st.header("🧹 Cleanup")

        col1, col2 = st.columns(2)

        with col1:
            if st.button(
                "🗑️ Clear Cache", help="Remove all files from .data_cache directory"
            ):
                try:
                    import shutil

                    cache_dir = ".data_cache"
                    if os.path.exists(cache_dir):
                        # Remove all files in the directory
                        for filename in os.listdir(cache_dir):
                            file_path = os.path.join(cache_dir, filename)
                            if os.path.isfile(file_path):
                                os.remove(file_path)
                        st.success(f"✅ Cleared cache directory")
                    else:
                        st.info("Cache directory doesn't exist")
                except Exception as e:
                    st.error(f"❌ Error clearing cache: {e}")

        with col2:
            if st.button("🖼️ Clear Plots", help="Remove all files from plots directory"):
                try:
                    plots_dir = "plots"
                    if os.path.exists(plots_dir):
                        # Remove all files in the directory
                        for filename in os.listdir(plots_dir):
                            file_path = os.path.join(plots_dir, filename)
                            if os.path.isfile(file_path):
                                os.remove(file_path)
                        st.success(f"✅ Cleared plots directory")
                    else:
                        st.info("Plots directory doesn't exist")
                except Exception as e:
                    st.error(f"❌ Error clearing plots: {e}")

        st.header("🛠️ Available Tools")

        # Get and display available tools
        try:
            tools_data = get_tools_info()

            if tools_data:
                # Sort tools alphabetically for consistent display
                tools_data.sort(key=lambda x: x["name"])

                # Display each tool as an expandable section
                for tool in tools_data:
                    with st.expander(f"🔧 {tool['name']}"):
                        st.markdown(tool["description"])
            else:
                st.info("No tools loaded yet.")

        except Exception as e:
            st.error(f"Could not load tools: {e}")

        # Stats section
        if st.session_state.messages:
            st.header("📊 Session Stats")
            user_msgs = len([m for m in st.session_state.messages if m[0] == "user"])
            agent_msgs = len([m for m in st.session_state.messages if m[0] == "agent"])
            st.metric("Your messages", user_msgs)
            st.metric("Agent responses", agent_msgs)


if __name__ == "__main__":
    main()
