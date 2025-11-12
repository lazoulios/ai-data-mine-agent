import sys
import asyncio
from mcp.server.fastmcp import FastMCP
from tools import register_all_tools

# Create an MCP server
mcp = FastMCP("Demo")
register_all_tools(mcp)


def main():
    try:
        print("Starting Educational MCP Server...", file=sys.stderr)
        mcp.run()
    except KeyboardInterrupt:
        print("👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"Failed to start server: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
