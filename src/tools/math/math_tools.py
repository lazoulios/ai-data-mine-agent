def register_math_tools(mcp):
    """Register math tools for the given MCP server instance"""

    @mcp.tool()
    def add(a: int, b: int) -> int:
        """Add two numbers

        Args:
            a (int): the first number
            b (int): the second number

        Returns:
            int: the result of a + b
        """
        return a + b