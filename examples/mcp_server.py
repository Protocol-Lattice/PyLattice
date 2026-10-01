"""Small offline MCP server: uv run python examples/mcp_server.py."""

from mcp.server.fastmcp import FastMCP

server = FastMCP("Example tools")


@server.tool()
def add(a: int, b: int) -> int:
    """Add two integers without changing any files."""
    return a + b


if __name__ == "__main__":
    server.run(transport="stdio")
