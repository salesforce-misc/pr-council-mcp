"""pr-council-mcp: a standalone local MCP server for durable pull-request reviews."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("pr-council-mcp")
except PackageNotFoundError:  # pragma: no cover - only during local, uninstalled use
    __version__ = "0.0.0"
