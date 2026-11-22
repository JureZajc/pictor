"""The tools the model can call.

Importing this package registers every tool. To add one, write a function in any module
here, decorate it with ``@tool("Label")`` and import the module below.
"""

from pictor.tools import (  # noqa: F401
    background,
    color,
    draw,
    generative,
    geometry,
    info,
    regions,
)
from pictor.tools.registry import (
    Tool,
    ToolContext,
    ToolError,
    ToolRegistry,
    ToolResult,
    registry,
    tool,
)

__all__ = ["Tool", "ToolContext", "ToolError", "ToolRegistry", "ToolResult", "registry", "tool"]
