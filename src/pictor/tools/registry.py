"""Tool plumbing: declare a tool as a plain function, get a JSON schema and safe calls.

A tool is a function whose first parameter is a :class:`ToolContext` and whose remaining
parameters are what the model fills in. The docstring becomes the description the model
reads, so it is written for the model.
"""

from __future__ import annotations

import inspect
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model
from pydantic.fields import FieldInfo

if TYPE_CHECKING:
    from pictor.session import Session

_NUMBER_WITH_UNIT = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*(%|px|pixels?|deg|degrees?|°)?\s*$", re.I)


class ToolError(Exception):
    """A problem the model can fix by calling the tool differently."""


@dataclass
class ToolResult:
    #: What the model is told happened.
    message: str
    #: A new picture, which becomes the next version. ``None`` leaves the picture alone.
    image: Image.Image | None = None
    #: Short description of the edit for the version strip, e.g. "16:9".
    detail: str = ""
    #: Ask the agent to show the current picture to the model on its next turn.
    show_image: bool = False


@dataclass
class ToolContext:
    session: Session
    #: Whether the model can see pictures.
    vision: bool = True
    #: Reports progress on slow steps (a model download, say) to the UI.
    status: Callable[[str], None] = lambda _text: None
    #: Settings from the user's config that some tools read.
    settings: dict[str, Any] = field(default_factory=dict)

    @property
    def image(self) -> Image.Image:
        """The current picture, RGB or RGBA. Never modify it in place."""
        return self.session.image()


def P(description: str, minimum: float | None = None, maximum: float | None = None) -> FieldInfo:
    """Describe a parameter for the model.

    The range is a hint in the schema, not a validation rule: tools clamp out-of-range
    numbers themselves, which serves a small model better than rejecting the call.
    """
    extra: dict[str, Any] = {}
    if minimum is not None:
        extra["minimum"] = minimum
    if maximum is not None:
        extra["maximum"] = maximum
    return Field(description=description, json_schema_extra=extra or None)


def _strip_titles(schema: Any) -> Any:
    if isinstance(schema, dict):
        return {k: _strip_titles(v) for k, v in schema.items() if k != "title"}
    if isinstance(schema, list):
        return [_strip_titles(item) for item in schema]
    return schema


def _clean(arguments: dict) -> dict:
    """Forgive the common ways a model writes numbers: ``"50%"``, ``"90 deg"``, ``"12px"``."""
    cleaned = {}
    for key, value in arguments.items():
        if isinstance(value, str):
            match = _NUMBER_WITH_UNIT.match(value)
            if match and match.group(2):
                value = float(match.group(1))
        cleaned[key] = value
    return cleaned


@dataclass
class Tool:
    name: str
    label: str
    description: str
    func: Callable[..., ToolResult]
    args_model: type[BaseModel]
    #: Only offered to models that can see pictures.
    requires_vision: bool = False

    def schema(self) -> dict:
        parameters = _strip_titles(self.args_model.model_json_schema())
        parameters.setdefault("properties", {})
        parameters.setdefault("required", [])
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": parameters,
            },
        }

    def run(self, ctx: ToolContext, arguments: dict | None) -> ToolResult:
        try:
            parsed = self.args_model.model_validate(_clean(arguments or {}))
        except ValidationError as exc:
            problems = "; ".join(
                f"{'.'.join(str(p) for p in err['loc']) or 'arguments'}: {err['msg']}"
                for err in exc.errors()
            )
            raise ToolError(f"Invalid arguments for {self.name}: {problems}") from exc
        return self.func(ctx, **parsed.model_dump())


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def tool(self, label: str, *, requires_vision: bool = False) -> Callable:
        """Register the decorated function as a tool named after it."""

        def decorate(func: Callable[..., ToolResult]) -> Callable[..., ToolResult]:
            signature = inspect.signature(func, eval_str=True)
            fields: dict[str, Any] = {}
            for name, parameter in list(signature.parameters.items())[1:]:  # skip ctx
                default = ... if parameter.default is inspect.Parameter.empty else parameter.default
                fields[name] = (parameter.annotation, default)
            args_model = create_model(
                f"{func.__name__}_args", __config__=ConfigDict(extra="ignore"), **fields
            )
            self._tools[func.__name__] = Tool(
                name=func.__name__,
                label=label,
                description=" ".join((func.__doc__ or "").split()),
                func=func,
                args_model=args_model,
                requires_vision=requires_vision,
            )
            return func

        return decorate

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def available(self, vision: bool = True) -> list[Tool]:
        return [t for t in self._tools.values() if vision or not t.requires_vision]

    def schemas(self, vision: bool = True) -> list[dict]:
        return [t.schema() for t in self.available(vision)]

    def names(self, vision: bool = True) -> list[str]:
        return [t.name for t in self.available(vision)]


registry = ToolRegistry()
tool = registry.tool
