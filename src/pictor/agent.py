"""The agent loop: show the model the picture and let it call tools until the job is done.

:meth:`Agent.run` is an async generator of plain dict events, which the server forwards
to the browser as they happen:

``status``       a short note while waiting, e.g. for the model to load
``thinking``     a piece of the model's reasoning (``delta``)
``text``         a piece of the model's reply (``delta``)
``tool_start``   a tool call began (``id``, ``name``, ``label``, ``args``)
``tool_status``  progress from a slow tool (``id``, ``text``)
``tool_end``     a tool call finished (``id``, ``ok``, ``result``, ``version``)
``state``        the session after a change (versions, cursor, size)
``done``         the run is over (``steps``, ``limit_reached``)
``error``        the run failed (``message``)
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass

import numpy as np
from PIL import Image

from pictor.catalog import suggest_num_ctx
from pictor.config import Config
from pictor.ollama import Ollama, OllamaError
from pictor.session import Session, has_transparency
from pictor.tools import ToolContext, ToolError, ToolRegistry, ToolResult, registry

log = logging.getLogger("pictor.agent")

#: How many earlier instructions the model is reminded of.
MAX_HISTORY_TURNS = 6

SYSTEM_PROMPT = """\
You are Pictor, an assistant that edits one picture for the user by calling tools. \
You run entirely on the user's own computer.

How you work
- The user has opened a picture. {vision}
- To change the picture you MUST call tools. Describing an edit does not perform it.
- Every edit applies to the result of the previous one. Call as many tools as the \
request needs, one after another, until it is completely done.
- If the user only asks a question about the picture, answer it and do not edit anything.

Coordinates
- Positions and regions are percentages of the CURRENT picture measured from its \
top-left corner: 0 is the left or top edge, 100 is the right or bottom edge.
- After a crop, resize or rotation, percentages refer to the new picture.

Doing it well
- Do exactly what was asked and nothing more. Never add edits the user did not ask for.
- Pick the most specific tool: crop_to_aspect_ratio for "make it square", \
anonymize_faces to hide faces, change_background for background effects.
- If an edit came out wrong, call undo_last_edit and try again with better values \
instead of stacking a correction on top.
- If something cannot be done with your tools, such as generating new content or \
erasing an object and inventing what was behind it, say so plainly and offer the \
closest thing you can do.
- If the request is unclear in a way that matters, ask one short question instead of \
guessing.

When you are finished, reply with one or two plain sentences saying what you did, in \
the language the user wrote in. Do not mention tool names or parameters."""

_VISION_YES = (
    "You can see it: the current picture is attached to the user's message. Call "
    "view_image to look again after edits whose result depends on position."
)
_VISION_NO = (
    "You cannot see pictures, so use get_image_info and detect_faces to learn about it, "
    "and tell the user when a request needs you to see the picture."
)


@dataclass(frozen=True)
class ModelInfo:
    tag: str
    vision: bool
    tools: bool
    #: Value for the API's ``think`` field, or ``None`` to leave the model's default.
    think: bool | str | None


def resolve_think(mode: str, details: dict) -> bool | str | None:
    """Translate the user's thinking preference into what this model accepts."""
    capabilities = details.get("capabilities") or []
    values = (details.get("thinking") or {}).get("values") or []
    if "thinking" not in capabilities and not values:
        return None
    if mode == "default":
        return None
    if mode == "off":
        return False
    # "auto": short reasoning helps choose tools, long reasoning mostly costs time.
    return "low" if "low" in values else None


def _checkerboard(size: tuple[int, int], tile: int = 16) -> Image.Image:
    ys, xs = np.indices((size[1], size[0]))
    pattern = ((xs // tile + ys // tile) % 2).astype(np.uint8)
    return Image.fromarray(np.where(pattern, 204, 255).astype(np.uint8)).convert("RGB")


def encode_for_model(image: Image.Image, max_side: int) -> str:
    """Base64 JPEG of the picture, scaled down for the model's vision encoder.

    Transparency is drawn as a checkerboard, the convention the model knows from
    screenshots of image editors.
    """
    if max(image.size) > max_side:
        image = image.copy()
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    if image.mode == "RGBA":
        backdrop = _checkerboard(image.size)
        backdrop.paste(image, mask=image.getchannel("A"))
        image = backdrop
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, "JPEG", quality=88)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def picture_note(session: Session) -> str:
    image = session.image()
    note = f"[Current picture: {image.width}×{image.height} px"
    if has_transparency(image):
        note += ", with transparent areas (drawn as a grey checkerboard)"
    return note + "]"


def explain_failure(message: str, model: str) -> str:
    """Add what to do about the failures a user can actually fix."""
    low = message.lower()
    if "memory" in low or "oom" in low:
        return (
            f"{model} does not fit in this computer's memory, so Ollama could not run it. "
            f"Choose a smaller model from the model menu. (Ollama said: {message})"
        )
    if "not found" in low and "model" in low:
        return f"{model} is not installed in Ollama any more. Choose a model from the model menu."
    return message


def _parse_arguments(raw: object) -> dict:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            return {}
    return dict(raw) if isinstance(raw, dict) else {}


def _stored(message: dict) -> dict:
    """A message as kept for later turns: no picture data and no reasoning trace."""
    kept = {k: v for k, v in message.items() if k not in ("images", "thinking")}
    if "images" in message:
        kept["content"] = f"{message.get('content', '')}\n[picture omitted]".strip()
    return kept


class Agent:
    def __init__(
        self,
        client: Ollama,
        config: Config,
        budget_gb: float,
        tools: ToolRegistry = registry,
    ) -> None:
        self.client = client
        self.config = config
        self.budget_gb = budget_gb
        self.tools = tools

    async def describe(self, model: str) -> ModelInfo:
        details = await self.client.show(model)
        capabilities = details.get("capabilities") or []
        return ModelInfo(
            tag=model,
            vision="vision" in capabilities,
            tools="tools" in capabilities,
            think=resolve_think(self.config.think, details),
        )

    async def run(self, session: Session, text: str) -> AsyncIterator[dict]:
        """Carry out one instruction from the user on the session's picture."""
        model = self.config.model
        if not model:
            yield {"type": "error", "message": "No model has been chosen yet."}
            return
        try:
            info = await self.describe(model)
        except OllamaError as exc:
            yield {"type": "error", "message": str(exc)}
            return
        if not info.tools:
            yield {
                "type": "error",
                "message": f"{model} cannot call tools. Choose another model in the settings.",
            }
            return

        system = SYSTEM_PROMPT.format(vision=_VISION_YES if info.vision else _VISION_NO)
        user: dict = {"role": "user", "content": f"{text}\n\n{picture_note(session)}"}
        if info.vision:
            user["images"] = [encode_for_model(session.image(), self.config.vision_max_side)]
        turn: list[dict] = [user]
        history = [m for past in session.turns[-MAX_HISTORY_TURNS:] for m in past]
        messages = [{"role": "system", "content": system}, *history, *turn]
        schemas = self.tools.schemas(info.vision)
        options = {"num_ctx": self.config.num_ctx or suggest_num_ctx(self.budget_gb)}

        loop = asyncio.get_running_loop()
        call_count = 0
        last_edit: str | None = None
        limit_reached = True
        yield {"type": "status", "text": "Waking the model…"}

        try:
            for _step in range(self.config.max_steps):
                content, thinking, calls = "", "", []
                async for chunk in self.client.chat(
                    model, messages, tools=schemas, think=info.think, options=options
                ):
                    message = chunk.get("message") or {}
                    if message.get("thinking"):
                        thinking += message["thinking"]
                        yield {"type": "thinking", "delta": message["thinking"]}
                    if message.get("content"):
                        content += message["content"]
                        yield {"type": "text", "delta": message["content"]}
                    if message.get("tool_calls"):
                        calls.extend(message["tool_calls"])

                reply: dict = {"role": "assistant", "content": content}
                if thinking:
                    reply["thinking"] = thinking
                if calls:
                    reply["tool_calls"] = calls
                messages.append(reply)
                turn.append(reply)
                if not calls:
                    limit_reached = False
                    break

                show_picture = False
                for call in calls:
                    function = call.get("function") or {}
                    name = str(function.get("name") or "")
                    arguments = _parse_arguments(function.get("arguments"))
                    call_count += 1
                    call_id = f"t{call_count}"
                    tool = self.tools.get(name)
                    if tool and tool.requires_vision and not info.vision:
                        tool = None
                    yield {
                        "type": "tool_start",
                        "id": call_id,
                        "name": name,
                        "label": tool.label if tool else name,
                        "args": arguments,
                    }

                    result: ToolResult | None = None
                    signature = json.dumps([name, arguments], sort_keys=True, default=str)
                    if tool is None:
                        known = ", ".join(self.tools.names(info.vision))
                        outcome = f"There is no tool named {name!r}. Available tools: {known}."
                    elif signature == last_edit:
                        outcome = (
                            "This exact edit was applied a moment ago and has not been repeated. "
                            "If the request is done, reply to the user now."
                        )
                    else:
                        statuses: asyncio.Queue[str] = asyncio.Queue()

                        def report(text: str, queue: asyncio.Queue = statuses) -> None:
                            loop.call_soon_threadsafe(queue.put_nowait, text)

                        ctx = ToolContext(
                            session=session,
                            vision=info.vision,
                            status=report,
                            settings={"background_model": self.config.background_model},
                        )
                        task = asyncio.ensure_future(asyncio.to_thread(tool.run, ctx, arguments))
                        while not task.done():
                            waiter = asyncio.ensure_future(statuses.get())
                            await asyncio.wait({task, waiter}, return_when=asyncio.FIRST_COMPLETED)
                            if waiter.done():
                                note = waiter.result()
                                yield {"type": "tool_status", "id": call_id, "text": note}
                            else:
                                waiter.cancel()
                        while not statuses.empty():  # notes sent just before it finished
                            yield {
                                "type": "tool_status",
                                "id": call_id,
                                "text": statuses.get_nowait(),
                            }
                        try:
                            result = task.result()
                            outcome = result.message
                        except ToolError as exc:
                            outcome = f"Error: {exc}"
                        except Exception as exc:  # noqa: BLE001 - a tool bug must not end the run
                            log.exception("Tool %s failed", name)
                            outcome = f"Error: the tool failed unexpectedly ({exc})."

                    version = None
                    if result is not None:
                        if result.image is not None:
                            version = session.push(result.image, tool.label, result.detail)
                            last_edit = signature
                        else:
                            last_edit = None
                        show_picture = show_picture or result.show_image
                    tool_message = {"role": "tool", "tool_name": name, "content": outcome}
                    messages.append(tool_message)
                    turn.append(tool_message)
                    yield {
                        "type": "tool_end",
                        "id": call_id,
                        "ok": result is not None,
                        "result": outcome,
                        "version": version.to_dict() if version else None,
                    }
                    yield {"type": "state", "state": session.state()}

                if show_picture and info.vision:
                    look = {
                        "role": "user",
                        "content": f"Here is the picture as it is now. {picture_note(session)}",
                        "images": [encode_for_model(session.image(), self.config.vision_max_side)],
                    }
                    messages.append(look)
                    turn.append(look)
        except OllamaError as exc:
            yield {"type": "error", "message": explain_failure(str(exc), model)}
            return

        if limit_reached:
            note = (
                f"I stopped after {self.config.max_steps} steps without finishing. "
                "Check the result, then tell me what is still missing."
            )
            turn.append({"role": "assistant", "content": note})
            yield {"type": "text", "delta": note}
        # Only a finished exchange is remembered; a cancelled or failed one could leave
        # tool calls without results, which the next request would choke on.
        session.turns.append([_stored(message) for message in turn])
        yield {"type": "done", "steps": call_count, "limit_reached": limit_reached}
