"""Tools that look at the picture or manage the session without drawing anything."""

from __future__ import annotations

from math import gcd
from typing import Annotated, Literal

from PIL import ImageOps, ImageStat

from pictor.session import has_transparency
from pictor.tools.common import clamp, size_text
from pictor.tools.registry import P, ToolContext, ToolError, ToolResult, tool


def _aspect(width: int, height: int) -> str:
    divisor = gcd(width, height)
    w, h = width // divisor, height // divisor
    if max(w, h) > 40:  # not a tidy ratio; a decimal says more
        return f"{width / height:.2f}:1"
    return f"{w}:{h}"


def _dominant_colors(image, count: int = 5) -> list[str]:
    small = image.convert("RGB").resize((64, 64))
    quantized = small.quantize(count)
    palette = quantized.getpalette() or []
    ranked = sorted(quantized.getcolors() or [], reverse=True)
    return [
        "#{:02x}{:02x}{:02x}".format(*palette[index * 3 : index * 3 + 3]) for _, index in ranked
    ]


@tool("Inspect")
def get_image_info(ctx: ToolContext) -> ToolResult:
    """Report facts about the current picture: pixel size, aspect ratio, transparency,
    overall brightness, dominant colours, file format and camera details when present.
    Call it when you need exact numbers, for example before resizing."""
    session = ctx.session
    image = ctx.image
    brightness = ImageStat.Stat(ImageOps.grayscale(image)).mean[0] / 255
    tone = "dark" if brightness < 0.35 else "bright" if brightness > 0.65 else "medium"
    lines = [
        f"Size: {size_text(image)} ({image.width * image.height / 1e6:.1f} megapixels)",
        f"Aspect ratio: {_aspect(image.width, image.height)}",
        f"Transparency: {'yes' if has_transparency(image) else 'no'}",
        f"Overall brightness: {brightness:.0%} ({tone})",
        f"Dominant colours: {', '.join(_dominant_colors(image))}",
        f"Original file: {session.filename} ({session.source_format.upper()})",
        f"Will be saved as: {session.effective_format.upper()}",
    ]
    lines += [f"{key.capitalize()}: {value}" for key, value in session.exif.items()]
    lines.append(f"Edits so far: {session.cursor}")
    return ToolResult("\n".join(lines))


@tool("Look", requires_vision=True)
def view_image(ctx: ToolContext) -> ToolResult:
    """Look at the picture as it is right now. Use it to check the result of an edit that
    depended on position, such as a crop, a blurred region, or text and shapes you placed.
    Simple global edits such as filters and resizing do not need checking."""
    return ToolResult("The current picture is attached to the next message.", show_image=True)


@tool("Output format")
def set_output_format(
    ctx: ToolContext,
    format: Annotated[Literal["png", "jpeg", "webp"], P("File format for the saved picture")],
    quality: Annotated[
        float, P("1 to 100 for JPEG and WebP; higher is better and larger", 1, 100)
    ] = 90,
) -> ToolResult:
    """Choose the file format the picture is saved in when the user downloads it. PNG is
    lossless and keeps transparency, JPEG is small but has no transparency, WebP is small
    and keeps transparency. The pixels are not changed."""
    session = ctx.session
    session.set_output(format, round(clamp(quality, 1, 100)))
    note = ""
    if format == "jpeg" and has_transparency(ctx.image):
        note = " JPEG has no transparency, so transparent areas will be saved as white."
    detail = "" if format == "png" else f" at quality {session.output_quality}"
    return ToolResult(f"The picture will be saved as {format.upper()}{detail}.{note}")


@tool("Undo")
def undo_last_edit(ctx: ToolContext) -> ToolResult:
    """Take back the most recent edit and return to the picture as it was before it. Use
    this when an edit came out wrong, then try again with better values."""
    session = ctx.session
    undone = session.current.label
    if not session.undo():
        raise ToolError("There is nothing to undo; this is the original picture.")
    return ToolResult(
        f"Undid '{undone}'. The picture is back to {size_text(ctx.image)} "
        f"(state: {session.current.label})."
    )


@tool("Reset")
def reset_to_original(ctx: ToolContext) -> ToolResult:
    """Throw away every edit and go back to the original picture the user opened. Only do
    this when the user asks to start over."""
    session = ctx.session
    if session.cursor == 0:
        raise ToolError("This already is the original picture.")
    session.goto(session.versions[0].id)
    return ToolResult(f"Back to the original picture, {size_text(ctx.image)}.")
