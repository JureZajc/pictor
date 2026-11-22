"""Tools that draw on the picture: text and outline shapes."""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Annotated, Literal

from PIL import Image, ImageDraw, ImageFont

from pictor.tools.common import clamp, parse_color, pct_box
from pictor.tools.registry import P, ToolContext, ToolError, ToolResult, tool

Position = Literal[
    "top-left",
    "top-center",
    "top-right",
    "center-left",
    "center",
    "center-right",
    "bottom-left",
    "bottom-center",
    "bottom-right",
]

# Pillow's built-in font has no accented letters (no č, š, ž, é), so a system font is
# preferred and the built-in one is only the last resort.
_FONT_FILES = {
    True: [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "C:/Windows/Fonts/segoeuib.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
        "/usr/share/fonts/noto/NotoSans-Bold.ttf",
    ],
    False: [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/noto/NotoSans-Regular.ttf",
    ],
}


@functools.lru_cache(maxsize=4)
def _font_file(bold: bool) -> str | None:
    for path in _FONT_FILES[bold] + _FONT_FILES[False]:
        if Path(path).is_file():
            return path
    return None


def load_font(size: int, bold: bool = True) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = _font_file(bold)
    if path:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


def _wrap(text: str, font, max_width: float) -> str:
    """Break lines at spaces so each fits ``max_width``; existing line breaks are kept."""
    lines = []
    for paragraph in text.splitlines() or [""]:
        line = ""
        for word in paragraph.split():
            candidate = f"{line} {word}".strip()
            if line and font.getlength(candidate) > max_width:
                lines.append(line)
                line = word
            else:
                line = candidate
        lines.append(line)
    return "\n".join(lines)


@tool("Add text")
def add_text(
    ctx: ToolContext,
    text: Annotated[str, P("The words to write. Use \\n for a line break")],
    position: Annotated[Position, P("Where to place the text")] = "bottom-center",
    size: Annotated[float, P("Letter height as % of the picture's height", 1, 50)] = 6,
    color: Annotated[str, P("Text colour name or hex code")] = "white",
    outline: Annotated[
        bool, P("Draw a contrasting outline so the text stays readable on any background")
    ] = True,
    bold: Annotated[bool, P("Use a bold typeface")] = True,
    x: Annotated[float, P("Optional exact centre of the text, % of the width. -1 = unused")] = -1,
    y: Annotated[float, P("Optional exact centre of the text, % of the height. -1 = unused")] = -1,
) -> ToolResult:
    """Write text on the picture, such as a caption, title or watermark. Choose one of the
    nine positions, or set both x and y to place the centre of the text exactly. Long text
    wraps onto several lines automatically."""
    if not text.strip():
        raise ToolError("The text is empty.")
    image = ctx.image
    result = image.copy()
    draw = ImageDraw.Draw(result)
    rgba = parse_color(color, allow_transparent=False)
    fill = rgba if result.mode == "RGBA" else rgba[:3]
    luminance = (0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]) / 255
    stroke_fill = (0, 0, 0) if luminance > 0.5 else (255, 255, 255)
    if result.mode == "RGBA":
        stroke_fill = (*stroke_fill, 255)

    margin = round(min(image.size) * 0.04)
    max_width = image.width - 2 * margin
    font_px = max(8, round(image.height * clamp(size, 1, 50) / 100))
    text = text.replace("\\n", "\n")
    while True:  # shrink until the widest line fits
        font = load_font(font_px, bold)
        stroke = max(1, round(font_px / 14)) if outline else 0
        wrapped = _wrap(text, font, max_width)
        box = draw.multiline_textbbox(
            (0, 0), wrapped, font=font, stroke_width=stroke, spacing=font_px * 0.2, align="center"
        )
        if box[2] - box[0] <= max_width or font_px <= 8:
            break
        font_px = max(8, round(font_px * 0.9))

    text_w, text_h = box[2] - box[0], box[3] - box[1]
    if x >= 0 and y >= 0:
        left = image.width * clamp(x, 0, 100) / 100 - text_w / 2
        top = image.height * clamp(y, 0, 100) / 100 - text_h / 2
        where = f"at ({x:g}%, {y:g}%)"
    else:
        vertical, _, horizontal = position.partition("-")
        horizontal = horizontal or "center"
        if position == "center":
            vertical = "center"
        left = {"left": margin, "right": image.width - margin - text_w}.get(
            horizontal, (image.width - text_w) / 2
        )
        top = {"top": margin, "bottom": image.height - margin - text_h}.get(
            vertical, (image.height - text_h) / 2
        )
        where = f"at the {position.replace('-', ' ')}"
    # Keep the whole block on the canvas, then cancel the font's own offset.
    left = clamp(left, 0, max(0, image.width - text_w)) - box[0]
    top = clamp(top, 0, max(0, image.height - text_h)) - box[1]
    draw.multiline_text(
        (left, top),
        wrapped,
        font=font,
        fill=fill,
        stroke_width=stroke,
        stroke_fill=stroke_fill,
        spacing=font_px * 0.2,
        align="center",
    )
    shown = text if len(text) <= 40 else text[:37] + "…"
    return ToolResult(f"Wrote {shown!r} {where} in {color}.", result, detail=shown)


@tool("Draw shape")
def draw_shape(
    ctx: ToolContext,
    left: Annotated[float, P("Left edge, % of the width from the left side", 0, 100)],
    top: Annotated[float, P("Top edge, % of the height from the top", 0, 100)],
    right: Annotated[float, P("Right edge, % of the width from the left side", 0, 100)],
    bottom: Annotated[float, P("Bottom edge, % of the height from the top", 0, 100)],
    shape: Annotated[Literal["rectangle", "ellipse"], P("Outline shape")] = "rectangle",
    color: Annotated[str, P("Line colour name or hex code")] = "red",
    thickness: Annotated[float, P("Line width as % of the picture's shorter side", 0.1, 10)] = 0.6,
) -> ToolResult:
    """Draw the outline of a rectangle or ellipse to highlight or mark part of the
    picture. Edges are percentages of the current picture measured from its top-left
    corner. The inside stays untouched."""
    image = ctx.image
    box = pct_box(image, left, top, right, bottom)
    rgba = parse_color(color, allow_transparent=False)
    result = image.copy()
    draw = ImageDraw.Draw(result)
    width = max(1, round(min(image.size) * clamp(thickness, 0.1, 10) / 100))
    outline = rgba if result.mode == "RGBA" else rgba[:3]
    corners = (box[0], box[1], box[2] - 1, box[3] - 1)
    if shape == "ellipse":
        draw.ellipse(corners, outline=outline, width=width)
    else:
        draw.rectangle(corners, outline=outline, width=width)
    return ToolResult(
        f"Drew a {color} {shape} from ({left:g}%, {top:g}%) to ({right:g}%, {bottom:g}%).",
        result,
        detail=f"{color} {shape}",
    )


__all__ = ["add_text", "draw_shape", "load_font", "Image"]
