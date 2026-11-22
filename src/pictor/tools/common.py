"""Helpers shared by the tools: percent boxes, colours, aspect ratios, alpha handling."""

from __future__ import annotations

import re
from collections.abc import Callable

from PIL import Image, ImageColor

from pictor.tools.registry import ToolError

MAX_SIDE = 12000


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def size_text(image: Image.Image) -> str:
    return f"{image.width}×{image.height} px"


def pct_box(
    image: Image.Image, left: float, top: float, right: float, bottom: float
) -> tuple[int, int, int, int]:
    """Convert edges given as percentages of the picture into a pixel box.

    Out-of-range values are clamped and swapped edges are put in order, because a model
    that is roughly right should still get its edit.
    """
    left, right = sorted((clamp(left, 0, 100), clamp(right, 0, 100)))
    top, bottom = sorted((clamp(top, 0, 100), clamp(bottom, 0, 100)))
    x0 = round(image.width * left / 100)
    x1 = round(image.width * right / 100)
    y0 = round(image.height * top / 100)
    y1 = round(image.height * bottom / 100)
    if x1 - x0 < 1 or y1 - y0 < 1:
        raise ToolError(
            "That region is empty. Edges are percentages from the top-left corner, "
            "and right must be larger than left, bottom larger than top."
        )
    return x0, y0, x1, y1


def parse_color(value: str, allow_transparent: bool = True) -> tuple[int, int, int, int]:
    """Read a colour name, hex code or ``rgb()`` string into an RGBA tuple."""
    text = (value or "").strip().lower()
    if text in ("transparent", "none", "clear"):
        if not allow_transparent:
            raise ToolError("A transparent colour makes no sense here; name a real colour.")
        return (0, 0, 0, 0)
    text = text.replace("grey", "gray")
    # Models write "light blue" or "dark-red"; the names Pillow knows have no separators.
    for candidate in (text, re.sub(r"[\s_-]+", "", text)):
        try:
            return tuple(ImageColor.getcolor(candidate, "RGBA"))  # type: ignore[return-value]
        except ValueError:
            continue
    raise ToolError(
        f"Unknown colour {value!r}. Use a name like 'white' or a hex code like '#ff8800'."
    )


_NAMED_RATIOS = {"square": 1.0, "landscape": 3 / 2, "portrait": 2 / 3, "widescreen": 16 / 9}


def parse_ratio(text: str) -> float:
    """Read ``"16:9"``, ``"4x5"``, ``"3/2"``, ``"1.5"`` or ``"square"`` as width / height."""
    value = (text or "").strip().lower()
    if value in _NAMED_RATIOS:
        return _NAMED_RATIOS[value]
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*[:x×/]\s*(\d+(?:\.\d+)?)", value)
    try:
        ratio = float(match.group(1)) / float(match.group(2)) if match else float(value)
    except (ValueError, ZeroDivisionError):
        ratio = 0
    if not 0.05 <= ratio <= 20:
        raise ToolError(f"Cannot read {text!r} as an aspect ratio. Write it like '16:9' or '1:1'.")
    return ratio


def on_rgb(image: Image.Image, edit: Callable[[Image.Image], Image.Image]) -> Image.Image:
    """Apply a colour edit to the RGB channels while leaving any transparency untouched."""
    if image.mode == "RGBA":
        alpha = image.getchannel("A")
        result = edit(image.convert("RGB")).convert("RGB")
        result.putalpha(alpha)
        return result
    return edit(image.convert("RGB")).convert("RGB")


def check_size(width: float, height: float) -> tuple[int, int]:
    width, height = round(width), round(height)
    if width < 1 or height < 1:
        raise ToolError("The result would be smaller than one pixel.")
    if max(width, height) > MAX_SIDE:
        raise ToolError(f"The result would be {width}×{height} px; the limit is {MAX_SIDE} px.")
    return width, height
