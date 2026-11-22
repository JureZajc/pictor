"""Tools that change the picture's shape: crop, resize, rotate, flip, borders."""

from __future__ import annotations

import math
from typing import Annotated, Literal

from PIL import Image, ImageFilter, ImageOps

from pictor.tools.common import check_size, clamp, parse_color, parse_ratio, pct_box, size_text
from pictor.tools.registry import P, ToolContext, ToolError, ToolResult, tool

Anchor = Literal["center", "top", "bottom", "left", "right"]


@tool("Crop")
def crop(
    ctx: ToolContext,
    left: Annotated[float, P("Left edge, % of the width from the left side", 0, 100)],
    top: Annotated[float, P("Top edge, % of the height from the top", 0, 100)],
    right: Annotated[float, P("Right edge, % of the width from the left side", 0, 100)],
    bottom: Annotated[float, P("Bottom edge, % of the height from the top", 0, 100)],
) -> ToolResult:
    """Cut the picture down to a rectangle. All four edges are percentages of the current
    picture measured from its top-left corner. Example: left=10, top=0, right=90,
    bottom=100 trims 10% off the left and right sides and keeps the full height."""
    image = ctx.image
    box = pct_box(image, left, top, right, bottom)
    result = image.crop(box)
    return ToolResult(
        f"Cropped to {size_text(result)} (was {size_text(image)}).",
        result,
        detail=f"{result.width}×{result.height}",
    )


@tool("Crop to ratio")
def crop_to_aspect_ratio(
    ctx: ToolContext,
    ratio: Annotated[str, P("Target width:height, e.g. '1:1', '16:9', '4:5', '3:2'")],
    anchor: Annotated[
        Anchor, P("Which part to keep when trimming: center, top, bottom, left or right")
    ] = "center",
) -> ToolResult:
    """Crop to a given aspect ratio, keeping as much of the picture as possible. Use this
    for requests like 'make it square' or 'crop to 16:9'. The anchor decides which side
    is kept when something has to be trimmed."""
    image = ctx.image
    target = parse_ratio(ratio)
    width, height = image.size
    if width / height > target:  # too wide: trim the sides
        new_width, new_height = max(1, round(height * target)), height
    else:  # too tall: trim top and bottom
        new_width, new_height = width, max(1, round(width / target))
    if (new_width, new_height) == (width, height):
        raise ToolError(f"The picture already has a {ratio} aspect ratio.")
    x = {"left": 0, "right": width - new_width}.get(anchor, (width - new_width) // 2)
    y = {"top": 0, "bottom": height - new_height}.get(anchor, (height - new_height) // 2)
    result = image.crop((x, y, x + new_width, y + new_height))
    return ToolResult(
        f"Cropped to {ratio} ({size_text(result)}, was {size_text(image)}).", result, detail=ratio
    )


@tool("Resize")
def resize(
    ctx: ToolContext,
    width: Annotated[float, P("Target width in pixels, or 0 to derive it", 0)] = 0,
    height: Annotated[float, P("Target height in pixels, or 0 to derive it", 0)] = 0,
    percent: Annotated[float, P("Scale factor in percent, e.g. 50 halves each side", 0)] = 0,
    mode: Annotated[
        Literal["fit", "fill", "stretch"],
        P("Only used when both width and height are given"),
    ] = "fit",
) -> ToolResult:
    """Change the pixel dimensions. Give either percent (50 halves each side, 200
    doubles), or a target width and/or height in pixels. With only one of width or height
    the other follows the aspect ratio. With both, mode 'fit' keeps the whole picture
    inside that box without distortion, 'fill' covers the box exactly and crops what
    overflows, and 'stretch' forces the exact size and may distort."""
    image = ctx.image
    src_w, src_h = image.size
    resample = Image.Resampling.LANCZOS
    if percent > 0:
        size = check_size(src_w * percent / 100, src_h * percent / 100)
        result = image.resize(size, resample)
    elif width > 0 and height > 0:
        size = check_size(width, height)
        if mode == "stretch":
            result = image.resize(size, resample)
        elif mode == "fill":
            result = ImageOps.fit(image, size, resample)
        else:
            scale = min(size[0] / src_w, size[1] / src_h)
            result = image.resize(check_size(src_w * scale, src_h * scale), resample)
    elif width > 0:
        result = image.resize(check_size(width, src_h * width / src_w), resample)
    elif height > 0:
        result = image.resize(check_size(src_w * height / src_h, height), resample)
    else:
        raise ToolError("Give percent, or a width and/or height in pixels.")
    return ToolResult(
        f"Resized to {size_text(result)} (was {size_text(image)}).",
        result,
        detail=f"{result.width}×{result.height}",
    )


def _inscribed_size(width: int, height: int, radians: float) -> tuple[float, float]:
    """Largest axis-aligned rectangle that fits inside a rotated width × height rectangle."""
    sin_a, cos_a = abs(math.sin(radians)), abs(math.cos(radians))
    long_side, short_side = max(width, height), min(width, height)
    if short_side <= 2 * sin_a * cos_a * long_side or abs(sin_a - cos_a) < 1e-10:
        half = 0.5 * short_side
        if width >= height:
            return half / sin_a, half / cos_a
        return half / cos_a, half / sin_a
    cos_2a = cos_a * cos_a - sin_a * sin_a
    return (width * cos_a - height * sin_a) / cos_2a, (height * cos_a - width * sin_a) / cos_2a


@tool("Rotate")
def rotate(
    ctx: ToolContext,
    degrees: Annotated[float, P("Clockwise angle; negative turns counter-clockwise", -360, 360)],
    crop_corners: Annotated[
        bool, P("Trim the empty corners left by an angle that is not a multiple of 90")
    ] = False,
) -> ToolResult:
    """Rotate the picture clockwise by a number of degrees. 90, 180 and 270 are exact
    quarter turns. Any other angle leaves empty corners, filled white (transparent if the
    picture already has transparency). Set crop_corners=true to trim them away instead,
    which is what you want when straightening a slightly tilted photo."""
    image = ctx.image
    angle = degrees % 360
    if angle == 0:
        raise ToolError("Rotating by 0 degrees changes nothing.")
    quarter_turns = {
        90: Image.Transpose.ROTATE_270,  # Pillow's names count counter-clockwise
        180: Image.Transpose.ROTATE_180,
        270: Image.Transpose.ROTATE_90,
    }
    if angle in quarter_turns:
        result = image.transpose(quarter_turns[angle])
    else:
        fill = (0, 0, 0, 0) if image.mode == "RGBA" else (255, 255, 255)
        result = image.rotate(-angle, Image.Resampling.BICUBIC, expand=True, fillcolor=fill)
        if crop_corners:
            inner_w, inner_h = _inscribed_size(image.width, image.height, math.radians(angle))
            inner_w = clamp(math.floor(inner_w), 1, result.width)
            inner_h = clamp(math.floor(inner_h), 1, result.height)
            x = (result.width - inner_w) // 2
            y = (result.height - inner_h) // 2
            result = result.crop((int(x), int(y), int(x + inner_w), int(y + inner_h)))
    return ToolResult(
        f"Rotated {degrees:g}° clockwise. The picture is now {size_text(result)}.",
        result,
        detail=f"{degrees:g}°",
    )


@tool("Flip")
def flip(
    ctx: ToolContext,
    direction: Annotated[
        Literal["horizontal", "vertical"],
        P("'horizontal' mirrors left and right, 'vertical' turns it upside down"),
    ],
) -> ToolResult:
    """Mirror the picture. 'horizontal' swaps left and right, like a mirror image;
    'vertical' swaps top and bottom."""
    method = (
        Image.Transpose.FLIP_LEFT_RIGHT
        if direction == "horizontal"
        else Image.Transpose.FLIP_TOP_BOTTOM
    )
    return ToolResult(f"Flipped {direction}ly.", ctx.image.transpose(method), detail=direction)


@tool("Border")
def add_border(
    ctx: ToolContext,
    size: Annotated[float, P("Border thickness as % of the picture's shorter side", 0.1, 50)] = 5,
    color: Annotated[str, P("Colour name or hex code, or 'transparent'")] = "white",
) -> ToolResult:
    """Add a border or frame of even thickness around the whole picture, making the canvas
    larger. Nothing is cropped."""
    image = ctx.image
    rgba = parse_color(color)
    thickness = max(1, round(min(image.size) * clamp(size, 0.1, 50) / 100))
    check_size(image.width + 2 * thickness, image.height + 2 * thickness)
    if rgba[3] < 255 and image.mode != "RGBA":
        image = image.convert("RGBA")
    fill = rgba if image.mode == "RGBA" else rgba[:3]
    result = ImageOps.expand(image, border=thickness, fill=fill)
    return ToolResult(
        f"Added a {thickness}px {color} border. The picture is now {size_text(result)}.",
        result,
        detail=color,
    )


@tool("Pad to ratio")
def pad_to_aspect_ratio(
    ctx: ToolContext,
    ratio: Annotated[str, P("Target width:height, e.g. '1:1', '16:9', '4:5'")],
    color: Annotated[
        str, P("Fill colour name or hex code, 'transparent', or 'blur' for a blurred copy")
    ] = "white",
) -> ToolResult:
    """Reach an aspect ratio WITHOUT cropping, by extending the canvas with bars on two
    sides. Use this when the whole picture must stay visible, e.g. fitting a wide photo
    into a square post. color='blur' fills the bars with a blurred copy of the picture."""
    image = ctx.image
    target = parse_ratio(ratio)
    width, height = image.size
    if width / height < target:
        new_width, new_height = round(height * target), height
    else:
        new_width, new_height = width, round(width / target)
    new_width, new_height = check_size(new_width, new_height)
    if (new_width, new_height) == (width, height):
        raise ToolError(f"The picture already has a {ratio} aspect ratio.")
    offset = ((new_width - width) // 2, (new_height - height) // 2)
    if color.strip().lower() == "blur":
        backdrop = ImageOps.fit(image.convert("RGB"), (new_width, new_height))
        canvas = backdrop.filter(ImageFilter.GaussianBlur(max(new_width, new_height) * 0.03))
        mask = image.getchannel("A") if image.mode == "RGBA" else None
        canvas.paste(image.convert("RGB"), offset, mask)
    else:
        rgba = parse_color(color)
        if rgba[3] < 255 or image.mode == "RGBA":
            canvas = Image.new("RGBA", (new_width, new_height), rgba)
            canvas.paste(image.convert("RGBA"), offset)
        else:
            canvas = Image.new("RGB", (new_width, new_height), rgba[:3])
            canvas.paste(image, offset)
    return ToolResult(
        f"Extended the canvas to {ratio} ({size_text(canvas)}) with {color} fill.",
        canvas,
        detail=ratio,
    )
