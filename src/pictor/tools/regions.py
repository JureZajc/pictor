"""Tools that act on part of the picture: obscuring regions and faces."""

from __future__ import annotations

from typing import Annotated, Literal

from PIL import Image, ImageDraw, ImageFilter

from pictor.tools.color import blur_radius, pixelate
from pictor.tools.common import pct_box
from pictor.tools.faces import detect_faces as find_faces
from pictor.tools.registry import P, ToolContext, ToolResult, tool

Style = Literal["blur", "pixelate"]
Shape = Literal["rectangle", "ellipse"]


def obscure(
    image: Image.Image, box: tuple[int, int, int, int], style: str, strength: float, shape: str
) -> Image.Image:
    """Return a copy of ``image`` with the pixels inside ``box`` blurred or pixelated."""
    result = image.copy()
    patch = image.crop(box)
    if style == "pixelate":
        # Scale the block size to the region so a small face still gets coarse blocks.
        hidden = pixelate(patch, max(strength, 20) * 2.5)
    else:
        hidden = patch.filter(ImageFilter.GaussianBlur(blur_radius(patch, strength) * 4))
    if shape == "ellipse":
        mask = Image.new("L", patch.size, 0)
        ImageDraw.Draw(mask).ellipse((0, 0, patch.width - 1, patch.height - 1), fill=255)
        result.paste(hidden, box[:2], mask)
    else:
        result.paste(hidden, box[:2])
    return result


def _percent(value: int, total: int) -> float:
    return round(100 * value / total, 1)


@tool("Blur region")
def blur_region(
    ctx: ToolContext,
    left: Annotated[float, P("Left edge, % of the width from the left side", 0, 100)],
    top: Annotated[float, P("Top edge, % of the height from the top", 0, 100)],
    right: Annotated[float, P("Right edge, % of the width from the left side", 0, 100)],
    bottom: Annotated[float, P("Bottom edge, % of the height from the top", 0, 100)],
    style: Annotated[Style, P("'blur' smooths the area, 'pixelate' turns it into blocks")] = "blur",
    strength: Annotated[float, P("0 to 100, how unrecognisable the area becomes", 0, 100)] = 60,
    shape: Annotated[Shape, P("Shape of the affected area inside the box")] = "rectangle",
) -> ToolResult:
    """Blur or pixelate one rectangular part of the picture, e.g. to hide a licence plate,
    a name tag or a screen. Edges are percentages of the current picture measured from
    its top-left corner. To hide faces use anonymize_faces, which finds them for you."""
    image = ctx.image
    box = pct_box(image, left, top, right, bottom)
    result = obscure(image, box, style, strength, shape)
    verb = "Pixelated" if style == "pixelate" else "Blurred"
    return ToolResult(
        f"{verb} the region from ({left:g}%, {top:g}%) to ({right:g}%, {bottom:g}%).",
        result,
        detail=style,
    )


@tool("Detect faces")
def detect_faces(ctx: ToolContext) -> ToolResult:
    """Find the faces in the picture and report where each one is, as percentages of the
    picture measured from its top-left corner. Use the result to crop around a person or
    to place something relative to a face."""
    image = ctx.image
    faces = find_faces(image, ctx.status)
    if not faces:
        return ToolResult("No faces were found in the picture.")
    lines = [
        f"#{number}: left={_percent(face.left, image.width)}, "
        f"top={_percent(face.top, image.height)}, "
        f"right={_percent(face.right, image.width)}, "
        f"bottom={_percent(face.bottom, image.height)} "
        f"(confidence {face.score:.2f})"
        for number, face in enumerate(faces, start=1)
    ]
    return ToolResult(
        f"Found {len(faces)} face(s), largest first, as % of the picture:\n" + "\n".join(lines)
    )


@tool("Anonymize faces")
def anonymize_faces(
    ctx: ToolContext,
    style: Annotated[Style, P("'blur' smooths each face, 'pixelate' makes blocks")] = "blur",
    strength: Annotated[float, P("0 to 100, how unrecognisable the faces become", 0, 100)] = 70,
) -> ToolResult:
    """Find every face in the picture and blur or pixelate it so the people cannot be
    recognised. No coordinates are needed."""
    image = ctx.image
    faces = find_faces(image, ctx.status)
    if not faces:
        return ToolResult("No faces were found, so nothing was changed.")
    result = image
    for face in faces:
        # Grow the box so hair and chin are covered too.
        grow_x = (face.right - face.left) * 0.15
        grow_y = (face.bottom - face.top) * 0.15
        box = (
            max(0, round(face.left - grow_x)),
            max(0, round(face.top - grow_y)),
            min(image.width, round(face.right + grow_x)),
            min(image.height, round(face.bottom + grow_y)),
        )
        result = obscure(result, box, style, strength, "ellipse")
    verb = "Pixelated" if style == "pixelate" else "Blurred"
    return ToolResult(f"{verb} {len(faces)} face(s).", result, detail=f"{len(faces)} face(s)")
