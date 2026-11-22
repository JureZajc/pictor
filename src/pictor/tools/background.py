"""Tools that separate the subject from the background, using rembg.

rembg downloads its segmentation model (about 180 MB for the default one) into
``~/.rembg`` the first time it is used.
"""

from __future__ import annotations

import threading
from typing import Annotated, Any, Literal

from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageStat

from pictor.session import has_transparency
from pictor.tools.color import blur_radius
from pictor.tools.common import clamp, parse_color
from pictor.tools.registry import P, ToolContext, ToolError, ToolResult, tool

DEFAULT_MODEL = "isnet-general-use"

_sessions: dict[str, Any] = {}
_lock = threading.Lock()


def _downloaded(name: str) -> bool:
    """Whether rembg already has this model on disk. Unknown counts as not downloaded."""
    try:
        from rembg.sessions import sessions

        return bool(sessions[name].resolve_existing(f"{name}.onnx"))
    except Exception:  # noqa: BLE001 - rembg internals; only used to word a status line
        return False


def _segmenter(name: str, ctx: ToolContext) -> Any:
    with _lock:
        if name not in _sessions:
            if _downloaded(name):
                ctx.status("Loading the background model…")
            else:
                ctx.status("Downloading the background model (about 180 MB, first use only)…")
            from rembg import new_session

            try:
                _sessions[name] = new_session(name)
            except Exception as exc:  # noqa: BLE001 - network, disk and model errors alike
                raise ToolError(
                    f"The background model {name!r} could not be loaded. It is downloaded "
                    f"once and needs an internet connection the first time. ({exc})"
                ) from exc
        return _sessions[name]


def subject_mask(ctx: ToolContext) -> Image.Image:
    """Greyscale mask of the current picture: white where the subject is.

    Cached per version, so removing the background and then restyling it segments once.
    """
    session = ctx.session
    scratch = session.scratch.setdefault(session.current.id, {})
    mask = scratch.get("subject_mask")
    if mask is None:
        from rembg import remove

        model = ctx.settings.get("background_model") or DEFAULT_MODEL
        mask = remove(
            ctx.image.convert("RGB"),
            session=_segmenter(model, ctx),
            only_mask=True,
            post_process_mask=True,
        ).convert("L")
        if mask.size != ctx.image.size:
            mask = mask.resize(ctx.image.size, Image.Resampling.BILINEAR)
        scratch["subject_mask"] = mask
    if ImageStat.Stat(mask).mean[0] < 2:
        raise ToolError("No clear subject could be separated from the background.")
    return mask


@tool("Remove background")
def remove_background(ctx: ToolContext) -> ToolResult:
    """Cut out the main subject and make everything else transparent. Works best when
    there is one clear subject such as a person, animal or product. The result has
    transparency, so it is saved as PNG unless another format is chosen."""
    image = ctx.image
    if has_transparency(image):
        raise ToolError("The background is already transparent.")
    result = image.convert("RGBA")
    result.putalpha(subject_mask(ctx))
    return ToolResult("Removed the background; it is now transparent.", result)


@tool("Change background")
def change_background(
    ctx: ToolContext,
    effect: Annotated[
        Literal["color", "blur", "grayscale", "darken"],
        P(
            "'color' replaces it with a solid colour, 'blur' gives a portrait look, "
            "'grayscale' keeps only the subject in colour, 'darken' dims it"
        ),
    ],
    color: Annotated[str, P("Colour name or hex code, used with effect='color'")] = "white",
    strength: Annotated[float, P("0 to 100, used with blur and darken", 0, 100)] = 60,
) -> ToolResult:
    """Restyle the background while leaving the main subject untouched: replace it with a
    solid colour, blur it for a shallow depth-of-field portrait look, turn it black and
    white so only the subject keeps its colour, or darken it. The subject is found
    automatically."""
    image = ctx.image
    cut_out = has_transparency(image)
    if effect == "color":
        rgba = parse_color(color)
        if rgba[3] == 0:
            return remove_background(ctx)
        mask = image.getchannel("A") if cut_out else subject_mask(ctx)
        result = Image.new("RGB", image.size, rgba[:3])
        result.paste(image.convert("RGB"), mask=mask)
        return ToolResult(f"Replaced the background with {color}.", result, detail=color)

    if cut_out:
        raise ToolError(
            "The background was already removed, so there is nothing to restyle. "
            "Use effect='color' to fill it, or undo the removal first."
        )
    rgb = image.convert("RGB")
    amount = clamp(strength, 0, 100)
    if effect == "blur":
        backdrop = rgb.filter(ImageFilter.GaussianBlur(blur_radius(rgb, amount) * 1.5))
        message = "Blurred the background."
    elif effect == "grayscale":
        backdrop = ImageOps.grayscale(rgb).convert("RGB")
        message = "Turned the background black and white; the subject keeps its colour."
    else:
        backdrop = ImageEnhance.Brightness(rgb).enhance(1 - 0.85 * amount / 100)
        message = "Darkened the background."
    result = Image.composite(rgb, backdrop, subject_mask(ctx))
    return ToolResult(message, result, detail=effect)
