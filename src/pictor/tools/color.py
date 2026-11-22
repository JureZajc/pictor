"""Tools that change how the picture looks without changing its shape."""

from __future__ import annotations

from typing import Annotated, Literal

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from pictor.tools.common import clamp, on_rgb
from pictor.tools.registry import P, ToolContext, ToolError, ToolResult, tool

FilterName = Literal[
    "grayscale",
    "sepia",
    "invert",
    "blur",
    "sharpen",
    "vintage",
    "posterize",
    "pixelate",
    "vignette",
    "sketch",
    "emboss",
]

#: Strength used when the model does not give one. Tone effects apply fully; effects
#: that can wreck a picture start gently.
DEFAULT_STRENGTH = {
    "grayscale": 100,
    "sepia": 100,
    "invert": 100,
    "blur": 30,
    "sharpen": 50,
    "vintage": 100,
    "posterize": 50,
    "pixelate": 30,
    "vignette": 50,
    "sketch": 100,
    "emboss": 100,
}

_SEPIA = (0.393, 0.769, 0.189, 0, 0.349, 0.686, 0.168, 0, 0.272, 0.534, 0.131, 0)


def blur_radius(image: Image.Image, strength: float) -> float:
    """Blur radius for a 0-100 strength, relative to the picture so it looks the same at
    any resolution."""
    return max(0.5, max(image.size) * 0.02 * clamp(strength, 0, 100) / 100)


def pixelate(image: Image.Image, strength: float) -> Image.Image:
    block = max(2, round(max(image.size) * 0.05 * clamp(strength, 1, 100) / 100))
    small = image.resize(
        (max(1, image.width // block), max(1, image.height // block)), Image.Resampling.BOX
    )
    return small.resize(image.size, Image.Resampling.NEAREST)


def _vignette(image: Image.Image, amount: float) -> Image.Image:
    height, width = image.height, image.width
    ys = np.linspace(-1, 1, height, dtype=np.float32)[:, None]
    xs = np.linspace(-1, 1, width, dtype=np.float32)[None, :]
    distance = np.clip((xs * xs + ys * ys) / 2, 0, 1)  # 0 at the centre, 1 in the corners
    falloff = 1 - amount * distance**1.5
    pixels = np.asarray(image, dtype=np.float32) * falloff[:, :, None]
    return Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))


def _sketch(image: Image.Image) -> Image.Image:
    """Pencil-drawing look: colour-dodge the greyscale picture with its blurred negative."""
    gray = ImageOps.grayscale(image)
    blurred = ImageOps.invert(gray).filter(ImageFilter.GaussianBlur(max(image.size) * 0.012))
    base = np.asarray(gray, dtype=np.float32)
    dodge = np.asarray(blurred, dtype=np.float32)
    result = np.clip(base * 255 / np.maximum(255 - dodge, 1), 0, 255).astype(np.uint8)
    return Image.fromarray(result).convert("RGB")


def _effect(name: str, image: Image.Image, strength: float) -> Image.Image:
    """Apply one named effect to an RGB picture."""
    mix = clamp(strength, 0, 100) / 100
    if name == "blur":
        return image.filter(ImageFilter.GaussianBlur(blur_radius(image, strength)))
    if name == "sharpen":
        sharpen = ImageFilter.UnsharpMask(radius=2, percent=round(mix * 300), threshold=2)
        return image.filter(sharpen)
    if name == "pixelate":
        return pixelate(image, strength)
    if name == "posterize":
        return ImageOps.posterize(image, round(clamp(8 - mix * 6, 2, 8)))
    if name == "vignette":
        return _vignette(image, mix * 0.9)
    # The rest are full looks that are blended back over the original by strength.
    if name == "grayscale":
        look = ImageOps.grayscale(image).convert("RGB")
    elif name == "sepia":
        look = image.convert("RGB", _SEPIA)
    elif name == "invert":
        look = ImageOps.invert(image)
    elif name == "sketch":
        look = _sketch(image)
    elif name == "emboss":
        look = image.filter(ImageFilter.EMBOSS)
    elif name == "vintage":
        look = ImageEnhance.Color(image).enhance(0.7)
        look = Image.blend(look, look.convert("RGB", _SEPIA), 0.35)
        look = ImageEnhance.Contrast(look).enhance(0.9)
        look = _vignette(look, 0.35)
    else:
        raise ToolError(f"Unknown filter {name!r}.")
    return look if mix >= 1 else Image.blend(image, look, mix)


@tool("Adjust")
def adjust(
    ctx: ToolContext,
    brightness: Annotated[float, P("-100 (black) to 100 (twice as bright), 0 = unchanged")] = 0,
    contrast: Annotated[float, P("-100 (flat grey) to 100 (strong), 0 = unchanged")] = 0,
    saturation: Annotated[float, P("-100 (no colour) to 100 (vivid), 0 = unchanged")] = 0,
    sharpness: Annotated[float, P("-100 (soft) to 100 (crisp), 0 = unchanged")] = 0,
    warmth: Annotated[float, P("-100 (cool, blue) to 100 (warm, orange), 0 = unchanged")] = 0,
) -> ToolResult:
    """Adjust the tone and colour of the whole picture. Every value runs from -100 to 100
    and 0 leaves that property alone, so set only the ones you want to change. Modest
    values around 15 to 30 are a visible but natural change; 60 and above is dramatic."""
    values = {
        "brightness": clamp(brightness, -100, 100),
        "contrast": clamp(contrast, -100, 100),
        "saturation": clamp(saturation, -100, 100),
        "sharpness": clamp(sharpness, -100, 100),
        "warmth": clamp(warmth, -100, 100),
    }
    changed = {name: value for name, value in values.items() if value}
    if not changed:
        raise ToolError("Every value is 0, so nothing would change. Set at least one.")

    def edit(rgb: Image.Image) -> Image.Image:
        if values["brightness"]:
            rgb = ImageEnhance.Brightness(rgb).enhance(1 + values["brightness"] / 100)
        if values["contrast"]:
            rgb = ImageEnhance.Contrast(rgb).enhance(1 + values["contrast"] / 100)
        if values["saturation"]:
            rgb = ImageEnhance.Color(rgb).enhance(1 + values["saturation"] / 100)
        if values["warmth"]:
            shift = values["warmth"] / 100 * 0.25
            red, green, blue = rgb.split()
            red = red.point(lambda v: min(255, round(v * (1 + shift))))
            blue = blue.point(lambda v: min(255, round(v * (1 - shift))))
            rgb = Image.merge("RGB", (red, green, blue))
        if values["sharpness"]:
            rgb = ImageEnhance.Sharpness(rgb).enhance(1 + values["sharpness"] / 50)
        return rgb

    summary = ", ".join(f"{name} {value:+g}" for name, value in changed.items())
    return ToolResult(f"Adjusted {summary}.", on_rgb(ctx.image, edit), detail=summary)


@tool("Filter")
def apply_filter(
    ctx: ToolContext,
    name: Annotated[FilterName, P("Which effect to apply")],
    strength: Annotated[
        float, P("0 to 100. Leave at -1 for a sensible default for that effect", -1, 100)
    ] = -1,
) -> ToolResult:
    """Apply an effect to the whole picture: grayscale (black and white), sepia, invert,
    blur, sharpen, vintage, posterize, pixelate, vignette (dark corners), sketch (pencil
    drawing) or emboss. To blur or pixelate only part of the picture use blur_region
    instead."""
    amount = DEFAULT_STRENGTH[name] if strength < 0 else clamp(strength, 0, 100)
    result = on_rgb(ctx.image, lambda rgb: _effect(name, rgb, amount))
    return ToolResult(f"Applied the {name} filter at strength {amount:g}.", result, detail=name)


@tool("Auto enhance")
def auto_enhance(ctx: ToolContext) -> ToolResult:
    """Improve a dull or flat photo in one step: stretches the tonal range so darks are
    dark and lights are light, then adds a touch of colour and sharpness. A good first
    move for vague requests like 'make it look better'."""

    def edit(rgb: Image.Image) -> Image.Image:
        rgb = ImageOps.autocontrast(rgb, cutoff=0.5, preserve_tone=True)
        rgb = ImageEnhance.Color(rgb).enhance(1.1)
        return ImageEnhance.Sharpness(rgb).enhance(1.2)

    return ToolResult(
        "Enhanced contrast, colour and sharpness automatically.", on_rgb(ctx.image, edit)
    )
