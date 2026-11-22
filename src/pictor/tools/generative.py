"""Tools that paint new content into the picture with a second, image-generating model.

An image model redraws everything it is shown, at about one megapixel, and shifts every
pixel a little while doing so. Handing it a whole photo and keeping its answer would lose
resolution and quietly change things nobody asked to change. So :func:`paint` shows it
only a window around the region, finds what it really changed there, and puts back just
that part. Every other pixel of the picture stays exactly as it was.
"""

from __future__ import annotations

import random
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated

import cv2
import numpy as np
from PIL import Image

from pictor.generation import SIZE_STEP, GenerationCancelled, GenerationError, ImageGenerator
from pictor.tools.common import clamp, pct_box
from pictor.tools.registry import P, ToolContext, ToolError, ToolResult, tool

Box = tuple[int, int, int, int]

#: The window shown to the image model is this many times the region's size. Tried on
#: real pictures: at 1.6 the model draws the thing inside the region and at its size; at
#: 2.5 and wider it starts to put it wherever it likes.
GROW = 1.6
#: ...but at least this many pixels a side, so a small region is not enlarged into mush.
MIN_WINDOW = 384
#: Windows more elongated than this are widened; the model draws badly in thin strips.
MAX_ASPECT = 2.0
#: A small window is enlarged to this long side for the model, which draws poorly small.
MIN_MODEL_SIDE = 768
#: How far around the region, as a share of its size, a change still counts as "there".
NEAR = 0.15
#: The thing itself must cover at least this share of the region to count as painted.
MIN_COVER = 0.01
#: A result that changes more of the window than this repainted the scene, not one thing.
MAX_CHANGED = 0.7
#: How different a blurred pixel must be, out of 255, to count as changed: never less
#: than the first, however calm the pair is, and never more than the second.
MIN_DIFFERENCE = 12.0
MAX_DIFFERENCE = 40.0
#: The most overall colour shift, out of 255, that is put down to the model's drift.
MAX_DRIFT = 16.0
#: How many times the image model is asked before giving up on a region.
ATTEMPTS = 3
#: What transparent areas look like to the image model.
BACKDROP = (128, 128, 128)


@dataclass(frozen=True)
class Window:
    """The part of the picture the image model works on."""

    #: The region asked for, in picture pixels.
    box: Box
    #: The window around it that is shown to the model, in picture pixels.
    area: Box
    #: The size the window is scaled to for the model. Both are multiples of 16.
    size: tuple[int, int]

    @property
    def scale(self) -> tuple[float, float]:
        """Model pixels per picture pixel, across and down."""
        return (
            self.size[0] / (self.area[2] - self.area[0]),
            self.size[1] / (self.area[3] - self.area[1]),
        )

    @property
    def inner(self) -> Box:
        """The region in the model's pixels."""
        sx, sy = self.scale
        left, top = self.area[:2]
        x0, y0, x1, y1 = self.box
        return (
            round((x0 - left) * sx),
            round((y0 - top) * sy),
            max(round((x0 - left) * sx) + 1, round((x1 - left) * sx)),
            max(round((y0 - top) * sy) + 1, round((y1 - top) * sy)),
        )


@dataclass(frozen=True)
class Change:
    """What the image model changed, judged against the region it was asked to use."""

    #: 0-255 mask in the model's pixels: the changed parts that belong to the region.
    mask: np.ndarray
    #: Share of the window that changed at all.
    changed: float
    #: Share of the window covered by the changes that belong to the region.
    kept: float

    @property
    def nothing(self) -> bool:
        return not self.mask.any()

    @property
    def too_much(self) -> bool:
        return self.kept > MAX_CHANGED


def _step(value: float) -> int:
    return max(SIZE_STEP, round(value / SIZE_STEP) * SIZE_STEP)


def plan_window(picture: tuple[int, int], box: Box, max_side: int = 1024) -> Window:
    """Choose the window around ``box`` and the size the model sees it at."""
    width, height = picture
    x0, y0, x1, y1 = box
    box_w, box_h = x1 - x0, y1 - y0

    # Tight enough that the region dictates where the thing goes and how large it is,
    # with enough around it for the model to match light and ground.
    win_w = min(width, max(GROW * box_w, MIN_WINDOW))
    win_h = min(height, max(GROW * box_h, MIN_WINDOW))
    if win_w > MAX_ASPECT * win_h:
        win_h = min(height, win_w / MAX_ASPECT)
    elif win_h > MAX_ASPECT * win_w:
        win_w = min(width, win_h / MAX_ASPECT)
    win_w, win_h = max(1, round(win_w)), max(1, round(win_h))

    # Centred on the region, then pushed back inside the picture.
    left = int(clamp(round((x0 + x1) / 2 - win_w / 2), 0, width - win_w))
    top = int(clamp(round((y0 + y1) / 2 - win_h / 2), 0, height - win_h))

    long_side = max(win_w, win_h)
    target = min(max_side, max(long_side, min(MIN_MODEL_SIDE, max_side)))
    ratio = target / long_side
    return Window(
        box=box,
        area=(left, top, left + win_w, top + win_h),
        size=(_step(win_w * ratio), _step(win_h * ratio)),
    )


def describe_place(window: Window) -> str:
    """Where the region sits in the window, in the words a prompt needs."""
    left, top, right, bottom = window.area
    x0, y0, x1, y1 = window.box
    across = ((x0 + x1) / 2 - left) / (right - left)
    down = ((y0 + y1) / 2 - top) / (bottom - top)
    column = 0 if across < 0.38 else 2 if across > 0.62 else 1
    row = 0 if down < 0.38 else 2 if down > 0.62 else 1
    return (
        ("in the upper left", "at the top, in the middle", "in the upper right"),
        ("on the left side", "in the centre", "on the right side"),
        ("in the lower left", "at the bottom, in the middle", "in the lower right"),
    )[row][column]


def describe_size(window: Window) -> str:
    """How large the region is in the window, in words."""
    left, top, right, bottom = window.area
    x0, y0, x1, y1 = window.box
    share = max((x1 - x0) / (right - left), (y1 - y0) / (bottom - top))
    percent = int(clamp(round(share * 20) * 5, 5, 100))
    longer = "width" if (x1 - x0) / (right - left) >= (y1 - y0) / (bottom - top) else "height"
    return f"about {percent}% of the picture's {longer}"


def find_change(before: Image.Image, after: Image.Image, inner: Box) -> Change:
    """Find what the model changed between two pictures of equal size.

    The model shifts every pixel a little and redraws fine texture such as grass, so the
    pictures are compared blurred, the overall drift is taken out, and the threshold
    adapts to how noisy this particular pair is.
    """
    a = np.asarray(before.convert("RGB"), dtype=np.float32)
    b = np.asarray(after.convert("RGB"), dtype=np.float32)
    height, width = a.shape[:2]
    x0, y0, x1, y1 = inner
    grow_x, grow_y = NEAR * (x1 - x0), NEAR * (y1 - y0)
    near = (
        max(0, round(x0 - grow_x)),
        max(0, round(y0 - grow_y)),
        min(width, round(x1 + grow_x)),
        min(height, round(y1 + grow_y)),
    )
    sigma = max(1.5, 0.004 * max(width, height))
    diff = cv2.GaussianBlur(b, (0, 0), sigma) - cv2.GaussianBlur(a, (0, 0), sigma)

    # Drift and noise are judged where nothing was asked for, outside the region, so a
    # thing that fills the region cannot pass for the background. Both are capped: a
    # large "drift" is a repainted scene, and must stay visible as change.
    threshold = MIN_DIFFERENCE
    calm = np.ones((height, width), dtype=bool)
    calm[near[1] : near[3], near[0] : near[2]] = False
    if calm.mean() >= 0.1:
        drift = np.median(diff[calm], axis=0)
        diff -= np.clip(drift, -MAX_DRIFT, MAX_DRIFT)
    strength = np.abs(diff).max(axis=2)
    if calm.mean() >= 0.1:
        middle = float(np.median(strength[calm]))
        spread = 1.4826 * float(np.median(np.abs(strength[calm] - middle)))
        threshold = float(clamp(middle + 6 * spread, MIN_DIFFERENCE, MAX_DIFFERENCE))
    changed = (strength > threshold).astype(np.uint8)

    # Drop specks, then look at each connected patch of change.
    speck = max(3, round(0.006 * max(width, height)) | 1)
    changed = cv2.morphologyEx(changed, cv2.MORPH_OPEN, np.ones((speck, speck), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(changed, connectivity=8)

    # The patch of change that overlaps the region most is the thing that was asked for,
    # shadow and all, even where it spills over the region's edge. Other patches count
    # only when they lie mostly inside the region. Whatever else the model altered, such
    # as a second cat it added on its own, is left out and the picture keeps its pixels.
    areas = [int(stats[label, cv2.CC_STAT_AREA]) for label in range(count)]
    overlap = [0] * count
    beside = [0] * count
    for label in range(1, count):
        patch = labels == label
        overlap[label] = int(patch[y0:y1, x0:x1].sum())
        beside[label] = int(patch[near[1] : near[3], near[0] : near[2]].sum())
    keep = np.zeros_like(changed)
    main = max(range(1, count), key=lambda label: overlap[label], default=0)
    if main and overlap[main] >= MIN_COVER * (x1 - x0) * (y1 - y0):
        for label in range(1, count):
            if label == main or beside[label] > 0.5 * areas[label]:
                keep |= (labels == label).astype(np.uint8)
    pixels = float(width * height)
    return Change(
        mask=keep * 255,
        changed=sum(areas[1:]) / pixels,
        kept=float(keep.sum()) / pixels,
    )


def soft_mask(change: np.ndarray, window: Window, picture: tuple[int, int]) -> Image.Image:
    """Turn the changed area into a paste mask: solid over the change, fading outside it.

    The mask is exactly zero everywhere else, and it fades out towards any window edge
    that is not also the picture's edge, so nothing can leave a visible seam there.
    """
    height, width = change.shape
    reach = max(4, round(0.012 * max(width, height)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4 * reach + 1, 4 * reach + 1))
    mask = cv2.dilate(change, kernel)
    mask = cv2.GaussianBlur(mask, (0, 0), reach * 0.6).astype(np.float32)

    left, top, right, bottom = window.area
    # Zero for the first two pixels, so resizing the mask cannot smear change onto the
    # window's very edge, then a ramp up to full strength.
    rise = np.clip((np.arange(2 * reach + 2, dtype=np.float32) - 2) / (2 * reach), 0.0, 1.0)

    def faded(length: int, at_start: bool, at_end: bool) -> np.ndarray:
        weights = np.ones(length, dtype=np.float32)
        edge = rise[:length]
        if at_start:
            weights[: len(edge)] = np.minimum(weights[: len(edge)], edge)
        if at_end:
            weights[-len(edge) :] = np.minimum(weights[-len(edge) :], edge[::-1])
        return weights

    across = faded(width, left > 0, right < picture[0])
    down = faded(height, top > 0, bottom < picture[1])
    mask *= down[:, None] * across[None, :]
    return Image.fromarray(np.clip(mask, 0, 255).astype(np.uint8), "L")


@dataclass(frozen=True)
class Painted:
    image: Image.Image
    #: How many times the image model was asked.
    attempts: int
    #: The area that was changed, in picture pixels.
    changed: Box


def paint(
    image: Image.Image,
    box: Box,
    prompt_for: Callable[[Window, int], str],
    generator: ImageGenerator,
    *,
    max_side: int = 1024,
    seed: int | None = None,
    status: Callable[[str], None] = lambda _text: None,
    cancelled: threading.Event | None = None,
    attempts: int = ATTEMPTS,
) -> Painted:
    """Have the image model change the region ``box`` of ``image`` and blend that in.

    ``prompt_for(window, attempt)`` writes the instruction for the model. Raises
    :class:`ToolError` when no attempt changed the region and only the region.
    """
    cancelled = cancelled or threading.Event()
    window = plan_window(image.size, box, max_side)
    crop = image.crop(window.area)
    alpha = crop.getchannel("A") if crop.mode == "RGBA" else None
    if alpha is not None:
        flat = Image.new("RGB", crop.size, BACKDROP)
        flat.paste(crop, mask=alpha)
    else:
        flat = crop.convert("RGB")
    shown = flat.resize(window.size, Image.Resampling.LANCZOS)

    first_seed = random.randrange(1, 2**31 - 1) if seed is None else seed
    problem = ""
    for attempt in range(attempts):
        if attempt:
            status(f"{problem} Trying again ({attempt + 1} of {attempts})…")
        redrawn = generator.edit(
            shown,
            prompt_for(window, attempt),
            seed=first_seed + attempt,
            status=status,
            cancelled=cancelled,
        )
        if redrawn.size != shown.size:
            redrawn = redrawn.resize(shown.size, Image.Resampling.LANCZOS)
        change = find_change(shown, redrawn, window.inner)
        if change.nothing:
            problem = "The image model changed nothing inside the region."
            continue
        if change.too_much:
            problem = "The image model repainted the whole area instead of one thing."
            continue
        return _blend(image, crop, alpha, redrawn, change, window, attempt + 1)
    raise ToolError(
        f"{problem} That happened {attempts} times, so the picture was left unchanged. "
        "Try once more with a larger region, or a simpler description."
    )


def _blend(
    image: Image.Image,
    crop: Image.Image,
    alpha: Image.Image | None,
    redrawn: Image.Image,
    change: Change,
    window: Window,
    attempts: int,
) -> Painted:
    mask = soft_mask(change.mask, window, image.size).resize(crop.size, Image.Resampling.BILINEAR)
    patch = crop.convert("RGB")
    patch.paste(redrawn.resize(crop.size, Image.Resampling.LANCZOS), mask=mask)
    if alpha is not None:
        # New content must be visible where the picture was see-through, but only the
        # content itself: the model painted its surroundings on a grey backdrop.
        solid = Image.fromarray(change.mask, "L").resize(crop.size, Image.Resampling.BILINEAR)
        patch.putalpha(Image.fromarray(np.maximum(np.asarray(alpha), np.asarray(solid)), "L"))
    result = image.copy()
    result.paste(patch, window.area[:2])

    touched = mask.getbbox() or (0, 0, crop.width, crop.height)
    left, top = window.area[:2]
    return Painted(
        image=result,
        attempts=attempts,
        changed=(left + touched[0], top + touched[1], left + touched[2], top + touched[3]),
    )


# --------------------------------------------------------------------------- tools


def _generate(ctx: ToolContext, box: Box, prompt_for: Callable[[Window, int], str]) -> Painted:
    generator = ctx.generator
    if generator is None:
        raise ToolError("The image generator is not set up, so nothing can be painted.")
    ctx.make_room(generator.memory_gb)
    try:
        return paint(
            ctx.image,
            box,
            prompt_for,
            generator,
            max_side=int(ctx.settings.get("image_max_side") or 1024),
            status=ctx.status,
            cancelled=ctx.cancelled,
        )
    except GenerationCancelled:
        raise ToolError("Stopped.") from None
    except GenerationError as exc:
        raise ToolError(str(exc)) from exc


def _tidy(description: str) -> str:
    text = " ".join((description or "").split()).strip(" .")
    if len(text) < 2:
        raise ToolError("Say what to paint, for example 'a small brown dog sitting'.")
    # Models like to write "A small dog."; it goes into the middle of a sentence.
    first, _, rest = text.partition(" ")
    if first in ("A", "An", "The"):
        text = f"{first.lower()} {rest}"
    return text[:400]


def _percent_region(left: float, top: float, right: float, bottom: float) -> str:
    return f"({left:g}%, {top:g}%) to ({right:g}%, {bottom:g}%)"


def add_prompt(description: str, window: Window, attempt: int = 0) -> str:
    """The instruction given to the image model for adding one thing."""
    # Worded after trying variants on real pictures. "Only this one thing" matters: without
    # it the model drew a horse with two heads and a second cat under the bench.
    place = describe_place(window)
    prompt = (
        f"Add {description} to this picture, {place}. "
        f"It should take up {describe_size(window)}. "
        "Add only this one thing. "
        "Match the light, shadows, perspective and style of the picture, "
        "and keep everything else exactly as it is."
    )
    if attempt:
        prompt += f" It must be {place}, nowhere else."
    return prompt


def remove_prompt(description: str, window: Window, attempt: int = 0) -> str:
    """The instruction given to the image model for taking one thing out."""
    return (
        f"Remove {description} {describe_place(window)} of the picture and fill the place "
        "with what is behind it, continuing the surroundings naturally. "
        "Keep everything else exactly as it is."
    )


@tool("Add object", requires={"generation"})
def add_object(
    ctx: ToolContext,
    description: Annotated[
        str,
        P(
            "What to paint, in English: the thing, its colour, size and pose, "
            "e.g. 'a small brown dog sitting, facing the camera'"
        ),
    ],
    left: Annotated[float, P("Left edge of the region, % of the width", 0, 100)],
    top: Annotated[float, P("Top edge of the region, % of the height", 0, 100)],
    right: Annotated[float, P("Right edge of the region, % of the width", 0, 100)],
    bottom: Annotated[float, P("Bottom edge of the region, % of the height", 0, 100)],
) -> ToolResult:
    """Paint something new into the picture inside a region: an animal, a person, an
    object, a plant. A second AI model draws it so that light, shadow and scale match
    the scene, and the rest of the picture is left untouched. Write the description in
    English and say what the thing is, its colour, its size and its pose. Choose the
    region where the thing could really be (on the ground, on a table, in the sky) and
    make it about as large as the thing should appear. This takes half a minute or more, so
    call it once for each thing to add."""
    thing = _tidy(description)
    box = pct_box(ctx.image, left, top, right, bottom)
    done = _generate(ctx, box, lambda window, attempt: add_prompt(thing, window, attempt))
    return ToolResult(
        f"Painted {thing} into the region {_percent_region(left, top, right, bottom)}. "
        "Look at the picture to check the result.",
        done.image,
        detail=thing[:48],
        show_image=True,
    )


@tool("Remove object", requires={"generation"})
def remove_object(
    ctx: ToolContext,
    description: Annotated[
        str, P("What to take out, in English, e.g. 'the red car' or 'the man on the left'")
    ],
    left: Annotated[float, P("Left edge of the region, % of the width", 0, 100)],
    top: Annotated[float, P("Top edge of the region, % of the height", 0, 100)],
    right: Annotated[float, P("Right edge of the region, % of the width", 0, 100)],
    bottom: Annotated[float, P("Bottom edge of the region, % of the height", 0, 100)],
) -> ToolResult:
    """Take something out of the picture and fill the gap with what was plausibly behind
    it. A second AI model repaints the region and the rest of the picture is left
    untouched. Describe the thing in English and give a region that covers all of it,
    including its shadow. This takes half a minute or more."""
    thing = _tidy(description)
    box = pct_box(ctx.image, left, top, right, bottom)
    done = _generate(ctx, box, lambda window, attempt: remove_prompt(thing, window, attempt))
    return ToolResult(
        f"Removed {thing} from the region {_percent_region(left, top, right, bottom)}. "
        "Look at the picture to check the result.",
        done.image,
        detail=thing[:48],
        show_image=True,
    )
