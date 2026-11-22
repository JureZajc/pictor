"""The edit pipeline: which part of the picture the image model sees, and what is kept."""

import threading

import numpy as np
import pytest
from PIL import Image, ImageDraw

from pictor.generation import GenerationError
from pictor.tools import ToolContext, ToolError, registry
from pictor.tools import generative as g
from tests.conftest import MAGENTA, FakeGenerator, make_photo


def changed_pixels(before: Image.Image, after: Image.Image) -> np.ndarray:
    a = np.asarray(before.convert("RGBA"), dtype=np.int16)
    b = np.asarray(after.convert("RGBA"), dtype=np.int16)
    return np.abs(a - b).max(axis=2) > 0


def bbox_of(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def big_photo() -> Image.Image:
    return make_photo(1600, 1200)


def is_magenta(pixel) -> bool:
    """The disc the fake generator paints, give or take what resizing does to a value."""
    return all(abs(a - b) <= 3 for a, b in zip(pixel[:3], MAGENTA, strict=True))


def paint(image, box, fake, **options):
    return g.paint(image, box, lambda window, attempt: f"add it ({attempt})", fake, **options)


# ---------------------------------------------------------------- the window


def test_the_window_is_centred_on_the_region_and_larger_than_it():
    window = g.plan_window((4000, 3000), (1800, 1300, 2400, 1700))
    left, top, right, bottom = window.area
    assert (right - left, bottom - top) == (960, 640)  # 1.6 times the region
    assert ((left + right) // 2, (top + bottom) // 2) == (2100, 1500)
    assert window.size == (960, 640)  # already a size the model works well at
    assert g.describe_place(window) == "in the centre"
    assert g.describe_size(window) == "about 60% of the picture's width"


@pytest.mark.parametrize(
    ("box", "place"),
    [
        ((0, 0, 500, 400), "in the upper left"),
        ((3500, 0, 4000, 400), "in the upper right"),
        ((0, 2600, 500, 3000), "in the lower left"),
        ((3500, 2600, 4000, 3000), "in the lower right"),
        ((1750, 0, 2250, 400), "at the top, in the middle"),
        ((0, 1300, 500, 1700), "on the left side"),
    ],
)
def test_a_region_at_an_edge_keeps_the_window_inside_the_picture(box, place):
    window = g.plan_window((4000, 3000), box)
    left, top, right, bottom = window.area
    assert 0 <= left < right <= 4000 and 0 <= top < bottom <= 3000
    assert left <= box[0] and top <= box[1] and right >= box[2] and bottom >= box[3]
    # The window was pushed inwards, so the prompt has to say where the region ended up.
    assert g.describe_place(window) == place


def test_a_tiny_region_still_gets_a_window_worth_looking_at():
    window = g.plan_window((4000, 3000), (2000, 1500, 2020, 1520))
    left, top, right, bottom = window.area
    assert (right - left, bottom - top) == (g.MIN_WINDOW, g.MIN_WINDOW)
    assert window.size == (768, 768)  # enlarged: the model draws poorly at 384 px
    assert g.describe_size(window) == "about 5% of the picture's width"


def test_a_region_covering_everything_uses_the_whole_picture():
    window = g.plan_window((3000, 2000), (0, 0, 3000, 2000))
    assert window.area == (0, 0, 3000, 2000)
    assert window.size == (1024, 688)
    assert window.inner == (0, 0, 1024, 688)
    assert g.describe_size(window) == "about 100% of the picture's width"


@pytest.mark.parametrize("picture", [(1001, 751), (333, 517), (97, 61), (5000, 301)])
def test_odd_sizes_are_rounded_to_what_the_model_accepts(picture):
    width, height = picture
    box = (width // 3, height // 3, 2 * width // 3, 2 * height // 3)
    for max_side in (512, 1024):
        window = g.plan_window(picture, box, max_side)
        assert window.size[0] % 16 == 0 and window.size[1] % 16 == 0
        assert min(window.size) >= 16 and max(window.size) <= max_side
        x0, y0, x1, y1 = window.inner
        assert 0 <= x0 < x1 <= window.size[0] and 0 <= y0 < y1 <= window.size[1]


def test_a_long_thin_region_gets_a_window_the_model_can_draw_in():
    window = g.plan_window((4000, 3000), (500, 1400, 3500, 1500))
    left, top, right, bottom = window.area
    assert (right - left) / (bottom - top) <= g.MAX_ASPECT + 0.01


def test_the_longest_side_setting_limits_what_the_model_is_given():
    assert max(g.plan_window((4000, 3000), (1000, 1000, 3000, 2500), 512).size) == 512


# ---------------------------------------------------------------- compositing


def test_only_the_painted_thing_changes_and_the_rest_is_identical():
    photo = big_photo()
    box = (900, 300, 1300, 600)  # on the plain blue part
    fake = FakeGenerator()
    done = paint(photo, box, fake)

    assert done.image.size == photo.size and done.image.mode == "RGB"
    assert done.attempts == 1 and len(fake.calls) == 1
    # The middle of the region now holds what the generator painted.
    assert is_magenta(done.image.getpixel((1100, 450)))
    changed = changed_pixels(photo, done.image)
    left, top, right, bottom = bbox_of(changed)
    window = g.plan_window(photo.size, box)
    assert window.area[0] <= left and right <= window.area[2]
    assert window.area[1] <= top and bottom <= window.area[3]
    # Everything outside the reported area is bit for bit what it was.
    outside = np.ones(changed.shape, dtype=bool)
    x0, y0, x1, y1 = done.changed
    outside[y0:y1, x0:x1] = False
    assert not changed[outside].any()
    assert 0.005 < changed.mean() < 0.04  # the disc and a soft rim, not the window


def test_the_model_only_sees_the_window_at_its_working_size():
    photo = big_photo()
    fake = FakeGenerator()
    paint(photo, (900, 300, 1300, 600), fake, seed=7)
    (call,) = fake.calls
    assert call["size"] == (768, 576) and call["seed"] == 7
    assert call["prompt"] == "add it (0)"


def test_the_overall_drift_of_a_real_model_is_not_pasted_back():
    photo = big_photo()
    fake = FakeGenerator(drift=6)  # every pixel a little brighter, as real models do
    done = paint(photo, (900, 300, 1300, 600), fake)
    changed = changed_pixels(photo, done.image)
    assert is_magenta(done.image.getpixel((1100, 450)))
    assert changed.mean() < 0.06
    assert done.image.getpixel((700, 200)) == photo.getpixel((700, 200))


def test_something_the_model_adds_elsewhere_is_left_out():
    photo = big_photo()
    box = (900, 300, 1300, 600)
    window = g.plan_window(photo.size, box)

    class TwoDiscs(FakeGenerator):
        def edit(self, image, prompt, **options):
            result = super().edit(image, prompt, **options)
            ImageDraw.Draw(result).ellipse((10, 10, 90, 90), fill=(0, 250, 0))  # uninvited
            return result

    done = paint(photo, box, TwoDiscs(radius=0.15))
    corner = (window.area[0] + 25, window.area[1] + 25)
    assert done.image.getpixel(corner) == photo.getpixel(corner)
    assert is_magenta(done.image.getpixel((1100, 450)))


def test_transparency_survives_and_new_content_is_visible():
    photo = big_photo().convert("RGBA")
    alpha = Image.new("L", photo.size, 255)
    ImageDraw.Draw(alpha).rectangle((800, 0, 1600, 1200), fill=0)  # right half see-through
    photo.putalpha(alpha)
    fake = FakeGenerator()
    done = paint(photo, (900, 300, 1300, 600), fake)

    assert done.image.mode == "RGBA" and done.image.size == photo.size
    # The model was shown the see-through part as plain grey, not as garbage colours.
    shown = fake.calls[0]["image"]
    assert shown.mode == "RGB" and shown.getpixel((shown.width - 5, 5)) == g.BACKDROP
    # The painted thing can be seen...
    assert is_magenta(done.image.getpixel((1100, 450)))
    assert done.image.getpixel((1100, 450))[3] == 255
    # ...while the rest keeps its transparency and its solid part.
    assert done.image.getpixel((1500, 100))[3] == 0
    assert done.image.getpixel((1500, 1100))[3] == 0
    assert done.image.getpixel((100, 100)) == photo.getpixel((100, 100))
    before, after = np.asarray(photo)[..., 3], np.asarray(done.image)[..., 3]
    assert (after >= before).all()


def test_a_thing_that_spills_over_the_window_fades_out_instead_of_leaving_a_seam():
    photo = big_photo()
    box = (1300, 800, 1600, 1200)  # bottom-right corner of the picture
    window = g.plan_window(photo.size, box)
    left, top = window.area[:2]
    assert (left, top) == (1120, 560)

    fake = FakeGenerator(at=(0.25, 0.2), radius=0.35)  # runs over the window's top and left
    done = paint(photo, box, fake)
    changed = changed_pixels(photo, done.image)
    assert changed[top + 60 : top + 120, left + 60 : left + 120].all()  # it did spill
    assert not changed[top, :].any() and not changed[:, left].any()  # no seam there
    assert not changed[:top, :].any() and not changed[:, :left].any()


def test_the_pictures_own_edge_needs_no_fade():
    photo = big_photo()
    fake = FakeGenerator(at=(0.95, 0.95), radius=0.3)
    done = paint(photo, (1300, 800, 1600, 1200), fake)
    assert is_magenta(done.image.getpixel((1598, 1198)))


# ---------------------------------------------------------------- checking the result


def test_a_thing_painted_in_the_wrong_place_is_tried_again():
    photo = big_photo()
    notes = []
    fake = FakeGenerator(at=[(0.08, 0.08), (0.5, 0.5)], radius=0.06)
    done = paint(photo, (900, 300, 1300, 600), fake, seed=100, status=notes.append)

    assert done.attempts == 2
    assert [call["seed"] for call in fake.calls] == [100, 101]  # another seed each time
    assert [call["prompt"] for call in fake.calls] == ["add it (0)", "add it (1)"]
    assert any("Trying again (2 of 3)" in note for note in notes)
    assert is_magenta(done.image.getpixel((1100, 450)))


def test_giving_up_after_three_tries_is_a_clear_error_and_changes_nothing():
    photo = big_photo()
    fake = FakeGenerator(at=(0.08, 0.08), radius=0.06)  # never in the region
    with pytest.raises(ToolError) as failure:
        paint(photo, (900, 300, 1300, 600), fake)
    assert len(fake.calls) == 3
    message = str(failure.value)
    assert "changed nothing inside the region" in message and "3 times" in message
    assert "larger region" in message  # tells the planning model what to try


def test_a_model_that_paints_nothing_at_all_is_reported():
    with pytest.raises(ToolError, match="changed nothing inside the region"):
        paint(big_photo(), (900, 300, 1300, 600), FakeGenerator(at=None, drift=5))


def test_a_model_that_repaints_the_whole_window_is_refused():
    class AnotherScene(FakeGenerator):
        def edit(self, image, prompt, **options):
            self.calls.append({"prompt": prompt})
            return Image.linear_gradient("L").resize(image.size).convert("RGB")

    fake = AnotherScene()
    with pytest.raises(ToolError, match="repainted the whole area"):
        paint(big_photo(), (900, 300, 1300, 600), fake)
    assert len(fake.calls) == 3


def test_a_thing_that_fills_most_of_the_window_is_still_told_from_the_background():
    photo = big_photo()
    done = paint(photo, (900, 300, 1300, 600), FakeGenerator(radius=0.42, drift=5))
    assert is_magenta(done.image.getpixel((1100, 450)))
    assert done.image.getpixel((700, 200)) == photo.getpixel((700, 200))


def test_find_change_ignores_texture_noise():
    rng = np.random.default_rng(3)
    base = rng.integers(60, 200, (384, 512, 3), dtype=np.uint8)
    noisy = np.clip(base.astype(np.int16) + rng.integers(-25, 26, base.shape), 0, 255)
    change = g.find_change(
        Image.fromarray(base), Image.fromarray(noisy.astype(np.uint8)), (150, 100, 350, 280)
    )
    assert change.nothing and change.changed < 0.01


# ---------------------------------------------------------------- the tools


def run(session, fake, name, /, **arguments):
    ctx = ToolContext(session=session, generator=fake, settings={"image_max_side": 1024})
    tool = registry.get(name)
    result = tool.run(ctx, arguments)
    if result.image is not None:
        session.push(result.image, tool.label, result.detail)
    return result


def test_add_object_paints_into_the_region_and_writes_a_prompt_for_the_image_model(session):
    fake = FakeGenerator()
    result = run(
        session,
        fake,
        "add_object",
        description="  a small brown dog,\n sitting. ",
        left=55,
        top=10,
        right=95,
        bottom=45,
    )
    assert session.cursor == 1 and session.image().size == (640, 480)
    assert is_magenta(session.image().getpixel((480, 132)))
    assert session.current.label == "Add object"
    assert session.current.detail == "a small brown dog, sitting"
    assert "Painted a small brown dog, sitting into the region (55%, 10%) to (95%, 45%)" in (
        result.message
    )
    assert result.show_image  # the planning model should look at what it got

    prompt = fake.calls[0]["prompt"]
    assert prompt.startswith("Add a small brown dog, sitting to this picture, ")
    assert "Add only this one thing." in prompt and "keep everything else" in prompt
    assert "of the picture's" in prompt  # how large it should be


def test_a_description_written_as_a_sentence_still_reads_well(session):
    fake = FakeGenerator()
    result = run(
        session, fake, "add_object", description="A red kite.", left=55, top=10, right=95, bottom=45
    )
    assert result.message.startswith("Painted a red kite into the region")
    assert fake.calls[0]["prompt"].startswith("Add a red kite to this picture, ")


def test_the_second_try_insists_on_the_place():
    window = g.plan_window((4000, 3000), (0, 0, 500, 400))
    first, second = g.add_prompt("a red kite", window, 0), g.add_prompt("a red kite", window, 1)
    assert "nowhere else" not in first
    assert second.endswith("It must be in the upper left, nowhere else.")


def test_remove_object_uses_the_same_machinery_with_a_removal_prompt(session):
    fake = FakeGenerator(color=(40, 90, 160))  # paints the background colour over the ball
    spot = (410, 290)
    assert session.image().getpixel(spot) == (240, 220, 60)
    result = run(
        session,
        fake,
        "remove_object",
        description="the yellow ball",
        left=50,
        top=50,
        right=95,
        bottom=95,
    )
    assert all(
        abs(a - b) <= 3 for a, b in zip(session.image().getpixel(spot), (40, 90, 160), strict=True)
    )
    assert session.current.label == "Remove object" and "Removed the yellow ball" in result.message
    prompt = fake.calls[0]["prompt"]
    assert prompt.startswith("Remove the yellow ball ") and "what is behind it" in prompt


def test_the_tools_ask_for_memory_and_pass_on_progress_and_cancellation(session):
    fake = FakeGenerator()
    asked, notes, stop = [], [], threading.Event()

    def make_room(need_gb: float) -> bool:
        asked.append(need_gb)
        return True

    ctx = ToolContext(
        session=session, generator=fake, status=notes.append, cancelled=stop, make_room=make_room
    )
    registry.get("add_object").run(
        ctx, {"description": "a kite", "left": 55, "top": 10, "right": 95, "bottom": 45}
    )
    assert asked == [fake.memory_gb]
    assert notes == ["Loading the image model…", "Painting: step 1 of 4…"]


def test_generator_failures_become_messages_the_planning_model_can_pass_on(session):
    arguments = {"description": "a kite", "left": 55, "top": 10, "right": 95, "bottom": 45}
    tool = registry.get("add_object")

    broken = FakeGenerator(error=GenerationError("The image generator ran out of memory."))
    with pytest.raises(ToolError, match="ran out of memory"):
        tool.run(ToolContext(session=session, generator=broken), arguments)

    stopping = threading.Event()
    stopping.set()
    waiting = FakeGenerator(wait_for_cancel=True)
    with pytest.raises(ToolError, match="Stopped"):
        tool.run(ToolContext(session=session, generator=waiting, cancelled=stopping), arguments)
    assert waiting.was_cancelled

    with pytest.raises(ToolError, match="not set up"):
        tool.run(ToolContext(session=session), arguments)
    with pytest.raises(ToolError, match="Say what to paint"):
        tool.run(
            ToolContext(session=session, generator=FakeGenerator()),
            {**arguments, "description": " "},
        )
    with pytest.raises(ToolError, match="region is empty"):
        tool.run(
            ToolContext(session=session, generator=FakeGenerator()), {**arguments, "right": 55}
        )
    assert session.cursor == 0
