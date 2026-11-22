import pytest
from PIL import Image, ImageChops

from pictor.tools import ToolContext, ToolError, registry
from pictor.tools import background as background_tools
from pictor.tools import regions as region_tools
from pictor.tools.common import parse_color, parse_ratio
from pictor.tools.faces import Face

RED = (200, 60, 40)  # top-left quarter of the test photo
BLUE = (40, 90, 160)  # its background
YELLOW = (240, 220, 60)  # ellipse in the bottom-right quarter
IN_RED = (100, 100)
IN_BLUE = (600, 200)  # clear of the ellipse and the white diagonal


def run(session, name, /, **arguments):
    """Call a tool the way the agent does; an edit becomes the next version."""
    tool = registry.get(name)
    result = tool.run(ToolContext(session=session), arguments)
    if result.image is not None:
        session.push(result.image, tool.label, result.detail)
    return result


def differs(a: Image.Image, b: Image.Image) -> bool:
    return ImageChops.difference(a.convert("RGB"), b.convert("RGB")).getbbox() is not None


def close(pixel, expected, tolerance=6) -> bool:
    return all(abs(a - b) <= tolerance for a, b in zip(pixel[:3], expected, strict=True))


# ---------------------------------------------------------------- registry


def test_every_tool_describes_itself_for_the_model():
    schemas = registry.schemas()
    assert len(schemas) >= 20
    for schema in schemas:
        function = schema["function"]
        assert schema["type"] == "function"
        assert len(function["description"]) > 40, function["name"]
        parameters = function["parameters"]
        assert parameters["type"] == "object"
        assert "title" not in str(parameters), "pydantic titles are noise for the model"
        for name, spec in parameters["properties"].items():
            assert spec.get("description"), f"{function['name']}.{name} has no description"
            assert "anyOf" not in spec, "optional types confuse small models"


def test_view_image_is_only_offered_to_models_that_can_see():
    assert "view_image" in registry.names(vision=True)
    assert "view_image" not in registry.names(vision=False)
    assert len(registry.schemas(vision=False)) == len(registry.schemas()) - 1


def test_numbers_written_with_units_are_understood(session):
    result = run(session, "crop", left="10%", top="0", right="90 %", bottom="100px")
    assert session.image().size == (512, 480)
    assert "512×480" in result.message


def test_bad_arguments_come_back_as_a_fixable_error(session):
    with pytest.raises(ToolError, match="Invalid arguments for flip"):
        run(session, "flip", direction="diagonal")
    with pytest.raises(ToolError, match="left"):
        run(session, "crop", top=0, right=50, bottom=50)
    with pytest.raises(ToolError):
        run(session, "rotate", degrees="a quarter turn")


def test_unknown_arguments_are_ignored(session):
    run(session, "flip", direction="horizontal", reason="the user asked")
    assert session.cursor == 1


# ---------------------------------------------------------------- geometry


def test_crop_keeps_the_requested_region(session):
    run(session, "crop", left=0, top=0, right=50, bottom=50)
    image = session.image()
    assert image.size == (320, 240)
    assert close(image.getpixel((160, 120)), RED)


def test_crop_tolerates_swapped_and_out_of_range_edges(session):
    run(session, "crop", left=150, top=-20, right=50, bottom=100)
    assert session.image().size == (320, 480)
    with pytest.raises(ToolError, match="empty"):
        run(session, "crop", left=40, top=0, right=40, bottom=100)


def test_crop_to_aspect_ratio(session):
    run(session, "crop_to_aspect_ratio", ratio="1:1")
    assert session.image().size == (480, 480)
    with pytest.raises(ToolError, match="already"):
        run(session, "crop_to_aspect_ratio", ratio="square")
    run(session, "crop_to_aspect_ratio", ratio="16:9", anchor="top")
    assert session.image().size == (480, 270)


def test_crop_anchor_decides_what_is_kept(session):
    run(session, "crop_to_aspect_ratio", ratio="1:2", anchor="left")
    image = session.image()
    assert image.size == (240, 480)
    assert close(image.getpixel((120, 60)), RED)


def test_resize_variants(session):
    run(session, "resize", percent=50)
    assert session.image().size == (320, 240)
    run(session, "resize", width=160)
    assert session.image().size == (160, 120)
    run(session, "resize", width=100, height=100)  # fit: no distortion
    assert session.image().size == (100, 75)
    run(session, "resize", width=60, height=60, mode="fill")
    assert session.image().size == (60, 60)
    run(session, "resize", width=90, height=30, mode="stretch")
    assert session.image().size == (90, 30)
    with pytest.raises(ToolError, match="percent"):
        run(session, "resize")
    with pytest.raises(ToolError, match="limit"):
        run(session, "resize", percent=100000)


def test_rotate_quarter_turns_are_clockwise_and_exact(session):
    run(session, "rotate", degrees=90)
    image = session.image()
    assert image.size == (480, 640)
    # A clockwise turn moves the red quarter from top-left to top-right, and the yellow
    # ellipse from bottom-right to bottom-left.
    assert close(image.getpixel((400, 100)), RED)
    assert close(image.getpixel((80, 500)), YELLOW)
    run(session, "rotate", degrees=-90)
    assert session.image().size == (640, 480)
    with pytest.raises(ToolError):
        run(session, "rotate", degrees=360)


def test_rotate_by_an_odd_angle_grows_or_crops(session):
    run(session, "rotate", degrees=10)
    grown = session.image()
    assert grown.width > 640 and grown.height > 480
    assert close(grown.getpixel((2, 2)), (255, 255, 255))  # empty corner filled white
    session.undo()
    run(session, "rotate", degrees=10, crop_corners=True)
    cropped = session.image()
    assert cropped.width < 640 and cropped.height < 480
    corners = [(1, 1), (cropped.width - 2, 1), (1, cropped.height - 2)]
    assert not any(close(cropped.getpixel(c), (255, 255, 255)) for c in corners)


def test_flip(session):
    run(session, "flip", direction="horizontal")
    assert close(session.image().getpixel((480, 120)), RED)
    run(session, "flip", direction="vertical")
    assert close(session.image().getpixel((480, 360)), RED)


def test_border_and_padding(session):
    run(session, "add_border", size=10, color="#00ff00")
    image = session.image()
    assert image.size == (736, 576)
    assert close(image.getpixel((5, 5)), (0, 255, 0))
    session.undo()

    run(session, "pad_to_aspect_ratio", ratio="1:1", color="black")
    image = session.image()
    assert image.size == (640, 640)
    assert close(image.getpixel((320, 10)), (0, 0, 0))
    assert close(image.getpixel((100, 180)), RED)  # the picture itself is untouched
    with pytest.raises(ToolError, match="already"):
        run(session, "pad_to_aspect_ratio", ratio="1:1")
    session.undo()

    run(session, "pad_to_aspect_ratio", ratio="16:9", color="blur")
    assert session.image().size == (853, 480)
    session.undo()

    run(session, "add_border", size=5, color="transparent")
    assert session.image().mode == "RGBA" and session.image().getpixel((2, 2))[3] == 0


# ---------------------------------------------------------------- colour


def test_adjust_changes_only_what_was_asked(session):
    before = session.image().getpixel(IN_BLUE)
    run(session, "adjust", brightness=40)
    after = session.image().getpixel(IN_BLUE)
    assert all(a > b for a, b in zip(after, before, strict=True))
    with pytest.raises(ToolError, match="nothing"):
        run(session, "adjust")
    result = run(session, "adjust", saturation=-100, contrast=500)
    assert "contrast +100" in result.message  # clamped
    r, g, b = session.image().getpixel(IN_BLUE)
    assert abs(r - g) <= 2 and abs(g - b) <= 2


def test_warmth_shifts_red_against_blue(session):
    before = session.image().getpixel(IN_BLUE)
    run(session, "adjust", warmth=80)
    after = session.image().getpixel(IN_BLUE)
    assert after[0] > before[0] and after[2] < before[2] and after[1] == before[1]


@pytest.mark.parametrize(
    "name",
    [
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
    ],
)
def test_every_filter_changes_the_picture_and_keeps_its_size(session, name):
    original = session.image()
    result = run(session, "apply_filter", name=name)
    assert session.image().size == original.size
    assert differs(session.image(), original)
    assert name in result.message


def test_filter_strength_blends_with_the_original(session):
    run(session, "apply_filter", name="grayscale", strength=0)
    assert not differs(session.image(), session.image(session.versions[0]))
    run(session, "apply_filter", name="grayscale")
    r, g, b = session.image().getpixel((100, 100))
    assert r == g == b


def test_colour_edits_keep_transparency(session):
    cut_out = session.image().convert("RGBA")
    cut_out.putalpha(Image.new("L", cut_out.size, 77))
    session.push(cut_out, "Cut out")
    for name, arguments in [
        ("apply_filter", {"name": "sepia"}),
        ("adjust", {"brightness": 20}),
        ("auto_enhance", {}),
    ]:
        run(session, name, **arguments)
        assert session.image().mode == "RGBA"
        assert session.image().getpixel((10, 10))[3] == 77


# ---------------------------------------------------------------- regions and faces


def test_blur_region_only_touches_the_region(session):
    original = session.image()
    run(session, "blur_region", left=40, top=40, right=60, bottom=60, style="pixelate")
    changed = ImageChops.difference(session.image(), original).getbbox()
    assert changed is not None
    left, top, right, bottom = changed
    assert left >= 256 and top >= 192 and right <= 384 and bottom <= 288


def test_face_tools_use_the_detector(session, monkeypatch):
    # A box over the corner of the red quarter, so there is detail to obscure.
    face = Face(left=256, top=180, right=384, bottom=300, score=0.97)
    monkeypatch.setattr(region_tools, "find_faces", lambda image, status=None: [face])

    report = run(session, "detect_faces")
    assert "Found 1 face" in report.message
    assert "left=40.0, top=37.5, right=60.0, bottom=62.5" in report.message
    assert session.cursor == 0  # looking does not create a version

    original = session.image()
    result = run(session, "anonymize_faces", style="pixelate")
    assert "1 face" in result.message
    left, top, right, bottom = ImageChops.difference(session.image(), original).getbbox()
    # Only the face, grown by 15% on each side, may change.
    assert left >= 236 and top >= 161 and right <= 404 and bottom <= 319


def test_no_faces_is_reported_without_changing_anything(session, monkeypatch):
    monkeypatch.setattr(region_tools, "find_faces", lambda image, status=None: [])
    assert "No faces" in run(session, "detect_faces").message
    assert "nothing was changed" in run(session, "anonymize_faces").message
    assert session.cursor == 0


# ---------------------------------------------------------------- background


@pytest.fixture
def left_half_subject(monkeypatch):
    """Pretend the segmentation model found the subject in the left half."""

    def fake_mask(ctx):
        mask = Image.new("L", ctx.image.size, 0)
        mask.paste(255, (0, 0, ctx.image.width // 2, ctx.image.height))
        return mask

    monkeypatch.setattr(background_tools, "subject_mask", fake_mask)


def test_remove_background_makes_it_transparent(session, left_half_subject):
    run(session, "remove_background")
    image = session.image()
    assert image.mode == "RGBA"
    assert image.getpixel((100, 100))[3] == 255 and image.getpixel((500, 100))[3] == 0
    assert session.state()["transparent"] is True
    with pytest.raises(ToolError, match="already transparent"):
        run(session, "remove_background")


def test_change_background_effects_leave_the_subject_alone(session, left_half_subject):
    original = session.image()
    for effect in ("blur", "grayscale", "darken"):
        run(session, "change_background", effect=effect)
        image = session.image()
        assert image.getpixel((100, 100)) == original.getpixel((100, 100)), effect
        assert differs(image.crop((320, 0, 640, 480)), original.crop((320, 0, 640, 480))), effect
        session.undo()

    run(session, "change_background", effect="color", color="#ff00ff")
    image = session.image()
    assert close(image.getpixel((500, 100)), (255, 0, 255)) and image.mode == "RGB"
    assert close(image.getpixel((100, 100)), RED)


def test_a_removed_background_can_be_filled_but_not_restyled(session, left_half_subject):
    run(session, "remove_background")
    with pytest.raises(ToolError, match="already removed"):
        run(session, "change_background", effect="blur")
    run(session, "change_background", effect="color", color="white")
    assert session.image().mode == "RGB"
    assert close(session.image().getpixel((500, 100)), (255, 255, 255))


# ---------------------------------------------------------------- drawing


def test_add_text_draws_where_it_was_asked(session):
    original = session.image()
    result = run(session, "add_text", text="Živjo, svet", position="bottom-center", size=8)
    left, top, right, bottom = ImageChops.difference(session.image(), original).getbbox()
    assert top > 480 * 0.7 and bottom <= 480
    assert left > 100 and right < 540
    assert "Živjo" in result.message

    session.undo()
    run(session, "add_text", text="TOP", position="top-left", size=8)
    left, top, right, bottom = ImageChops.difference(session.image(), original).getbbox()
    assert bottom < 480 * 0.3 and right < 320

    session.undo()
    run(session, "add_text", text="X", x=75, y=25, size=10, outline=False, color="#00ff00")
    left, top, right, bottom = ImageChops.difference(session.image(), original).getbbox()
    assert abs((left + right) / 2 - 480) < 25 and abs((top + bottom) / 2 - 120) < 25


def test_long_text_wraps_inside_the_picture(session):
    original = session.image()
    words = "a caption that is far too long to fit on one line of a small picture"
    run(session, "add_text", text=words, size=10, position="center")
    left, top, right, bottom = ImageChops.difference(session.image(), original).getbbox()
    assert left >= 0 and right <= 640
    assert bottom - top > 480 * 0.2  # several lines tall
    with pytest.raises(ToolError, match="empty"):
        run(session, "add_text", text="   ")


def test_draw_shape_outlines_without_filling(session):
    original = session.image()
    run(session, "draw_shape", left=60, top=10, right=90, bottom=40, color="lime", thickness=1)
    image = session.image()
    assert close(image.getpixel((384 + 2, 120)), (0, 255, 0))  # on the left edge of the box
    assert image.getpixel((480, 120)) == original.getpixel((480, 120))  # inside is untouched


def test_colours_and_ratios_are_parsed_leniently():
    assert parse_color("White") == (255, 255, 255, 255)
    assert parse_color("#f80") == (255, 136, 0, 255)
    assert parse_color("light grey") == parse_color("lightgray")
    assert parse_color("Dark-Red") == (139, 0, 0, 255)
    assert parse_color("rgb(10, 20, 30)") == (10, 20, 30, 255)
    assert parse_color("transparent") == (0, 0, 0, 0)
    with pytest.raises(ToolError):
        parse_color("transparent", allow_transparent=False)
    with pytest.raises(ToolError, match="Unknown colour"):
        parse_color("blurple-ish")
    assert parse_ratio("16:9") == pytest.approx(16 / 9)
    assert parse_ratio("4x5") == pytest.approx(0.8)
    assert parse_ratio("1.5") == 1.5
    assert parse_ratio("square") == 1
    for bad in ("", "wide", "0:1", "1:0", "1000:1"):
        with pytest.raises(ToolError):
            parse_ratio(bad)


# ---------------------------------------------------------------- session tools


def test_info_reports_the_facts(session):
    message = run(session, "get_image_info").message
    assert "640×480 px" in message and "Aspect ratio: 4:3" in message
    assert "Transparency: no" in message and "holiday.png (PNG)" in message
    assert session.cursor == 0


def test_output_format_does_not_touch_the_pixels(session):
    result = run(session, "set_output_format", format="webp", quality=70)
    assert "WEBP at quality 70" in result.message
    assert session.cursor == 0 and session.effective_format == "webp"
    assert session.export()[1].endswith(".webp")


def test_undo_and_reset_tools(session):
    with pytest.raises(ToolError, match="nothing to undo"):
        run(session, "undo_last_edit")
    with pytest.raises(ToolError, match="already"):
        run(session, "reset_to_original")
    run(session, "resize", percent=50)
    run(session, "flip", direction="horizontal")
    assert "Undid 'Flip'" in run(session, "undo_last_edit").message
    assert session.cursor == 1
    run(session, "apply_filter", name="invert")
    assert "640×480" in run(session, "reset_to_original").message
    assert session.cursor == 0 and session.image().size == (640, 480)
