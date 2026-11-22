import io

import pytest
from PIL import Image

from pictor.config import Config, normalize_host
from pictor.session import ImageLoadError, normalize
from tests.conftest import encode, make_photo


def test_opening_a_picture_creates_the_original_version(store, photo):
    session = store.create(encode(photo, "JPEG"), "trip/../holiday.jpg")
    state = session.state()
    assert state["filename"] == "holiday.jpg"  # no path components survive
    assert (state["width"], state["height"]) == (640, 480)
    assert state["versions"] == [
        {"id": 0, "label": "Original", "detail": "", "width": 640, "height": 480}
    ]
    assert state["cursor"] == 0 and not state["can_undo"] and not state["can_redo"]
    assert state["output"]["format"] == "jpeg"
    assert store.get(session.id) is session


def test_undo_redo_and_jumping_between_versions(session):
    session.push(session.image().resize((320, 240)), "Resize", "320×240")
    session.push(session.image().rotate(90, expand=True), "Rotate", "90°")
    assert session.state()["cursor"] == 2 and session.image().size == (240, 320)

    assert session.undo() and session.image().size == (320, 240)
    assert session.undo() and session.image().size == (640, 480)
    assert not session.undo()
    assert session.redo() and session.cursor == 1
    assert session.goto(2) and session.image().size == (240, 320)
    assert not session.goto(99)
    assert session.redo() is False


def test_a_new_edit_after_undo_discards_the_undone_versions(session):
    session.push(session.image().resize((320, 240)), "Resize")
    doomed = session.push(session.image().resize((160, 120)), "Resize")
    session.undo()
    session.undo()
    fresh = session.push(session.image().convert("L").convert("RGB"), "Filter")

    assert [v.label for v in session.versions] == ["Original", "Filter"]
    assert not doomed.path.exists()
    assert session.find(doomed.id) is None
    # Ids are never reused, so cached previews in the browser can never go stale.
    assert fresh.id == 3


def test_versions_survive_a_reload_from_disk(session):
    edited = session.image().resize((100, 75))
    version = session.push(edited, "Resize")
    session._images.clear()
    assert session.image(version).size == (100, 75)


def test_exif_orientation_is_applied_on_open(store):
    sideways = make_photo(400, 200)
    exif = Image.Exif()
    exif[0x0112] = 6  # stored rotated; viewers turn it 90° clockwise
    session = store.create(encode(sideways, "JPEG", exif=exif.tobytes()), "phone.jpg")
    assert session.image().size == (200, 400)


def test_unusual_modes_become_rgb_or_rgba():
    assert normalize(Image.new("L", (4, 4), 128)).mode == "RGB"
    assert normalize(Image.new("LA", (4, 4), (128, 10))).mode == "RGBA"
    assert normalize(Image.new("CMYK", (4, 4))).mode == "RGB"
    palette = Image.new("P", (4, 4))
    palette.info["transparency"] = 0
    assert normalize(palette).mode == "RGBA"
    deep = Image.fromarray(
        __import__("numpy").linspace(0, 65535, 16).reshape(4, 4).astype("uint16")
    )
    converted = normalize(deep)
    assert converted.mode == "RGB"
    assert converted.getpixel((0, 0))[0] == 0 and converted.getpixel((3, 3))[0] == 255


def test_files_that_are_not_pictures_are_rejected(store):
    with pytest.raises(ImageLoadError):
        store.create(b"this is not a picture", "notes.txt")
    with pytest.raises(ImageLoadError):
        store.create(b"", "empty.png")


def test_export_uses_the_source_format_by_default(session, store, photo):
    data, filename, media_type = session.export()
    assert (filename, media_type) == ("holiday-pictor.png", "image/png")
    assert Image.open(io.BytesIO(data)).size == (640, 480)

    jpeg_session = store.create(encode(photo, "JPEG"), "holiday.jpeg")
    data, filename, media_type = jpeg_session.export()
    assert (filename, media_type) == ("holiday-pictor.jpg", "image/jpeg")
    assert Image.open(io.BytesIO(data)).format == "JPEG"


def test_transparency_switches_a_jpeg_to_png_unless_jpeg_was_asked_for(store, photo):
    session = store.create(encode(photo, "JPEG"), "portrait.jpg")
    cut_out = photo.convert("RGBA")
    cut_out.putalpha(Image.new("L", photo.size, 0))
    session.push(cut_out, "Remove background")

    assert session.effective_format == "png"
    assert session.state()["transparent"] is True
    data, filename, _ = session.export()
    assert filename.endswith(".png") and Image.open(io.BytesIO(data)).mode == "RGBA"

    session.set_output("jpeg", 80)
    data, filename, _ = session.export()
    flattened = Image.open(io.BytesIO(data))
    assert filename.endswith(".jpg") and flattened.mode == "RGB"
    assert min(flattened.getpixel((5, 5))) > 240  # transparent areas become white


def test_export_to_webp_and_bad_formats(session):
    session.set_output("webp", 500)
    assert session.output_quality == 100
    data, filename, media_type = session.export()
    assert media_type == "image/webp" and filename.endswith(".webp")
    assert Image.open(io.BytesIO(data)).format == "WEBP"
    with pytest.raises(ValueError):
        session.set_output("bmp")


def test_previews_are_scaled_down_webp(session):
    data = session.preview(session.current, 128)
    preview = Image.open(io.BytesIO(data))
    assert preview.format == "WEBP" and max(preview.size) == 128
    assert session.preview(session.current, 128) is data  # cached


def test_discarding_a_session_removes_its_files(store, session):
    folder = session.dir
    assert folder.exists()
    store.discard(session.id)
    assert not folder.exists() and store.get(session.id) is None


def test_config_round_trip_and_defaults():
    config = Config.load()
    assert config.model is None and config.think == "auto"
    config.model = "gemma4:12b"
    config.think = "off"
    config.save()
    loaded = Config.load()
    assert (loaded.model, loaded.think) == ("gemma4:12b", "off")

    Config.path().write_text('{"model": "x", "think": "sideways", "unknown": 1}')
    assert Config.load().think == "auto"
    Config.path().write_text("not json")
    assert Config.load().model is None


def test_ollama_host_forms_are_normalized(monkeypatch):
    assert normalize_host(None) == "http://127.0.0.1:11434"
    assert normalize_host("0.0.0.0:11434") == "http://127.0.0.1:11434"
    assert normalize_host("0.0.0.0") == "http://127.0.0.1:11434"
    assert normalize_host("192.168.1.5:8000") == "http://192.168.1.5:8000"
    assert normalize_host("https://ollama.example.com/") == "https://ollama.example.com"
    assert normalize_host("http://localhost:11434") == "http://localhost:11434"
    monkeypatch.setenv("OLLAMA_HOST", "10.0.0.2:9999")
    assert Config().resolved_ollama_host == "http://10.0.0.2:9999"
    assert Config(ollama_host="box:1234").resolved_ollama_host == "http://box:1234"
