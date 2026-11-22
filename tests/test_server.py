import io
import json

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from pictor.catalog import InstalledModel
from pictor.config import Config
from pictor.generation import GenerationError
from pictor.server import create_app
from tests.conftest import (
    ALL_CAPABILITIES,
    FakeGenerator,
    FakeOllama,
    call,
    encode,
    make_photo,
    say,
)

MODEL = "gemma4:12b"
LOCAL = "http://127.0.0.1:8765"


def make_client(
    fake: FakeOllama,
    config: Config | None = None,
    generator: FakeGenerator | None = None,
    **options,
) -> TestClient:
    # Never the real image generator: what is installed on this machine must not matter.
    generator = generator or FakeGenerator(state="not_installed")
    app = create_app(
        config or Config(model=MODEL),
        client=fake,
        start_ollama=False,
        painter=lambda _config: generator,
        **options,
    )
    return TestClient(app, base_url=LOCAL)


@pytest.fixture
def fake() -> FakeOllama:
    return FakeOllama(installed=[InstalledModel(MODEL, 7.7, ALL_CAPABILITIES, "12B")])


@pytest.fixture
def client(fake):
    with make_client(fake) as test_client:
        yield test_client


def upload(client: TestClient, image: Image.Image | None = None, name: str = "cat.png") -> dict:
    data = encode(image or make_photo(), "PNG")
    response = client.post("/api/session", files={"file": (name, data, "image/png")})
    assert response.status_code == 200, response.text
    return response.json()["state"]


def stream(client: TestClient, path: str, payload: dict) -> list[dict]:
    response = client.post(path, json=payload)
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("application/x-ndjson")
    return [json.loads(line) for line in response.text.splitlines() if line]


# ---------------------------------------------------------------- interface and status


def test_the_interface_is_served(client):
    page = client.get("/")
    assert page.status_code == 200 and "<title>Pictor</title>" in page.text
    assert page.headers["cache-control"] == "no-store"
    for asset in ("app.js", "style.css", "favicon.svg"):
        response = client.get(f"/static/{asset}")
        assert response.status_code == 200
        # Browsers must re-check, or an updated app keeps running its old script.
        assert response.headers["cache-control"] == "no-cache"


def test_status_when_everything_is_ready(client):
    status = client.get("/api/status").json()
    assert status["ready"] is True
    assert status["ollama"]["running"] and status["ollama"]["version"] == "0.40.16"
    assert status["model"] == {
        "tag": MODEL,
        "installed": True,
        "vision": True,
        "tools": True,
        "thinking": False,
    }
    assert status["hardware"]["budget_gb"] > 0
    assert {"name": "crop", "label": "Crop"} in status["tools"]


def test_first_start_has_no_model_and_is_not_ready(fake):
    with make_client(fake, Config()) as client:
        status = client.get("/api/status").json()
        assert status["model"] is None and status["ready"] is False


def test_status_without_ollama(fake):
    fake.running = False
    with make_client(fake) as client:
        status = client.get("/api/status").json()
        assert status["ollama"]["running"] is False and status["ready"] is False
        assert status["ollama"]["download_url"] == "https://ollama.com/download"
        assert client.get("/api/models").status_code == 503


def test_a_chosen_model_that_was_removed_is_reported(fake):
    with make_client(fake, Config(model="gone:1b")) as client:
        status = client.get("/api/status").json()
        assert status["ready"] is False
        assert status["model"]["installed"] is False and "not found" in status["model"]["problem"]


# ---------------------------------------------------------------- choosing a model


def test_models_lists_options_with_one_recommendation(client):
    data = client.get("/api/models").json()
    assert data["selected"] == MODEL and data["budget_gb"] > 0 and data["catalog_updated"]
    options = data["options"]
    assert sum(option["recommended"] for option in options) == 1
    assert any(o["installed"] and o["name"] == "Gemma 4 12B" for o in options)
    assert {"tag", "size_gb", "need_gb", "fit", "usable", "vision", "note"} <= set(options[0])


def test_first_start_flow_downloads_and_selects_a_model():
    fake = FakeOllama()
    with make_client(fake, Config()) as client:
        events = stream(client, "/api/models/pull", {"model": "qwen3.5:4b"})
        assert [e["type"] for e in events] == ["progress", "progress", "progress", "done"]
        assert events[1]["percent"] == 50.0
        assert fake.pulled == ["qwen3.5:4b"]

        status = client.post("/api/models/select", json={"model": "qwen3.5:4b"}).json()
        assert status["ready"] is True and status["model"]["tag"] == "qwen3.5:4b"
        assert Config.load().model == "qwen3.5:4b"  # remembered for the next start


def test_models_that_cannot_run_the_agent_are_refused(fake):
    fake.models.append(InstalledModel("chatty:7b", 4.0, ("completion", "vision")))
    with make_client(fake) as client:
        refused = client.post("/api/models/select", json={"model": "chatty:7b"})
        assert refused.status_code == 400 and "cannot call tools" in refused.json()["detail"]
        missing = client.post("/api/models/select", json={"model": "nope:1b"})
        assert missing.status_code == 400 and "not installed" in missing.json()["detail"]
        assert Config.load().model is None  # nothing was saved


def test_a_failed_download_is_reported_in_the_stream(fake):
    from pictor.ollama import OllamaError

    async def failing_pull(model):
        yield {"status": "pulling manifest", "completed": 0, "total": 0, "percent": None}
        raise OllamaError("pull model manifest: file does not exist")

    fake.pull = failing_pull
    with make_client(fake) as client:
        events = stream(client, "/api/models/pull", {"model": "typo:1b"})
        assert events[-1] == {
            "type": "error",
            "message": "pull model manifest: file does not exist",
        }
        assert client.post("/api/models/pull", json={"model": "  "}).status_code == 400


def test_think_setting_is_validated_and_saved(client):
    assert client.post("/api/settings", json={"think": "off"}).json()["think"] == "off"
    assert Config.load().think == "off"
    assert client.post("/api/settings", json={"think": "maybe"}).status_code == 400


# ---------------------------------------------------------------- painting new things


@pytest.mark.parametrize(
    ("state", "supported", "installed", "ready"),
    [
        ("unsupported", False, False, False),
        ("not_installed", True, False, False),
        ("no_weights", True, True, False),
        ("ready", True, True, True),
    ],
)
def test_status_tells_the_four_states_of_the_image_generator_apart(
    fake, state, supported, installed, ready
):
    with make_client(fake, generator=FakeGenerator(state=state)) as client:
        status = client.get("/api/status").json()
    painting = status["generation"]
    assert (painting["state"], painting["supported"]) == (state, supported)
    assert (painting["installed"], painting["ready"]) == (installed, ready)
    # Nothing is switched on, or offered to the model, without the user asking for it.
    assert painting["enabled"] is False and painting["active"] is False
    assert painting["busy"] is False
    assert painting["install_command"].startswith("uv tool install mflux==")
    assert painting["download_gb"] == 16.0 and painting["model"] == "flux2-klein-4b"
    assert bool(painting["reason"]) is not ready
    assert not any(tool["name"] == "add_object" for tool in status["tools"])


def test_setting_up_streams_progress_and_switches_painting_on(fake):
    generator = FakeGenerator(state="no_weights")
    with make_client(fake, generator=generator) as client:
        events = stream(client, "/api/generation/setup", {})
        assert [e["type"] for e in events] == ["progress", "progress", "progress", "done"]
        assert events[0] == {
            "type": "progress",
            "status": "Downloading the image model",
            "completed": 500,
            "total": 1000,
            "percent": 50.0,
        }
        assert events[1]["status"] == "Preparing the image model…"
        assert generator.setups == 1

        status = client.get("/api/status").json()
        painting = status["generation"]
        assert painting["ready"] and painting["enabled"] and painting["active"]
        assert any(tool["name"] == "add_object" for tool in status["tools"])
        assert Config.load().image_generator == "mflux"  # remembered for the next start


def test_a_failed_setup_is_reported_and_leaves_painting_off(fake):
    generator = FakeGenerator(state="no_weights", setup_error=GenerationError("The disk is full."))
    with make_client(fake, generator=generator) as client:
        events = stream(client, "/api/generation/setup", {})
        assert events[-1] == {"type": "error", "message": "The disk is full."}
        assert [e["type"] for e in events[:-1]] == ["progress", "progress"]
        painting = client.get("/api/status").json()["generation"]
        assert painting["enabled"] is False and painting["busy"] is False
        assert Config.load().image_generator == "off"


def test_the_model_is_told_where_painting_can_be_switched_on(fake):
    for state, hinted in (("no_weights", True), ("ready", True), ("unsupported", False)):
        fake.requests.clear()
        fake.script = [say("I cannot paint a dog into it.")]
        with make_client(fake, generator=FakeGenerator(state=state)) as client:
            sid = upload(client)["id"]
            stream(client, f"/api/session/{sid}/chat", {"message": "add a dog"})
        system = fake.requests[0]["messages"][0]["content"]
        assert ("'Add things to pictures'" in system) is hinted


def test_setup_is_refused_when_it_cannot_work(fake):
    for state, words in (("unsupported", "Apple Silicon"), ("not_installed", "not installed")):
        generator = FakeGenerator(state=state)
        with make_client(fake, generator=generator) as client:
            refused = client.post("/api/generation/setup", json={})
            assert refused.status_code == 409 and words in refused.json()["detail"]
            assert generator.setups == 0


def test_the_language_model_makes_room_for_the_setup(fake):
    fake.in_memory = {MODEL: 11.1}
    with make_client(fake, generator=FakeGenerator(state="no_weights")) as client:
        stream(client, "/api/generation/setup", {})
    assert fake.unloaded == [MODEL]


def test_painting_can_be_switched_off_and_on_again(fake):
    config = Config(model=MODEL, image_generator="mflux")
    with make_client(fake, config, generator=FakeGenerator()) as client:
        assert client.get("/api/status").json()["generation"]["active"] is True
        off = client.post("/api/settings", json={"image_generator": "off"}).json()
        assert off["generation"]["enabled"] is False and off["generation"]["ready"] is True
        assert Config.load().image_generator == "off"
        on = client.post("/api/settings", json={"image_generator": "mflux"}).json()
        assert on["generation"]["active"] is True
        assert client.post("/api/settings", json={"image_generator": "dall-e"}).status_code == 400


def test_a_new_object_is_painted_over_http_only_when_painting_is_on(fake):
    dog = {"description": "a small brown dog", "left": 55, "top": 10, "right": 95, "bottom": 45}
    generator = FakeGenerator()
    with make_client(fake, generator=generator) as client:  # ready, but not switched on
        sid = upload(client)["id"]
        fake.script = [call("add_object", **dog), say("I cannot paint.")]
        events = stream(client, f"/api/session/{sid}/chat", {"message": "add a dog"})
        end = next(e for e in events if e["type"] == "tool_end")
        assert end["ok"] is False and "no tool named" in end["result"]
        assert generator.calls == []

        client.post("/api/settings", json={"image_generator": "mflux"})
        fake.script = [call("add_object", **dog), say("Added a dog.")]
        events = stream(client, f"/api/session/{sid}/chat", {"message": "add a dog"})
        end = next(e for e in events if e["type"] == "tool_end")
        assert end["ok"] is True and end["version"]["label"] == "Add object"
        assert [e["text"] for e in events if e["type"] == "tool_status"] == [
            "Loading the image model…",
            "Painting: step 1 of 4…",
        ]
        state = client.get(f"/api/session/{sid}").json()["state"]
        assert state["cursor"] == 1 and (state["width"], state["height"]) == (640, 480)


# ---------------------------------------------------------------- pictures


def test_open_view_and_download_a_picture(client):
    state = upload(client)
    assert (state["width"], state["height"], state["filename"]) == (640, 480, "cat.png")
    sid = state["id"]

    preview = client.get(f"/api/session/{sid}/image/0?w=200")
    assert preview.headers["content-type"] == "image/webp"
    assert "max-age" in preview.headers["cache-control"]
    assert max(Image.open(io.BytesIO(preview.content)).size) == 200

    download = client.get(f"/api/session/{sid}/download")
    assert download.headers["content-type"] == "image/png"
    assert "cat-pictor.png" in download.headers["content-disposition"]
    assert Image.open(io.BytesIO(download.content)).size == (640, 480)

    assert client.get(f"/api/session/{sid}/image/7").status_code == 404
    assert client.get("/api/session/nope/image/0").status_code == 404


def test_a_file_that_is_not_a_picture_is_refused(client):
    response = client.post("/api/session", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert response.status_code == 400 and "could not be opened" in response.json()["detail"]


def test_unicode_filenames_survive_the_download_header(client):
    state = upload(client, name="počitnice šola.png")
    download = client.get(f"/api/session/{state['id']}/download")
    assert "UTF-8''po%C4%8Ditnice%20%C5%A1ola-pictor.png" in download.headers["content-disposition"]


def test_closing_a_picture(client):
    sid = upload(client)["id"]
    assert client.delete(f"/api/session/{sid}").json() == {"closed": True}
    assert client.get(f"/api/session/{sid}").status_code == 404


# ---------------------------------------------------------------- the agent over HTTP


def test_chat_streams_the_run_and_keeps_a_transcript(fake, client):
    fake.script = [
        call("crop_to_aspect_ratio", ratio="1:1"),
        say("Cropped it to a square.", thinking="A square is 1:1."),
    ]
    sid = upload(client)["id"]
    events = stream(client, f"/api/session/{sid}/chat", {"message": "  make it square  "})

    types = [e["type"] for e in events]
    assert types[0] == "status" and types[-1] == "done"
    assert {"tool_start", "tool_end", "state", "thinking", "text"} <= set(types)
    assert fake.requests[0]["messages"][1]["content"].startswith("make it square")

    restored = client.get(f"/api/session/{sid}").json()
    assert restored["busy"] is False
    assert restored["state"]["width"] == restored["state"]["height"] == 480
    user, assistant = restored["transcript"]
    assert user == {"role": "user", "text": "make it square"}
    parts = assistant["parts"]
    assert [p["type"] for p in parts] == ["tool", "thinking", "text"]
    assert parts[0]["state"] == "ok" and parts[0]["label"] == "Crop to ratio"
    assert parts[0]["version"]["id"] == 1 and parts[0]["args"] == {"ratio": "1:1"}
    assert parts[1]["text"] == "A square is 1:1."
    assert parts[2]["text"] == "Cropped it to a square."

    # The edited picture is what gets downloaded.
    download = client.get(f"/api/session/{sid}/download")
    assert Image.open(io.BytesIO(download.content)).size == (480, 480)


def test_undo_redo_and_goto_over_http(fake, client):
    fake.script = [call("resize", percent=50), call("flip", direction="vertical"), say("Done.")]
    sid = upload(client)["id"]
    stream(client, f"/api/session/{sid}/chat", {"message": "halve and flip"})

    state = client.post(f"/api/session/{sid}/undo").json()["state"]
    assert state["cursor"] == 1 and state["can_redo"] and state["can_undo"]
    state = client.post(f"/api/session/{sid}/goto", json={"version": 0}).json()["state"]
    assert (state["cursor"], state["width"]) == (0, 640)
    state = client.post(f"/api/session/{sid}/redo").json()["state"]
    assert (state["cursor"], state["width"]) == (1, 320)
    assert client.post(f"/api/session/{sid}/goto", json={"version": 42}).status_code == 404
    assert client.post("/api/session/nope/undo").status_code == 404


def test_errors_from_the_model_reach_the_transcript(fake, client):
    from pictor.ollama import OllamaError

    fake.script = [OllamaError("connection reset")]
    sid = upload(client)["id"]
    events = stream(client, f"/api/session/{sid}/chat", {"message": "hello"})
    assert events[-1] == {"type": "error", "message": "connection reset"}
    parts = client.get(f"/api/session/{sid}").json()["transcript"][1]["parts"]
    assert parts == [{"type": "error", "text": "connection reset"}]


def test_chat_needs_a_message_and_an_open_picture(client):
    sid = upload(client)["id"]
    assert client.post(f"/api/session/{sid}/chat", json={"message": "   "}).status_code == 400
    assert client.post("/api/session/nope/chat", json={"message": "hi"}).status_code == 404


# ---------------------------------------------------------------- local-only guard


def test_requests_from_other_sites_are_refused(client):
    # DNS rebinding: the browser thinks it is talking to another host.
    rebound = client.get("/api/status", headers={"Host": "evil.example"})
    assert rebound.status_code == 403
    # A page on another site posting to the local server.
    cross_site = client.post(
        "/api/models/select", json={"model": MODEL}, headers={"Origin": "https://evil.example"}
    )
    assert cross_site.status_code == 403
    # The app's own page is fine, on either loopback name.
    assert client.get("/api/status", headers={"Origin": LOCAL}).status_code == 200
    assert client.get("/api/status", headers={"Host": "localhost:8765"}).status_code == 200


def test_the_guard_is_off_when_the_app_is_deliberately_exposed(fake):
    with make_client(fake, allow_remote=True) as client:
        assert client.get("/api/status", headers={"Host": "192.168.1.20:8765"}).status_code == 200
