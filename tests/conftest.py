"""Shared fixtures: an isolated profile, a synthetic picture, a scripted fake Ollama and a
fake image generator."""

from __future__ import annotations

import copy
import io
import threading
from collections.abc import Callable, Iterator

import numpy as np
import pytest
from PIL import Image, ImageDraw

from pictor.catalog import InstalledModel
from pictor.generation import Availability, GenerationCancelled
from pictor.session import Session, SessionStore

ALL_CAPABILITIES = ("completion", "vision", "tools", "thinking")


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    """Keep config, cache and sessions inside the test's temp folder."""
    monkeypatch.setenv("PICTOR_HOME", str(tmp_path / "pictor-home"))
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    return tmp_path / "pictor-home"


def make_photo(width: int = 640, height: int = 480) -> Image.Image:
    """A picture with distinct regions, so tests can tell where pixels came from."""
    image = Image.new("RGB", (width, height), (40, 90, 160))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, width // 2, height // 2), fill=(200, 60, 40))
    draw.ellipse((width * 0.55, height * 0.55, width * 0.9, height * 0.9), fill=(240, 220, 60))
    draw.line((0, height - 1, width - 1, 0), fill=(255, 255, 255), width=4)
    return image


def encode(image: Image.Image, fmt: str = "JPEG", **options) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, fmt, **options)
    return buffer.getvalue()


@pytest.fixture
def photo() -> Image.Image:
    return make_photo()


@pytest.fixture
def store() -> SessionStore:
    return SessionStore()


@pytest.fixture
def session(store, photo) -> Session:
    return store.create(encode(photo, "PNG"), "holiday.png")


def say(text: str, thinking: str = "") -> list[dict]:
    """Chunks of an assistant turn that only talks."""
    chunks = []
    if thinking:
        chunks.append({"message": {"role": "assistant", "thinking": thinking}})
    half = len(text) // 2
    chunks += [
        {"message": {"role": "assistant", "content": text[:half]}},
        {"message": {"role": "assistant", "content": text[half:]}},
        {"message": {"role": "assistant", "content": ""}, "done": True},
    ]
    return chunks


def call(name: str, /, **arguments) -> list[dict]:
    """Chunks of an assistant turn that calls one tool."""
    return calls((name, arguments))


def calls(*requests: tuple[str, dict]) -> list[dict]:
    tool_calls = [
        {"function": {"index": i, "name": name, "arguments": arguments}}
        for i, (name, arguments) in enumerate(requests)
    ]
    return [
        {"message": {"role": "assistant", "content": "", "tool_calls": tool_calls}},
        {"message": {"role": "assistant", "content": ""}, "done": True},
    ]


class FakeOllama:
    """Stands in for :class:`pictor.ollama.Ollama` and replays scripted model turns."""

    host = "http://fake-ollama"

    def __init__(
        self,
        script: list | None = None,
        capabilities: tuple[str, ...] = ALL_CAPABILITIES,
        thinking: dict | None = None,
        installed: list[InstalledModel] | None = None,
        running: bool = True,
    ) -> None:
        self.script = list(script or [])
        self.capabilities = capabilities
        self.thinking = thinking
        self.models = installed or []
        self.running = running
        self.requests: list[dict] = []
        self.pulled: list[str] = []
        #: What Ollama holds in memory, tag -> GB, and what it was told to drop.
        self.in_memory: dict[str, float] = {}
        self.unloaded: list[str] = []

    async def version(self) -> str | None:
        return "0.40.16" if self.running else None

    @staticmethod
    def binary() -> str | None:
        return "/usr/local/bin/ollama"

    async def ensure_running(self, wait: float = 0) -> str | None:
        return await self.version()

    async def show(self, model: str, refresh: bool = False) -> dict:
        from pictor.ollama import OllamaError

        known = {m.tag: m for m in self.models}
        if self.models and model not in known:
            raise OllamaError(f"model '{model}' not found")
        capabilities = known[model].capabilities if model in known else self.capabilities
        return {"capabilities": list(capabilities), "thinking": self.thinking}

    def forget(self, model: str | None = None) -> None:
        pass

    async def installed(self) -> list[InstalledModel]:
        return list(self.models)

    async def loaded(self) -> dict[str, float]:
        return dict(self.in_memory)

    async def unload(self, model: str) -> None:
        self.unloaded.append(model)
        self.in_memory.pop(model, None)

    async def pull(self, model: str):
        self.pulled.append(model)
        yield {"status": "pulling manifest", "completed": 0, "total": 0, "percent": None}
        yield {"status": "pulling model", "completed": 50, "total": 100, "percent": 50.0}
        self.models.append(InstalledModel(model, 1.0, ALL_CAPABILITIES))
        yield {"status": "success", "completed": 100, "total": 100, "percent": 100.0}

    async def chat(self, model, messages, tools=None, think=None, options=None):
        self.requests.append(
            {
                "model": model,
                "messages": copy.deepcopy(messages),
                "tools": tools,
                "think": think,
                "options": options,
            }
        )
        if not self.script:
            raise AssertionError("The model was asked for more turns than the test scripted.")
        turn = self.script.pop(0)
        if isinstance(turn, Exception):
            raise turn
        for chunk in turn:
            yield chunk

    async def aclose(self) -> None:
        pass


MAGENTA = (230, 30, 200)


class FakeGenerator:
    """Stands in for the image model: paints a disc into the picture it is given.

    ``at`` is where the disc goes, as fractions of the picture's width and height. Give a
    list to paint somewhere else on each successive call. ``drift`` shifts every pixel a
    little, the way a real model does.
    """

    memory_gb = 8.5
    download_bytes = 16_000_000_000
    setup_memory_gb = 16.0

    def __init__(
        self,
        at: tuple[float, float] | list[tuple[float, float] | None] = (0.5, 0.5),
        radius: float = 0.2,
        color: tuple[int, int, int] = MAGENTA,
        drift: int = 0,
        state: str = "ready",
        error: Exception | None = None,
        wait_for_cancel: bool = False,
        setup_events: list[dict] | None = None,
        setup_error: Exception | None = None,
    ) -> None:
        self.at = at
        self.radius = radius
        self.color = color
        self.drift = drift
        self.state = state
        self.error = error
        self.wait_for_cancel = wait_for_cancel
        self.setup_events = setup_events
        self.setup_error = setup_error
        self.calls: list[dict] = []
        self.started = threading.Event()
        self.was_cancelled = False
        self.setups = 0

    def available(self) -> Availability:
        reasons = {
            "unsupported": "Painting new things needs an Apple Silicon Mac.",
            "not_installed": "mflux is not installed.",
            "no_weights": "The image model has not been downloaded yet.",
        }
        return Availability(self.state, reasons.get(self.state, ""))

    def edit(
        self,
        image: Image.Image,
        prompt: str,
        *,
        seed: int,
        status: Callable[[str], None],
        cancelled: threading.Event,
    ) -> Image.Image:
        number = len(self.calls)
        self.calls.append({"prompt": prompt, "seed": seed, "size": image.size, "image": image})
        self.started.set()
        status("Loading the image model…")
        if self.wait_for_cancel:
            self.was_cancelled = cancelled.wait(10)
            raise GenerationCancelled("Stopped.")
        if self.error is not None:
            raise self.error
        status("Painting: step 1 of 4…")
        spot = self.at[min(number, len(self.at) - 1)] if isinstance(self.at, list) else self.at
        result = image.convert("RGB")
        if self.drift:
            shifted = np.asarray(result, dtype=np.int16) + self.drift
            result = Image.fromarray(np.clip(shifted, 0, 255).astype(np.uint8))
        if spot is not None:
            x, y = spot[0] * image.width, spot[1] * image.height
            r = self.radius * min(image.size)
            ImageDraw.Draw(result).ellipse((x - r, y - r, x + r, y + r), fill=self.color)
        return result

    def setup(self, cancelled: threading.Event | None = None) -> Iterator[dict]:
        self.setups += 1
        events = self.setup_events or [
            {
                "status": "Downloading the image model",
                "completed": 500,
                "total": 1000,
                "percent": 50.0,
            },
            {"status": "Preparing the image model…", "percent": None},
        ]
        yield from events
        if self.setup_error is not None:
            raise self.setup_error
        self.state = "ready"
        yield {"status": "The image model is ready.", "percent": 100.0}


@pytest.fixture
def generator() -> FakeGenerator:
    return FakeGenerator()
