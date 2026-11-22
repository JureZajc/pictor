"""Shared fixtures: an isolated profile, a synthetic picture and a scripted fake Ollama."""

from __future__ import annotations

import copy
import io

import pytest
from PIL import Image, ImageDraw

from pictor.catalog import InstalledModel
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
