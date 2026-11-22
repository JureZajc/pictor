"""The local web server: a JSON API for the browser UI plus the UI's static files."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from importlib import resources
from pathlib import Path
from urllib.parse import quote, urlparse

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from pictor import __version__, catalog, hardware
from pictor.agent import Agent
from pictor.config import THINK_MODES, Config
from pictor.ollama import DOWNLOAD_URL, Ollama, OllamaError
from pictor.session import MAX_UPLOAD_BYTES, ImageLoadError, Session, SessionStore
from pictor.tools import registry

log = logging.getLogger("pictor.server")

LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}
NO_STORE = {"Cache-Control": "no-store"}


class ModelChoice(BaseModel):
    model: str


class Settings(BaseModel):
    think: str | None = None


class ChatRequest(BaseModel):
    message: str


class GotoRequest(BaseModel):
    version: int


class Transcript:
    """Rebuilds the conversation as the UI shows it, so a page reload can restore it."""

    def __init__(self, session: Session, text: str) -> None:
        self.parts: list[dict] = []
        session.transcript.append({"role": "user", "text": text})
        session.transcript.append({"role": "assistant", "parts": self.parts})

    def _append_text(self, kind: str, delta: str) -> None:
        if self.parts and self.parts[-1]["type"] == kind:
            self.parts[-1]["text"] += delta
        else:
            self.parts.append({"type": kind, "text": delta})

    def record(self, event: dict) -> None:
        kind = event["type"]
        if kind in ("thinking", "text"):
            self._append_text(kind, event["delta"])
        elif kind == "tool_start":
            self.parts.append(
                {
                    "type": "tool",
                    "id": event["id"],
                    "name": event["name"],
                    "label": event["label"],
                    "args": event["args"],
                    "state": "running",
                }
            )
        elif kind == "tool_end":
            for part in self.parts:
                if part.get("id") == event["id"]:
                    part.update(
                        state="ok" if event["ok"] else "failed",
                        result=event["result"],
                        version=event["version"],
                    )
        elif kind == "error":
            self.parts.append({"type": "error", "text": event["message"]})

    def close(self, stopped: bool) -> None:
        """Settle anything still marked as running when the stream ends early."""
        for part in self.parts:
            if part.get("state") == "running":
                part["state"] = "failed"
                part["result"] = "Stopped."
        if stopped:
            self.parts.append({"type": "stopped"})


class RevalidatedStatic(StaticFiles):
    """Static files the browser must re-check on every load.

    Without this a browser keeps using its cached script after the app is updated. The
    check is a cheap conditional request that usually answers "not modified".
    """

    async def get_response(self, path: str, scope) -> Response:
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response


def ndjson(events: AsyncIterator[dict]) -> StreamingResponse:
    async def lines() -> AsyncIterator[bytes]:
        async for event in events:
            yield (json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8")

    return StreamingResponse(
        lines(),
        media_type="application/x-ndjson",
        headers={**NO_STORE, "X-Accel-Buffering": "no"},
    )


def create_app(
    config: Config | None = None,
    client: Ollama | None = None,
    allow_remote: bool = False,
    start_ollama: bool = True,
) -> FastAPI:
    config = config or Config.load()
    client = client or Ollama(config.resolved_ollama_host)
    store = SessionStore()
    machine = hardware.detect()
    agent = Agent(client, config, machine.budget_gb)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        store.cleanup()
        starting = None
        if start_ollama:
            # Bring Ollama up in the background so the page can load meanwhile. The
            # reference keeps the task from being garbage-collected mid-flight.
            starting = asyncio.ensure_future(client.ensure_running())
        yield
        if starting and not starting.done():
            starting.cancel()
        await client.aclose()

    app = FastAPI(title="Pictor", version=__version__, lifespan=lifespan, docs_url=None)
    app.state.config = config
    app.state.store = store

    @app.middleware("http")
    async def local_only(request: Request, call_next):
        """Refuse requests that a web page on another site could make to this server.

        The server only listens on loopback, but a browser will happily send requests
        there on behalf of any page. Checking the Host header stops DNS rebinding and
        checking Origin stops cross-site form posts.
        """
        if not allow_remote:
            host = (request.headers.get("host") or "").rsplit(":", 1)[0].strip("[]")
            if host not in LOCAL_HOSTS:
                return JSONResponse({"detail": "Pictor only answers on localhost."}, 403)
            origin = request.headers.get("origin")
            if origin and (urlparse(origin).hostname or "") not in LOCAL_HOSTS:
                return JSONResponse({"detail": "Cross-site requests are not allowed."}, 403)
        return await call_next(request)

    # ------------------------------------------------------------------ helpers

    def session_or_404(session_id: str) -> Session:
        session = store.get(session_id)
        if session is None:
            raise HTTPException(404, "That picture is no longer open. Choose it again.")
        return session

    def idle_or_409(session: Session) -> None:
        if session.lock.locked():
            raise HTTPException(409, "The agent is still working on this picture.")

    async def model_status() -> dict | None:
        if not config.model:
            return None
        try:
            info = await agent.describe(config.model)
        except OllamaError as exc:
            return {"tag": config.model, "installed": False, "problem": str(exc)}
        return {
            "tag": config.model,
            "installed": True,
            "vision": info.vision,
            "tools": info.tools,
            "thinking": info.think is not None or config.think != "auto",
        }

    # ------------------------------------------------------------------ status

    @app.get("/api/status")
    async def status() -> dict:
        version = await client.version()
        model = await model_status() if version else None
        return {
            "version": __version__,
            "ollama": {
                "running": version is not None,
                "version": version,
                "installed": client.binary() is not None or version is not None,
                "host": client.host,
                "download_url": DOWNLOAD_URL,
            },
            "hardware": machine.to_dict(),
            "model": model,
            "think": config.think,
            "ready": bool(model and model.get("installed") and model.get("tools")),
            "tools": [{"name": t.name, "label": t.label} for t in registry.available()],
        }

    @app.get("/api/models")
    async def models() -> dict:
        version = await client.version()
        if version is None:
            raise HTTPException(503, "Ollama is not running.")
        try:
            installed = await client.installed()
        except OllamaError as exc:
            raise HTTPException(503, str(exc)) from exc
        options = catalog.build_options(machine, installed, version)
        return {
            "budget_gb": machine.budget_gb,
            "selected": config.model,
            "catalog_updated": catalog.load_catalog().updated,
            "options": [option.to_dict() for option in options],
        }

    @app.post("/api/models/pull")
    async def pull(choice: ModelChoice) -> StreamingResponse:
        tag = choice.model.strip()
        if not tag:
            raise HTTPException(400, "Give a model name.")

        async def events() -> AsyncIterator[dict]:
            try:
                async for progress in client.pull(tag):
                    yield {"type": "progress", **progress}
                yield {"type": "done"}
            except OllamaError as exc:
                yield {"type": "error", "message": str(exc)}

        return ndjson(events())

    @app.post("/api/models/select")
    async def select(choice: ModelChoice) -> dict:
        tag = choice.model.strip()
        try:
            details = await client.show(tag, refresh=True)
        except OllamaError as exc:
            raise HTTPException(400, f"{tag} is not installed in Ollama: {exc}") from exc
        if "tools" not in (details.get("capabilities") or []):
            raise HTTPException(400, f"{tag} cannot call tools, so it cannot run the agent.")
        config.model = tag
        config.save()
        return await status()

    @app.post("/api/settings")
    async def settings(body: Settings) -> dict:
        if body.think is not None:
            if body.think not in THINK_MODES:
                raise HTTPException(400, f"think must be one of {', '.join(THINK_MODES)}.")
            config.think = body.think
        config.save()
        return await status()

    # ------------------------------------------------------------------ pictures

    @app.post("/api/session")
    async def open_picture(file: UploadFile) -> dict:
        data = await file.read(MAX_UPLOAD_BYTES + 1)
        try:
            session = await asyncio.to_thread(store.create, data, file.filename or "picture")
        except ImageLoadError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"state": session.state(), "transcript": []}

    @app.get("/api/session/{session_id}")
    async def restore(session_id: str) -> dict:
        session = session_or_404(session_id)
        return {
            "state": session.state(),
            "transcript": session.transcript,
            "busy": session.lock.locked(),
        }

    @app.delete("/api/session/{session_id}")
    async def close(session_id: str) -> dict:
        session = session_or_404(session_id)
        idle_or_409(session)
        store.discard(session_id)
        return {"closed": True}

    @app.get("/api/session/{session_id}/image/{version_id}")
    async def image(session_id: str, version_id: int, w: int = 2048) -> Response:
        session = session_or_404(session_id)
        version = session.find(version_id)
        if version is None:
            raise HTTPException(404, "That version no longer exists.")
        data = await asyncio.to_thread(session.preview, version, max(32, min(w, 4096)))
        # Version ids are never reused, so the browser may keep these forever.
        return Response(
            data, media_type="image/webp", headers={"Cache-Control": "private, max-age=86400"}
        )

    @app.get("/api/session/{session_id}/download")
    async def download(session_id: str) -> Response:
        session = session_or_404(session_id)
        data, filename, media_type = await asyncio.to_thread(session.export)
        return Response(
            data,
            media_type=media_type,
            headers={
                **NO_STORE,
                "Content-Disposition": f"attachment; filename*=UTF-8''{quote(filename)}",
            },
        )

    @app.post("/api/session/{session_id}/undo")
    async def undo(session_id: str) -> dict:
        session = session_or_404(session_id)
        idle_or_409(session)
        session.undo()
        return {"state": session.state()}

    @app.post("/api/session/{session_id}/redo")
    async def redo(session_id: str) -> dict:
        session = session_or_404(session_id)
        idle_or_409(session)
        session.redo()
        return {"state": session.state()}

    @app.post("/api/session/{session_id}/goto")
    async def goto(session_id: str, body: GotoRequest) -> dict:
        session = session_or_404(session_id)
        idle_or_409(session)
        if not session.goto(body.version):
            raise HTTPException(404, "That version no longer exists.")
        return {"state": session.state()}

    @app.post("/api/session/{session_id}/chat")
    async def chat(session_id: str, body: ChatRequest) -> StreamingResponse:
        session = session_or_404(session_id)
        text = body.message.strip()
        if not text:
            raise HTTPException(400, "Say what should be done with the picture.")
        idle_or_409(session)

        async def events() -> AsyncIterator[dict]:
            async with session.lock:
                transcript = Transcript(session, text)
                finished = False
                try:
                    async for event in agent.run(session, text):
                        transcript.record(event)
                        yield event
                    finished = True
                finally:
                    # Reached on a clean end and also when the browser stops the stream.
                    transcript.close(stopped=not finished)

        return ndjson(events())

    # ------------------------------------------------------------------ interface

    static = Path(str(resources.files("pictor").joinpath("static")))

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(static / "index.html", headers=NO_STORE)

    app.mount("/static", RevalidatedStatic(directory=static), name="static")
    return app
