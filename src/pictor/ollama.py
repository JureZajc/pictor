"""A small async client for the parts of the Ollama HTTP API that Pictor uses."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
from collections.abc import AsyncIterator
from typing import Any

import httpx2 as httpx

from pictor.catalog import InstalledModel

DOWNLOAD_URL = "https://ollama.com/download"
#: Minimum seconds between download progress updates passed on to the UI.
PULL_PROGRESS_INTERVAL = 0.25


class OllamaError(Exception):
    """Ollama answered with an error, or could not be reached."""


class Ollama:
    def __init__(self, host: str, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.host = host.rstrip("/")
        # Loading a large model can take minutes before the first byte arrives, so reads
        # are unbounded; only connecting has a deadline.
        self._client = httpx.AsyncClient(
            base_url=self.host,
            timeout=httpx.Timeout(connect=5.0, read=None, write=60.0, pool=5.0),
            transport=transport,
        )
        self._show_cache: dict[str, dict] = {}
        self._process: subprocess.Popen | None = None

    async def aclose(self) -> None:
        await self._client.aclose()
        if self._process and self._process.poll() is None:
            self._process.terminate()

    # ------------------------------------------------------------------ plumbing

    @staticmethod
    async def _error_text(response: httpx.Response) -> str:
        body = (await response.aread()).decode("utf-8", "replace").strip()
        try:
            return str(json.loads(body).get("error") or body)
        except (ValueError, AttributeError):
            return body or f"HTTP {response.status_code}"

    async def _json(self, method: str, path: str, payload: dict | None = None) -> Any:
        try:
            response = await self._client.request(method, path, json=payload, timeout=30.0)
        except httpx.HTTPError as exc:
            reason = str(exc) or type(exc).__name__
            raise OllamaError(f"Cannot reach Ollama at {self.host}: {reason}") from exc
        if response.status_code >= 400:
            raise OllamaError(await self._error_text(response))
        return response.json()

    async def _stream(self, path: str, payload: dict) -> AsyncIterator[dict]:
        """POST and yield one decoded JSON object per line of the streamed reply."""
        try:
            async with self._client.stream("POST", path, json=payload) as response:
                if response.status_code >= 400:
                    raise OllamaError(await self._error_text(response))
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        event = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(event, dict) and event.get("error"):
                        raise OllamaError(str(event["error"]))
                    yield event
        except httpx.HTTPError as exc:
            reason = str(exc) or type(exc).__name__
            raise OllamaError(f"Lost the connection to Ollama at {self.host}: {reason}") from exc

    # ------------------------------------------------------------------ server

    async def version(self) -> str | None:
        """The running server's version, or ``None`` if nothing answers."""
        try:
            response = await self._client.get("/api/version", timeout=3.0)
            return response.json().get("version") if response.status_code == 200 else None
        except (httpx.HTTPError, ValueError):
            return None

    @staticmethod
    def binary() -> str | None:
        return shutil.which("ollama")

    async def ensure_running(self, wait: float = 12.0) -> str | None:
        """Start ``ollama serve`` if Ollama is installed but not answering.

        Only attempted for a local default host. Returns the version once the server is
        up, or ``None`` if Ollama is missing or did not come up in time.
        """
        version = await self.version()
        if version or not self.binary():
            return version
        if not any(h in self.host for h in ("127.0.0.1", "localhost")):
            return None
        kwargs: dict = {}
        if os.name == "nt":
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            self._process = subprocess.Popen(
                [self.binary() or "ollama", "serve"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                **kwargs,
            )
        except OSError:
            return None
        loop = asyncio.get_running_loop()
        deadline = loop.time() + wait
        while loop.time() < deadline:
            await asyncio.sleep(0.4)
            version = await self.version()
            if version:
                return version
        return None

    # ------------------------------------------------------------------ models

    async def show(self, model: str, refresh: bool = False) -> dict:
        """Model details, including ``capabilities`` and the ``thinking`` controls."""
        if refresh or model not in self._show_cache:
            data = await self._json("POST", "/api/show", {"model": model})
            # The licence text and template are large and never needed here.
            self._show_cache[model] = {
                key: data.get(key) for key in ("capabilities", "details", "thinking", "model_info")
            }
        return self._show_cache[model]

    def forget(self, model: str | None = None) -> None:
        if model is None:
            self._show_cache.clear()
        else:
            self._show_cache.pop(model, None)

    async def installed(self) -> list[InstalledModel]:
        """Every locally installed model together with its capabilities."""
        data = await self._json("GET", "/api/tags")
        entries = data.get("models") or []

        async def describe(entry: dict) -> InstalledModel | None:
            tag = entry.get("name") or entry.get("model")
            if not tag:
                return None
            details = entry.get("details") or {}
            capabilities = entry.get("capabilities")
            if capabilities is None:
                try:
                    capabilities = (await self.show(tag)).get("capabilities") or []
                except OllamaError:
                    capabilities = []
            return InstalledModel(
                tag=tag,
                size_gb=round((entry.get("size") or 0) / 1e9, 2),
                capabilities=tuple(capabilities),
                params=details.get("parameter_size") or "",
            )

        found = await asyncio.gather(*(describe(entry) for entry in entries))
        return [model for model in found if model]

    async def loaded(self) -> dict[str, float]:
        """The models Ollama holds in memory right now, as tag -> size in GB."""
        data = await self._json("GET", "/api/ps")
        return {
            str(entry.get("name") or entry.get("model")): round((entry.get("size") or 0) / 1e9, 2)
            for entry in data.get("models") or []
            if entry.get("name") or entry.get("model")
        }

    async def unload(self, model: str) -> None:
        """Have Ollama drop a model from memory now. It loads again on its next use."""
        await self._json("POST", "/api/generate", {"model": model, "keep_alive": 0})

    async def pull(self, model: str) -> AsyncIterator[dict]:
        """Download a model, yielding progress with an overall ``percent`` when known."""
        totals: dict[str, int] = {}
        done: dict[str, int] = {}
        loop = asyncio.get_running_loop()
        last_status, last_sent, progress = None, 0.0, None
        async for event in self._stream("/api/pull", {"model": model, "stream": True}):
            digest = event.get("digest")
            if digest and event.get("total"):
                totals[digest] = int(event["total"])
                done[digest] = int(event.get("completed") or 0)
            total = sum(totals.values())
            completed = sum(done.values())
            status = event.get("status", "")
            progress = {
                "status": status,
                "completed": completed,
                "total": total,
                "percent": round(100 * completed / total, 1) if total else None,
            }
            # Ollama reports thousands of times a second; a few updates are plenty.
            now = loop.time()
            if status != last_status or now - last_sent >= PULL_PROGRESS_INTERVAL:
                last_status, last_sent = status, now
                yield progress
                progress = None
        if progress:
            yield progress  # always deliver the final state
        self.forget(model)

    async def chat(
        self,
        model: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        think: bool | str | None = None,
        options: dict | None = None,
    ) -> AsyncIterator[dict]:
        """Stream one assistant turn. Chunks carry partial ``message`` fields."""
        payload: dict[str, Any] = {"model": model, "messages": messages, "stream": True}
        if tools:
            payload["tools"] = tools
        if think is not None:
            payload["think"] = think
        if options:
            payload["options"] = options
        async for event in self._stream("/api/chat", payload):
            yield event
