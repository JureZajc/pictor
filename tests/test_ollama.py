import json

import httpx2 as httpx
import pytest

from pictor import ollama as ollama_module
from pictor.ollama import Ollama, OllamaError


def ndjson(*events: dict) -> bytes:
    return "".join(json.dumps(event) + "\n" for event in events).encode()


def client_for(handler) -> Ollama:
    return Ollama("http://ollama.test:11434/", transport=httpx.MockTransport(handler))


async def test_version_and_unreachable_server():
    client = client_for(lambda request: httpx.Response(200, json={"version": "0.40.16"}))
    assert client.host == "http://ollama.test:11434"
    assert await client.version() == "0.40.16"

    def refuse(request):
        raise httpx.ConnectError("connection refused")

    down = client_for(refuse)
    assert await down.version() is None
    with pytest.raises(OllamaError, match="Cannot reach Ollama"):
        await down.installed()


async def test_installed_models_come_with_their_capabilities():
    shown = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(
                200,
                json={
                    "models": [
                        {
                            "name": "gemma4:12b",
                            "size": 7_700_000_000,
                            "details": {"parameter_size": "12B"},
                        },
                        {"name": "old-llm:7b", "size": 4_000_000_000},
                    ]
                },
            )
        model = json.loads(request.content)["model"]
        shown.append(model)
        if model == "old-llm:7b":
            return httpx.Response(200, json={"capabilities": ["completion"], "license": "x" * 9999})
        return httpx.Response(
            200,
            json={
                "capabilities": ["completion", "vision", "tools", "thinking"],
                "thinking": {"values": [True, False]},
            },
        )

    client = client_for(handler)
    models = {m.tag: m for m in await client.installed()}
    assert models["gemma4:12b"].size_gb == 7.7 and models["gemma4:12b"].params == "12B"
    assert "tools" in models["gemma4:12b"].capabilities
    assert models["old-llm:7b"].capabilities == ("completion",)

    # Details are cached and stripped of the bulky fields.
    details = await client.show("old-llm:7b")
    assert "license" not in details and shown.count("old-llm:7b") == 1
    await client.show("old-llm:7b", refresh=True)
    assert shown.count("old-llm:7b") == 2


async def test_errors_from_ollama_are_raised_with_their_message():
    client = client_for(lambda request: httpx.Response(404, json={"error": "model 'x' not found"}))
    with pytest.raises(OllamaError, match="model 'x' not found"):
        await client.show("x")
    with pytest.raises(OllamaError, match="not found"):
        async for _ in client.chat("x", [{"role": "user", "content": "hi"}]):
            pass


async def test_chat_streams_chunks_and_sends_the_right_payload():
    sent = {}

    def handler(request: httpx.Request) -> httpx.Response:
        sent.update(json.loads(request.content))
        return httpx.Response(
            200,
            content=ndjson(
                {"message": {"role": "assistant", "thinking": "hm"}},
                {"message": {"role": "assistant", "content": "Hel"}},
                {"message": {"role": "assistant", "content": "lo"}},
                {"message": {"role": "assistant", "content": ""}, "done": True},
            ),
        )

    client = client_for(handler)
    tools = [{"type": "function", "function": {"name": "crop", "parameters": {}}}]
    chunks = [
        chunk
        async for chunk in client.chat(
            "gemma4:12b",
            [{"role": "user", "content": "hi"}],
            tools=tools,
            think="low",
            options={"num_ctx": 8192},
        )
    ]
    assert "".join(c["message"].get("content", "") for c in chunks) == "Hello"
    assert chunks[-1]["done"] is True
    assert sent["stream"] is True and sent["think"] == "low" and sent["tools"] == tools
    assert sent["options"] == {"num_ctx": 8192}

    sent.clear()
    async for _ in client.chat("gemma4:12b", [{"role": "user", "content": "hi"}]):
        pass
    assert "think" not in sent and "tools" not in sent and "options" not in sent


async def test_an_error_in_the_middle_of_a_stream_is_raised():
    body = ndjson({"message": {"content": "Hi"}}, {"error": "mlx runner failed: out of memory"})
    client = client_for(lambda request: httpx.Response(200, content=body))
    seen = []
    with pytest.raises(OllamaError, match="out of memory"):
        async for chunk in client.chat("m", []):
            seen.append(chunk)
    assert len(seen) == 1


async def test_pull_adds_up_layers_and_limits_how_often_it_reports(monkeypatch):
    layer = [
        {"status": "pulling abc", "digest": "abc", "total": 1000, "completed": n}
        for n in range(0, 1001, 10)
    ]
    body = ndjson(
        {"status": "pulling manifest"},
        *layer,
        {"status": "pulling def", "digest": "def", "total": 1000, "completed": 1000},
        {"status": "success"},
    )
    client = client_for(lambda request: httpx.Response(200, content=body))

    updates = [update async for update in client.pull("gemma4:12b")]
    # 104 raw events arrive at once; only status changes and the final state get through.
    assert [u["status"] for u in updates] == [
        "pulling manifest",
        "pulling abc",
        "pulling def",
        "success",
    ]
    assert updates[0]["percent"] is None
    assert updates[-1] == {"status": "success", "completed": 2000, "total": 2000, "percent": 100.0}

    monkeypatch.setattr(ollama_module, "PULL_PROGRESS_INTERVAL", 0)
    assert len([u async for u in client.pull("gemma4:12b")]) == 104


async def test_ensure_running_does_nothing_when_ollama_answers_or_is_missing(monkeypatch):
    up = client_for(lambda request: httpx.Response(200, json={"version": "0.40.16"}))
    assert await up.ensure_running() == "0.40.16"

    def refuse(request):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(Ollama, "binary", staticmethod(lambda: None))
    missing = client_for(refuse)
    assert await missing.ensure_running(wait=0.1) is None
    assert missing._process is None


async def test_models_in_memory_are_listed_and_can_be_unloaded():
    sent = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/ps":
            return httpx.Response(
                200,
                json={"models": [{"name": "gemma4:12b-mlx", "size": 11_100_000_000}, {"size": 5}]},
            )
        sent.append((request.url.path, json.loads(request.content)))
        return httpx.Response(200, json={"done": True, "done_reason": "unload"})

    client = client_for(handler)
    assert await client.loaded() == {"gemma4:12b-mlx": 11.1}
    await client.unload("gemma4:12b-mlx")
    # Ollama's documented way to drop a model now: an empty request that keeps it for 0 s.
    assert sent == [("/api/generate", {"model": "gemma4:12b-mlx", "keep_alive": 0})]

    idle = client_for(lambda request: httpx.Response(200, json={"models": []}))
    assert await idle.loaded() == {}
