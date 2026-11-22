"""Command line entry point: ``pictor`` starts the app, ``pictor doctor`` checks the setup."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import os
import socket
import sys
import threading
import webbrowser

from pictor import __version__, catalog, hardware
from pictor.config import Config
from pictor.ollama import DOWNLOAD_URL, Ollama, OllamaError

DEFAULT_PORT = 8765


def find_port(host: str, preferred: int, attempts: int = 20) -> int:
    """The preferred port if it is free, else the next free one after it."""
    for port in range(preferred, preferred + attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            if os.name != "nt":
                # Match the server's own socket option, or a port that was closed a
                # moment ago (a quick restart) looks busy and the address changes.
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind((host, port))
            except OSError:
                continue
            return port
    raise SystemExit(f"No free port between {preferred} and {preferred + attempts - 1}.")


async def _doctor(config: Config) -> int:
    machine = hardware.detect()
    print(f"Pictor {__version__}\n")
    print("Hardware")
    print(f"  System        {machine.os} {machine.os_version} ({machine.arch})")
    print(f"  CPU           {machine.cpu}, {machine.cores} cores")
    print(f"  Memory        {machine.ram_gb:g} GB")
    for gpu in machine.gpus:
        memory = f", {gpu.vram_gb:g} GB" if gpu.vram_gb else ""
        print(f"  GPU           {gpu.name}{memory}")
    print(f"  Runs on       {machine.accelerator}")
    print(f"  Model budget  {machine.budget_gb:g} GB\n")

    client = Ollama(config.resolved_ollama_host)
    try:
        version = await client.version()
        print("Ollama")
        if version is None:
            state = "installed but not running" if client.binary() else "not installed"
            print(f"  {state} at {client.host}")
            print(f"  Get it from {DOWNLOAD_URL}\n")
            installed = []
        else:
            print(f"  Version {version} at {client.host}\n")
            try:
                installed = await client.installed()
            except OllamaError as exc:
                print(f"  Could not list models: {exc}\n")
                installed = []
        options = catalog.build_options(machine, installed, version)
    finally:
        await client.aclose()

    print("Models (memory needed against the budget above)")
    marks = {"fits": "fits", "tight": "tight", "too_big": "too big"}
    for option in options:
        flags = [marks[option.fit]]
        if option.installed:
            flags.append("installed")
        if not option.usable:
            flags.append("no tool calling")
        if option.recommended:
            flags.append("RECOMMENDED")
        chosen = "*" if option.tag == config.model else " "
        print(f" {chosen} {option.tag:<44} {option.need_gb:>5g} GB  {', '.join(flags)}")
    print(f"\nChosen model: {config.model or 'none yet (picked on first start)'}")
    print(f"Settings file: {Config.path()}")
    return 0 if version else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pictor",
        description="Edit pictures by telling a local AI model what to do.",
    )
    parser.add_argument("command", nargs="?", choices=["run", "doctor"], default="run")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="port to listen on")
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="address to listen on (anything but 127.0.0.1 exposes the app to your network)",
    )
    parser.add_argument("--model", help="use this Ollama model tag and remember it")
    parser.add_argument("--no-browser", action="store_true", help="do not open the browser")
    parser.add_argument("--version", action="version", version=f"pictor {__version__}")
    args = parser.parse_args(argv)

    config = Config.load()
    if args.model:
        config.model = args.model
        config.save()

    if args.command == "doctor":
        return asyncio.run(_doctor(config))

    import uvicorn

    from pictor.server import create_app

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    local = args.host in ("127.0.0.1", "localhost")
    port = find_port(args.host, args.port)
    url = f"http://{'127.0.0.1' if local else args.host}:{port}"
    print(f"Pictor {__version__} is running at {url}  (Ctrl+C to stop)", flush=True)
    if not local:
        print("Warning: listening beyond localhost. Anyone on your network can use the app.")
    if not args.no_browser:
        threading.Timer(0.8, webbrowser.open, args=(url,)).start()

    app = create_app(config, allow_remote=not local)
    with contextlib.suppress(KeyboardInterrupt):
        uvicorn.run(app, host=args.host, port=port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
