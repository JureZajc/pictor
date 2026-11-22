"""Painting new content into a picture with an image model that runs outside Pictor.

Ollama runs the model that plans an edit but cannot generate pictures, so a second
program does the painting. The only backend today is `mflux
<https://github.com/filipstrand/mflux>`_ with FLUX.2 klein, behind the small
:class:`ImageGenerator` interface so another one can be added.

mflux is run as a separate process for every edit, never imported: it brings PyTorch and a
second OpenCV build that would clash with Pictor's own, and a process that exits is the
one sure way to hand several gigabytes back to the system.
"""

from __future__ import annotations

import contextlib
import os
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from PIL import Image

from pictor import hardware
from pictor.config import Config, cache_dir

EDIT_PROGRAM = "mflux-generate-flux2-edit"
SAVE_PROGRAM = "mflux-save"
#: The mflux release this was written and measured against.
MFLUX_VERSION = "0.24.0"
INSTALL_COMMAND = f"uv tool install mflux=={MFLUX_VERSION}"
#: Longest one edit may take before it is given up on, in seconds.
EDIT_TIMEOUT_SECONDS = 900.0
#: Longest the one-time download and preparation may take, in seconds.
SETUP_TIMEOUT_SECONDS = 6 * 3600.0
#: How often a running program is checked for cancellation, in seconds.
POLL_SECONDS = 0.1
#: Minimum seconds between two progress updates of the setup step.
SETUP_PROGRESS_INTERVAL = 0.5
#: Sides of the picture handed to the model must be multiples of this many pixels;
#: mflux silently rounds anything else down.
SIZE_STEP = 16
#: MLX keeps freed buffers for reuse, by default until nearly all memory is taken: an
#: edit that needs 6 GB grew to 18 GB on a 24 GB Mac. Capped at this many GB, the same
#: edit stayed at 8 GB and was faster.
MLX_CACHE_LIMIT_GB = 2

#: The models mflux knows by name, and the Hugging Face repository each one downloads.
MODEL_REPOS = {
    "flux2-klein-4b": "black-forest-labs/FLUX.2-klein-4B",
    "flux2-klein-9b": "black-forest-labs/FLUX.2-klein-9B",
    "flux2-klein-9b-kv": "black-forest-labs/FLUX.2-klein-9b-kv",
    "flux2-klein-base-4b": "black-forest-labs/FLUX.2-klein-base-4B",
    "flux2-klein-base-9b": "black-forest-labs/FLUX.2-klein-base-9B",
}
#: Size of the download where it was measured, in bytes.
DOWNLOAD_BYTES = {"flux2-klein-4b": 15_980_000_000}
#: Folders of a model that must hold weights before it can run.
WEIGHT_FOLDERS = ("transformer", "text_encoder", "vae")
#: Written by ``mflux-save`` into every copy it makes.
SAVED_MARKER = "mflux_model.json"

#: Peak memory of one 1024 px edit with FLUX.2 klein 4B, in GB, by bits per weight.
#: From a prepared copy: 8 bits was measured at 8.1 GB, the others are estimates.
PEAK_GB_PREPARED = {3: 6.0, 4: 6.5, 5: 7.0, 6: 7.5, 8: 8.5}
#: Quantizing while loading holds the full-precision weights for a moment: 11.6 GB
#: measured at 8 bits, and no less at 4.
PEAK_GB_ON_THE_FLY = 12.0
#: The preparation step loads the full-precision model once (15.5 GB measured).
PEAK_GB_SETUP = 16.0

_SHARD = re.compile(r"-(\d+)-of-(\d+)\.safetensors$")
#: A bare progress bar such as " 50%|#####     | 2/4 [00:06<00:06,  3.04s/it]". Bars with
#: a caption in front ("Fetching 16 files: ...") count something other than steps.
_STEP = re.compile(r"^\s*\d+%\|.*?(\d+)/(\d+) \[")


class GenerationError(Exception):
    """The image generator produced no picture. The message says what to do about it."""


class GenerationCancelled(GenerationError):
    """The work was stopped on purpose."""


@dataclass(frozen=True)
class Availability:
    """Whether the generator can paint right now, and if not, which step is missing."""

    #: "ready", "unsupported", "not_installed" or "no_weights".
    state: str
    #: A sentence for the user when the state is not "ready".
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.state == "ready"


class ImageGenerator(Protocol):
    #: Peak memory one edit needs, in GB. Decides whether the language model must be
    #: unloaded first.
    memory_gb: float

    def available(self) -> Availability: ...

    def edit(
        self,
        image: Image.Image,
        prompt: str,
        *,
        seed: int,
        status: Callable[[str], None],
        cancelled: threading.Event,
    ) -> Image.Image:
        """Return ``image`` redrawn as ``prompt`` asks, at the same size.

        The sides of ``image`` are multiples of :data:`SIZE_STEP`. Raises
        :class:`GenerationCancelled` once ``cancelled`` is set and
        :class:`GenerationError` for every other failure.
        """
        ...


class InstallableGenerator(ImageGenerator, Protocol):
    """What the setup dialog needs on top of painting."""

    #: Size of the download that :meth:`setup` starts, where it is known.
    download_bytes: int | None
    #: Peak memory :meth:`setup` needs, in GB; zero when there is nothing left to do.
    setup_memory_gb: float

    def setup(self, cancelled: threading.Event | None = None) -> Iterator[dict]:
        """Fetch and prepare whatever painting needs, yielding progress dicts with a
        ``status`` and, while downloading, ``completed``, ``total`` and ``percent``."""
        ...


# --------------------------------------------------------------------------- locating


def find_program(name: str = EDIT_PROGRAM) -> str | None:
    """Path of an mflux command, also when uv's tool folder is not on ``PATH``.

    ``uv tool install`` links programs into ``~/.local/bin`` and only warns when the shell
    does not look there, so that folder is checked as well.
    """
    found = shutil.which(name)
    if found:
        return found
    env = os.environ
    folders = [env.get("UV_TOOL_BIN_DIR"), env.get("XDG_BIN_HOME")]
    if env.get("XDG_DATA_HOME"):
        folders.append(str(Path(env["XDG_DATA_HOME"]).parent / "bin"))
    folders.append(str(Path.home() / ".local" / "bin"))
    for folder in folders:
        if folder:
            candidate = Path(folder).expanduser() / name
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return str(candidate)
    return None


def hub_cache() -> Path:
    """Where Hugging Face downloads live, following the same variables its library reads."""
    env = os.environ
    if env.get("HF_HUB_CACHE"):
        return Path(env["HF_HUB_CACHE"]).expanduser()
    if env.get("HF_HOME"):
        return Path(env["HF_HOME"]).expanduser() / "hub"
    base = Path(env["XDG_CACHE_HOME"]).expanduser() if env.get("XDG_CACHE_HOME") else None
    return (base or Path.home() / ".cache") / "huggingface" / "hub"


def _folder_complete(folder: Path) -> bool:
    """Whether a weights folder holds every shard its file names announce."""
    files = [p for p in folder.glob("*.safetensors") if p.is_file()]  # skips dangling links
    if not files:
        return False
    for path in files:
        match = _SHARD.search(path.name)
        if match:
            prefix = path.name[: match.start()]
            shards = [p for p in files if p.name.startswith(prefix) and _SHARD.search(p.name)]
            if len(shards) < int(match.group(2)):
                return False
    return True


def _has_weights(folder: Path, small_text_encoder: bool = False) -> bool:
    needed = [f for f in WEIGHT_FOLDERS if not (small_text_encoder and f == "text_encoder")]
    return all(_folder_complete(folder / name) for name in needed)


def downloaded_weights(model: str, small_text_encoder: bool = False) -> Path | None:
    """The folder holding a model's complete download, or ``None`` if it is not there.

    ``model`` is one of mflux's names, a Hugging Face ``org/name`` or a local folder.
    """
    local = Path(model).expanduser()
    if local.is_dir():
        candidates = [local]
    else:
        repo = MODEL_REPOS.get(model.lower(), model if "/" in model else None)
        if repo is None:
            return None
        snapshots = hub_cache() / f"models--{repo.replace('/', '--')}" / "snapshots"
        try:
            candidates = sorted(snapshots.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        except OSError:
            return None
    for folder in candidates:
        if _has_weights(folder, small_text_encoder):
            return folder
    return None


def prepared_copy(model: str, quantize: int) -> Path:
    """Where Pictor keeps a model's ready-quantized copy, whether or not it exists yet."""
    name = re.sub(r"[^a-z0-9._-]+", "-", model.lower()).strip("-.") or "model"
    return cache_dir() / "image-models" / f"{name}-q{quantize}"


def _is_prepared(folder: Path) -> bool:
    return (folder / SAVED_MARKER).is_file() and _has_weights(folder)


def supported(machine: hardware.Hardware | None = None) -> bool:
    """mflux runs on Apple Silicon Macs and on Linux with an NVIDIA card."""
    machine = machine or hardware.detect()
    if machine.accelerator == "apple-silicon":
        return True
    return machine.os == "Linux" and machine.accelerator == "nvidia"


def _folder_bytes(folder: Path) -> int:
    """Bytes on disk under a folder, counting each file once and not following links."""
    total = 0
    with contextlib.suppress(OSError):
        for root, _dirs, files in os.walk(folder):
            for name in files:
                with contextlib.suppress(OSError):
                    info = os.lstat(os.path.join(root, name))
                    total += info.st_size
    return total


# --------------------------------------------------------------------------- failures


def explain_failure(output: str, returncode: int | None) -> str:
    """Turn what mflux printed into something a person can act on."""
    low = output.lower()
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    last = lines[-1][:300] if lines else ""
    out_of_memory = ("insufficient memory", "out of memory", "bad_alloc", "outofmemory")
    killed = returncode is not None and returncode in (-signal.SIGKILL, 128 + signal.SIGKILL)
    if killed or any(k in low for k in out_of_memory):
        return (
            "The image generator ran out of memory. Close other programs and try again, or "
            'set "image_quantize" to 4 in the settings file so it needs less.'
        )
    missing = ("offline mode", "localentrynotfound", "model not found", "requested files")
    if any(k in low for k in missing):
        return (
            "The image model's files are missing or incomplete. Open the model menu and "
            "set up the image generator again."
        )
    if "no space left" in low:
        return "The disk is full, so the image generator could not finish."
    if "battery" in low:
        return f"The image generator stopped because the battery is nearly empty. ({last})"
    if any(k in low for k in ("connection", "timed out", "name resolution", "max retries")):
        return f"The download failed; check the internet connection and try again. ({last})"
    return f"The image generator failed. ({last or f'exit code {returncode}'})"


# --------------------------------------------------------------------------- mflux


class MfluxGenerator:
    """Runs ``mflux-generate-flux2-edit`` once per edit."""

    def __init__(
        self,
        *,
        model: str = "flux2-klein-4b",
        quantize: int = 8,
        steps: int = 4,
        small_text_encoder: bool = False,
        program: str | None = None,
        save_program: str | None = None,
        work_dir: Path | None = None,
        timeout: float = EDIT_TIMEOUT_SECONDS,
        machine: hardware.Hardware | None = None,
    ) -> None:
        self.model = model
        self.quantize = quantize
        self.steps = steps
        self.small_text_encoder = small_text_encoder
        self._program = program
        self._save_program = save_program
        self._work_dir = work_dir
        self.timeout = timeout
        self._machine = machine

    @classmethod
    def from_config(cls, config: Config) -> MfluxGenerator:
        return cls(
            model=config.image_model,
            quantize=config.image_quantize,
            steps=config.image_steps,
            small_text_encoder=config.image_small_text_encoder,
        )

    # ------------------------------------------------------------------ state

    @property
    def program(self) -> str | None:
        return self._program or find_program(EDIT_PROGRAM)

    @property
    def save_program(self) -> str | None:
        if self._save_program:
            return self._save_program
        beside = Path(self.program).with_name(SAVE_PROGRAM) if self.program else None
        return str(beside) if beside and beside.is_file() else find_program(SAVE_PROGRAM)

    @property
    def is_folder(self) -> bool:
        """Whether the model setting is a folder of weights the user points at."""
        return Path(self.model).expanduser().is_dir()

    @property
    def can_prepare(self) -> bool:
        """Whether a quantized copy can be made. mflux cannot save the small text encoder."""
        return not self.small_text_encoder and not self.is_folder

    def prepared(self) -> Path | None:
        """The ready-quantized copy to load from, if one has been made."""
        if not self.can_prepare:
            return None
        folder = prepared_copy(self.model, self.quantize)
        return folder if _is_prepared(folder) else None

    @property
    def memory_gb(self) -> float:
        peak = PEAK_GB_PREPARED.get(self.quantize, max(PEAK_GB_PREPARED.values()))
        if self.small_text_encoder:
            peak -= 1.5
        elif not self.is_folder and self.prepared() is None:
            peak = PEAK_GB_ON_THE_FLY
        return peak * (2.0 if "9b" in self.model.lower() else 1.0)

    @property
    def download_bytes(self) -> int | None:
        """Size of the download the setup step starts, where it is known."""
        return DOWNLOAD_BYTES.get(self.model.lower())

    @property
    def setup_memory_gb(self) -> float:
        """Peak memory of the setup step: it loads the full-precision model once."""
        if self.prepared() or self.is_folder:
            return 0.0
        return PEAK_GB_SETUP if self.can_prepare else self.memory_gb

    def available(self) -> Availability:
        if not supported(self._machine):
            return Availability(
                "unsupported",
                "Painting new things needs an Apple Silicon Mac or Linux with an NVIDIA card.",
            )
        if not self.program:
            return Availability(
                "not_installed", f"mflux is not installed. Install it with: {INSTALL_COMMAND}"
            )
        if self.prepared() or self.is_folder:
            return Availability("ready")
        known = self.model.lower() in MODEL_REPOS or "/" in self.model
        if known and downloaded_weights(self.model, self.small_text_encoder) is None:
            return Availability("no_weights", "The image model has not been downloaded yet.")
        return Availability("ready")

    # ------------------------------------------------------------------ painting

    def command(
        self, source: Path, output: Path, prompt: str, seed: int, size: tuple[int, int]
    ) -> list[str]:
        """The argument list for one edit. No shell is involved at any point."""
        program = self.program
        if not program:
            raise GenerationError(f"mflux is not installed. Install it with: {INSTALL_COMMAND}")
        prepared = self.prepared()
        folder = Path(self.model).expanduser()
        if prepared:
            model = ["--model", str(prepared), "--base-model", self.model]
        elif self.is_folder and (folder / SAVED_MARKER).is_file():
            model = ["--model", str(folder)]  # a saved copy carries its own quantization
        else:
            model = ["--model", self.model, "--quantize", str(self.quantize)]
        command = [
            program,
            *model,
            "--steps",
            str(self.steps),
            "--seed",
            str(seed),
            "--width",
            str(size[0]),
            "--height",
            str(size[1]),
            "--mlx-cache-limit-gb",
            str(MLX_CACHE_LIMIT_GB),
            "--image-paths",
            str(source),
            "--output",
            str(output),
            "--no-exif",
        ]
        if self.small_text_encoder:
            command.append("--small-text-encoder")
        # Last, and as one argument with "=": text that starts with a dash stays a prompt.
        command.append(f"--prompt={prompt}")
        return command

    def _scratch(self) -> Path:
        root = self._work_dir or cache_dir() / "generation"
        root.mkdir(parents=True, exist_ok=True)
        cutoff = time.time() - 3600
        for stale in root.glob("edit-*"):  # left behind if Pictor itself was killed
            with contextlib.suppress(OSError):
                if stale.stat().st_mtime < cutoff:
                    shutil.rmtree(stale, ignore_errors=True)
        return root

    def edit(
        self,
        image: Image.Image,
        prompt: str,
        *,
        seed: int,
        status: Callable[[str], None],
        cancelled: threading.Event,
    ) -> Image.Image:
        if image.width % SIZE_STEP or image.height % SIZE_STEP:
            raise GenerationError(f"The picture's sides must be multiples of {SIZE_STEP} pixels.")
        with tempfile.TemporaryDirectory(prefix="edit-", dir=self._scratch()) as folder:
            source, output = Path(folder) / "input.png", Path(folder) / "output.png"
            image.convert("RGB").save(source, "PNG", compress_level=1)
            command = self.command(source, output, prompt, seed, image.size)
            # Offline: an edit must never start a multi-gigabyte download by itself. The
            # weights come down in the setup step, after the user has agreed to it.
            env = {**os.environ, "HF_HUB_OFFLINE": "1", "PYTHONUNBUFFERED": "1"}
            status("Loading the image model…")
            returncode, text = run_watched(
                command,
                env=env,
                cwd=folder,
                on_line=_step_reporter(status),
                cancelled=cancelled,
                timeout=self.timeout,
            )
            if returncode != 0 or not output.is_file():
                raise GenerationError(explain_failure(text, returncode))
            try:
                with Image.open(output) as result:
                    painted = result.convert("RGB")
            except OSError as exc:
                raise GenerationError(
                    f"The image generator's result could not be read ({exc})."
                ) from exc
        if painted.size != image.size:
            painted = painted.resize(image.size, Image.Resampling.LANCZOS)
        return painted

    # ------------------------------------------------------------------ setting up

    def setup(self, cancelled: threading.Event | None = None) -> Iterator[dict]:
        """Download the model and make its quantized copy, yielding progress.

        One ``mflux-save`` run does both: it fetches whatever is missing from Hugging
        Face, then writes the copy edits load from. Each yielded dict has ``status`` and,
        while downloading, ``completed``, ``total`` and ``percent``. Raises
        :class:`GenerationError` on failure and :class:`GenerationCancelled` when stopped;
        a stopped download resumes where it was the next time.
        """
        if not self.program:
            raise GenerationError(f"mflux is not installed. Install it with: {INSTALL_COMMAND}")
        if self.prepared() or self.is_folder:
            yield {"status": "The image model is ready.", "percent": 100.0}
            return
        if not self.can_prepare:
            yield from self._download_only(cancelled)
            return
        save = self.save_program
        if not save:
            raise GenerationError("mflux-save was not found next to mflux; reinstall mflux.")

        final = prepared_copy(self.model, self.quantize)
        partial = final.with_name(final.name + ".partial")
        shutil.rmtree(partial, ignore_errors=True)
        final.parent.mkdir(parents=True, exist_ok=True)
        command = [
            save,
            "--model",
            self.model,
            "--quantize",
            str(self.quantize),
            "--path",
            str(partial),
        ]
        try:
            yield from self._run_setup(command, cancelled)
            if not _has_weights(partial):
                raise GenerationError("The image model was not written completely.")
            shutil.rmtree(final, ignore_errors=True)
            partial.replace(final)
        finally:
            shutil.rmtree(partial, ignore_errors=True)
        yield {"status": "The image model is ready.", "percent": 100.0}

    def _download_only(self, cancelled: threading.Event | None) -> Iterator[dict]:
        """For setups that cannot keep a copy: fetch the weights by painting a tiny picture."""
        with tempfile.TemporaryDirectory(prefix="edit-", dir=self._scratch()) as folder:
            source, output = Path(folder) / "input.png", Path(folder) / "output.png"
            Image.new("RGB", (256, 256), (120, 140, 120)).save(source, "PNG")
            command = self.command(source, output, "a small red ball", 1, (256, 256))
            yield from self._run_setup(command, cancelled)
        yield {"status": "The image model is ready.", "percent": 100.0}

    def _run_setup(self, command: list[str], cancelled: threading.Event | None) -> Iterator[dict]:
        need_download = downloaded_weights(self.model, self.small_text_encoder) is None
        total = self.download_bytes if need_download else None
        hub = hub_cache()
        before = _folder_bytes(hub) if need_download else 0
        stage = {"saving": not need_download}
        stop = cancelled or threading.Event()
        preparing = {"status": "Preparing the image model…", "percent": None}

        def on_line(line: str) -> None:
            if "Saving components" in line or "Image saved" in line:
                stage["saving"] = True

        outcome: dict = {}

        def work() -> None:
            try:
                outcome["result"] = run_watched(
                    command,
                    env={**os.environ, "PYTHONUNBUFFERED": "1"},
                    on_line=on_line,
                    cancelled=stop,
                    timeout=SETUP_TIMEOUT_SECONDS,
                )
            except BaseException as exc:  # noqa: BLE001 - re-raised in the caller's thread
                outcome["error"] = exc

        worker = threading.Thread(target=work, name="pictor-generator-setup", daemon=True)
        worker.start()
        last: dict | None = None
        try:
            while worker.is_alive():
                event = preparing
                if not stage["saving"]:
                    # What has arrived on disk is the one progress measure that does not
                    # depend on how the downloader words its output.
                    done = max(0, _folder_bytes(hub) - before)
                    if not total or done < total:
                        event = {
                            "status": "Downloading the image model",
                            "completed": done,
                            "total": total or 0,
                            "percent": round(100 * done / total, 1) if total else None,
                        }
                if event != last:
                    last = event
                    yield event
                worker.join(SETUP_PROGRESS_INTERVAL)
        finally:
            if worker.is_alive():  # the consumer went away: stop the program too
                stop.set()
                worker.join(10)
        if "error" in outcome:
            raise outcome["error"]
        returncode, text = outcome["result"]
        if returncode != 0:
            raise GenerationError(explain_failure(text, returncode))
        if last != preparing:  # a quick last step still gets its line
            yield preparing


def _step_reporter(status: Callable[[str], None]) -> Callable[[str], None]:
    """Turn mflux's progress bar into one status line per denoising step."""
    last = ""

    def on_line(line: str) -> None:
        nonlocal last
        match = _STEP.search(line)
        if not match:
            return
        done, total = int(match.group(1)), int(match.group(2))
        if not 0 < total <= 100 or done > total:
            return
        note = "Finishing the picture…"
        if done < total:
            note = f"Painting: step {done + 1} of {total}…"
        if note != last:
            last = note
            status(note)

    return on_line


# --------------------------------------------------------------------------- processes


def run_watched(
    command: list[str],
    *,
    env: dict[str, str] | None = None,
    cwd: str | Path | None = None,
    on_line: Callable[[str], None] = lambda _line: None,
    cancelled: threading.Event | None = None,
    timeout: float = EDIT_TIMEOUT_SECONDS,
) -> tuple[int, str]:
    """Run a program to its end, passing on what it prints and killing it when asked.

    Returns the exit code and the last part of its output. Raises
    :class:`GenerationCancelled` when ``cancelled`` is set and :class:`GenerationError`
    when the program cannot be started or runs past ``timeout``. In both of those cases
    the process, and anything it started, is dead when this returns.
    """
    options: dict = {}
    if os.name == "nt":
        options["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    else:
        options["start_new_session"] = True  # its own group, so helpers die with it
    try:
        process = subprocess.Popen(  # an argument list, never a shell
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=cwd,
            **options,
        )
    except OSError as exc:
        raise GenerationError(f"The image generator could not be started ({exc}).") from exc

    tail: deque[str] = deque(maxlen=60)

    def pump() -> None:
        for line in _lines(process.stdout):
            tail.append(line)
            with contextlib.suppress(Exception):  # a bad progress line must not stop the pump
                on_line(line)

    reader = threading.Thread(target=pump, name="pictor-generator-output", daemon=True)
    reader.start()
    deadline = time.monotonic() + timeout
    try:
        while process.poll() is None:
            if cancelled is not None and cancelled.is_set():
                raise GenerationCancelled("Stopped.")
            if time.monotonic() > deadline:
                raise GenerationError(
                    f"The image generator did not finish within {timeout / 60:.0f} minutes "
                    "and was stopped."
                )
            time.sleep(POLL_SECONDS)
    finally:
        _kill(process)  # does nothing to a process that ended by itself
        reader.join(timeout=2)
        if process.stdout:
            process.stdout.close()
    return process.returncode, "\n".join(tail)


def _lines(stream) -> Iterator[str]:
    """Split a byte stream on newlines and on the carriage returns of a progress bar."""
    pending = b""
    while True:
        try:
            chunk = stream.read1(4096)
        except (OSError, ValueError):  # closed under us after a kill
            break
        if not chunk:
            break
        *complete, pending = re.split(rb"[\r\n]", pending + chunk)
        for raw in complete:
            if raw.strip():
                yield raw.decode("utf-8", "replace").strip()
    if pending.strip():
        yield pending.decode("utf-8", "replace").strip()


def _kill(process: subprocess.Popen) -> None:
    """Stop a process and its children at once, and wait until it is really gone."""
    if process.poll() is None:
        try:
            if os.name == "nt":
                process.kill()
            else:
                os.killpg(process.pid, signal.SIGKILL)
        except OSError:
            with contextlib.suppress(OSError):
                process.kill()
    with contextlib.suppress(subprocess.TimeoutExpired, OSError):
        process.wait(timeout=5)


def from_config(config: Config) -> ImageGenerator | None:
    """The generator the settings ask for, or ``None`` when painting is switched off."""
    if config.image_generator == "mflux":
        return MfluxGenerator.from_config(config)
    return None


def describe(candidate: InstallableGenerator, config: Config) -> dict:
    """Where painting stands, for the status API and the setup dialog."""
    state = candidate.available()
    enabled = config.image_generator != "off"
    download = candidate.download_bytes
    return {
        "state": state.state,
        "reason": state.reason,
        "supported": state.state != "unsupported",
        "installed": state.state in ("no_weights", "ready"),
        "ready": state.ok,
        "enabled": enabled,
        #: Whether the model is offered the painting tools right now.
        "active": enabled and state.ok,
        "model": config.image_model,
        "install_command": INSTALL_COMMAND,
        "download_gb": round(download / 1e9, 1) if download else None,
        "memory_gb": candidate.memory_gb,
    }
