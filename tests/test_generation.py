"""The image generator backend, with a small script standing in for mflux."""

import json
import os
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image, ImageChops

from pictor import generation
from pictor.config import Config, cache_dir
from pictor.generation import (
    GenerationCancelled,
    GenerationError,
    MfluxGenerator,
    explain_failure,
    find_program,
    prepared_copy,
)
from pictor.hardware import Gpu, Hardware
from tests.conftest import make_photo

pytestmark = pytest.mark.skipif(os.name == "nt", reason="the stand-in program is a POSIX script")

MAC = Hardware("macOS", "27", "arm64", "Apple M4", 10, 24.0, (), "apple-silicon", 16.8, "")
LINUX_NVIDIA = Hardware(
    "Linux", "6", "x86_64", "Ryzen", 8, 32.0, (Gpu("RTX", "nvidia", 16, True),), "nvidia", 15.2, ""
)
WINDOWS_NVIDIA = Hardware("Windows", "11", "amd64", "Ryzen", 8, 32.0, (), "nvidia", 15.2, "")
INTEL_MAC = Hardware("macOS", "14", "x86_64", "Core i7", 4, 16.0, (), "cpu", 8.0, "")

# Behaves like mflux as far as Pictor can tell: the same options, a progress bar with
# carriage returns on stderr, and a picture written to --output. STUB_MODE picks a failure.
STUB = r"""
import json, os, shutil, subprocess, sys, time

args = sys.argv[1:]
mode = os.environ.get("STUB_MODE", "")
record = os.environ.get("STUB_RECORD")
if record:
    kept = {k: os.environ.get(k) for k in ("HF_HUB_OFFLINE", "PYTHONUNBUFFERED")}
    with open(record, "w") as handle:
        json.dump({"argv": args, "env": kept, "cwd": os.getcwd()}, handle)


def value(option):
    return args[args.index(option) + 1]


def say(text, end="\n"):
    sys.stderr.write(text + end)
    sys.stderr.flush()


if mode == "sleep":
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    with open(os.environ["STUB_PIDS"], "w") as handle:
        handle.write(f"{os.getpid()} {child.pid}")
    time.sleep(60)
if mode == "oom":
    say("RuntimeError: [METAL] Command buffer execution failed: Insufficient Memory")
    sys.exit(1)
if mode == "offline":
    say("huggingface_hub.errors.OfflineModeIsEnabled: Cannot reach https://huggingface.co/x: "
        "offline mode is enabled.")
    sys.exit(1)
if mode == "odd":
    say("Traceback (most recent call last):")
    say("ValueError: something nobody expected")
    sys.exit(3)
if mode == "disk-full":
    say("OSError: [Errno 28] No space left on device")
    sys.exit(1)

if "--path" in args:  # mflux-save
    if mode == "download":
        with open(os.path.join(os.environ["HF_HUB_CACHE"], "blob"), "wb") as handle:
            handle.write(b"x" * 4096)
        time.sleep(0.12)
    say("Saving components:   0%|          | 0/3 [00:00<?, ?component/s]", end="\r")
    target = value("--path")
    for folder in ("transformer", "text_encoder", "vae"):
        os.makedirs(os.path.join(target, folder))
        open(os.path.join(target, folder, "0.safetensors"), "wb").close()
    with open(os.path.join(target, "mflux_model.json"), "w") as handle:
        json.dump({"quantize": int(value("--quantize"))}, handle)
    sys.exit(0)

say("Fetching 16 files:  44%|####      | 7/16 [00:00<00:00, 14.96it/s]")
for step in range(5):
    say(f"{step * 25:3d}%|{'#' * step:10s}| {step}/4 [00:0{step}<00:04,  1.86s/it]", end="\r")
say("")
if mode != "no-output":
    shutil.copyfile(os.environ["STUB_ANSWER"], value("--output"))
    say("INFO: Image saved successfully")
"""


@pytest.fixture
def stub(tmp_path, monkeypatch) -> SimpleNamespace:
    program = tmp_path / "bin" / "mflux-generate-flux2-edit"
    program.parent.mkdir()
    program.write_text(f"#!{sys.executable}\n{STUB}", encoding="utf-8")
    program.chmod(0o755)
    answer = make_photo(256, 192).transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    answer.save(tmp_path / "answer.png")
    hub = tmp_path / "hub"
    hub.mkdir()
    monkeypatch.setenv("STUB_ANSWER", str(tmp_path / "answer.png"))
    monkeypatch.setenv("STUB_RECORD", str(tmp_path / "record.json"))
    monkeypatch.setenv("STUB_PIDS", str(tmp_path / "pids"))
    monkeypatch.setenv("HF_HUB_CACHE", str(hub))
    monkeypatch.delenv("STUB_MODE", raising=False)
    monkeypatch.setattr(generation, "SETUP_PROGRESS_INTERVAL", 0.02)
    monkeypatch.setattr(generation, "POLL_SECONDS", 0.01)  # keep the tests quick
    return SimpleNamespace(
        program=program,
        answer=answer,
        record=tmp_path / "record.json",
        pids=tmp_path / "pids",
        work=tmp_path / "work",
        hub=hub,
        root=tmp_path,
    )


def make(stub: SimpleNamespace, **options) -> MfluxGenerator:
    options.setdefault("machine", MAC)
    return MfluxGenerator(
        program=str(stub.program), save_program=str(stub.program), work_dir=stub.work, **options
    )


def edit(generator: MfluxGenerator, prompt: str = "add a dog", **options):
    options.setdefault("seed", 42)
    options.setdefault("status", lambda _text: None)
    options.setdefault("cancelled", threading.Event())
    return generator.edit(make_photo(256, 192), prompt, **options)


def fake_download(hub: Path, repo: str = "black-forest-labs/FLUX.2-klein-4B", parts=None) -> Path:
    """Lay out a Hugging Face cache entry the way a finished download looks."""
    snapshot = hub / f"models--{repo.replace('/', '--')}" / "snapshots" / "e7b7dc27"
    parts = parts or {
        "transformer": ["diffusion_pytorch_model.safetensors"],
        "text_encoder": ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"],
        "vae": ["diffusion_pytorch_model.safetensors"],
    }
    for folder, names in parts.items():
        (snapshot / folder).mkdir(parents=True)
        for name in names:
            (snapshot / folder / name).write_bytes(b"weights")
    return snapshot


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def press_when_running(stub: SimpleNamespace, stop: threading.Event) -> None:
    """Set ``stop`` from another thread once the stand-in program is up, like a user would."""

    def press() -> None:
        deadline = time.monotonic() + 5
        while not stub.pids.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        time.sleep(0.05)
        stop.set()

    threading.Thread(target=press, daemon=True).start()


def wait_until_dead(pids: list[int], seconds: float = 3.0) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if not any(alive(pid) for pid in pids):
            return True
        time.sleep(0.02)
    return False


# ---------------------------------------------------------------- one edit


def test_an_edit_runs_the_program_and_returns_its_picture(stub):
    notes = []
    result = edit(make(stub), status=notes.append)

    assert result.mode == "RGB" and result.size == (256, 192)
    assert ImageChops.difference(result, stub.answer).getbbox() is None
    assert notes == [
        "Loading the image model…",
        "Painting: step 1 of 4…",
        "Painting: step 2 of 4…",
        "Painting: step 3 of 4…",
        "Painting: step 4 of 4…",
        "Finishing the picture…",
    ]  # one line per step; the "Fetching 16 files" bar is not a step
    assert list(stub.work.iterdir()) == []  # the input and output files are gone again


def test_the_command_is_an_argument_list_and_the_prompt_one_argument(stub):
    prompt = '--help "quoted" $(touch pwned) ; rm -rf ~ && `id`\nsecond line — ünïcode'
    edit(make(stub, quantize=4, steps=6), prompt, seed=7)

    record = json.loads(stub.record.read_text())
    argv = record["argv"]
    assert argv[-1] == f"--prompt={prompt}"  # whole, unsplit, and never read as an option
    assert argv[:4] == ["--model", "flux2-klein-4b", "--quantize", "4"]
    for option, value in (
        ("--steps", "6"),
        ("--seed", "7"),
        ("--width", "256"),
        ("--height", "192"),
        ("--mlx-cache-limit-gb", "2"),
    ):
        assert argv[argv.index(option) + 1] == value
    assert "--no-exif" in argv and "--small-text-encoder" not in argv
    source, output = argv[argv.index("--image-paths") + 1], argv[argv.index("--output") + 1]
    assert Path(source).parent == Path(output).parent and stub.work in Path(source).parents
    # No shell ever saw the prompt.
    assert not list(stub.root.rglob("pwned"))
    # An edit may not start a download on its own.
    assert record["env"]["HF_HUB_OFFLINE"] == "1"


def test_the_small_text_encoder_is_passed_on_when_chosen(stub):
    edit(make(stub, small_text_encoder=True))
    assert "--small-text-encoder" in json.loads(stub.record.read_text())["argv"]


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("oom", "ran out of memory"),
        ("offline", "files are missing or incomplete"),
        ("disk-full", "disk is full"),
        ("odd", "ValueError: something nobody expected"),
        ("no-output", "The image generator failed"),
    ],
)
def test_failures_are_explained_in_terms_a_person_can_act_on(stub, monkeypatch, mode, expected):
    monkeypatch.setenv("STUB_MODE", mode)
    with pytest.raises(GenerationError, match=expected):
        edit(make(stub))
    assert list(stub.work.iterdir()) == []


def test_explain_failure_reads_the_usual_ways_a_run_dies():
    assert "out of memory" in explain_failure("", -9)  # killed by the system
    assert "battery" in explain_failure("Stopping: battery level is below 5 percent", 0)
    assert "internet connection" in explain_failure("ConnectionError: Max retries exceeded", 1)
    assert explain_failure("", 2) == "The image generator failed. (exit code 2)"


def test_a_missing_program_is_reported_before_and_during_an_edit(stub, monkeypatch, tmp_path):
    monkeypatch.setenv("PATH", str(tmp_path / "nowhere"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    missing = MfluxGenerator(machine=MAC, work_dir=stub.work)
    state = missing.available()
    assert state.state == "not_installed" and not state.ok
    assert "uv tool install mflux==" in state.reason
    with pytest.raises(GenerationError, match="not installed"):
        edit(missing)

    gone = MfluxGenerator(program=str(tmp_path / "deleted"), machine=MAC, work_dir=stub.work)
    with pytest.raises(GenerationError, match="could not be started"):
        edit(gone)


def test_sizes_the_model_cannot_take_are_refused(stub):
    with pytest.raises(GenerationError, match="multiples of 16"):
        make(stub).edit(
            Image.new("RGB", (250, 192)),
            "x",
            seed=1,
            status=lambda _text: None,
            cancelled=threading.Event(),
        )


# ---------------------------------------------------------------- stopping


def test_cancelling_kills_the_program_and_everything_it_started(stub, monkeypatch):
    monkeypatch.setenv("STUB_MODE", "sleep")
    stop = threading.Event()
    press_when_running(stub, stop)
    started = time.monotonic()
    with pytest.raises(GenerationCancelled):
        edit(make(stub), cancelled=stop)
    assert time.monotonic() - started < 4
    pids = [int(pid) for pid in stub.pids.read_text().split()]
    assert len(pids) == 2 and wait_until_dead(pids)
    assert list(stub.work.iterdir()) == []


def test_an_edit_that_never_finishes_is_stopped_after_the_timeout(stub, monkeypatch):
    monkeypatch.setenv("STUB_MODE", "sleep")
    with pytest.raises(GenerationError, match="did not finish"):
        edit(make(stub, timeout=0.15))
    assert wait_until_dead([int(pid) for pid in stub.pids.read_text().split()])


# ---------------------------------------------------------------- what is installed


def test_the_four_states_of_availability(stub, tmp_path, monkeypatch):
    assert make(stub, machine=INTEL_MAC).available().state == "unsupported"
    assert make(stub, machine=WINDOWS_NVIDIA).available().state == "unsupported"
    assert "Apple Silicon" in make(stub, machine=INTEL_MAC).available().reason

    waiting = make(stub).available()
    assert waiting.state == "no_weights" and "not been downloaded" in waiting.reason

    fake_download(stub.hub)
    assert make(stub).available().ok
    assert make(stub, machine=LINUX_NVIDIA).available().state == "ready"


def test_a_download_that_stopped_halfway_does_not_count(stub):
    fake_download(
        stub.hub,
        parts={
            "transformer": ["diffusion_pytorch_model.safetensors"],
            "text_encoder": ["model-00001-of-00002.safetensors"],  # the second shard is missing
            "vae": ["diffusion_pytorch_model.safetensors"],
        },
    )
    assert make(stub).available().state == "no_weights"
    # The small text encoder comes from another repository, so this is enough for it.
    assert make(stub, small_text_encoder=True).available().ok


def test_other_models_and_folders(stub, tmp_path):
    assert make(stub, model="someone/their-klein").available().state == "no_weights"
    fake_download(stub.hub, "someone/their-klein")
    assert make(stub, model="someone/their-klein").available().ok
    # A folder of weights the user points at is taken as it is.
    folder = tmp_path / "my-model"
    folder.mkdir()
    assert make(stub, model=str(folder)).available().ok
    # A name mflux may know and Pictor does not is given the benefit of the doubt.
    assert make(stub, model="some-future-model").available().ok


def test_mflux_is_found_in_the_uv_tool_folder_even_when_it_is_not_on_the_path(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("PATH", str(tmp_path / "nowhere"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    for name in ("UV_TOOL_BIN_DIR", "XDG_BIN_HOME", "XDG_DATA_HOME"):
        monkeypatch.delenv(name, raising=False)
    assert find_program() is None
    local = tmp_path / "home" / ".local" / "bin"
    local.mkdir(parents=True)
    (local / "mflux-generate-flux2-edit").write_text("#!/bin/sh\n")
    assert find_program() is None  # there, but not executable
    (local / "mflux-generate-flux2-edit").chmod(0o755)
    assert find_program() == str(local / "mflux-generate-flux2-edit")

    monkeypatch.setenv("PATH", str(local))
    assert find_program() == str(local / "mflux-generate-flux2-edit")


def test_the_generator_follows_the_settings():
    assert generation.from_config(Config()) is None  # off until the user turns it on
    chosen = generation.from_config(
        Config(image_generator="mflux", image_quantize=4, image_steps=6, image_model="x/y")
    )
    assert isinstance(chosen, MfluxGenerator)
    assert (chosen.model, chosen.quantize, chosen.steps) == ("x/y", 4, 6)


# ---------------------------------------------------------------- setting up


def test_setup_downloads_and_prepares_a_copy_that_later_edits_load(stub, monkeypatch):
    monkeypatch.setenv("STUB_MODE", "download")
    generator = make(stub)
    assert generator.available().state == "no_weights" and generator.prepared() is None
    assert generator.memory_gb == generation.PEAK_GB_ON_THE_FLY

    events = list(generator.setup())
    statuses = [event["status"] for event in events]
    assert statuses[0] == "Downloading the image model"
    assert "Preparing the image model…" in statuses
    assert events[-1] == {"status": "The image model is ready.", "percent": 100.0}
    downloading = [e for e in events if e["status"] == "Downloading the image model"]
    assert downloading[-1]["completed"] >= 4096  # grows with what arrives on disk
    assert downloading[-1]["total"] == generator.download_bytes
    assert 0 <= downloading[-1]["percent"] < 1

    record = json.loads(stub.record.read_text())
    assert record["argv"][:4] == ["--model", "flux2-klein-4b", "--quantize", "8"]
    assert record["env"]["HF_HUB_OFFLINE"] != "1"  # this is the step that may download

    folder = prepared_copy("flux2-klein-4b", 8)
    assert folder == cache_dir() / "image-models" / "flux2-klein-4b-q8"
    assert generator.prepared() == folder and generator.available().ok
    assert not folder.with_name(folder.name + ".partial").exists()
    assert generator.memory_gb == 8.5

    # From now on edits load the prepared copy and do not quantize again.
    edit(generator)
    argv = json.loads(stub.record.read_text())["argv"]
    assert argv[:4] == ["--model", str(folder), "--base-model", "flux2-klein-4b"]
    assert "--quantize" not in argv

    # Asking again does nothing, and another quantization has its own copy.
    stub.record.unlink()
    assert list(generator.setup()) == [{"status": "The image model is ready.", "percent": 100.0}]
    assert not stub.record.exists()
    assert make(stub, quantize=4).prepared() is None


def test_setup_that_fails_leaves_nothing_half_made(stub, monkeypatch):
    monkeypatch.setenv("STUB_MODE", "disk-full")
    generator = make(stub)
    with pytest.raises(GenerationError, match="disk is full"):
        list(generator.setup())
    assert generator.prepared() is None
    assert not (cache_dir() / "image-models").exists() or not list(
        (cache_dir() / "image-models").iterdir()
    )


def test_setup_can_be_stopped(stub, monkeypatch):
    monkeypatch.setenv("STUB_MODE", "sleep")
    stop = threading.Event()
    press_when_running(stub, stop)
    seen = []
    with pytest.raises(GenerationCancelled):
        for event in make(stub).setup(stop):
            seen.append(event)
    # One line while nothing arrives, not one every poll.
    assert [event["status"] for event in seen] == ["Downloading the image model"]
    assert wait_until_dead([int(pid) for pid in stub.pids.read_text().split()])
    assert make(stub).prepared() is None


def test_setup_without_mflux_says_how_to_install_it(stub, monkeypatch, tmp_path):
    monkeypatch.setenv("PATH", str(tmp_path / "nowhere"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    with pytest.raises(GenerationError, match="uv tool install mflux"):
        list(MfluxGenerator(machine=MAC).setup())
