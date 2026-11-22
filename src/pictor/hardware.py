"""Detect CPU, memory and GPU, and work out how much memory a local model may use.

Detection shells out to a few well-known system tools and never raises: anything that
cannot be determined is reported as unknown and the budget falls back to a CPU-safe value.
"""

from __future__ import annotations

import contextlib
import functools
import os
import platform
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import psutil

GIB = 1024**3

#: Share of unified memory a model may take on Apple Silicon. macOS caps what the GPU
#: can wire at roughly this much, and the rest keeps the system and browser usable.
APPLE_FRACTION = 0.70
#: Share of dedicated VRAM counted as usable; the desktop and drivers hold the rest.
GPU_FRACTION = 0.95
#: Without a GPU, speed is the limit long before memory is, so the budget is capped.
CPU_FRACTION = 0.50
CPU_CAP_GB = 8.0
#: Smaller cards are treated as display adapters, not as somewhere to run a model.
MIN_DISCRETE_VRAM_GB = 3.5

_PCI_VENDORS = {"0x10de": "nvidia", "0x1002": "amd", "0x1022": "amd", "0x8086": "intel"}


@dataclass(frozen=True)
class Gpu:
    name: str
    vendor: str  # "nvidia" | "amd" | "intel" | "apple" | "other"
    vram_gb: float | None = None
    discrete: bool = False


@dataclass(frozen=True)
class Hardware:
    os: str
    os_version: str
    arch: str
    cpu: str
    cores: int
    ram_gb: float
    gpus: tuple[Gpu, ...]
    accelerator: str  # "apple-silicon" | "nvidia" | "amd" | "intel" | "cpu"
    budget_gb: float
    budget_note: str

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def summary(self) -> str:
        parts = [self.cpu, f"{self.ram_gb:.0f} GB RAM"]
        parts += [
            f"{g.name} ({g.vram_gb:.0f} GB)" if g.vram_gb else g.name
            for g in self.gpus
            if g.discrete
        ]
        return " · ".join(parts)


# --------------------------------------------------------------------------- budget


def compute_budget(
    os_name: str, arch: str, ram_gb: float, gpus: tuple[Gpu, ...] | list[Gpu]
) -> tuple[str, float, str]:
    """Return ``(accelerator, budget_gb, explanation)`` for the given machine."""
    if os_name == "macOS" and arch in ("arm64", "aarch64"):
        budget = round(ram_gb * APPLE_FRACTION, 1)
        note = (
            f"Apple Silicon shares its {ram_gb:.0f} GB between the CPU and GPU, so about "
            f"{budget:g} GB can go to a model while the rest keeps the system responsive."
        )
        return "apple-silicon", budget, note

    for vendor, label in (("nvidia", "NVIDIA"), ("amd", "AMD"), ("intel", "Intel")):
        cards = [g for g in gpus if g.vendor == vendor and g.discrete and g.vram_gb]
        total = sum(g.vram_gb or 0 for g in cards)
        if total >= MIN_DISCRETE_VRAM_GB:
            budget = round(total * GPU_FRACTION, 1)
            where = f"{len(cards)} {label} GPUs" if len(cards) > 1 else f"the {label} GPU"
            note = (
                f"{where[0].upper()}{where[1:]} has {total:g} GB of video memory; a model "
                f"that fits in about {budget:g} GB runs entirely on it."
            )
            return vendor, budget, note

    budget = round(min(ram_gb * CPU_FRACTION, CPU_CAP_GB), 1)
    note = (
        "No supported GPU was found, so the model will run on the CPU. A small model is "
        "chosen to keep it responsive."
    )
    return "cpu", budget, note


# --------------------------------------------------------------------------- helpers


def _run(cmd: list[str], timeout: float = 4.0) -> str | None:
    """Run a command and return its stdout, or ``None`` if it is missing or fails."""
    if shutil.which(cmd[0]) is None:
        return None
    kwargs: dict = {}
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        done = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            stdin=subprocess.DEVNULL,
            **kwargs,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout if done.returncode == 0 else None


def guess_vendor(name: str) -> str:
    low = name.lower()
    if any(k in low for k in ("nvidia", "geforce", "quadro", "rtx", "gtx", "tesla")):
        return "nvidia"
    if any(k in low for k in ("amd", "radeon", "firepro")):
        return "amd"
    if "intel" in low or re.search(r"\barc\b", low):
        return "intel"
    if "apple" in low:
        return "apple"
    return "other"


def looks_discrete(name: str, vendor: str, vram_gb: float | None) -> bool:
    """Tell a card that can host a model from an integrated display adapter."""
    if vendor not in ("nvidia", "amd", "intel"):
        return False
    if vram_gb is None or vram_gb < MIN_DISCRETE_VRAM_GB:
        return False
    if vendor == "intel":
        return bool(re.search(r"\barc\b", name.lower()))
    return True


def parse_nvidia_smi(text: str) -> list[Gpu]:
    """Parse ``nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits``."""
    gpus = []
    for line in text.splitlines():
        name, _, memory = line.rpartition(",")
        name, memory = name.strip(), memory.strip()
        if not name:
            continue
        try:
            vram_gb = round(float(memory) / 1024, 1)  # reported in MiB
        except ValueError:
            vram_gb = None
        gpus.append(Gpu(name, "nvidia", vram_gb, looks_discrete(name, "nvidia", vram_gb)))
    return gpus


# --------------------------------------------------------------------------- probes


def _nvidia_gpus() -> list[Gpu]:
    out = _run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"])
    return parse_nvidia_smi(out) if out else []


def _linux_sysfs_gpus() -> list[Gpu]:
    gpus, seen = [], set()
    for card in sorted(Path("/sys/class/drm").glob("card[0-9]*")):
        device = card / "device"
        try:
            real = device.resolve()
            if real in seen:
                continue
            seen.add(real)
            vendor = _PCI_VENDORS.get((device / "vendor").read_text().strip().lower(), "other")
        except OSError:
            continue
        vram_gb = None
        with contextlib.suppress(OSError, ValueError):
            vram_gb = round(int((device / "mem_info_vram_total").read_text()) / GIB, 1)
        name = {"nvidia": "NVIDIA GPU", "amd": "AMD Radeon GPU", "intel": "Intel GPU"}.get(
            vendor, "GPU"
        )
        gpus.append(Gpu(name, vendor, vram_gb, looks_discrete(name, vendor, vram_gb)))
    return gpus


def _windows_registry_gpus() -> list[Gpu]:
    try:
        import winreg
    except ImportError:
        return []
    gpus = []
    base = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
    try:
        root = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, base)
    except OSError:
        return []
    with root:
        for index in range(32):
            try:
                sub = winreg.OpenKey(root, f"{index:04d}")
            except OSError:
                break
            with sub:
                try:
                    name = str(winreg.QueryValueEx(sub, "DriverDesc")[0])
                except OSError:
                    continue
                vram_gb = None
                # The 64-bit value is the reliable one; the 32-bit one wraps at 4 GB.
                for value in ("HardwareInformation.qwMemorySize", "HardwareInformation.MemorySize"):
                    try:
                        raw = winreg.QueryValueEx(sub, value)[0]
                    except OSError:
                        continue
                    if isinstance(raw, bytes):
                        raw = int.from_bytes(raw, "little")
                    if isinstance(raw, int) and raw > 0:
                        vram_gb = round(raw / GIB, 1)
                        break
                vendor = guess_vendor(name)
                gpus.append(Gpu(name, vendor, vram_gb, looks_discrete(name, vendor, vram_gb)))
    return gpus


def _detect_gpus(os_name: str, arch: str, cpu: str) -> list[Gpu]:
    if os_name == "macOS":
        if arch in ("arm64", "aarch64"):
            return [Gpu(f"{cpu} GPU", "apple", None, False)]
        return []
    gpus = _nvidia_gpus()
    others = _windows_registry_gpus() if os_name == "Windows" else _linux_sysfs_gpus()
    have_nvidia = bool(gpus)
    # nvidia-smi is authoritative for NVIDIA cards; take only the other vendors from the OS.
    gpus += [g for g in others if not (have_nvidia and g.vendor == "nvidia")]
    return gpus


def _cpu_name(os_name: str) -> str:
    name = ""
    if os_name == "macOS":
        name = (_run(["sysctl", "-n", "machdep.cpu.brand_string"]) or "").strip()
    elif os_name == "Linux":
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.lower().startswith(("model name", "hardware")):
                    name = line.split(":", 1)[1].strip()
                    break
        except OSError:
            pass
    elif os_name == "Windows":
        try:
            import winreg

            key = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key) as handle:
                name = str(winreg.QueryValueEx(handle, "ProcessorNameString")[0]).strip()
        except (ImportError, OSError):
            pass
    return name or platform.processor() or platform.machine() or "Unknown CPU"


def _os_info() -> tuple[str, str]:
    system = platform.system()
    if system == "Darwin":
        return "macOS", platform.mac_ver()[0]
    if system == "Windows":
        return "Windows", platform.release()
    if system == "Linux":
        try:
            for line in Path("/etc/os-release").read_text().splitlines():
                if line.startswith("PRETTY_NAME="):
                    return "Linux", line.split("=", 1)[1].strip().strip('"')
        except OSError:
            pass
        return "Linux", platform.release()
    return system or "Unknown", platform.release()


@functools.lru_cache(maxsize=1)
def detect() -> Hardware:
    """Probe this machine once and cache the result for the life of the process."""
    os_name, os_version = _os_info()
    arch = platform.machine().lower()
    cpu = _cpu_name(os_name)
    ram_gb = round(psutil.virtual_memory().total / GIB, 1)
    cores = psutil.cpu_count(logical=False) or os.cpu_count() or 1
    try:
        gpus = tuple(_detect_gpus(os_name, arch, cpu))
    except Exception:  # noqa: BLE001 - detection must never stop the app from starting
        gpus = ()
    accelerator, budget, note = compute_budget(os_name, arch, ram_gb, gpus)
    return Hardware(
        os=os_name,
        os_version=os_version,
        arch=arch,
        cpu=cpu,
        cores=cores,
        ram_gb=ram_gb,
        gpus=gpus,
        accelerator=accelerator,
        budget_gb=budget,
        budget_note=note,
    )
