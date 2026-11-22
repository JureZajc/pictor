"""User settings and on-disk locations.

Everything lives in the platform's standard per-user folders. Set ``PICTOR_HOME`` to keep
config and cache together somewhere else (the tests do this).
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from platformdirs import user_cache_dir, user_config_dir

APP_NAME = "pictor"
DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
THINK_MODES = ("auto", "off", "default")
IMAGE_GENERATORS = ("off", "mflux")
#: The quantization levels mflux accepts for ``-q``.
IMAGE_QUANTIZE_LEVELS = (3, 4, 5, 6, 8)


def config_dir() -> Path:
    home = os.environ.get("PICTOR_HOME")
    path = Path(home) / "config" if home else Path(user_config_dir(APP_NAME, appauthor=False))
    path.mkdir(parents=True, exist_ok=True)
    return path


def cache_dir() -> Path:
    home = os.environ.get("PICTOR_HOME")
    path = Path(home) / "cache" if home else Path(user_cache_dir(APP_NAME, appauthor=False))
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalize_host(host: str | None) -> str:
    """Turn the forms Ollama accepts in ``OLLAMA_HOST`` into a URL a client can call."""
    host = (host or "").strip().rstrip("/")
    if not host:
        return DEFAULT_OLLAMA_HOST
    explicit_scheme = "://" in host
    if not explicit_scheme:
        host = f"http://{host}"
    scheme, _, rest = host.partition("://")
    address, slash, path = rest.partition("/")
    # A server bound to every interface is still reached through loopback.
    if address.startswith("0.0.0.0"):
        address = "127.0.0.1" + address[len("0.0.0.0") :]
    # A bare host means Ollama's port; a full URL keeps the scheme's own default.
    if not explicit_scheme and ":" not in address.rsplit("]", 1)[-1]:
        address += ":11434"
    return f"{scheme}://{address}{slash}{path}"


@dataclass
class Config:
    #: Ollama tag of the model the agent runs, chosen on first start.
    model: str | None = None
    #: "auto" keeps reasoning short when the model allows it, "off" disables it,
    #: "default" leaves the model's own setting alone.
    think: str = "auto"
    #: Context window in tokens. ``None`` picks one from the memory budget.
    num_ctx: int | None = None
    #: Longest side, in pixels, of the copy of the picture the model looks at.
    vision_max_side: int = 1024
    #: Most tool rounds the agent may take for one instruction.
    max_steps: int = 12
    #: rembg model used by the background tools.
    background_model: str = "isnet-general-use"
    #: Where Ollama listens. ``None`` follows ``OLLAMA_HOST`` or the default port.
    ollama_host: str | None = None
    #: What paints new objects into a picture: "off", or "mflux" once it is set up.
    image_generator: str = "off"
    #: The image model, passed to mflux as ``--model``.
    image_model: str = "flux2-klein-4b"
    #: Bits per weight the image model is loaded with. Fewer bits need less memory.
    image_quantize: int = 8
    #: Denoising steps. Four is what the distilled klein models are made for.
    image_steps: int = 4
    #: Longest side, in pixels, of the part of the picture handed to the image model.
    image_max_side: int = 1024
    #: Use mflux's experimental small text encoder: less memory, weaker prompts.
    image_small_text_encoder: bool = False

    @property
    def resolved_ollama_host(self) -> str:
        return normalize_host(self.ollama_host or os.environ.get("OLLAMA_HOST"))

    @staticmethod
    def path() -> Path:
        return config_dir() / "config.json"

    @classmethod
    def load(cls) -> Config:
        try:
            raw = json.loads(cls.path().read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return cls()
        if not isinstance(raw, dict):
            return cls()
        known = {f.name for f in fields(cls)}
        config = cls(**{k: v for k, v in raw.items() if k in known})
        if config.think not in THINK_MODES:
            config.think = "auto"
        # A hand-edited file must not be able to break the image generator's command line.
        defaults = cls()
        if config.image_generator not in IMAGE_GENERATORS:
            config.image_generator = defaults.image_generator
        if config.image_quantize not in IMAGE_QUANTIZE_LEVELS:
            config.image_quantize = defaults.image_quantize
        if not isinstance(config.image_model, str) or not config.image_model.strip():
            config.image_model = defaults.image_model
        for key, low, high in (("image_steps", 1, 50), ("image_max_side", 256, 2048)):
            value = getattr(config, key)
            if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
                setattr(config, key, getattr(defaults, key))
        config.image_small_text_encoder = config.image_small_text_encoder is True
        return config

    def save(self) -> None:
        path = self.path()
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")
        tmp.replace(path)
