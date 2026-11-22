"""A picture being worked on: its versions, undo history and export settings."""

from __future__ import annotations

import asyncio
import io
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from pictor.config import cache_dir

try:  # HEIC/HEIF is what phones save by default
    import pillow_heif

    pillow_heif.register_heif_opener()
except Exception:  # noqa: BLE001 - optional codec; other formats still work
    pass

#: name -> (Pillow format, media type, file extension)
EXPORT_FORMATS = {
    "png": ("PNG", "image/png", ".png"),
    "jpeg": ("JPEG", "image/jpeg", ".jpg"),
    "webp": ("WEBP", "image/webp", ".webp"),
}
MAX_UPLOAD_BYTES = 80 * 1024 * 1024
MAX_SIDE = 12000
SESSION_TTL_SECONDS = 24 * 3600


class ImageLoadError(ValueError):
    """The uploaded file is not a picture Pictor can open."""


def has_transparency(image: Image.Image) -> bool:
    return image.mode == "RGBA" and image.getextrema()[3][0] < 255


def normalize(image: Image.Image) -> Image.Image:
    """Return an upright RGB or RGBA copy, whatever mode or orientation the file had."""
    image = ImageOps.exif_transpose(image)
    if image.mode in ("RGB", "RGBA"):
        return image.copy()
    if image.mode in ("I", "F") or image.mode.startswith("I;16"):
        # High bit-depth greyscale: rescale to 8 bits instead of clipping.
        data = np.asarray(image, dtype=np.float64)
        low, high = float(data.min()), float(data.max())
        scale = 255.0 / (high - low) if high > low else 0.0
        return Image.fromarray(((data - low) * scale).astype(np.uint8)).convert("RGB")
    alpha = image.mode in ("LA", "PA", "RGBa") or "transparency" in image.info
    return image.convert("RGBA" if alpha else "RGB")


def read_exif(image: Image.Image) -> dict[str, str]:
    """A few human-readable facts from the file's metadata, if it has any."""
    facts: dict[str, str] = {}
    try:
        exif = image.getexif()
    except Exception:  # noqa: BLE001 - damaged metadata must not block opening the picture
        return facts
    if not exif:
        return facts
    camera = " ".join(str(exif.get(tag, "")).strip() for tag in (271, 272)).strip()
    if camera:
        facts["camera"] = camera
    detail = exif.get_ifd(0x8769)
    taken = detail.get(36867) or exif.get(306)
    if taken:
        facts["taken"] = str(taken)
    if detail.get(42036):
        facts["lens"] = str(detail[42036]).strip()
    settings = []
    try:
        if detail.get(37386):
            settings.append(f"{float(detail[37386]):g} mm")
        if detail.get(33437):
            settings.append(f"f/{float(detail[33437]):g}")
        if detail.get(33434):
            exposure = float(detail[33434])
            settings.append(f"1/{round(1 / exposure)} s" if 0 < exposure < 1 else f"{exposure:g} s")
        if detail.get(34855):
            settings.append(f"ISO {detail[34855]}")
    except (TypeError, ValueError, ZeroDivisionError):
        pass
    if settings:
        facts["settings"] = ", ".join(settings)
    if exif.get_ifd(0x8825):
        facts["gps"] = "location recorded"
    return facts


@dataclass
class Version:
    id: int
    label: str
    detail: str
    path: Path
    width: int
    height: int

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "detail": self.detail,
            "width": self.width,
            "height": self.height,
        }


@dataclass
class Session:
    id: str
    dir: Path
    filename: str
    source_format: str
    icc_profile: bytes | None = None
    exif: dict[str, str] = field(default_factory=dict)
    versions: list[Version] = field(default_factory=list)
    cursor: int = -1
    output_format: str = "png"
    output_quality: int = 90
    #: True once the format was asked for, so transparency no longer overrides it.
    output_explicit: bool = False
    #: Past exchanges as sent to the model, one list of messages per user instruction,
    #: without picture data.
    turns: list[list[dict]] = field(default_factory=list)
    #: Conversation as shown in the UI, so a page reload can restore it.
    transcript: list[dict] = field(default_factory=list)
    #: Per-version scratch space for tools, e.g. a computed subject mask.
    scratch: dict = field(default_factory=dict)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _next_id: int = 0
    _images: dict[int, Image.Image] = field(default_factory=dict)
    _previews: dict[tuple[int, int], bytes] = field(default_factory=dict)
    #: Tools run in worker threads while the server reads previews, so caches are guarded.
    _cache_lock: threading.RLock = field(default_factory=threading.RLock)

    # ------------------------------------------------------------------ versions

    @property
    def current(self) -> Version:
        return self.versions[self.cursor]

    def image(self, version: Version | None = None) -> Image.Image:
        """The pixels of a version (default: the current one). Treat as read-only."""
        version = version or self.current
        with self._cache_lock:
            cached = self._images.get(version.id)
            if cached is None:
                with Image.open(version.path) as handle:
                    cached = handle.copy()
                self._remember(version.id, cached)
            return cached

    def _remember(self, version_id: int, image: Image.Image) -> None:
        with self._cache_lock:
            self._images[version_id] = image
            original = self.versions[0].id
            while len(self._images) > 3:  # the original, plus the last couple of edits
                oldest = next(k for k in self._images if k not in (original, version_id))
                del self._images[oldest]

    def push(self, image: Image.Image, label: str, detail: str = "") -> Version:
        """Add a new version after the current one, discarding anything that was undone."""
        if image.mode not in ("RGB", "RGBA"):
            image = image.convert("RGBA" if "A" in image.mode else "RGB")
        for stale in self.versions[self.cursor + 1 :]:
            self._drop(stale)
        del self.versions[self.cursor + 1 :]
        version = Version(
            id=self._next_id,
            label=label,
            detail=detail,
            path=self.dir / f"v{self._next_id}.png",
            width=image.width,
            height=image.height,
        )
        self._next_id += 1
        # Fast, lossless working copy; the chosen output format applies on export only.
        image.save(version.path, "PNG", compress_level=1)
        self.versions.append(version)
        self.cursor = len(self.versions) - 1
        self._remember(version.id, image)
        return version

    def _drop(self, version: Version) -> None:
        version.path.unlink(missing_ok=True)
        with self._cache_lock:
            self._images.pop(version.id, None)
            self.scratch.pop(version.id, None)
            for key in [k for k in self._previews if k[0] == version.id]:
                del self._previews[key]

    def undo(self) -> bool:
        if self.cursor <= 0:
            return False
        self.cursor -= 1
        return True

    def redo(self) -> bool:
        if self.cursor >= len(self.versions) - 1:
            return False
        self.cursor += 1
        return True

    def goto(self, version_id: int) -> bool:
        for index, version in enumerate(self.versions):
            if version.id == version_id:
                self.cursor = index
                return True
        return False

    def find(self, version_id: int) -> Version | None:
        return next((v for v in self.versions if v.id == version_id), None)

    # ------------------------------------------------------------------ output

    @property
    def effective_format(self) -> str:
        """The format a download will really use. A JPEG cannot hold transparency, so a
        see-through picture is saved as PNG unless JPEG was asked for explicitly."""
        if (
            self.output_format == "jpeg"
            and not self.output_explicit
            and has_transparency(self.image())
        ):
            return "png"
        return self.output_format

    def set_output(self, fmt: str, quality: int | None = None) -> None:
        if fmt not in EXPORT_FORMATS:
            raise ValueError(f"Unsupported format {fmt!r}")
        self.output_format = fmt
        self.output_explicit = True
        if quality is not None:
            self.output_quality = max(1, min(100, int(quality)))

    def export(self) -> tuple[bytes, str, str]:
        """Encode the current version. Returns ``(data, filename, media type)``."""
        image = self.image()
        fmt = self.effective_format
        pil_format, media_type, extension = EXPORT_FORMATS[fmt]
        options: dict = {}
        if self.icc_profile:
            options["icc_profile"] = self.icc_profile
        if fmt == "jpeg":
            if image.mode == "RGBA":
                flat = Image.new("RGB", image.size, "white")
                flat.paste(image, mask=image.getchannel("A"))
                image = flat
            options.update(quality=self.output_quality, optimize=True)
        elif fmt == "webp":
            options.update(quality=self.output_quality, method=4)
        else:
            options.update(optimize=True)
        buffer = io.BytesIO()
        image.save(buffer, pil_format, **options)
        stem = Path(self.filename).stem or "picture"
        return buffer.getvalue(), f"{stem}-pictor{extension}", media_type

    def preview(self, version: Version, max_side: int) -> bytes:
        """A WebP copy of a version for the browser, no larger than ``max_side``."""
        key = (version.id, max_side)
        with self._cache_lock:
            cached = self._previews.get(key)
        if cached is None:
            image = self.image(version)
            if max(image.size) > max_side:
                image = image.copy()
                image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
            options: dict = {"quality": 90, "method": 2}
            if self.icc_profile:
                options["icc_profile"] = self.icc_profile
            buffer = io.BytesIO()
            image.save(buffer, "WEBP", **options)
            cached = buffer.getvalue()
            with self._cache_lock:
                if self.find(version.id):  # skip if it was undone away meanwhile
                    self._previews[key] = cached
        return cached

    # ------------------------------------------------------------------ state

    def state(self) -> dict:
        image = self.image()
        return {
            "id": self.id,
            "filename": self.filename,
            "cursor": self.cursor,
            "current": self.current.id,
            "can_undo": self.cursor > 0,
            "can_redo": self.cursor < len(self.versions) - 1,
            "width": image.width,
            "height": image.height,
            "transparent": has_transparency(image),
            "output": {"format": self.effective_format, "quality": self.output_quality},
            "versions": [v.to_dict() for v in self.versions],
        }


class SessionStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or cache_dir() / "sessions"
        self.root.mkdir(parents=True, exist_ok=True)
        self._sessions: dict[str, Session] = {}

    def cleanup(self) -> None:
        """Delete working folders left behind by earlier runs."""
        cutoff = time.time() - SESSION_TTL_SECONDS
        for folder in self.root.iterdir():
            try:
                if folder.is_dir() and folder.stat().st_mtime < cutoff:
                    shutil.rmtree(folder, ignore_errors=True)
            except OSError:
                continue

    def get(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    def create(self, data: bytes, filename: str) -> Session:
        if len(data) > MAX_UPLOAD_BYTES:
            raise ImageLoadError("That file is larger than 80 MB.")
        try:
            with Image.open(io.BytesIO(data)) as source:
                source_format = (source.format or "PNG").lower()
                icc_profile = source.info.get("icc_profile")
                exif = read_exif(source)
                image = normalize(source)
        except (UnidentifiedImageError, Image.DecompressionBombError, OSError, ValueError) as exc:
            raise ImageLoadError(
                "That file could not be opened as a picture. "
                "Supported: JPEG, PNG, WebP, HEIC, GIF, BMP, TIFF."
            ) from exc
        if max(image.size) > MAX_SIDE:
            image.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)

        session_id = uuid.uuid4().hex[:12]
        folder = self.root / session_id
        folder.mkdir(parents=True)
        fmt = {"jpg": "jpeg", "mpo": "jpeg"}.get(source_format, source_format)
        session = Session(
            id=session_id,
            dir=folder,
            filename=Path(filename or "picture").name,
            source_format=source_format,
            icc_profile=icc_profile if isinstance(icc_profile, bytes) else None,
            exif=exif,
            output_format=fmt if fmt in EXPORT_FORMATS else "png",
        )
        session.push(image, "Original")
        self._sessions[session_id] = session
        return session

    def discard(self, session_id: str) -> None:
        session = self._sessions.pop(session_id, None)
        if session:
            shutil.rmtree(session.dir, ignore_errors=True)
