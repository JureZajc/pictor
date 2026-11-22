"""Face detection with OpenCV's YuNet model.

The model file (about 230 KB, MIT licensed, from the OpenCV Model Zoo) is fetched the
first time a face tool runs and checked against a pinned SHA-256 before use.
"""

from __future__ import annotations

import hashlib
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import httpx2 as httpx
import numpy as np
from PIL import Image

from pictor.config import cache_dir
from pictor.tools.registry import ToolError

YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
YUNET_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
DETECT_MAX_SIDE = 1280
SCORE_THRESHOLD = 0.6

_download_lock = threading.Lock()


@dataclass(frozen=True)
class Face:
    """A detected face as a pixel box in the picture, with the detector's confidence."""

    left: int
    top: int
    right: int
    bottom: int
    score: float


def model_path() -> Path:
    return cache_dir() / "models" / "face_detection_yunet_2023mar.onnx"


def _checksum_ok(path: Path) -> bool:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest() == YUNET_SHA256
    except OSError:
        return False


def ensure_model(status: Callable[[str], None] | None = None) -> Path:
    path = model_path()
    with _download_lock:
        if _checksum_ok(path):
            return path
        if status:
            status("Downloading the face detection model (230 KB, first use only)…")
        try:
            response = httpx.get(YUNET_URL, follow_redirects=True, timeout=60)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ToolError(
                "The face detection model could not be downloaded. It is fetched once from "
                f"GitHub and needs an internet connection. ({exc})"
            ) from exc
        if hashlib.sha256(response.content).hexdigest() != YUNET_SHA256:
            raise ToolError("The downloaded face detection model failed its integrity check.")
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        tmp.write_bytes(response.content)
        tmp.replace(path)
        return path


def detect_faces(image: Image.Image, status: Callable[[str], None] | None = None) -> list[Face]:
    """Find faces in a picture, largest first."""
    import cv2

    path = ensure_model(status)
    rgb = image.convert("RGB")
    scale = min(1.0, DETECT_MAX_SIDE / max(rgb.size))
    if scale < 1:
        rgb = rgb.resize(
            (max(1, round(rgb.width * scale)), max(1, round(rgb.height * scale))),
            Image.Resampling.BILINEAR,
        )
    bgr = np.ascontiguousarray(np.asarray(rgb)[:, :, ::-1])
    detector = cv2.FaceDetectorYN.create(
        str(path), "", (rgb.width, rgb.height), SCORE_THRESHOLD, 0.3, 5000
    )
    _, found = detector.detect(bgr)
    faces = []
    for row in found if found is not None else []:
        x, y, w, h = (float(v) / scale for v in row[:4])
        left = max(0, round(x))
        top = max(0, round(y))
        right = min(image.width, round(x + w))
        bottom = min(image.height, round(y + h))
        if right > left and bottom > top:
            faces.append(Face(left, top, right, bottom, float(row[-1])))
    faces.sort(key=lambda f: (f.right - f.left) * (f.bottom - f.top), reverse=True)
    return faces
