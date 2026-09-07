from __future__ import annotations

import io
import json
import os
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from .contracts import MediaRef, storage_key
from .screen import ScreenFrame

ScreenEncoding = Literal["jpeg", "webp", "png"]
ScreenSource = Literal["screen", "camera"]

_FRAME_ID = re.compile(r"^[A-Za-z0-9_.:-]{1,256}$")
_SOURCES: frozenset[str] = frozenset({"screen", "camera"})
_ENCODING_EXTENSIONS: dict[str, str] = {
    "jpeg": ".jpg",
    "webp": ".webp",
    "png": ".png",
}
_ENCODING_FORMATS: dict[str, str] = {
    "jpeg": "JPEG",
    "webp": "WEBP",
    "png": "PNG",
}


@dataclass(frozen=True, slots=True)
class ScreenFrameHeader:
    """Metadata sent immediately before one encoded screen image."""

    frame_id: str
    captured_at_ms: int
    encoding: ScreenEncoding
    asset_id: str | None = None
    display_id: str | None = None
    scale_factor: float | None = None
    # Which capture surface produced this frame. The model treats both identically;
    # this only lets the backbrain describe what it is looking at truthfully. A typed
    # field rather than free-form metadata because the server routes on it.
    video_source: ScreenSource | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "ScreenFrameHeader":
        if payload.get("type") not in {None, "screen.frame"}:
            raise ValueError("screen frame metadata type must be 'screen.frame'")
        frame_id = str(payload.get("frame_id") or "")
        if not _FRAME_ID.fullmatch(frame_id):
            raise ValueError("invalid screen frame_id")
        if payload.get("captured_at_ms") is None:
            raise ValueError("captured_at_ms is required")
        captured_at_ms = int(payload["captured_at_ms"])
        if captured_at_ms < 0:
            raise ValueError("captured_at_ms must not be negative")
        encoding = str(payload.get("encoding") or "").lower()
        if encoding == "jpg":
            encoding = "jpeg"
        if encoding not in _ENCODING_EXTENSIONS:
            raise ValueError(f"unsupported screen encoding: {encoding!r}")
        metadata = payload.get("metadata") or {}
        if not isinstance(metadata, dict):
            raise ValueError("screen metadata must be an object")
        if len(json.dumps(metadata, ensure_ascii=False)) > 16_384:
            raise ValueError("screen metadata is too large")
        video_source = payload.get("video_source")
        if video_source is not None:
            video_source = str(video_source)
            if video_source not in _SOURCES:
                raise ValueError(f"unsupported screen video_source: {video_source!r}")
        return cls(
            frame_id=frame_id,
            captured_at_ms=captured_at_ms,
            encoding=encoding,  # type: ignore[arg-type]
            asset_id=_optional_id(payload.get("asset_id"), "asset_id"),
            display_id=_optional_text(payload.get("display_id"), "display_id", 256),
            scale_factor=_optional_positive_float(
                payload.get("scale_factor"), "scale_factor"
            ),
            video_source=video_source,  # type: ignore[arg-type]
            metadata=dict(metadata),
        )

    def to_metadata(self) -> dict[str, Any]:
        value = dict(self.metadata)
        value.update(
            {
                "frame_id": self.frame_id,
                "asset_id": self.asset_id or self.frame_id,
                "encoding": self.encoding,
            }
        )
        if self.display_id is not None:
            value["display_id"] = self.display_id
        if self.scale_factor is not None:
            value["scale_factor"] = self.scale_factor
        if self.video_source is not None:
            value["video_source"] = self.video_source
        return value


@dataclass(frozen=True, slots=True)
class DecodedScreenFrame:
    frame: ScreenFrame
    width: int
    height: int
    asset_id: str


def decode_screen_frame(
    header: ScreenFrameHeader,
    payload: bytes,
    *,
    max_bytes: int,
    max_pixels: int,
) -> DecodedScreenFrame:
    """Decode and fully validate one bounded raster image."""

    if not payload:
        raise ValueError("screen frame payload must not be empty")
    if len(payload) > max_bytes:
        raise ValueError(
            f"screen frame exceeds byte limit: {len(payload)} > {max_bytes}"
        )
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - installed with MiniCPM vision.
        raise RuntimeError("screen decoding requires Pillow") from exc

    try:
        with Image.open(io.BytesIO(payload)) as source:
            if source.format != _ENCODING_FORMATS[header.encoding]:
                raise ValueError(
                    f"screen payload format {source.format!r} does not match "
                    f"{header.encoding!r}"
                )
            width, height = source.size
            if width <= 0 or height <= 0 or width * height > max_pixels:
                raise ValueError(
                    f"screen dimensions exceed pixel limit: {width}x{height}"
                )
            source.load()
            image = source.convert("RGB")
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("invalid encoded screen frame") from exc

    asset_id = header.asset_id or header.frame_id
    metadata = header.to_metadata()
    metadata.update(width=width, height=height)
    return DecodedScreenFrame(
        frame=ScreenFrame(
            frame_id=header.frame_id,
            image=image,
            captured_at_ms=header.captured_at_ms,
            metadata=metadata,
        ),
        width=width,
        height=height,
        asset_id=asset_id,
    )


def persist_screen_frame(
    root: str | Path,
    session_id: str,
    header: ScreenFrameHeader,
    payload: bytes,
    decoded: DecodedScreenFrame,
    *,
    kind: Literal["screen", "frame"] = "screen",
) -> MediaRef:
    """Persist the encoded source for context/Codex without re-encoding it.

    ``kind`` distinguishes a shared screen from a camera frame so the backbrain
    never narrates a webcam image as the user's screen. The storage layout is
    shared regardless, so an existing cache stays valid.
    """

    directory = Path(root).expanduser().resolve() / storage_key(session_id) / "screen"
    directory.mkdir(parents=True, exist_ok=True)
    extension = _ENCODING_EXTENSIONS[header.encoding]
    # asset_id is client metadata, not immutable content identity. A client may
    # legitimately reuse it for a changing capture surface, so every accepted
    # frame gets its own path while the original ID stays in MediaRef metadata.
    filename = f"{header.captured_at_ms}_{uuid.uuid4().hex}{extension}"
    destination = directory / filename
    temporary = directory / f".{filename}.tmp"
    with temporary.open("xb") as handle:
        handle.write(payload)
    try:
        os.replace(temporary, destination)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    metadata = decoded.frame.metadata
    return MediaRef(
        kind=kind,
        path=str(destination),
        timestamp_ms=header.captured_at_ms,
        mime_type=f"image/{header.encoding}",
        metadata=dict(metadata),
    )


def _optional_id(value: Any, name: str) -> str | None:
    if value is None:
        return None
    text = str(value)
    if not _FRAME_ID.fullmatch(text):
        raise ValueError(f"invalid screen {name}")
    return text


def _optional_text(value: Any, name: str, limit: int) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or len(text) > limit:
        raise ValueError(f"invalid screen {name}")
    return text


def _optional_positive_float(value: Any, name: str) -> float | None:
    if value is None:
        return None
    number = float(value)
    if number <= 0:
        raise ValueError(f"{name} must be positive")
    return number
