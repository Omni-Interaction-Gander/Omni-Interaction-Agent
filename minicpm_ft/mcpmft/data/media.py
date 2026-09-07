from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlsplit

from mcpmft.data.sample import OmniSample


@dataclass(frozen=True)
class PublicVideoReference:
    locator: str
    namespace: str
    relative_path: Path
    kind: str
    frame_ms: int | None = None


def parse_public_video_reference(value: str) -> PublicVideoReference:
    parsed = urlsplit(value)
    if parsed.scheme != "public" or not parsed.netloc:
        raise ValueError(f"Not a public video reference: {value!r}")
    parts = [unquote(part) for part in parsed.path.split("/") if part]
    if not parts or any(part in {".", ".."} for part in parts):
        raise ValueError(f"Invalid public video path: {value!r}")

    frame_ms = None
    if parsed.fragment == "audio":
        kind = "audio"
    elif parsed.fragment.startswith("frame_ms="):
        kind = "frame"
        try:
            frame_ms = int(parsed.fragment.split("=", 1)[1])
        except ValueError as exc:
            raise ValueError(f"Invalid public video frame time: {value!r}") from exc
        if frame_ms < 0:
            raise ValueError(f"Invalid public video frame time: {value!r}")
    else:
        raise ValueError(
            f"Public video reference must end in #audio or #frame_ms=<time>: {value!r}"
        )

    locator = f"public://{parsed.netloc}{parsed.path}"
    return PublicVideoReference(
        locator=locator,
        namespace=parsed.netloc,
        relative_path=Path(*parts),
        kind=kind,
        frame_ms=frame_ms,
    )


def is_public_video_reference(value: str | None) -> bool:
    return bool(value and value.startswith("public://"))


class ReleaseMediaResolver:
    """Resolve published relative paths without changing their stable manifest identity."""

    def __init__(
        self,
        *,
        release_root: str | Path | None = None,
        public_video_root: str | Path | None = None,
    ) -> None:
        self.release_root = (
            Path(release_root).expanduser() if release_root is not None else None
        )
        self.public_video_root = (
            Path(public_video_root).expanduser()
            if public_video_root is not None
            else None
        )

    def resolve_path(self, value: str) -> str:
        if is_public_video_reference(value):
            if self.public_video_root is None:
                raise ValueError(
                    "The manifest contains public:// video references but "
                    "data.public_video_root is not configured"
                )
            reference = parse_public_video_reference(value)
            return str(
                self.public_video_root
                / reference.namespace
                / reference.relative_path
            )

        path = Path(value).expanduser()
        if path.is_absolute() or self.release_root is None:
            return str(path)
        return str(self.release_root / path)

    def resolve_sample(self, sample: OmniSample) -> OmniSample:
        for turn in sample.turns:
            for audio_ref in (turn.audio_in, turn.speech_out):
                if audio_ref is not None and audio_ref.path:
                    audio_ref._local_path = self.resolve_path(audio_ref.path)
            for image_ref in turn.images:
                image_ref._local_path = self.resolve_path(image_ref.path)
        return sample
