from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Sequence

from mcpmft.data.media import (
    is_public_video_reference,
    parse_public_video_reference,
)
from mcpmft.data.sample import ImageRef


def _pillow():
    try:
        from PIL import Image, ImageOps
    except ImportError as exc:
        raise RuntimeError("Vision training requires Pillow") from exc
    return Image, ImageOps


def _load_image(path: Path):
    Image, ImageOps = _pillow()
    if not path.is_file():
        raise FileNotFoundError(f"image frame does not exist: {path}")
    try:
        with Image.open(path) as image:
            return ImageOps.exif_transpose(image).convert("RGB").copy()
    except Exception as exc:
        raise ValueError(f"failed to load image frame {path}") from exc


def _frame_time(frame, stream) -> float:
    if frame.time is not None:
        return float(frame.time)
    if frame.pts is None:
        raise ValueError("video frame has neither time nor PTS")
    return float(frame.pts * (frame.time_base or stream.time_base))


def _load_video_frames(path: Path, times_ms: Sequence[int]) -> dict[int, object]:
    if not path.is_file():
        raise FileNotFoundError(f"public source video does not exist: {path}")
    try:
        import av
    except ImportError as exc:
        raise RuntimeError("Public video training requires PyAV") from exc

    targets = sorted(set(int(value) for value in times_ms))
    images: dict[int, object] = {}
    target_index = 0
    previous = None
    previous_time = None
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError(f"public source has no video track: {path}")
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            current_time = _frame_time(frame, stream)
            while (
                target_index < len(targets)
                and current_time * 1000 >= targets[target_index]
            ):
                target = targets[target_index]
                chosen = frame
                if previous is not None and previous_time is not None:
                    before = abs(previous_time * 1000 - target)
                    after = abs(current_time * 1000 - target)
                    if before <= after:
                        chosen = previous
                images[target] = chosen.to_image().convert("RGB")
                target_index += 1
            previous = frame
            previous_time = current_time
            if target_index == len(targets):
                break

    if previous is None:
        raise ValueError(f"public source has an empty video track: {path}")
    while target_index < len(targets):
        target = targets[target_index]
        images[target] = previous.to_image().convert("RGB")
        target_index += 1
    return images


def load_image_refs(image_refs: Sequence[ImageRef]) -> list[object]:
    """Load image files and batch all requested frames from each public source video."""

    loaded: list[object | None] = [None] * len(image_refs)
    video_groups: dict[Path, list[tuple[int, int]]] = defaultdict(list)
    for index, image_ref in enumerate(image_refs):
        if not is_public_video_reference(image_ref.path):
            loaded[index] = _load_image(Path(image_ref.local_path()).expanduser())
            continue
        reference = parse_public_video_reference(image_ref.path)
        if reference.kind != "frame" or reference.frame_ms is None:
            raise ValueError(f"image reference points to video audio: {image_ref.path}")
        if image_ref._local_path is None:
            raise ValueError("public:// image references require data.public_video_root")
        video_groups[Path(image_ref._local_path)].append((index, reference.frame_ms))

    for path, requests in video_groups.items():
        frames = _load_video_frames(path, [time_ms for _, time_ms in requests])
        for index, time_ms in requests:
            loaded[index] = frames[time_ms].copy()
    if any(image is None for image in loaded):
        raise RuntimeError("failed to load every image reference")
    return [image for image in loaded if image is not None]


def load_image_ref(image_ref: ImageRef):
    """Load one image reference; collators should prefer ``load_image_refs``."""

    return load_image_refs([image_ref])[0]
