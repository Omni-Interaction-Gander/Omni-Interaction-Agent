from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ScreenFrame:
    """A decoded screen image ready for visual prefill."""

    frame_id: str
    image: Any = field(repr=False, compare=False)
    captured_at_ms: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.frame_id, str):
            raise TypeError("frame_id must be a string")
        frame_id = self.frame_id.strip()
        if not frame_id or len(frame_id) > 256:
            raise ValueError("frame_id must contain 1 to 256 characters")
        object.__setattr__(self, "frame_id", frame_id)
        if self.image is None:
            raise ValueError("screen frame image must not be None")
        if self.captured_at_ms is not None and self.captured_at_ms < 0:
            raise ValueError("captured_at_ms must not be negative")


class LatestScreenFrameBuffer:
    """Capture-ordered screen buffer consumed at the MiniCPM unit clock.

    Producers publish independently of inference. Timestamped consumers select the newest
    frame available at audio-unit start; continuous visual mode may reuse the last frame.
    """

    def __init__(self, *, max_pending_frames: int = 128) -> None:
        if max_pending_frames < 1:
            raise ValueError("max_pending_frames must be positive")
        self._lock = threading.Lock()
        self.max_pending_frames = int(max_pending_frames)
        self._pending: list[ScreenFrame] = []
        self._last: ScreenFrame | None = None

    @staticmethod
    def _capture_order(frame: ScreenFrame) -> int:
        return -1 if frame.captured_at_ms is None else frame.captured_at_ms

    def publish(self, frame: ScreenFrame) -> None:
        if not isinstance(frame, ScreenFrame):
            raise TypeError("frame must be a ScreenFrame")
        with self._lock:
            self._pending.append(frame)
            overflow = len(self._pending) - self.max_pending_frames
            if overflow > 0:
                del self._pending[:overflow]

    def consume_for_unit(
        self,
        *,
        reuse_base: bool = False,
        captured_not_after_ms: float | None = None,
    ) -> tuple[ScreenFrame, ...]:
        """Atomically select frames for one model unit.

        ``reuse_base`` implements continuous visual mode: after the newest frame is
        consumed, static-screen units reuse it.
        """

        with self._lock:
            frame: ScreenFrame | None = None
            if captured_not_after_ms is None:
                if self._pending:
                    frame = self._pending[-1]
                    self._pending.clear()
            else:
                eligible = [
                    (index, pending)
                    for index, pending in enumerate(self._pending)
                    if self._capture_order(pending) <= captured_not_after_ms
                ]
                if eligible:
                    _, frame = max(
                        eligible,
                        key=lambda item: (self._capture_order(item[1]), item[0]),
                    )
                    self._pending = [
                        pending
                        for pending in self._pending
                        if self._capture_order(pending) > captured_not_after_ms
                    ]

            if frame is not None:
                if (
                    self._last is None
                    or self._capture_order(frame) >= self._capture_order(self._last)
                ):
                    self._last = frame
                else:
                    frame = None
            if frame is None and reuse_base:
                if (
                    self._last is not None
                    and (
                        captured_not_after_ms is None
                        or self._capture_order(self._last) <= captured_not_after_ms
                    )
                ):
                    frame = self._last
        return (frame,) if frame is not None else ()

    def reset(self) -> None:
        """Clear pending and reusable frames across media-mode transitions."""

        with self._lock:
            self._pending.clear()
            self._last = None


class ScreenFrameRateGate:
    """Keep at most one frame per capture-time interval."""

    def __init__(self, min_interval_ms: float) -> None:
        if min_interval_ms <= 0:
            raise ValueError("min_interval_ms must be positive")
        self.min_interval_ms = float(min_interval_ms)
        self._last_accepted_ms: int | None = None

    def accept(self, captured_at_ms: int) -> bool:
        if captured_at_ms < 0:
            raise ValueError("captured_at_ms must not be negative")
        previous = self._last_accepted_ms
        if previous is not None:
            if captured_at_ms <= previous:
                return False
            if captured_at_ms - previous < self.min_interval_ms:
                return False
        self._last_accepted_ms = captured_at_ms
        return True
