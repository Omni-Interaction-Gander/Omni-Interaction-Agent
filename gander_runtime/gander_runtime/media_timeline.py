from __future__ import annotations

from dataclasses import dataclass
from typing import Any


AUDIO_INPUT_PROTOCOL = "metadata-json+pcm16-binary-v1"


def _required_int(payload: dict[str, Any], name: str) -> int:
    value = payload.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"audio frame {name} must be an integer")
    return value


@dataclass(frozen=True, slots=True)
class AudioFrameHeader:
    """Metadata for the following PCM16 message; time names its first sample."""

    sequence: int
    start_sample: int
    sample_count: int
    captured_at_ms: int

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "AudioFrameHeader":
        if payload.get("type") != "audio.frame":
            raise ValueError("audio frame metadata type must be 'audio.frame'")
        header = cls(
            sequence=_required_int(payload, "sequence"),
            start_sample=_required_int(payload, "start_sample"),
            sample_count=_required_int(payload, "sample_count"),
            captured_at_ms=_required_int(payload, "captured_at_ms"),
        )
        if header.sequence <= 0:
            raise ValueError("audio frame sequence must be positive")
        if header.start_sample < 0:
            raise ValueError("audio frame start_sample must not be negative")
        if header.sample_count <= 0:
            raise ValueError("audio frame sample_count must be positive")
        if header.captured_at_ms < 0:
            raise ValueError("audio frame captured_at_ms must not be negative")
        return header


@dataclass(frozen=True, slots=True)
class TimedPcm16Part:
    data: bytes
    unit_capture_start_ms: tuple[float, ...]


class AudioCaptureTimeline:
    """Map ordered browser PCM frames onto the model's fixed-duration units."""

    def __init__(self, *, sample_rate: int, unit_ms: int) -> None:
        if sample_rate <= 0 or unit_ms <= 0:
            raise ValueError("audio timeline sample_rate and unit_ms must be positive")
        self.sample_rate = int(sample_rate)
        self.unit_samples = int(self.sample_rate * unit_ms / 1000)
        if self.unit_samples <= 0:
            raise ValueError("audio timeline unit must contain at least one sample")
        self.reset()

    def reset(self) -> None:
        self._next_sequence = 1
        self._next_sample = 0
        self._next_unit_start_sample = 0
        self._next_unit_end_sample = self.unit_samples
        self._unit_capture_start_ms: float | None = None

    @property
    def pending_unit_capture_start_ms(self) -> float | None:
        """Capture origin for the incomplete model unit, if one exists."""

        return self._unit_capture_start_ms

    def split_frame(
        self,
        header: AudioFrameHeader,
        pcm16: bytes,
        *,
        max_part_bytes: int,
    ) -> tuple[TimedPcm16Part, ...]:
        if len(pcm16) != header.sample_count * 2:
            raise ValueError("audio frame sample_count does not match its PCM16 payload")
        if header.sequence != self._next_sequence:
            raise ValueError(
                f"audio frame sequence must be {self._next_sequence}, got {header.sequence}"
            )
        if header.start_sample != self._next_sample:
            raise ValueError(
                f"audio frame start_sample must be {self._next_sample}, "
                f"got {header.start_sample}"
            )
        if max_part_bytes <= 0 or max_part_bytes % 2:
            raise ValueError("max_part_bytes must be a positive PCM16 byte count")

        frame_end = header.start_sample + header.sample_count
        parts: list[TimedPcm16Part] = []
        for byte_offset in range(0, len(pcm16), max_part_bytes):
            data = pcm16[byte_offset : byte_offset + max_part_bytes]
            part_start = header.start_sample + byte_offset // 2
            part_end = part_start + len(data) // 2
            if (
                self._unit_capture_start_ms is None
                and part_start <= self._next_unit_start_sample < part_end
            ):
                self._unit_capture_start_ms = (
                    header.captured_at_ms
                    + (self._next_unit_start_sample - header.start_sample)
                    * 1000.0
                    / self.sample_rate
                )

            unit_starts: list[float] = []
            while self._next_unit_end_sample <= part_end:
                if self._unit_capture_start_ms is None:
                    raise RuntimeError("audio unit has no capture-time origin")
                unit_starts.append(self._unit_capture_start_ms)
                self._next_unit_start_sample = self._next_unit_end_sample
                self._next_unit_end_sample += self.unit_samples
                if self._next_unit_start_sample < frame_end:
                    self._unit_capture_start_ms = (
                        header.captured_at_ms
                        + (self._next_unit_start_sample - header.start_sample)
                        * 1000.0
                        / self.sample_rate
                    )
                else:
                    self._unit_capture_start_ms = None
            parts.append(TimedPcm16Part(data, tuple(unit_starts)))

        self._next_sequence += 1
        self._next_sample += header.sample_count
        return tuple(parts)
