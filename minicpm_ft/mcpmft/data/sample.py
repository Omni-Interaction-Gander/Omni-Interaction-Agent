from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AudioRef:
    path: str | None = None
    sample_rate: int | None = None
    start_ms: int | None = None
    end_ms: int | None = None
    channel: int | None = None
    source: dict[str, Any] = field(default_factory=dict)
    _local_path: str | None = field(default=None, repr=False, compare=False)

    def id(self) -> str:
        if self.path:
            suffix = f":{self.start_ms or 0}-{self.end_ms or 'end'}"
            if self.channel is not None:
                suffix += f":ch{self.channel}"
            return self.path + suffix
        return "|".join(f"{k}={v}" for k, v in sorted(self.source.items()))

    def local_path(self) -> str | None:
        return self._local_path or self.path


@dataclass
class ImageRef:
    path: str
    start_ms: int | None = None
    end_ms: int | None = None
    source: dict[str, Any] = field(default_factory=dict)
    public_video: dict[str, Any] | None = None
    _local_path: str | None = field(default=None, repr=False, compare=False)

    def local_path(self) -> str:
        return self._local_path or self.path


@dataclass
class Capabilities:
    has_audio_in: bool = False
    has_text_out: bool = False
    has_speech_out: bool = False
    has_timestamps: bool = False
    multichannel: bool = False
    has_vision: bool = False
    has_omniflow: bool = False
    has_backbrain: bool = False
    has_tools: bool = False


@dataclass
class Turn:
    role: str
    text: str | None = None
    audio_in: AudioRef | None = None
    speech_out: AudioRef | None = None
    images: list[ImageRef] = field(default_factory=list)
    start_ms: int | None = None
    end_ms: int | None = None
    channel: int | None = None
    turn_state: str | None = None
    word_timestamps: list[tuple[str, int, int]] | None = None
    # Native MiniCPM tool protocol. Tool actions are assistant turns with one or more complete
    # calls; tool results are role=tool turns and carry their structured/string payload here.
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_response: Any | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class OmniSample:
    id: str
    turns: list[Turn]
    caps: Capabilities
    tools: list[dict[str, Any]] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _without_runtime_fields(dataclasses.asdict(self))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OmniSample":
        turns = [
            Turn(
                role=turn["role"],
                text=turn.get("text"),
                audio_in=_audio_ref(turn.get("audio_in")),
                speech_out=_audio_ref(turn.get("speech_out")),
                images=[ImageRef(**image) for image in turn.get("images", [])],
                start_ms=turn.get("start_ms"),
                end_ms=turn.get("end_ms"),
                channel=turn.get("channel"),
                turn_state=turn.get("turn_state"),
                word_timestamps=turn.get("word_timestamps"),
                tool_calls=list(turn.get("tool_calls") or []),
                tool_response=turn.get("tool_response"),
                meta=turn.get("meta", {}),
            )
            for turn in data.get("turns", [])
        ]
        caps_data = data.get("caps", {})
        return cls(
            id=data["id"],
            turns=turns,
            caps=Capabilities(**caps_data),
            tools=list(data.get("tools") or []),
            meta=data.get("meta", {}),
        )


def _audio_ref(value: dict[str, Any] | None) -> AudioRef | None:
    if not value:
        return None
    return AudioRef(**value)


def _without_runtime_fields(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _without_runtime_fields(item)
            for key, item in value.items()
            if key != "_local_path"
        }
    if isinstance(value, list):
        return [_without_runtime_fields(item) for item in value]
    return value


def infer_capabilities(
    turns: list[Turn],
    *,
    omniflow: bool = False,
    tools: list[dict[str, Any]] | None = None,
) -> Capabilities:
    return Capabilities(
        has_audio_in=any(turn.audio_in is not None for turn in turns),
        has_text_out=any(turn.role == "assistant" and bool(turn.text) for turn in turns),
        has_speech_out=any(turn.speech_out is not None for turn in turns),
        has_timestamps=any(turn.start_ms is not None and turn.end_ms is not None for turn in turns),
        multichannel=len({turn.channel for turn in turns if turn.channel is not None}) > 1,
        has_vision=any(turn.images for turn in turns),
        has_omniflow=omniflow,
        has_backbrain=False,
        has_tools=bool(tools) or any(
            turn.tool_calls or turn.role == "tool" or turn.tool_response is not None
            for turn in turns
        ),
    )



def sample_total_audio_ms(sample: "OmniSample") -> int:
    """Total audio duration (ms) across all turns' audio_in + speech_out.

    Used to drop over-long samples at the data layer (bounds OOM and shrinks per-rank length
    variance so variable-length batches stay roughly synchronized across ranks).
    Falls back to AudioRef.start_ms/end_ms span; turns without timing contribute 0.
    """
    total = 0
    for turn in sample.turns:
        for ref in (turn.audio_in, turn.speech_out):
            if ref is None:
                continue
            if turn.start_ms is not None and turn.end_ms is not None and turn.end_ms > turn.start_ms:
                total += turn.end_ms - turn.start_ms
            elif ref.start_ms is not None and ref.end_ms is not None and ref.end_ms > ref.start_ms:
                total += ref.end_ms - ref.start_ms
    return total
