"""Per-session media-mode policy for online duplex inference.

Pure policy functions combine model capability, deployment settings, client intent,
source warnings, and per-unit token cost. Transport and session state stay separate.
"""
from __future__ import annotations

from typing import Literal

MediaMode = Literal["voice", "omni", "auto"]
VideoSource = Literal["camera", "screen"]
VIDEO_SOURCES: tuple[VideoSource, ...] = ("camera", "screen")
CLIENT_VIDEO_MODES: tuple[MediaMode, ...] = ("omni", "auto")

# Duplex vision uses one unsliced frame with IMAGE_FEATURE_SIZE placeholders.
TOKENS_PER_FRAME = 66
# `<unit>` plus pooled Whisper positions for one second at 16 kHz.
TOKENS_PER_AUDIO_UNIT = 11

# Screen sharing is routed to the back brain; the front brain receives only the
# 448 px frame representation used for coarse visual context.
SCREEN_OUT_OF_DISTRIBUTION = "screen_content_out_of_distribution"


def vision_available(model: object) -> bool:
    """Report whether the loaded model exposes its vision encoder and resampler."""

    if model is None:
        return False
    return (
        getattr(model, "vpm", None) is not None
        and getattr(model, "resampler", None) is not None
    )


def client_video_allowed(
    *,
    vision_ok: bool,
    media_mode: str,
    allow_client_video: bool,
) -> bool:
    """Report whether a session may enable video through client control."""

    if not vision_ok:
        return False
    return media_mode != "voice" or bool(allow_client_video)


def resolve_target_mode(*, want_video: bool, client_video_mode: str) -> MediaMode:
    """Map client video intent to the configured model-side media mode."""

    if not want_video:
        return "voice"
    if client_video_mode not in CLIENT_VIDEO_MODES:
        raise ValueError(f"unsupported client_video_mode: {client_video_mode!r}")
    return client_video_mode  # type: ignore[return-value]


def source_warnings(source: object) -> tuple[str, ...]:
    """Return capability warnings for a video source."""

    return (SCREEN_OUT_OF_DISTRIBUTION,) if source == "screen" else ()


def estimated_tokens_per_unit(
    *,
    media_mode: str,
    speak_text_tokens_per_unit: int,
) -> int:
    """Estimate per-unit KV tokens for the selected media mode."""

    tokens = TOKENS_PER_AUDIO_UNIT + max(int(speak_text_tokens_per_unit), 0)
    if media_mode != "voice":
        tokens += TOKENS_PER_FRAME
    return tokens
