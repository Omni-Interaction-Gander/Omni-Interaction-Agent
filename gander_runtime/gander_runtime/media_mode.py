"""Per-session media-mode policy for the online duplex service.

``duplex.media_mode`` selects the mode a session starts in. When
``duplex.allow_client_video`` is enabled, a session may turn vision on or off
itself
through the ``media.mode`` control event. This module owns every decision that
choice implies, as pure functions, so the transport layer stays a thin dispatch:

  - is client-driven video permitted at all (vision tower + operator policy)
  - which model-side mode a "video on/off" intent maps to
  - what the caller must be warned about
  - what the mode costs per unit

Nothing here touches the model, the network, or session state.
"""
from __future__ import annotations

from typing import Literal

MediaMode = Literal["voice", "omni", "auto"]
VideoSource = Literal["camera", "screen"]
VIDEO_SOURCES: tuple[VideoSource, ...] = ("camera", "screen")
CLIENT_VIDEO_MODES: tuple[MediaMode, ...] = ("omni", "auto")

# One frame serializes to <image> + IMAGE_FEATURE_SIZE placeholders + </image>,
# matching mcpmft.data.serialize_duplex. The duplex path fixes max_slice_nums=1,
# so a frame always costs exactly this, whatever its resolution.
TOKENS_PER_FRAME = 66
# <unit> plus the pooled whisper placeholders for one 1s chunk at 16 kHz.
TOKENS_PER_AUDIO_UNIT = 11

# The frontbrain has never seen screen recordings or desktop UI: every video row
# in the training mix is natural footage, and a whole frame is 64 tokens at
# 448px. Small on-screen text is unreadable, so screen share is a backbrain
# capability that the frontbrain only sees the gist of.
SCREEN_OUT_OF_DISTRIBUTION = "screen_content_out_of_distribution"


def vision_available(model: object) -> bool:
    """Report whether the loaded model can actually embed a frame.

    ``init_vision: false`` (the audio-only serving config) skips building
    ``vpm``/``resampler`` entirely, so feeding a frame would raise
    ``AttributeError`` on the model thread. Probing the instance is the only
    honest signal: the config that built it is not carried on the bundle.

    """

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
    """Report whether a session may turn video on by itself.

    A session that already starts in a vision mode keeps that capability. An
    audio-first deployment must opt in explicitly, so ``media_mode: voice``
    alone never gains a video surface.
    """

    if not vision_ok:
        return False
    return media_mode != "voice" or bool(allow_client_video)


def resolve_target_mode(*, want_video: bool, client_video_mode: str) -> MediaMode:
    """Map a client's video on/off intent to a model-side media mode.

    Clients send intent rather than a mode so policy stays server-side and a
    client can never select a costlier mode than the operator allowed.
    """

    if not want_video:
        return "voice"
    if client_video_mode not in CLIENT_VIDEO_MODES:
        raise ValueError(f"unsupported client_video_mode: {client_video_mode!r}")
    return client_video_mode  # type: ignore[return-value]


def source_warnings(source: object) -> tuple[str, ...]:
    """Return the honest caveats for one video source."""

    return (SCREEN_OUT_OF_DISTRIBUTION,) if source == "screen" else ()


def estimated_tokens_per_unit(
    *,
    media_mode: str,
    speak_text_tokens_per_unit: int,
) -> int:
    """Estimate the KV cost of one unit, so the caller can surface it.

    Context eviction is unit-counted, not token-counted, so a vision session
    silently multiplies the KV window. Reporting the number lets an operator
    lower ``duplex.context_max_units`` deliberately instead of discovering the cost
    as latency.
    """

    tokens = TOKENS_PER_AUDIO_UNIT + max(int(speak_text_tokens_per_unit), 0)
    if media_mode != "voice":
        tokens += TOKENS_PER_FRAME
    return tokens
