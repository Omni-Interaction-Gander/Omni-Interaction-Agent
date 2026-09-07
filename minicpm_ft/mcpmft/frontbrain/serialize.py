from __future__ import annotations

from mcpmft.data.feature import AudioGeometry
from mcpmft.data.sample import OmniSample
from mcpmft.data.serialize_turn import SerializedSample
from mcpmft.prompts import GANDER_DUPLEX_SYSTEM_PROMPT


def serialize_frontbrain_sample(
    sample: OmniSample,
    tokenizer,
    *,
    max_seq_length: int = 4096,
    block_ms: int = 1000,
    audio_geometry: AudioGeometry | None = None,
    truncate_block: int | None = None,
    turn_gap_ms: int = 2000,
    text_tokens_per_block: int = 4,
    codes_per_text_token: int = 6,
    speech_tokens_per_unit: int = 25,
    include_system_prompt: bool = True,
    system_prompt: str = GANDER_DUPLEX_SYSTEM_PROMPT,
    pinned_context: str | None = None,
) -> SerializedSample:
    """Serialize the realtime front brain with MiniCPM's native tool protocol.

    Listen, speak, backchannel and interrupt retain their existing duplex tokens. A tool call is a
    separate silent action branch containing complete ``<tool_call>`` JSON and ``<|chunk_eos|>``.
    Tool results are masked ``<tool_response>`` context appended after the continuous media in a
    later unit. Only the native MiniCPM interaction and tool tokens are emitted;
    every backend action is an ordinary, explicitly versioned function declared in ``sample.tools``.
    """
    from mcpmft.data.serialize_duplex import serialize_duplex_sample

    resolved_pinned_context = (
        sample.meta.get("pinned_context", "")
        if pinned_context is None
        else pinned_context
    )
    if not isinstance(resolved_pinned_context, str):
        raise TypeError("front-brain pinned_context must be a string")

    serialized = serialize_duplex_sample(
        sample,
        tokenizer,
        max_seq_length=max_seq_length,
        block_ms=block_ms,
        audio_geometry=audio_geometry,
        truncate_block=truncate_block,
        turn_gap_ms=turn_gap_ms,
        text_tokens_per_block=text_tokens_per_block,
        codes_per_text_token=codes_per_text_token,
        speech_tokens_per_unit=speech_tokens_per_unit,
        include_system_prompt=include_system_prompt,
        system_prompt=system_prompt,
        pinned_context=resolved_pinned_context,
    )
    if pinned_context is not None and not resolved_pinned_context:
        # The source row may retain a static SLATE. An explicit empty
        # override is a training-view decision, so stale source metadata must not make sliding
        # context reserve nonexistent pinned tokens.
        serialized.meta.pop("pinned_context", None)
        serialized.meta.pop("pinned_context_tokens", None)
        serialized.meta["sample_pinned_context_disabled"] = True
    serialized.meta["paradigm"] = "frontbrain_native_tools"
    return serialized
