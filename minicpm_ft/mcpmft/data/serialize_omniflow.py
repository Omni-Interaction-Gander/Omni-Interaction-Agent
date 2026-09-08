from __future__ import annotations

from mcpmft.data.feature import AudioGeometry
from mcpmft.data.sample import OmniSample
from mcpmft.data.serialize_turn import SerializedSample
from mcpmft.prompts import GANDER_DUPLEX_SYSTEM_PROMPT
from mcpmft.tokenizer_tools import INTERRUPT, LISTEN, SPEAK


BACKCHANNEL_CLASS = "backchannel"


def turn_state_to_control(state: str | None, *, is_agent_turn: bool) -> str:
    """Map a normalized turn state to the agent control decision for a block.

    Complete maps to speak. Incomplete content, wait, and silence remain listen decisions;
    backchannels retain their own class.
    """
    normalized = (state or "").strip("<|>").lower()
    if not is_agent_turn:
        return "listen"
    if normalized == "backchannel":
        return BACKCHANNEL_CLASS
    if normalized == "complete":
        return "speak"
    # Non-speaking states map to listen.
    return "listen"


def control_token_id(control: str, tokenizer) -> int:
    """Resolve the supervised control-token id for a block."""
    if control == BACKCHANNEL_CLASS:
        return int(tokenizer.convert_tokens_to_ids("<|backchannel|>"))
    if control == "speak":
        return SPEAK.token_id
    if control == "interrupt":
        return INTERRUPT.token_id
    return LISTEN.token_id


def serialize_omniflow_sample(
    sample: OmniSample,
    tokenizer,
    *,
    max_seq_length: int = 4096,
    block_ms: int = 1000,
    audio_geometry: AudioGeometry | None = None,
    turn_gap_ms: int = 2000,
    text_tokens_per_block: int = 4,
    codes_per_text_token: int = 6,
    speech_tokens_per_unit: int = 25,
    include_system_prompt: bool = True,
    system_prompt: str = GANDER_DUPLEX_SYSTEM_PROMPT,
) -> SerializedSample:
    """Serialize single-model full-duplex interaction with an always-on microphone.

    This wrapper delegates to ``serialize_duplex_sample``. Each one-second unit carries
    perception, control, optional assistant text, boundaries, and aligned Talker targets.
    """
    from mcpmft.data.serialize_duplex import serialize_duplex_sample

    return serialize_duplex_sample(
        sample,
        tokenizer,
        max_seq_length=max_seq_length,
        block_ms=block_ms,
        audio_geometry=audio_geometry,
        turn_gap_ms=turn_gap_ms,
        text_tokens_per_block=text_tokens_per_block,
        codes_per_text_token=codes_per_text_token,
        speech_tokens_per_unit=speech_tokens_per_unit,
        include_system_prompt=include_system_prompt,
        system_prompt=system_prompt,
    )


def unit_ids_from_boundaries(input_ids: list[int], unit_start_id: int) -> list[int]:
    """Tag each token with its unit index: a new unit begins at every <unit> token.

    Used by the optional KV-delete mask. Tokens before the first <unit> are marked as -2 =
    protected prefix/system, matching MiniCPM-o online inference which keeps the initial system
    prompt outside eviction. Padding remains -1 in the collator.
    """
    unit_ids: list[int] = []
    cur = 0
    started = False
    for tid in input_ids:
        if tid == unit_start_id:
            cur = cur + 1 if started else 0
            started = True
        unit_ids.append(cur if started else -2)
    return unit_ids


def duplex_system_prompt_prefix(system_prompt: str) -> str:
    """MiniCPMODuplex.prepare() prefix before the context-mode `previous:` insertion point."""
    return f"<|im_start|>system\n{system_prompt}"


def duplex_system_prompt_suffix() -> str:
    """MiniCPMODuplex.prepare() suffix protected after context-mode `previous:`."""
    return "<|im_end|>"


def _format_duplex_system_prompt(system_prompt: str) -> str:
    """Exact MiniCPMODuplex.prepare() text form without reference-audio prompt."""
    return duplex_system_prompt_prefix(system_prompt) + duplex_system_prompt_suffix()


def _is_complete(state: str | None) -> bool:
    """True for a natural completion; interruption closure is carried separately in turn meta."""
    return (state or "").strip("<|>").lower() == "complete"
