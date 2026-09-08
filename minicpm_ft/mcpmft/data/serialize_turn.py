from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from mcpmft.data.feature import AudioGeometry, audio_placeholder_for_duration_ms
from mcpmft.data.labels import labels_from_assistant_spans
from mcpmft.data.sample import OmniSample, Turn
from mcpmft.tool_protocol import (
    format_tool_calls,
    format_tool_response,
    system_prompt_with_tools,
)
from mcpmft.tokenizer_tools import IMAGE_END, IMAGE_FEATURE_SIZE, IMAGE_START


@dataclass
class SpeechSegment:
    batch_index: int | None
    text_token_positions: list[int]
    text_token_ids: list[int]
    audio_ref_id: str
    s3_codes: list[int] | None = None
    turn_group: int = 0  # shared Talker context
    is_turn_final: bool = True  # final duplex unit resets Talker KV
    should_predict_audio_eos: bool = True
    unit_index: int | None = None  # streaming order
    real_text_token_count: int | None = None  # excludes control tokens
    unit_start_ms: int | None = None
    unit_end_ms: int | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class SerializedSample:
    id: str
    input_ids: list[int]
    labels: list[int]
    image_bounds: list[tuple[int, int]] = field(default_factory=list)
    image_inputs: list[Any] = field(default_factory=list)  # aligned with image_bounds
    audio_bounds: list[tuple[int, int]] = field(default_factory=list)
    audio_inputs: list[Any] = field(default_factory=list)  # aligned with audio_bounds
    speech_segments: list[SpeechSegment] = field(default_factory=list)
    unit_ids: list[int] = field(default_factory=list)  # per-token streaming unit
    meta: dict[str, Any] = field(default_factory=dict)


def serialize_turn_sample(
    sample: OmniSample,
    tokenizer,
    *,
    max_seq_length: int = 4096,
    audio_geometry: AudioGeometry | None = None,
    enable_thinking: bool = False,
) -> SerializedSample:
    """Serialize a sample into a turn-level SFT sequence.

    The implementation intentionally keeps labels unshifted. The training wrapper applies the
    only causal shift when computing text CE.
    """
    input_ids: list[int] = []
    assistant_spans: list[tuple[int, int]] = []
    image_bounds: list[tuple[int, int]] = []
    image_inputs: list[Any] = []
    audio_bounds: list[tuple[int, int]] = []
    audio_inputs: list[Any] = []
    speech_segments: list[SpeechSegment] = []
    structured_spans: list[tuple[int, int]] = []
    turn_group = -1

    turns = list(sample.turns)
    start_index = 0
    if sample.tools:
        base_system = ""
        if turns and turns[0].role == "system":
            base_system = turns[0].text or ""
            start_index = 1
        tool_system = system_prompt_with_tools(base_system, sample.tools)
        input_ids.extend(_encode(tokenizer, "<|im_start|>system\n" + tool_system))
        input_ids.append(_im_end_id(tokenizer))
        input_ids.extend(_encode(tokenizer, "\n"))

    for turn_index in range(start_index, len(turns)):
        turn = turns[turn_index]
        if turn.role == "tool":
            if turn.audio_in is not None or turn.speech_out is not None or turn.images:
                raise ValueError(f"tool response turn cannot carry media: sample={sample.id!r}")
            previous_is_tool = turn_index > start_index and turns[turn_index - 1].role == "tool"
            next_is_tool = turn_index + 1 < len(turns) and turns[turn_index + 1].role == "tool"
            if not previous_is_tool:
                input_ids.extend(_encode(tokenizer, "<|im_start|>user"))
            response = (
                turn.tool_response if turn.tool_response is not None else (turn.text or "")
            )
            input_ids.extend(_encode(tokenizer, "\n" + format_tool_response(response)))
            if not next_is_tool:
                input_ids.append(_im_end_id(tokenizer))
                input_ids.extend(_encode(tokenizer, "\n"))
            continue
        if turn.role == "assistant" and turn.images:
            raise ValueError(
                f"assistant image output is not supported: sample={sample.id!r}"
            )

        prefix = _encode(tokenizer, _role_prefix(turn.role))
        input_ids.extend(prefix)
        # Match the empty think prefix used by offline and online inference when
        # enable_thinking is false; the prefix itself is not supervised.
        if turn.role == "assistant" and not enable_thinking:
            input_ids.extend(_encode(tokenizer, "<think>\n\n</think>\n\n"))
        content_start = len(input_ids)

        for image_ref in turn.images:
            input_ids.append(IMAGE_START.token_id)
            image_start = len(input_ids)
            input_ids.extend([_unk_id(tokenizer)] * IMAGE_FEATURE_SIZE)
            image_bounds.append((image_start, len(input_ids)))
            image_inputs.append(image_ref)
            input_ids.append(IMAGE_END.token_id)

        if turn.audio_in is not None:
            placeholder = _audio_placeholder(turn, audio_geometry)
            audio_start = len(input_ids) + 1
            input_ids.extend(placeholder)
            audio_end = len(input_ids) - 1
            audio_bounds.append((audio_start, audio_end))
            audio_inputs.append(turn.audio_in)

        text_ids = _encode(tokenizer, turn.text or "")
        text_start = len(input_ids)
        input_ids.extend(text_ids)
        if turn.tool_calls:
            if turn.role != "assistant":
                raise ValueError(
                    f"only assistant turns may emit tool calls: sample={sample.id!r}"
                )
            if turn.speech_out is not None:
                raise ValueError(
                    f"silent tool-call turn cannot carry speech_out: sample={sample.id!r}"
                )
            if text_ids:
                input_ids.extend(_encode(tokenizer, "\n"))
            structured_start = len(input_ids)
            input_ids.extend(_encode(tokenizer, format_tool_calls(turn.tool_calls)))
            structured_spans.append((structured_start, len(input_ids)))
        # Assistant supervision includes the chat-template turn terminator.
        im_end_id = _im_end_id(tokenizer)
        input_ids.append(im_end_id)
        content_end = len(input_ids)
        input_ids.extend(_encode(tokenizer, "\n"))

        if turn.role == "assistant" and (text_ids or turn.tool_calls):
            # The supervised span includes content and <|im_end|>.
            assistant_spans.append((content_start, content_end))
            turn_group += 1
            if turn.speech_out is not None:
                speech_segments.append(
                    SpeechSegment(
                        batch_index=None,
                        text_token_positions=list(range(text_start, text_start + len(text_ids))),
                        text_token_ids=text_ids,
                        audio_ref_id=turn.speech_out.id(),
                        s3_codes=turn.meta.get("s3_codes"),
                        turn_group=turn_group,
                        meta={"turn_meta": turn.meta},
                    )
                )

    if len(input_ids) > max_seq_length:
        offset = _left_truncation_offset(input_ids, max_seq_length, _im_start_id(tokenizer))
        if any(start < offset < end for start, end in structured_spans):
            raise ValueError(
                "max_seq_length would cut through a complete tool call: "
                f"sample={sample.id!r}, offset={offset}"
            )
        input_ids = input_ids[offset:]
        assistant_spans = _shift_spans(assistant_spans, -offset, len(input_ids))
        image_bounds, image_inputs = _shift_image_pairs(image_bounds, image_inputs, offset)
        audio_bounds, audio_inputs = _shift_audio_pairs(audio_bounds, audio_inputs, offset)
        speech_segments = _shift_complete_speech_segments(speech_segments, offset)

    labels = labels_from_assistant_spans(input_ids, assistant_spans)
    return SerializedSample(
        id=sample.id,
        input_ids=input_ids,
        labels=labels,
        image_bounds=image_bounds,
        image_inputs=image_inputs,
        audio_bounds=audio_bounds,
        audio_inputs=audio_inputs,
        speech_segments=[seg for seg in speech_segments if seg.text_token_positions],
        meta={"caps": sample.caps.__dict__, **sample.meta},
    )


def _role_prefix(role: str) -> str:
    if role == "system":
        return "<|im_start|>system\n"
    if role == "assistant":
        return "<|im_start|>assistant\n"
    return "<|im_start|>user\n"


def _encode(tokenizer, text: str) -> list[int]:
    if not text:
        return []
    return tokenizer.encode(text, add_special_tokens=False)


def _im_end_id(tokenizer) -> int:
    tid = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if tid is None or tid < 0:
        tid = tokenizer.eos_token_id
    return tid


def _im_start_id(tokenizer) -> int:
    return int(tokenizer.convert_tokens_to_ids("<|im_start|>"))


def _unk_id(tokenizer) -> int:
    value = getattr(tokenizer, "unk_token_id", None)
    if value is None:
        value = tokenizer.convert_tokens_to_ids("<unk>")
    if value is None or int(value) < 0:
        raise ValueError("tokenizer does not expose the MiniCPM-o <unk> token")
    return int(value)


def _audio_placeholder(turn: Turn, geometry: AudioGeometry | None) -> list[int]:
    from mcpmft.data.feature import audio_placeholder_len
    from mcpmft.tokenizer_tools import AUDIO_END, AUDIO_START

    geometry = geometry or AudioGeometry()
    num_samples = turn.audio_in.meta.get("num_samples") if turn.audio_in and isinstance(getattr(turn.audio_in, "meta", None), dict) else None
    if num_samples is None and turn.audio_in is not None and turn.audio_in.path:
        # Derive placeholder length from the same sliced, resampled waveform used by
        # the collator so audio bounds and pooled features remain aligned.
        try:
            from mcpmft.data.feature import load_audio_ref_waveform

            wav = load_audio_ref_waveform(turn.audio_in)
            num_samples = int(len(wav))
        except Exception:
            num_samples = None
    if num_samples is not None:
        n = audio_placeholder_len(num_samples, geometry)
        return [AUDIO_START.token_id] + [0] * n + [AUDIO_END.token_id]
    duration_ms = 1000
    if turn.audio_in is not None and turn.audio_in.start_ms is not None and turn.audio_in.end_ms is not None:
        duration_ms = max(turn.audio_in.end_ms - turn.audio_in.start_ms, 1)
    return audio_placeholder_for_duration_ms(duration_ms, geometry)


def _shift_spans(spans: list[tuple[int, int]], delta: int, max_len: int) -> list[tuple[int, int]]:
    shifted: list[tuple[int, int]] = []
    for start, end in spans:
        start = max(start + delta, 0)
        end = min(end + delta, max_len)
        if end > start:
            shifted.append((start, end))
    return shifted


def _left_truncation_offset(input_ids: list[int], max_length: int, boundary_id: int) -> int:
    """Prefer dropping a complete turn/unit instead of cutting through structured content."""
    overflow = len(input_ids) - max_length
    if overflow <= 0:
        return 0
    return next(
        (index for index in range(overflow, len(input_ids)) if input_ids[index] == boundary_id),
        overflow,
    )


def _shift_audio_pairs(
    bounds: list[tuple[int, int]],
    refs: list[Any],
    offset: int,
) -> tuple[list[tuple[int, int]], list[Any]]:
    return _shift_complete_media_pairs(bounds, refs, offset, media="audio")


def _shift_image_pairs(
    bounds: list[tuple[int, int]],
    refs: list[Any],
    offset: int,
) -> tuple[list[tuple[int, int]], list[Any]]:
    return _shift_complete_media_pairs(bounds, refs, offset, media="image")


def _shift_complete_media_pairs(
    bounds: list[tuple[int, int]],
    refs: list[Any],
    offset: int,
    *,
    media: str,
) -> tuple[list[tuple[int, int]], list[Any]]:
    if len(bounds) != len(refs):
        raise ValueError(
            f"{media}_bounds/{media}_inputs mismatch before truncation: "
            f"{len(bounds)} != {len(refs)}"
        )
    shifted_bounds: list[tuple[int, int]] = []
    shifted_refs: list[Any] = []
    for (start, end), ref in zip(bounds, refs):
        if start < offset:
            continue
        shifted_bounds.append((start - offset, end - offset))
        shifted_refs.append(ref)
    return shifted_bounds, shifted_refs


def _shift_complete_speech_segments(
    segments: list[SpeechSegment],
    offset: int,
) -> list[SpeechSegment]:
    shifted: list[SpeechSegment] = []
    for segment in segments:
        positions = segment.text_token_positions
        token_ids = segment.text_token_ids
        if len(positions) != len(token_ids):
            raise ValueError(
                "speech segment condition mismatch before truncation: "
                f"positions={len(positions)} token_ids={len(token_ids)} "
                f"audio_ref_id={segment.audio_ref_id!r}"
            )
        # Drop an S3 target when truncation removes part of its text condition because
        # the remaining codes have no reliable token boundary.
        if not positions or positions[0] < offset:
            continue
        shifted.append(
            replace(
                segment,
                text_token_positions=[position - offset for position in positions],
                text_token_ids=list(token_ids),
            )
        )
    return shifted
