from __future__ import annotations

from dataclasses import replace
from typing import Any

from mcpmft.data.labels import IGNORE_INDEX
from mcpmft.data.serialize_omniflow import (
    duplex_system_prompt_prefix,
    duplex_system_prompt_suffix,
    unit_ids_from_boundaries,
)
from mcpmft.data.serialize_turn import SerializedSample, SpeechSegment
from mcpmft.prompts import GANDER_DUPLEX_SYSTEM_PROMPT
from mcpmft.tokenizer_tools import (
    AUDIO_END,
    AUDIO_START,
    CHUNK_EOS,
    IMAGE_END,
    IMAGE_START,
    INTERRUPT,
    LISTEN,
    SPEAK,
    SPK_BOS,
    SPK_EOS,
    TTS_BOS,
    TTS_EOS,
    TURN_EOS,
    UNIT_END,
    UNIT_START,
    _resolved_or_none,
)


def build_sampled_context_window(
    item: SerializedSample,
    tokenizer: Any,
    *,
    target_unit: int,
    context_max_units: int,
    context_previous_max_tokens: int = 500,
    max_seq_length: int | None = None,
    system_prompt: str = GANDER_DUPLEX_SYSTEM_PROMPT,
    previous_marker: str = "\n\nprevious: ",
) -> SerializedSample:
    """Materialize a target-consistent context-mode training view.

    Target unit ``t`` sees the preceding K completed units and older generated text in
    ``previous:``. Only the target retains labels. History tokens are recomputed in the
    snapshot, which is built before left truncation and requires sufficient max_seq_length.
    """
    ordinary_units = sorted({unit_id for unit_id in item.unit_ids if unit_id >= 0})
    if target_unit not in ordinary_units:
        raise ValueError(f"target_unit={target_unit} is not present in sample {item.id!r}")

    keep_units = max(0, int(context_max_units))
    target_ordinal = ordinary_units.index(target_unit)
    history_start = max(0, target_ordinal - keep_units)
    history_units = ordinary_units[history_start:target_ordinal]
    dropped_units = set(ordinary_units[:history_start])
    selected_units = set(history_units + [target_unit])

    first_unit_pos = _first_unit_position(item.unit_ids)
    protected_ids = item.input_ids[:first_unit_pos]
    prefix_ids, _suffix_ids = _split_system_prompt(
        protected_ids,
        tokenizer,
        system_prompt=system_prompt,
    )

    previous_content_ids = _previous_content_token_ids(item, dropped_units, tokenizer)
    previous_limit = max(0, int(context_previous_max_tokens))
    previous_content_ids = previous_content_ids[-previous_limit:] if previous_limit else []
    if previous_content_ids and not _suffix_ids:
        raise ValueError(
            "Cannot place previous context: the serialized duplex system-prompt suffix was not "
            f"found in sample {item.id!r}"
        )
    marker_ids = _encode(tokenizer, previous_marker) if previous_content_ids else []
    previous_ids = marker_ids + previous_content_ids

    selected_indices = [
        index for index, unit_id in enumerate(item.unit_ids) if unit_id in selected_units
    ]
    snapshot_length = len(protected_ids) + len(previous_ids) + len(selected_indices)
    if max_seq_length is not None and snapshot_length > int(max_seq_length):
        raise ValueError(
            "Sampled context does not fit max_seq_length: "
            f"sample={item.id!r}, target_unit={target_unit}, length={snapshot_length}, "
            f"max_seq_length={max_seq_length}, history_units={len(history_units)}, "
            f"previous_tokens={len(previous_content_ids)}. Increase max_seq_length or reduce "
            "context_max_units/context_previous_max_tokens."
        )

    old_to_new: dict[int, int] = {}
    new_input_ids: list[int] = []
    new_labels: list[int] = []

    def append_original(old_index: int, *, supervise: bool = False) -> None:
        old_to_new[old_index] = len(new_input_ids)
        new_input_ids.append(item.input_ids[old_index])
        new_labels.append(item.labels[old_index] if supervise else IGNORE_INDEX)

    prefix_len = len(prefix_ids)
    for old_index in range(prefix_len):
        append_original(old_index)
    new_input_ids.extend(previous_ids)
    new_labels.extend([IGNORE_INDEX] * len(previous_ids))
    for old_index in range(prefix_len, first_unit_pos):
        append_original(old_index)
    for old_index in selected_indices:
        append_original(old_index, supervise=item.unit_ids[old_index] == target_unit)

    new_image_bounds, new_image_inputs = _shift_image_bounds(item, old_to_new)
    new_audio_bounds, new_audio_inputs = _shift_audio_bounds(item, old_to_new)
    new_speech_segments = _shift_target_speech_segments(
        item,
        target_unit=target_unit,
        old_to_new=old_to_new,
    )
    meta = dict(item.meta)
    meta["sampled_context"] = {
        "mode": "sampled_context",
        "source_unit_count": len(ordinary_units),
        "target_unit": target_unit,
        "target_ordinal": target_ordinal,
        "history_units": len(history_units),
        "dropped_units": len(dropped_units),
        "previous_content_tokens": len(previous_content_ids),
        "previous_marker_tokens": len(marker_ids),
    }

    return SerializedSample(
        id=f"{item.id}#context{target_unit}",
        input_ids=new_input_ids,
        labels=new_labels,
        image_bounds=new_image_bounds,
        image_inputs=new_image_inputs,
        audio_bounds=new_audio_bounds,
        audio_inputs=new_audio_inputs,
        speech_segments=new_speech_segments,
        unit_ids=unit_ids_from_boundaries(new_input_ids, UNIT_START.token_id),
        meta=meta,
    )


def build_sampled_pinned_context_window(
    item: SerializedSample,
    *,
    target_unit: int,
    context_max_units: int,
    max_seq_length: int | None = None,
) -> SerializedSample:
    """Build one compressed-position training view for ``context_memory``.

    The already-serialized protected prefix contains ``[MEMORY]+[SLATE]`` between the system
    prefix and suffix. Older ordinary units are removed rather than copied into ``previous``;
    retained history and the supervised target are then reindexed contiguously, matching the
    inference controller's cache rebuild.
    """

    ordinary_units = sorted({unit_id for unit_id in item.unit_ids if unit_id >= 0})
    if target_unit not in ordinary_units:
        raise ValueError(f"target_unit={target_unit} is not present in sample {item.id!r}")
    keep_units = max(0, int(context_max_units))
    target_ordinal = ordinary_units.index(target_unit)
    history_start = max(0, target_ordinal - keep_units)
    history_units = ordinary_units[history_start:target_ordinal]
    dropped_units = ordinary_units[:history_start]
    selected_units = set(history_units + [target_unit])
    first_unit_pos = _first_unit_position(item.unit_ids)
    selected_indices = [
        index for index, unit_id in enumerate(item.unit_ids) if unit_id in selected_units
    ]
    snapshot_length = first_unit_pos + len(selected_indices)
    if max_seq_length is not None and snapshot_length > int(max_seq_length):
        raise ValueError(
            "Pinned-context sample does not fit max_seq_length: "
            f"sample={item.id!r}, target_unit={target_unit}, length={snapshot_length}, "
            f"max_seq_length={max_seq_length}, history_units={len(history_units)}"
        )

    old_to_new: dict[int, int] = {}
    new_input_ids: list[int] = []
    new_labels: list[int] = []

    def append_original(old_index: int, *, supervise: bool = False) -> None:
        old_to_new[old_index] = len(new_input_ids)
        new_input_ids.append(item.input_ids[old_index])
        new_labels.append(item.labels[old_index] if supervise else IGNORE_INDEX)

    for old_index in range(first_unit_pos):
        append_original(old_index)
    for old_index in selected_indices:
        append_original(old_index, supervise=item.unit_ids[old_index] == target_unit)

    new_image_bounds, new_image_inputs = _shift_image_bounds(item, old_to_new)
    new_audio_bounds, new_audio_inputs = _shift_audio_bounds(item, old_to_new)
    meta = dict(item.meta)
    meta["sampled_context"] = {
        "mode": "context_memory",
        "source_unit_count": len(ordinary_units),
        "target_unit": target_unit,
        "target_ordinal": target_ordinal,
        "history_units": len(history_units),
        "dropped_units": len(dropped_units),
        "pinned_context_tokens": int(meta.get("pinned_context_tokens") or 0),
    }
    return SerializedSample(
        id=f"{item.id}#memory{target_unit}",
        input_ids=new_input_ids,
        labels=new_labels,
        image_bounds=new_image_bounds,
        image_inputs=new_image_inputs,
        audio_bounds=new_audio_bounds,
        audio_inputs=new_audio_inputs,
        speech_segments=_shift_target_speech_segments(
            item,
            target_unit=target_unit,
            old_to_new=old_to_new,
        ),
        unit_ids=unit_ids_from_boundaries(new_input_ids, UNIT_START.token_id),
        meta=meta,
    )


def _first_unit_position(unit_ids: list[int]) -> int:
    for idx, unit_id in enumerate(unit_ids):
        if unit_id >= 0:
            return idx
    return len(unit_ids)


def _split_system_prompt(
    protected_ids: list[int],
    tokenizer: Any,
    *,
    system_prompt: str,
) -> tuple[list[int], list[int]]:
    prefix_ids = _encode(tokenizer, duplex_system_prompt_prefix(system_prompt))
    suffix_ids = _encode(tokenizer, duplex_system_prompt_suffix())
    expected = prefix_ids + suffix_ids
    if protected_ids == expected:
        return list(prefix_ids), list(suffix_ids)
    if suffix_ids and protected_ids[-len(suffix_ids) :] == suffix_ids:
        return list(protected_ids[: -len(suffix_ids)]), list(suffix_ids)
    return list(protected_ids), []


def _previous_content_token_ids(
    item: SerializedSample,
    drop_units: set[int],
    tokenizer: Any,
) -> list[int]:
    banned = _special_token_ids(tokenizer)
    out: list[int] = []
    for token_id, label, unit_id in zip(item.input_ids, item.labels, item.unit_ids):
        if unit_id not in drop_units:
            continue
        if label == IGNORE_INDEX or token_id in banned:
            continue
        out.append(int(token_id))
    return out


def _special_token_ids(tokenizer: Any) -> set[int]:
    ids = {
        AUDIO_START.token_id,
        AUDIO_END.token_id,
        IMAGE_START.token_id,
        IMAGE_END.token_id,
        UNIT_START.token_id,
        UNIT_END.token_id,
        CHUNK_EOS.token_id,
        TURN_EOS.token_id,
        LISTEN.token_id,
        SPEAK.token_id,
        INTERRUPT.token_id,
        SPK_BOS.token_id,
        SPK_EOS.token_id,
        TTS_BOS.token_id,
        TTS_EOS.token_id,
    }
    ids.update(int(token_id) for token_id in getattr(tokenizer, "all_special_ids", ()))
    for text in [
        "<|chunk_bos|>",
        "<|chunk_tts_bos|>",
        "<|chunk_tts_eos|>",
        "<|turn_bos|>",
        "<|audio|>",
        "<|spk|>",
        "<|tts_pad|>",
        "<|vad_start|>",
        "<|vad_end|>",
    ]:
        resolved = _resolved_or_none(tokenizer, text)
        if resolved is not None:
            ids.add(resolved)
    return ids


def _shift_audio_bounds(
    item: SerializedSample,
    old_to_new: dict[int, int],
) -> tuple[list[tuple[int, int]], list[Any]]:
    return _shift_media_bounds(
        item.audio_bounds,
        item.audio_inputs,
        old_to_new,
        media="audio",
    )


def _shift_image_bounds(
    item: SerializedSample,
    old_to_new: dict[int, int],
) -> tuple[list[tuple[int, int]], list[Any]]:
    return _shift_media_bounds(
        item.image_bounds,
        item.image_inputs,
        old_to_new,
        media="image",
    )


def _shift_media_bounds(
    source_bounds: list[tuple[int, int]],
    source_inputs: list[Any],
    old_to_new: dict[int, int],
    *,
    media: str,
) -> tuple[list[tuple[int, int]], list[Any]]:
    if len(source_bounds) != len(source_inputs):
        raise ValueError(
            f"{media}_bounds/{media}_inputs mismatch: "
            f"{len(source_bounds)} != {len(source_inputs)}"
        )
    bounds: list[tuple[int, int]] = []
    refs: list[Any] = []
    for bound, ref in zip(source_bounds, source_inputs):
        start, end = bound
        if end <= start:
            continue
        if start not in old_to_new or (end - 1) not in old_to_new:
            continue
        bounds.append((old_to_new[start], old_to_new[end - 1] + 1))
        refs.append(ref)
    return bounds, refs


def _shift_target_speech_segments(
    item: SerializedSample,
    *,
    target_unit: int,
    old_to_new: dict[int, int],
) -> list[SpeechSegment]:
    """Keep talker supervision only for the sampled target unit."""
    shifted: list[SpeechSegment] = []
    for segment in item.speech_segments:
        pairs = [
            (old_to_new[position], token_id)
            for position, token_id in zip(segment.text_token_positions, segment.text_token_ids)
            if position in old_to_new
            and position < len(item.unit_ids)
            and item.unit_ids[position] == target_unit
        ]
        if not pairs:
            continue
        shifted.append(
            replace(
                segment,
                text_token_positions=[position for position, _ in pairs],
                text_token_ids=[token_id for _, token_id in pairs],
            )
        )
    return shifted


def _encode(tokenizer: Any, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False) if text else []
