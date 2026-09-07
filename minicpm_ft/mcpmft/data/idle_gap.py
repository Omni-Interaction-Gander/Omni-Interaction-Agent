from __future__ import annotations

import copy
import hashlib
import math
import random
from typing import Any, Mapping, Sequence

from mcpmft.data.augmentation_profile import TimelinePolicy
from mcpmft.data.interrupts import (
    CONTROL_BLOCK_FIELD,
    INTERRUPT_BLOCK_FIELD,
    REQUESTED_CONTROL_BLOCK_FIELD,
    SUPERVISED_SPEAK_BLOCKS_FIELD,
    SUPPRESSED_SPEAK_BLOCKS_FIELD,
)
from mcpmft.data.sample import OmniSample, Turn


IDLE_GAP_META_FIELD = "idle_gap_augmentation"
IDLE_GAP_BLOCKS_META_FIELD = "idle_gap_block_indices"

_PENDING_TASK_CALL_NAMES = {"task_start", "task_send"}

_BLOCK_INDEX_FIELDS = {
    CONTROL_BLOCK_FIELD,
    INTERRUPT_BLOCK_FIELD,
    REQUESTED_CONTROL_BLOCK_FIELD,
    "tool_block_index",
}
_BLOCK_LIST_FIELDS = {
    "speak_block_indices",
    SUPERVISED_SPEAK_BLOCKS_FIELD,
    SUPPRESSED_SPEAK_BLOCKS_FIELD,
}
_SAMPLE_BLOCK_INDEX_FIELDS = {
    "action_block",
    "tool_response_block",
    "worker_delivery_block",
}


def insert_random_idle_gaps(
    sample: OmniSample,
    *,
    seed: int,
    min_units: int = 1,
    max_units: int = 10,
    block_ms: int = 1000,
    turn_gap_ms: int = 2000,
    timeline_policy: TimelinePolicy | Mapping[str, Any] | None = None,
    tokenizer: Any | None = None,
    text_tokens_per_unit: int = 4,
) -> OmniSample:
    """Insert deterministic, label-safe idle mic units without changing source manifests.

    Short waits are placed only at the beginning and at completed assistant-to-user boundaries.
    A profile may also opt into short waits between complete, non-overlapping turns from distinct
    human speakers. This path is disabled by default and never applies to reactive speech events.
    A profile may replace a short wait with one or more 5-20-unit no-command spans, including a
    tail span. Long replacements never apply to human-to-human waits. Overlaps, interruptions,
    backchannels, and incomplete turns retain native timing.
    """
    if block_ms <= 0:
        raise ValueError("block_ms must be positive")
    if turn_gap_ms < 0:
        raise ValueError("turn_gap_ms must be non-negative")
    if not sample.turns:
        return sample
    if any(
        turn.start_ms is None
        or turn.end_ms is None
        or turn.start_ms < 0
        or turn.end_ms <= turn.start_ms
        for turn in sample.turns
    ):
        return sample

    if timeline_policy is None:
        if min_units <= 0 or max_units < min_units:
            raise ValueError(
                f"Idle-gap unit range must satisfy 0 < min <= max, got {min_units}..{max_units}"
            )
        policy = TimelinePolicy(
            enabled=True,
            short_gap_bands=((min_units, max_units, 1.0),),
        )
    elif isinstance(timeline_policy, TimelinePolicy):
        policy = timeline_policy
    else:
        policy = TimelinePolicy(**dict(timeline_policy))
    if not policy.enabled:
        return sample
    _validate_gap_bands(policy.short_gap_bands, "short_gap_bands")
    _validate_gap_bands(
        policy.human_to_human_gap_bands,
        "human_to_human_gap_bands",
    )
    _validate_gap_bands(policy.long_gap_bands, "long_gap_bands")
    if policy.pending_task_gap_bands:
        _validate_gap_bands(
            policy.pending_task_gap_bands,
            "pending_task_gap_bands",
        )
        if tokenizer is None:
            raise ValueError("pending_task_gap_bands requires a tokenizer")
        if text_tokens_per_unit <= 0:
            raise ValueError("text_tokens_per_unit must be positive")
    if not 0.0 <= policy.human_to_human_probability <= 1.0:
        raise ValueError("human_to_human_probability must be in [0, 1]")
    if not 0.0 <= policy.long_gap_probability <= 1.0:
        raise ValueError("long_gap_probability must be in [0, 1]")
    if policy.max_long_spans < 0:
        raise ValueError("max_long_spans must be non-negative")

    rng = random.Random(_sample_seed(seed, sample.id))
    start_units = (
        _sample_gap_units(policy.short_gap_bands, rng)
        if policy.insert_start
        else 0
    )
    assistant_boundaries = (
        _safe_assistant_to_user_boundaries(sample.turns)
        if policy.insert_between_turns
        else []
    )
    inserted: list[dict[str, Any]] = [
        {
            "boundary_ms": boundary_ms,
            "before_turn_index": turn_index,
            "units": _sample_gap_units(policy.short_gap_bands, rng),
            "kind": "short",
            "boundary_type": "assistant_to_user",
        }
        for boundary_ms, turn_index in assistant_boundaries
    ]
    human_boundaries = (
        _safe_human_to_human_boundaries(sample.turns)
        if policy.human_to_human_probability > 0
        else []
    )
    for boundary_ms, turn_index in human_boundaries:
        if rng.random() >= policy.human_to_human_probability:
            continue
        inserted.append(
            {
                "boundary_ms": boundary_ms,
                "before_turn_index": turn_index,
                "units": _sample_gap_units(
                    policy.human_to_human_gap_bands,
                    rng,
                ),
                "kind": "human_short",
                "boundary_type": "human_to_human",
            }
        )
    inserted.sort(
        key=lambda item: (
            int(item["boundary_ms"]),
            int(item["before_turn_index"]),
        )
    )
    start_kind = "short" if start_units else "none"
    tail_units = 0
    long_locations: list[str] = []
    candidates: list[tuple[str, int | None]] = []
    allowed = set(policy.long_gap_locations)
    if start_units and "start" in allowed:
        candidates.append(("start", None))
    if "between" in allowed:
        candidates.extend(
            ("between", index)
            for index, item in enumerate(inserted)
            if item["boundary_type"] == "assistant_to_user"
        )
    if "tail" in allowed:
        candidates.append(("tail", None))
    rng.shuffle(candidates)
    for span_index in range(min(policy.max_long_spans, len(candidates))):
        if rng.random() >= policy.long_gap_probability:
            break
        location, index = candidates[span_index]
        units = _sample_gap_units(policy.long_gap_bands, rng)
        if location == "start":
            start_units = units
            start_kind = "long"
        elif location == "between":
            assert index is not None
            inserted[index]["units"] = units
            inserted[index]["kind"] = "long"
        else:
            tail_units = units
        long_locations.append(location)

    idle_gap_blocks = set(range(start_units))
    prior_shift_ms = start_units * block_ms
    for item in inserted:
        boundary_ms = int(item["boundary_ms"])
        units = int(item["units"])
        augmented_boundary_ms = boundary_ms + prior_shift_ms
        alignment_ms = (-augmented_boundary_ms) % block_ms
        augmented_start_ms = augmented_boundary_ms + alignment_ms
        augmented_end_ms = augmented_start_ms + units * block_ms
        first_block = augmented_start_ms // block_ms
        last_block = max(augmented_end_ms - 1, augmented_start_ms) // block_ms
        idle_gap_blocks.update(range(first_block, last_block + 1))
        item["alignment_ms"] = alignment_ms
        item["augmented_start_ms"] = augmented_start_ms
        item["augmented_end_ms"] = augmented_end_ms
        item["shift_ms"] = alignment_ms + units * block_ms
        prior_shift_ms += int(item["shift_ms"])

    result = copy.deepcopy(sample)
    native_block_shifts: list[int] = []
    for original_turn, turn in zip(sample.turns, result.turns):
        assert original_turn.start_ms is not None
        assert turn.start_ms is not None and turn.end_ms is not None
        original_start_ms = int(original_turn.start_ms)
        shift_ms = start_units * block_ms + sum(
            int(item["shift_ms"])
            for item in inserted
            if int(item["boundary_ms"]) <= original_start_ms
        )
        turn.start_ms += shift_ms
        turn.end_ms += shift_ms
        for image_ref in turn.images:
            if image_ref.start_ms is not None:
                image_ref.start_ms += shift_ms
            if image_ref.end_ms is not None:
                image_ref.end_ms += shift_ms
        native_block_shifts.append(
            turn.start_ms // block_ms - original_start_ms // block_ms
        )
    metadata_block_shifts = _assistant_group_block_shifts(
        sample.turns,
        result.turns,
        native_block_shifts,
        block_ms=block_ms,
        turn_gap_ms=turn_gap_ms,
    )
    for turn, block_shift in zip(result.turns, metadata_block_shifts):
        _shift_training_block_metadata(turn.meta, block_shift)
    _shift_interrupted_plans_after_prior_assistant_units(result.turns)
    _synchronize_paired_interrupt_controls(sample.turns, result.turns)

    shifted_end_ms = max(int(turn.end_ms or 0) for turn in result.turns)
    if tail_units:
        tail_start_ms = int(math.ceil(shifted_end_ms / block_ms) * block_ms)
        first_tail_block = tail_start_ms // block_ms
        tail_end_ms = tail_start_ms + tail_units * block_ms
        last_tail_block = max(tail_end_ms - 1, tail_start_ms) // block_ms
        idle_gap_blocks.update(range(first_tail_block, last_tail_block + 1))
    else:
        tail_start_ms = shifted_end_ms
        tail_end_ms = shifted_end_ms

    total_units = start_units + sum(int(item["units"]) for item in inserted) + tail_units
    total_inserted_ms = (
        start_units * block_ms
        + sum(int(item["shift_ms"]) for item in inserted)
        + max(tail_start_ms - shifted_end_ms, 0)
        + tail_units * block_ms
    )
    result.meta = copy.deepcopy(result.meta)
    _shift_sample_block_metadata(
        result.meta,
        start_units=start_units,
        inserted=inserted,
        block_ms=block_ms,
    )
    if isinstance(result.meta.get("timeline_blocks"), int) and not isinstance(
        result.meta["timeline_blocks"], bool
    ):
        result.meta["timeline_blocks"] = int(math.ceil(tail_end_ms / block_ms))
    result.meta[IDLE_GAP_META_FIELD] = {
        "block_ms": block_ms,
        "start_units": start_units,
        "start_kind": start_kind,
        "between_turns": inserted,
        "tail_units": tail_units,
        "tail_start_ms": tail_start_ms if tail_units else None,
        "long_locations": long_locations,
        "total_inserted_units": total_units,
        "total_inserted_ms": total_inserted_ms,
    }
    result.meta[IDLE_GAP_BLOCKS_META_FIELD] = sorted(idle_gap_blocks)
    result.meta["augmented_timeline_end_ms"] = tail_end_ms
    result.meta["augmented_total_timeline_sec"] = tail_end_ms / 1000.0
    if policy.pending_task_gap_bands:
        result = insert_pending_task_idle_gaps(
            result,
            seed=seed,
            gap_bands=policy.pending_task_gap_bands,
            tokenizer=tokenizer,
            text_tokens_per_unit=text_tokens_per_unit,
            block_ms=block_ms,
        )
    return result


def insert_pending_task_idle_gaps(
    sample: OmniSample,
    *,
    seed: int,
    gap_bands: Sequence[Sequence[float]],
    tokenizer: Any,
    text_tokens_per_unit: int = 4,
    block_ms: int = 1000,
    max_timeline_units: int | None = None,
) -> OmniSample:
    """Teach event causality while a background task is pending.

    A successful ``task_start``/``task_send`` may be acknowledged immediately. After that spoken
    turn has actually finished, no new assistant turn is justified until user input or a Runtime
    event arrives. Reviewed agent rows contain either a later ``worker_delivery`` or end at the
    acknowledgement, so this transform creates a deterministic 10-30-unit (profile-configured)
    listen span without inventing task state or semantic content.
    """

    if block_ms <= 0:
        raise ValueError("block_ms must be positive")
    if text_tokens_per_unit <= 0:
        raise ValueError("text_tokens_per_unit must be positive")
    if max_timeline_units is not None and max_timeline_units <= 0:
        raise ValueError("max_timeline_units must be positive when provided")
    _validate_gap_bands(gap_bands, "pending_task_gap_bands")
    minimum_listen_units = min(
        int(band[0]) for band in gap_bands if float(band[2]) > 0
    )

    result = copy.deepcopy(sample)
    rng = random.Random(_sample_seed(seed, f"{sample.id}\0pending-task"))
    records: list[dict[str, Any]] = []
    idle_blocks = set(result.meta.get(IDLE_GAP_BLOCKS_META_FIELD) or ())

    for action_index, action in enumerate(result.turns):
        call_names = {
            str(call.get("name") or "")
            for call in action.tool_calls
            if isinstance(call, Mapping)
        }
        task_call_names = call_names.intersection(_PENDING_TASK_CALL_NAMES)
        if action.role != "assistant" or len(task_call_names) != 1:
            continue

        response_index = _next_turn_index(
            result.turns,
            action_index + 1,
            lambda turn: turn.role == "tool"
            and not bool(turn.meta.get("runtime_event")),
        )
        if response_index is None:
            continue
        response = result.turns[response_index].tool_response
        if not isinstance(response, Mapping) or response.get("status") != "ok":
            continue

        acknowledgement_index = _next_turn_index(
            result.turns,
            response_index + 1,
            lambda turn: turn.role == "assistant"
            and turn.meta.get("phase") == "task_response_answer"
            and bool(turn.text),
        )
        if acknowledgement_index is None:
            continue

        delivery_index = _next_turn_index(
            result.turns,
            acknowledgement_index + 1,
            _is_worker_delivery_turn,
        )
        next_action_index = _next_turn_index(
            result.turns,
            action_index + 1,
            lambda turn: turn.role == "assistant" and bool(turn.tool_calls),
        )
        if (
            delivery_index is not None
            and next_action_index is not None
            and next_action_index < delivery_index
        ):
            delivery_index = None

        # This is deliberately an event-free augmentation. Do not move real user input or another
        # action merely to manufacture a clean waiting span.
        span_end = delivery_index if delivery_index is not None else len(result.turns)
        if any(
            _is_observable_intervening_turn(turn)
            for turn in result.turns[acknowledgement_index + 1 : span_end]
        ):
            continue
        if delivery_index is None and acknowledgement_index != len(result.turns) - 1:
            continue

        assistant_spans = _planned_assistant_spans(
            result.turns,
            tokenizer=tokenizer,
            text_tokens_per_unit=text_tokens_per_unit,
            block_ms=block_ms,
        )
        acknowledgement_span = assistant_spans.get(acknowledgement_index)
        if acknowledgement_span is None:
            continue
        acknowledgement_end_block = acknowledgement_span[1]
        sampled_listen_units = _sample_gap_units(gap_bands, rng)
        original_sampled_listen_units = sampled_listen_units
        baseline_end_block = _planned_timeline_end_block(
            result,
            tokenizer=tokenizer,
            text_tokens_per_unit=text_tokens_per_unit,
            block_ms=block_ms,
        )

        if delivery_index is not None:
            delivery = result.turns[delivery_index]
            delivery_block = int(
                delivery.meta.get(
                    "tool_block_index",
                    int(delivery.start_ms or 0) // block_ms,
                )
            )
            if max_timeline_units is not None:
                max_shift_units = max_timeline_units - baseline_end_block
                max_listen_units = (
                    delivery_block + max_shift_units - acknowledgement_end_block
                )
                if max_listen_units < minimum_listen_units:
                    continue
                sampled_listen_units = min(
                    sampled_listen_units,
                    max_listen_units,
                )
            desired_delivery_block = max(
                delivery_block,
                acknowledgement_end_block + sampled_listen_units,
            )
            inserted_units = desired_delivery_block - delivery_block
            if inserted_units:
                _shift_turn_suffix(
                    result.turns,
                    delivery_index,
                    inserted_units,
                    block_ms=block_ms,
                )
                idle_blocks = {
                    block + inserted_units if block >= delivery_block else block
                    for block in idle_blocks
                }
            actual_delivery_block = desired_delivery_block
            idle_blocks.update(
                range(acknowledgement_end_block, actual_delivery_block)
            )
            records.append(
                {
                    "task_call": next(iter(task_call_names)),
                    "acknowledgement_turn_index": acknowledgement_index,
                    "acknowledgement_end_block": acknowledgement_end_block,
                    "worker_delivery_turn_index": delivery_index,
                    "worker_delivery_block_before": delivery_block,
                    "worker_delivery_block_after": actual_delivery_block,
                    "sampled_listen_units": sampled_listen_units,
                    "unclamped_sampled_listen_units": original_sampled_listen_units,
                    "actual_listen_units": (
                        actual_delivery_block - acknowledgement_end_block
                    ),
                    "inserted_units": inserted_units,
                    "kind": "before_worker_delivery",
                }
            )
        else:
            if max_timeline_units is not None:
                max_listen_units = max_timeline_units - acknowledgement_end_block
                if max_listen_units < minimum_listen_units:
                    continue
                sampled_listen_units = min(
                    sampled_listen_units,
                    max_listen_units,
                )
            pending_end_block = acknowledgement_end_block + sampled_listen_units
            idle_blocks.update(range(acknowledgement_end_block, pending_end_block))
            records.append(
                {
                    "task_call": next(iter(task_call_names)),
                    "acknowledgement_turn_index": acknowledgement_index,
                    "acknowledgement_end_block": acknowledgement_end_block,
                    "worker_delivery_turn_index": None,
                    "sampled_listen_units": sampled_listen_units,
                    "unclamped_sampled_listen_units": original_sampled_listen_units,
                    "actual_listen_units": sampled_listen_units,
                    "inserted_units": sampled_listen_units,
                    "kind": "pending_tail",
                }
            )

    if not records:
        return result

    planned_spans = _planned_assistant_spans(
        result.turns,
        tokenizer=tokenizer,
        text_tokens_per_unit=text_tokens_per_unit,
        block_ms=block_ms,
    )
    planned_end_ms = max(
        (end_block * block_ms for _, end_block in planned_spans.values()),
        default=0,
    )
    turn_end_ms = max((int(turn.end_ms or 0) for turn in result.turns), default=0)
    pending_tail_end_ms = max(
        (
            (record["acknowledgement_end_block"] + record["actual_listen_units"])
            * block_ms
            for record in records
            if record["kind"] == "pending_tail"
        ),
        default=0,
    )
    existing_end_ms = int(result.meta.get("augmented_timeline_end_ms") or 0)
    timeline_end_ms = max(
        existing_end_ms,
        turn_end_ms,
        planned_end_ms,
        pending_tail_end_ms,
    )
    result.meta = copy.deepcopy(result.meta)
    result.meta[IDLE_GAP_BLOCKS_META_FIELD] = sorted(idle_blocks)
    idle_meta = dict(result.meta.get(IDLE_GAP_META_FIELD) or {})
    idle_meta["pending_tasks"] = records
    idle_meta["pending_task_inserted_units"] = sum(
        int(record["inserted_units"]) for record in records
    )
    idle_meta["total_inserted_units"] = int(
        idle_meta.get("total_inserted_units") or 0
    ) + idle_meta["pending_task_inserted_units"]
    idle_meta["total_inserted_ms"] = int(
        idle_meta.get("total_inserted_ms") or 0
    ) + idle_meta["pending_task_inserted_units"] * block_ms
    result.meta[IDLE_GAP_META_FIELD] = idle_meta
    result.meta["augmented_timeline_end_ms"] = timeline_end_ms
    result.meta["augmented_total_timeline_sec"] = timeline_end_ms / 1000.0
    if isinstance(result.meta.get("timeline_blocks"), int) and not isinstance(
        result.meta["timeline_blocks"], bool
    ):
        result.meta["timeline_blocks"] = int(math.ceil(timeline_end_ms / block_ms))
    worker_delivery_turns = [
        turn for turn in result.turns if _is_worker_delivery_turn(turn)
    ]
    if len(worker_delivery_turns) == 1:
        result.meta["worker_delivery_block"] = int(
            worker_delivery_turns[0].meta.get(
                "tool_block_index",
                int(worker_delivery_turns[0].start_ms or 0) // block_ms,
            )
        )
    return result


def _next_turn_index(
    turns: Sequence[Turn],
    start: int,
    predicate: Any,
) -> int | None:
    for index in range(start, len(turns)):
        if predicate(turns[index]):
            return index
    return None


def _is_worker_delivery_turn(turn: Turn) -> bool:
    response = turn.tool_response
    return (
        turn.role == "tool"
        and bool(turn.meta.get("runtime_event"))
        and isinstance(response, Mapping)
        and response.get("type") == "worker_delivery"
    )


def _is_observable_intervening_turn(turn: Turn) -> bool:
    return bool(
        turn.role == "user"
        or turn.tool_calls
        or turn.tool_response is not None
        or turn.text
        or turn.audio_in is not None
        or turn.images
    )


def _planned_assistant_spans(
    turns: Sequence[Turn],
    *,
    tokenizer: Any,
    text_tokens_per_unit: int,
    block_ms: int,
) -> dict[int, tuple[int, int]]:
    """Mirror the duplex serializer's text-driven assistant block planner."""

    spans: dict[int, tuple[int, int]] = {}
    next_free = 0
    ordered = sorted(
        (
            (index, turn)
            for index, turn in enumerate(turns)
            if turn.role == "assistant" and turn.text
        ),
        key=lambda item: (int(item[1].start_ms or 0), item[0]),
    )
    for index, turn in ordered:
        text_ids = tokenizer.encode(turn.text or "", add_special_tokens=False)
        units = max(1, int(math.ceil(len(text_ids) / text_tokens_per_unit)))
        requested = turn.meta.get("speak_block_indices")
        if isinstance(requested, list) and len(requested) == units:
            assigned: list[int] = []
            for block in requested:
                assigned.append(
                    max(int(block), next_free if not assigned else assigned[-1] + 1)
                )
        else:
            start_block = max(int(turn.start_ms or 0) // block_ms, next_free)
            assigned = list(range(start_block, start_block + units))
        spans[index] = (assigned[0], assigned[-1] + 1)
        interrupt_block = turn.meta.get(INTERRUPT_BLOCK_FIELD)
        next_free = (
            int(interrupt_block) + 1
            if isinstance(interrupt_block, int) and not isinstance(interrupt_block, bool)
            else assigned[-1] + 1
        )
    return spans


def _planned_timeline_end_block(
    sample: OmniSample,
    *,
    tokenizer: Any,
    text_tokens_per_unit: int,
    block_ms: int,
) -> int:
    assistant_spans = _planned_assistant_spans(
        sample.turns,
        tokenizer=tokenizer,
        text_tokens_per_unit=text_tokens_per_unit,
        block_ms=block_ms,
    )
    assistant_end = max((end for _, end in assistant_spans.values()), default=0)
    turn_end = max(
        (int(math.ceil(int(turn.end_ms or 0) / block_ms)) for turn in sample.turns),
        default=0,
    )
    augmented_end = int(
        math.ceil(int(sample.meta.get("augmented_timeline_end_ms") or 0) / block_ms)
    )
    return max(assistant_end, turn_end, augmented_end)


def _shift_turn_suffix(
    turns: Sequence[Turn],
    start_index: int,
    units: int,
    *,
    block_ms: int,
) -> None:
    shift_ms = units * block_ms
    for turn in turns[start_index:]:
        if turn.start_ms is not None:
            turn.start_ms += shift_ms
        if turn.end_ms is not None:
            turn.end_ms += shift_ms
        for image_ref in turn.images:
            if image_ref.start_ms is not None:
                image_ref.start_ms += shift_ms
            if image_ref.end_ms is not None:
                image_ref.end_ms += shift_ms
        _shift_training_block_metadata(turn.meta, units)


def _shift_sample_block_metadata(
    meta: dict[str, Any],
    *,
    start_units: int,
    inserted: Sequence[Mapping[str, Any]],
    block_ms: int,
) -> None:
    """Keep materializer timing anchors consistent with shifted tool turns."""
    for field in _SAMPLE_BLOCK_INDEX_FIELDS:
        value = meta.get(field)
        if not isinstance(value, int) or isinstance(value, bool):
            continue
        original_ms = value * block_ms
        shift_ms = start_units * block_ms + sum(
            int(item["shift_ms"])
            for item in inserted
            if int(item["boundary_ms"]) <= original_ms
        )
        meta[field] = (original_ms + shift_ms) // block_ms


def _safe_assistant_to_user_boundaries(turns: list[Turn]) -> list[tuple[int, int]]:
    ordered = sorted(
        enumerate(turns),
        key=lambda item: (
            int(item[1].start_ms or 0),
            int(item[1].end_ms or 0),
            item[0],
        ),
    )
    boundaries: list[tuple[int, int]] = []
    for (previous_index, previous), (current_index, current) in zip(ordered, ordered[1:]):
        if previous.role != "assistant" or current.role != "user":
            continue
        if not _is_naturally_complete(previous):
            continue
        if current.meta.get("duplex_control") == "interrupt":
            continue
        assert previous.end_ms is not None and current.start_ms is not None
        if current.start_ms < previous.end_ms:
            continue
        boundary_ms = int(current.start_ms)
        if any(
            index not in {previous_index, current_index}
            and turn.start_ms is not None
            and turn.end_ms is not None
            and turn.start_ms < boundary_ms < turn.end_ms
            for index, turn in enumerate(turns)
        ):
            continue
        boundaries.append((boundary_ms, current_index))
    return boundaries


def _safe_human_to_human_boundaries(turns: list[Turn]) -> list[tuple[int, int]]:
    ordered = sorted(
        enumerate(turns),
        key=lambda item: (
            int(item[1].start_ms or 0),
            int(item[1].end_ms or 0),
            item[0],
        ),
    )
    boundaries: list[tuple[int, int]] = []
    for (previous_index, previous), (current_index, current) in zip(ordered, ordered[1:]):
        if previous.role != "user" or current.role != "user":
            continue
        if not _is_naturally_complete(previous) or not _is_naturally_complete(current):
            continue
        previous_speaker = previous.meta.get("speaker_id")
        current_speaker = current.meta.get("speaker_id")
        if (
            previous_speaker is None
            or current_speaker is None
            or str(previous_speaker) == str(current_speaker)
        ):
            continue
        if _is_reactive_event(previous) or _is_reactive_event(current):
            continue
        assert previous.end_ms is not None and current.start_ms is not None
        if current.start_ms < previous.end_ms:
            continue
        boundary_ms = int(current.start_ms)
        if any(
            index not in {previous_index, current_index}
            and turn.start_ms is not None
            and turn.end_ms is not None
            and turn.start_ms < boundary_ms < turn.end_ms
            for index, turn in enumerate(turns)
        ):
            continue
        boundaries.append((boundary_ms, current_index))
    return boundaries


def _is_naturally_complete(turn: Turn) -> bool:
    state = (turn.turn_state or "complete").strip("<|>").lower()
    return state == "complete" and not bool(turn.meta.get("is_interrupted"))


def _is_reactive_event(turn: Turn) -> bool:
    state = (turn.turn_state or "").strip("<|>").lower()
    control = str(turn.meta.get("duplex_control") or "").strip("<|>").lower()
    event_type = str(turn.meta.get("event_type") or "").lower()
    return (
        state == "backchannel"
        or control in {"backchannel", "interrupt"}
        or bool(turn.meta.get("is_interrupted"))
        or any(label in event_type for label in ("backchannel", "interrupt", "overlap"))
    )


def _shift_training_block_metadata(meta: dict[str, Any], shift_units: int) -> None:
    if shift_units <= 0:
        return
    for field in _BLOCK_INDEX_FIELDS:
        value = meta.get(field)
        if isinstance(value, int) and not isinstance(value, bool):
            meta[field] = value + shift_units
    for field in _BLOCK_LIST_FIELDS:
        value = meta.get(field)
        if isinstance(value, list) and all(
            isinstance(block, int) and not isinstance(block, bool) for block in value
        ):
            meta[field] = [block + shift_units for block in value]


def _assistant_group_block_shifts(
    original_turns: Sequence[Turn],
    shifted_turns: Sequence[Turn],
    native_shifts: Sequence[int],
    *,
    block_ms: int,
    turn_gap_ms: int,
) -> list[int]:
    """Use one block delta for assistant fragments serialized as one logical turn."""
    shifts = list(native_shifts)
    ordered = sorted(
        (index for index, turn in enumerate(original_turns) if turn.role == "assistant"),
        key=lambda index: (
            int(original_turns[index].start_ms or 0),
            int(original_turns[index].end_ms or 0),
            index,
        ),
    )
    group_shift: int | None = None
    last_end_ms: int | None = None
    previous_closed = True
    for index in ordered:
        original = original_turns[index]
        start_ms = int(original.start_ms or 0)
        if (
            last_end_ms is None
            or previous_closed
            or start_ms - last_end_ms > turn_gap_ms
        ):
            assert shifted_turns[index].start_ms is not None
            group_shift = (
                int(shifted_turns[index].start_ms) // block_ms
                - start_ms // block_ms
            )
        assert group_shift is not None
        shifts[index] = group_shift
        last_end_ms = int(original.end_ms or start_ms)
        state = (original.turn_state or "").strip("<|>").lower()
        previous_closed = state == "complete" or bool(
            original.meta.get("is_interrupted")
        )
    return shifts


def _synchronize_paired_interrupt_controls(
    original_turns: Sequence[Turn],
    shifted_turns: Sequence[Turn],
) -> None:
    """Keep the two labels for one observed interrupt on the same shifted block.

    An inserted gap may contain a sub-block alignment offset. The overlapping assistant and user
    can then cross different block boundaries even though both move by the same number of
    milliseconds. Their per-turn block deltas differ, so independently shifting the duplicate
    interrupt labels would make the user control collide with a still-supervised assistant unit.
    Only pairs that agreed in the source are synchronized; malformed source labels remain visible
    to the serializer instead of being silently repaired.
    """
    for index in range(1, min(len(original_turns), len(shifted_turns))):
        original_user = original_turns[index]
        original_assistant = original_turns[index - 1]
        if (
            original_user.role != "user"
            or original_user.meta.get("duplex_control") != "interrupt"
            or original_assistant.role != "assistant"
        ):
            continue
        source_control = original_user.meta.get(CONTROL_BLOCK_FIELD)
        source_interrupt = original_assistant.meta.get(INTERRUPT_BLOCK_FIELD)
        if (
            not isinstance(source_control, int)
            or isinstance(source_control, bool)
            or source_control != source_interrupt
        ):
            continue
        shifted_interrupt = shifted_turns[index - 1].meta.get(INTERRUPT_BLOCK_FIELD)
        if isinstance(shifted_interrupt, int) and not isinstance(shifted_interrupt, bool):
            shifted_turns[index].meta[CONTROL_BLOCK_FIELD] = shifted_interrupt


def _shift_interrupted_plans_after_prior_assistant_units(
    turns: Sequence[Turn],
) -> None:
    """Preserve an interrupted plan when quantization moves it onto a prior assistant unit."""
    next_free = 0
    ordered = sorted(
        (turn for turn in turns if turn.role == "assistant"),
        key=lambda turn: int(turn.start_ms or 0),
    )
    for turn in ordered:
        requested = turn.meta.get("speak_block_indices")
        if (
            not isinstance(requested, list)
            or not requested
            or any(
                not isinstance(block, int) or isinstance(block, bool) or block < 0
                for block in requested
            )
            or any(left >= right for left, right in zip(requested, requested[1:]))
        ):
            continue
        assigned: list[int] = []
        for block in requested:
            assigned.append(
                max(block, next_free if not assigned else assigned[-1] + 1)
            )
        if turn.meta.get("is_interrupted") and assigned != requested:
            shift_units = max(next_free - requested[0], 0)
            if shift_units:
                _shift_training_block_metadata(turn.meta, shift_units)
                requested = turn.meta["speak_block_indices"]
                assigned = list(requested)
        interrupt_block = turn.meta.get(INTERRUPT_BLOCK_FIELD)
        next_free = (
            int(interrupt_block) + 1
            if isinstance(interrupt_block, int) and not isinstance(interrupt_block, bool)
            else assigned[-1] + 1
        )


def _sample_seed(seed: int, sample_id: str) -> int:
    digest = hashlib.blake2b(
        f"{int(seed)}\0{sample_id}".encode("utf-8"),
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, "little")


def _sample_gap_units(
    bands: Sequence[Sequence[float]],
    rng: random.Random,
) -> int:
    weighted: list[tuple[int, int, float]] = []
    total = 0.0
    for band in bands:
        low, high, weight = int(band[0]), int(band[1]), float(band[2])
        if weight <= 0:
            continue
        weighted.append((low, high, weight))
        total += weight
    if total <= 0:
        raise ValueError("Gap bands require a positive total weight")
    point = rng.random() * total
    cumulative = 0.0
    selected = weighted[-1]
    for band in weighted:
        cumulative += band[2]
        if point < cumulative:
            selected = band
            break
    return rng.randint(selected[0], selected[1])


def _validate_gap_bands(
    bands: Sequence[Sequence[float]],
    field_name: str,
) -> None:
    if not bands:
        raise ValueError(f"{field_name} must not be empty")
    total = 0.0
    for band in bands:
        if len(band) != 3:
            raise ValueError(f"{field_name} entries must be [min, max, weight]")
        low, high, weight = int(band[0]), int(band[1]), float(band[2])
        if low <= 0 or high < low or weight < 0:
            raise ValueError(f"Invalid {field_name} entry: {band!r}")
        total += weight
    if total <= 0:
        raise ValueError(f"{field_name} must have a positive total weight")
