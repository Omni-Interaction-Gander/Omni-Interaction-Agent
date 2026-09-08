from __future__ import annotations

from typing import Any


COMPETITIVE_INTERRUPTION = "competitive_interruption"
INTERRUPT_CONTROL = "interrupt"
CONTROL_FIELD = "duplex_control"
CONTROL_BLOCK_FIELD = "control_block_index"
INTERRUPT_BLOCK_FIELD = "interrupt_block_index"
SUPERVISED_SPEAK_BLOCKS_FIELD = "supervised_speak_block_indices"
SUPPRESSED_SPEAK_BLOCKS_FIELD = "interrupt_suppressed_speak_block_indices"
REQUESTED_CONTROL_BLOCK_FIELD = "requested_control_block_index"


def label_interruption_controls(
    turns: list[dict[str, Any]],
    *,
    block_ms: int = 1000,
    max_decision_delay_ms: int | None = None,
) -> int:
    """Attach a causal interrupt control to each competitive user event.

    Control occurs at the observed assistant release or the latest unit within the
    configured delay. The predecessor retains supervised speech before control and marks
    its suppressed suffix.
    """
    if block_ms <= 0:
        raise ValueError("block_ms must be positive")
    if max_decision_delay_ms is not None and max_decision_delay_ms < block_ms:
        raise ValueError(
            "max_decision_delay_ms must be at least one block so every onset phase is feasible"
        )

    labeled = 0
    for index, turn in enumerate(turns):
        if turn.get("role") != "user":
            continue
        meta = _meta(turn)
        if str(meta.get("event_type") or "") != COMPETITIVE_INTERRUPTION:
            continue
        if index == 0:
            raise ValueError("competitive interruption has no preceding assistant turn")
        # Match an interruption to the nearest preceding assistant, allowing
        # interleaved user or bystander turns.
        predecessor = None
        for candidate in reversed(turns[:index]):
            if candidate.get("role") == "assistant":
                predecessor = candidate
                break
        if predecessor is None:
            raise ValueError("competitive interruption has no preceding assistant turn")
        predecessor_meta = _meta(predecessor)
        if not predecessor_meta.get("is_interrupted"):
            raise ValueError(
                "competitive interruption must follow an assistant marked is_interrupted"
            )

        start_ms = _non_negative_int(turn.get("start_ms"), field="user.start_ms")
        end_ms = _non_negative_int(turn.get("end_ms"), field="user.end_ms")
        if end_ms <= start_ms:
            raise ValueError("competitive interruption user turn must have positive duration")
        release_ms = _non_negative_int(
            predecessor.get("end_ms"),
            field="assistant.end_ms",
        )
        if release_ms <= start_ms:
            raise ValueError(
                "competitive interruption must overlap the predecessor until its observed release"
            )
        onset_block = start_ms // block_ms
        control_block = (release_ms - 1) // block_ms
        planned = _block_list(predecessor_meta.get("speak_block_indices"))
        if onset_block not in planned:
            raise ValueError(
                "competitive interruption onset must overlap the predecessor speak plan: "
                f"onset_block={onset_block}, speak_blocks={planned}"
            )
        source_release_block = control_block
        requested_control_block = meta.get(REQUESTED_CONTROL_BLOCK_FIELD)
        if requested_control_block is not None:
            if (
                not isinstance(requested_control_block, int)
                or isinstance(requested_control_block, bool)
                or requested_control_block <= onset_block
                or requested_control_block > planned[-1] + 1
            ):
                raise ValueError(
                    "requested interrupt control must follow at least one overlapping speak "
                    "block and be no later than the unit after the speak plan: "
                    f"requested={requested_control_block!r}, onset={onset_block}, "
                    f"speak_blocks={planned}"
                )
            last_user_block = max(onset_block, (end_ms - 1) // block_ms)
            if requested_control_block > last_user_block:
                raise ValueError(
                    "requested interrupt control must occur while interrupting user audio is "
                    f"present: requested={requested_control_block}, "
                    f"last_user_block={last_user_block}"
                )
            control_block = requested_control_block
        elif max_decision_delay_ms is not None:
            latest_bounded_block = max(
                onset_block,
                (start_ms + max_decision_delay_ms) // block_ms - 1,
            )
            last_user_block = max(onset_block, (end_ms - 1) // block_ms)
            control_block = min(
                source_release_block,
                latest_bounded_block,
                last_user_block,
            )
            decision_end_ms = (control_block + 1) * block_ms
            decision_delay_ms = decision_end_ms - start_ms
            if not 0 < decision_delay_ms <= max_decision_delay_ms:
                raise ValueError(
                    "bounded interrupt decision is outside the requested latency: "
                    f"delay_ms={decision_delay_ms}"
                )
        if control_block > planned[-1] + 1:
            raise ValueError(
                "interrupt decision block is detached from the predecessor speak plan: "
                f"control_block={control_block}, speak_blocks={planned}"
            )

        supervised = [block for block in planned if block < control_block]
        suppressed = [block for block in planned if block >= control_block]

        _set_or_validate(meta, CONTROL_FIELD, INTERRUPT_CONTROL)
        _set_or_validate(meta, CONTROL_BLOCK_FIELD, control_block)
        _set_or_validate(predecessor_meta, INTERRUPT_BLOCK_FIELD, control_block)
        _set_or_validate(
            predecessor_meta,
            SUPERVISED_SPEAK_BLOCKS_FIELD,
            supervised,
        )
        _set_or_validate(
            predecessor_meta,
            SUPPRESSED_SPEAK_BLOCKS_FIELD,
            suppressed,
        )
        if requested_control_block is not None or max_decision_delay_ms is not None:
            decision_end_ms = (control_block + 1) * block_ms
            decision_delay_ms = decision_end_ms - start_ms
            _set_or_validate(
                predecessor_meta,
                "source_release_interrupt_block_index",
                source_release_block,
            )
            _set_or_validate(
                meta,
                "source_release_control_block_index",
                source_release_block,
            )
            _set_or_validate(meta, "control_evidence_target_ms", decision_delay_ms)
            _set_or_validate(
                meta,
                "control_evidence_ms",
                min(end_ms - start_ms, decision_delay_ms),
            )
        labeled += 1
    return labeled


def _meta(turn: dict[str, Any]) -> dict[str, Any]:
    value = turn.get("meta")
    if not isinstance(value, dict):
        value = {}
        turn["meta"] = value
    return value


def _block_list(value: Any) -> list[int]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(block, int) or block < 0 for block in value)
        or any(left >= right for left, right in zip(value, value[1:]))
    ):
        raise ValueError(f"invalid predecessor speak_block_indices: {value!r}")
    return list(value)


def _non_negative_int(value: Any, *, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer, got {value!r}")
    return value


def _set_or_validate(target: dict[str, Any], key: str, value: Any) -> None:
    if key in target and target[key] != value:
        raise ValueError(
            f"conflicting interruption label {key}: existing={target[key]!r}, expected={value!r}"
        )
    target[key] = value
