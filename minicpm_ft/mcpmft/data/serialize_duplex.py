"""Unified block-driven full-duplex serializer.

Full-duplex means the model perceives a continuous input stream and, on a fixed 1s time grid,
decides at each block whether to listen, speak, backchannel, or interrupt while STILL receiving
env-audio (always-on mic).
The control space is ``listen``/``speak``/``backchannel``/``interrupt`` plus complete native
tool calls. Runtime task state returns later as masked ``<tool_response>`` context.

Per 1s block, in strict causal order (matching MiniCPMODuplex streaming_prefill/generate and the
official schema `<unit>[audio_embed]<|listen|or|speak|>content</unit>`):
    <unit>                                              (boundary, not supervised)
    [video-frame embeddings]                            (perception input, zero or more frames)
    [env-audio placeholders]                            (perception input, ALWAYS present:
                                                         real user speech overlapping this block,
                                                         else 1s silence — always-on mic)
    [arriving user text]                                (optional text_list input, -100)
    [arriving tool response(s)]                         (optional structured input, -100)
    <|listen|> | <|speak|> | <|backchannel|> |
    <|interrupt|>                                       (dialogue control, SUPERVISED)
    [K agent text tokens]      (speak blocks)            (SUPERVISED, + per-unit s3 target)
    [<|turn_eos|>]             (naturally completed text turn only) (SUPERVISED)
    <|chunk_eos|>              (speak blocks)             (SUPERVISED)
    </unit>                                              (boundary, not supervised)

Text is chunked at K tokens/block (fixed-unit streaming). Thinker units are text/control-driven:
S3 length never creates extra empty-text speak units. If S3 targets are present, each non-final text
unit receives the next speech_tokens_per_unit codes and the final text unit absorbs every remaining
code, without a final-unit cap. Length disagreement is alignment metadata, not a reason to discard or
truncate a speech target.
"""
from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

from mcpmft.data.feature import AudioGeometry, block_mic_audio
from mcpmft.data.idle_gap import IDLE_GAP_BLOCKS_META_FIELD
from mcpmft.data.interrupts import (
    CONTROL_BLOCK_FIELD,
    CONTROL_FIELD,
    INTERRUPT_BLOCK_FIELD,
    INTERRUPT_CONTROL,
    SUPERVISED_SPEAK_BLOCKS_FIELD,
)
from mcpmft.data.labels import IGNORE_INDEX
from mcpmft.data.sample import OmniSample, Turn
from mcpmft.data.serialize_turn import (
    SerializedSample,
    SpeechSegment,
    _left_truncation_offset,
    _shift_audio_pairs,
    _shift_image_pairs,
    _shift_complete_speech_segments,
    _unk_id,
)
from mcpmft.prompts import GANDER_DUPLEX_SYSTEM_PROMPT
from mcpmft.tool_protocol import (
    format_tool_calls,
    format_tool_response,
    system_prompt_with_tools,
)
from mcpmft.tokenizer_tools import (
    CHUNK_EOS,
    IMAGE_END,
    IMAGE_FEATURE_SIZE,
    IMAGE_START,
    TURN_EOS,
    UNIT_END,
    UNIT_START,
)

# Shared pure helpers stay in serialize_omniflow (imported by callers too).
from mcpmft.data.serialize_omniflow import (
    BACKCHANNEL_CLASS,
    duplex_system_prompt_prefix,
    duplex_system_prompt_suffix,
    _is_complete,
    control_token_id,
    turn_state_to_control,
    unit_ids_from_boundaries,
)


def _enc(tokenizer, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False) if text else []


def _id(tokenizer, token_text: str) -> int:
    return int(tokenizer.convert_tokens_to_ids(token_text))


def split_s3_codes_for_text_units(
    s3_codes: list[int],
    *,
    text_units: int,
    speech_tokens_per_unit: int,
) -> list[list[int]]:
    """Split S3 loss targets without dropping codes.

    The first ``text_units - 1`` slices follow the configured streaming cadence. The final text
    unit owns the complete remainder, whether that remainder is empty, shorter than one nominal
    unit, or spans several nominal units. This is deliberately asymmetric: Thinker determines the
    number of conditioning units, while the target waveform determines when Talker predicts EOS.
    """
    if text_units <= 0:
        raise ValueError("text_units must be positive")
    if speech_tokens_per_unit <= 0:
        raise ValueError("speech_tokens_per_unit must be positive")

    codes = list(s3_codes)
    slices = [
        codes[index * speech_tokens_per_unit : (index + 1) * speech_tokens_per_unit]
        for index in range(text_units - 1)
    ]
    slices.append(codes[(text_units - 1) * speech_tokens_per_unit :])
    if [code for unit in slices for code in unit] != codes:
        raise AssertionError("S3 unit planning must preserve every target code in order")
    return slices


def serialize_duplex_sample(
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
    pinned_context: str = "",
) -> SerializedSample:
    geometry = audio_geometry or AudioGeometry()
    K = max(1, int(text_tokens_per_block))
    codes_per_text_token = max(1, int(codes_per_text_token))
    speech_tokens_per_unit = int(speech_tokens_per_unit)
    idle_gap_blocks = {
        int(block)
        for block in sample.meta.get(IDLE_GAP_BLOCKS_META_FIELD, [])
        if isinstance(block, int) and not isinstance(block, bool) and block >= 0
    }

    # ---- 1. Bucket turns and video frames onto the 1s block grid. ----
    agent_buckets: dict[int, list[Turn]] = defaultdict(list)
    env_buckets: dict[int, list[Turn]] = defaultdict(list)
    env_text_by_block: dict[int, list[Turn]] = defaultdict(list)
    image_buckets: dict[int, list] = defaultdict(list)
    tool_action_by_block: dict[int, Turn] = {}
    tool_responses_by_block: dict[int, list[Turn]] = defaultdict(list)
    for turn in sample.turns:
        start = turn.start_ms or 0
        if turn.role == "assistant" and turn.images:
            raise ValueError(
                f"assistant image output is not supported: sample={sample.id!r}"
            )
        if turn.role == "tool" and turn.images:
            raise ValueError(f"tool response turn cannot carry media: sample={sample.id!r}")
        for image_ref in turn.images:
            frame_ms = image_ref.start_ms if image_ref.start_ms is not None else start
            if (
                not isinstance(frame_ms, (int, float))
                or isinstance(frame_ms, bool)
                or frame_ms < 0
            ):
                raise ValueError(
                    f"Invalid image start_ms={frame_ms!r} in sample={sample.id!r}"
                )
            image_buckets[int(frame_ms) // block_ms].append(image_ref)
        block = turn.meta.get("tool_block_index", start // block_ms)
        if not isinstance(block, int) or isinstance(block, bool) or block < 0:
            raise ValueError(
                f"Invalid tool_block_index={block!r} in sample={sample.id!r}"
            )
        if turn.tool_calls:
            if turn.role != "assistant":
                raise ValueError(
                    f"only assistant turns may emit tool calls: sample={sample.id!r}"
                )
            if (
                turn.text
                or turn.speech_out is not None
                or turn.audio_in is not None
                or turn.images
            ):
                raise ValueError(
                    "duplex tool calls are silent actions and cannot carry text/media: "
                    f"sample={sample.id!r}"
                )
            if block in tool_action_by_block:
                raise ValueError(
                    f"Multiple tool actions occupy block={block} in sample={sample.id!r}"
                )
            tool_action_by_block[block] = turn
            continue
        if turn.role == "tool" or turn.tool_response is not None:
            if turn.audio_in is not None or turn.speech_out is not None or turn.images:
                raise ValueError(
                    f"tool response turn cannot carry media: sample={sample.id!r}"
                )
            tool_responses_by_block[block].append(turn)
            continue
        end = turn.end_ms or start
        first = start // block_ms
        last = max(end - 1, start) // block_ms
        if turn.role not in {"assistant", "system"} and turn.text:
            # Official streaming_prefill accepts text_list beside video/audio. Text is an input
            # event, not a duration, so expose it exactly once when it arrives.
            #
            # A spoken turn is perceived through the microphone ONLY. Feeding its transcript
            # alongside the audio lets the model read the request instead of listening to it,
            # which no realtime deployment can reproduce: at inference the runtime has audio and
            # no transcript. Text arriving as a genuine typed/structured event (no audio on the
            # same turn) is still a legitimate input channel and stays.
            if turn.audio_in is None:
                env_text_by_block[first].append(turn)
        for b in range(first, last + 1):
            (agent_buckets if turn.role == "assistant" else env_buckets)[b].append(turn)

    interrupt_by_block: dict[int, Turn] = {}
    for turn in sample.turns:
        control = turn.meta.get(CONTROL_FIELD)
        if control is None:
            continue
        if control != INTERRUPT_CONTROL:
            raise ValueError(
                f"Unsupported duplex control {control!r} in sample={sample.id!r}"
            )
        block = turn.meta.get(CONTROL_BLOCK_FIELD)
        if not isinstance(block, int) or isinstance(block, bool) or block < 0:
            raise ValueError(
                f"Invalid {CONTROL_BLOCK_FIELD}={block!r} in sample={sample.id!r}"
            )
        if block in interrupt_by_block:
            raise ValueError(
                f"Multiple interrupt controls occupy block={block} in sample={sample.id!r}"
            )
        interrupt_by_block[block] = turn

    # ---- 2. Pre-plan each agent turn into TEXT-DRIVEN speak units ----
    # Thinker streaming is controlled by text/control tokens: every non-final speak unit carries K
    # text tokens, and the final unit carries the remaining text plus optional <|turn_eos|>. S3 codes
    # are attached only as Talker targets and must never create extra empty-text Thinker units. The
    # final text unit owns all remaining S3 codes; it is not limited to one nominal S3 unit.
    S = int(speech_tokens_per_unit)
    if S <= 0:
        S = K * codes_per_text_token
    turn_plans: list[dict] = []
    cur_turn_group = -1
    last_end_ms: int | None = None
    previous_turn_closed = True
    for turn in sorted((t for t in sample.turns if t.role == "assistant"), key=lambda t: (t.start_ms or 0)):
        text_ids = _enc(tokenizer, turn.text or "")
        if not text_ids:
            continue
        t_start = turn.start_ms or 0
        turn_complete = _is_complete(turn.turn_state)
        turn_interrupted = bool(turn.meta.get("is_interrupted"))
        turn_closed = turn_complete or turn_interrupted
        interrupt_block = turn.meta.get(INTERRUPT_BLOCK_FIELD)
        if turn_interrupted and (
            not isinstance(interrupt_block, int)
            or isinstance(interrupt_block, bool)
            or interrupt_block < 0
        ):
            raise ValueError(
                "Interrupted assistant turns require an explicit interrupt control block: "
                f"sample={sample.id!r}, value={interrupt_block!r}"
            )
        if not turn_interrupted and interrupt_block is not None:
            raise ValueError(
                "Only assistant turns marked is_interrupted may carry interrupt_block_index: "
                f"sample={sample.id!r}, value={interrupt_block!r}"
            )
        if (
            last_end_ms is None
            or previous_turn_closed
            or (t_start - last_end_ms) > turn_gap_ms
        ):
            cur_turn_group += 1
        last_end_ms = turn.end_ms or t_start
        previous_turn_closed = turn_closed
        s3 = turn.meta.get("s3_codes") or []
        text_chunks = [text_ids[i : i + K] for i in range(0, len(text_ids), K)]
        n_units = max(len(text_chunks), 1)
        code_slices = split_s3_codes_for_text_units(
            s3,
            text_units=n_units,
            speech_tokens_per_unit=S,
        )
        requested_blocks = turn.meta.get("speak_block_indices")
        if requested_blocks is not None:
            if (
                not isinstance(requested_blocks, list)
                or len(requested_blocks) != n_units
                or any(not isinstance(block, int) or block < 0 for block in requested_blocks)
                or any(left >= right for left, right in zip(requested_blocks, requested_blocks[1:]))
            ):
                raise ValueError(
                    f"Invalid speak_block_indices for sample={sample.id!r}: "
                    f"expected {n_units} strictly increasing non-negative integers, "
                    f"got {requested_blocks!r}"
                )
        supervised_blocks = turn.meta.get(SUPERVISED_SPEAK_BLOCKS_FIELD)
        if turn_interrupted:
            if requested_blocks is None:
                raise ValueError(
                    "Interrupted assistant turns require explicit speak_block_indices: "
                    f"sample={sample.id!r}"
                )
            expected_supervised = [
                block for block in requested_blocks if block < interrupt_block
            ]
            if supervised_blocks != expected_supervised:
                raise ValueError(
                    "Interrupted assistant supervised speak blocks do not match the causal prefix: "
                    f"sample={sample.id!r}, expected={expected_supervised}, "
                    f"got={supervised_blocks!r}"
                )
        elif supervised_blocks is not None and supervised_blocks != requested_blocks:
            raise ValueError(
                "Non-interrupted assistant cannot suppress planned speak blocks: "
                f"sample={sample.id!r}, planned={requested_blocks!r}, "
                f"supervised={supervised_blocks!r}"
            )
        turn_plans.append({
            "turn": turn,
            "turn_group": cur_turn_group,
            "start_block": t_start // block_ms,
            "text_chunks": text_chunks,
            "code_slices": code_slices,
            "n_units": n_units,
            "requested_blocks": requested_blocks,
            "supervised_blocks": supervised_blocks,
            "is_backchannel": (turn.turn_state or "").strip("<|>").lower() == "backchannel",
            "complete": turn_complete,
            "interrupt_block": interrupt_block,
        })

    # Assign each turn's speak units to consecutive blocks by default. A materialized corpus may
    # provide sparse explicit blocks to preserve a known late full-duplex event (for example, an
    # interruption during a long assistant waveform). Collisions still shift later units forward.
    speak_by_block: dict[int, dict] = {}   # block_index -> {plan, unit_idx}
    next_free = 0
    for plan in turn_plans:
        requested = plan["requested_blocks"]
        assigned = []
        if requested is None:
            b0 = max(plan["start_block"], next_free)
            assigned = list(range(b0, b0 + plan["n_units"]))
        else:
            for block in requested:
                assigned.append(max(block, next_free if not assigned else assigned[-1] + 1))
        if plan["interrupt_block"] is not None and assigned != requested:
            raise ValueError(
                "Interrupted assistant speak blocks cannot be shifted during serialization: "
                f"sample={sample.id!r}, requested={requested!r}, assigned={assigned!r}"
            )
        active_unit_indices: list[int] = []
        supervised_blocks = plan["supervised_blocks"]
        for unit_index, block in enumerate(assigned):
            if supervised_blocks is not None and block not in supervised_blocks:
                continue
            if block in interrupt_by_block:
                raise ValueError(
                    f"Speak and interrupt controls collide at block={block} "
                    f"in sample={sample.id!r}"
                )
            speak_by_block[block] = {"plan": plan, "unit_idx": unit_index}
            active_unit_indices.append(unit_index)
        plan["_assigned_start"] = assigned[0]
        plan["_assigned_blocks"] = assigned
        plan["_active_unit_indices"] = active_unit_indices
        # Planned suffixes at/after an interrupt control are masked counterfactual text, not emitted
        # speech. They must not delay a later assistant turn after the interrupting user is done.
        next_free = (
            int(plan["interrupt_block"]) + 1
            if plan["interrupt_block"] is not None
            else assigned[-1] + 1
        )

    # ---- 3. Block grid = 0 .. last block needed by any audio turn / env turn / speak unit ----
    max_block = -1
    for buckets in (agent_buckets, env_buckets):
        if buckets:
            max_block = max(max_block, max(buckets))
    if image_buckets:
        max_block = max(max_block, max(image_buckets))
    if env_text_by_block:
        max_block = max(max_block, max(env_text_by_block))
    if speak_by_block:
        max_block = max(max_block, max(speak_by_block))
    if interrupt_by_block:
        max_block = max(max_block, max(interrupt_by_block))
    if tool_action_by_block:
        max_block = max(max_block, max(tool_action_by_block))
    if tool_responses_by_block:
        max_block = max(max_block, max(tool_responses_by_block))
    augmented_timeline_end_ms = sample.meta.get("augmented_timeline_end_ms")
    if (
        isinstance(augmented_timeline_end_ms, (int, float))
        and not isinstance(augmented_timeline_end_ms, bool)
        and augmented_timeline_end_ms > 0
    ):
        max_block = max(
            max_block,
            max(int(math.ceil(float(augmented_timeline_end_ms) / block_ms)) - 1, 0),
        )
    blocks_range = list(range(max_block + 1))
    if truncate_block is not None:
        blocks_range = [b for b in blocks_range if b <= truncate_block]

    # ---- 4. Emit ----
    input_ids: list[int] = []
    labels: list[int] = []
    image_bounds: list[tuple[int, int]] = []
    image_inputs: list = []
    audio_bounds: list[tuple[int, int]] = []
    audio_inputs: list = []
    speech_segments: list[SpeechSegment] = []
    structured_spans: list[tuple[int, int]] = []
    resolved_system_prompt = system_prompt_with_tools(system_prompt, sample.tools)

    def emit(tids: list[int], supervise: bool) -> None:
        input_ids.extend(tids)
        labels.extend(tids if supervise else [IGNORE_INDEX] * len(tids))

    if include_system_prompt:
        # Match MiniCPMODuplex.prepare(): context mode registers the prefix first, then feeds the
        # suffix after the future `previous:` insertion point.
        emit(_enc(tokenizer, duplex_system_prompt_prefix(resolved_system_prompt)), False)
        if pinned_context:
            emit(_enc(tokenizer, pinned_context), False)
        emit(_enc(tokenizer, duplex_system_prompt_suffix()), False)

    # accumulate speech-unit records keyed by plan id (for SpeechSegment build after emission)
    plan_units: dict[int, list[dict]] = defaultdict(list)

    for b in blocks_range:
        block_start = b * block_ms
        block_end = (b + 1) * block_ms
        env_turns = sorted(env_buckets.get(b, []), key=lambda t: (t.start_ms or 0, t.channel or 0))
        agent_turns = sorted(agent_buckets.get(b, []), key=lambda t: (t.start_ms or 0))
        speak_here = speak_by_block.get(b)
        interrupt_here = interrupt_by_block.get(b)
        tool_action_here = tool_action_by_block.get(b)
        tool_responses_here = tool_responses_by_block.get(b, [])
        image_refs_here = image_buckets.get(b, [])
        text_inputs_here = env_text_by_block.get(b, [])

        if tool_action_here is not None and (
            speak_here is not None or interrupt_here is not None
        ):
            raise ValueError(
                f"tool action collides with dialogue action at block={b} in sample={sample.id!r}"
            )

        emit([UNIT_START.token_id], False)

        # --- perception: video frames, then microphone audio (official streaming order). ---
        # MiniCPM-o's low-latency duplex path fixes max_slice_nums=1. Each frame therefore maps to
        # one overview embedding of IMAGE_FEATURE_SIZE positions and never emits HD slice markers.
        for image_ref in image_refs_here:
            emit([IMAGE_START.token_id], False)
            image_start = len(input_ids)
            emit([_unk_id(tokenizer)] * IMAGE_FEATURE_SIZE, False)
            image_bounds.append((image_start, len(input_ids)))
            image_inputs.append(image_ref)
            emit([IMAGE_END.token_id], False)

        # The microphone remains continuous even when a structured Runtime event arrives.  Audio
        # placeholders are replaced by get_omni_embedding via audio_bounds; the masked tool
        # response is appended afterwards at the streaming protocol's optional-text position.
        ref, n = block_mic_audio(env_turns, block_start, block_end, geometry)
        if ref is not None and n > 0:
            ref.source = {
                **(ref.source or {}),
                "block_index": b,
                "timeline_start_ms": block_start,
                "timeline_end_ms": block_end,
            }
            if b in idle_gap_blocks and (ref.source or {}).get("kind") == "silence":
                ref.source["idle_gap"] = True
            audio_bounds.append((len(input_ids), len(input_ids) + n))
            emit([0] * n, False)
            audio_inputs.append(ref)

        # Official per-unit order is frame(s), audio, then optional text inputs. Preserve arriving
        # user text first, followed by bounded structured responses in their Runtime order.
        text_events = [turn.text or "" for turn in text_inputs_here]
        for turn in tool_responses_here:
            value = (
                turn.tool_response
                if turn.tool_response is not None
                else (turn.text or "")
            )
            text_events.append(format_tool_response(value))
        if text_events:
            emit(_enc(tokenizer, "\n".join(text_events)), False)

        # --- fifth native action branch: a complete tool call is silent and never reaches Talker. ---
        if tool_action_here is not None:
            structured_start = len(input_ids)
            emit(_enc(tokenizer, format_tool_calls(tool_action_here.tool_calls)), True)
            structured_spans.append((structured_start, len(input_ids)))
            emit([CHUNK_EOS.token_id], True)
        else:
            # --- dialogue control. Interrupt owns the complete control-only unit. ---
            if interrupt_here is not None:
                ls = INTERRUPT_CONTROL
            elif speak_here is not None:
                plan = speak_here["plan"]
                ls = BACKCHANNEL_CLASS if plan["is_backchannel"] else "speak"
            else:
                ls = "listen"
            emit([control_token_id(ls, tokenizer)], True)

        # --- speak unit: K text tokens (final may be shorter) + turn_eos on final ---
        if speak_here is not None:
            plan = speak_here["plan"]
            u = speak_here["unit_idx"]
            is_original_final = u == plan["n_units"] - 1
            is_effective_final = u == plan["_active_unit_indices"][-1]
            chunk = plan["text_chunks"][u]
            cond_positions: list[int] = []
            cond_ids: list[int] = []
            if chunk:
                start = len(input_ids)
                emit(chunk, True)
                cond_positions = list(range(start, start + len(chunk)))
                cond_ids = list(chunk)
            if is_original_final and plan["complete"]:
                emit([TURN_EOS.token_id], True)
                cond_positions = cond_positions + [len(input_ids) - 1]
                cond_ids = cond_ids + [TURN_EOS.token_id]
            plan_units[id(plan)].append({
                "plan": plan,
                "positions": cond_positions,
                "token_ids": cond_ids,
                "real_text_token_count": len(chunk),
                "s3_codes": plan["code_slices"][u],
                "start_ms": block_start,
                "end_ms": block_end,
                "is_turn_final": is_effective_final,
                "block_index": b,
            })
            emit([CHUNK_EOS.token_id], True)

        emit([UNIT_END.token_id], False)

    # ---- 5. One SpeechSegment per speak unit; same turn shares turn_group (continuous talker KV) ----
    for plan in turn_plans:
        turn = plan["turn"]
        if turn.speech_out is None:
            continue
        units = plan_units.get(id(plan), [])
        for unit in units:
            speech_segments.append(
                SpeechSegment(
                    batch_index=None,
                    text_token_positions=unit["positions"],
                    text_token_ids=unit["token_ids"],
                    audio_ref_id=turn.speech_out.id(),
                    s3_codes=unit["s3_codes"],
                    turn_group=plan["turn_group"],
                    is_turn_final=bool(unit["is_turn_final"]),
                    should_predict_audio_eos=bool(unit["is_turn_final"] and plan["complete"]),
                    real_text_token_count=int(unit["real_text_token_count"]),
                    unit_index=int(unit["block_index"]),
                    unit_start_ms=unit["start_ms"],
                    unit_end_ms=unit["end_ms"],
                    meta={"turn_state": turn.turn_state, "block": unit["block_index"], "layout": "duplex_unit"},
                )
            )


    # ---- 5. Left-truncate to max_seq_length (drop earliest tokens; keep alignment) ----
    if len(input_ids) > max_seq_length:
        offset = _left_truncation_offset(input_ids, max_seq_length, UNIT_START.token_id)
        if any(start < offset < end for start, end in structured_spans):
            raise ValueError(
                "max_seq_length would cut through a complete duplex tool action: "
                f"sample={sample.id!r}, offset={offset}"
            )
        input_ids = input_ids[offset:]
        labels = labels[offset:]
        image_bounds, image_inputs = _shift_image_pairs(
            image_bounds, image_inputs, offset
        )
        audio_bounds, audio_inputs = _shift_audio_pairs(audio_bounds, audio_inputs, offset)
        speech_segments = _shift_complete_speech_segments(speech_segments, offset)

    paradigm = "omniflow"
    serialized_meta = {
        **sample.meta,
        "caps": sample.caps.__dict__,
        "paradigm": paradigm,
        # Keep the exact rendered prompt, including per-row tool schemas. Sliding-context
        # reconstruction must not fall back to the unexpanded base prompt stored by materializers.
        "duplex_system_prompt": resolved_system_prompt,
    }
    if pinned_context:
        serialized_meta["pinned_context_tokens"] = len(_enc(tokenizer, pinned_context))
    playback_segments = _assistant_playback_segments(sample)
    if playback_segments:
        serialized_meta["assistant_playback_segments"] = playback_segments
    return SerializedSample(
        id=sample.id if truncate_block is None else f"{sample.id}#trunc{truncate_block}",
        input_ids=input_ids,
        labels=labels,
        image_bounds=image_bounds,
        image_inputs=image_inputs,
        audio_bounds=audio_bounds,
        audio_inputs=audio_inputs,
        speech_segments=speech_segments,
        unit_ids=unit_ids_from_boundaries(input_ids, UNIT_START.token_id),
        meta=serialized_meta,
    )

def _assistant_playback_segments(sample: OmniSample) -> list[dict]:
    segments: list[dict] = []
    forbidden_basenames = {"full_session.wav", "mixed_input.wav"}
    for turn in sample.turns:
        if turn.role != "assistant" or turn.start_ms is None:
            continue
        path = None
        source_start_ms = None
        source_end_ms = None
        channel = None
        origin = None
        if turn.speech_out is not None and turn.speech_out.path:
            path = turn.speech_out.path
            source_start_ms = turn.speech_out.start_ms
            source_end_ms = turn.speech_out.end_ms
            channel = turn.speech_out.channel
            origin = "speech_out"
        elif turn.meta.get("raw_agent_audio_path"):
            path = str(turn.meta["raw_agent_audio_path"])
            source_start_ms = turn.meta.get("raw_agent_audio_start_ms")
            source_end_ms = turn.meta.get("raw_agent_audio_end_ms")
            channel = turn.meta.get("raw_agent_audio_channel")
            origin = "raw_agent_audio"
        if not path or Path(path).name.lower() in forbidden_basenames:
            continue
        item = {
            "path": path,
            "origin": origin,
            "timeline_start_ms": int(turn.start_ms),
            "timeline_end_ms": (
                int(turn.end_ms) if turn.end_ms is not None else None
            ),
            "source_start_ms": source_start_ms,
            "source_end_ms": source_end_ms,
            "channel": channel,
        }
        segments.append(item)
    return segments
