from __future__ import annotations

import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Any, Sequence

from mcpmft.data.feature import AudioGeometry
from mcpmft.data.frontbrain_training import validate_frontbrain_training_sample
from mcpmft.data.idle_gap import (
    IDLE_GAP_META_FIELD,
    insert_pending_task_idle_gaps,
    insert_random_idle_gaps,
)
from mcpmft.data.labels import IGNORE_INDEX
from mcpmft.data.s3_target import S3TokenCache
from mcpmft.data.sliding_context import build_sampled_context_window
from mcpmft.data.tool_augmentation import (
    TOOL_AUGMENTATION_META_FIELD,
    augment_frontbrain_tool_context,
)


from mcpmft.data.serialize_omniflow import serialize_omniflow_sample
from mcpmft.data.serialize_turn import SerializedSample, serialize_turn_sample
from mcpmft.frontbrain.serialize import serialize_frontbrain_sample
from mcpmft.prompts import GANDER_DUPLEX_SYSTEM_PROMPT


LOGGER = logging.getLogger(__name__)
IDLE_GAP_BUDGET_META_FIELD = "idle_gap_budget"
TOOL_TOKEN_BUDGET_META_FIELD = "tool_token_budget"


def _serialized_unit_count(item: SerializedSample) -> int:
    ordinary_units = [unit_id for unit_id in item.unit_ids if unit_id >= 0]
    return max(ordinary_units, default=-1) + 1


@dataclass
class OmniCollator:
    tokenizer: Any
    paradigm: str = "turn"
    max_seq_length: int = 4096
    strict_no_truncation: bool = False
    s3_cache: S3TokenCache | None = None
    audio_geometry: AudioGeometry | None = None
    pad_to_multiple_of: int | None = 8
    audio_processor: Any = None  # processor with .audio_feature_extract for S2T audio input
    audio_augmenter: Any = None  # task-aware training-only microphone waveform augmenter
    enable_thinking: bool = False  # False = inject empty <think></think> prefix (official default)
    kv_delete_mask: bool = False
    kv_keep_previous_units: int = 128
    sliding_window_training: str = "off"
    context_max_units: int = 128
    context_previous_max_tokens: int = 500
    frontbrain_tool_protocol: str = "none"
    frontbrain_use_sample_pinned_context: bool = True
    frontbrain_pinned_context_requires_marker: bool = False
    frontbrain_business_tool_catalog: tuple[dict[str, Any], ...] = ()
    frontbrain_business_tool_forbidden_by_sample: dict[str, tuple[str, ...]] = field(
        default_factory=dict
    )
    frontbrain_business_tool_augmentation_probability: float = 0.0
    frontbrain_business_tool_augmentation_min: int = 1
    frontbrain_business_tool_augmentation_max: int = 3
    frontbrain_max_tools_per_sample: int = 6
    frontbrain_max_tool_schema_tokens: int = 1024
    turn_gap_ms: int = 2000             # max gap for continuing adjacent incomplete Talker chunks
    duplex_text_tokens_per_unit: int = 4
    codes_per_text_token: int = 6
    duplex_speech_tokens_per_unit: int = 25
    idle_gap_augmentation: bool = False
    include_duplex_system_prompt: bool = True
    duplex_system_prompt: str = GANDER_DUPLEX_SYSTEM_PROMPT
    seed: int = 42
    _sample_rng: random.Random | None = field(default=None, init=False, repr=False)
    _sample_rng_identity: tuple[int, int, int] | None = field(default=None, init=False, repr=False)

    def __call__(self, samples):
        if self.sliding_window_training in {"context", "mask_context"}:
            raise ValueError(
                f"sliding_window_training={self.sliding_window_training!r} is disabled because "
                "its shared previous copy exposes future assistant tokens. Use "
                "'window_no_previous', 'sampled_context', or 'context_memory'."
            )
        if self.sliding_window_training not in {
            "off",
            "window_no_previous",
            "sampled_context",
            "context_memory",
        }:
            raise ValueError(
                f"Unsupported sliding_window_training: {self.sliding_window_training!r}"
            )
        if (
            self.sliding_window_training in {"sampled_context", "context_memory"}
            and not self.include_duplex_system_prompt
        ):
            raise ValueError(
                f"sliding_window_training={self.sliding_window_training!r} requires "
                "include_duplex_system_prompt=true"
            )
        if self.sliding_window_training == "context_memory" and self.paradigm != "frontbrain":
            raise ValueError(
                "sliding_window_training='context_memory' requires paradigm='frontbrain'"
            )
        import torch

        serialized = [self._serialize_training_view(sample) for sample in samples]
        oversized = [item for item in serialized if len(item.input_ids) > self.max_seq_length]
        if oversized:
            item = oversized[0]
            budget = item.meta.get(IDLE_GAP_BUDGET_META_FIELD, {})
            if not isinstance(budget, dict):
                budget = {}
            source_detail = (
                " The online idle-gap augmentation was removed, but the original source view "
                "is still oversized."
                if budget.get("action") == "source_still_oversized"
                else ""
            )
            raise ValueError(
                f"Serialized sample {item.id!r} has {len(item.input_ids)} tokens, exceeding "
                f"max_seq_length={self.max_seq_length}. Strict/sliding-context serialization "
                f"never silently token-truncates.{source_detail}"
            )
        max_len = max(len(item.input_ids) for item in serialized)
        if self.pad_to_multiple_of:
            remainder = max_len % self.pad_to_multiple_of
            if remainder:
                max_len += self.pad_to_multiple_of - remainder
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id or 0

        input_ids = []
        labels = []
        attention_mask = []
        position_ids = []
        speech_segments = []
        image_bounds_batch = []
        audio_bounds_batch = []
        unit_ids_batch = []
        has_audio = False
        has_images = False
        has_units = False
        for batch_idx, item in enumerate(serialized):
            pad_len = max_len - len(item.input_ids)
            input_ids.append(item.input_ids + [pad_id] * pad_len)
            labels.append(item.labels + [IGNORE_INDEX] * pad_len)
            attention_mask.append([1] * len(item.input_ids) + [0] * pad_len)
            position_ids.append(list(range(max_len)))
            if len(item.image_bounds) != len(item.image_inputs):
                raise ValueError(
                    f"image bounds/input mismatch for sample {item.id!r}: "
                    f"{len(item.image_bounds)} != {len(item.image_inputs)}"
                )
            image_bounds_batch.append(item.image_bounds)
            has_images = has_images or bool(item.image_inputs)
            audio_bounds_batch.append(item.audio_bounds)
            if item.unit_ids:
                has_units = True
                unit_ids_batch.append(item.unit_ids + [-1] * pad_len)  # -1 = padding (masked)
            else:
                unit_ids_batch.append([-1] * max_len)
            has_audio = has_audio or bool(item.audio_inputs)
            for segment in item.speech_segments:
                segment.batch_index = batch_idx
                if self.s3_cache is not None and segment.s3_codes is None:
                    codes = self.s3_cache.get_by_id(segment.audio_ref_id)
                    if codes is None:
                        raise RuntimeError(
                            "Missing S3 codes for speech segment "
                            f"{segment.audio_ref_id!r}; materialize the target or embed "
                            "turn.meta['s3_codes'] in the manifest."
                        )
                    segment.s3_codes = codes
                speech_segments.append(segment.__dict__)

        batch = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "position_ids": torch.tensor(position_ids, dtype=torch.long),
            "speech_segments": speech_segments,
            "sample_ids": [item.id for item in serialized],
        }
        # Both full-supervision modes keep every target while hiding evicted ordinary units with
        # the runtime-equivalent last-K mask. PFC and Slate tokens stay in the protected prefix.
        use_kv_delete_mask = self.sliding_window_training in {
            "window_no_previous",
            "context_memory",
        } or (
            self.sliding_window_training == "off" and self.kv_delete_mask
        )
        if has_units and use_kv_delete_mask:
            batch["attention_mask"] = self._build_kv_delete_mask(
                torch.tensor(unit_ids_batch, dtype=torch.long),
                torch,
            )
        if has_images:
            if self.audio_processor is None:
                raise RuntimeError("Vision samples require a processor with process_image")
            self._attach_vision_features(batch, serialized, image_bounds_batch, torch)
        if has_audio and self.audio_processor is not None:
            self._attach_audio_features(batch, serialized, audio_bounds_batch, torch)
        return batch

    def _build_kv_delete_mask(
        self,
        unit_ids,
        torch,
    ):
        """4D additive mask simulating unit-level KV deletion without RoPE reindex.

        In full-supervision window modes, ``context_max_units=N`` means a query in unit u can
        attend ordinary-unit keys in [u-N, ..., u] plus every protected prefix/system/Slate key.
        Position ids remain absolute. The standalone compatibility flag uses
        ``kv_keep_previous_units``.
        """
        bsz, seqlen = unit_ids.shape
        dtype = torch.float32
        neg = torch.finfo(dtype).min
        ui = unit_ids.unsqueeze(2)  # [b, q, 1]
        uj = unit_ids.unsqueeze(1)  # [b, 1, k]
        q_idx = torch.arange(seqlen).view(1, seqlen, 1)
        k_idx = torch.arange(seqlen).view(1, 1, seqlen)
        causal = k_idx <= q_idx
        valid_query = ui != -1
        protected_key = uj < -1
        ordinary_query = ui >= 0
        ordinary_key = uj >= 0
        keep_prev = max(
            0,
            int(
                self.context_max_units
                if self.sliding_window_training
                in {"window_no_previous", "context_memory"}
                else self.kv_keep_previous_units
            ),
        )
        recent = ordinary_query & ordinary_key & (uj <= ui) & ((ui - uj) <= keep_prev)
        # Padding keys are excluded because they are neither protected nor ordinary.
        allow = causal & valid_query & (protected_key | recent)
        mask = torch.where(allow, torch.zeros((), dtype=dtype), torch.full((), neg, dtype=dtype))
        return mask.unsqueeze(1)

    def _attach_audio_features(self, batch, serialized, audio_bounds_batch, torch) -> None:
        """Attach turn audio or the exact MiniCPM-o realtime duplex frontend."""

        from mcpmft.data.feature import load_audio_ref_waveform
        from mcpmft.data.streaming_audio import (
            STREAMING_MAX_GROUP_UNITS,
            build_streaming_mel_batch,
        )

        per_sample_feats: list[list] = []
        per_sample_lens: list[list] = []
        per_sample_group_bounds: list[list[list[tuple[int, int]]]] = []
        streaming_audio = self.paradigm in {
            "omniflow",
            "frontbrain",
        }
        streaming_first_unit_masks: list[list[bool]] = []
        for item, item_bounds in zip(serialized, audio_bounds_batch):
            feats: list = []
            lens: list = []
            group_bounds_all: list[list[tuple[int, int]]] = []
            refs = item.audio_inputs
            if len(refs) != len(item_bounds):
                raise ValueError(
                    f"audio bounds/input mismatch for sample {item.id!r}: "
                    f"{len(item_bounds)} != {len(refs)}"
                )
            waveforms = []
            for ref_index, ref in enumerate(refs):
                try:
                    waveforms.append(load_audio_ref_waveform(ref))
                except Exception as exc:
                    source = getattr(ref, "source", None) or {}
                    location = getattr(ref, "path", None) or source.get("parquet")
                    raise RuntimeError(
                        f"Failed to load training audio sample_id={item.id!r} "
                        f"ref_index={ref_index} source_kind={source.get('kind')!r} "
                        f"location={location!r}"
                    ) from exc
            if self.audio_augmenter is not None:
                try:
                    waveforms = self.audio_augmenter.augment(
                        waveforms,
                        refs,
                        item.meta,
                        self._get_sample_rng(),
                    )
                except Exception as exc:
                    raise RuntimeError(
                        f"Failed to augment training audio sample_id={item.id!r}"
                    ) from exc
            if streaming_audio:
                feature_extractor = getattr(self.audio_processor, "audio_processor", None)
                if feature_extractor is None:
                    raise RuntimeError(
                        "Realtime duplex audio training requires processor.audio_processor"
                    )
                streaming = build_streaming_mel_batch(
                    waveforms,
                    feature_extractor,
                )
                unit_features = [
                    torch.from_numpy(feature).unsqueeze(0)
                    for feature in streaming.features
                ]
                for start in range(0, len(refs), STREAMING_MAX_GROUP_UNITS):
                    end = min(start + STREAMING_MAX_GROUP_UNITS, len(refs))
                    feats.extend(unit_features[start:end])
                    lens.extend(
                        torch.tensor(
                            [unit_features[index].shape[-1]],
                            dtype=torch.long,
                        )
                        for index in range(start, end)
                    )
                    group_bounds_all.append(
                        [tuple(bound) for bound in item_bounds[start:end]]
                    )
                per_sample_feats.append(feats)
                per_sample_lens.append(lens)
                per_sample_group_bounds.append(group_bounds_all)
                streaming_first_unit_masks.append(
                    streaming.first_unit_mask.tolist()
                )
                continue

            for waveform, bound in zip(waveforms, item_bounds):
                af, _, _ = self.audio_processor.audio_feature_extract(
                    [[waveform]], sampling_rate=16000, chunk_length=1
                )
                merged_mel = af[0] if af.dim() == 3 else af  # (80, F)
                if not bool(torch.isfinite(merged_mel).all().item()):
                    raise RuntimeError(
                        f"Non-finite audio features for training sample_id={item.id!r} "
                        f"audio_group={len(feats)}"
                    )
                feats.append(merged_mel.unsqueeze(0))
                lens.append(torch.tensor([merged_mel.shape[-1]], dtype=torch.long))
                group_bounds_all.append([tuple(bound)])
            per_sample_feats.append(feats)
            per_sample_lens.append(lens)
            per_sample_group_bounds.append(group_bounds_all)
            streaming_first_unit_masks.append([])

        # Flatten feature rows and preserve one length tensor per batch sample. The turn path
        # consumes these lengths through get_audio_embedding; the duplex path consumes the
        # explicit unit/group metadata below.
        flat_feats = [f for feats in per_sample_feats for f in feats]
        if not flat_feats:
            return
        max_frames = max(f.shape[-1] for f in flat_feats)
        padded = []
        for f in flat_feats:
            pad = max_frames - f.shape[-1]
            if pad:
                f = torch.nn.functional.pad(f, (0, pad))
            padded.append(f)
        batch["audio_features"] = torch.cat(padded, dim=0)
        # per-sample: concat that sample's per-audio length tensors into a single 1-D tensor.
        grouped_lens = []
        for lens in per_sample_lens:
            if lens:
                grouped_lens.append(torch.cat([ln.reshape(-1) for ln in lens]))
            else:
                grouped_lens.append(torch.zeros((0,), dtype=torch.long))
        batch["audio_feature_lens"] = grouped_lens
        batch["audio_bounds"] = [torch.tensor(b, dtype=torch.long) if b else torch.zeros((0, 2), dtype=torch.long) for b in audio_bounds_batch]
        batch["audio_group_bounds"] = per_sample_group_bounds
        if streaming_audio:
            batch["audio_streaming_first_unit_mask"] = torch.tensor(
                [
                    value
                    for sample_mask in streaming_first_unit_masks
                    for value in sample_mask
                ],
                dtype=torch.bool,
            )
            batch["audio_streaming_group_unit_counts"] = [
                [len(group) for group in sample_groups]
                for sample_groups in per_sample_group_bounds
            ]

    def _attach_vision_features(
        self,
        batch,
        serialized,
        image_bounds_batch,
        torch,
    ) -> None:
        """Process sampled video frames with MiniCPM-o's official realtime vision path."""

        from mcpmft.data.vision import load_image_refs
        from mcpmft.tokenizer_tools import IMAGE_FEATURE_SIZE

        process_image = getattr(self.audio_processor, "process_image", None)
        if not callable(process_image):
            raise RuntimeError("Vision samples require processor.process_image")

        images_batch = [
            load_image_refs(item.image_inputs)
            for item in serialized
        ]
        processed = process_image(
            images=images_batch,
            do_pad=True,
            max_slice_nums=1,
            return_tensors="pt",
        )
        pixel_values = processed["pixel_values"]
        tgt_sizes = processed["tgt_sizes"]
        if len(pixel_values) != len(serialized) or len(tgt_sizes) != len(serialized):
            raise ValueError(
                "vision processor batch mismatch: "
                f"pixel_values={len(pixel_values)}, tgt_sizes={len(tgt_sizes)}, "
                f"samples={len(serialized)}"
            )

        for item, bounds, pixels in zip(serialized, image_bounds_batch, pixel_values):
            invalid_bounds = [bound for bound in bounds if bound[1] - bound[0] != IMAGE_FEATURE_SIZE]
            if invalid_bounds:
                raise ValueError(
                    f"image placeholder length mismatch for sample {item.id!r}: {invalid_bounds}"
                )
            if len(pixels) != len(bounds):
                raise ValueError(
                    "max_slice_nums=1 must produce one vision tensor per frame: "
                    f"sample={item.id!r}, tensors={len(pixels)}, bounds={len(bounds)}"
                )

        batch["pixel_values"] = pixel_values
        batch["tgt_sizes"] = tgt_sizes
        batch["image_bound"] = [
            torch.tensor(bounds, dtype=torch.long)
            if bounds
            else torch.zeros((0, 2), dtype=torch.long)
            for bounds in image_bounds_batch
        ]

    def _sample_context_window(self, item: SerializedSample) -> SerializedSample:
        ordinary_units = sorted({unit_id for unit_id in item.unit_ids if unit_id >= 0})
        if not ordinary_units:
            raise ValueError(
                f"sliding_window_training={self.sliding_window_training!r} requires the "
                "omniflow or frontbrain "
                f"unit serializer; sample {item.id!r} has no unit boundaries"
            )

        keep_units = max(0, int(self.context_max_units))
        # K retained completed units plus the current unit still fits the initial causal stream.
        # The first target that actually follows an eviction is ordinal K + 1.
        first_evicted_target = keep_units + 1
        if len(ordinary_units) <= first_evicted_target:
            return item

        target_unit = self._get_sample_rng().choice(ordinary_units[first_evicted_target:])
        return build_sampled_context_window(
            item,
            self.tokenizer,
            target_unit=target_unit,
            context_max_units=keep_units,
            context_previous_max_tokens=self.context_previous_max_tokens,
            max_seq_length=self.max_seq_length,
            system_prompt=str(
                item.meta.get("duplex_system_prompt") or self.duplex_system_prompt
            ),
        )

    def _get_sample_rng(self) -> random.Random:
        import torch

        worker = torch.utils.data.get_worker_info()
        rank = int(os.environ.get("RANK", "0"))
        worker_id = worker.id if worker is not None else -1
        identity = (os.getpid(), rank, worker_id)
        if self._sample_rng is None or self._sample_rng_identity != identity:
            worker_seed = int(worker.seed) if worker is not None else 0
            combined_seed = int(self.seed) + rank * 1_000_003 + worker_seed
            self._sample_rng = random.Random(combined_seed)
            self._sample_rng_identity = identity
        return self._sample_rng

    def _augment_tool_context(
        self,
        sample,
        *,
        business_tool_probability: float | None = None,
    ):
        """Build the exact runtime-visible tool context for one training draw."""
        return augment_frontbrain_tool_context(
            sample,
            protocol=self.frontbrain_tool_protocol,
            business_tool_catalog=self.frontbrain_business_tool_catalog,
            business_tool_probability=(
                self.frontbrain_business_tool_augmentation_probability
                if business_tool_probability is None
                else business_tool_probability
            ),
            business_tool_min=self.frontbrain_business_tool_augmentation_min,
            business_tool_max=self.frontbrain_business_tool_augmentation_max,
            max_tools=self.frontbrain_max_tools_per_sample,
            tokenizer=self.tokenizer,
            max_schema_tokens=self.frontbrain_max_tool_schema_tokens,
            forbidden_tool_names=self.frontbrain_business_tool_forbidden_by_sample.get(
                sample.id, ()
            ),
            rng=self._get_sample_rng(),
        )

    def _serialize_training_view(self, sample) -> SerializedSample:
        source_sample = sample
        sample = self._augment_tool_context(source_sample)
        item = self._serialize_augmented_training_view(sample)
        augmentation = sample.meta.get(TOOL_AUGMENTATION_META_FIELD) or {}
        added_tools = augmentation.get("business_tools_added") or ()
        if len(item.input_ids) <= self.max_seq_length or not added_tools:
            return item

        # Business-tool distractors are optional context augmentation. A long but otherwise valid
        # source row must not become a stochastic distributed-training failure merely because one
        # draw selected a larger schema set. Replay the same row with the mandatory task trio and
        # its source-required business tools only. Source rows that still exceed the hard token
        # budget remain invalid and are rejected before the batch reaches the model.
        mandatory_sample = self._augment_tool_context(
            source_sample,
            business_tool_probability=0.0,
        )
        mandatory_item = self._serialize_augmented_training_view(mandatory_sample)
        mandatory_item.meta = {
            **mandatory_item.meta,
            TOOL_TOKEN_BUDGET_META_FIELD: {
                "action": (
                    "dropped_optional_business_tools"
                    if len(mandatory_item.input_ids) <= self.max_seq_length
                    else "mandatory_tool_context_still_oversized"
                ),
                "max_seq_length": int(self.max_seq_length),
                "augmented_tokens": len(item.input_ids),
                "mandatory_tokens": len(mandatory_item.input_ids),
                "dropped_business_tools": list(added_tools),
            },
        }
        return mandatory_item

    def _serialize_augmented_training_view(self, sample) -> SerializedSample:
        """Serialize a sample whose runtime-visible tool context is already fixed."""
        item = self._serialize(sample)
        unit_cap = self._idle_gap_unit_cap(sample)
        if unit_cap is not None:
            augmented_units = _serialized_unit_count(item)
            if augmented_units > unit_cap:
                pending_only = self._serialize(
                    sample,
                    apply_idle_gap=False,
                    apply_pending_task_gap=True,
                )
                pending_meta = pending_only.meta.get(IDLE_GAP_META_FIELD) or {}
                pending_applied = bool(pending_meta.get("pending_tasks"))
                pending_units = _serialized_unit_count(pending_only)
                if pending_applied:
                    # The unit cap budgets optional gaps. A pending-task wait is causal
                    # supervision, so preserve it even when the source already exceeds the cap.
                    action = (
                        "used_pending_task_timeline_unit_cap"
                        if pending_units <= unit_cap
                        else "used_pending_task_timeline_over_unit_cap"
                    )
                    pending_only.meta = {
                        **pending_only.meta,
                        IDLE_GAP_BUDGET_META_FIELD: {
                            "action": action,
                            "max_timeline_units": unit_cap,
                            "augmented_units": augmented_units,
                            "pending_task_units": pending_units,
                        },
                    }
                    item = pending_only
                else:
                    original = self._serialize(
                        sample,
                        apply_idle_gap=False,
                        apply_pending_task_gap=False,
                    )
                    original_units = _serialized_unit_count(original)
                    # ``max_timeline_units`` is an augmentation budget, not a source-row
                    # validity limit.  In window_no_previous mode an arbitrary-length source
                    # timeline is valid: the 4-D attention mask keeps at most
                    # ``context_max_units`` completed units visible to each query.  If the source
                    # itself is already beyond the optional augmentation budget, preserve it
                    # unchanged and let the independent max_seq_length guard decide whether it
                    # is computationally trainable.
                    action = (
                        "used_original_timeline_unit_cap"
                        if original_units <= unit_cap
                        else "used_source_timeline_over_augmentation_cap"
                    )
                    original.meta = {
                        **original.meta,
                        IDLE_GAP_BUDGET_META_FIELD: {
                            "action": action,
                            "max_timeline_units": unit_cap,
                            "augmented_units": augmented_units,
                            "original_units": original_units,
                        },
                    }
                    item = original
        if self.sliding_window_training == "sampled_context":
            item = self._sample_context_window(item)
        if not self.idle_gap_augmentation or len(item.input_ids) <= self.max_seq_length:
            return item

        augmented_tokens = len(item.input_ids)
        pending_only = self._serialize(
            sample,
            apply_idle_gap=False,
            apply_pending_task_gap=True,
        )
        pending_meta = pending_only.meta.get(IDLE_GAP_META_FIELD) or {}
        if pending_meta.get("pending_tasks"):
            pending_tokens = len(pending_only.input_ids)
            sampled_pending_tokens = pending_tokens
            if pending_tokens > self.max_seq_length:
                timeline_policy = (
                    self.audio_augmenter.resolve_profile(sample.meta).timeline
                    if self.audio_augmenter is not None
                    else None
                )
                pending_gap_bands = (
                    timeline_policy.pending_task_gap_bands
                    if timeline_policy is not None
                    else ()
                )
                minimum_pending_units = min(
                    (
                        int(band[0])
                        for band in pending_gap_bands
                        if float(band[2]) > 0
                    ),
                    default=0,
                )
                if minimum_pending_units > 0:
                    minimum_pending = self._serialize(
                        sample,
                        apply_idle_gap=False,
                        apply_pending_task_gap=True,
                        pending_task_gap_bands_override=(
                            (minimum_pending_units, minimum_pending_units, 1.0),
                        ),
                    )
                    minimum_meta = minimum_pending.meta.get(IDLE_GAP_META_FIELD) or {}
                    if minimum_meta.get("pending_tasks"):
                        pending_only = minimum_pending
                        pending_tokens = len(pending_only.input_ids)
            action = (
                "used_pending_task_timeline"
                if pending_tokens == sampled_pending_tokens
                and pending_tokens <= self.max_seq_length
                else "used_minimum_pending_task_timeline"
                if pending_tokens <= self.max_seq_length
                else "pending_task_timeline_exceeds_max_seq_length"
            )
            pending_only.meta = {
                **pending_only.meta,
                IDLE_GAP_BUDGET_META_FIELD: {
                    "action": action,
                    "max_seq_length": int(self.max_seq_length),
                    "augmented_tokens": augmented_tokens,
                    "sampled_pending_task_tokens": sampled_pending_tokens,
                    "pending_task_tokens": pending_tokens,
                },
            }
            return pending_only
        original = self._serialize(
            sample,
            apply_idle_gap=False,
            apply_pending_task_gap=False,
        )
        if self.sliding_window_training == "sampled_context":
            original = self._sample_context_window(original)
        action = (
            "used_original_timeline"
            if len(original.input_ids) <= self.max_seq_length
            else "source_still_oversized"
        )
        original.meta = {
            **original.meta,
            IDLE_GAP_BUDGET_META_FIELD: {
                "action": action,
                "max_seq_length": int(self.max_seq_length),
                "augmented_tokens": augmented_tokens,
                "original_tokens": len(original.input_ids),
            },
        }
        if action == "used_original_timeline":
            LOGGER.warning(
                "Skipping optional idle-gap augmentation for sample %r: augmented_tokens=%s "
                "would exceed max_seq_length=%s; using the original %s-token timeline",
                original.id,
                augmented_tokens,
                self.max_seq_length,
                len(original.input_ids),
            )
        return original

    def _idle_gap_unit_cap(self, sample) -> int | None:
        if not self.idle_gap_augmentation or self.audio_augmenter is None:
            return None
        value = self.audio_augmenter.resolve_profile(sample.meta).timeline.max_timeline_units
        return int(value) if value is not None else None

    def _serialize(
        self,
        sample,
        *,
        apply_idle_gap: bool = True,
        apply_pending_task_gap: bool | None = None,
        pending_task_gap_bands_override: Sequence[Sequence[float]] | None = None,
    ) -> SerializedSample:
        use_sample_pinned_context = self.frontbrain_use_sample_pinned_context
        if (
            use_sample_pinned_context
            and self.frontbrain_pinned_context_requires_marker
            and sample.meta.get("requires_pinned_context") is not True
        ):
            use_sample_pinned_context = False
        if self.paradigm == "frontbrain":
            validate_frontbrain_training_sample(
                sample,
                protocol=self.frontbrain_tool_protocol,
                use_sample_pinned_context=use_sample_pinned_context,
            )
        if apply_pending_task_gap is None:
            apply_pending_task_gap = apply_idle_gap
        if self.idle_gap_augmentation and (apply_idle_gap or apply_pending_task_gap):
            if self.paradigm not in {"omniflow", "frontbrain"}:
                raise ValueError(
                    "Idle-gap augmentation is only supported by omniflow/frontbrain serialization"
                )
            timeline_policy = (
                self.audio_augmenter.resolve_profile(sample.meta).timeline
                if self.audio_augmenter is not None
                else None
            )
            if apply_idle_gap:
                sample = insert_random_idle_gaps(
                    sample,
                    seed=self.seed,
                    turn_gap_ms=self.turn_gap_ms,
                    timeline_policy=timeline_policy,
                    tokenizer=self.tokenizer,
                    text_tokens_per_unit=self.duplex_text_tokens_per_unit,
                )
            elif (
                timeline_policy is not None
                and (
                    pending_task_gap_bands_override
                    or timeline_policy.pending_task_gap_bands
                )
            ):
                sample = insert_pending_task_idle_gaps(
                    sample,
                    seed=self.seed,
                    gap_bands=(
                        pending_task_gap_bands_override
                        or timeline_policy.pending_task_gap_bands
                    ),
                    tokenizer=self.tokenizer,
                    text_tokens_per_unit=self.duplex_text_tokens_per_unit,
                )
        if self.s3_cache is not None and self.paradigm in {
            "omniflow",
            "frontbrain",
        }:
            self._attach_s3_codes_before_serialization(sample)
        if self.strict_no_truncation or self.sliding_window_training in {
            "window_no_previous",
            "sampled_context",
            "context_memory",
        }:
            serialization_max_length = sys.maxsize
        else:
            serialization_max_length = self.max_seq_length
        # The run config is authoritative. Source-row prompt metadata is provenance from the
        # release that created the row and must not split one training run across prompt versions.
        sample_system_prompt = str(self.duplex_system_prompt)
        if self.paradigm == "omniflow":
            return serialize_omniflow_sample(
                sample,
                self.tokenizer,
                max_seq_length=serialization_max_length,
                audio_geometry=self.audio_geometry,
                turn_gap_ms=self.turn_gap_ms,
                text_tokens_per_block=self.duplex_text_tokens_per_unit,
                codes_per_text_token=self.codes_per_text_token,
                speech_tokens_per_unit=self.duplex_speech_tokens_per_unit,
                include_system_prompt=self.include_duplex_system_prompt,
                system_prompt=sample_system_prompt,
            )
        if self.paradigm == "frontbrain":
            return serialize_frontbrain_sample(
                sample,
                self.tokenizer,
                max_seq_length=serialization_max_length,
                audio_geometry=self.audio_geometry,
                turn_gap_ms=self.turn_gap_ms,
                text_tokens_per_block=self.duplex_text_tokens_per_unit,
                codes_per_text_token=self.codes_per_text_token,
                speech_tokens_per_unit=self.duplex_speech_tokens_per_unit,
                include_system_prompt=self.include_duplex_system_prompt,
                system_prompt=sample_system_prompt,
                pinned_context=(
                    None if use_sample_pinned_context else ""
                ),
            )
        return serialize_turn_sample(
            sample,
            self.tokenizer,
            max_seq_length=serialization_max_length,
            audio_geometry=self.audio_geometry,
            enable_thinking=self.enable_thinking,
        )

    def _attach_s3_codes_before_serialization(self, sample) -> None:
        """Load turn-level S3 codes before duplex unit planning.

        The duplex serializer is text-driven, but S3 codes still need to be present before
        serialization so every emitted speak unit can receive its Talker target slice and the final
        unit can retain the full remainder. Loading cache entries after serialization leaves every
        unit with an empty target for manifests that do not embed ``turn.meta["s3_codes"]``.
        """
        assert self.s3_cache is not None
        for turn in getattr(sample, "turns", []):
            if turn.role != "assistant" or turn.speech_out is None:
                continue
            codes = turn.meta.get("s3_codes") if isinstance(turn.meta, dict) else None
            if isinstance(codes, list) and codes:
                continue
            loaded = self.s3_cache.get(turn.speech_out)
            if loaded is None:
                raise RuntimeError(
                    "Missing S3 codes for assistant speech "
                    f"{turn.speech_out.id()!r}; materialize the target or embed "
                    "turn.meta['s3_codes'] in the manifest."
                )
            turn.meta["s3_codes"] = loaded
