from __future__ import annotations

import json
import logging
import math
from pathlib import Path

from mcpmft.args import parse_project_config, write_resolved_config
from mcpmft.data.audio_augment import AudioAugmenter
from mcpmft.data.augmentation_profile import ProfileResolver
from mcpmft.data.collator import OmniCollator
from mcpmft.data.dataset import ManifestDataset, validate_training_schedule
from mcpmft.data.feature import AudioGeometry
from mcpmft.data.media import ReleaseMediaResolver
from mcpmft.data.s3_target import S3TokenCache, require_complete_s3_cache
from mcpmft.data.streaming_audio import (
    STREAMING_AUDIO_POOL_STEP,
    STREAMING_CHUNK_MS,
    STREAMING_HOP_LENGTH,
    STREAMING_SAMPLE_RATE,
)
from mcpmft.data.tool_augmentation import (
    filter_business_tool_catalog_for_realtime,
    load_business_tool_augmentation_catalog,
)
from mcpmft.modeling.freeze import apply_freeze, param_stats
from mcpmft.modeling.load import (
    load_minicpmo_model,
    load_selected_state_dict,
    load_tokenizer_and_processor,
)
from mcpmft.modeling.omni_forward import OmniTrainWrapper
from mcpmft.tokenizer_tools import (
    AUDIO_BOS_ID,
    S3_EOS_ID,
    S3_NUM_AUDIO_TOKENS,
    assert_minicpmo_tokenizer,
)
from mcpmft.train.run_state import validate_run_directory
from mcpmft.utils.logging import is_rank_zero, rank_zero_info, setup_logging

LOGGER = logging.getLogger(__name__)


def _validate_training_mode(config) -> None:
    mode = config.train.mode
    if mode == "custom":
        return

    checks = {
        "path.paradigm=frontbrain": config.path.paradigm == "frontbrain",
        "model.init_audio=true": config.model.init_audio is True,
        "path.enable_s2t=true": config.path.enable_s2t is True,
        "path.enable_omniflow=true": config.path.enable_omniflow is True,
        "freeze.tune_vision=false": config.freeze.tune_vision is False,
        "freeze.tune_resampler=false": config.freeze.tune_resampler is False,
        "freeze.tune_audio_encoder=false": config.freeze.tune_audio_encoder is False,
        "train.save_trainable_only=true": config.train.save_trainable_only is True,
    }
    if mode == "thinker":
        checks.update(
            {
                "model.init_tts=false": config.model.init_tts is False,
                "path.enable_t2s=false": config.path.enable_t2s is False,
                "freeze.tune_audio_proj=true": config.freeze.tune_audio_proj is True,
                "freeze.tune_llm=true": config.freeze.tune_llm is True,
                "freeze.tune_tts_proj=false": config.freeze.tune_tts_proj is False,
                "freeze.tune_tts_decoder=false": config.freeze.tune_tts_decoder is False,
                "train.text_loss_weight>0": config.train.text_loss_weight > 0,
                "train.audio_loss_weight=0": config.train.audio_loss_weight == 0,
            }
        )
    elif mode == "talker":
        checks.update(
            {
                "model.init_vision=false": config.model.init_vision is False,
                "model.init_tts=true": config.model.init_tts is True,
                "path.enable_vision=false": config.path.enable_vision is False,
                "path.enable_t2s=true": config.path.enable_t2s is True,
                "freeze.tune_audio_proj=false": config.freeze.tune_audio_proj is False,
                "freeze.tune_llm=false": config.freeze.tune_llm is False,
                "freeze.tune_tts_proj=true": config.freeze.tune_tts_proj is True,
                "freeze.tune_tts_decoder=true": config.freeze.tune_tts_decoder is True,
                "talker.detach_llm_for_tts=always": (
                    config.talker.detach_llm_for_tts == "always"
                ),
                "train.text_loss_weight=0": config.train.text_loss_weight == 0,
                "train.audio_loss_weight>0": config.train.audio_loss_weight > 0,
                "train.control_loss_weight=0": config.train.control_loss_weight == 0,
                "runtime.init_checkpoint": bool(config.runtime.init_checkpoint),
                "runtime.init_checkpoint_prefixes": (
                    config.runtime.init_checkpoint_prefixes
                    == ["llm.", "audio_projection_layer."]
                ),
            }
        )
    elif mode == "joint":
        checks.update(
            {
                "model.init_tts=true": config.model.init_tts is True,
                "path.enable_t2s=true": config.path.enable_t2s is True,
                "freeze.tune_audio_proj=true": config.freeze.tune_audio_proj is True,
                "freeze.tune_llm=true": config.freeze.tune_llm is True,
                "freeze.tune_tts_proj=true": config.freeze.tune_tts_proj is True,
                "freeze.tune_tts_decoder=true": config.freeze.tune_tts_decoder is True,
                "talker.detach_llm_for_tts!=always": (
                    config.talker.detach_llm_for_tts != "always"
                ),
                "train.text_loss_weight>0": config.train.text_loss_weight > 0,
                "train.audio_loss_weight>0": config.train.audio_loss_weight > 0,
            }
        )
    failed = [name for name, valid in checks.items() if not valid]
    if failed:
        raise ValueError(f"train.mode={mode!r} contract mismatch: {failed}")


def _validate_weighted_bands(
    bands,
    field_name: str,
    *,
    positive_bounds: bool = False,
) -> None:
    if not bands:
        raise ValueError(f"{field_name} cannot be empty")
    total_weight = 0.0
    for band in bands:
        if len(band) != 3:
            raise ValueError(f"{field_name} rows must be [min, max, weight]")
        low, high, weight = (float(value) for value in band)
        if not all(math.isfinite(value) for value in (low, high, weight)):
            raise ValueError(f"{field_name} must contain finite values")
        if high < low or weight < 0 or (positive_bounds and low <= 0):
            raise ValueError(f"{field_name} contains an invalid band: {band}")
        total_weight += weight
    if total_weight <= 0:
        raise ValueError(f"{field_name} must have positive band weight")


def _validate_finite_pair(values, field_name: str) -> None:
    if (
        not isinstance(values, (list, tuple))
        or len(values) != 2
        or not all(math.isfinite(float(value)) for value in values)
        or float(values[1]) < float(values[0])
    ):
        raise ValueError(f"{field_name} must be a finite [min, max] pair")


def validate_project_config(config) -> None:
    _validate_training_mode(config)
    talker_only = (
        config.train.audio_loss_weight > 0
        and config.train.text_loss_weight == 0
        and config.train.control_loss_weight == 0
    )
    numeric_positive = {
        "audio.sample_rate": config.audio.sample_rate,
        "audio.audio_chunk_length": config.audio.audio_chunk_length,
        "audio.whisper_hop_length": config.audio.whisper_hop_length,
        "audio.audio_pool_step": config.audio.audio_pool_step,
        "data.max_seq_length": config.data.max_seq_length,
        "data.duplex_text_tokens_per_unit": config.data.duplex_text_tokens_per_unit,
        "data.shuffle_block_bytes": config.data.shuffle_block_bytes,
        "talker.codes_per_text_token": config.talker.codes_per_text_token,
        "talker.max_talker_tokens": config.talker.max_talker_tokens,
        "train.per_device_train_batch_size": config.train.per_device_train_batch_size,
        "train.gradient_accumulation_steps": config.train.gradient_accumulation_steps,
        "train.logging_steps": config.train.logging_steps,
        "train.save_steps": config.train.save_steps,
    }
    invalid_positive = {
        name: value for name, value in numeric_positive.items() if value <= 0
    }
    if invalid_positive:
        raise ValueError(f"Training lengths/counts must be positive: {invalid_positive}")
    if config.path.paradigm in {"omniflow", "frontbrain"} and config.path.enable_s2t:
        expected_audio = {
            "audio.sample_rate": STREAMING_SAMPLE_RATE,
            "audio.audio_chunk_length": STREAMING_CHUNK_MS // 1000,
            "audio.whisper_hop_length": STREAMING_HOP_LENGTH,
            "audio.audio_pool_step": STREAMING_AUDIO_POOL_STEP,
        }
        actual_audio = {
            "audio.sample_rate": config.audio.sample_rate,
            "audio.audio_chunk_length": config.audio.audio_chunk_length,
            "audio.whisper_hop_length": config.audio.whisper_hop_length,
            "audio.audio_pool_step": config.audio.audio_pool_step,
        }
        mismatched_audio = {
            name: {"expected": expected_audio[name], "actual": value}
            for name, value in actual_audio.items()
            if value != expected_audio[name]
        }
        if mismatched_audio:
            raise ValueError(
                "Realtime duplex training must use the MiniCPM-o 4.5 streaming audio "
                f"contract: {mismatched_audio}"
            )
    if config.data.duplex_speech_tokens_per_unit < 0:
        raise ValueError("data.duplex_speech_tokens_per_unit must be non-negative")
    if config.data.context_training_semantics != "all_unit_4d_mask_v1":
        raise ValueError(
            "data.context_training_semantics must be all_unit_4d_mask_v1"
        )
    if config.data.frontbrain_tool_protocol == "task_tools_v1":
        if config.path.paradigm != "frontbrain":
            raise ValueError(
                "data.frontbrain_tool_protocol=task_tools_v1 requires path.paradigm=frontbrain"
            )
        if not config.data.include_duplex_system_prompt:
            raise ValueError(
                "task_tools_v1 requires data.include_duplex_system_prompt=true because the "
                "three task schemas are unconditional runtime context"
            )
        probability = float(
            config.data.frontbrain_business_tool_augmentation_probability
        )
        business_min = int(config.data.frontbrain_business_tool_augmentation_min)
        business_max = int(config.data.frontbrain_business_tool_augmentation_max)
        max_tools = int(config.data.frontbrain_max_tools_per_sample)
        if not 0.0 <= probability <= 1.0:
            raise ValueError(
                "data.frontbrain_business_tool_augmentation_probability must be in [0, 1]"
            )
        if business_min < 0 or business_max < business_min:
            raise ValueError(
                "frontbrain business-tool bounds must satisfy 0 <= min <= max"
            )
        if max_tools < 3:
            raise ValueError(
                "data.frontbrain_max_tools_per_sample must be at least 3 for task_tools_v1"
            )
        if config.data.frontbrain_max_tool_schema_tokens < 1:
            raise ValueError(
                "data.frontbrain_max_tool_schema_tokens must be positive"
            )
        if business_max > max_tools - 3:
            raise ValueError(
                "frontbrain business-tool max exceeds the capacity left after the three "
                "always-visible task schemas"
            )
        if probability > 0 and not config.data.frontbrain_business_tool_catalog_path:
            raise ValueError(
                "business-tool augmentation requires "
                "data.frontbrain_business_tool_catalog_path"
            )
        # Both policies supervise the full causal stream with a last-K attention mask;
        # context_memory also retains the PFC/Slate prefix.
        if config.data.sliding_window_training not in {
            "context_memory",
            "window_no_previous",
        }:
            raise ValueError(
                "task_tools_v1 mixed training requires data.sliding_window_training="
                "context_memory or window_no_previous"
            )
        if (
            config.data.sliding_window_training == "window_no_previous"
            and config.data.frontbrain_use_sample_pinned_context
        ):
            raise ValueError(
                "data.sliding_window_training=window_no_previous cannot serve a pinned static "
                "slate; set data.frontbrain_use_sample_pinned_context=false or use context_memory"
            )
        if (
            config.data.frontbrain_pinned_context_requires_marker
            and not config.data.frontbrain_use_sample_pinned_context
        ):
            raise ValueError(
                "data.frontbrain_pinned_context_requires_marker=true requires "
                "data.frontbrain_use_sample_pinned_context=true"
            )
        if not talker_only and (
            not config.model.init_vision or not config.path.enable_vision
        ):
            raise ValueError(
                "task_tools_v1 mixed training requires the MiniCPM-o vision path: "
                "model.init_vision=true and path.enable_vision=true"
            )
        if (
            not config.model.init_audio
            or not config.path.enable_s2t
            or not config.path.enable_omniflow
        ):
            raise ValueError(
                "task_tools_v1 mixed training requires realtime audio perception: "
                "model.init_audio=true, path.enable_s2t=true, and path.enable_omniflow=true"
            )
        required_task_calls = list(config.data.frontbrain_required_task_calls)
        allowed_task_calls = {"task_start", "task_send", "task_resolve"}
        if (
            not required_task_calls
            or len(required_task_calls) != len(set(required_task_calls))
            or not set(required_task_calls).issubset(allowed_task_calls)
        ):
            raise ValueError(
                "data.frontbrain_required_task_calls must be a non-empty unique subset of "
                f"{sorted(allowed_task_calls)}"
            )
        if (
            config.data.frontbrain_require_vision_source
            and not config.data.frontbrain_omni_manifest_paths
        ):
            raise ValueError(
                "task_tools_v1 mixed training requires explicit timestamped audio-video "
                "sources in data.frontbrain_omni_manifest_paths unless "
                "data.frontbrain_require_vision_source=false is selected"
            )
        if not config.data.frontbrain_tool_manifest_paths:
            raise ValueError(
                "task_tools_v1 mixed training requires explicit "
                "data.frontbrain_tool_manifest_paths"
            )
        missing_omni_sources = sorted(
            set(config.data.frontbrain_omni_manifest_paths)
            - set(config.data.manifest_paths)
        )
        if missing_omni_sources:
            raise ValueError(
                "frontbrain multimodal manifests must also be present in data.manifest_paths: "
                f"{missing_omni_sources}"
            )
        missing_tool_sources = sorted(
            set(config.data.frontbrain_tool_manifest_paths)
            - set(config.data.manifest_paths)
        )
        if missing_tool_sources:
            raise ValueError(
                "frontbrain tool manifests must also be present in data.manifest_paths: "
                f"{missing_tool_sources}"
            )
        overlap = sorted(
            set(config.data.frontbrain_omni_manifest_paths)
            & set(config.data.frontbrain_tool_manifest_paths)
        )
        if overlap:
            raise ValueError(
                "audio-video multimodal and task-tool manifest sets must be disjoint: "
                f"{overlap}"
            )
        speech_sources = sorted(
            set(config.data.manifest_paths)
            - set(config.data.frontbrain_omni_manifest_paths)
            - set(config.data.frontbrain_tool_manifest_paths)
        )
        if not speech_sources and not talker_only:
            raise ValueError(
                "task_tools_v1 training requires at least one ordinary speech manifest in "
                "addition to audio-video multimodal and task-tool manifests"
            )
    elif (
        config.data.frontbrain_omni_manifest_paths
        or config.data.frontbrain_tool_manifest_paths
        or not config.data.frontbrain_require_vision_source
        or config.data.frontbrain_required_task_calls
        != ["task_start", "task_send", "task_resolve"]
        or config.data.frontbrain_business_tool_catalog_path
        or config.data.frontbrain_business_tool_augmentation_probability != 0.0
    ):
        raise ValueError(
            "frontbrain_omni_manifest_paths/frontbrain_tool_manifest_paths require an explicit "
            "data.frontbrain_tool_protocol"
        )
    if config.data.idle_gap_augmentation and config.path.paradigm not in {
        "omniflow",
        "frontbrain",
    }:
        raise ValueError(
            "data.idle_gap_augmentation requires an omniflow/frontbrain paradigm"
        )
    if config.data.idle_gap_augmentation and not config.audio_augment.enabled:
        raise ValueError(
            "data.idle_gap_augmentation requires audio_augment.enabled so added user-silent "
            "listen units contain a sampled microphone noise floor"
        )
    if config.data.release_root and not Path(config.data.release_root).is_dir():
        raise FileNotFoundError(f"Release root not found: {config.data.release_root}")
    if config.data.public_video_root and not Path(config.data.public_video_root).is_dir():
        raise FileNotFoundError(
            "Public video root not found: "
            f"{config.data.public_video_root}. Arrange source videos as "
            "<root>/<public-namespace>/<relative-video-path>."
        )
    if (
        not config.data.frontbrain_use_sample_pinned_context
        and config.path.paradigm != "frontbrain"
    ):
        raise ValueError(
            "data.frontbrain_use_sample_pinned_context=false requires paradigm=frontbrain"
        )
    losses = {
        "text_loss_weight": config.train.text_loss_weight,
        "audio_loss_weight": config.train.audio_loss_weight,
        "control_loss_weight": config.train.control_loss_weight,
    }
    if any(not math.isfinite(float(value)) or value < 0 for value in losses.values()):
        raise ValueError(f"Loss weights must be finite and non-negative: {losses}")
    if config.train.text_loss_weight == 0 and config.train.audio_loss_weight == 0:
        raise ValueError("At least one of text_loss_weight or audio_loss_weight must be positive")
    learning_rates = {
        "learning_rate": config.train.learning_rate,
        "projector_learning_rate": config.train.projector_learning_rate,
        "tts_learning_rate": config.train.tts_learning_rate,
    }
    invalid_lrs = {
        name: value
        for name, value in learning_rates.items()
        if value is not None and (not math.isfinite(float(value)) or value <= 0)
    }
    if invalid_lrs:
        raise ValueError(f"Learning rates must be finite and positive: {invalid_lrs}")
    if not math.isfinite(float(config.train.weight_decay)) or config.train.weight_decay < 0:
        raise ValueError("train.weight_decay must be finite and non-negative")
    if not math.isfinite(float(config.train.max_grad_norm)) or config.train.max_grad_norm < 0:
        raise ValueError("train.max_grad_norm must be finite and non-negative")
    if not 0.0 <= config.train.adam_beta1 < 1.0:
        raise ValueError("train.adam_beta1 must be in [0, 1)")
    if not 0.0 <= config.train.adam_beta2 < 1.0:
        raise ValueError("train.adam_beta2 must be in [0, 1)")
    if not math.isfinite(float(config.train.adam_epsilon)) or config.train.adam_epsilon <= 0:
        raise ValueError("train.adam_epsilon must be finite and positive")
    if not 0.0 <= config.train.warmup_ratio <= 1.0:
        raise ValueError("train.warmup_ratio must be in [0, 1]")
    if config.train.max_steps == 0 or config.train.max_steps < -1:
        raise ValueError("train.max_steps must be -1 or a positive integer")
    if config.runtime.skip_final_save and config.train.max_steps < 1:
        raise ValueError("runtime.skip_final_save requires a positive train.max_steps")
    if config.train.max_steps < 0 and config.train.num_train_epochs <= 0:
        raise ValueError("train.num_train_epochs must be positive when max_steps=-1")
    if config.train.save_total_limit is not None and config.train.save_total_limit <= 0:
        raise ValueError("train.save_total_limit must be positive or null")
    if (
        config.runtime.init_checkpoint
        and config.runtime.resume_from_checkpoint
        and config.train.mode != "talker"
    ):
        raise ValueError(
            "init_checkpoint and resume_from_checkpoint may be combined only in Talker mode, "
            "where the frozen Thinker overlay and resumed Talker tensors are disjoint"
        )
    if config.runtime.init_checkpoint and not config.runtime.init_checkpoint_prefixes:
        raise ValueError("runtime.init_checkpoint_prefixes must be explicit for a warm start")
    if config.train.deepspeed:
        deepspeed_path = Path(config.train.deepspeed).expanduser()
        if not deepspeed_path.is_file():
            raise FileNotFoundError(f"DeepSpeed config not found: {deepspeed_path}")
        with deepspeed_path.open("r", encoding="utf-8") as handle:
            deepspeed_config = json.load(handle)
        if not isinstance(deepspeed_config, dict):
            raise ValueError(f"DeepSpeed config must be a JSON object: {deepspeed_path}")
        if "optimizer" in deepspeed_config:
            raise ValueError(
                "DeepSpeed config must not define optimizer: CPMTrainer constructs AdamW "
                "parameter groups for module-specific learning rates and bias/norm decay "
                f"exclusions; remove optimizer from {deepspeed_path}"
            )
        if "scheduler" in deepspeed_config:
            raise ValueError(
                "DeepSpeed config must not define scheduler: Hugging Face builds the scheduler "
                "from train.max_steps/num_train_epochs and warmup_ratio; defining both makes "
                f"resume behavior ambiguous. Remove scheduler from {deepspeed_path}"
            )
        zero_stage = int((deepspeed_config.get("zero_optimization") or {}).get("stage", 0))
        if zero_stage == 3 and config.train.save_trainable_only:
            raise ValueError(
                "train.save_trainable_only is not supported with DeepSpeed ZeRO-3 because "
                "individual parameters are partitioned during canonical checkpoint export"
            )
    augment = config.audio_augment
    if augment.enabled:
        if not augment.noise_index_path:
            raise ValueError(
                "audio_augment.noise_index_path is required when audio augmentation is enabled"
            )
        if not Path(augment.noise_index_path).is_file():
            raise FileNotFoundError(
                f"Audio augmentation noise index not found: {augment.noise_index_path}"
            )
        resolver = ProfileResolver(
            augment.profiles,
            augment.profile_rules,
            default_profile=augment.default_profile,
        )
        for profile_name, profile in resolver.profiles.items():
            probabilities = {
                "background_probability": profile.background_probability,
                "floor_probability": profile.floor_probability,
                "transient_probability": profile.transient_probability,
                "device_probability": profile.device_probability,
                "room_probability": profile.room_probability,
                "speaker_spatial_probability": profile.speaker_spatial_probability,
                "echo_probability": profile.echo_probability,
                "combined_probability": profile.combined_probability,
                "timeline.human_to_human_probability": (
                    profile.timeline.human_to_human_probability
                ),
                "timeline.long_gap_probability": profile.timeline.long_gap_probability,
            }
            invalid = {
                name: value
                for name, value in probabilities.items()
                if not 0.0 <= float(value) <= 1.0
            }
            if invalid:
                raise ValueError(
                    f"audio_augment profile {profile_name!r} probabilities must be in "
                    f"[0, 1]: {invalid}"
                )
            timeline_cap = profile.timeline.max_timeline_units
            if timeline_cap is not None and (
                not isinstance(timeline_cap, int)
                or isinstance(timeline_cap, bool)
                or timeline_cap <= 0
            ):
                raise ValueError(
                    f"audio_augment profile {profile_name!r} timeline.max_timeline_units "
                    "must be a positive integer or null"
                )
            for name, weights in {
                "category_weights": profile.category_weights,
                "floor_category_weights": profile.floor_category_weights,
            }.items():
                if (
                    not weights
                    or any(
                        not math.isfinite(float(value)) or float(value) < 0
                        for value in weights.values()
                    )
                    or sum(float(value) for value in weights.values()) <= 0
                ):
                    raise ValueError(
                        f"audio_augment profile {profile_name!r} {name} must have "
                        "non-negative weights with a positive total"
                    )
            for name, weights in {
                "room_preset_weights": profile.room_preset_weights,
                "speaker_distance_weights": profile.speaker_distance_weights,
            }.items():
                if weights and (
                    any(
                        not math.isfinite(float(value)) or float(value) < 0
                        for value in weights.values()
                    )
                    or sum(float(value) for value in weights.values()) <= 0
                ):
                    raise ValueError(
                        f"audio_augment profile {profile_name!r} {name} must have "
                        "non-negative weights with a positive total"
                    )
            _validate_weighted_bands(
                profile.snr_bands,
                f"audio_augment.profiles.{profile_name}.snr_bands",
            )
            _validate_weighted_bands(
                profile.floor_snr_bands,
                f"audio_augment.profiles.{profile_name}.floor_snr_bands",
            )
            _validate_weighted_bands(
                profile.timeline.short_gap_bands,
                f"audio_augment.profiles.{profile_name}.timeline.short_gap_bands",
                positive_bounds=True,
            )
            _validate_weighted_bands(
                profile.timeline.human_to_human_gap_bands,
                (
                    f"audio_augment.profiles.{profile_name}."
                    "timeline.human_to_human_gap_bands"
                ),
                positive_bounds=True,
            )
            _validate_weighted_bands(
                profile.timeline.long_gap_bands,
                f"audio_augment.profiles.{profile_name}.timeline.long_gap_bands",
                positive_bounds=True,
            )
            invalid_locations = set(profile.timeline.long_gap_locations) - {
                "start",
                "between",
                "tail",
            }
            if invalid_locations:
                raise ValueError(
                    f"audio_augment profile {profile_name!r} has invalid long-gap "
                    f"locations: {sorted(invalid_locations)}"
                )
            if profile.timeline.max_long_spans < 0:
                raise ValueError("timeline.max_long_spans must be non-negative")
        if augment.silence_dbfs_max < augment.silence_dbfs_min:
            raise ValueError("audio_augment.silence_dbfs_max must be >= silence_dbfs_min")
        if not 0.0 < augment.peak_limit <= 1.0:
            raise ValueError("audio_augment.peak_limit must be in (0, 1]")
        if augment.crossfade_ms < 0 or augment.max_noise_segment_seconds <= 0:
            raise ValueError("Audio augmentation segment and crossfade lengths must be positive")
        if (
            augment.transient_min_seconds <= 0
            or augment.transient_max_seconds < augment.transient_min_seconds
        ):
            raise ValueError("Audio augmentation transient duration range is invalid")
        weight_maps = {
            "source_weights": augment.source_weights,
            "transient_category_weights": augment.transient_category_weights,
        }
        for name, weights in weight_maps.items():
            invalid_weights = any(
                not math.isfinite(float(value)) or float(value) < 0
                for value in weights.values()
            )
            if not weights or invalid_weights:
                raise ValueError(f"audio_augment.{name} must contain non-negative weights")
            if sum(float(value) for value in weights.values()) <= 0:
                raise ValueError(f"audio_augment.{name} must have a positive total weight")
        if any(
            not math.isfinite(float(value))
            for value in augment.category_snr_offsets_db.values()
        ):
            raise ValueError("audio_augment.category_snr_offsets_db must be finite")
        _validate_weighted_bands(
            augment.transient_sir_bands,
            "audio_augment.transient_sir_bands",
        )
        for name in (
            "device_gain_db",
            "device_low_cut_hz",
            "device_high_cut_hz",
            "device_compression",
            "room_rt60_seconds",
            "room_predelay_ms",
            "room_wet",
            "room_hf_damping",
            "echo_delay_ms",
            "echo_erl_db",
            "echo_room_mix",
        ):
            values = getattr(augment, name)
            if (
                len(values) != 2
                or not all(math.isfinite(float(value)) for value in values)
                or float(values[1]) < float(values[0])
            ):
                raise ValueError(f"audio_augment.{name} must be finite [min, max]")
        if augment.rir_max_seconds <= 0:
            raise ValueError("audio_augment.rir_max_seconds must be positive")
        if min(augment.room_rt60_seconds) <= 0:
            raise ValueError("audio_augment.room_rt60_seconds must be positive")
        if min(augment.room_predelay_ms) < 0:
            raise ValueError("audio_augment.room_predelay_ms must be non-negative")
        if not 0.0 <= min(augment.room_wet) <= max(augment.room_wet) <= 1.0:
            raise ValueError("audio_augment.room_wet must stay in [0, 1]")
        if not 0.0 <= min(augment.room_hf_damping) <= max(augment.room_hf_damping) < 1.0:
            raise ValueError("audio_augment.room_hf_damping must stay in [0, 1)")
        if min(augment.device_low_cut_hz) <= 0 or min(augment.device_high_cut_hz) <= 0:
            raise ValueError("Audio device cutoff frequencies must be positive")
        if min(augment.device_compression) < 1.0:
            raise ValueError("audio_augment.device_compression must be >= 1")
        if (
            min(augment.echo_delay_ms) < 0
            or min(augment.echo_erl_db) < 0
            or not 0.0 <= min(augment.echo_room_mix) <= max(augment.echo_room_mix) <= 1.0
        ):
            raise ValueError("Audio echo delay and ERL must be non-negative")
        for preset_name, preset in augment.room_presets.items():
            unknown = set(preset) - {
                "weight",
                "rt60_seconds",
                "predelay_ms",
                "wet",
                "hf_damping",
                "early_reflection_count",
            }
            if unknown:
                raise ValueError(
                    f"audio_augment.room_presets.{preset_name} has unknown fields: "
                    f"{sorted(unknown)}"
                )
            for field_name in ("rt60_seconds", "predelay_ms", "wet", "hf_damping"):
                _validate_finite_pair(
                    preset.get(field_name),
                    f"audio_augment.room_presets.{preset_name}.{field_name}",
                )
            early_reflection_count = preset.get("early_reflection_count", [6, 11])
            _validate_finite_pair(
                early_reflection_count,
                f"audio_augment.room_presets.{preset_name}.early_reflection_count",
            )
            if (
                min(preset["rt60_seconds"]) <= 0
                or min(preset["predelay_ms"]) < 0
                or not 0.0 <= min(preset["wet"]) <= max(preset["wet"]) <= 1.0
                or not 0.0
                <= min(preset["hf_damping"])
                <= max(preset["hf_damping"])
                < 1.0
                or not math.isfinite(float(preset.get("weight", 1.0)))
                or float(preset.get("weight", 1.0)) < 0
                or min(early_reflection_count) < 0
                or max(early_reflection_count) > 32
                or any(
                    not float(value).is_integer()
                    for value in early_reflection_count
                )
            ):
                raise ValueError(
                    f"audio_augment.room_presets.{preset_name} has invalid acoustic ranges"
                )
        for preset_name, preset in augment.speaker_distance_presets.items():
            unknown = set(preset) - {
                "weight",
                "distance_m",
                "gain_db",
                "wet_scale",
                "lowpass_hz",
            }
            if unknown:
                raise ValueError(
                    f"audio_augment.speaker_distance_presets.{preset_name} has unknown "
                    f"fields: {sorted(unknown)}"
                )
            for field_name in ("distance_m", "gain_db", "wet_scale", "lowpass_hz"):
                _validate_finite_pair(
                    preset.get(field_name),
                    f"audio_augment.speaker_distance_presets.{preset_name}.{field_name}",
                )
            if (
                min(preset["distance_m"]) <= 0
                or min(preset["wet_scale"]) <= 0
                or min(preset["lowpass_hz"]) <= 0
                or max(preset["lowpass_hz"]) >= config.audio.sample_rate / 2
                or not math.isfinite(float(preset.get("weight", 1.0)))
                or float(preset.get("weight", 1.0)) < 0
            ):
                raise ValueError(
                    f"audio_augment.speaker_distance_presets.{preset_name} has invalid ranges"
                )
        if augment.room_presets and sum(
            float(value.get("weight", 1.0))
            for value in augment.room_presets.values()
        ) <= 0:
            raise ValueError("audio_augment.room_presets must have positive default weight")
        if augment.speaker_distance_presets and sum(
            float(value.get("weight", 1.0))
            for value in augment.speaker_distance_presets.values()
        ) <= 0:
            raise ValueError(
                "audio_augment.speaker_distance_presets must have positive default weight"
            )
        room_names = set(augment.room_presets)
        distance_names = set(augment.speaker_distance_presets)
        for profile_name, profile in resolver.profiles.items():
            unknown_rooms = set(profile.room_preset_weights) - room_names
            unknown_distances = set(profile.speaker_distance_weights) - distance_names
            if unknown_rooms or unknown_distances:
                raise ValueError(
                    f"audio_augment profile {profile_name!r} references unknown room/distance "
                    f"presets: {sorted(unknown_rooms | unknown_distances)}"
                )
            if profile.speaker_spatial_probability > 0 and not distance_names:
                raise ValueError(
                    f"audio_augment profile {profile_name!r} enables speaker spatialization "
                    "without speaker_distance_presets"
                )
        if augment.rir_index_path and not Path(augment.rir_index_path).is_file():
            raise FileNotFoundError(
                f"Audio augmentation RIR index not found: {augment.rir_index_path}"
            )
    if config.train.audio_loss_weight > 0:
        if not config.model.init_tts:
            raise ValueError("audio_loss_weight > 0 requires model.init_tts=true")
        if not (config.freeze.tune_tts_proj or config.freeze.tune_tts_decoder):
            raise ValueError("audio_loss_weight > 0 but every Talker module is frozen")


def validate_loaded_training_contract(config, tokenizer, model) -> None:
    assert_minicpmo_tokenizer(tokenizer)
    if config.train.audio_loss_weight <= 0:
        return

    tts = model.tts
    if tts is None:
        raise RuntimeError("Talker supervision is enabled but the loaded model has no TTS module")
    tts_config = getattr(tts, "config", None)
    if tts_config is None:
        raise RuntimeError("Loaded TTS module has no config")
    expected = {
        "audio_bos_token_id": config.talker.audio_bos_id,
        "num_audio_tokens": config.talker.audio_eos_id + 1,
        "num_vq": 1,
        "condition_type": "hidden_text_merge",
        "normalize_projected_hidden": True,
    }
    actual = {name: getattr(tts_config, name, None) for name in expected}
    mismatched = {
        name: {"expected": value, "actual": actual[name]}
        for name, value in expected.items()
        if actual[name] != value
    }
    if config.talker.audio_bos_id != AUDIO_BOS_ID:
        mismatched["configured_audio_bos_id"] = {
            "expected": AUDIO_BOS_ID,
            "actual": config.talker.audio_bos_id,
        }
    if config.talker.audio_eos_id != S3_EOS_ID:
        mismatched["configured_audio_eos_id"] = {
            "expected": S3_EOS_ID,
            "actual": config.talker.audio_eos_id,
        }
    if S3_NUM_AUDIO_TOKENS != config.talker.audio_eos_id + 1:
        mismatched["configured_num_audio_tokens"] = {
            "expected": S3_NUM_AUDIO_TOKENS,
            "actual": config.talker.audio_eos_id + 1,
        }
    if mismatched:
        raise ValueError(f"Loaded Talker token contract mismatch: {mismatched}")

    context = int(getattr(tts_config, "max_position_embeddings", 0) or 0)
    if context <= 0 or config.talker.max_talker_tokens > context:
        raise ValueError(
            f"talker.max_talker_tokens={config.talker.max_talker_tokens} exceeds loaded "
            f"TTS max_position_embeddings={context}"
        )
    if len(getattr(tts, "emb_code", ())) != 1 or len(getattr(tts, "head_code", ())) != 1:
        raise ValueError("The current Talker loss supports exactly one S3 codebook/head")
    code_embeddings = int(getattr(tts.emb_code[0], "num_embeddings", 0) or 0)
    code_outputs = int(getattr(tts.head_code[0], "out_features", 0) or 0)
    if code_embeddings < S3_NUM_AUDIO_TOKENS or code_outputs != S3_NUM_AUDIO_TOKENS:
        raise ValueError(
            "Loaded Talker S3 embedding/head size mismatch: "
            f"embeddings={code_embeddings}, outputs={code_outputs}, "
            f"expected={S3_NUM_AUDIO_TOKENS}"
        )
    llm = model.llm
    num_layers = int(getattr(getattr(llm, "config", None), "num_hidden_layers", 0) or 0)
    hidden_state_count = num_layers + 1
    layer = config.talker.tts_proj_layer
    if not -hidden_state_count <= layer < hidden_state_count:
        raise ValueError(
            f"talker.tts_proj_layer={layer} is outside the loaded hidden-state range "
            f"[-{hidden_state_count}, {hidden_state_count - 1}]"
        )


def build_hf_training_args(config):
    try:
        from transformers import TrainingArguments
    except ImportError as exc:
        raise RuntimeError(
            "Training requires the dependencies declared by the mcpmft package."
        ) from exc

    train = config.train
    kwargs = {
        "output_dir": train.output_dir,
        "per_device_train_batch_size": train.per_device_train_batch_size,
        "gradient_accumulation_steps": train.gradient_accumulation_steps,
        "num_train_epochs": train.num_train_epochs,
        "max_steps": train.max_steps,
        "learning_rate": train.learning_rate,
        "lr_scheduler_type": train.lr_scheduler_type,
        "weight_decay": train.weight_decay,
        "warmup_ratio": train.warmup_ratio,
        "max_grad_norm": train.max_grad_norm,
        "adam_beta1": train.adam_beta1,
        "adam_beta2": train.adam_beta2,
        "adam_epsilon": train.adam_epsilon,
        "logging_steps": train.logging_steps,
        # Bounded integration runs disable both periodic and final checkpoints.
        "save_strategy": "no" if config.runtime.skip_final_save else "steps",
        "save_steps": train.save_steps,
        "save_total_limit": train.save_total_limit,
        "bf16": train.bf16,
        "gradient_checkpointing": train.gradient_checkpointing,
        "gradient_checkpointing_kwargs": {"use_reentrant": False} if train.gradient_checkpointing else None,
        "remove_unused_columns": train.remove_unused_columns,
        "label_names": train.label_names,
        "dataloader_num_workers": train.dataloader_num_workers,
        "report_to": train.report_to,
        "seed": config.data.seed,
        "data_seed": config.data.seed,
        # Pad variable-length batches per rank; Accelerate shards the iterable dataset.
        "accelerator_config": {"dispatch_batches": False},
    }
    if train.deepspeed:
        kwargs["deepspeed"] = train.deepspeed
    if train.ddp_find_unused_parameters is not None:
        kwargs["ddp_find_unused_parameters"] = train.ddp_find_unused_parameters
    return TrainingArguments(**kwargs)


def build_dataset(config):
    cap = config.data.max_audio_seconds
    # Accelerate shards IterableDataset across ranks; only DataLoader workers shard here.
    manifest_paths = config.data.manifest_paths
    if not manifest_paths:
        raise ValueError("data.manifest_paths cannot be empty")
    rank_zero_info(
        LOGGER,
        "Training data shuffle: enabled=%s seed=%d epoch-aware=true mode=%s block_bytes=%s",
        config.data.shuffle,
        config.data.seed,
        "read-only-jsonl-blocks",
        config.data.shuffle_block_bytes,
    )
    source_counts = config.data.manifest_row_counts or ["unknown"] * len(manifest_paths)
    rank_zero_info(
        LOGGER,
        "Training manifest mix: strategy=%s sources=%s total_rows=%s",
        config.data.manifest_mix_strategy,
        [f"{Path(path).name}:{count}" for path, count in zip(manifest_paths, source_counts)],
        sum(source_counts) if all(isinstance(value, int) for value in source_counts) else "unknown",
    )
    return ManifestDataset(
        manifest_paths,
        shard_by_rank=False,
        shard_by_worker=True,
        max_audio_seconds=cap,
        mix_strategy=config.data.manifest_mix_strategy,
        mix_weights=config.data.manifest_mix_weights,
        manifest_row_counts=config.data.manifest_row_counts,
        manifest_sample_counts=config.data.manifest_sample_counts,
        shuffle=config.data.shuffle,
        seed=config.data.seed,
        shuffle_block_bytes=config.data.shuffle_block_bytes,
        media_resolver=ReleaseMediaResolver(
            release_root=config.data.release_root,
            public_video_root=config.data.public_video_root,
        ),
    )


def main(argv: list[str] | None = None) -> None:
    setup_logging()
    config = parse_project_config(argv)
    validate_project_config(config)
    if config.train.audio_loss_weight > 0 and config.data.s3_cache_dir:
        require_complete_s3_cache(
            config.data.s3_cache_dir,
            manifest_paths=config.data.manifest_paths,
            release_root=config.data.release_root,
        )
    raw_business_tool_catalog, forbidden_business_tools_by_sample = (
        load_business_tool_augmentation_catalog(
            config.data.frontbrain_business_tool_catalog_path
        )
        if config.data.frontbrain_business_tool_catalog_path
        else ((), {})
    )
    dataset = build_dataset(config)
    validate_training_schedule(
        dataset,
        max_steps=config.train.max_steps,
        dry_run=config.runtime.dry_run,
    )
    if is_rank_zero() and not config.runtime.dry_run:
        validate_run_directory(config)
        Path(config.train.output_dir).mkdir(parents=True, exist_ok=True)
        write_resolved_config(config, Path(config.train.output_dir) / "resolved_config.yaml")

    tokenizer, processor = load_tokenizer_and_processor(config.model)
    business_tool_catalog, rejected_business_tools = (
        filter_business_tool_catalog_for_realtime(
            raw_business_tool_catalog,
            tokenizer,
            max_tools=config.data.frontbrain_max_tools_per_sample,
            max_schema_tokens=config.data.frontbrain_max_tool_schema_tokens,
        )
        if raw_business_tool_catalog
        else ((), ())
    )
    if (
        config.data.frontbrain_business_tool_augmentation_probability > 0
        and not business_tool_catalog
    ):
        raise ValueError(
            "business-tool augmentation has no catalog schemas within the realtime token budget"
        )
    if config.data.frontbrain_tool_protocol == "task_tools_v1":
        rank_zero_info(
            LOGGER,
            "Training tool context: task_start/task_send/task_resolve always visible; "
            "business_catalog=%s/%s schemas probability=%.3f bounds=%s..%s "
            "max_tools=%s max_schema_tokens=%s",
            len(business_tool_catalog),
            len(raw_business_tool_catalog),
            config.data.frontbrain_business_tool_augmentation_probability,
            config.data.frontbrain_business_tool_augmentation_min,
            config.data.frontbrain_business_tool_augmentation_max,
            config.data.frontbrain_max_tools_per_sample,
            config.data.frontbrain_max_tool_schema_tokens,
        )
        if rejected_business_tools:
            rank_zero_info(
                LOGGER,
                "Excluded %s oversized business schema variants from random augmentation",
                len(rejected_business_tools),
            )
    # Token2wav is not part of either training loss.
    model = load_minicpmo_model(config.model, init_token2wav=False)
    if config.path.paradigm in {"omniflow", "frontbrain"}:
        # Resize the base embedding before loading a Thinker checkpoint.
        from mcpmft.frontbrain.setup import ensure_native_frontbrain_tokens

        ids = ensure_native_frontbrain_tokens(model, tokenizer)
        rank_zero_info(LOGGER, "Added realtime interaction tokens: %s", ids)
    validate_loaded_training_contract(config, tokenizer, model)
    if config.runtime.init_checkpoint:
        loaded = load_selected_state_dict(
            model,
            config.runtime.init_checkpoint,
            prefixes=config.runtime.init_checkpoint_prefixes,
        )
        rank_zero_info(
            LOGGER,
            "Warm-started %s tensors from %s with prefixes=%s",
            loaded,
            config.runtime.init_checkpoint,
            config.runtime.init_checkpoint_prefixes,
        )
    touched = apply_freeze(model, config.freeze)
    stats = param_stats(model)
    rank_zero_info(
        LOGGER,
        "Applied freeze groups=%s trainable=%s/%s (%.2f%%)",
        touched,
        stats.trainable,
        stats.total,
        stats.trainable_ratio * 100,
    )

    wrapper = OmniTrainWrapper(
        model,
        tokenizer=tokenizer,
        talker_args=config.talker,
        text_loss_weight=config.train.text_loss_weight,
        audio_loss_weight=config.train.audio_loss_weight,
        control_loss_weight=config.train.control_loss_weight,
        tune_llm=config.freeze.tune_llm,
        audio_chunk_length=config.audio.audio_chunk_length,
        freeze_args=config.freeze,
    )
    if config.train.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    s3_cache = S3TokenCache(config.data.s3_cache_dir) if config.data.s3_cache_dir else None
    audio_augmenter = (
        AudioAugmenter(config.audio_augment, sample_rate=config.audio.sample_rate)
        if config.audio_augment.enabled
        else None
    )
    collator = OmniCollator(
        tokenizer=tokenizer,
        paradigm=config.path.paradigm,
        max_seq_length=config.data.max_seq_length,
        strict_no_truncation=config.data.strict_no_truncation,
        s3_cache=s3_cache,
        audio_processor=processor,
        audio_augmenter=audio_augmenter,
        kv_delete_mask=config.data.kv_delete_mask,
        kv_keep_previous_units=config.data.kv_keep_previous_units,
        sliding_window_training=config.data.sliding_window_training,
        context_max_units=config.data.context_max_units,
        context_previous_max_tokens=config.data.context_previous_max_tokens,
        frontbrain_tool_protocol=config.data.frontbrain_tool_protocol,
        frontbrain_use_sample_pinned_context=(
            config.data.frontbrain_use_sample_pinned_context
        ),
        frontbrain_pinned_context_requires_marker=(
            config.data.frontbrain_pinned_context_requires_marker
        ),
        frontbrain_business_tool_catalog=business_tool_catalog,
        frontbrain_business_tool_forbidden_by_sample=(
            forbidden_business_tools_by_sample
        ),
        frontbrain_business_tool_augmentation_probability=(
            config.data.frontbrain_business_tool_augmentation_probability
        ),
        frontbrain_business_tool_augmentation_min=(
            config.data.frontbrain_business_tool_augmentation_min
        ),
        frontbrain_business_tool_augmentation_max=(
            config.data.frontbrain_business_tool_augmentation_max
        ),
        frontbrain_max_tools_per_sample=(
            config.data.frontbrain_max_tools_per_sample
        ),
        frontbrain_max_tool_schema_tokens=(
            config.data.frontbrain_max_tool_schema_tokens
        ),
        turn_gap_ms=config.talker.turn_gap_ms,
        duplex_text_tokens_per_unit=config.data.duplex_text_tokens_per_unit,
        codes_per_text_token=config.talker.codes_per_text_token,
        duplex_speech_tokens_per_unit=config.data.duplex_speech_tokens_per_unit,
        idle_gap_augmentation=config.data.idle_gap_augmentation,
        include_duplex_system_prompt=config.data.include_duplex_system_prompt,
        duplex_system_prompt=config.data.duplex_system_prompt,
        seed=config.data.seed,
        audio_geometry=AudioGeometry(
            sample_rate=config.audio.sample_rate,
            hop_length=config.audio.whisper_hop_length,
            pool_step=config.audio.audio_pool_step,
        ),
    )
    if config.runtime.dry_run:
        rank_zero_info(LOGGER, "Dry run completed before Trainer construction")
        return
    from mcpmft.train.trainer import CPMTrainer

    training_args = build_hf_training_args(config)
    trainer = CPMTrainer(
        model=wrapper,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        processing_class=tokenizer,
        projector_learning_rate=config.train.projector_learning_rate,
        tts_learning_rate=config.train.tts_learning_rate,
        project_config=config.to_dict(),
    )
    trainer.save_trainable_only = config.train.save_trainable_only
    trainer.train(resume_from_checkpoint=config.runtime.resume_from_checkpoint)
    if config.runtime.skip_final_save:
        rank_zero_info(LOGGER, "Skipping final checkpoint for bounded integration run")
    else:
        trainer.save_final_checkpoint()
        trainer.save_model()


if __name__ == "__main__":
    main()
