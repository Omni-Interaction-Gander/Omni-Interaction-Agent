from __future__ import annotations

import argparse
import copy
import dataclasses
import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Iterable, Literal

import yaml

from mcpmft.prompts import GANDER_DUPLEX_SYSTEM_PROMPT
from mcpmft.utils.io import atomic_write_text


def _deep_update(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_yaml_files(paths: Iterable[str | Path]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for value in paths:
        _, document = _load_yaml_document(value)
        _deep_update(merged, document)
    return merged


def _load_yaml_document(value: str | Path) -> tuple[Path, dict[str, Any]]:
    path = Path(value).expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle) or {}
    if not isinstance(document, dict):
        raise ValueError(f"Top-level YAML must be a mapping: {path}")

    data = document.get("data")
    if isinstance(data, dict) and data.get("release_root"):
        release_root = Path(str(data["release_root"])).expanduser()
        if not release_root.is_absolute():
            release_root = path.parent / release_root
        data["release_root"] = str(release_root.resolve())
    return path, document


def _resolve_release_paths(config: dict[str, Any]) -> dict[str, Any]:
    data = _section(config, "data")
    value = data.get("release_root")
    if not value:
        return config
    root = Path(str(value)).expanduser().resolve()
    data["release_root"] = str(root)

    def resolve(path_value: str | None) -> str | None:
        if not path_value:
            return path_value
        path = Path(path_value).expanduser()
        return str(path if path.is_absolute() else (root / path).resolve())

    for name in (
        "manifest_paths",
        "frontbrain_omni_manifest_paths",
        "frontbrain_tool_manifest_paths",
    ):
        data[name] = [resolve(path) for path in data.get(name, [])]
    for name in (
        "frontbrain_business_tool_catalog_path",
        "public_video_root",
        "s3_cache_dir",
    ):
        if name in data:
            data[name] = resolve(data.get(name))

    augment = _section(config, "audio_augment")
    for name in ("noise_index_path", "archive_cache_dir", "rir_index_path"):
        if name in augment:
            augment[name] = resolve(augment.get(name))
    return config


def _parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _set_dotted(config: dict[str, Any], key: str, value: Any) -> None:
    cursor = config
    parts = key.split(".")
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = value


def apply_overrides(config: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply ``--section.key value`` overrides to a loaded project document."""

    index = 0
    while index < len(overrides):
        option = overrides[index]
        if not option.startswith("--"):
            raise ValueError(f"Unexpected argument: {option}")
        if index + 1 == len(overrides) or overrides[index + 1].startswith("--"):
            value = True
            index += 1
        else:
            value = _parse_scalar(overrides[index + 1])
            index += 2
        _set_dotted(config, option[2:].replace("-", "_"), value)
    return config


def load_project_document(
    paths: Iterable[str | Path],
    overrides: list[str] | None = None,
) -> dict[str, Any]:
    """Load the base recipe, expand its selected mode, then apply release overlays."""

    overrides = list(overrides or [])
    loaded = [_load_yaml_document(value) for value in paths]
    if not loaded:
        raise ValueError("At least one training config is required")
    documents = [document for _, document in loaded]
    raw: dict[str, Any] = {}
    for document in documents:
        _deep_update(raw, copy.deepcopy(document))
    modes = raw.get("train_modes")
    if modes is None:
        return _resolve_release_paths(apply_overrides(raw, overrides))
    if not isinstance(modes, dict) or not modes:
        raise ValueError("train_modes must be a non-empty mapping")

    selection = apply_overrides(copy.deepcopy(raw), overrides)
    train = _section(selection, "train")
    mode = train.get("mode")
    if not isinstance(mode, str) or not mode:
        raise ValueError("train.mode must select one entry from train_modes")
    profile = modes.get(mode)
    if not isinstance(profile, dict):
        raise ValueError(
            f"Unknown train.mode {mode!r}; available modes: {sorted(modes)}"
        )

    document = copy.deepcopy(documents[0])
    document.pop("train_modes")
    _deep_update(document, copy.deepcopy(profile))
    for overlay in documents[1:]:
        overlay = copy.deepcopy(overlay)
        overlay.pop("train_modes", None)
        _deep_update(document, overlay)
    apply_overrides(document, overrides)
    _section(document, "train")["mode"] = mode
    return _resolve_release_paths(document)


def parse_config(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", action="append", required=True)
    known, overrides = parser.parse_known_args(argv)
    return load_project_document(known.config, overrides)


def _section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"Config section {name!r} must be a mapping")
    return value


def _make(cls: type[Any], config: dict[str, Any], name: str) -> Any:
    values = _section(config, name)
    unknown = sorted(set(values) - {item.name for item in fields(cls)})
    if unknown:
        raise ValueError(
            "Unknown config key(s): " + ", ".join(f"{name}.{key}" for key in unknown)
        )
    return cls(**values)


@dataclass
class ModelArguments:
    model_name_or_path: str = "external/models/minicpm-o-4_5"
    processor_name_or_path: str | None = None
    trust_remote_code: bool = True
    torch_dtype: str = "bfloat16"
    attn_implementation: str = "sdpa"
    init_vision: bool = True
    init_audio: bool = True
    init_tts: bool = True
    token2wav_dir: str | None = None
    token2wav_n_timesteps: int = 10
    token2wav_enable_float16: bool = False
    low_cpu_mem_usage: bool = True
    device_map: str | None = None
    train_disable_stream_input: bool = True


@dataclass
class AudioArguments:
    sample_rate: int = 16000
    audio_chunk_length: int = 1
    whisper_hop_length: int = 160
    audio_pool_step: int = 5


@dataclass
class AudioAugmentArguments:
    enabled: bool = False
    noise_index_path: str | None = None
    archive_cache_dir: str = "data/noise/archive_cache"
    source_weights: dict[str, float] = field(default_factory=dict)
    default_profile: str = "clean"
    profiles: dict[str, dict[str, Any]] = field(default_factory=dict)
    profile_rules: list[dict[str, Any]] = field(default_factory=list)
    category_snr_offsets_db: dict[str, float] = field(default_factory=dict)
    transient_category_weights: dict[str, float] = field(default_factory=dict)
    transient_sir_bands: list[list[float]] = field(default_factory=lambda: [[8.0, 20.0, 1.0]])
    transient_min_seconds: float = 0.5
    transient_max_seconds: float = 6.0
    rir_index_path: str | None = None
    rir_max_seconds: float = 1.5
    room_rt60_seconds: list[float] = field(default_factory=lambda: [0.25, 1.10])
    room_predelay_ms: list[float] = field(default_factory=lambda: [6.0, 35.0])
    room_wet: list[float] = field(default_factory=lambda: [0.22, 0.48])
    room_hf_damping: list[float] = field(default_factory=lambda: [0.25, 0.70])
    room_presets: dict[str, dict[str, Any]] = field(default_factory=dict)
    speaker_distance_presets: dict[str, dict[str, Any]] = field(default_factory=dict)
    device_gain_db: list[float] = field(default_factory=lambda: [-4.0, 2.0])
    device_low_cut_hz: list[float] = field(default_factory=lambda: [50.0, 180.0])
    device_high_cut_hz: list[float] = field(default_factory=lambda: [4500.0, 7600.0])
    device_compression: list[float] = field(default_factory=lambda: [1.0, 1.8])
    echo_delay_ms: list[float] = field(default_factory=lambda: [35.0, 220.0])
    echo_erl_db: list[float] = field(default_factory=lambda: [6.0, 18.0])
    echo_room_mix: list[float] = field(default_factory=lambda: [0.20, 0.50])
    silence_dbfs_min: float = -34.0
    silence_dbfs_max: float = -22.0
    crossfade_ms: int = 400
    max_noise_segment_seconds: float = 60.0
    peak_limit: float = 0.98


@dataclass
class TalkerArguments:
    tts_proj_layer: int = -1
    codes_per_text_token: int = 6
    audio_eos_id: int = 6561
    audio_bos_id: int = 151687
    detach_llm_for_tts: Literal["auto", "always", "never"] = "auto"
    turn_gap_ms: int = 2000
    max_talker_tokens: int = 4096


@dataclass
class FreezeArguments:
    tune_vision: bool = True
    tune_resampler: bool = True
    tune_audio_encoder: bool = True
    tune_audio_proj: bool = True
    tune_llm: bool = True
    tune_tts_proj: bool = True
    tune_tts_decoder: bool = True


@dataclass
class PathArguments:
    paradigm: Literal["turn", "omniflow", "frontbrain"] = "frontbrain"
    enable_s2t: bool = True
    enable_t2s: bool = False
    enable_vision: bool = True
    enable_omniflow: bool = True


@dataclass
class DataArguments:
    release_root: str | None = None
    manifest_paths: list[str] = field(default_factory=list)
    manifest_row_counts: list[int] = field(default_factory=list)
    manifest_sample_counts: list[int] = field(default_factory=list)
    manifest_mix_strategy: Literal["concat", "smooth"] = "smooth"
    manifest_mix_weights: list[float] = field(default_factory=list)
    frontbrain_tool_protocol: Literal["none", "task_tools_v1"] = "task_tools_v1"
    frontbrain_omni_manifest_paths: list[str] = field(default_factory=list)
    frontbrain_tool_manifest_paths: list[str] = field(default_factory=list)
    frontbrain_require_vision_source: bool = True
    frontbrain_required_task_calls: list[str] = field(
        default_factory=lambda: ["task_start", "task_send", "task_resolve"]
    )
    frontbrain_business_tool_catalog_path: str | None = None
    frontbrain_business_tool_augmentation_probability: float = 0.0
    frontbrain_business_tool_augmentation_min: int = 1
    frontbrain_business_tool_augmentation_max: int = 3
    frontbrain_max_tools_per_sample: int = 6
    frontbrain_max_tool_schema_tokens: int = 1024
    frontbrain_use_sample_pinned_context: bool = False
    frontbrain_pinned_context_requires_marker: bool = False
    shuffle: bool = True
    shuffle_block_bytes: int = 262144
    public_video_root: str | None = None
    s3_cache_dir: str | None = None
    max_seq_length: int = 16384
    strict_no_truncation: bool = True
    max_audio_seconds: float | None = None
    kv_delete_mask: bool = True
    kv_keep_previous_units: int = 128
    sliding_window_training: Literal[
        "off", "window_no_previous", "sampled_context", "context_memory"
    ] = "window_no_previous"
    context_training_semantics: Literal["all_unit_4d_mask_v1"] = "all_unit_4d_mask_v1"
    context_max_units: int = 128
    context_previous_max_tokens: int = 1500
    duplex_text_tokens_per_unit: int = 4
    duplex_speech_tokens_per_unit: int = 25
    idle_gap_augmentation: bool = False
    include_duplex_system_prompt: bool = True
    duplex_system_prompt: str = GANDER_DUPLEX_SYSTEM_PROMPT
    seed: int = 42


@dataclass
class TrainArguments:
    mode: Literal["thinker", "talker", "joint", "custom"] = "custom"
    output_dir: str = "outputs/gander"
    deepspeed: str | None = None
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    num_train_epochs: float = 1.0
    max_steps: int = -1
    learning_rate: float = 2.0e-6
    projector_learning_rate: float | None = 1.0e-5
    tts_learning_rate: float | None = None
    lr_scheduler_type: str = "linear"
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1.0e-8
    logging_steps: int = 10
    save_steps: int = 2000
    save_total_limit: int | None = 2
    bf16: bool = True
    gradient_checkpointing: bool = True
    remove_unused_columns: bool = False
    label_names: list[str] = field(default_factory=lambda: ["labels"])
    text_loss_weight: float = 1.0
    audio_loss_weight: float = 0.0
    control_loss_weight: float = 1.5
    save_trainable_only: bool = True
    dataloader_num_workers: int = 0
    ddp_find_unused_parameters: bool | None = False
    report_to: str | list[str] | None = None


@dataclass
class RuntimeArguments:
    local_rank: int = -1
    init_checkpoint: str | None = None
    init_checkpoint_prefixes: list[str] = field(default_factory=list)
    resume_from_checkpoint: str | None = None
    dry_run: bool = False
    skip_final_save: bool = False


@dataclass
class LaunchArguments:
    hosts: list[str] = field(default_factory=list)
    hostfile: str | None = None
    ssh_user: str = "root"
    gpus_per_node: int = 8
    master_addr: str = ""
    master_port: int = 29500
    python: str = "python"
    project_dir: str = ""
    log_dir: str = "outputs/launcher"
    local_cache_dir: str = "/tmp/gander"
    nccl_socket_ifname: str | None = None
    nccl_ib_disable: bool = False
    omp_num_threads: int = 8
    ssh_options: list[str] = field(
        default_factory=lambda: ["-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no"]
    )


@dataclass
class ProjectConfig:
    model: ModelArguments = field(default_factory=ModelArguments)
    audio: AudioArguments = field(default_factory=AudioArguments)
    audio_augment: AudioAugmentArguments = field(default_factory=AudioAugmentArguments)
    talker: TalkerArguments = field(default_factory=TalkerArguments)
    freeze: FreezeArguments = field(default_factory=FreezeArguments)
    path: PathArguments = field(default_factory=PathArguments)
    data: DataArguments = field(default_factory=DataArguments)
    train: TrainArguments = field(default_factory=TrainArguments)
    runtime: RuntimeArguments = field(default_factory=RuntimeArguments)
    launch: LaunchArguments = field(default_factory=LaunchArguments)

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "ProjectConfig":
        types = {
            "model": ModelArguments,
            "audio": AudioArguments,
            "audio_augment": AudioAugmentArguments,
            "talker": TalkerArguments,
            "freeze": FreezeArguments,
            "path": PathArguments,
            "data": DataArguments,
            "train": TrainArguments,
            "runtime": RuntimeArguments,
            "launch": LaunchArguments,
        }
        unknown = sorted(set(config) - set(types))
        if unknown:
            raise ValueError("Unknown top-level config section(s): " + ", ".join(unknown))
        values = {name: _make(section_type, config, name) for name, section_type in types.items()}
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return {
            field_.name: dataclasses.asdict(getattr(self, field_.name))
            for field_ in fields(self)
        }


def parse_project_config(argv: list[str] | None = None) -> ProjectConfig:
    return ProjectConfig.from_dict(parse_config(argv))


def write_resolved_config(config: ProjectConfig, path: str | Path) -> None:
    atomic_write_text(
        path,
        yaml.safe_dump(config.to_dict(), sort_keys=False, allow_unicode=True),
    )
