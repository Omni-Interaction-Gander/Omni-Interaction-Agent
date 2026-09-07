from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import shutil
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Literal

import yaml

from mcpmft.args import ModelArguments
from mcpmft.prompts import GANDER_DUPLEX_SYSTEM_PROMPT
from mcpmft.tool_protocol import ensure_lean_task_tools, normalize_tool_schema

from .asr_process import AsrConfig, AsrService, asr_base_url, validate_asr_config
from .media_mode import VIDEO_SOURCES


@dataclass(frozen=True)
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 7975
    ws_ping_interval: float = 20.0
    ws_ping_timeout: float = 120.0
    runtime_dir: str = "var/gander"
    ledger_dir: str | None = None
    mode: Literal["lean", "coordinator"] = "lean"
    log_level: str = "info"
    context_max_events: int = 512
    cuda_visible_devices: str | None = None


@dataclass(frozen=True)
class DuplexConfig:
    checkpoint: str = ""
    talker_checkpoint: str | None = None
    detached_talker_device: str | None = None
    talker_emit_speech_tokens: int = 25
    generate_audio: bool = False
    decode_mode: Literal["sampling", "greedy"] = "sampling"
    system_prompt: str = GANDER_DUPLEX_SYSTEM_PROMPT
    ref_audio_path: str | None = None
    tools_path: str | None = None
    trailing_silence_sec: float = 8.0
    turn_bind_grace_sec: float = 5.0
    media_mode: Literal["voice", "omni", "auto"] = "voice"
    allow_client_video: bool = False
    client_video_mode: Literal["omni", "auto"] = "omni"
    client_video_sources: tuple[str, ...] = VIDEO_SOURCES
    vision_max_slice_nums: int = 1
    vision_batch_feed: bool = False
    max_screen_frame_bytes: int = 4 * 1024 * 1024
    max_screen_pixels: int = 4096 * 4096
    codex_frame_rate_multiplier: float = 3.0
    codex_screen_history_seconds: float = 8.0
    sliding_window_mode: Literal[
        "context_memory", "context_no_previous", "context_slate"
    ] = "context_no_previous"
    expose_task_slate_to_model: bool = False
    speak_text_tokens_per_unit: int = 4
    talker_speech_tokens_per_unit: int = 25
    talker_final_speech_tokens_max: int = 0
    max_new_speak_tokens_per_chunk: int | None = None
    max_new_tool_tokens: int = 256
    max_tool_response_tokens: int = 256
    context_max_units: int = 128
    context_previous_max_tokens: int = 0
    memory_slate_max_tokens: int = 256
    memory_lead_ratio: float = 0.7
    memory_soft_ratio: float = 0.9
    memory_hard_ratio: float = 1.0
    memory_kv_ceiling_units: int | None = None
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    warm_first_unit: bool = True


@dataclass(frozen=True)
class WorkerConfig:
    provider: str = "codex"
    cwd: str = "."
    profile: Literal["task_scoped", "full"] = "task_scoped"
    settings: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CoordinatorConfig:
    cwd: str | None = None
    codex_bin: str = "codex"
    model: str | None = None
    reasoning_effort: str = "low"
    codex_home: str | None = None
    timeout_sec: float = 30.0
    max_turns: int = 8


@dataclass(frozen=True)
class MemoryConfig:
    url: str | None = None
    token_env: str = "GANDER_MEMORY_TOKEN"
    timeout_sec: float = 10.0


@dataclass(frozen=True)
class ReleaseConfig:
    model: ModelArguments
    server: ServerConfig
    asr: AsrConfig
    duplex: DuplexConfig
    worker: WorkerConfig
    coordinator: CoordinatorConfig
    memory: MemoryConfig


def _section(cls: type[Any], document: dict[str, Any], name: str) -> Any:
    values = document.get(name) or {}
    if not isinstance(values, dict):
        raise ValueError(f"{name} must be a YAML mapping")
    known = {item.name for item in fields(cls)}
    unknown = sorted(set(values) - known)
    if unknown:
        raise ValueError(
            "Unknown setting(s): " + ", ".join(f"{name}.{key}" for key in unknown)
        )
    return cls(**values)


def load_config(path: str | Path) -> ReleaseConfig:
    path = Path(path).expanduser()
    with path.open("r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle) or {}
    if not isinstance(document, dict):
        raise ValueError("Top-level YAML must be a mapping")
    sections = {
        "model",
        "server",
        "asr",
        "duplex",
        "worker",
        "coordinator",
        "memory",
    }
    unknown = sorted(set(document) - sections)
    if unknown:
        raise ValueError("Unknown top-level section(s): " + ", ".join(unknown))
    config = ReleaseConfig(
        model=_section(ModelArguments, document, "model"),
        server=_section(ServerConfig, document, "server"),
        asr=_section(AsrConfig, document, "asr"),
        duplex=_section(DuplexConfig, document, "duplex"),
        worker=_section(WorkerConfig, document, "worker"),
        coordinator=_section(CoordinatorConfig, document, "coordinator"),
        memory=_section(MemoryConfig, document, "memory"),
    )
    validate_release_config(config)
    return config


def validate_release_config(config: ReleaseConfig) -> None:
    server = config.server
    duplex = config.duplex
    if server.mode not in {"lean", "coordinator"}:
        raise ValueError("server.mode must be 'lean' or 'coordinator'")
    if not 1 <= server.port <= 65535:
        raise ValueError("server.port must be between 1 and 65535")
    if server.ws_ping_interval <= 0 or server.ws_ping_timeout <= 0:
        raise ValueError("server WebSocket ping values must be positive")
    if server.context_max_events < 1:
        raise ValueError("server.context_max_events must be positive")
    if config.worker.profile not in {"task_scoped", "full"}:
        raise ValueError("worker.profile must be 'task_scoped' or 'full'")
    if not config.worker.provider:
        raise ValueError("worker.provider must not be empty")
    if not isinstance(config.worker.settings, dict):
        raise ValueError("worker.settings must be a YAML mapping")
    validate_asr_config(config.asr)
    if config.asr.mode == "managed" and config.asr.port == server.port:
        raise ValueError("managed ASR and the Gander server must use different ports")
    if not duplex.checkpoint:
        raise ValueError("duplex.checkpoint is required")
    if duplex.generate_audio and not config.model.init_tts:
        raise ValueError("duplex.generate_audio requires model.init_tts=true")
    if duplex.generate_audio and not duplex.ref_audio_path:
        raise ValueError("duplex.generate_audio requires duplex.ref_audio_path")
    if duplex.detached_talker_device and not duplex.generate_audio:
        raise ValueError("detached Talker requires duplex.generate_audio=true")
    if duplex.detached_talker_device and not duplex.talker_checkpoint:
        raise ValueError("detached Talker requires duplex.talker_checkpoint")
    if duplex.detached_talker_device and not config.model.token2wav_dir:
        raise ValueError("detached Talker requires model.token2wav_dir")
    if duplex.allow_client_video and not config.model.init_vision:
        raise ValueError("client video requires model.init_vision=true")
    if duplex.media_mode != "voice" and not config.model.init_vision:
        raise ValueError("omni/auto media mode requires model.init_vision=true")
    if (
        duplex.sliding_window_mode == "context_no_previous"
        and duplex.context_previous_max_tokens != 0
    ):
        raise ValueError(
            "context_no_previous requires duplex.context_previous_max_tokens=0"
        )
    if (
        duplex.sliding_window_mode in {"context_memory", "context_slate"}
        and duplex.context_previous_max_tokens <= 0
    ):
        raise ValueError(
            "context_memory/context_slate require "
            "duplex.context_previous_max_tokens > 0"
        )
    if (
        duplex.sliding_window_mode == "context_slate"
        and not duplex.expose_task_slate_to_model
    ):
        raise ValueError(
            "context_slate requires duplex.expose_task_slate_to_model=true"
        )
    if (
        duplex.sliding_window_mode == "context_no_previous"
        and duplex.expose_task_slate_to_model
    ):
        raise ValueError(
            "context_no_previous requires duplex.expose_task_slate_to_model=false"
        )
    if duplex.context_max_units < 1:
        raise ValueError("duplex.context_max_units must be positive")
    if config.coordinator.timeout_sec <= 0 or config.coordinator.max_turns < 1:
        raise ValueError("coordinator timeout and max_turns must be positive")
    if config.memory.timeout_sec <= 0:
        raise ValueError("memory.timeout_sec must be positive")


def preflight_config(config: ReleaseConfig) -> None:
    from .providers import builtin_provider_registry

    configured_provider = builtin_provider_registry().configure(
        config.worker.provider,
        config.worker.settings,
    )
    _require_directory("model.model_name_or_path", config.model.model_name_or_path)
    if config.model.processor_name_or_path:
        _require_directory(
            "model.processor_name_or_path",
            config.model.processor_name_or_path,
        )
    _require_path("duplex.checkpoint", config.duplex.checkpoint)
    if config.duplex.talker_checkpoint:
        _require_path("duplex.talker_checkpoint", config.duplex.talker_checkpoint)
    if config.duplex.generate_audio:
        assert config.duplex.ref_audio_path is not None
        _require_file("duplex.ref_audio_path", config.duplex.ref_audio_path)
    if config.model.token2wav_dir:
        _require_directory("model.token2wav_dir", config.model.token2wav_dir)
    if config.duplex.tools_path:
        _require_file("duplex.tools_path", config.duplex.tools_path)
        _tool_schemas(config.duplex.tools_path)
    _require_directory("worker.cwd", config.worker.cwd)
    if config.server.mode == "coordinator" and config.coordinator.cwd:
        _require_directory("coordinator.cwd", config.coordinator.cwd)

    if config.worker.provider == "codex":
        _require_executable(
            "worker.settings.codex_bin",
            str(getattr(configured_provider.settings, "codex_bin")),
        )
    if config.server.mode == "coordinator":
        _require_executable("coordinator.codex_bin", config.coordinator.codex_bin)

    if config.asr.mode == "managed":
        assert config.asr.model_path is not None
        _require_directory("asr.model_path", config.asr.model_path)
        if importlib.util.find_spec("faster_whisper") is None:
            raise RuntimeError(
                "managed ASR requires faster-whisper; install minicpm_ft[asr]"
            )
    _validate_gpu_assignment(config)


def _require_path(label: str, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def _require_directory(label: str, value: str) -> Path:
    path = _require_path(label, value)
    if not path.is_dir():
        raise ValueError(f"{label} must be a directory: {path}")
    return path


def _require_file(label: str, value: str) -> Path:
    path = _require_path(label, value)
    if not path.is_file():
        raise ValueError(f"{label} must be a file: {path}")
    return path


def _require_executable(label: str, command: str) -> None:
    expanded = str(Path(command).expanduser()) if "/" in command else command
    resolved = shutil.which(expanded)
    if resolved is None:
        raise FileNotFoundError(
            f"{label} executable not found: {command}; install it or configure "
            "an absolute executable path"
        )


def _cuda_devices(value: str | None) -> tuple[str, ...]:
    if value is None:
        return ()
    devices = tuple(item.strip() for item in value.split(",") if item.strip())
    if not devices:
        raise ValueError("CUDA_VISIBLE_DEVICES must contain at least one device")
    if len(set(devices)) != len(devices):
        raise ValueError("CUDA_VISIBLE_DEVICES must not contain duplicates")
    return devices


def _validate_gpu_assignment(config: ReleaseConfig) -> None:
    server_devices = _cuda_devices(config.server.cuda_visible_devices)
    talker_device = config.duplex.detached_talker_device
    if talker_device:
        if not talker_device.startswith("cuda:"):
            raise ValueError("duplex.detached_talker_device must be cuda:<index>")
        try:
            talker_index = int(talker_device.removeprefix("cuda:"))
        except ValueError as exc:
            raise ValueError(
                "duplex.detached_talker_device must be cuda:<index>"
            ) from exc
        if talker_index < 0:
            raise ValueError("detached Talker CUDA index must not be negative")
        if server_devices and talker_index >= len(server_devices):
            raise ValueError(
                "detached Talker CUDA index is outside server.cuda_visible_devices"
            )

    if config.asr.mode != "managed" or config.asr.device != "cuda":
        return
    asr_devices = _cuda_devices(config.asr.cuda_visible_devices)
    if not server_devices or not asr_devices:
        raise ValueError(
            "managed CUDA ASR requires explicit, separate server.cuda_visible_devices "
            "and asr.cuda_visible_devices"
        )
    overlap = sorted(set(server_devices) & set(asr_devices))
    if overlap:
        raise ValueError(
            "ASR and Gander CUDA assignments overlap: " + ", ".join(overlap)
        )
    if config.asr.device_index >= len(asr_devices):
        raise ValueError("asr.device_index is outside asr.cuda_visible_devices")


def _tool_schemas(path: str | None) -> tuple[dict[str, Any], ...]:
    schemas: list[dict[str, Any]] = []
    if path:
        value = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
        value = value.get("tools") if isinstance(value, dict) else value
        if not isinstance(value, list):
            raise ValueError("duplex.tools_path must contain a JSON tool list")
        schemas = [normalize_tool_schema(item) for item in value]
    return tuple(ensure_lean_task_tools(schemas))


def _duplex_params(config: DuplexConfig) -> DuplexParams:
    from mcpmft.infer.online import DuplexParams

    max_new_speak = (
        config.max_new_speak_tokens_per_chunk
        if config.max_new_speak_tokens_per_chunk is not None
        else config.speak_text_tokens_per_unit + 3
    )
    return DuplexParams(
        generate_audio=config.generate_audio,
        decode_mode=config.decode_mode,
        sliding_window_mode=config.sliding_window_mode,
        speak_text_tokens_per_unit=config.speak_text_tokens_per_unit,
        talker_speech_tokens_per_unit=config.talker_speech_tokens_per_unit,
        talker_final_speech_tokens_max=config.talker_final_speech_tokens_max,
        max_new_speak_tokens_per_chunk=max_new_speak,
        max_new_tool_tokens=config.max_new_tool_tokens,
        max_tool_response_tokens=config.max_tool_response_tokens,
        context_max_units=config.context_max_units,
        context_previous_max_tokens=config.context_previous_max_tokens,
        memory_slate_max_tokens=config.memory_slate_max_tokens,
        memory_lead_ratio=config.memory_lead_ratio,
        memory_soft_ratio=config.memory_soft_ratio,
        memory_hard_ratio=config.memory_hard_ratio,
        memory_kv_ceiling_units=config.memory_kv_ceiling_units,
        temperature=config.temperature,
        top_p=config.top_p,
        top_k=config.top_k,
    )


def build_app(config: ReleaseConfig):
    from mcpmft.infer.common import load_for_infer

    from .codex_coordinator import CodexCoordinator, CodexCoordinatorConfig
    from .contracts import storage_key
    from .gateway import GanderGateway, ProviderRegistry
    from .memory_provider import HttpMemoryProvider
    from .online_duplex import OnlineDuplexSettings, create_online_duplex_app
    from .providers import ProviderBuildContext, builtin_provider_registry
    from .supervision import TaskLedger

    server = config.server
    duplex = config.duplex

    runtime_dir = Path(server.runtime_dir).expanduser().resolve()
    runtime_dir.mkdir(parents=True, exist_ok=True)
    ledger_dir = (
        Path(server.ledger_dir).expanduser().resolve()
        if server.ledger_dir
        else runtime_dir / "gateway"
    )
    worker_cwd = str(Path(config.worker.cwd).expanduser().resolve())
    coordinator_cwd = str(
        Path(config.coordinator.cwd or worker_cwd).expanduser().resolve()
    )
    provider_factory = builtin_provider_registry().configure(
        config.worker.provider,
        config.worker.settings,
    )
    detached = duplex.detached_talker_device is not None
    thinker_model = replace(config.model, init_tts=False) if detached else config.model
    bundle = load_for_infer(
        thinker_model,
        checkpoint=duplex.checkpoint,
        talker_checkpoint=None if detached else duplex.talker_checkpoint,
        init_token2wav=duplex.generate_audio and not detached,
    )
    detached_talker = None
    if duplex.detached_talker_device:
        import torch

        from mcpmft.infer.detached_talker import (
            DetachedTalkerConfig,
            DetachedTalkerRuntime,
        )

        thinker_device = torch.device(next(bundle.model.parameters()).device)
        talker_device = torch.device(duplex.detached_talker_device)
        if talker_device.type != "cuda":
            raise ValueError("duplex.detached_talker_device must be a CUDA device")
        if thinker_device == talker_device:
            raise ValueError("Thinker and detached Talker must use different devices")
        detached_talker = DetachedTalkerRuntime.from_thinker_model(
            bundle.model,
            base_model_checkpoint=config.model.model_name_or_path,
            talker_checkpoint=duplex.talker_checkpoint,
            token2wav_dir=config.model.token2wav_dir,
            prompt_wav_path=duplex.ref_audio_path,
            device=str(talker_device),
            n_timesteps=config.model.token2wav_n_timesteps,
            enable_float16=config.model.token2wav_enable_float16,
            config=DetachedTalkerConfig(
                speech_tokens_per_unit=duplex.talker_speech_tokens_per_unit,
                emit_speech_tokens=duplex.talker_emit_speech_tokens,
                final_speech_tokens_max=duplex.talker_final_speech_tokens_max,
            ),
        )
    params = _duplex_params(duplex)
    settings = OnlineDuplexSettings(
        decode_mode=duplex.decode_mode,
        system_prompt=duplex.system_prompt,
        ref_audio_path=duplex.ref_audio_path,
        trailing_silence_sec=duplex.trailing_silence_sec,
        asr_base_url=asr_base_url(config.asr),
        asr_timeout_sec=config.asr.request_timeout_sec,
        turn_bind_grace_sec=duplex.turn_bind_grace_sec,
        media_mode=duplex.media_mode,
        allow_client_video=duplex.allow_client_video,
        client_video_mode=duplex.client_video_mode,
        client_video_sources=tuple(duplex.client_video_sources),
        vision_max_slice_nums=duplex.vision_max_slice_nums,
        vision_batch_feed=duplex.vision_batch_feed,
        max_screen_frame_bytes=duplex.max_screen_frame_bytes,
        max_screen_pixels=duplex.max_screen_pixels,
        codex_frame_rate_multiplier=duplex.codex_frame_rate_multiplier,
        codex_screen_history_seconds=duplex.codex_screen_history_seconds,
        tool_schemas=_tool_schemas(duplex.tools_path),
        expose_task_slate_to_model=duplex.expose_task_slate_to_model,
        warm_first_unit=duplex.warm_first_unit,
    )
    memory_provider = (
        HttpMemoryProvider(
            config.memory.url,
            bearer_token=os.environ.get(config.memory.token_env),
            timeout_s=config.memory.timeout_sec,
        )
        if config.memory.url
        else None
    )

    def gateway_factory(session_id: str) -> GanderGateway:
        session_key = storage_key(session_id)
        ledger_dir.mkdir(parents=True, exist_ok=True)
        ledger = TaskLedger(ledger_dir / f"{session_key}.sqlite")
        provider = provider_factory.create(
            ProviderBuildContext(
                workspace=Path(worker_cwd),
                runtime_dir=runtime_dir / config.worker.provider / session_key,
                runtime_profile=config.worker.profile,
            )
        )
        coordinator = (
            CodexCoordinator(
                CodexCoordinatorConfig(
                    cwd=coordinator_cwd,
                    runtime_dir=str(runtime_dir / "coordinator" / session_key),
                    codex_bin=config.coordinator.codex_bin,
                    model=config.coordinator.model,
                    reasoning_effort=config.coordinator.reasoning_effort,
                    codex_home=config.coordinator.codex_home,
                    turn_timeout_s=config.coordinator.timeout_sec,
                    max_turns_per_thread=config.coordinator.max_turns,
                ),
                ledger,
            )
            if server.mode == "coordinator"
            else None
        )
        return GanderGateway(
            coordinator=coordinator,
            providers=ProviderRegistry((provider,)),
            ledger=ledger,
            mode=server.mode,
            memory_provider=memory_provider,
            memory_search_timeout_s=config.memory.timeout_sec,
            max_realtime_context_events=server.context_max_events,
            coordinator_timeout_s=config.coordinator.timeout_sec + 5.0,
        )

    return create_online_duplex_app(
        bundle,
        params=params,
        settings=settings,
        gateway_factory=gateway_factory,
        provider_name=provider_factory.provider_name,
        media_dir=runtime_dir / "media",
        detached_talker=detached_talker,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Serve the complete Gander runtime")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="validate dependencies and paths without loading models",
    )
    args = parser.parse_args(argv)
    config = load_config(args.config)
    logging.basicConfig(
        level=config.server.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    preflight_config(config)
    if args.check_config:
        print(
            json.dumps(
                {
                    "status": "ok",
                    "server": f"{config.server.host}:{config.server.port}",
                    "asr": {
                        "mode": config.asr.mode,
                        "url": asr_base_url(config.asr),
                    },
                    "worker": {
                        "provider": config.worker.provider,
                        "profile": config.worker.profile,
                    },
                    "sliding_window_mode": config.duplex.sliding_window_mode,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    with AsrService(config.asr):
        if config.server.cuda_visible_devices:
            os.environ["CUDA_VISIBLE_DEVICES"] = config.server.cuda_visible_devices
        app = build_app(config)

        import uvicorn

        uvicorn.run(
            app,
            host=config.server.host,
            port=config.server.port,
            log_level=config.server.log_level.lower(),
            ws_ping_interval=config.server.ws_ping_interval,
            ws_ping_timeout=config.server.ws_ping_timeout,
        )


if __name__ == "__main__":
    main()
