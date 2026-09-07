from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Literal

from mcpmft.args import (
    ModelArguments,
    apply_overrides,
    load_yaml_files,
)
from mcpmft.infer.common import load_for_infer
from mcpmft.infer.offline import OfflineRunner, build_user_messages
from mcpmft.infer.online import DuplexParams, OnlineRunner
from mcpmft.prompts import GANDER_DUPLEX_SYSTEM_PROMPT
from mcpmft.tool_protocol import ensure_lean_task_tools, normalize_tool_schema


@dataclass(frozen=True)
class InputConfig:
    text: str | None = None
    audio: str | None = None
    image: str | None = None
    ref_audio: str | None = None


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = 512
    min_new_tokens: int = 0
    do_sample: bool = True
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 100
    repetition_penalty: float = 1.02
    max_inp_length: int = 8192
    max_slice_nums: int = 1


@dataclass(frozen=True)
class DuplexConfig:
    system_prompt: str = GANDER_DUPLEX_SYSTEM_PROMPT
    tools_path: str | None = None
    enable_task_tools: bool = False
    trailing_silence_sec: float = 20.0
    stop_on_turn_end: bool = True
    chunk_ms: int = 1000
    first_chunk_ms: int = 1035
    ls_mode: str = "explicit"
    speak_text_tokens_per_unit: int = 4
    max_new_speak_tokens_per_chunk: int | None = None
    max_new_tool_tokens: int = 96
    max_tool_response_tokens: int = 256
    max_tool_calls_per_unit: int = 4
    max_tool_schemas: int = 6
    max_tool_schema_tokens: int = 1024
    inject_search_time_context: bool = False
    decode_mode: Literal["sampling", "greedy"] = "sampling"
    n_timesteps: int = 10
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    sliding_window_mode: Literal[
        "off",
        "basic",
        "context",
        "context_no_previous",
        "context_memory",
        "context_slate",
    ] = "context_no_previous"
    basic_window_high_tokens: int = 8000
    basic_window_low_tokens: int = 6000
    context_previous_max_tokens: int = 500
    context_max_units: int = 128
    memory_slate_max_tokens: int = 256
    memory_lead_ratio: float = 0.7
    memory_soft_ratio: float = 0.9
    memory_hard_ratio: float = 1.0
    memory_kv_ceiling_units: int | None = None
    talker_speech_tokens_per_unit: int = 25
    talker_final_speech_tokens_max: int = 0


@dataclass(frozen=True)
class InferenceConfig:
    mode: Literal["turn", "duplex"] = "turn"
    checkpoint: str | None = None
    talker_checkpoint: str | None = None
    generate_audio: bool = False
    output_audio: str | None = None
    enable_thinking: bool = False
    omni_mode: bool = False
    sys_mode: str = "audio_assistant"
    language: str = "zh"
    strict_checkpoint: bool = False
    input: InputConfig = InputConfig()
    generation: GenerationConfig = GenerationConfig()
    duplex: DuplexConfig = DuplexConfig()


@dataclass(frozen=True)
class OfflineConfig:
    model: ModelArguments
    inference: InferenceConfig


def _typed(cls: type[Any], values: Any, name: str) -> Any:
    if not isinstance(values, dict):
        raise ValueError(f"{name} must be a YAML mapping")
    known = {item.name for item in fields(cls)}
    unknown = sorted(set(values) - known)
    if unknown:
        raise ValueError(
            "Unknown setting(s): " + ", ".join(f"{name}.{key}" for key in unknown)
        )
    return cls(**values)


def load_config(path: str | Path, overrides: list[str] | None = None) -> OfflineConfig:
    document = apply_overrides(load_yaml_files([path]), overrides or [])
    unknown = sorted(set(document) - {"model", "inference"})
    if unknown:
        raise ValueError("Unknown top-level section(s): " + ", ".join(unknown))

    inference_values = dict(document.get("inference") or {})
    input_config = _typed(
        InputConfig,
        inference_values.pop("input", {}),
        "inference.input",
    )
    generation_config = _typed(
        GenerationConfig,
        inference_values.pop("generation", {}),
        "inference.generation",
    )
    duplex_config = _typed(
        DuplexConfig,
        inference_values.pop("duplex", {}),
        "inference.duplex",
    )
    inference = _typed(
        InferenceConfig,
        {
            **inference_values,
            "input": input_config,
            "generation": generation_config,
            "duplex": duplex_config,
        },
        "inference",
    )
    model = _typed(ModelArguments, document.get("model") or {}, "model")
    _validate(model, inference)
    return OfflineConfig(model=model, inference=inference)


def _validate(model: ModelArguments, inference: InferenceConfig) -> None:
    if inference.mode not in {"turn", "duplex"}:
        raise ValueError("inference.mode must be 'turn' or 'duplex'")
    if inference.mode == "turn":
        generation = inference.generation
        if generation.max_new_tokens < 1:
            raise ValueError("inference.generation.max_new_tokens must be positive")
        if not 0 <= generation.min_new_tokens <= generation.max_new_tokens:
            raise ValueError(
                "inference.generation.min_new_tokens must be between zero and max_new_tokens"
            )
    else:
        duplex = inference.duplex
        if not inference.input.audio:
            raise ValueError("Duplex inference requires inference.input.audio")
        if inference.input.text or inference.input.image:
            raise ValueError(
                "Duplex inference consumes inference.input.audio; text/image belong to turn mode"
            )
        if duplex.chunk_ms <= 0:
            raise ValueError("inference.duplex.chunk_ms must be positive")
        if duplex.trailing_silence_sec < 0:
            raise ValueError(
                "inference.duplex.trailing_silence_sec must not be negative"
            )
        if duplex.speak_text_tokens_per_unit < 1:
            raise ValueError(
                "inference.duplex.speak_text_tokens_per_unit must be positive"
            )
        minimum_budget = duplex.speak_text_tokens_per_unit + 3
        if (
            duplex.max_new_speak_tokens_per_chunk is not None
            and duplex.max_new_speak_tokens_per_chunk < minimum_budget
        ):
            raise ValueError(
                "inference.duplex.max_new_speak_tokens_per_chunk must be at least "
                f"speak_text_tokens_per_unit + 3 ({minimum_budget})"
            )
        if duplex.talker_speech_tokens_per_unit < 1:
            raise ValueError(
                "inference.duplex.talker_speech_tokens_per_unit must be positive"
            )
        if duplex.talker_final_speech_tokens_max < 0:
            raise ValueError(
                "inference.duplex.talker_final_speech_tokens_max must be non-negative"
            )
    if inference.generate_audio:
        if not model.init_tts or not model.token2wav_dir:
            raise ValueError(
                "Audio generation requires model.init_tts=true and model.token2wav_dir"
            )
        if not inference.input.ref_audio or not inference.output_audio:
            raise ValueError(
                "Audio generation requires inference.input.ref_audio and inference.output_audio"
            )
    elif inference.output_audio:
        raise ValueError(
            "inference.output_audio is only valid when inference.generate_audio=true"
        )


def run(config: OfflineConfig) -> dict[str, Any]:
    if config.inference.mode == "duplex":
        return _run_duplex(config)
    return _run_turn(config)


def _run_turn(config: OfflineConfig) -> dict[str, Any]:
    inference = config.inference
    bundle = load_for_infer(
        config.model,
        checkpoint=inference.checkpoint,
        talker_checkpoint=inference.talker_checkpoint,
        strict=inference.strict_checkpoint,
        init_token2wav=inference.generate_audio,
        load_processor=True,
    )
    runner = OfflineRunner(bundle)
    generation_kwargs = vars(inference.generation).copy()
    if not inference.generation.do_sample:
        for name in ("temperature", "top_p", "top_k"):
            generation_kwargs.pop(name)
    return runner.generate(
        build_user_messages(
            text=inference.input.text,
            audio_path=inference.input.audio,
            image_path=inference.input.image,
        ),
        generate_audio=inference.generate_audio,
        output_audio_path=inference.output_audio,
        ref_audio_path=inference.input.ref_audio,
        enable_thinking=inference.enable_thinking,
        omni_mode=inference.omni_mode,
        sys_mode=inference.sys_mode,
        language=inference.language,
        **generation_kwargs,
    )


def _run_duplex(config: OfflineConfig) -> dict[str, Any]:
    inference = config.inference
    duplex = inference.duplex
    bundle = load_for_infer(
        config.model,
        checkpoint=inference.checkpoint,
        talker_checkpoint=inference.talker_checkpoint,
        strict=inference.strict_checkpoint,
        # MiniCPM-o's as_duplex() owns Token2wav initialization.
        init_token2wav=False,
        load_processor=False,
    )
    runner = OnlineRunner(bundle, _duplex_params(duplex, inference.generate_audio))
    runner.prepare(
        system_prompt=duplex.system_prompt,
        ref_audio_path=inference.input.ref_audio,
        tools=_load_tools(duplex),
    )
    outputs = runner.run_audio_file(
        inference.input.audio,
        output_wav=inference.output_audio,
        trailing_silence_sec=duplex.trailing_silence_sec,
        stop_on_turn_end=duplex.stop_on_turn_end,
    )
    units = [_duplex_unit(index, output) for index, output in enumerate(outputs, 1)]
    result: dict[str, Any] = {
        "mode": "duplex",
        "text": "".join(unit["text"] for unit in units),
        "units": units,
    }
    if inference.output_audio:
        result["output_audio"] = inference.output_audio
    return result


def _duplex_params(config: DuplexConfig, generate_audio: bool) -> DuplexParams:
    max_new_speak = (
        config.max_new_speak_tokens_per_chunk
        if config.max_new_speak_tokens_per_chunk is not None
        else config.speak_text_tokens_per_unit + 3
    )
    return DuplexParams(
        chunk_ms=config.chunk_ms,
        first_chunk_ms=config.first_chunk_ms,
        ls_mode=config.ls_mode,
        speak_text_tokens_per_unit=config.speak_text_tokens_per_unit,
        max_new_speak_tokens_per_chunk=max_new_speak,
        max_new_tool_tokens=config.max_new_tool_tokens,
        max_tool_response_tokens=config.max_tool_response_tokens,
        max_tool_calls_per_unit=config.max_tool_calls_per_unit,
        max_tool_schemas=config.max_tool_schemas,
        max_tool_schema_tokens=config.max_tool_schema_tokens,
        inject_search_time_context=config.inject_search_time_context,
        decode_mode=config.decode_mode,
        generate_audio=generate_audio,
        n_timesteps=config.n_timesteps,
        temperature=config.temperature,
        top_p=config.top_p,
        top_k=config.top_k,
        sliding_window_mode=config.sliding_window_mode,
        basic_window_high_tokens=config.basic_window_high_tokens,
        basic_window_low_tokens=config.basic_window_low_tokens,
        context_previous_max_tokens=config.context_previous_max_tokens,
        context_max_units=config.context_max_units,
        memory_slate_max_tokens=config.memory_slate_max_tokens,
        memory_lead_ratio=config.memory_lead_ratio,
        memory_soft_ratio=config.memory_soft_ratio,
        memory_hard_ratio=config.memory_hard_ratio,
        memory_kv_ceiling_units=config.memory_kv_ceiling_units,
        talker_speech_tokens_per_unit=config.talker_speech_tokens_per_unit,
        talker_final_speech_tokens_max=config.talker_final_speech_tokens_max,
    )


def _load_tools(config: DuplexConfig) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    if config.tools_path:
        value = json.loads(
            Path(config.tools_path).expanduser().read_text(encoding="utf-8")
        )
        value = value.get("tools") if isinstance(value, dict) else value
        if not isinstance(value, list):
            raise ValueError(
                "inference.duplex.tools_path must contain a JSON tool list"
            )
        tools = [normalize_tool_schema(item) for item in value]
    task_names = {"task_start", "task_send", "task_resolve"}
    if config.enable_task_tools or any(tool["name"] in task_names for tool in tools):
        tools = ensure_lean_task_tools(tools)
    return tools


def _duplex_unit(index: int, output: Any) -> dict[str, Any]:
    if not isinstance(output, dict):
        raise TypeError(
            f"Unsupported duplex inference result: {type(output).__name__}"
        )
    is_interrupt = bool(output.get("is_interrupt", False))
    is_tool_call = bool(output.get("is_tool_call", False))
    is_listen = bool(output.get("is_listen", True))
    decision = (
        "tool"
        if is_tool_call
        else ("interrupt" if is_interrupt else ("listen" if is_listen else "speak"))
    )
    metrics: dict[str, int | float] = {}
    for key in ("cost_llm", "cost_tts_prep", "cost_tts", "cost_token2wav", "cost_all"):
        if output.get(key) is not None:
            metrics[key] = float(output[key])
    for key in ("n_tokens", "n_tts_tokens"):
        if output.get(key) is not None:
            metrics[key] = int(output[key])
    waveform = output.get("audio_waveform")
    current_time = output.get("current_time")
    unit: dict[str, Any] = {
        "index": index,
        "decision": decision,
        "is_listen": is_listen,
        "is_interrupt": is_interrupt,
        "is_tool_call": is_tool_call,
        "text": str(output.get("text") or ""),
        "end_of_turn": bool(output.get("end_of_turn", False)),
        "current_time": int(current_time) if current_time is not None else None,
        "generated_token_ids": [
            int(token_id) for token_id in output.get("generated_token_ids", ())
        ],
        "audio_samples": (
            int(waveform.numel())
            if waveform is not None and hasattr(waveform, "numel")
            else (int(getattr(waveform, "size", 0)) if waveform is not None else 0)
        ),
        "tool_calls": list(output.get("tool_calls") or []),
        "tool_error": output.get("tool_error"),
        "raw_tool_text": str(output.get("raw_tool_text") or ""),
        "tool_generation_complete": bool(
            output.get("tool_generation_complete", False)
        ),
        "tool_parse_valid": bool(output.get("tool_parse_valid", False)),
        "tool_schema_valid": bool(output.get("tool_schema_valid", False)),
        "metrics": metrics,
    }
    return unit


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run MiniCPM-o offline inference")
    parser.add_argument("--config", required=True)
    args, overrides = parser.parse_known_args(argv)
    config = load_config(args.config, overrides)
    result = run(config)
    if config.inference.mode == "duplex":
        summary = result
    else:
        summary = {"mode": "turn", "text": result.get("text", "")}
        if result.get("audio") is not None:
            summary["audio_samples"] = int(result["audio"].size)
            summary["output_audio"] = config.inference.output_audio
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
