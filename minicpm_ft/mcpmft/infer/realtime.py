from __future__ import annotations

import logging
import math
import sys
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Literal, Mapping, Protocol, Sequence

import numpy as np

from mcpmft.infer.common import InferBundle
from mcpmft.infer.detached_talker import (
    AsyncTalkerWorker,
    DetachedTalkerRuntime,
    PlaybackCancel,
    SpeechSynthesisChunk,
    SpeechSynthesisDone,
    SpeechSynthesisError,
)
from mcpmft.infer.online import DuplexParams, DuplexPrefixSnapshot, OnlineRunner

LOGGER = logging.getLogger(__name__)

MediaMode = Literal["voice", "omni", "auto"]


class UnitFrame(Protocol):
    frame_id: str
    image: Any


class UnitFrameSource(Protocol):
    def consume_for_unit(
        self,
        *,
        reuse_base: bool = False,
        captured_not_after_ms: float | None = None,
    ) -> tuple[UnitFrame, ...]: ...


def pcm16_bytes_to_float32(data: bytes | bytearray | memoryview) -> np.ndarray:
    """Convert little-endian mono PCM16 bytes to float32 waveform in [-1, 1]."""
    raw = bytes(data)
    if len(raw) % 2:
        raise ValueError("pcm16 audio byte length must be even")
    if not raw:
        return np.zeros(0, dtype=np.float32)
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


def waveform_to_pcm16_bytes(waveform: Any) -> bytes:
    """Convert a numpy/torch/list waveform to little-endian PCM16 bytes."""
    torch = sys.modules.get("torch")
    if torch is not None:
        if torch.is_tensor(waveform):
            waveform = waveform.detach().cpu().numpy()
    audio = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        return b""
    audio = np.clip(audio, -1.0, 1.0)
    pcm = np.where(audio < 0, audio * 32768.0, audio * 32767.0).astype("<i2")
    return pcm.tobytes()


@dataclass
class Pcm16Chunker:
    sample_rate: int = 16000
    chunk_ms: int = 1000

    def __post_init__(self) -> None:
        self.chunk_samples = int(self.sample_rate * self.chunk_ms / 1000)
        self.chunk_bytes = self.chunk_samples * 2
        self._buffer = bytearray()

    def push(self, data: bytes | bytearray | memoryview) -> list[np.ndarray]:
        self._buffer.extend(data)
        chunks: list[np.ndarray] = []
        while len(self._buffer) >= self.chunk_bytes:
            raw = bytes(self._buffer[: self.chunk_bytes])
            del self._buffer[: self.chunk_bytes]
            chunks.append(pcm16_bytes_to_float32(raw))
        return chunks

    def flush_padded(self) -> list[np.ndarray]:
        if not self._buffer:
            return []
        if len(self._buffer) % 2:
            raise ValueError("pcm16 audio byte length must be even")
        chunk = pcm16_bytes_to_float32(bytes(self._buffer))
        self._buffer.clear()
        if len(chunk) < self.chunk_samples:
            chunk = np.pad(chunk, (0, self.chunk_samples - len(chunk)))
        return [chunk.astype(np.float32, copy=False)]

    def clear(self) -> None:
        self._buffer.clear()


@dataclass
class DuplexLiveConfig:
    trailing_silence_sec: float = 20.0
    stop_on_turn_end: bool = True
    send_listen_audio: bool = False
    stop_reply_wait_sec: float = 8.0
    input_speech_rms: float = 1e-4
    media_mode: MediaMode = "voice"
    max_slice_nums: int = 1
    batch_vision_feed: bool = False

    def __post_init__(self) -> None:
        if self.media_mode not in {"voice", "omni", "auto"}:
            raise ValueError(f"unsupported media_mode: {self.media_mode!r}")
        if (
            isinstance(self.max_slice_nums, bool)
            or not isinstance(self.max_slice_nums, int)
            or self.max_slice_nums <= 0
        ):
            raise ValueError("max_slice_nums must be a positive integer")
        if not isinstance(self.batch_vision_feed, bool):
            raise TypeError("batch_vision_feed must be a bool")


@dataclass
class DuplexStepEvent:
    index: int
    is_listen: bool
    text: str
    end_of_turn: bool
    current_time: int | None
    audio_waveform: Any | None
    metrics: dict[str, Any]
    generated_token_ids: list[int] = field(default_factory=list)
    generation_id: int = 0
    unit_id: int | None = None
    interrupted: bool = False
    is_tool_call: bool = False
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_error: str | None = None
    raw_tool_text: str = ""
    tool_generation_complete: bool = False
    tool_parse_valid: bool = False
    tool_schema_valid: bool = False
    tool_response_expected: bool = False

    def to_payload(self) -> dict[str, Any]:
        return {
            "type": (
                "tool.error"
                if self.is_tool_call and self.tool_error
                else ("tool.call" if self.is_tool_call else "chunk")
            ),
            "index": self.index,
            "is_listen": self.is_listen,
            "text": self.text,
            "end_of_turn": self.end_of_turn,
            "current_time": self.current_time,
            "audio": self.audio_waveform is not None,
            "generated_token_ids": self.generated_token_ids,
            "generation_id": self.generation_id,
            "unit_id": self.unit_id,
            "interrupted": self.interrupted,
            "tool_calls": self.tool_calls,
            "tool_error": self.tool_error,
            "raw_tool_text": self.raw_tool_text,
            "tool_generation_complete": self.tool_generation_complete,
            "tool_parse_valid": self.tool_parse_valid,
            "tool_schema_valid": self.tool_schema_valid,
            "tool_response_expected": self.tool_response_expected,
            "metrics": self.metrics,
        }


class DuplexLiveSession:
    """Stateful single-browser live session backed by MiniCPMODuplex.

    MiniCPMODuplex stores KV, audio, and Token2wav state on the model, so one instance
    serves one active model session.
    """

    def __init__(
        self,
        bundle: InferBundle,
        *,
        params: DuplexParams | None = None,
        system_prompt: str | None = None,
        ref_audio_path: str | None = None,
        config: DuplexLiveConfig | None = None,
        detached_talker: DetachedTalkerRuntime | Any | None = None,
        frame_source: UnitFrameSource | None = None,
        tools: Sequence[Mapping[str, Any]] | None = None,
        prefix_snapshot: DuplexPrefixSnapshot | None = None,
    ) -> None:
        runner_params = params or DuplexParams()
        if detached_talker is not None:
            runner_params = replace(runner_params, generate_audio=False)
        self.runner = OnlineRunner(bundle, runner_params)
        if prefix_snapshot is None:
            self.runner.prepare(
                system_prompt=system_prompt,
                ref_audio_path=None if detached_talker is not None else ref_audio_path,
                tools=tools,
            )
        else:
            self.runner.restore_prefix_snapshot(prefix_snapshot)
        self.config = config or DuplexLiveConfig()
        self.chunker = Pcm16Chunker(chunk_ms=self.runner.params.chunk_ms)
        self.speech_worker = (
            AsyncTalkerWorker(detached_talker) if detached_talker is not None else None
        )
        self.step_index = 0
        self.timeline_sec = 0.0
        self.output_unit_id = 0
        self.spoke_any = False
        self.awaiting_reply = False
        self._assistant_turn_open = False
        self._needs_new_output_generation = False
        self._frame_source = frame_source
        self._tool_response_pending = False
        self._queued_tool_responses: list[Any] = []
        self.closed = False

    def set_frame_source(self, frame_source: UnitFrameSource | None) -> None:
        self._frame_source = frame_source

    def feed_memory_episode(self, episode: Mapping[str, Any] | Any) -> bool:
        """Push one completed MM-Mem ``EpisodeCreated.entry`` into the protected context."""
        self._ensure_open()
        return self.runner.add_memory_episode(episode)

    def set_task_slate(self, slate: str) -> bool:
        self._ensure_open()
        return self.runner.set_task_slate(slate)

    def set_summary_needed_callback(
        self,
        callback: Callable[[dict[str, Any]], None] | None,
    ) -> None:
        self._ensure_open()
        self.runner.set_summary_needed_callback(callback)

    def feed_pcm16(
        self,
        data: bytes | bytearray | memoryview,
        *,
        unit_capture_start_ms: Sequence[float] | None = None,
    ) -> list[DuplexStepEvent]:
        self._ensure_open()
        chunks = self.chunker.push(data)
        if unit_capture_start_ms is None:
            unit_starts: Sequence[float | None] = [None] * len(chunks)
        else:
            if len(unit_capture_start_ms) != len(chunks):
                raise ValueError(
                    "unit_capture_start_ms must contain one timestamp per completed audio unit"
                )
            unit_starts = unit_capture_start_ms
        return [
            self._step(chunk, input_capture_start_ms=unit_start)
            for chunk, unit_start in zip(chunks, unit_starts)
        ]

    def flush_pending(
        self, *, unit_capture_start_ms: float | None = None
    ) -> list[DuplexStepEvent]:
        self._ensure_open()
        return [
            self._step(
                chunk,
                input_capture_start_ms=unit_capture_start_ms,
            )
            for chunk in self.chunker.flush_padded()
        ]

    def max_trailing_silence_chunks(self) -> int:
        return int(round(self.config.trailing_silence_sec * 1000 / self.runner.params.chunk_ms))

    def step_silence(self) -> DuplexStepEvent:
        self._ensure_open()
        silence = np.zeros(self.chunker.chunk_samples, dtype=np.float32)
        return self._step(silence)

    def should_stop_after(self, event: DuplexStepEvent) -> bool:
        if self.speech_worker is not None:
            return bool(self.config.stop_on_turn_end and self.spoke_any and event.end_of_turn)
        return bool(
            self.config.stop_on_turn_end
            and self.spoke_any
            and event.end_of_turn
            and self.talker_state()["drained"]
        )

    def talker_state(self) -> dict[str, Any]:
        if self.speech_worker is not None:
            return {
                **self.speech_worker.state(),
                "turn_ended": not self._assistant_turn_open,
                "text_position": 0,
                "pending_audio_tokens": 0,
                "mode": "detached",
            }
        if not self.runner.params.generate_audio:
            return {
                "active": False,
                "drained": True,
                "turn_ended": not self._assistant_turn_open,
                "text_position": 0,
                "pending_audio_tokens": 0,
            }
        duplex = self.runner.duplex
        turn_ended = duplex.current_turn_ended
        text_position = int(duplex.tts_text_start_pos or 0)
        kv_active = duplex.tts_past_key_values is not None
        turn_started = duplex.tts_current_turn_start_time is not None
        token_buffer = duplex.token2wav_buffer or ()
        buffered_tokens = len(token_buffer)
        active = turn_ended is False or kv_active or text_position > 0 or turn_started
        return {
            "active": active,
            "drained": not active,
            "turn_ended": turn_ended,
            "text_position": text_position,
            "pending_audio_tokens": buffered_tokens if active else 0,
        }

    def should_continue_draining(self, trailing_steps: int) -> bool:
        max_steps = self.max_trailing_silence_chunks()
        if trailing_steps >= max_steps:
            return False
        if self._queued_tool_responses:
            return True
        if not self.config.stop_on_turn_end:
            return True
        if self.speech_worker is None and not self.talker_state()["drained"]:
            return True
        if getattr(self, "_assistant_turn_open", False):
            # Advance the microphone timeline until the Thinker emits turn EOS, then
            # wait for queued detached Talker output.
            return True
        if not self.awaiting_reply:
            return False
        reply_wait_steps = math.ceil(
            self.config.stop_reply_wait_sec * 1000 / self.runner.params.chunk_ms
        )
        return trailing_steps < min(max_steps, reply_wait_steps)

    def close(self, *, drain_speech: bool = False) -> None:
        if self.closed:
            return
        self.closed = True
        self.chunker.clear()
        self._queued_tool_responses.clear()
        if self.speech_worker is not None:
            self.speech_worker.close(drain=drain_speech)

    def finish(self) -> list[DuplexStepEvent]:
        self._ensure_open()
        events = self.flush_pending()
        trailing_steps = 0
        while self.should_continue_draining(trailing_steps):
            event = self.step_silence()
            events.append(event)
            trailing_steps += 1
            if self.should_stop_after(event):
                break
        if self.speech_worker is not None:
            self.wait_for_speech()
        self.close(drain_speech=True)
        return events

    def set_break(self) -> None:
        if self.speech_worker is not None:
            self.speech_worker.cancel("explicit_break")
        self.runner.reset_streaming_text()
        self.runner.duplex.set_break_event()

    def interrupt_output(self) -> None:
        """End the current assistant output without pausing future model units."""

        if self.speech_worker is not None:
            self.speech_worker.cancel("runtime_interrupt")
        self.runner.reset_streaming_text()
        self.awaiting_reply = True
        self._assistant_turn_open = False
        self._needs_new_output_generation = False

    def clear_break(self) -> None:
        self.runner.duplex.clear_break_event()

    def poll_output(
        self,
        timeout: float = 0.0,
    ) -> (
        SpeechSynthesisChunk
        | SpeechSynthesisDone
        | PlaybackCancel
        | SpeechSynthesisError
        | None
    ):
        if self.speech_worker is None:
            return None
        return self.speech_worker.poll(timeout)

    def drain_outputs(
        self,
    ) -> list[
        SpeechSynthesisChunk
        | SpeechSynthesisDone
        | PlaybackCancel
        | SpeechSynthesisError
    ]:
        if self.speech_worker is None:
            return []
        return self.speech_worker.drain_outputs()

    def wait_for_speech(self, timeout: float | None = None) -> bool:
        if self.speech_worker is None:
            return True
        return self.speech_worker.wait_until_drained(timeout)

    def feed_tool_response(self, response: Any) -> None:
        """Queue a tool result for the next continuous audio-video input unit."""
        self._ensure_open()
        if not self._tool_response_pending:
            raise RuntimeError("there is no pending tool call for this response")
        self.runner.validate_tool_response(response)
        self._queued_tool_responses.append(response)
        self._tool_response_pending = False

    def feed_runtime_event(self, event: Any) -> None:
        """Queue one asynchronous Runtime event for the next media input unit.

        Worker deliveries use the training ``<tool_response>`` envelope and a slot
        separate from synchronous task-tool responses.
        """

        self._ensure_open()
        if self._tool_response_pending:
            raise RuntimeError(
                "cannot inject a runtime event while a tool response is pending"
            )
        self.runner.validate_tool_response(event)
        self._queued_tool_responses.append(event)

    def _step(
        self,
        waveform: np.ndarray,
        *,
        input_capture_start_ms: float | None = None,
    ) -> DuplexStepEvent:
        waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
        input_rms = float(np.sqrt(np.mean(np.square(waveform), dtype=np.float64)))
        input_peak = float(np.max(np.abs(waveform))) if waveform.size else 0.0
        frame_list, consumed_frame_ids = self._frames_for_unit(
            captured_not_after_ms=input_capture_start_ms
        )
        prefill_kwargs: dict[str, Any] = {"audio_waveform": waveform}
        if frame_list:
            # Duplex units combine video and one second of audio.
            prefill_kwargs.update(
                frame_list=frame_list,
                max_slice_nums=self.config.max_slice_nums,
                batch_vision_feed=self.config.batch_vision_feed,
            )
        queued_tool_response = (
            self._queued_tool_responses[0] if self._queued_tool_responses else None
        )
        if queued_tool_response is None:
            self.runner.duplex.streaming_prefill(**prefill_kwargs)
        else:
            self.runner.prefill_tool_response(
                queued_tool_response,
                **prefill_kwargs,
            )
            self._queued_tool_responses.pop(0)
        prefill_mode = "OMNI" if frame_list else "AUDIO"
        if queued_tool_response is not None:
            prefill_mode += "+TOOL"
        unit_start_sec = self.timeline_sec
        unit_end_sec = unit_start_sec + self.runner.params.chunk_ms / 1000.0
        event = self._generate_event(
            input_rms=input_rms,
            input_peak=input_peak,
            prefill_mode=prefill_mode,
            consumed_frame_ids=consumed_frame_ids,
            input_has_speech=input_rms >= self.config.input_speech_rms,
            unit_start_sec=unit_start_sec,
            unit_end_sec=unit_end_sec,
        )
        self.timeline_sec = unit_end_sec
        return event

    def _generate_event(
        self,
        *,
        input_rms: float,
        input_peak: float,
        prefill_mode: str,
        consumed_frame_ids: list[str],
        input_has_speech: bool,
        unit_start_sec: float,
        unit_end_sec: float,
    ) -> DuplexStepEvent:
        out = self.runner.streaming_generate(
            max_new_speak_tokens_per_chunk=self.runner.params.max_new_speak_tokens_per_chunk,
            decode_mode=self.runner.params.decode_mode,
            temperature=self.runner.params.temperature,
            top_p=self.runner.params.top_p,
            top_k=self.runner.params.top_k,
        )
        if not isinstance(out, dict):
            out = {"raw": out}
        record_unit_time = getattr(self.runner, "record_unit_time", None)
        if record_unit_time is not None:
            record_unit_time(unit_start_sec, unit_end_sec)

        is_interrupt = bool(out.get("is_interrupt", False))
        is_tool_call = bool(out.get("is_tool_call", False))
        if is_tool_call:
            if self._tool_response_pending:
                out = dict(out)
                out["tool_calls"] = []
                out["tool_error"] = "a previous tool call is still awaiting its response"
                out["tool_response_expected"] = False
            else:
                # Return one bounded result for a structurally complete malformed call.
                self._tool_response_pending = True
                out["tool_response_expected"] = True
        is_listen = bool(out.get("is_listen", True))
        talker_condition = None
        if self.speech_worker is not None and not is_tool_call:
            talker_condition = self.runner.take_talker_condition()
        if input_has_speech:
            self.awaiting_reply = True
        if is_interrupt:
            if self.speech_worker is not None:
                self.speech_worker.cancel("model_interrupt")
            self.awaiting_reply = True
            self._assistant_turn_open = False
            self._needs_new_output_generation = False
        elif not is_listen and not is_tool_call:
            self.spoke_any = True
            self.awaiting_reply = False
        end_of_turn = bool(out.get("end_of_turn", False))
        if (
            self.speech_worker is not None
            and not is_listen
            and talker_condition is not None
            and self._needs_new_output_generation
        ):
            self.speech_worker.cancel("new_assistant_turn")
            self._needs_new_output_generation = False
        if not is_listen and not end_of_turn and not is_tool_call:
            self._assistant_turn_open = True
        if end_of_turn:
            self._assistant_turn_open = False
            self._needs_new_output_generation = True
            if not input_has_speech:
                self.awaiting_reply = False
        audio_waveform = out.get("audio_waveform")
        if is_listen and not self.config.send_listen_audio:
            audio_waveform = None

        self.step_index += 1
        unit_id = None
        if not is_listen and not is_tool_call:
            self.output_unit_id += 1
            unit_id = self.output_unit_id
        if talker_condition is not None and unit_id is not None:
            token_ids, hidden_states = talker_condition
            self.speech_worker.submit(
                unit_id=unit_id,
                current_time=out.get("current_time"),
                token_ids=token_ids,
                hidden_states=hidden_states,
                end_of_turn=end_of_turn,
            )
        metric_keys = (
            "cost_llm",
            "cost_tts_prep",
            "cost_tts",
            "cost_token2wav",
            "cost_all",
            "n_tokens",
            "n_tts_tokens",
        )
        metrics = {key: out.get(key) for key in metric_keys if key in out}
        metrics.update({"input_rms": input_rms, "input_peak": input_peak})
        metrics.update(
            {
                "media_mode": self.config.media_mode,
                "prefill_mode": prefill_mode,
                "consumed_frame_ids": consumed_frame_ids,
            }
        )
        metrics["talker"] = self.talker_state()
        context_window = self._context_window_metrics()
        if context_window is not None:
            metrics["context_window"] = context_window
        event = DuplexStepEvent(
            index=self.step_index,
            is_listen=is_listen,
            text=str(out.get("text") or ""),
            end_of_turn=end_of_turn,
            current_time=out.get("current_time"),
            audio_waveform=audio_waveform,
            metrics=metrics,
            generated_token_ids=[int(value) for value in out.get("generated_token_ids", ())],
            generation_id=(
                self.speech_worker.generation_id if self.speech_worker is not None else 0
            ),
            unit_id=unit_id,
            interrupted=is_interrupt,
            is_tool_call=is_tool_call,
            tool_calls=list(out.get("tool_calls") or []),
            tool_error=out.get("tool_error"),
            raw_tool_text=str(out.get("raw_tool_text") or ""),
            tool_generation_complete=bool(out.get("tool_generation_complete", False)),
            tool_parse_valid=bool(out.get("tool_parse_valid", False)),
            tool_schema_valid=bool(out.get("tool_schema_valid", False)),
            tool_response_expected=bool(out.get("tool_response_expected", False)),
        )
        LOGGER.info(
            "duplex chunk=%d decision=%s input_rms=%.6f input_peak=%.6f text=%r",
            event.index,
            (
                "tool"
                if event.is_tool_call
                else ("interrupt" if event.interrupted else ("listen" if event.is_listen else "speak"))
            ),
            input_rms,
            input_peak,
            event.text,
        )
        return event

    def _frames_for_unit(
        self, *, captured_not_after_ms: float | None = None
    ) -> tuple[list[Any], list[str]]:
        media_mode = self.config.media_mode
        if media_mode == "voice":
            return [], []

        source = self._frame_source
        if source is None:
            return [], []

        if captured_not_after_ms is None:
            frames = source.consume_for_unit(reuse_base=media_mode == "omni")
        else:
            frames = source.consume_for_unit(
                reuse_base=media_mode == "omni",
                captured_not_after_ms=captured_not_after_ms,
            )
        return (
            [frame.image for frame in frames],
            [frame.frame_id for frame in frames],
        )

    def _context_window_metrics(self) -> dict[str, Any] | None:
        decoder = getattr(self.runner.duplex, "decoder", None)
        if decoder is None:
            return None
        get_stats = getattr(decoder, "get_window_stats", None)
        stats: dict[str, Any] = {}
        try:
            if callable(get_stats):
                stats = get_stats()
        except Exception:  # pragma: no cover
            LOGGER.debug("failed to collect decoder window stats", exc_info=True)
        config = stats.get("config") or getattr(decoder, "_window_config", None)

        def config_value(name: str, default: Any = 0) -> Any:
            if isinstance(config, dict):
                return config.get(name, default)
            return getattr(config, name, default)

        unit_count = stats.get("unit_count")
        if unit_count is None:
            unit_count = len(getattr(decoder, "_unit_history", ()))
        previous_tokens = stats.get("previous_token_count")
        if previous_tokens is None:
            previous_tokens = len(getattr(decoder, "_previous_token_ids", ()))
        return {
            "mode": getattr(
                decoder,
                "_reported_sliding_window_mode",
                config_value("sliding_window_mode"),
            ),
            "unit_count": unit_count,
            "max_units": config_value("context_max_units"),
            "previous_tokens": previous_tokens,
            "previous_max_tokens": config_value("context_previous_max_tokens"),
            "sliding_events": getattr(decoder, "_sliding_event_count", 0),
            "dropped_units": getattr(decoder, "_total_dropped_units", 0),
        }

    def _ensure_open(self) -> None:
        if self.closed:
            raise RuntimeError("duplex live session is already closed")
