from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import secrets
import time
import urllib.parse
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

from .contracts import MediaRef
from .duplex_bridge import GanderDuplexSession
from .media_mode import (
    CLIENT_VIDEO_MODES,
    VIDEO_SOURCES,
    client_video_allowed,
    estimated_tokens_per_unit,
    resolve_target_mode,
    source_warnings,
    vision_available,
)
from .media_timeline import (
    AUDIO_INPUT_PROTOCOL,
    AudioCaptureTimeline,
    AudioFrameHeader,
)
from .screen import LatestScreenFrameBuffer, ScreenFrameRateGate
from .screen_transport import (
    DecodedScreenFrame,
    ScreenFrameHeader,
    decode_screen_frame,
    persist_screen_frame,
)
from .task_tools_online import TaskToolsRealtimeCoordinator

LOGGER = logging.getLogger(__name__)
_SESSION_ID = re.compile(r"^(?!\.{1,2}$)[A-Za-z0-9_.-]{1,128}$")
ONLINE_TASK_PROTOCOL = "task_tools_v1"
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse


@dataclass(frozen=True)
class OnlineDuplexSettings:
    decode_mode: Literal["sampling", "greedy"] = "sampling"
    system_prompt: str | None = None
    ref_audio_path: str | None = None
    trailing_silence_sec: float = 20.0
    send_listen_audio: bool = False
    input_sample_rate: int = 16000
    output_sample_rate: int = 24000
    asr_base_url: str | None = None
    asr_timeout_sec: float = 120.0
    turn_bind_grace_sec: float = 5.0
    # The mode every session STARTS in. "voice" keeps the audio-only path.
    media_mode: Literal["voice", "omni", "auto"] = "voice"
    # Whether a session may switch vision on itself through `media.mode`. An
    # audio-first deployment must opt in, so `media_mode="voice"` alone never
    # grows a video surface.
    allow_client_video: bool = False
    client_video_mode: Literal["omni", "auto"] = "omni"
    client_video_sources: tuple[str, ...] = VIDEO_SOURCES
    vision_max_slice_nums: int = 1
    vision_batch_feed: bool = False
    max_screen_frame_bytes: int = 4 * 1024 * 1024
    max_screen_pixels: int = 4096 * 4096
    codex_frame_rate_multiplier: float = 3.0
    codex_screen_history_seconds: float = 8.0
    tool_schemas: tuple[dict[str, Any], ...] = ()
    expose_task_slate_to_model: bool = False
    warm_first_unit: bool = True


@dataclass
class _ActiveSession:
    duplex: GanderDuplexSession
    screen_token: str
    codex_frame_gate: ScreenFrameRateGate
    # Live per-session media state. The settings value only seeds it; a client may
    # move it with `media.mode` for the rest of the session.
    media_mode: str = "voice"
    video_source: str | None = None
    coordinator: Any | None = None
    ended: asyncio.Event = field(default_factory=asyncio.Event)


@dataclass
class _Runtime:
    bundle: Any
    params: Any
    settings: OnlineDuplexSettings
    media_dir: Path
    gateway_factory: Callable[[str], Any]
    provider_name: str
    detached_talker: Any | None = None
    prefix_snapshot: Any | None = None
    prefix_cache_status: str = "pending"
    prefix_prepare_seconds: float | None = None
    first_unit_warmup_seconds: float | None = None
    model_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    sessions: dict[str, _ActiveSession] = field(default_factory=dict)


def _json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str)


async def _warm_backbrain_providers(gateway: Any) -> None:
    """Start each worker process before the realtime session is announced ready."""

    for provider in gateway.providers.values():
        await provider.warmup()


async def _screen_media_ref(
    runtime: _Runtime,
    session_id: str,
    header: ScreenFrameHeader,
    payload: bytes,
    decoded: DecodedScreenFrame,
    *,
    kind: Literal["screen", "frame"] = "screen",
) -> MediaRef:
    media = await asyncio.to_thread(
        persist_screen_frame,
        runtime.media_dir,
        session_id,
        header,
        payload,
        decoded,
        kind=kind,
    )
    return media


def _build_session(
    runtime: _Runtime,
    *,
    media_mode: str | None = None,
    screen_frames: LatestScreenFrameBuffer | None = None,
) -> GanderDuplexSession:
    from mcpmft.infer.realtime import DuplexLiveConfig, DuplexLiveSession
    from mcpmft.prompts import GANDER_DUPLEX_SYSTEM_PROMPT
    from mcpmft.tool_protocol import ensure_lean_task_tools

    tools = ensure_lean_task_tools(runtime.settings.tool_schemas)

    live = DuplexLiveSession(
        runtime.bundle,
        params=runtime.params,
        system_prompt=runtime.settings.system_prompt or GANDER_DUPLEX_SYSTEM_PROMPT,
        ref_audio_path=runtime.settings.ref_audio_path,
        config=DuplexLiveConfig(
            trailing_silence_sec=runtime.settings.trailing_silence_sec,
            send_listen_audio=runtime.settings.send_listen_audio,
            media_mode=(
                media_mode if media_mode is not None else runtime.settings.media_mode
            ),
            max_slice_nums=runtime.settings.vision_max_slice_nums,
            batch_vision_feed=runtime.settings.vision_batch_feed,
        ),
        detached_talker=runtime.detached_talker,
        tools=tools,
        prefix_snapshot=runtime.prefix_snapshot,
    )
    if screen_frames is None:
        # One browser frame represents one model unit. Keep a full context
        # window so inference lag cannot evict the frame needed by queued audio.
        screen_frames = LatestScreenFrameBuffer(
            max_pending_frames=runtime.params.context_max_units
        )
    return GanderDuplexSession(
        live,
        decode_mode=runtime.settings.decode_mode,
        screen_frames=screen_frames,
    )


def _prepare_static_prefix(runtime: _Runtime) -> None:
    """Prefill the process-static system/tool prefix once for all live sessions."""
    from mcpmft.infer.online import OnlineRunner
    from mcpmft.prompts import GANDER_DUPLEX_SYSTEM_PROMPT
    from mcpmft.tool_protocol import ensure_lean_task_tools

    started = time.perf_counter()
    runner_params = runtime.params
    if runtime.detached_talker is not None:
        runner_params = replace(runner_params, generate_audio=False)
    runner = OnlineRunner(runtime.bundle, runner_params)
    runner.prepare(
        system_prompt=(
            runtime.settings.system_prompt or GANDER_DUPLEX_SYSTEM_PROMPT
        ),
        ref_audio_path=(
            None if runtime.detached_talker is not None else runtime.settings.ref_audio_path
        ),
        tools=ensure_lean_task_tools(runtime.settings.tool_schemas),
    )
    runtime.prefix_snapshot = runner.capture_prefix_snapshot()
    runtime.prefix_prepare_seconds = time.perf_counter() - started
    runtime.prefix_cache_status = "ready"
    LOGGER.info(
        "Prepared static Gander prefix cache: tokens=%d seconds=%.3f",
        runtime.prefix_snapshot.token_count,
        runtime.prefix_prepare_seconds,
    )

    if not runtime.settings.warm_first_unit:
        return
    if runtime.detached_talker is not None:
        started = time.perf_counter()
        runtime.detached_talker.warm_token2wav()
        runtime.first_unit_warmup_seconds = time.perf_counter() - started
        LOGGER.info(
            "Warmed detached Token2wav first chunk: seconds=%.3f",
            runtime.first_unit_warmup_seconds,
        )
        return
    started = time.perf_counter()
    warm_session = _build_session(runtime)
    try:
        event = warm_session.step_silence()
    finally:
        warm_session.close()
    runtime.first_unit_warmup_seconds = time.perf_counter() - started
    LOGGER.info(
        "Warmed first Gander duplex unit: decision=%s seconds=%.3f",
        "listen" if event.is_listen else "speak",
        runtime.first_unit_warmup_seconds,
    )


async def _open_session(
    runtime: _Runtime,
    *,
    media_mode: str | None = None,
    screen_frames: LatestScreenFrameBuffer | None = None,
) -> GanderDuplexSession:
    return await asyncio.to_thread(
        _build_session,
        runtime,
        media_mode=media_mode,
        screen_frames=screen_frames,
    )


def _tool_names(runtime: _Runtime) -> list[str]:
    names = [str(schema.get("name") or "") for schema in runtime.settings.tool_schemas]
    names = [name for name in names if name]
    required = ("task_start", "task_send", "task_resolve")
    return list(dict.fromkeys([*names, *required]))


def _codex_frame_interval_ms(runtime: _Runtime) -> float:
    return runtime.params.chunk_ms / runtime.settings.codex_frame_rate_multiplier


def _codex_frame_rate(runtime: _Runtime) -> float:
    return 1000.0 / _codex_frame_interval_ms(runtime)


def _max_recent_screen_frames(runtime: _Runtime) -> int:
    return max(
        1,
        math.ceil(
            _codex_frame_rate(runtime)
            * runtime.settings.codex_screen_history_seconds
        ),
    )


def _vision_available(runtime: _Runtime) -> bool:
    return vision_available(runtime.bundle.model)


def _client_video_allowed(runtime: _Runtime) -> bool:
    return client_video_allowed(
        vision_ok=_vision_available(runtime),
        media_mode=runtime.settings.media_mode,
        allow_client_video=runtime.settings.allow_client_video,
    )


def _unit_frame_rate(runtime: _Runtime) -> float:
    """Frames per second the client should send: one per model unit."""

    return 1000.0 / runtime.params.chunk_ms


def _client_video_capabilities(runtime: _Runtime) -> dict[str, Any]:
    return {
        "enabled": _client_video_allowed(runtime),
        "vision_available": _vision_available(runtime),
        "sources": list(runtime.settings.client_video_sources),
        "mode": runtime.settings.client_video_mode,
        "recommended_frame_rate": _unit_frame_rate(runtime),
    }


def _screen_channel(runtime: _Runtime, active: _ActiveSession) -> dict[str, Any]:
    """Everything a client needs to start uploading frames on ``/ws/screen``."""

    return {
        "path": "/ws/screen",
        "token": active.screen_token,
        "protocol": "metadata-json+encoded-binary-v1",
        "encodings": ["jpeg", "webp", "png"],
        "max_frame_bytes": runtime.settings.max_screen_frame_bytes,
        "max_pixels": runtime.settings.max_screen_pixels,
        "recommended_frame_rate": _unit_frame_rate(runtime),
        "codex_frame_rate": _codex_frame_rate(runtime),
        "codex_screen_history_seconds": runtime.settings.codex_screen_history_seconds,
    }


def _reject_media_mode(
    runtime: _Runtime,
    *,
    want_video: bool,
    source: Any,
) -> str | None:
    """Return why a ``media.mode`` request is refused, or None to accept it."""

    if not want_video:
        return None
    if not _vision_available(runtime):
        return "the loaded model has no vision tower"
    if not _client_video_allowed(runtime):
        return "client video is disabled"
    if source not in runtime.settings.client_video_sources:
        return f"unsupported video source: {source!r}"
    return None


async def _send_chunk(
    websocket: WebSocket,
    event: Any,
    output_sample_rate: int,
) -> None:
    from mcpmft.infer.detached_talker import (
        PlaybackCancel,
        SpeechSynthesisChunk,
        SpeechSynthesisDone,
        SpeechSynthesisError,
    )
    from mcpmft.infer.realtime import waveform_to_pcm16_bytes

    audio = b""
    if isinstance(event, PlaybackCancel):
        payload = {
            "type": "playback.cancel",
            "generation_id": event.generation_id,
            "cancelled_generation_id": event.cancelled_generation_id,
            "reason": event.reason,
        }
    elif isinstance(event, SpeechSynthesisError):
        payload = {
            "type": "error",
            "message": f"Detached Talker failed: {event.message}",
            "generation_id": event.generation_id,
            "unit_id": event.unit_id,
        }
    elif isinstance(event, SpeechSynthesisDone):
        payload = {
            "type": "audio.done",
            "generation_id": event.generation_id,
            "unit_id": event.unit_id,
            "end_of_turn": event.end_of_turn,
            "metrics": event.metrics,
        }
    elif isinstance(event, SpeechSynthesisChunk):
        audio = waveform_to_pcm16_bytes(event.waveform)
        payload = {
            "type": "audio.chunk",
            "generation_id": event.generation_id,
            "unit_id": event.unit_id,
            "sequence": event.sequence,
            "current_time": event.current_time,
            "end_of_turn": event.end_of_turn,
            "metrics": event.metrics,
        }
    else:
        audio = (
            waveform_to_pcm16_bytes(event.audio_waveform)
            if event.audio_waveform is not None
            else b""
        )
        payload = event.to_payload()
    payload.update(
        audio=bool(audio),
        audio_bytes=len(audio),
        audio_format="pcm16",
        audio_sample_rate=output_sample_rate,
    )
    await websocket.send_text(_json(payload))
    if audio:
        await websocket.send_bytes(audio)


class _WebSocketOutbox:
    """Serialize websocket writes without blocking model ingestion on audio I/O."""

    MODEL_PRIORITY = 0
    AUDIO_PRIORITY = 10

    def __init__(self, websocket: WebSocket, output_sample_rate: int) -> None:
        self.websocket = websocket
        self.output_sample_rate = output_sample_rate
        self.queue: asyncio.PriorityQueue[
            tuple[int, int, float, str, Any, asyncio.Future[None] | None]
        ] = asyncio.PriorityQueue()
        self.sequence = 0
        self.audio_generation_floor = 0
        self.task = asyncio.create_task(
            self._run(), name="gander-duplex-websocket-writer"
        )

    def _raise_if_failed(self) -> None:
        if not self.task.done():
            return
        if self.task.cancelled():
            raise WebSocketDisconnect(code=1006)
        error = self.task.exception()
        if error is not None:
            raise error
        raise WebSocketDisconnect(code=1006)

    async def send_text(
        self,
        payload: dict[str, Any],
        *,
        wait_sent: bool = False,
    ) -> None:
        self._raise_if_failed()
        self.sequence += 1
        sent = asyncio.get_running_loop().create_future() if wait_sent else None
        self.queue.put_nowait(
            (
                self.MODEL_PRIORITY,
                self.sequence,
                time.perf_counter(),
                f"text:{payload.get('type', 'unknown')}",
                payload,
                sent,
            )
        )
        # Queueing alone does not suspend the caller. Yield once so the writer
        # can put latency-sensitive UI/control JSON on the wire before any
        # synchronous coordinator bookkeeping resumes on this event loop.
        await asyncio.sleep(0)
        if sent is not None:
            await sent

    async def send_event(
        self,
        event: Any,
        *,
        audio: bool = False,
        wait_sent: bool = False,
    ) -> None:
        self._raise_if_failed()
        if type(event).__name__ == "PlaybackCancel":
            self.audio_generation_floor = max(
                self.audio_generation_floor,
                int(event.generation_id),
            )
        self.sequence += 1
        sent = asyncio.get_running_loop().create_future() if wait_sent else None
        self.queue.put_nowait(
            (
                self.AUDIO_PRIORITY if audio else self.MODEL_PRIORITY,
                self.sequence,
                time.perf_counter(),
                f"event:{type(event).__name__}",
                event,
                sent,
            )
        )
        await asyncio.sleep(0)
        if sent is not None:
            await sent

    async def join(self) -> None:
        self._raise_if_failed()
        joined = asyncio.create_task(self.queue.join())
        done, _ = await asyncio.wait(
            (joined, self.task), return_when=asyncio.FIRST_COMPLETED
        )
        if self.task in done and not joined.done():
            joined.cancel()
            await asyncio.gather(joined, return_exceptions=True)
            self._raise_if_failed()
        await joined
        self._raise_if_failed()

    def discard_pending(self) -> None:
        """Drop output queued before a conversation reset."""

        while True:
            try:
                _priority, _sequence, _queued_at, _label, _payload, sent = (
                    self.queue.get_nowait()
                )
            except asyncio.QueueEmpty:
                return
            if sent is not None and not sent.done():
                sent.cancel()
            self.queue.task_done()

    async def close(self) -> None:
        if not self.task.done():
            self.task.cancel()
        await asyncio.gather(self.task, return_exceptions=True)
        while True:
            try:
                _priority, _sequence, _queued_at, _label, _payload, sent = (
                    self.queue.get_nowait()
                )
            except asyncio.QueueEmpty:
                break
            if sent is not None and not sent.done():
                sent.cancel()
            self.queue.task_done()

    def _stale_audio(self, event: Any) -> bool:
        return (
            type(event).__name__
            in {
                "SpeechSynthesisChunk",
                "SpeechSynthesisDone",
                "SpeechSynthesisError",
            }
            and int(event.generation_id) < self.audio_generation_floor
        )

    async def _run(self) -> None:
        while True:
            _priority, _sequence, queued_at, label, payload, sent = (
                await self.queue.get()
            )
            started = time.perf_counter()
            try:
                if self._stale_audio(payload):
                    LOGGER.debug(
                        "discarding stale %s generation=%s floor=%s",
                        type(payload).__name__,
                        payload.generation_id,
                        self.audio_generation_floor,
                    )
                elif label.startswith("text:"):
                    await self.websocket.send_text(_json(payload))
                else:
                    await _send_chunk(
                        self.websocket, payload, self.output_sample_rate
                    )
                if sent is not None and not sent.done():
                    sent.set_result(None)
            except RuntimeError as exc:
                error = WebSocketDisconnect(code=1006)
                if sent is not None and not sent.done():
                    sent.set_exception(error)
                raise error from exc
            except asyncio.CancelledError:
                if sent is not None and not sent.done():
                    sent.cancel()
                raise
            except Exception as exc:
                if sent is not None and not sent.done():
                    sent.set_exception(exc)
                raise
            finally:
                self.queue.task_done()
            finished = time.perf_counter()
            queue_ms = (started - queued_at) * 1000
            send_ms = (finished - started) * 1000
            if queue_ms >= 100 or send_ms >= 100:
                LOGGER.info(
                    "websocket outbound %s queue_ms=%.1f send_ms=%.1f pending=%d",
                    label,
                    queue_ms,
                    send_ms,
                    self.queue.qsize(),
                )


async def _drain(
    session: GanderDuplexSession,
    emit_model_event: Callable[[Any], Awaitable[None]],
    *,
    pending_unit_capture_start_ms: float | None = None,
) -> None:
    for event in await asyncio.to_thread(
        session.flush_pending,
        unit_capture_start_ms=pending_unit_capture_start_ms,
    ):
        await emit_model_event(event)

    trailing_steps = 0
    while session.should_continue_draining(trailing_steps):
        event = await asyncio.to_thread(session.step_silence)
        await emit_model_event(event)
        trailing_steps += 1
        if session.should_stop_after(event):
            break
    await asyncio.to_thread(session.wait_for_speech)
    await asyncio.to_thread(session.close, drain_speech=True)


def create_online_duplex_app(
    bundle: Any,
    *,
    params: Any,
    gateway_factory: Callable[[str], Any],
    provider_name: str,
    settings: OnlineDuplexSettings | None = None,
    media_dir: str | Path,
    detached_talker: Any | None = None,
):
    """Create the persistent MiniCPM, task-tools and screen endpoints."""

    from mcpmft.infer.web import STATIC_DIR, request_json

    runtime = _Runtime(
        bundle=bundle,
        params=params,
        settings=settings or OnlineDuplexSettings(),
        gateway_factory=gateway_factory,
        provider_name=provider_name,
        media_dir=Path(media_dir).expanduser().resolve(),
        detached_talker=detached_talker,
    )
    if runtime.params.sliding_window_mode not in {
        "context_memory",
        "context_no_previous",
        "context_slate",
    }:
        raise ValueError(
            "task_tools_v1 requires a bounded live context mode "
            "('context_memory', 'context_slate', or 'context_no_previous')"
        )
    if (
        runtime.params.sliding_window_mode == "context_slate"
        and not runtime.settings.expose_task_slate_to_model
    ):
        raise ValueError("context_slate requires task-slate exposure")
    if (
        runtime.params.sliding_window_mode == "context_no_previous"
        and runtime.settings.expose_task_slate_to_model
    ):
        raise ValueError("context_no_previous cannot expose a task slate")
    if runtime.settings.media_mode not in {"voice", "omni", "auto"}:
        raise ValueError(f"unsupported media_mode: {runtime.settings.media_mode!r}")
    if runtime.settings.client_video_mode not in CLIENT_VIDEO_MODES:
        raise ValueError(
            f"unsupported client_video_mode: {runtime.settings.client_video_mode!r}"
        )
    if not runtime.settings.client_video_sources or not set(
        runtime.settings.client_video_sources
    ) <= set(VIDEO_SOURCES):
        raise ValueError(
            f"client_video_sources must be a non-empty subset of {VIDEO_SOURCES}"
        )
    # Refuse at startup rather than crashing on the model thread the moment a user
    # enables their camera: with init_vision=false the vision tower does not exist.
    if (
        runtime.settings.media_mode != "voice" or runtime.settings.allow_client_video
    ) and not _vision_available(runtime):
        raise ValueError(
            "video input requires model.init_vision=true in the serving configuration"
        )
    if runtime.settings.vision_max_slice_nums <= 0:
        raise ValueError("vision_max_slice_nums must be positive")
    if runtime.settings.max_screen_frame_bytes <= 0:
        raise ValueError("max_screen_frame_bytes must be positive")
    if runtime.settings.max_screen_pixels <= 0:
        raise ValueError("max_screen_pixels must be positive")
    if not 1 <= runtime.settings.codex_frame_rate_multiplier <= 10:
        raise ValueError("codex_frame_rate_multiplier must be between 1 and 10")
    if runtime.settings.codex_screen_history_seconds <= 0:
        raise ValueError("codex_screen_history_seconds must be positive")
    if runtime.params.chunk_ms <= 0:
        raise ValueError("chunk_ms must be positive")
    if runtime.settings.asr_timeout_sec <= 0:
        raise ValueError("asr_timeout_sec must be positive")
    if runtime.settings.turn_bind_grace_sec < 0:
        raise ValueError("turn_bind_grace_sec cannot be negative")
    app = FastAPI(title="Gander Online Duplex", version="1.0.0")

    async def prepare_static_prefix() -> None:
        await asyncio.to_thread(_prepare_static_prefix, runtime)

    app.router.add_event_handler("startup", prepare_static_prefix)

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(
            STATIC_DIR / "index.html",
            media_type="text/html",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/assets/app.css")
    async def app_css() -> FileResponse:
        return FileResponse(
            STATIC_DIR / "app.css",
            media_type="text/css",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/assets/app.js")
    async def app_js() -> FileResponse:
        return FileResponse(
            STATIC_DIR / "app.js",
            media_type="text/javascript",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/assets/video.js")
    async def video_js() -> FileResponse:
        return FileResponse(
            STATIC_DIR / "video.js",
            media_type="text/javascript",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/assets/mic-worklet.js")
    async def mic_worklet_js() -> FileResponse:
        return FileResponse(
            STATIC_DIR / "mic-worklet.js",
            media_type="text/javascript",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "busy": runtime.model_lock.locked(),
            "active_sessions": list(runtime.sessions),
            "session_media_modes": {
                name: active.media_mode
                for name, active in runtime.sessions.items()
            },
            "vision_available": _vision_available(runtime),
            "client_video": _client_video_capabilities(runtime),
            "task_protocol": ONLINE_TASK_PROTOCOL,
            "audio_input_protocol": AUDIO_INPUT_PROTOCOL,
            "input_sample_rate": runtime.settings.input_sample_rate,
            "output_sample_rate": runtime.settings.output_sample_rate,
            "generate_audio": bool(
                runtime.params.generate_audio or runtime.detached_talker is not None
            ),
            "detached_talker": runtime.detached_talker is not None,
            "chunk_ms": runtime.params.chunk_ms,
            "sliding_window_mode": runtime.params.sliding_window_mode,
            "memory_episode_channel": (
                "enabled"
                if runtime.params.sliding_window_mode == "context_memory"
                else "disabled"
            ),
            "context_max_units": runtime.params.context_max_units,
            "context_previous_max_tokens": runtime.params.context_previous_max_tokens,
            "prefix_cache_status": runtime.prefix_cache_status,
            "prefix_cache_tokens": (
                runtime.prefix_snapshot.token_count
                if runtime.prefix_snapshot is not None
                else 0
            ),
            "prefix_prepare_seconds": runtime.prefix_prepare_seconds,
            "first_unit_warmup_seconds": runtime.first_unit_warmup_seconds,
            "task_slate_visible_to_model": (
                runtime.settings.expose_task_slate_to_model
            ),
            "asr_enabled": bool(runtime.settings.asr_base_url),
            "tools": _tool_names(runtime),
        }

    @app.get("/api/asr/health")
    async def asr_health() -> JSONResponse:
        if not runtime.settings.asr_base_url:
            return JSONResponse(
                {"status": "disabled", "message": "ASR upstream is not configured"},
                status_code=503,
            )
        try:
            payload = await asyncio.to_thread(
                request_json,
                f"{runtime.settings.asr_base_url.rstrip('/')}/health",
                timeout=min(runtime.settings.asr_timeout_sec, 2.0),
            )
        except RuntimeError as exc:
            return JSONResponse(
                {"status": "unavailable", "message": str(exc)},
                status_code=502,
            )
        return JSONResponse(payload)

    @app.post("/api/asr/transcribe")
    async def asr_transcribe(request: Request) -> JSONResponse:
        if not runtime.settings.asr_base_url:
            return JSONResponse(
                {"type": "error", "message": "ASR upstream is not configured"},
                status_code=503,
            )
        pcm = await request.body()
        if not pcm or len(pcm) % 2:
            return JSONResponse(
                {"type": "error", "message": "ASR input must be non-empty PCM16"},
                status_code=400,
            )
        max_bytes = runtime.settings.input_sample_rate * 2 * 120
        if len(pcm) > max_bytes:
            return JSONResponse(
                {"type": "error", "message": "ASR input exceeds 120 seconds"},
                status_code=413,
            )
        query = urllib.parse.urlencode(
            {
                "sample_rate": runtime.settings.input_sample_rate,
                "start_ms": request.query_params.get("start_ms", "0"),
                "language": request.query_params.get("language", ""),
            }
        )
        url = f"{runtime.settings.asr_base_url.rstrip('/')}/transcribe?{query}"
        try:
            payload = await asyncio.to_thread(
                request_json,
                url,
                data=pcm,
                headers={"Content-Type": "application/octet-stream"},
                timeout=runtime.settings.asr_timeout_sec,
            )
        except RuntimeError as exc:
            return JSONResponse(
                {"type": "error", "message": str(exc)},
                status_code=502,
            )
        return JSONResponse(payload)

    @app.websocket("/ws/screen")
    async def screen(websocket: WebSocket) -> None:
        await websocket.accept()
        session_id = websocket.query_params.get("session_id") or ""
        token = websocket.query_params.get("token") or ""
        active = runtime.sessions.get(session_id)
        if not _SESSION_ID.fullmatch(session_id) or active is None:
            await websocket.send_text(
                _json({"type": "error", "message": "duplex session is not active"})
            )
            await websocket.close(code=1008)
            return
        if active.media_mode == "voice":
            await websocket.send_text(
                _json({"type": "error", "message": "screen input is disabled in voice mode"})
            )
            await websocket.close(code=1008)
            return
        if not token or not secrets.compare_digest(token, active.screen_token):
            await websocket.send_text(
                _json({"type": "error", "message": "invalid screen session token"})
            )
            await websocket.close(code=1008)
            return

        await websocket.send_text(
            _json(
                {
                    "type": "screen.ready",
                    "session_id": session_id,
                    "media_mode": active.media_mode,
                    "video_source": active.video_source,
                    "max_frame_bytes": runtime.settings.max_screen_frame_bytes,
                    "max_pixels": runtime.settings.max_screen_pixels,
                    "recommended_frame_rate": _unit_frame_rate(runtime),
                    "codex_frame_rate": _codex_frame_rate(runtime),
                    "codex_screen_history_seconds": (
                        runtime.settings.codex_screen_history_seconds
                    ),
                }
            )
        )

        async def receive_while_session_active() -> dict[str, Any] | None:
            receive_task = asyncio.create_task(websocket.receive())
            ended_task = asyncio.create_task(active.ended.wait())
            done, pending = await asyncio.wait(
                (receive_task, ended_task),
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            if ended_task in done:
                if not receive_task.done():
                    receive_task.cancel()
                await asyncio.gather(receive_task, return_exceptions=True)
                try:
                    await websocket.close()
                except RuntimeError:
                    pass
                return None
            return receive_task.result()

        try:
            while runtime.sessions.get(session_id) is active:
                metadata_message = await receive_while_session_active()
                if metadata_message is None:
                    return
                if metadata_message.get("type") == "websocket.disconnect":
                    return
                raw_metadata = metadata_message.get("text")
                if raw_metadata is None:
                    raise ValueError("screen frame metadata must be JSON text")
                payload = json.loads(raw_metadata)
                if not isinstance(payload, dict):
                    raise ValueError("screen frame metadata must be an object")
                header = ScreenFrameHeader.from_payload(payload)

                image_message = await receive_while_session_active()
                if image_message is None:
                    return
                if image_message.get("type") == "websocket.disconnect":
                    return
                image_payload = image_message.get("bytes")
                if image_payload is None:
                    raise ValueError("screen frame image must be binary")
                decoded = await asyncio.to_thread(
                    decode_screen_frame,
                    header,
                    image_payload,
                    max_bytes=runtime.settings.max_screen_frame_bytes,
                    max_pixels=runtime.settings.max_screen_pixels,
                )
                if runtime.sessions.get(session_id) is not active:
                    return
                if active.media_mode == "voice":
                    # The session switched back to audio while this frame was in
                    # flight. Dropping one frame is benign; closing the socket over
                    # a race during a source switch is not.
                    await websocket.send_text(
                        _json(
                            {
                                "type": "screen.frame.dropped",
                                "frame_id": header.frame_id,
                                "reason": "media_mode is voice",
                            }
                        )
                    )
                    continue

                context_sampled = active.codex_frame_gate.accept(
                    header.captured_at_ms
                )
                # The ACK means the frame is available to the frontbrain, so publish
                # it before acknowledging. Shared-storage persistence remains off the
                # event loop and must not delay the browser's initial handshake.
                active.duplex.enqueue_screen_frame(decoded.frame)
                await websocket.send_text(
                    _json(
                        {
                            "type": "screen.frame.accepted",
                            "frame_id": header.frame_id,
                            "captured_at_ms": header.captured_at_ms,
                            "width": decoded.width,
                            "height": decoded.height,
                            "asset_id": decoded.asset_id,
                            "context_sampled": context_sampled,
                        }
                    )
                )
                if context_sampled:
                    # A webcam frame is not the user's screen; tagging it keeps the
                    # backbrain from narrating one as the other.
                    source = header.video_source or active.video_source
                    media = await _screen_media_ref(
                        runtime,
                        session_id,
                        header,
                        image_payload,
                        decoded,
                        kind="frame" if source == "camera" else "screen",
                    )
                    if runtime.sessions.get(session_id) is not active:
                        return
                    coordinator = active.coordinator
                    if coordinator is not None:
                        coordinator.remember_media(media)
        except WebSocketDisconnect:
            return
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            try:
                await websocket.send_text(
                    _json({"type": "error", "message": str(exc)})
                )
                await websocket.close(code=1003)
            except RuntimeError:
                pass

    @app.websocket("/ws/duplex")
    async def duplex(websocket: WebSocket) -> None:
        await websocket.accept()
        requested_id = websocket.query_params.get("session_id")
        session_id = requested_id or f"duplex_{secrets.token_hex(8)}"
        if not _SESSION_ID.fullmatch(session_id):
            await websocket.send_text(_json({"type": "error", "message": "invalid session_id"}))
            await websocket.close(code=1008)
            return
        if runtime.model_lock.locked():
            await websocket.send_text(
                _json({"type": "error", "message": "model is busy with another session"})
            )
            await websocket.close(code=1013)
            return

        await runtime.model_lock.acquire()
        session: GanderDuplexSession | None = None
        coordinator: Any | None = None
        active: _ActiveSession | None = None
        outbound_task: asyncio.Task[None] | None = None
        speech_output_task: asyncio.Task[None] | None = None
        speech_output_stop: asyncio.Event | None = None
        pending_audio_header: AudioFrameHeader | None = None
        audio_timeline = AudioCaptureTimeline(
            sample_rate=runtime.settings.input_sample_rate,
            unit_ms=runtime.params.chunk_ms,
        )
        websocket_outbox = _WebSocketOutbox(
            websocket, runtime.settings.output_sample_rate
        )

        async def send_text(
            payload: dict[str, Any],
            *,
            wait_sent: bool = False,
        ) -> None:
            await websocket_outbox.send_text(payload, wait_sent=wait_sent)

        async def send_model_event(
            event: Any,
            *,
            wait_sent: bool = False,
        ) -> None:
            await websocket_outbox.send_event(event, wait_sent=wait_sent)

        def start_speech_output_pump(
            target: GanderDuplexSession,
        ) -> tuple[asyncio.Event, asyncio.Task[None]]:
            stop = asyncio.Event()

            async def pump() -> None:
                while True:
                    event = await asyncio.to_thread(target.poll_output, 0.05)
                    if event is not None:
                        await websocket_outbox.send_event(
                            event,
                            audio=type(event).__name__
                            in {"SpeechSynthesisChunk", "SpeechSynthesisDone"},
                        )
                        continue
                    if stop.is_set():
                        for remaining in target.drain_outputs():
                            await websocket_outbox.send_event(
                                remaining,
                                audio=(
                                    type(remaining).__name__
                                    in {
                                        "SpeechSynthesisChunk",
                                        "SpeechSynthesisDone",
                                    }
                                ),
                            )
                        return

            return stop, asyncio.create_task(
                pump(), name=f"gander-detached-talker-output-{session_id}"
            )

        async def stop_speech_output_pump() -> None:
            nonlocal speech_output_task, speech_output_stop
            if speech_output_stop is not None:
                speech_output_stop.set()
            if speech_output_task is not None:
                try:
                    await speech_output_task
                except (WebSocketDisconnect, RuntimeError):
                    pass
            speech_output_task = None
            speech_output_stop = None

        async def emit_model_event(event: Any) -> None:
            assert coordinator is not None
            output = coordinator.model_output(event)
            await send_model_event(
                event,
                wait_sent=output.delivery_id is not None,
            )
            if output.delivery_id is not None:
                coordinator.acknowledge_output(output)
            coordinator.observe_frontbrain(event)

        async def build_coordinator(
            model_session: GanderDuplexSession,
        ) -> TaskToolsRealtimeCoordinator:
            gateway = runtime.gateway_factory(session_id)
            task_coordinator = TaskToolsRealtimeCoordinator(
                gateway,
                owner_id=session_id,
                session_id=session_id,
                session=model_session,
                provider_name=runtime.provider_name,
                media_dir=runtime.media_dir,
                input_sample_rate=runtime.settings.input_sample_rate,
                max_recent_media=_max_recent_screen_frames(runtime),
                turn_bind_grace_sec=runtime.settings.turn_bind_grace_sec,
                expose_task_slate_to_model=(
                    runtime.settings.expose_task_slate_to_model
                ),
                close_ledger=True,
            )
            await task_coordinator.start()
            await _warm_backbrain_providers(gateway)
            return task_coordinator

        try:
            session = await _open_session(runtime)
            if runtime.detached_talker is not None:
                speech_output_stop, speech_output_task = start_speech_output_pump(
                    session
                )
            active = _ActiveSession(
                duplex=session,
                screen_token=secrets.token_urlsafe(32),
                codex_frame_gate=ScreenFrameRateGate(
                    _codex_frame_interval_ms(runtime)
                ),
                media_mode=runtime.settings.media_mode,
            )
            runtime.sessions[session_id] = active
            coordinator = await build_coordinator(session)
            active.coordinator = coordinator

            async def forward_tool_outputs() -> None:
                assert coordinator is not None
                while True:
                    output = await coordinator.next_output()
                    if output.kind == "control":
                        await send_text(
                            output.value,
                            wait_sent=output.delivery_id is not None,
                        )
                    else:
                        await send_model_event(
                            output.value,
                            wait_sent=output.delivery_id is not None,
                        )
                    if output.delivery_id is not None:
                        coordinator.acknowledge_output(output)

            outbound_task = asyncio.create_task(
                forward_tool_outputs(),
                name=f"gander-native-tool-output-{session_id}",
            )
            outbound_task.add_done_callback(
                lambda task: (
                    None
                    if task.cancelled()
                    else LOGGER.error(
                        "native tool output loop failed for %s: %s",
                        session_id,
                        task.exception(),
                    )
                    if task.exception() is not None
                    else None
                )
            )
            screen_enabled = (
                active.media_mode != "voice" or _client_video_allowed(runtime)
            )
            await send_text(
                {
                    "type": "ready",
                    "session_id": session_id,
                    "input_sample_rate": runtime.settings.input_sample_rate,
                    "output_sample_rate": runtime.settings.output_sample_rate,
                    "chunk_ms": runtime.params.chunk_ms,
                    "audio_input": {
                        "protocol": AUDIO_INPUT_PROTOCOL,
                        "encoding": "pcm_s16le",
                        "clock": "unix_ms",
                    },
                    "generate_audio": bool(
                        runtime.params.generate_audio
                        or runtime.detached_talker is not None
                    ),
                    "detached_talker": runtime.detached_talker is not None,
                    "generation_id": int(
                        session.talker_state().get("generation_id", 0)
                    ),
                    "sliding_window_mode": runtime.params.sliding_window_mode,
                    "context_max_units": runtime.params.context_max_units,
                    "context_previous_max_tokens": (
                        runtime.params.context_previous_max_tokens
                    ),
                    "transport": "ws",
                    "decode_mode": runtime.settings.decode_mode,
                    "media_mode": active.media_mode,
                    "video_source": active.video_source,
                    "tool_protocol": "native_complete_call_v1",
                    "task_protocol": ONLINE_TASK_PROTOCOL,
                    "turn_bind_grace_ms": round(
                        runtime.settings.turn_bind_grace_sec * 1000
                    ),
                    "tools": _tool_names(runtime),
                    "context_events": [
                        "turn.final",
                        *(
                            ["memory.episode"]
                            if runtime.params.sliding_window_mode == "context_memory"
                            else []
                        ),
                        "task_status",
                        "screen",
                    ],
                    "screen": {
                        "enabled": screen_enabled,
                        "path": "/ws/screen" if screen_enabled else None,
                        "token": active.screen_token if screen_enabled else None,
                        "protocol": "metadata-json+encoded-binary-v1",
                        "encodings": ["jpeg", "webp", "png"],
                        "max_frame_bytes": runtime.settings.max_screen_frame_bytes,
                        "max_pixels": runtime.settings.max_screen_pixels,
                        "vision_max_slice_nums": runtime.settings.vision_max_slice_nums,
                        "vision_batch_feed": runtime.settings.vision_batch_feed,
                        "recommended_frame_rate": _unit_frame_rate(runtime),
                        "codex_frame_rate": _codex_frame_rate(runtime),
                        "codex_screen_history_seconds": (
                            runtime.settings.codex_screen_history_seconds
                        ),
                        "client_video": _client_video_capabilities(runtime),
                    },
                }
            )

            while True:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    return
                if message.get("bytes") is not None:
                    audio = message["bytes"]
                    await asyncio.to_thread(coordinator.record_pcm16, audio)
                    # Bound each model call to one coordination unit. This gives provider
                    # tasks an event-loop opportunity between units even if a client batches PCM.
                    chunk_bytes = int(
                        runtime.settings.input_sample_rate * runtime.params.chunk_ms / 1000 * 2
                    )
                    if pending_audio_header is None:
                        parts = tuple(
                            (
                                audio[offset : offset + chunk_bytes],
                                None,
                            )
                            for offset in range(0, len(audio), chunk_bytes)
                        )
                    else:
                        timed_parts = audio_timeline.split_frame(
                            pending_audio_header,
                            audio,
                            max_part_bytes=chunk_bytes,
                        )
                        pending_audio_header = None
                        parts = tuple(
                            (part.data, part.unit_capture_start_ms)
                            for part in timed_parts
                        )
                    for part, capture_starts in parts:
                        events = await asyncio.to_thread(
                            session.feed_pcm16,
                            part,
                            unit_capture_start_ms=capture_starts,
                        )
                        for event in events:
                            await emit_model_event(event)
                    continue

                raw = message.get("text")
                if raw is None:
                    continue
                try:
                    control = json.loads(raw)
                except json.JSONDecodeError:
                    await send_text(
                        {"type": "error", "message": "invalid json control event"}
                    )
                    continue
                if not isinstance(control, dict):
                    await send_text(
                        {"type": "error", "message": "control event must be an object"}
                    )
                    continue

                event_type = control.get("type")
                if event_type == "audio.frame":
                    if pending_audio_header is not None:
                        await send_text(
                            {
                                "type": "error",
                                "message": (
                                    "audio frame metadata requires a following "
                                    "binary payload"
                                ),
                            }
                        )
                        continue
                    try:
                        pending_audio_header = AudioFrameHeader.from_payload(control)
                    except ValueError as exc:
                        await send_text({"type": "error", "message": str(exc)})
                elif event_type == "ping":
                    await send_text({"type": "pong", "id": control.get("id")})
                elif event_type == "stop":
                    await _drain(
                        session,
                        emit_model_event,
                        pending_unit_capture_start_ms=(
                            audio_timeline.pending_unit_capture_start_ms
                        ),
                    )
                    await stop_speech_output_pump()
                    await websocket_outbox.join()
                    await send_text({"type": "session.done"})
                    await websocket_outbox.join()
                    await websocket.close()
                    return
                elif event_type == "reset":
                    pending_audio_header = None
                    audio_timeline.reset()
                    if runtime.detached_talker is not None:
                        # Reset discards the old conversation. Cancel its Talker
                        # generation before waiting for the output pump, otherwise
                        # a long synthesis can hold the only model slot after the
                        # browser has already stopped playing it.
                        await asyncio.to_thread(session.interrupt_output)
                    await stop_speech_output_pump()
                    if outbound_task is not None:
                        outbound_task.cancel()
                        await asyncio.gather(outbound_task, return_exceptions=True)
                        outbound_task = None
                    websocket_outbox.discard_pending()
                    assert active is not None
                    screen_frames = active.duplex.screen_frames
                    # Reset is a new conversation on the same transport. Keep the
                    # selected video source, but never let the new model session
                    # reuse a frame captured for the conversation being discarded.
                    screen_frames.reset()
                    active.coordinator = None
                    await coordinator.close(discard_state=True)
                    coordinator = None
                    await asyncio.to_thread(session.close)
                    session = await _open_session(
                        runtime,
                        media_mode=active.media_mode,
                        screen_frames=screen_frames,
                    )
                    if runtime.detached_talker is not None:
                        speech_output_stop, speech_output_task = (
                            start_speech_output_pump(session)
                        )
                    coordinator = await build_coordinator(session)
                    active.duplex = session
                    active.coordinator = coordinator
                    outbound_task = asyncio.create_task(
                        forward_tool_outputs(),
                        name=f"gander-native-tool-output-{session_id}",
                    )
                    await send_text(
                        {
                            "type": "reset.done",
                            "generation_id": int(
                                session.talker_state().get("generation_id", 0)
                            ),
                            "media_mode": active.media_mode,
                            "video_source": active.video_source,
                        }
                    )
                elif event_type == "media.mode":
                    assert active is not None
                    want_video = bool(control.get("video"))
                    source = control.get("source") if want_video else None
                    reason = _reject_media_mode(
                        runtime, want_video=want_video, source=source
                    )
                    if reason is not None:
                        # Deliberately not an `error`: browser clients treat that as
                        # fatal, and a refused mode request must not end the session.
                        await send_text(
                            {"type": "media.mode.rejected", "reason": reason}
                        )
                        continue
                    target = resolve_target_mode(
                        want_video=want_video,
                        client_video_mode=runtime.settings.client_video_mode,
                    )
                    previous = await asyncio.to_thread(
                        active.duplex.set_media_mode, target
                    )
                    active.media_mode = target
                    active.video_source = source
                    warnings = source_warnings(source)
                    if warnings:
                        LOGGER.warning(
                            "session %s enabled %s video: %s",
                            session_id,
                            source,
                            ", ".join(warnings),
                        )
                    await send_text(
                        {
                            "type": "media.mode.done",
                            "video": want_video,
                            "source": source,
                            "media_mode": target,
                            "previous_media_mode": previous,
                            "estimated_tokens_per_unit": estimated_tokens_per_unit(
                                media_mode=target,
                                speak_text_tokens_per_unit=(
                                    runtime.params.speak_text_tokens_per_unit
                                ),
                            ),
                            "warnings": list(warnings),
                            "screen": (
                                _screen_channel(runtime, active)
                                if want_video
                                else None
                            ),
                        }
                    )
                elif event_type == "break":
                    await asyncio.to_thread(session.set_break)
                    await send_text({"type": "break.done"})
                elif event_type == "clear_break":
                    await asyncio.to_thread(session.clear_break)
                    await send_text({"type": "clear_break.done"})
                elif event_type == "tool.response":
                    try:
                        response_event = await coordinator.inject_external_tool_response(
                            control.get("content")
                        )
                    except (RuntimeError, ValueError) as exc:
                        await send_text({"type": "error", "message": str(exc)})
                        continue
                    if response_event is None:
                        await send_text({"type": "tool.response.queued"})
                    else:
                        await emit_model_event(response_event)
                elif event_type == "turn.final":
                    try:
                        turn = coordinator.bind_final_turn(control)
                    except (TypeError, ValueError) as exc:
                        await send_text({"type": "error", "message": str(exc)})
                        continue
                    await send_text(
                        {
                            "type": "turn.final.accepted",
                            "turn_id": turn.turn_id,
                            "context_revision": turn.context_revision,
                        }
                    )
                elif event_type == "memory.episode":
                    if runtime.params.sliding_window_mode != "context_memory":
                        await send_text(
                            {
                                "type": "error",
                                "message": "memory.episode is disabled for this context mode",
                            }
                        )
                        continue
                    if "entry" not in control:
                        await send_text(
                            {
                                "type": "error",
                                "message": "memory.episode requires entry",
                            }
                        )
                        continue
                    try:
                        changed = await coordinator.inject_memory_episode(
                            control["entry"]
                        )
                    except (TypeError, ValueError, RuntimeError) as exc:
                        await send_text({"type": "error", "message": str(exc)})
                        continue
                    await send_text(
                        {
                            "type": "memory.episode.done",
                            "context_changed": changed,
                        }
                    )
                elif event_type == "context":
                    await send_text(
                        {
                            "type": "error",
                            "message": (
                                "task_tools_v1 binds user text only through turn.final"
                            ),
                        }
                    )
                elif event_type == "task_status":
                    status = coordinator.task_status()
                    await send_text({"type": "task_status", "task": status})
                else:
                    await send_text(
                        {
                            "type": "error",
                            "message": f"unsupported control event: {event_type}",
                        }
                    )
        except WebSocketDisconnect:
            LOGGER.info("duplex websocket disconnected: %s", session_id)
        except Exception as exc:
            LOGGER.exception("duplex websocket failed: %s", session_id)
            try:
                await send_text({"type": "error", "message": str(exc)})
                await websocket_outbox.join()
            except (RuntimeError, WebSocketDisconnect):
                pass
        finally:
            if (
                runtime.detached_talker is not None
                and session is not None
                and not session.closed
            ):
                try:
                    await asyncio.to_thread(session.interrupt_output)
                except RuntimeError:
                    pass
            await stop_speech_output_pump()
            if outbound_task is not None:
                outbound_task.cancel()
                await asyncio.gather(outbound_task, return_exceptions=True)
            await websocket_outbox.close()
            if active is not None:
                if runtime.sessions.get(session_id) is active:
                    runtime.sessions.pop(session_id, None)
                active.ended.set()
            if coordinator is not None:
                await coordinator.close()
            if session is not None and not session.closed:
                # All model-owning coordinator jobs have stopped at this point;
                # close is a short lifecycle operation and must finish even
                # after the loop's executor-backed work has drained.
                session.close()
            runtime.model_lock.release()

    return app
