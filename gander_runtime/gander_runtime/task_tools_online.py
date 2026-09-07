from __future__ import annotations

import asyncio
import dataclasses
import logging
import os
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

from .contracts import (
    EPHEMERAL_VISUAL_MEDIA_KINDS,
    ContextEvent,
    MediaRef,
    now_ms,
    stable_id,
    storage_key,
)
from .coordination import TurnEnvelope
from .lean_realtime import TaskToolHandler, worker_delivery_response

LOGGER = logging.getLogger(__name__)
_TASK_TOOLS = frozenset({"task_start", "task_send", "task_resolve"})
MAX_TASK_TOOL_CALLS_PER_UNIT = 4


@dataclass(frozen=True)
class TaskToolsOnlineOutput:
    kind: Literal["model", "control"]
    value: Any
    delivery_id: str | None = None
    claim_token: str | None = None
    delivery_attempt: int | None = None

    def __post_init__(self) -> None:
        claim_fields = (self.delivery_id, self.claim_token, self.delivery_attempt)
        if any(value is None for value in claim_fields) != all(
            value is None for value in claim_fields
        ):
            raise ValueError(
                "delivery_id, claim_token and delivery_attempt must be provided together"
            )
        if self.delivery_attempt is not None and (
            not isinstance(self.delivery_attempt, int)
            or isinstance(self.delivery_attempt, bool)
            or self.delivery_attempt < 1
        ):
            raise ValueError("delivery_attempt must be a positive integer")


class TaskToolsRealtimeCoordinator:
    """Bind native task tools to one realtime Gateway owner.

    User text is accepted only through an explicit ``turn.final`` transport
    event. The model's task-tool arguments never carry or rewrite that text.
    Synchronous control receipts consume the pending tool-response slot, while
    later worker deliveries enter through a separate Runtime input unit.
    """

    def __init__(
        self,
        gateway: Any,
        owner_id: str,
        session_id: str,
        session: Any,
        *,
        provider_name: str | None = None,
        media_dir: str | Path | None = None,
        input_sample_rate: int = 16000,
        delivery_poll_sec: float = 0.02,
        max_recent_media: int = 16,
        max_turn_media: int = 32,
        max_turn_text_chars: int = 32_768,
        turn_bind_grace_sec: float = 5.0,
        expose_task_slate_to_model: bool = False,
        close_ledger: bool = False,
    ) -> None:
        if delivery_poll_sec <= 0:
            raise ValueError("delivery_poll_sec must be positive")
        if max_recent_media < 1:
            raise ValueError("max_recent_media must be positive")
        if max_turn_media < 1:
            raise ValueError("max_turn_media must be positive")
        if max_turn_text_chars < 1:
            raise ValueError("max_turn_text_chars must be positive")
        if turn_bind_grace_sec < 0:
            raise ValueError("turn_bind_grace_sec cannot be negative")
        self.gateway = gateway
        self.owner_id = owner_id
        self.session_id = session_id
        self.session = session
        self.task_tools = TaskToolHandler(
            gateway,
            owner_id,
            provider_name=provider_name,
        )
        self.input_sample_rate = input_sample_rate
        self.delivery_poll_sec = delivery_poll_sec
        self.max_turn_media = max_turn_media
        self.max_turn_text_chars = max_turn_text_chars
        self.turn_bind_grace_sec = turn_bind_grace_sec
        self.expose_task_slate_to_model = bool(expose_task_slate_to_model)
        self.close_ledger = close_ledger
        self._outputs: asyncio.Queue[TaskToolsOnlineOutput] = asyncio.Queue()
        self._jobs: set[asyncio.Task[Any]] = set()
        self._delivery_task: asyncio.Task[None] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._latest_turn: TurnEnvelope | None = None
        self._turns: dict[str, TurnEnvelope] = {}
        self._turn_identities: dict[str, tuple[Any, ...]] = {}
        self._next_context_revision = 0
        self._context_seq = max(
            (
                event.seq
                for event in gateway.ledger.list_realtime_context(owner_id)
                if event.session_id == session_id
            ),
            default=0,
        )
        self._consumed_turns: set[str] = set()
        self._control_pending = False
        self._pending_task_calls: tuple[dict[str, Any], ...] | None = None
        self._pending_task_nonce = 0
        self._external_tool_pending: str | None = None
        self._delivery_inflight = False
        self._delivery_outputs_pending: set[tuple[str, str, int]] = set()
        self._recovering_tool_error = False
        self._last_slate: str | None = None
        self._recent_media: deque[MediaRef] = deque(maxlen=max_recent_media)
        self._state_lock = threading.RLock()
        self._audio_lock = threading.Lock()
        self._closed = False
        self._audio_fd: int | None = None
        self._audio_path: Path | None = None
        if media_dir is not None:
            root = Path(media_dir).expanduser().resolve()
            root.mkdir(parents=True, exist_ok=True)
            self._audio_path = root / f"{storage_key(session_id)}.input.pcm"

    async def start(self) -> None:
        self._ensure_open()
        self._loop = asyncio.get_running_loop()
        await self.gateway.start()
        # This is a lightweight lifecycle setter, not a model invocation. Keep
        # it on the event-loop thread so startup does not depend on executor
        # wake-up latency before the delivery loop exists.
        self.session.set_summary_needed_callback(self._on_summary_needed)
        await self._sync_slate(force=True)
        self._delivery_task = asyncio.create_task(
            self._delivery_loop(),
            name=f"gander-task-tools-delivery-{self.session_id}",
        )
        self._delivery_task.add_done_callback(self._log_job_failure)

    def bind_final_turn(self, payload: Mapping[str, Any]) -> TurnEnvelope:
        """Bind one transport-authenticated final ASR/text event exactly once."""

        self._ensure_open()
        if not isinstance(payload, Mapping):
            raise ValueError("turn.final payload must be an object")
        allowed = {
            "type",
            "turn_id",
            "text",
            "final_asr",
            "media",
            "environment_ref",
            "context_revision",
            "timezone",
            "start_ms",
            "end_ms",
            "timestamp_ms",
        }
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError(f"turn.final has unexpected fields: {sorted(unknown)}")
        if payload.get("type") not in {None, "turn.final"}:
            raise ValueError("turn.final type must be 'turn.final'")
        turn_id_value = payload.get("turn_id")
        if not isinstance(turn_id_value, str):
            raise ValueError("turn.final turn_id must be a string")
        turn_id = turn_id_value.strip()
        text_value = payload.get("text")
        final_asr_value = payload.get("final_asr")
        if text_value is not None and not isinstance(text_value, str):
            raise ValueError("turn.final text must be a string")
        if final_asr_value is not None and not isinstance(final_asr_value, str):
            raise ValueError("turn.final final_asr must be a string")
        if (
            text_value is not None
            and final_asr_value is not None
            and text_value.strip() != final_asr_value.strip()
        ):
            raise ValueError("turn.final text and final_asr disagree")
        final_asr = (final_asr_value or text_value or "").strip()
        if not turn_id or not final_asr:
            raise ValueError("turn.final requires non-empty turn_id and text")
        if len(final_asr) > self.max_turn_text_chars:
            raise ValueError(
                "turn.final text exceeds "
                f"max_turn_text_chars={self.max_turn_text_chars}"
            )
        environment_value = payload.get("environment_ref")
        timezone_value = payload.get("timezone")
        if environment_value is not None and not isinstance(environment_value, str):
            raise ValueError("turn.final environment_ref must be a string")
        if timezone_value is not None and not isinstance(timezone_value, str):
            raise ValueError("turn.final timezone must be a string")
        environment_ref = environment_value or ""
        timezone = timezone_value or "UTC"
        media_value = payload.get("media")
        explicit_media = _parse_media(() if media_value is None else media_value)
        if len(explicit_media) > self.max_turn_media:
            raise ValueError(
                f"turn.final media exceeds max_turn_media={self.max_turn_media}"
            )
        supplied_revision = payload.get("context_revision")
        if supplied_revision is not None and (
            not isinstance(supplied_revision, int)
            or isinstance(supplied_revision, bool)
        ):
            raise ValueError("turn.final context_revision must be an integer")
        start_ms = _optional_int(payload, "start_ms")
        end_ms = _optional_int(payload, "end_ms")
        timestamp_ms = _optional_int(payload, "timestamp_ms")
        if (start_ms is None) != (end_ms is None):
            raise ValueError("turn.final start_ms and end_ms must be provided together")
        with self._state_lock:
            existing = self._turns.get(turn_id)
            if existing is not None:
                replay_revision = (
                    existing.context_revision
                    if supplied_revision is None
                    else int(supplied_revision)
                )
                identity = (
                    final_asr,
                    explicit_media,
                    environment_ref,
                    timezone,
                    replay_revision,
                    start_ms,
                    end_ms,
                    timestamp_ms,
                )
                if self._turn_identities.get(turn_id) != identity:
                    raise ValueError(f"turn.final identity collision: {turn_id}")
                return existing
            revision = (
                self._next_context_revision + 1
                if supplied_revision is None
                else int(supplied_revision)
            )
            if revision < self._next_context_revision:
                raise ValueError("turn.final context_revision moved backwards")
            media = list(explicit_media)
            reference_ms = timestamp_ms or now_ms()
            screen_window_ms = int(
                getattr(self.gateway, "screen_context_window_ms", 8_000)
            )
            media.extend(
                item
                for item in self._recent_media
                if item.kind not in EPHEMERAL_VISUAL_MEDIA_KINDS
                or item.timestamp_ms is None
                or item.timestamp_ms >= reference_ms - screen_window_ms
            )
            if self._audio_path is not None and self._audio_path.exists():
                media.append(
                    MediaRef(
                        kind="audio",
                        path=str(self._audio_path),
                        mime_type="audio/L16",
                        metadata={
                            "sample_rate": self.input_sample_rate,
                            "channels": 1,
                            "sample_width_bytes": 2,
                            "append_only_live_stream": True,
                        },
                    )
                )
            turn = TurnEnvelope(
                owner_id=self.owner_id,
                voice_session_id=self.session_id,
                turn_id=turn_id,
                final_asr=final_asr,
                media_refs=_dedupe_media(media),
                environment_ref=environment_ref,
                context_revision=revision,
                timezone=timezone,
                start_ms=start_ms,
                end_ms=end_ms,
                timestamp_ms=timestamp_ms,
            )
            context_event = ContextEvent(
                session_id=self.session_id,
                seq=self._next_context_seq(),
                role="user",
                kind="audio_transcript",
                text=final_asr,
                # Never freeze a reference to the append-only live PCM file:
                # its bytes would keep changing after task_start. ASR spans are
                # immutable; sampled screen/image files are immutable as well.
                media=tuple(
                    item
                    for item in turn.media_refs
                    if not (
                        item.kind == "audio"
                        and item.metadata.get("append_only_live_stream") is True
                    )
                ),
                start_ms=start_ms,
                end_ms=end_ms,
                timestamp_ms=timestamp_ms or turn.created_at_ms,
                event_id=stable_id("ctx_turn", turn.receipt_key),
            )
            self.gateway.record_realtime_context(
                owner_id=self.owner_id, event=context_event
            )
            self._turns[turn_id] = turn
            self._turn_identities[turn_id] = (
                final_asr,
                explicit_media,
                environment_ref,
                timezone,
                revision,
                start_ms,
                end_ms,
                timestamp_ms,
            )
            self._latest_turn = turn
            self._next_context_revision = revision
            self._recent_media.clear()
            pending_calls = self._pending_task_calls
            if pending_calls is not None:
                self._pending_task_calls = None
                self._pending_task_nonce += 1
                self._spawn(
                    self._handle_task_tools(pending_calls, turn),
                    "late-bound-task-tool-batch",
                )
            return turn

    def remember_media(self, media: MediaRef) -> None:
        self._ensure_open()
        if not isinstance(media, MediaRef):
            raise TypeError("remember_media requires MediaRef")
        with self._state_lock:
            self._recent_media.append(media)
            self._record_realtime_context(
                role="user",
                kind=_media_context_kind(media.kind),
                media=(media,),
                timestamp_ms=media.timestamp_ms or now_ms(),
            )

    def record_pcm16(self, data: bytes) -> None:
        self._ensure_open()
        if not data or self._audio_path is None:
            return
        if len(data) % 2:
            raise ValueError("pcm16 audio byte length must be even")
        with self._audio_lock:
            if self._audio_fd is None:
                self._audio_fd = os.open(
                    self._audio_path,
                    os.O_APPEND | os.O_CREAT | os.O_WRONLY,
                    0o600,
                )
            view = memoryview(data)
            while view:
                written = os.write(self._audio_fd, view)
                if written <= 0:
                    raise OSError("short write to realtime PCM journal")
                view = view[written:]

    def observe_frontbrain(self, event: Any) -> None:
        """Dispatch a completed native call without blocking the model thread."""

        self._ensure_open()
        if not event.is_tool_call:
            text = event.text
            if (
                isinstance(text, str)
                and text.strip()
                and not event.is_listen
            ):
                with self._state_lock:
                    self._record_realtime_context(
                        role="frontbrain",
                        kind="frontbrain_reply",
                        text=text.strip(),
                    )
            if (
                event.end_of_turn
                and not event.is_listen
            ):
                with self._state_lock:
                    if self._latest_turn is not None:
                        self._consumed_turns.add(self._latest_turn.receipt_key)
            return
        error = event.tool_error
        calls = list(event.tool_calls)
        response_expected = event.tool_response_expected
        if error or not calls or len(calls) > MAX_TASK_TOOL_CALLS_PER_UNIT:
            detail = str(
                error
                or (
                    "expected 1.."
                    f"{MAX_TASK_TOOL_CALLS_PER_UNIT} tool calls, received {len(calls)}"
                )
            )
            if response_expected:
                self._spawn(self._deliver_tool_error(detail), "tool-error")
            else:
                self._emit_control(
                    {"type": "tool.error", "message": detail[:512]}
                )
            return

        names = tuple(str(call.get("name") or "") for call in calls)
        if any(name in _TASK_TOOLS for name in names):
            if not all(name in _TASK_TOOLS for name in names):
                self._spawn(
                    self._deliver_tool_error(
                        "task tools cannot share a unit with external tools"
                    ),
                    "mixed-tool-batch-error",
                )
                return
            try:
                normalized_calls = tuple(
                    self.task_tools.validate_native_tool_call(call) for call in calls
                )
            except (TypeError, ValueError) as exc:
                self._spawn(
                    self._deliver_tool_error(str(exc)),
                    "task-tool-validation-error",
                )
                return
            if (
                len(normalized_calls) > 1
                and self.gateway.mode != "lean"
            ):
                self._spawn(
                    self._deliver_tool_error(
                        "multi-call task units currently require lean gateway mode"
                    ),
                    "coordinator-tool-batch-error",
                )
                return
            with self._state_lock:
                turn = self._latest_turn
                if self._control_pending:
                    detail = "a task tool response is still pending"
                    buffered_nonce = None
                elif turn is None or turn.receipt_key in self._consumed_turns:
                    if self.turn_bind_grace_sec == 0:
                        detail = (
                            "task tools require a trusted turn.final event"
                            if turn is None
                            else "the current final turn already consumed its task action"
                        )
                        buffered_nonce = None
                    else:
                        detail = ""
                        self._control_pending = True
                        self._pending_task_calls = normalized_calls
                        self._pending_task_nonce += 1
                        buffered_nonce = self._pending_task_nonce
                else:
                    detail = ""
                    self._control_pending = True
                    buffered_nonce = None
            if detail:
                self._spawn(self._deliver_tool_error(detail), "task-tool-state-error")
                return
            if buffered_nonce is not None:
                self._emit_control(
                    {
                        "type": "turn.final.required",
                        "grace_ms": round(self.turn_bind_grace_sec * 1000),
                    }
                )
                self._spawn(
                    self._expire_pending_task_call(buffered_nonce),
                    "turn-final-timeout",
                )
                return
            assert turn is not None
            self._spawn(
                self._handle_task_tools(normalized_calls, turn),
                "task-tool-batch",
            )
            return

        if len(calls) != 1:
            self._spawn(
                self._deliver_tool_error(
                    "multiple external tool calls are not supported by this transport"
                ),
                "external-tool-batch-error",
            )
            return
        call = calls[0]
        name = names[0]
        arguments = call.get("arguments")
        if not isinstance(arguments, dict):
            self._spawn(
                self._deliver_tool_error("tool arguments must be an object"),
                "tool-arguments-error",
            )
            return
        with self._state_lock:
            if self._external_tool_pending is not None:
                detail = f"tool {self._external_tool_pending!r} is still pending"
            else:
                detail = ""
                self._external_tool_pending = name
        if detail:
            self._spawn(self._deliver_tool_error(detail), "external-tool-overlap")

    def _next_context_seq(self) -> int:
        self._context_seq += 1
        return self._context_seq

    def _record_realtime_context(
        self,
        *,
        role: Literal["user", "frontbrain", "system", "tool"],
        kind: str,
        text: str = "",
        media: tuple[MediaRef, ...] = (),
        timestamp_ms: int | None = None,
    ) -> ContextEvent:
        event = ContextEvent(
            session_id=self.session_id,
            seq=self._next_context_seq(),
            role=role,
            kind=kind,
            text=text,
            media=media,
            timestamp_ms=timestamp_ms or now_ms(),
        )
        self.gateway.record_realtime_context(owner_id=self.owner_id, event=event)
        return event

    async def inject_external_tool_response(self, response: Any) -> Any | None:
        self._ensure_open()
        with self._state_lock:
            name = self._external_tool_pending
        if name is None:
            raise RuntimeError("there is no pending external tool call")
        event = await self._call_session("feed_tool_response", response)
        with self._state_lock:
            if self._external_tool_pending == name:
                self._external_tool_pending = None
        if event is not None:
            self.observe_frontbrain(event)
        return event

    async def inject_memory_episode(self, episode: Any) -> bool:
        self._ensure_open()
        return bool(await self._call_session("feed_memory_episode", episode))

    def task_status(self) -> dict[str, Any]:
        entries = self.gateway.task_slate(self.owner_id)
        return {
            "slate": self.task_tools.system_prompt_slate(),
            "tasks": [dataclasses.asdict(entry) for entry in entries],
        }

    async def next_output(self) -> TaskToolsOnlineOutput:
        self._ensure_open()
        return await self._outputs.get()

    def model_output(self, event: Any) -> TaskToolsOnlineOutput:
        """Wrap one generated model event and preserve a consumed Delivery claim."""

        self._ensure_open()
        receipt = self.session.take_native_input_receipt(event)
        if not isinstance(receipt, Mapping) or receipt.get("kind") != "worker_delivery":
            return TaskToolsOnlineOutput("model", event)
        delivery_id = receipt.get("delivery_id")
        claim_token = receipt.get("claim_token")
        delivery_attempt = receipt.get("delivery_attempt")
        if (
            not isinstance(delivery_id, str)
            or not isinstance(claim_token, str)
            or not isinstance(delivery_attempt, int)
            or isinstance(delivery_attempt, bool)
            or delivery_attempt < 1
        ):
            raise RuntimeError("consumed worker delivery is missing its private claim receipt")
        key = (delivery_id, claim_token, delivery_attempt)
        with self._state_lock:
            if key not in self._delivery_outputs_pending:
                raise RuntimeError("consumed worker delivery is not awaiting client delivery")
        return TaskToolsOnlineOutput(
            "model",
            event,
            delivery_id=delivery_id,
            claim_token=claim_token,
            delivery_attempt=delivery_attempt,
        )

    def acknowledge_output(
        self,
        output: TaskToolsOnlineOutput,
        *,
        delivered: bool = True,
    ) -> Any | None:
        """Commit a worker delivery only after its model event reached the client."""

        if output.delivery_id is None or output.claim_token is None:
            return None
        assert output.delivery_attempt is not None
        key = (
            output.delivery_id,
            output.claim_token,
            output.delivery_attempt,
        )
        with self._state_lock:
            if key not in self._delivery_outputs_pending:
                raise ValueError("worker delivery output is not awaiting acknowledgement")
        record = self.gateway.delivery.acknowledge_consumed(
            output.delivery_id,
            output.claim_token,
            claim_attempt=output.delivery_attempt,
            delivered=delivered,
        )
        with self._state_lock:
            self._delivery_outputs_pending.discard(key)
        return record

    async def close(self, *, discard_state: bool = False) -> None:
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
        jobs = tuple(self._jobs)
        if self._delivery_task is not None:
            jobs = (*jobs, self._delivery_task)
        for job in jobs:
            if not job.done():
                job.cancel()
        if jobs:
            await asyncio.gather(*jobs, return_exceptions=True)
        # Stop every task that can enter the model lock before making the final
        # session call. Otherwise the delivery loop can race teardown and keep
        # WebSocket shutdown waiting behind an in-flight model operation.
        try:
            self.session.set_summary_needed_callback(None)
        except Exception:
            LOGGER.debug("could not clear summary callback", exc_info=True)
        with self._state_lock:
            pending_outputs = tuple(self._delivery_outputs_pending)
        for delivery_id, claim_token, delivery_attempt in pending_outputs:
            try:
                self.gateway.delivery.acknowledge_consumed(
                    delivery_id,
                    claim_token,
                    claim_attempt=delivery_attempt,
                    delivered=False,
                )
            except Exception:
                LOGGER.warning(
                    "could not release unsent worker delivery %s",
                    delivery_id,
                    exc_info=True,
                )
            else:
                with self._state_lock:
                    self._delivery_outputs_pending.discard(
                        (delivery_id, claim_token, delivery_attempt)
                    )
        try:
            await self.gateway.close()
            if discard_state:
                self.gateway.ledger.clear()
        finally:
            self._close_audio()
            if discard_state and self._audio_path is not None:
                self._audio_path.unlink(missing_ok=True)
            if self.close_ledger:
                self.gateway.ledger.close()

    async def _handle_task_tools(
        self,
        calls: tuple[dict[str, Any], ...],
        turn: TurnEnvelope,
    ) -> None:
        try:
            responses = []
            for call in calls:
                responses.append(
                    await self.task_tools.handle_native_tool_call(
                        call["name"], call["arguments"], turn
                    )
                )
            with self._state_lock:
                self._consumed_turns.add(turn.receipt_key)
            await self._sync_slate()
            response: Any = responses[0] if len(responses) == 1 else {
                "status": "batch",
                "results": responses,
            }
            event = await self._call_session("feed_tool_response", response)
            if event is not None:
                self.observe_frontbrain(event)
                await self._outputs.put(TaskToolsOnlineOutput("model", event))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            LOGGER.warning("task tool dispatch failed", exc_info=True)
            await self._deliver_tool_error(str(exc))
        finally:
            with self._state_lock:
                self._control_pending = False

    async def _expire_pending_task_call(self, nonce: int) -> None:
        await asyncio.sleep(self.turn_bind_grace_sec)
        with self._state_lock:
            if nonce != self._pending_task_nonce or self._pending_task_calls is None:
                return
            self._pending_task_calls = None
            self._pending_task_nonce += 1
        try:
            await self._deliver_tool_error(
                "task tools require a trusted turn.final event within the ASR grace window"
            )
        finally:
            with self._state_lock:
                self._control_pending = False

    async def _deliver_tool_error(self, detail: str) -> None:
        with self._state_lock:
            if self._recovering_tool_error:
                self._emit_control(
                    {
                        "type": "tool.error",
                        "message": "tool error recovery is already in progress",
                    }
                )
                return
            self._recovering_tool_error = True
        try:
            event = await self._call_session(
                "feed_tool_response",
                {
                    "status": "invalid_action",
                    "task_ids": [],
                    "reason": str(detail)[:512],
                },
            )
            if event is not None:
                self.observe_frontbrain(event)
                await self._outputs.put(TaskToolsOnlineOutput("model", event))
        except Exception as exc:
            self._emit_control(
                {"type": "tool.error", "message": str(exc)[:512]}
            )
        finally:
            with self._state_lock:
                self._recovering_tool_error = False

    async def _delivery_loop(self) -> None:
        while not self._closed:
            try:
                await self._sync_slate()
                pending = self.gateway.pending_deliveries(self.owner_id)
                if not pending or self._model_input_busy():
                    await asyncio.sleep(self.delivery_poll_sec)
                    continue
                first = pending[0]
                if first.timing == "safe_pause" and not await self._safe_pause():
                    await asyncio.sleep(self.delivery_poll_sec)
                    continue
                delivery = self.gateway.delivery.claim(self.owner_id)
                if delivery is None:
                    await asyncio.sleep(self.delivery_poll_sec)
                    continue
                with self._state_lock:
                    self._delivery_inflight = True
                try:
                    await self._sync_slate()
                    if delivery.timing == "interrupt":
                        # A worker interrupt stops the current output, but it must
                        # not set the persistent client-controlled break flag: the
                        # very next unit has to consume this delivery.
                        await self._call_session("interrupt_output")
                    response = worker_delivery_response(self.gateway, delivery)
                    key = (
                        delivery.delivery_id,
                        delivery.claim_token,
                        delivery.attempts,
                    )
                    with self._state_lock:
                        self._delivery_outputs_pending.add(key)
                    event = await self._call_session(
                        "feed_runtime_event",
                        response,
                        delivery_id=delivery.delivery_id,
                        claim_token=delivery.claim_token,
                        delivery_attempt=delivery.attempts,
                    )
                    if event is not None:
                        self.observe_frontbrain(event)
                        await self._outputs.put(
                            TaskToolsOnlineOutput(
                                "model",
                                event,
                                delivery_id=delivery.delivery_id,
                                claim_token=delivery.claim_token,
                                delivery_attempt=delivery.attempts,
                            )
                        )
                except asyncio.CancelledError:
                    key = (
                        delivery.delivery_id,
                        delivery.claim_token,
                        delivery.attempts,
                    )
                    with self._state_lock:
                        self._delivery_outputs_pending.discard(key)
                    try:
                        self.gateway.delivery.acknowledge_consumed(
                            delivery.delivery_id,
                            delivery.claim_token,
                            claim_attempt=delivery.attempts,
                            delivered=False,
                        )
                    except Exception:
                        LOGGER.warning(
                            "could not release cancelled worker delivery %s",
                            delivery.delivery_id,
                            exc_info=True,
                        )
                    raise
                except Exception as exc:
                    LOGGER.warning("worker delivery injection failed", exc_info=True)
                    key = (
                        delivery.delivery_id,
                        delivery.claim_token,
                        delivery.attempts,
                    )
                    with self._state_lock:
                        self._delivery_outputs_pending.discard(key)
                    try:
                        self.gateway.delivery.acknowledge_consumed(
                            delivery.delivery_id,
                            delivery.claim_token,
                            claim_attempt=delivery.attempts,
                            delivered=False,
                        )
                    except Exception:
                        LOGGER.warning("could not mark delivery failed", exc_info=True)
                    self._emit_control(
                        {
                            "type": "delivery.failed",
                            "delivery_id": delivery.delivery_id,
                            "message": str(exc)[:512],
                        }
                    )
                    await asyncio.sleep(
                        min(
                            1.0,
                            self.delivery_poll_sec
                            * (2 ** min(delivery.attempts, 6)),
                        )
                    )
                finally:
                    with self._state_lock:
                        self._delivery_inflight = False
            except asyncio.CancelledError:
                raise
            except Exception:
                LOGGER.warning("task-tools delivery loop failed", exc_info=True)
                await asyncio.sleep(self.delivery_poll_sec)

    async def _sync_slate(self, *, force: bool = False) -> bool:
        if not self.expose_task_slate_to_model:
            return False
        slate = self.task_tools.system_prompt_slate()
        with self._state_lock:
            if not force and slate == self._last_slate:
                return False
        changed = bool(await self._call_session("set_task_slate", slate))
        with self._state_lock:
            self._last_slate = slate
        return changed

    async def _safe_pause(self) -> bool:
        state = await self._call_session("talker_state")
        return not isinstance(state, Mapping) or (
            bool(state.get("drained", True))
            and bool(state.get("turn_ended", True))
        )

    def _model_input_busy(self) -> bool:
        with self._state_lock:
            return bool(
                self._control_pending
                or self._external_tool_pending is not None
                or self._delivery_inflight
                or self._delivery_outputs_pending
                or self._recovering_tool_error
            )

    def _on_summary_needed(self, payload: dict[str, Any]) -> None:
        loop = self._loop
        if loop is None or self._closed:
            return
        loop.call_soon_threadsafe(
            self._outputs.put_nowait,
            TaskToolsOnlineOutput(
                "control",
                {"type": "memory.summary_needed", "segment": dict(payload)},
            ),
        )

    def _emit_control(self, payload: dict[str, Any]) -> None:
        loop = self._loop
        if loop is None or self._closed:
            return
        loop.call_soon_threadsafe(
            self._outputs.put_nowait,
            TaskToolsOnlineOutput("control", payload),
        )

    def _spawn(self, coroutine: Any, suffix: str) -> None:
        task = asyncio.create_task(
            coroutine,
            name=f"gander-task-tools-{suffix}-{self.session_id}",
        )
        self._jobs.add(task)
        task.add_done_callback(self._jobs.discard)
        task.add_done_callback(self._log_job_failure)

    @staticmethod
    def _log_job_failure(task: asyncio.Task[Any]) -> None:
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            LOGGER.error("task-tools realtime job failed: %s", error)

    async def _call_session(self, method: str, *args: Any, **kwargs: Any) -> Any:
        function = getattr(self.session, method)
        return await asyncio.to_thread(function, *args, **kwargs)

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("task-tools realtime coordinator is closed")

    def _close_audio(self) -> None:
        with self._audio_lock:
            fd = self._audio_fd
            self._audio_fd = None
        if fd is None:
            return
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def _parse_media(values: Any) -> tuple[MediaRef, ...]:
    if not isinstance(values, (list, tuple)):
        raise ValueError("turn.final media must be an array")
    output: list[MediaRef] = []
    for value in values:
        if isinstance(value, MediaRef):
            output.append(value)
        elif isinstance(value, Mapping):
            output.append(MediaRef(**dict(value)))
        else:
            raise ValueError("turn.final media entries must be objects")
    return tuple(output)


def _optional_int(payload: Mapping[str, Any], name: str) -> int | None:
    value = payload.get(name)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"turn.final {name} must be an integer")
    return value


def _media_context_kind(kind: str) -> str:
    return {
        "screen": "screen",
        "image": "image",
        "frame": "video_frame",
        "video": "video_frame",
        "audio": "audio_transcript",
    }[kind]


def _dedupe_media(values: list[MediaRef]) -> tuple[MediaRef, ...]:
    output: list[MediaRef] = []
    seen: set[tuple[str, str]] = set()
    for value in values:
        key = (value.kind, value.path)
        if key in seen:
            continue
        seen.add(key)
        output.append(value)
    return tuple(output)
