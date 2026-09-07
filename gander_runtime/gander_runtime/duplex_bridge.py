from __future__ import annotations

import threading
from collections import deque
from typing import Any, Literal

from .screen import LatestScreenFrameBuffer, ScreenFrame


class GanderDuplexSession:
    """Thread-safe facade over one native-tool ``DuplexLiveSession``."""

    def __init__(
        self,
        live_session: Any,
        *,
        decode_mode: Literal["sampling", "greedy"] = "sampling",
        screen_frames: LatestScreenFrameBuffer | None = None,
    ) -> None:
        self.live = live_session
        self._model_lock = threading.RLock()
        self._pending_native_inputs: deque[dict[str, str | int | None]] = deque()
        self._native_input_receipts: dict[int, dict[str, str | int | None]] = {}
        self.screen_frames = (
            screen_frames
            if screen_frames is not None
            else LatestScreenFrameBuffer()
        )
        live_session.runner.params.decode_mode = decode_mode
        live_session.set_frame_source(self.screen_frames)

    def enqueue_screen_frame(self, frame: ScreenFrame) -> None:
        self.screen_frames.publish(frame)

    def set_media_mode(self, mode: str) -> str:
        """Switch vision on or off at a model-clock boundary; return the old mode.

        ``DuplexLiveSession`` re-reads ``config.media_mode`` for every unit, so
        this needs no session rebuild: the live KV cache, prefix snapshot, tools
        and Talker state all stay valid and vision simply starts or stops
        contributing frame tokens from the next unit onward.

        Holding the model lock makes the mode change and the buffer reset atomic
        with respect to unit generation. The order between them differs by
        direction: turning vision off must stop consumption before dropping the
        frames, and turning it on must drop stale frames before consumption can
        reach them.
        """

        if mode not in {"voice", "omni", "auto"}:
            raise ValueError(f"unsupported media_mode: {mode!r}")
        with self._model_lock:
            config = self.live.config
            previous = str(config.media_mode)
            if mode == previous:
                return previous
            if mode == "voice":
                config.media_mode = "voice"
                self.screen_frames.reset()
            else:
                self.screen_frames.reset()
                config.media_mode = mode
            return previous

    def feed_pcm16(
        self,
        data: bytes,
        *,
        unit_capture_start_ms: tuple[float, ...] | None = None,
    ) -> list[Any]:
        with self._model_lock:
            if unit_capture_start_ms is None:
                return self._annotate(self.live.feed_pcm16(data))
            return self._annotate(
                self.live.feed_pcm16(
                    data,
                    unit_capture_start_ms=unit_capture_start_ms,
                )
            )

    def flush_pending(
        self, *, unit_capture_start_ms: float | None = None
    ) -> list[Any]:
        with self._model_lock:
            return self._annotate(
                self.live.flush_pending(
                    unit_capture_start_ms=unit_capture_start_ms
                )
            )

    def step_silence(self) -> Any:
        with self._model_lock:
            return self._annotate([self.live.step_silence()])[0]

    def feed_tool_response(self, response: Any) -> Any | None:
        with self._model_lock:
            event = self.live.feed_tool_response(response)
            if event is None:
                self._pending_native_inputs.append(
                    {
                        "kind": "tool_response",
                        "delivery_id": None,
                        "claim_token": None,
                    }
                )
                return None
            return self._annotate([event])[0]

    def feed_runtime_event(
        self,
        event: Any,
        *,
        delivery_id: str | None = None,
        claim_token: str | None = None,
        delivery_attempt: int | None = None,
    ) -> Any | None:
        """Inject one later worker delivery without consuming a tool response slot."""

        claim_fields = (delivery_id, claim_token, delivery_attempt)
        if any(value is None for value in claim_fields) != all(
            value is None for value in claim_fields
        ):
            raise ValueError(
                "delivery_id, claim_token and delivery_attempt must be provided together"
            )
        if delivery_attempt is not None and (
            not isinstance(delivery_attempt, int)
            or isinstance(delivery_attempt, bool)
            or delivery_attempt < 1
        ):
            raise ValueError("delivery_attempt must be a positive integer")
        with self._model_lock:
            fitted = self._fit_runtime_event(event)
            generated = self.live.feed_runtime_event(fitted)
            if generated is None:
                self._pending_native_inputs.append(
                    {
                        "kind": "worker_delivery",
                        "delivery_id": delivery_id,
                        "claim_token": claim_token,
                        "delivery_attempt": delivery_attempt,
                    }
                )
                return None
            return self._annotate([generated])[0]

    def feed_memory_episode(self, episode: Any) -> bool:
        """Serialize protected-memory mutation with every other model call."""

        with self._model_lock:
            if self.live.runner.pinned_context is None:
                return False
            return bool(self.live.feed_memory_episode(episode))

    def set_task_slate(self, slate: str) -> bool:
        """Refresh the PFC task slate at a model-clock boundary."""

        with self._model_lock:
            if self.live.runner.pinned_context is None:
                return False
            return bool(self.live.set_task_slate(slate))

    def set_summary_needed_callback(self, callback: Any) -> None:
        with self._model_lock:
            if self.live.runner.pinned_context is None:
                return
            self.live.set_summary_needed_callback(callback)

    def talker_state(self) -> dict[str, Any]:
        with self._model_lock:
            return dict(self.live.talker_state())

    def poll_output(self, timeout: float = 0.0) -> Any | None:
        return self.live.poll_output(timeout)

    def drain_outputs(self) -> list[Any]:
        return self.live.drain_outputs()

    def wait_for_speech(self, timeout: float | None = None) -> bool:
        return bool(self.live.wait_for_speech(timeout))

    def close(self, *args: Any, **kwargs: Any) -> Any:
        with self._model_lock:
            return self.live.close(*args, **kwargs)

    @property
    def closed(self) -> bool:
        return bool(self.live.closed)

    def set_break(self) -> None:
        with self._model_lock:
            self.live.set_break()

    def interrupt_output(self) -> None:
        with self._model_lock:
            self.live.interrupt_output()

    def clear_break(self) -> None:
        with self._model_lock:
            self.live.clear_break()

    def should_continue_draining(self, trailing_steps: int) -> bool:
        return bool(self.live.should_continue_draining(trailing_steps))

    def should_stop_after(self, event: Any) -> bool:
        return bool(self.live.should_stop_after(event))

    def take_native_input_receipt(
        self, event: Any
    ) -> dict[str, str | int | None] | None:
        return self._native_input_receipts.pop(id(event), None)

    def _annotate(self, events: list[Any]) -> list[Any]:
        for event in events:
            event.metrics["gander"] = {
                "protocol": "native_tools",
                "front_control": None,
                "terminal_share_ids": [],
            }
            prefill_mode = str(event.metrics.get("prefill_mode") or "")
            if prefill_mode.endswith("+TOOL"):
                if not self._pending_native_inputs:
                    raise RuntimeError(
                        "MiniCPM consumed a native tool input without a matching Runtime receipt"
                    )
                receipt = self._pending_native_inputs.popleft()
                event.metrics["gander"]["native_input_kind"] = receipt["kind"]
                self._native_input_receipts[id(event)] = receipt
        return events

    def _fit_runtime_event(self, event: Any) -> Any:
        """Keep a worker delivery within the live model's response-token budget."""

        from mcpmft.tool_protocol import format_tool_response

        if not isinstance(event, dict) or event.get("type") != "worker_delivery":
            raise ValueError("runtime event must be a worker_delivery object")
        runner = self.live.runner
        tokenizer = runner.bundle.tokenizer
        limit = int(runner.params.max_tool_response_tokens)

        def fits(value: dict[str, Any]) -> bool:
            rendered = format_tool_response(value)
            return len(tokenizer.encode(rendered, add_special_tokens=False)) <= limit

        if fits(event):
            return event
        candidate = {key: value for key, value in event.items() if key != "unresolved"}
        content = str(candidate.get("content") or "")
        low, high = 0, len(content)
        while low < high:
            middle = (low + high + 1) // 2
            bounded = {**candidate, "content": content[:middle]}
            if fits(bounded):
                low = middle
            else:
                high = middle - 1
        bounded = {**candidate, "content": content[:low]}
        if (content and not bounded["content"]) or not fits(bounded):
            raise ValueError(
                f"worker delivery envelope exceeds max_tool_response_tokens={limit}"
            )
        return bounded
