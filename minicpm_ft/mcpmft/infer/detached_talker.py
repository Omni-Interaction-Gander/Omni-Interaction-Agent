from __future__ import annotations

import importlib
import logging
import queue
import threading
import time
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from mcpmft.modeling.load import load_prefixed_submodule_state_dict


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SpeechSynthesisRequest:
    generation_id: int
    unit_id: int
    current_time: int | None
    token_ids: tuple[int, ...]
    hidden_states: Any
    end_of_turn: bool


@dataclass
class SpeechSynthesisChunk:
    generation_id: int
    unit_id: int
    sequence: int
    waveform: np.ndarray
    current_time: int | None
    end_of_turn: bool
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PlaybackCancel:
    generation_id: int
    cancelled_generation_id: int
    reason: str


@dataclass(frozen=True)
class SpeechSynthesisError:
    generation_id: int
    unit_id: int
    message: str


@dataclass(frozen=True)
class SpeechSynthesisDone:
    generation_id: int
    unit_id: int
    end_of_turn: bool
    metrics: dict[str, Any]


class SynthesisCancelled(RuntimeError):
    pass


@dataclass
class DetachedTalkerConfig:
    speech_tokens_per_unit: int = 25
    emit_speech_tokens: int = 25
    final_speech_tokens_max: int = 0
    sample_rate: int = 24000
    temperature: float = 0.8
    repetition_penalty: float = 1.05
    silence_token_id: int = 4218

    def __post_init__(self) -> None:
        if self.speech_tokens_per_unit <= 0:
            raise ValueError("speech_tokens_per_unit must be positive")
        if self.emit_speech_tokens <= 0:
            raise ValueError("emit_speech_tokens must be positive")
        if self.emit_speech_tokens != 25:
            raise ValueError(
                "StepAudio Token2wav streaming requires 25 S3 tokens per emitted chunk"
            )
        if self.final_speech_tokens_max < 0:
            raise ValueError("final_speech_tokens_max must be non-negative")


class DetachedTalkerRuntime:
    """Stateful Talker and Token2wav runtime hosted independently from the Thinker.

    One instance serves one session at a time. It keeps Talker KV and Token2wav caches on its own
    CUDA device, allowing the Thinker to continue ingesting microphone units on another device.
    """

    def __init__(
        self,
        *,
        tts: Any,
        model_module: Any,
        device: str,
        prompt_wav_path: str,
        config: DetachedTalkerConfig | None = None,
    ) -> None:
        self.tts = tts
        self.model_module = model_module
        self.device = str(device)
        self.prompt_wav_path = str(prompt_wav_path)
        self.config = config or DetachedTalkerConfig()
        self.eos_token_id = int(self.tts.config.num_audio_tokens) - 1
        self._clone = getattr(model_module, "torch_clone_recursive")
        self._voice_ready = False
        self._flow_cache_base = None
        self._hift_cache_base = None
        self._pre_lookahead = 3
        self._past_key_values = None
        self._text_start_pos = 0
        self._token2wav_buffer: list[int] = []
        self._prepare_voice()
        self.reset()

    @classmethod
    def from_thinker_model(
        cls,
        thinker_model: Any,
        *,
        base_model_checkpoint: str,
        talker_checkpoint: str,
        token2wav_dir: str,
        prompt_wav_path: str,
        device: str,
        n_timesteps: int = 10,
        enable_float16: bool = False,
        config: DetachedTalkerConfig | None = None,
    ) -> "DetachedTalkerRuntime":
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("Detached Talker requires CUDA")
        target = torch.device(device)
        if target.type != "cuda":
            raise ValueError(f"Detached Talker device must be CUDA, got {device!r}")

        model_module = importlib.import_module(type(thinker_model).__module__)
        tts_class = getattr(model_module, "MiniCPMTTS")
        tts_config = deepcopy(thinker_model.config.tts_config)
        if getattr(thinker_model.config, "_attn_implementation", None) == "flash_attention_2":
            tts_config.attn_implementation = "flash_attention_2"
        else:
            tts_config.attn_implementation = "eager"

        dtype = next(thinker_model.parameters()).dtype
        with torch.cuda.device(target):
            tts = tts_class(config=tts_config, audio_tokenizer=None)
            tts.to(device=target, dtype=dtype)
            base_loaded = load_prefixed_submodule_state_dict(
                tts,
                base_model_checkpoint,
                prefix="tts.",
                strict=True,
            )
            overlay_loaded = load_prefixed_submodule_state_dict(
                tts,
                talker_checkpoint,
                prefix="tts.",
                strict=False,
            )
            if overlay_loaded <= 0:
                raise RuntimeError(f"No Talker tensors loaded from {talker_checkpoint}")
            LOGGER.info(
                "Initialized detached Talker with %d base and %d fine-tuned tensors",
                base_loaded,
                overlay_loaded,
            )
            tts.eval()

            try:
                from stepaudio2 import Token2wav
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    "Detached Talker requires stepaudio2/Token2wav from minicpmo-utils"
                ) from exc
            tts.config.audio_tokenizer_type = "s3tokenizer_step_audio"
            tts.audio_tokenizer = Token2wav(
                str(token2wav_dir),
                float16=bool(enable_float16),
                n_timesteps=int(n_timesteps),
            )

        return cls(
            tts=tts,
            model_module=model_module,
            device=str(target),
            prompt_wav_path=prompt_wav_path,
            config=config,
        )

    def reset(self) -> None:
        self._past_key_values = None
        self._text_start_pos = 0
        self._reset_token2wav()

    def warm_token2wav(self) -> None:
        """Run the real first streaming vocoder chunk without retaining session state."""
        import torch

        target = torch.device(self.device)
        with torch.cuda.device(target), torch.inference_mode():
            cpu_rng_state = torch.random.get_rng_state()
            cuda_rng_state = torch.cuda.get_rng_state(target)
            try:
                self.reset()
                codes = np.full(
                    (1, self.config.emit_speech_tokens, int(self.tts.num_vq)),
                    self.config.silence_token_id,
                    dtype=np.int64,
                )
                self._tokens_to_waveforms(
                    codes,
                    is_last_chunk=False,
                    force_flush=True,
                )
                torch.cuda.synchronize(target)
            finally:
                self.reset()
                torch.random.set_rng_state(cpu_rng_state)
                torch.cuda.set_rng_state(cuda_rng_state, target)

    def synthesize(
        self,
        request: SpeechSynthesisRequest,
        cancel_event: threading.Event,
        emit: Callable[[SpeechSynthesisChunk], None],
    ) -> dict[str, Any]:
        import torch

        started = time.monotonic()
        sequence = 0
        emitted_codes = 0
        emitted_audio_sec = 0.0
        token2wav_cost = 0.0
        force_flush = self._text_start_pos == 0

        with torch.cuda.device(torch.device(self.device)), torch.inference_mode():
            condition = self._build_condition(request)
            condition_length = int(condition.shape[1])
            visible_budget = self._visible_budget(condition_length, request.end_of_turn)
            min_new_tokens = 0 if request.end_of_turn else visible_budget + 1

            def emit_codes(codes, *, final: bool) -> None:
                nonlocal sequence, emitted_codes, emitted_audio_sec, token2wav_cost
                if cancel_event.is_set():
                    raise SynthesisCancelled()
                wav_started = time.monotonic()
                waveforms = self._tokens_to_waveforms(
                    codes,
                    is_last_chunk=final,
                    force_flush=force_flush and emitted_codes == 0,
                )
                token2wav_cost += time.monotonic() - wav_started
                if cancel_event.is_set():
                    raise SynthesisCancelled()
                emitted_codes += int(codes.numel())
                for waveform in waveforms:
                    if waveform.size == 0:
                        continue
                    sequence += 1
                    duration = float(waveform.size / self.config.sample_rate)
                    emitted_audio_sec += duration
                    emit(
                        SpeechSynthesisChunk(
                            generation_id=request.generation_id,
                            unit_id=request.unit_id,
                            sequence=sequence,
                            waveform=waveform,
                            current_time=request.current_time,
                            end_of_turn=bool(request.end_of_turn and final),
                            metrics={
                                "n_tts_tokens": emitted_codes,
                                "audio_duration_sec": duration,
                                "cost_token2wav": token2wav_cost,
                            },
                        )
                    )

            past_key_values, visible_count, ar_cost = self._generate_incremental(
                inputs_embeds=condition,
                visible_budget=visible_budget,
                min_new_tokens=min_new_tokens,
                cancel_event=cancel_event,
                emit_codes=emit_codes,
                final=request.end_of_turn,
            )

            if cancel_event.is_set():
                raise SynthesisCancelled()
            if request.end_of_turn:
                self._past_key_values = None
                self._text_start_pos = 0
                self._reset_token2wav()
            else:
                self._past_key_values = past_key_values
                self._text_start_pos += condition_length + visible_count

        return {
            "generation_id": request.generation_id,
            "unit_id": request.unit_id,
            "audio_chunks": sequence,
            "n_tts_tokens": visible_count,
            "audio_duration_sec": emitted_audio_sec,
            "cost_tts": ar_cost,
            "cost_token2wav": token2wav_cost,
            "cost_all": time.monotonic() - started,
        }

    def _prepare_voice(self) -> None:
        import torch

        path = Path(self.prompt_wav_path)
        if not path.is_file():
            raise FileNotFoundError(f"Detached Talker reference audio not found: {path}")
        with torch.cuda.device(torch.device(self.device)):
            self.tts.audio_tokenizer.cache = None
            flow_cache, hift_cache = self.tts.audio_tokenizer.set_stream_cache(str(path))
            self._flow_cache_base = self._clone(flow_cache)
            self._hift_cache_base = self._clone(hift_cache)
            self._pre_lookahead = int(self.tts.audio_tokenizer.flow.pre_lookahead_len)
            self._voice_ready = True

    def _reset_token2wav(self) -> None:
        if not self._voice_ready:
            return
        self.tts.audio_tokenizer.stream_cache = self._clone(self._flow_cache_base)
        self.tts.audio_tokenizer.hift_cache_dict = self._clone(self._hift_cache_base)
        self._token2wav_buffer = [self.config.silence_token_id] * self._pre_lookahead

    def _build_condition(self, request: SpeechSynthesisRequest):
        import torch
        import torch.nn.functional as functional

        if not request.token_ids:
            raise ValueError("A Talker request must contain at least one condition token")
        hidden = request.hidden_states
        if not torch.is_tensor(hidden):
            hidden = torch.as_tensor(hidden)
        if hidden.ndim != 2 or hidden.shape[0] != len(request.token_ids):
            raise ValueError(
                "Talker hidden states must have shape [num_tokens, hidden_size]: "
                f"tokens={len(request.token_ids)} hidden_shape={tuple(hidden.shape)}"
            )

        token_ids = torch.tensor(request.token_ids, dtype=torch.long, device=self.device)
        text_embeds = self.tts.emb_text(token_ids)
        hidden = hidden.to(device=self.device, dtype=text_embeds.dtype)
        semantic = self.tts.projector_semantic(hidden)
        semantic = functional.normalize(semantic, p=2, dim=-1)
        condition = text_embeds + semantic
        audio_bos = self.tts.emb_text(
            torch.tensor(
                [self.tts.audio_bos_token_id],
                dtype=torch.long,
                device=self.device,
            )
        )
        return torch.cat([condition, audio_bos], dim=0).unsqueeze(0)

    def _visible_budget(self, condition_length: int, final: bool) -> int:
        context_limit = int(self.tts.config.max_position_embeddings)
        remaining = context_limit - self._text_start_pos - condition_length
        if remaining <= 0:
            raise RuntimeError(
                "Detached Talker positional context exhausted: "
                f"limit={context_limit} text_start={self._text_start_pos} "
                f"condition={condition_length}"
            )
        if final:
            cap = self.config.final_speech_tokens_max
            return min(cap, remaining) if cap else remaining
        if remaining < self.config.speech_tokens_per_unit:
            raise RuntimeError(
                "Detached Talker cannot fit a fixed non-final speech unit: "
                f"required={self.config.speech_tokens_per_unit} remaining={remaining}"
            )
        return self.config.speech_tokens_per_unit

    def _generate_incremental(
        self,
        *,
        inputs_embeds,
        visible_budget: int,
        min_new_tokens: int,
        cancel_event: threading.Event,
        emit_codes: Callable[..., None],
        final: bool,
    ):
        """Run the upstream AR loop while releasing complete S3 subchunks immediately."""
        import torch
        import torch.nn.functional as functional
        import torch.nn.utils.parametrize as parametrize

        _, logits_processors = self.model_module.gen_logits(
            num_code=self.tts.config.num_audio_tokens,
            repetition_penalty=self.config.repetition_penalty,
        )
        batch_size = int(inputs_embeds.shape[0])
        if batch_size != 1:
            raise ValueError("Detached Talker currently supports batch size 1")

        max_new_token = visible_budget + 1
        condition_length = int(inputs_embeds.shape[1])
        new_tokens = torch.zeros(
            batch_size,
            max_new_token,
            self.tts.num_vq,
            dtype=torch.long,
            device=self.device,
        )
        temperature = torch.tensor(
            [self.config.temperature],
            dtype=torch.float,
            device=self.device,
        ).view(-1, 1)
        eos_token = torch.tensor([self.eos_token_id], dtype=torch.long, device=self.device)
        past_key_values = self._past_key_values
        finish = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        emitted = 0
        ar_cost = 0.0
        total_visible = 0

        for step in range(max_new_token):
            if cancel_event.is_set():
                raise SynthesisCancelled()
            ar_started = time.monotonic()
            audio_bos = step == 0
            if audio_bos:
                model_input = inputs_embeds
                position_ids = torch.arange(
                    self._text_start_pos,
                    self._text_start_pos + condition_length,
                    dtype=torch.long,
                    device=self.device,
                ).unsqueeze(0)
            else:
                model_input = self.tts.emb_code[0](new_tokens[:, step - 1 : step, 0])
                position_ids = torch.tensor(
                    [self._text_start_pos + condition_length + step - 1],
                    dtype=torch.long,
                    device=self.device,
                ).unsqueeze(0)

            outputs = self.tts.model(
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=model_input,
                use_cache=True,
                output_attentions=False,
            )
            hidden_states = outputs.last_hidden_state
            past_key_values = outputs.past_key_values
            with parametrize.cached():
                logits = torch.empty(
                    hidden_states.size(0),
                    hidden_states.size(1),
                    self.tts.num_audio_tokens,
                    self.tts.num_vq,
                    dtype=torch.float,
                    device=self.device,
                )
                for codebook in range(self.tts.num_vq):
                    logits[..., codebook] = self.tts.head_code[codebook](hidden_states)

            logits = logits[:, -1].float().permute(0, 2, 1)
            logits = logits.reshape(-1, logits.size(2))
            logits /= temperature
            if not audio_bos:
                previous = new_tokens[:, :step].permute(0, 2, 1)
                previous = previous.reshape(previous.size(0) * previous.size(1), -1)
                for processor in logits_processors:
                    logits = processor(previous.to(self.device), logits)
            if step < min_new_tokens:
                logits[:, eos_token] = -torch.inf

            scores = functional.softmax(logits, dim=-1)
            next_token = torch.multinomial(scores, num_samples=1).to(finish.device)
            next_token = next_token.view(-1, self.tts.num_vq)
            finish.logical_or_(next_token.eq(eos_token).any(1))
            new_tokens[:, step] = next_token
            ar_cost += time.monotonic() - ar_started

            if finish.all():
                total_visible = step
                break

            safe_visible = step + 1
            while safe_visible - emitted >= self.config.emit_speech_tokens:
                stop = emitted + self.config.emit_speech_tokens
                emit_codes(new_tokens[:, emitted:stop].clone(), final=False)
                emitted = stop
        else:
            # Hold the upstream look-ahead token for the next step.
            total_visible = max_new_token - 1

        if total_visible > emitted:
            emit_codes(new_tokens[:, emitted:total_visible].clone(), final=final)
            emitted = total_visible
        elif final:
            # Flush Token2wav look-ahead state at EOS.
            emit_codes(new_tokens[:, 0:0].clone(), final=True)

        return past_key_values, total_visible, ar_cost

    def _tokens_to_waveforms(
        self,
        new_tokens,
        *,
        is_last_chunk: bool,
        force_flush: bool,
    ) -> list[np.ndarray]:
        token_ids = new_tokens.reshape(-1).tolist()
        self._token2wav_buffer.extend(token_ids)
        chunk_size = int(self.config.emit_speech_tokens)
        pcm_chunks: list[Any] = []

        if force_flush:
            while len(self._token2wav_buffer) >= self._pre_lookahead + 5:
                process_count = min(
                    chunk_size + self._pre_lookahead,
                    len(self._token2wav_buffer),
                )
                pcm_chunks.append(
                    self.tts.audio_tokenizer.stream(
                        self._token2wav_buffer[:process_count],
                        prompt_wav=self.prompt_wav_path,
                    )
                )
                consumed = min(chunk_size, process_count - self._pre_lookahead)
                self._token2wav_buffer = self._token2wav_buffer[consumed:]
        else:
            while len(self._token2wav_buffer) >= chunk_size + self._pre_lookahead:
                pcm_chunks.append(
                    self.tts.audio_tokenizer.stream(
                        self._token2wav_buffer[: chunk_size + self._pre_lookahead],
                        prompt_wav=self.prompt_wav_path,
                    )
                )
                self._token2wav_buffer = self._token2wav_buffer[chunk_size:]

        if is_last_chunk and self._token2wav_buffer:
            pcm_chunks.append(
                self.tts.audio_tokenizer.stream(
                    self._token2wav_buffer,
                    prompt_wav=self.prompt_wav_path,
                    last_chunk=True,
                )
            )
            self._token2wav_buffer = []

        return [self._pcm_to_float32(value) for value in pcm_chunks if value is not None]

    @staticmethod
    def _pcm_to_float32(value: Any) -> np.ndarray:
        if isinstance(value, bytes):
            return np.frombuffer(value, dtype="<i2").astype(np.float32) / 32768.0
        array = np.asarray(value)
        if np.issubdtype(array.dtype, np.integer):
            return array.astype(np.float32).reshape(-1) / 32768.0
        return array.astype(np.float32, copy=False).reshape(-1)


@dataclass(frozen=True)
class _QueuedRequest:
    request: SpeechSynthesisRequest
    cancel_event: threading.Event


class AsyncTalkerWorker:
    """Single-session background Talker with epoch-based stale-output rejection."""

    def __init__(self, runtime: DetachedTalkerRuntime | Any) -> None:
        self.runtime = runtime
        self._requests: queue.Queue[_QueuedRequest | None] = queue.Queue()
        self._outputs: queue.Queue[
            SpeechSynthesisChunk
            | SpeechSynthesisDone
            | PlaybackCancel
            | SpeechSynthesisError
        ] = queue.Queue()
        self._lock = threading.Lock()
        self._drained = threading.Condition(self._lock)
        self._generation_id = 1
        self._cancel_event = threading.Event()
        self._pending = 0
        self._active = False
        self._closed = False
        self._runtime_generation: int | None = None
        self._thread = threading.Thread(
            target=self._run,
            name="detached-talker",
            daemon=True,
        )
        self._thread.start()

    @property
    def generation_id(self) -> int:
        with self._lock:
            return self._generation_id

    def submit(
        self,
        *,
        unit_id: int,
        current_time: int | None,
        token_ids: tuple[int, ...],
        hidden_states: Any,
        end_of_turn: bool,
    ) -> SpeechSynthesisRequest:
        with self._lock:
            if self._closed:
                raise RuntimeError("Detached Talker worker is closed")
            request = SpeechSynthesisRequest(
                generation_id=self._generation_id,
                unit_id=unit_id,
                current_time=current_time,
                token_ids=token_ids,
                hidden_states=hidden_states,
                end_of_turn=end_of_turn,
            )
            queued = _QueuedRequest(request=request, cancel_event=self._cancel_event)
            self._pending += 1
        self._requests.put(queued)
        return request

    def cancel(self, reason: str) -> PlaybackCancel:
        with self._lock:
            cancelled = self._generation_id
            self._cancel_event.set()
            self._generation_id += 1
            self._cancel_event = threading.Event()
            event = PlaybackCancel(
                generation_id=self._generation_id,
                cancelled_generation_id=cancelled,
                reason=reason,
            )
        LOGGER.info(
            "Detached Talker cancelled generation=%d next_generation=%d reason=%s",
            cancelled,
            event.generation_id,
            reason,
        )
        self._outputs.put(event)
        return event

    def poll(
        self,
        timeout: float = 0.0,
    ) -> (
        SpeechSynthesisChunk
        | SpeechSynthesisDone
        | PlaybackCancel
        | SpeechSynthesisError
        | None
    ):
        try:
            return self._outputs.get(timeout=max(0.0, timeout))
        except queue.Empty:
            return None

    def drain_outputs(
        self,
    ) -> list[
        SpeechSynthesisChunk
        | SpeechSynthesisDone
        | PlaybackCancel
        | SpeechSynthesisError
    ]:
        values = []
        while True:
            value = self.poll()
            if value is None:
                return values
            values.append(value)

    def state(self) -> dict[str, Any]:
        with self._lock:
            return {
                "active": self._active,
                "drained": not self._active and self._pending == 0,
                "pending_requests": self._pending,
                "generation_id": self._generation_id,
                "pending_output_events": self._outputs.qsize(),
            }

    def wait_until_drained(self, timeout: float | None = None) -> bool:
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._drained:
            while self._active or self._pending:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return False
                self._drained.wait(remaining)
            return True

    def close(self, *, drain: bool = False, timeout: float = 10.0) -> None:
        drained = self.wait_until_drained(timeout) if drain else False
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if not drained:
                self._cancel_event.set()
        self._requests.put(None)
        self._thread.join(timeout)
        if self._thread.is_alive():
            raise RuntimeError("Detached Talker thread did not stop before timeout")

    def _run(self) -> None:
        while True:
            queued = self._requests.get()
            if queued is None:
                return
            request = queued.request
            with self._drained:
                self._pending -= 1
                if queued.cancel_event.is_set():
                    self._drained.notify_all()
                    continue
                self._active = True

            try:
                if self._runtime_generation != request.generation_id:
                    self.runtime.reset()
                    self._runtime_generation = request.generation_id
                metrics = self.runtime.synthesize(
                    request,
                    queued.cancel_event,
                    self._outputs.put,
                )
                if not queued.cancel_event.is_set():
                    if int((metrics or {}).get("audio_chunks", 0)) == 0:
                        LOGGER.warning(
                            "Detached Talker produced no audio generation=%d unit=%d "
                            "end_of_turn=%s metrics=%s",
                            request.generation_id,
                            request.unit_id,
                            request.end_of_turn,
                            metrics,
                        )
                    self._outputs.put(
                        SpeechSynthesisDone(
                            generation_id=request.generation_id,
                            unit_id=request.unit_id,
                            end_of_turn=request.end_of_turn,
                            metrics=dict(metrics or {}),
                        )
                    )
            except SynthesisCancelled:
                try:
                    self.runtime.reset()
                except Exception:
                    LOGGER.exception("Detached Talker reset failed after cancellation")
                self._runtime_generation = None
            except Exception as exc:  # pragma: no cover
                LOGGER.exception(
                    "Detached Talker failed for generation=%d unit=%d",
                    request.generation_id,
                    request.unit_id,
                )
                try:
                    self.runtime.reset()
                except Exception:
                    LOGGER.exception("Detached Talker reset also failed")
                self._runtime_generation = None
                self._outputs.put(
                    SpeechSynthesisError(
                        generation_id=request.generation_id,
                        unit_id=request.unit_id,
                        message=str(exc),
                    )
                )
            finally:
                with self._drained:
                    self._active = False
                    self._drained.notify_all()
