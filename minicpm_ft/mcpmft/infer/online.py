from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Literal, Mapping, Sequence, cast

from mcpmft.infer.common import InferBundle, save_wav
from mcpmft.infer.context_window import CONTEXT_NO_PREVIOUS, install_context_no_previous
from mcpmft.infer.pinned_context import (
    CONTEXT_MEMORY,
    CONTEXT_SLATE,
    PinnedContextConfig,
    PinnedContextController,
    install_context_memory,
    install_context_slate,
)
from mcpmft.infer.streaming_text import DuplexStreamingText
from mcpmft.tool_protocol import (
    MAX_TOOL_CALLS_PER_UNIT,
    ToolProtocolError,
    format_tool_response,
    parse_tool_calls,
    system_prompt_with_tools,
    validate_realtime_tool_context,
    validate_tool_calls,
)


_AS_DUPLEX_LOCK = threading.Lock()
RuntimeWindowMode = Literal[
    "off", "basic", "context", "context_no_previous", "context_memory", "context_slate"
]


def _mask_generation_logits(
    logits: Any,
    *,
    allowed: Iterable[int | None] | None = None,
    forbidden: Iterable[int | None] = (),
):
    """Return a cloned logits tensor with the current unit grammar applied."""

    constrained = logits.clone()
    vocab_size = int(constrained.shape[-1])
    if allowed is not None:
        valid = tuple(
            dict.fromkeys(
                int(token_id)
                for token_id in allowed
                if token_id is not None and 0 <= int(token_id) < vocab_size
            )
        )
        # Lightweight tests may use a one-logit fake decoder with real tokenizer ids.
        if not valid:
            return constrained
        masked = constrained.new_full(constrained.shape, float("-inf"))
        masked[..., list(valid)] = constrained[..., list(valid)]
        constrained = masked
    invalid = tuple(
        dict.fromkeys(
            int(token_id)
            for token_id in forbidden
            if token_id is not None and 0 <= int(token_id) < vocab_size
        )
    )
    if invalid:
        constrained[..., list(invalid)] = float("-inf")
    return constrained


def _extract_decode_logits(
    args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[Any, tuple[Any, ...], dict[str, Any]]:
    """Normalize decoder.decode's positional or keyword logits argument."""

    normalized = dict(kwargs)
    if "logits" in normalized:
        return normalized.pop("logits"), args, normalized
    if not args:
        raise TypeError("decoder.decode requires logits")
    return args[0], args[1:], normalized


@contextmanager
def _allow_lexical_tool_tokens(decoder: Any, tokenizer: Any) -> Iterator[None]:
    """Disable speech-only lexical masks while decoding a native tool action.

    MiniCPM-o initializes ``StreamDecoder.forbidden_token_ids`` with the tokenizer's
    ``bad_token_ids`` for spoken-response generation. Native tool actions are text, and their
    canonical supervised representation contains lexical tokens from that list (notably the
    leading newline and common JSON-closing token). Applying the speech mask here makes the
    decoder leave its trained trajectory and continue argument values indefinitely.

    Keep all non-lexical/special-token restrictions intact, including the decoder's chunk-EOS
    handling. Restore the exact original object on exit so dialogue/Talker decoding is unchanged.
    """

    original = getattr(decoder, "forbidden_token_ids", None)
    if not isinstance(original, (list, tuple)):
        yield
        return

    bad_token_ids = {
        int(token_id)
        for token_id in (getattr(tokenizer, "bad_token_ids", ()) or ())
    }
    special_token_ids = {
        int(token_id)
        for token_id in (getattr(tokenizer, "all_special_ids", ()) or ())
    }
    speech_only_lexical_ids = bad_token_ids - special_token_ids
    filtered = [
        token_id
        for token_id in original
        if int(token_id) not in speech_only_lexical_ids
    ]
    if len(filtered) == len(original):
        yield
        return

    decoder.forbidden_token_ids = filtered
    try:
        yield
    finally:
        decoder.forbidden_token_ids = original


@dataclass
class DuplexPrefixSnapshot:
    """Immutable prepared system/tool prefix used to seed independent live sessions."""

    decoder_cache: Any
    decoder_state: dict[str, Any]
    duplex_state: dict[str, Any]
    tool_schemas: list[dict[str, Any]]
    token_count: int
    sliding_window_mode: RuntimeWindowMode
    generate_audio: bool
    token2wav_state: dict[str, Any] | None = None


_PREFIX_DECODER_STATE = (
    "_system_preserve_length",
    "_preserve_prefix_length",
    "_previous_content_length",
    "_suffix_token_ids",
    "_previous_marker",
    "_previous_marker_token_ids",
    "_has_previous",
    "_previous_text",
    "_previous_token_ids",
)
_PREFIX_DUPLEX_STATE = (
    "_prefix_system_prompt",
    "_suffix_system_prompt",
    "_ref_audio",
)


def _clone_decoder_cache(cache: Any) -> Any:
    """Clone the Transformers DynamicCache used by MiniCPM-o 4.5."""

    cloned = type(cache)()
    cloned.key_cache = [tensor.clone() for tensor in cache.key_cache]
    cloned.value_cache = [tensor.clone() for tensor in cache.value_cache]
    cloned._seen_tokens = cache._seen_tokens
    return cloned


def _clone_tensor_tree(value: Any) -> Any:
    """Clone nested token2wav state without sharing mutable tensor storage."""

    import torch

    if torch.is_tensor(value):
        return value.clone()
    if isinstance(value, dict):
        return {key: _clone_tensor_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_tensor_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_tensor_tree(item) for item in value)
    return deepcopy(value)


def runtime_window_mode_for_training(mode: str | None) -> RuntimeWindowMode:
    """Map an additive training layout name to its matching online KV policy."""

    resolved = {
        "window_no_previous": "context_no_previous",
        "sampled_context": "context",
        "context_memory": "context_memory",
        "context_slate": "context_slate",
    }.get(str(mode or ""), "off")
    return cast(RuntimeWindowMode, resolved)


@dataclass
class DuplexParams:
    chunk_ms: int = 1000
    first_chunk_ms: int = 1035
    cnn_redundancy_ms: int = 20
    ls_mode: str = "explicit"
    # The training serializer emits one action token, at most K lexical tokens, an optional turn
    # terminator, and the chunk terminator. These MUST match the trained unit contract
    # (data.duplex_text_tokens_per_unit / data.duplex_speech_tokens_per_unit): the decode loop
    # force-closes a speak unit once `speak_text_tokens_per_unit` lexical tokens are emitted, so a
    # value below the trained K silently truncates generation. Serving configurations must set
    # this to the same value as data.duplex_text_tokens_per_unit used for training.
    speak_text_tokens_per_unit: int = 4
    max_new_speak_tokens_per_chunk: int = 7
    max_new_tool_tokens: int = 96
    max_tool_response_tokens: int = 256
    max_tool_calls_per_unit: int = MAX_TOOL_CALLS_PER_UNIT
    max_tool_schemas: int = 6
    max_tool_schema_tokens: int = 1024
    # Keep the model-visible prefix identical to training by default. Deployments that
    # deliberately trained with a live clock hint may opt in explicitly.
    inject_search_time_context: bool = False
    decode_mode: Literal["sampling", "greedy"] = "sampling"
    generate_audio: bool = True
    n_timesteps: int = 10
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    sliding_window_mode: RuntimeWindowMode = "off"
    basic_window_high_tokens: int = 8000
    basic_window_low_tokens: int = 6000
    context_previous_max_tokens: int = 500
    context_max_units: int = 128
    memory_slate_max_tokens: int = 256
    memory_lead_ratio: float = 0.7
    memory_soft_ratio: float = 0.9
    memory_hard_ratio: float = 1.0
    memory_kv_ceiling_units: int | None = None
    # Talker generation uses one look-ahead token internally, so an S-token non-final unit is
    # requested as max_new_token=S+1. The final unit uses the remaining decoder context and may
    # continue until S3 EOS.
    # Must match data.duplex_speech_tokens_per_unit in the training configuration.
    talker_speech_tokens_per_unit: int = 25
    # Zero lets the final unit continue until S3 EOS or the decoder context limit.
    talker_final_speech_tokens_max: int = 0


class OnlineRunner:
    def __init__(self, bundle: InferBundle, params: DuplexParams | None = None) -> None:
        self.bundle = bundle
        self.params = params or DuplexParams()
        if self.params.speak_text_tokens_per_unit < 1:
            raise ValueError("speak_text_tokens_per_unit must be positive")
        minimum_speak_budget = self.params.speak_text_tokens_per_unit + 3
        if self.params.max_new_speak_tokens_per_chunk < minimum_speak_budget:
            raise ValueError(
                "max_new_speak_tokens_per_chunk must fit the trained speak grammar: "
                "action + lexical tokens + optional turn_eos + chunk_eos; "
                f"minimum={minimum_speak_budget}, "
                f"actual={self.params.max_new_speak_tokens_per_chunk}"
            )
        if not hasattr(bundle.model, "as_duplex"):
            raise RuntimeError("Loaded model does not expose as_duplex()")
        duplex_kwargs = dict(self.params.__dict__)
        duplex_kwargs.pop("decode_mode")
        duplex_kwargs.pop("speak_text_tokens_per_unit")
        duplex_kwargs.pop("talker_speech_tokens_per_unit")
        duplex_kwargs.pop("talker_final_speech_tokens_max")
        duplex_kwargs.pop("max_new_tool_tokens")
        duplex_kwargs.pop("max_tool_response_tokens")
        duplex_kwargs.pop("max_tool_calls_per_unit")
        duplex_kwargs.pop("max_tool_schemas")
        duplex_kwargs.pop("max_tool_schema_tokens")
        duplex_kwargs.pop("inject_search_time_context")
        duplex_kwargs.pop("memory_slate_max_tokens")
        duplex_kwargs.pop("memory_lead_ratio")
        duplex_kwargs.pop("memory_soft_ratio")
        duplex_kwargs.pop("memory_hard_ratio")
        duplex_kwargs.pop("memory_kv_ceiling_units")
        requested_window_mode = duplex_kwargs["sliding_window_mode"]
        if requested_window_mode in {
            CONTEXT_NO_PREVIOUS,
            CONTEXT_MEMORY,
            CONTEXT_SLATE,
        }:
            # Keep upstream context-mode preparation and enforcement dispatch. The decoder patch
            # replaces only the selected context policy; existing modes remain unchanged.
            duplex_kwargs["sliding_window_mode"] = "context"
        if requested_window_mode == CONTEXT_NO_PREVIOUS:
            duplex_kwargs["context_previous_max_tokens"] = 0
        self.duplex = _build_duplex(bundle.model, duplex_kwargs)
        self.tool_schemas: list[dict[str, Any]] = []
        self.pinned_context: PinnedContextController | None = None
        self._configure_control_tokens()
        if requested_window_mode == CONTEXT_NO_PREVIOUS:
            install_context_no_previous(self.duplex.decoder)
        elif requested_window_mode == CONTEXT_MEMORY:
            self.pinned_context = install_context_memory(
                self.duplex.decoder,
                config=PinnedContextConfig(
                    max_units=self.params.context_max_units,
                    max_tokens=self.params.context_previous_max_tokens,
                    slate_max_tokens=self.params.memory_slate_max_tokens,
                    lead_ratio=self.params.memory_lead_ratio,
                    soft_ratio=self.params.memory_soft_ratio,
                    hard_ratio=self.params.memory_hard_ratio,
                    kv_ceiling_units=self.params.memory_kv_ceiling_units,
                ),
            )
        elif requested_window_mode == CONTEXT_SLATE:
            self.pinned_context = install_context_slate(
                self.duplex.decoder,
                config=PinnedContextConfig(
                    max_units=self.params.context_max_units,
                    max_tokens=self.params.context_previous_max_tokens,
                    slate_max_tokens=self.params.memory_slate_max_tokens,
                    lead_ratio=self.params.memory_lead_ratio,
                    soft_ratio=self.params.memory_soft_ratio,
                    hard_ratio=self.params.memory_hard_ratio,
                    kv_ceiling_units=self.params.memory_kv_ceiling_units,
                ),
            )
        if self.params.generate_audio:
            configure_duplex_talker_generation(
                self.duplex,
                speech_tokens_per_unit=self.params.talker_speech_tokens_per_unit,
                final_speech_tokens_max=self.params.talker_final_speech_tokens_max,
            )
        self._streaming_text = DuplexStreamingText(self.duplex, self.bundle.tokenizer)

    def prepare(
        self,
        system_prompt: str | None = None,
        ref_audio_path: str | None = None,
        tools: Sequence[Mapping[str, Any]] | None = None,
    ) -> None:
        self._streaming_text.reset()
        self.tool_schemas = validate_realtime_tool_context(
            tools,
            self.bundle.tokenizer,
            max_tools=self.params.max_tool_schemas,
            max_schema_tokens=self.params.max_tool_schema_tokens,
        )
        base_system_prompt = system_prompt or "Streaming Omni Conversation."
        if self.params.inject_search_time_context and any(
            tool["name"] == "search" for tool in self.tool_schemas
        ):
            now = datetime.now().astimezone().isoformat(timespec="seconds")
            base_system_prompt += (
                f"\nCurrent local time: {now}. Resolve relative dates to absolute dates "
                "when constructing search queries."
            )
        kwargs = {}
        if system_prompt or self.tool_schemas:
            kwargs["prefix_system_prompt"] = system_prompt_with_tools(
                base_system_prompt,
                self.tool_schemas,
            )
        if ref_audio_path:
            kwargs["prompt_wav_path"] = ref_audio_path
        # MiniCPMODuplex.prepare() resets its decoder and mel processor, but the shared base
        # MiniCPMO keeps Whisper's streaming KV in model.audio_past_key_values.  Reusing one base
        # model across serialized web sessions would otherwise let a new session inherit several
        # seconds of the preceding session's audio state.  Reset the base streaming session before
        # prefilling the immutable system/tool prefix.
        _reset_shared_model_streaming_session(self.duplex.model)
        self.duplex.prepare(**kwargs)
        if self.pinned_context is not None:
            self.pinned_context.refresh()

    def capture_prefix_snapshot(self) -> DuplexPrefixSnapshot:
        """Capture the prepared system/tool KV prefix without sharing mutable session state."""

        decoder = self.duplex.decoder
        token_count = int(decoder.get_cache_length())
        if token_count <= 0 or decoder.cache is None:
            raise RuntimeError("cannot cache an empty duplex prefix")
        token2wav_state = None
        if self.params.generate_audio and self.duplex.token2wav_initialized:
            token2wav_state = {
                # prompt_wav=None is valid only while this initialized prompt cache exists.
                "audio_tokenizer_cache": _clone_tensor_tree(
                    self.duplex.model.tts.audio_tokenizer.cache
                ),
                "flow_cache_base": _clone_tensor_tree(self.duplex.flow_cache_base),
                "hift_cache_base": _clone_tensor_tree(self.duplex.hift_cache_base),
                "pre_lookahead": int(self.duplex.pre_lookahead),
            }
        return DuplexPrefixSnapshot(
            decoder_cache=_clone_decoder_cache(decoder.cache),
            decoder_state={
                name: deepcopy(getattr(decoder, name))
                for name in _PREFIX_DECODER_STATE
                if hasattr(decoder, name)
            },
            duplex_state={
                name: deepcopy(getattr(self.duplex, name))
                for name in _PREFIX_DUPLEX_STATE
                if hasattr(self.duplex, name)
            },
            tool_schemas=deepcopy(self.tool_schemas),
            token_count=token_count,
            sliding_window_mode=self.params.sliding_window_mode,
            generate_audio=bool(self.params.generate_audio),
            token2wav_state=token2wav_state,
        )

    def restore_prefix_snapshot(self, snapshot: DuplexPrefixSnapshot) -> None:
        """Reset a fresh runner and seed it from a cached, session-isolated KV prefix."""

        self._streaming_text.reset()

        if snapshot.sliding_window_mode != self.params.sliding_window_mode:
            raise ValueError(
                "prefix snapshot window mode mismatch: "
                f"snapshot={snapshot.sliding_window_mode}, runner={self.params.sliding_window_mode}"
            )
        if snapshot.generate_audio != bool(self.params.generate_audio):
            raise ValueError("prefix snapshot audio mode does not match the runner")

        duplex = self.duplex
        decoder = duplex.decoder
        duplex.clear_break_event()
        duplex.clear_session_stop()
        duplex._reset_streaming_state()
        decoder.reset()
        _reset_shared_model_streaming_session(duplex.model)
        duplex.model.init_streaming_processor()

        decoder.cache = _clone_decoder_cache(snapshot.decoder_cache)
        for name, value in snapshot.decoder_state.items():
            setattr(decoder, name, deepcopy(value))
        for name, value in snapshot.duplex_state.items():
            setattr(duplex, name, deepcopy(value))
        if snapshot.token2wav_state is not None:
            duplex.model.tts.audio_tokenizer.cache = _clone_tensor_tree(
                snapshot.token2wav_state["audio_tokenizer_cache"]
            )
            duplex.flow_cache_base = _clone_tensor_tree(
                snapshot.token2wav_state["flow_cache_base"]
            )
            duplex.hift_cache_base = _clone_tensor_tree(
                snapshot.token2wav_state["hift_cache_base"]
            )
            duplex.pre_lookahead = int(snapshot.token2wav_state["pre_lookahead"])
            duplex.token2wav_initialized = True
            duplex._reset_token2wav_for_new_turn()
        self.tool_schemas = deepcopy(snapshot.tool_schemas)

        restored_count = int(decoder.get_cache_length())
        if restored_count != snapshot.token_count:
            raise RuntimeError(
                "restored duplex prefix has the wrong length: "
                f"expected={snapshot.token_count}, actual={restored_count}"
            )
        if self.pinned_context is not None:
            self.pinned_context.refresh()

    def add_memory_episode(self, episode: Mapping[str, Any] | Any) -> bool:
        if self.pinned_context is None or not self.pinned_context.allow_memory:
            raise RuntimeError("memory episodes require sliding_window_mode='context_memory'")
        return self.pinned_context.add_episode(episode)

    def set_task_slate(self, slate: str) -> bool:
        if self.pinned_context is None:
            raise RuntimeError(
                "task slate requires sliding_window_mode='context_slate' or 'context_memory'"
            )
        return self.pinned_context.set_slate(slate)

    def set_summary_needed_callback(
        self,
        callback: Callable[[dict[str, Any]], None] | None,
    ) -> None:
        if self.pinned_context is None:
            raise RuntimeError("summary callbacks require sliding_window_mode='context_memory'")
        if not self.pinned_context.allow_memory:
            self.pinned_context.set_summary_needed_callback(None)
            return
        self.pinned_context.set_summary_needed_callback(callback)

    def record_unit_time(self, start_sec: float, end_sec: float) -> None:
        if self.pinned_context is not None:
            self.pinned_context.record_last_unit_time(start_sec, end_sec)

    def streaming_generate(self, **kwargs):
        """Generate one unit after activating this session's shared Talker wrapper."""
        tts = getattr(self.duplex.model, "tts", None)
        if tts is not None and hasattr(tts, "_mcpmft_duplex_owner"):
            tts._mcpmft_duplex_owner = self.duplex
        self._streaming_text.begin_unit()
        before = len(self.duplex.total_ids)
        output = dict(self._streaming_generate_with_tools(**kwargs))
        stable_text = self._streaming_text.take_unit_text()
        if stable_text is not None:
            output["text"] = stable_text
        generated_token_ids = list(self.duplex.total_ids[before:])
        output["generated_token_ids"] = generated_token_ids
        if self.interrupt_token_id in generated_token_ids:
            self._reset_after_interrupt()
            output.update(
                {
                    "is_listen": True,
                    "is_interrupt": True,
                    "text": "",
                    "audio_waveform": None,
                    # An interrupt closes the in-flight assistant turn, but it must not make
                    # a file/live runner stop before the user's correction is complete.
                    "end_of_turn": False,
                    "n_tts_tokens": 0,
                }
            )
        if not self.params.generate_audio:
            output["audio_waveform"] = None
        return output

    def reset_streaming_text(self) -> None:
        self._streaming_text.reset()

    def validate_tool_response(self, response: Any) -> str:
        """Format and bound a tool result before it enters the live media queue."""
        formatted = format_tool_response(response)
        token_ids = self.bundle.tokenizer.encode(formatted, add_special_tokens=False)
        limit = int(self.params.max_tool_response_tokens)
        if len(token_ids) > limit:
            raise ToolProtocolError(
                f"tool response has {len(token_ids)} tokens; maximum is {limit}"
            )
        return formatted

    def prefill_tool_response(
        self,
        response: Any,
        *,
        audio_waveform: Any,
        text_list: Sequence[str] | None = None,
        **prefill_kwargs: Any,
    ) -> dict[str, Any]:
        """Inject one bounded tool result after media in the same duplex input unit."""
        formatted = self.validate_tool_response(response)
        if audio_waveform is None:
            raise ToolProtocolError(
                "tool response prefill requires the unit's always-on microphone audio"
            )
        text_content = "\n".join([*(text_list or ()), formatted])
        text_token_ids = self.bundle.tokenizer.encode(
            text_content,
            add_special_tokens=False,
        )
        result = self.duplex.streaming_prefill(
            audio_waveform=audio_waveform,
            **prefill_kwargs,
        )
        if isinstance(result, dict) and not result.get("success", True):
            raise RuntimeError(
                f"duplex rejected tool response: {result.get('reason') or 'unknown reason'}"
            )

        # MiniCPM-o 4.5 accepts audio_waveform + text_list, but its mixed-mode path keeps
        # pending_logits from the audio position after feeding the text without return_logits.
        # Feed the bounded text into the already-open unit ourselves and refresh logits at the
        # actual final input position. This preserves causal audio -> tool-response -> action order.
        decoder = getattr(self.duplex, "decoder", None)
        if decoder is None:
            raise RuntimeError("duplex tool response prefill requires decoder access")
        if hasattr(decoder, "embed_tokens"):
            text_embeds = decoder.embed_tokens(text_token_ids)
        else:
            import torch

            text_embeds = torch.cat(
                [decoder.embed_token(token_id) for token_id in text_token_ids],
                dim=0,
            )
        self.duplex.pending_logits, _ = decoder.feed(
            text_embeds,
            return_logits=True,
        )
        schema = getattr(self.duplex, "prefill_schema_tokens", None)
        if isinstance(schema, list) and schema and isinstance(schema[-1], list):
            schema[-1].extend(text_token_ids)
        return result

    def _streaming_generate_with_tools(self, **kwargs):
        duplex = self.duplex
        decoder = getattr(duplex, "decoder", None)
        pending_logits = getattr(duplex, "pending_logits", None)
        if (
            decoder is None
            or pending_logits is None
            or getattr(self, "tool_call_start_token_id", None) is None
            or self._must_force_listen()
            or _runtime_stop_requested(duplex)
        ):
            return duplex.streaming_generate(**kwargs)

        decode_kwargs = _decoder_kwargs(kwargs)
        import torch

        first_logits = _mask_generation_logits(
            pending_logits,
            allowed=self._allowed_first_action_token_ids(),
        )
        with torch.no_grad():
            first_token = decoder.decode(logits=first_logits, **decode_kwargs)
        if int(first_token.item()) == self.tool_call_start_token_id:
            return self._generate_tool_action(first_token, pending_logits, decode_kwargs)

        original_decode = decoder.decode
        replayed = False
        content_tokens = 0
        turn_closed = False
        first_token_id = int(first_token.item())
        speak_action = first_token_id in getattr(self, "speak_action_token_ids", ())

        def replay_first_token(*args, **decode_call_kwargs):
            nonlocal replayed, content_tokens, turn_closed
            if not replayed:
                replayed = True
                return first_token
            if turn_closed:
                return first_token.new_full(first_token.shape, self.chunk_eos_token_id)

            logits, args, decode_call_kwargs = _extract_decode_logits(
                args, decode_call_kwargs
            )
            if speak_action and content_tokens >= self.params.speak_text_tokens_per_unit:
                constrained = _mask_generation_logits(
                    logits,
                    allowed=(self.chunk_eos_token_id, self.turn_eos_token_id),
                )
            else:
                constrained = _mask_generation_logits(
                    logits,
                    forbidden=getattr(self, "dialogue_continuation_forbidden_token_ids", ()),
                )
            token = original_decode(
                *args,
                logits=constrained,
                **decode_call_kwargs,
            )
            token_id = int(token.item())
            if speak_action:
                if token_id == self.turn_eos_token_id:
                    turn_closed = True
                elif token_id != self.chunk_eos_token_id:
                    content_tokens += 1
            return token

        decoder.decode = replay_first_token
        try:
            return duplex.streaming_generate(**kwargs)
        finally:
            decoder.decode = original_decode

    def _generate_tool_action(self, first_token, pending_logits, decode_kwargs):
        """Decode a complete silent tool unit without ever entering the Talker path."""
        import torch

        duplex = self.duplex
        decoder = duplex.decoder
        start_time = time.time()
        llm_start_time = start_time
        current_time = getattr(duplex, "audio_chunk_idx", None)
        duplex.pending_logits = None
        duplex._streaming_generate_count += 1

        generated: list[int] = []
        logits = pending_logits
        token = first_token
        ended = False
        budget = max(4, int(self.params.max_new_tool_tokens))
        # The upstream decoder's bad-token list is a speech-generation constraint. Tool JSON is
        # native text and must be able to follow the exact tokenization used by training. This
        # scope changes no protocol grammar and deliberately permits multiple ordered tool calls.
        with _allow_lexical_tool_tokens(decoder, self.bundle.tokenizer):
            with torch.no_grad():
                for index in range(budget):
                    if index:
                        constrained = _mask_generation_logits(
                            logits,
                            forbidden=getattr(
                                self, "tool_generation_forbidden_token_ids", ()
                            ),
                        )
                        token = decoder.decode(logits=constrained, **decode_kwargs)
                    token_id = int(token.item())
                    generated.append(token_id)
                    duplex.total_ids.append(token_id)
                    logits, _hidden = decoder.feed(
                        decoder.embed_token(token_id),
                        return_logits=True,
                    )
                    if token_id == self.chunk_eos_token_id:
                        ended = True
                        break

                if not ended:
                    generated.append(self.chunk_eos_token_id)
                    duplex.total_ids.append(self.chunk_eos_token_id)
                    decoder.feed(decoder.embed_token(self.chunk_eos_token_id))

                decoder.feed(decoder.embed_token(self.unit_end_token_id))
                duplex.total_ids.append(self.unit_end_token_id)

        payload_ids = [
            token_id for token_id in generated if token_id != self.chunk_eos_token_id
        ]
        raw_text = _decode_raw(self.bundle.tokenizer, payload_ids)
        calls: list[dict[str, Any]] = []
        error = None
        parse_valid = False
        schema_valid = False
        if not ended:
            error = (
                f"tool call exceeded max_new_tool_tokens={self.params.max_new_tool_tokens}"
            )
        else:
            try:
                parsed = parse_tool_calls(raw_text)
            except ToolProtocolError as exc:
                error = str(exc)
            else:
                parse_valid = True
                validation = validate_tool_calls(
                    parsed,
                    getattr(self, "tool_schemas", ()),
                    max_calls=self.params.max_tool_calls_per_unit,
                )
                calls = list(validation.calls)
                error = validation.error
                schema_valid = validation.error is None

        duplex.current_turn_ended = True
        if isinstance(getattr(duplex, "total_hidden", None), list):
            duplex.total_hidden.append([])
        self._register_tool_unit(payload_ids, raw_text)
        llm_end_time = time.time()
        return {
            "is_listen": False,
            "is_tool_call": True,
            "tool_calls": calls,
            "tool_error": error,
            "raw_tool_text": raw_text if error else "",
            "tool_generation_complete": ended,
            "tool_parse_valid": parse_valid,
            "tool_schema_valid": schema_valid,
            "text": "",
            "audio_waveform": None,
            "end_of_turn": False,
            "current_time": current_time,
            "cost_llm": llm_end_time - llm_start_time,
            "cost_tts_prep": 0.0,
            "cost_tts": 0.0,
            "cost_token2wav": 0.0,
            "cost_all": time.time() - start_time,
            "n_tokens": len(generated),
            "n_tts_tokens": 0,
        }

    def _register_tool_unit(self, generated_ids: list[int], raw_text: str) -> None:
        decoder = self.duplex.decoder
        register = getattr(decoder, "register_unit_end", None)
        if callable(register):
            register(
                input_type=(
                    self.duplex.current_mode.lower()
                    if getattr(self.duplex, "current_mode", None)
                    else "audio"
                ),
                generated_tokens=generated_ids,
                is_listen=True,
                generated_text=raw_text,
            )
        config = getattr(decoder, "_window_config", None)
        mode = getattr(config, "sliding_window_mode", "off")
        if mode == "context":
            decoder.enforce_window_with_context()
        elif mode == "basic":
            decoder.enforce_window()

    def _must_force_listen(self) -> bool:
        return int(getattr(self.duplex, "_streaming_generate_count", 0)) < int(
            getattr(self.duplex, "force_listen_count", 0)
        )

    def take_talker_condition(self) -> tuple[tuple[int, ...], object] | None:
        """Move the latest unit's Talker condition off the Thinker device."""
        total_hidden = getattr(self.duplex, "total_hidden", None)
        if not isinstance(total_hidden, list) or not total_hidden:
            return None
        unit = total_hidden.pop()
        if not unit:
            return None

        import torch

        token_ids = tuple(int(item[0]) for item in unit)
        hidden = torch.cat([item[1].detach().squeeze(0) for item in unit], dim=0)
        return token_ids, hidden.to(device="cpu", copy=True)

    def _configure_control_tokens(self) -> None:
        """Install Gander's native action and tool grammar on MiniCPM-o 4.5."""

        tokenizer = self.bundle.tokenizer
        unk_id = tokenizer.unk_token_id

        def required(text: str) -> int:
            value = tokenizer.convert_tokens_to_ids(text)
            if value is None or value < 0 or value == unk_id:
                raise RuntimeError(f"required realtime token is missing: {text}")
            return int(value)

        self.tool_call_start_token_id = required("<tool_call>")
        self.tool_call_end_token_id = required("</tool_call>")
        self.tool_response_start_token_id = required("<tool_response>")
        self.tool_response_end_token_id = required("</tool_response>")
        self.chunk_eos_token_id = required("<|chunk_eos|>")
        self.turn_eos_token_id = required("<|turn_eos|>")
        self.unit_end_token_id = required("</unit>")
        listen_id = required("<|listen|>")
        speak_id = required("<|speak|>")
        self.interrupt_token_id = required("<|interrupt|>")
        backchannel_id = required("<|backchannel|>")

        if backchannel_id not in self.duplex.chunk_speak_token_ids:
            self.duplex.chunk_speak_token_ids.append(backchannel_id)
        action_ids = (listen_id, speak_id, self.interrupt_token_id, backchannel_id)
        protocol_ids = (
            self.tool_call_start_token_id,
            self.tool_call_end_token_id,
            self.tool_response_start_token_id,
            self.tool_response_end_token_id,
        )
        self.speak_action_token_ids = (speak_id, backchannel_id)
        self.dialogue_continuation_forbidden_token_ids = tuple(
            dict.fromkeys((*action_ids, *protocol_ids))
        )
        self.tool_generation_forbidden_token_ids = tuple(
            dict.fromkeys(
                (
                    *action_ids,
                    self.tool_response_start_token_id,
                    self.tool_response_end_token_id,
                    self.turn_eos_token_id,
                )
            )
        )

        self.duplex.interrupt_token_id = self.interrupt_token_id
        self.duplex._mcpmft_unit_end_token_id = self.unit_end_token_id
        if self.interrupt_token_id not in self.duplex.chunk_terminator_token_ids:
            self.duplex.chunk_terminator_token_ids.append(self.interrupt_token_id)

    def _allowed_first_action_token_ids(self) -> tuple[int, ...]:
        ids = list(self.dialogue_continuation_forbidden_token_ids)
        protocol_ids = {
            self.tool_call_end_token_id,
            self.tool_response_start_token_id,
            self.tool_response_end_token_id,
        }
        ids = [token_id for token_id in ids if token_id not in protocol_ids]
        if not self.tool_schemas:
            ids = [token_id for token_id in ids if token_id != self.tool_call_start_token_id]
        return tuple(dict.fromkeys(ids))

    def _reset_after_interrupt(self) -> None:
        """Discard Talker state belonging to the assistant turn that was just interrupted."""
        self._streaming_text.reset()
        duplex = self.duplex
        duplex.current_turn_ended = True
        for name, value in (
            ("tts_text_start_pos", 0),
            ("tts_past_key_values", None),
            ("tts_current_turn_start_time", None),
        ):
            setattr(duplex, name, value)
        duplex._reset_token2wav_for_new_turn()
        if duplex.decoder._unit_history:
            # Interrupt is a control-only unit and should not retain assistant content.
            duplex.decoder._unit_history[-1]["is_listen"] = True

    def run_audio_file(
        self,
        audio_path: str | Path,
        *,
        output_wav: str | None = None,
        trailing_silence_sec: float = 20.0,
        stop_on_turn_end: bool = True,
    ) -> list[dict]:
        """Stream a mono audio file through the duplex model, 1s at a time.

        Training uses an ALWAYS-ON mic: after the user stops talking, the serializer keeps emitting
        1s SILENCE audio blocks for the whole answer, and the model is supervised to <|speak|> over
        them. So at inference we must ALSO keep feeding silence after the file's audio ends —
        otherwise a long reply (or any reply that only starts once the user finishes) is truncated,
        because streaming_generate has no further chunk to hang the speak units on (see the old
        EOF-only loop). We feed up to `trailing_silence_sec` of silence and stop early once the model
        signals end_of_turn (after having spoken), matching a single completed response.
        """
        import numpy as np

        chunk_samples = int(16000 * self.params.chunk_ms / 1000)
        silence = np.zeros(chunk_samples, dtype=np.float32)

        outputs: list[dict] = []
        wav_chunks = []
        spoke_any = False

        def step(chunk):
            nonlocal spoke_any
            self.duplex.streaming_prefill(audio_waveform=chunk)
            out = self.streaming_generate(
                max_new_speak_tokens_per_chunk=self.params.max_new_speak_tokens_per_chunk,
                decode_mode=self.params.decode_mode,
                temperature=self.params.temperature,
                top_p=self.params.top_p,
                top_k=self.params.top_k,
            )
            outputs.append(out)
            if isinstance(out, dict):
                if out.get("audio_waveform") is not None:
                    wav_chunks.append(out["audio_waveform"])
                if out.get("is_listen") is False:
                    spoke_any = True
            return out

        # 1) real audio from the file
        for chunk in iter_audio_chunks(audio_path, chunk_ms=self.params.chunk_ms):
            step(chunk)

        # 2) trailing always-on-mic silence so the answer can complete (matches training)
        max_silence_chunks = int(round(trailing_silence_sec * 1000 / self.params.chunk_ms))
        for _ in range(max_silence_chunks):
            out = step(silence)
            if stop_on_turn_end and spoke_any and isinstance(out, dict) and out.get("end_of_turn"):
                break

        if output_wav and wav_chunks:
            save_wav(output_wav, np.concatenate(wav_chunks))
        return outputs


def _runtime_stop_requested(duplex: Any) -> bool:
    return bool(duplex.is_session_stop_set() or duplex.is_break_set())


def _reset_shared_model_streaming_session(model: Any) -> None:
    """Clear base-model streaming caches that are not owned by MiniCPMODuplex."""
    model.reset_session(reset_token2wav_cache=False)


def _decoder_kwargs(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "mode": kwargs.get("decode_mode", "sampling"),
        "temperature": kwargs.get("temperature", 0.7),
        "top_k": kwargs.get("top_k", 100),
        "top_p": kwargs.get("top_p", 0.8),
        "listen_top_k": kwargs.get("listen_top_k"),
        "listen_prob_scale": kwargs.get("listen_prob_scale", 1.0),
        "text_repetition_penalty": kwargs.get("text_repetition_penalty", 1.05),
        "text_repetition_window_size": kwargs.get("text_repetition_window_size", 512),
    }


def _decode_raw(tokenizer: Any, token_ids: Sequence[int]) -> str:
    return tokenizer.decode(
        token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def _build_duplex(model, duplex_kwargs: dict):
    """Construct upstream duplex without loading an unused token2wav stack in text-only mode."""
    if duplex_kwargs.get("generate_audio", True):
        return model.as_duplex(**duplex_kwargs)
    with _AS_DUPLEX_LOCK:
        model.init_tts = lambda *args, **kwargs: None
        try:
            return model.as_duplex(**duplex_kwargs)
        finally:
            delattr(model, "init_tts")


def iter_audio_chunks(
    audio_path: str | Path,
    *,
    chunk_ms: int = 1000,
    sample_rate: int = 16000,
) -> Iterable:
    import librosa
    import numpy as np

    audio, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
    chunk_samples = int(sample_rate * chunk_ms / 1000)
    for start in range(0, len(audio), chunk_samples):
        chunk = audio[start : start + chunk_samples]
        if len(chunk) < chunk_samples:
            chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
        yield chunk.astype(np.float32, copy=False)


def configure_duplex_talker_generation(
    duplex: Any,
    *,
    speech_tokens_per_unit: int,
    final_speech_tokens_max: int,
) -> None:
    """Apply the K-text/S-S3 streaming cadence without capping the final speech remainder.

    MiniCPM-o 4.5 intentionally withholds its newest prediction, so requesting N visible codes
    requires ``max_new_token=N+1``. Non-final units expose exactly the configured S3 count. Once
    Thinker emits turn EOS, Talker may emit zero or more remaining codes until S3 EOS; the only
    implicit bound is its remaining positional context. A positive final cap remains available as
    an operational limit, but is not part of the training unit contract.
    """
    unit_codes = int(speech_tokens_per_unit)
    final_codes = int(final_speech_tokens_max)
    if unit_codes <= 0:
        raise ValueError("talker_speech_tokens_per_unit must be positive")
    if final_codes < 0:
        raise ValueError("talker_final_speech_tokens_max must be non-negative")

    tts = duplex.model.tts
    tts._mcpmft_duplex_owner = duplex
    tts._mcpmft_speech_tokens_per_unit = unit_codes
    tts._mcpmft_final_speech_tokens_max = final_codes
    if hasattr(tts, "_mcpmft_original_generate_chunk"):
        return

    original = tts.generate_chunk
    tts._mcpmft_original_generate_chunk = original

    @wraps(original)
    def generate_fixed_s3_chunk(*args, **kwargs):
        owner = tts._mcpmft_duplex_owner
        total_ids = owner.total_ids
        if len(total_ids) >= 2 and total_ids[-2:] == [
            owner.interrupt_token_id,
            owner._mcpmft_unit_end_token_id,
        ]:
            # Interrupt is a control-only unit and must not advance Talker/token2wav state.
            import torch

            inputs_embeds = kwargs.get("inputs_embeds")
            device = inputs_embeds.device if inputs_embeds is not None else None
            empty_tokens = torch.empty((1, 0), dtype=torch.long, device=device)
            return empty_tokens, kwargs.get("past_key_values")

        current_unit_codes = int(tts._mcpmft_speech_tokens_per_unit)
        is_final = bool(owner.current_turn_ended)
        context_limit = _talker_context_limit(tts)
        text_start_pos = int(kwargs.get("text_start_pos") or 0)
        inputs_embeds = kwargs.get("inputs_embeds")
        condition_length = int(inputs_embeds.shape[1]) if inputs_embeds is not None else 0
        remaining_context = context_limit - text_start_pos - condition_length
        if remaining_context <= 0:
            raise RuntimeError(
                "Talker has no positional context left for this S3 unit: "
                f"max_positions={context_limit} text_start_pos={text_start_pos} "
                f"condition_length={condition_length}"
            )

        visible_budget = current_unit_codes
        if is_final:
            configured_cap = int(tts._mcpmft_final_speech_tokens_max)
            visible_budget = (
                min(configured_cap, remaining_context)
                if configured_cap
                else remaining_context
            )
        elif remaining_context < current_unit_codes:
            raise RuntimeError(
                "Talker cannot fit the fixed non-final S3 unit in its remaining context: "
                f"required_visible_codes={current_unit_codes} "
                f"remaining_context={remaining_context} "
                f"max_positions={context_limit} text_start_pos={text_start_pos} "
                f"condition_length={condition_length}"
            )
        kwargs["max_new_token"] = visible_budget + 1
        kwargs["min_new_tokens"] = 0 if is_final else current_unit_codes + 1
        return original(*args, **kwargs)

    tts.generate_chunk = generate_fixed_s3_chunk


def _talker_context_limit(tts) -> int:
    return int(tts.config.max_position_embeddings)
