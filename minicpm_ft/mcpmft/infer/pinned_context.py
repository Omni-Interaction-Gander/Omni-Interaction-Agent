from __future__ import annotations

import json
import math
import threading
from dataclasses import dataclass
from types import MethodType
from typing import Any, Callable, Mapping


CONTEXT_MEMORY = "context_memory"
CONTEXT_SLATE = "context_slate"


def render_slate_context(slate: str) -> str:
    normalized = str(slate or "").strip()
    return f"\n[SLATE]\n{normalized}\n" if normalized else ""


@dataclass(frozen=True)
class MemoryEpisode:
    key: str
    start_sec: float
    end_sec: float
    event_summary: str

    @classmethod
    def from_value(cls, value: Mapping[str, Any] | Any) -> "MemoryEpisode":
        def field(name: str, default: Any = None) -> Any:
            return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)

        start_sec = float(field("start_sec"))
        end_sec = float(field("end_sec"))
        if not math.isfinite(start_sec) or not math.isfinite(end_sec) or end_sec <= start_sec:
            raise ValueError("memory episode requires finite start_sec < end_sec")
        summary = field("event_summary")
        if summary is None:
            summary = _event_summary(field("summary", ""))
        summary = str(summary or "").strip()
        if not summary:
            raise ValueError("memory episode event_summary is empty")
        index = field("episode_index")
        key = str(field("key") or (f"episode:{index}" if index is not None else f"{start_sec:g}:{end_sec:g}"))
        return cls(key=key, start_sec=start_sec, end_sec=end_sec, event_summary=summary)


@dataclass(frozen=True)
class PinnedContextConfig:
    max_units: int = 128
    max_tokens: int = 1500
    slate_max_tokens: int = 256
    lead_ratio: float = 0.7
    soft_ratio: float = 0.9
    hard_ratio: float = 1.0
    kv_ceiling_units: int | None = None

    def __post_init__(self) -> None:
        if self.max_units <= 0 or self.max_tokens <= 0 or self.slate_max_tokens <= 0:
            raise ValueError("pinned context unit/token budgets must be positive")
        if not 0 < self.lead_ratio <= self.soft_ratio <= self.hard_ratio:
            raise ValueError("pinned context ratios must satisfy 0 < lead <= soft <= hard")
        if self.hard_units < self.soft_units:
            raise ValueError("pinned context hard threshold must not precede soft")
        if self.ceiling_units < self.hard_units:
            raise ValueError("pinned context KV ceiling must be >= hard threshold")

    @property
    def lead_units(self) -> int:
        return max(1, math.ceil(self.max_units * self.lead_ratio))

    @property
    def soft_units(self) -> int:
        return max(self.lead_units, math.ceil(self.max_units * self.soft_ratio))

    @property
    def hard_units(self) -> int:
        return max(self.soft_units, math.ceil(self.max_units * self.hard_ratio))

    @property
    def ceiling_units(self) -> int:
        return self.kv_ceiling_units or max(self.hard_units + 1, math.ceil(self.max_units * 1.125))


@dataclass(frozen=True)
class _PinnedRecord:
    key: str
    start_sec: float
    end_sec: float
    text: str
    raw_capture: bool = False


class PinnedContextController:
    """Own the protected memory/slate prefix for one StreamDecoder."""

    def __init__(
        self,
        decoder: Any,
        config: PinnedContextConfig,
        *,
        mode: str = CONTEXT_MEMORY,
        allow_memory: bool = True,
        on_summary_needed: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.decoder = decoder
        self.config = config
        self.mode = mode
        self.allow_memory = bool(allow_memory)
        self.on_summary_needed = on_summary_needed if self.allow_memory else None
        self._episodes: dict[str, MemoryEpisode] = {}
        self._rolling: list[_PinnedRecord] = []
        self._slate = ""
        self._last_tokens: list[int] = []
        self._requested_oldest_unit: int | None = None
        self._dropped_until_sec = 0.0
        self._emergency_captures = 0
        self._blocked_evictions = 0
        self._lock = threading.RLock()

    def reset_runtime(self) -> None:
        with self._lock:
            self._episodes.clear()
            self._rolling.clear()
            self._slate = ""
            self._last_tokens = []
            self._requested_oldest_unit = None
            self._dropped_until_sec = 0.0
            self._emergency_captures = 0
            self._blocked_evictions = 0

    def record_last_unit_time(self, start_sec: float, end_sec: float) -> None:
        if end_sec < start_sec:
            raise ValueError("unit end time must not precede start time")
        with self._lock:
            history = getattr(self.decoder, "_unit_history", ())
            for entry in reversed(history):
                if "start_sec" not in entry and entry.get("type") != "system":
                    entry["start_sec"] = float(start_sec)
                    entry["end_sec"] = float(end_sec)
                    return

    def add_episode(self, value: Mapping[str, Any] | Any) -> bool:
        if not self.allow_memory:
            raise RuntimeError("memory episodes are disabled in context_slate mode")
        episode = MemoryEpisode.from_value(value)
        with self._lock:
            previous = self._episodes.get(episode.key)
            if previous == episode:
                return False
            self._episodes[episode.key] = episode
            if episode.end_sec <= self._dropped_until_sec:
                self._replace_captures_with_episode(episode)
                return self.refresh()
            return self.enforce()

    def set_summary_needed_callback(
        self,
        callback: Callable[[dict[str, Any]], None] | None,
    ) -> None:
        with self._lock:
            self.on_summary_needed = callback if self.allow_memory else None

    def set_slate(self, slate: str) -> bool:
        normalized = str(slate or "").strip()
        with self._lock:
            if normalized == self._slate:
                return False
            slate_tokens = self._encode(self._render_slate(normalized)) if normalized else []
            if len(slate_tokens) > min(self.config.slate_max_tokens, self.config.max_tokens):
                raise ValueError(
                    f"task slate has {len(slate_tokens)} tokens; maximum is "
                    f"{min(self.config.slate_max_tokens, self.config.max_tokens)}"
                )
            self._slate = normalized
            return self.refresh()

    def enforce(self) -> bool:
        with self._lock:
            history = self._ordinary_history()
            if not self.allow_memory:
                if len(history) <= self.config.max_units:
                    return False
                overflow = len(history) - self.config.max_units
                return self._drop_segment(
                    history[:overflow],
                    episode=None,
                    capture_raw=False,
                )
            count = len(history)
            if count >= self.config.lead_units:
                self._request_summary(history[0])
            if count <= self.config.soft_units:
                return False

            changed = False
            while len(self._ordinary_history()) > self.config.soft_units:
                history = self._ordinary_history()
                episode = self._episode_covering(history[0])
                if episode is not None:
                    segment = [
                        entry
                        for entry in history
                        if float(entry.get("end_sec", math.inf)) <= episode.end_sec
                    ]
                    if segment and self._drop_segment(segment, episode=episode):
                        changed = True
                        continue
                if len(history) <= self.config.ceiling_units:
                    break
                overflow = max(1, len(history) - self.config.hard_units)
                if not self._drop_segment(history[:overflow], episode=None):
                    self._blocked_evictions += 1
                    break
                changed = True
            return changed

    def refresh(self) -> bool:
        with self._lock:
            tokens, rendered = self._render_tokens()
            if tokens == self._last_tokens:
                return False
            if getattr(self.decoder, "cache", None) is None:
                return False
            if int(getattr(self.decoder, "_position_offset", 0)) != 0:
                raise RuntimeError(
                    f"{self.mode} requires compressed context positions (offset=0)"
                )
            if not self.decoder._rebuild_cache_with_previous(tokens):
                raise RuntimeError("failed to refresh the protected memory/slate cache block")
            self.decoder._previous_token_ids = list(tokens)
            self.decoder._previous_text = rendered
            self.decoder._has_previous = bool(tokens)
            self._last_tokens = list(tokens)
            return True

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "mode": self.mode,
                "lead_units": self.config.lead_units,
                "soft_units": self.config.soft_units,
                "hard_units": self.config.hard_units,
                "kv_ceiling_units": self.config.ceiling_units,
                "rolling_segments": len(self._rolling),
                "ready_episodes": len(self._episodes),
                "pinned_tokens": len(self._last_tokens),
                "pinned_token_budget": self.config.max_tokens,
                "pinned_token_overflow": max(0, len(self._last_tokens) - self.config.max_tokens),
                "emergency_captures": self._emergency_captures,
                "blocked_evictions": self._blocked_evictions,
                "dropped_until_sec": self._dropped_until_sec,
            }

    def _ordinary_history(self) -> list[dict[str, Any]]:
        return [
            entry
            for entry in getattr(self.decoder, "_unit_history", ())
            if entry.get("type") != "system"
        ]

    def _request_summary(self, oldest: Mapping[str, Any]) -> None:
        unit_id = int(oldest.get("unit_id", -1))
        if unit_id == self._requested_oldest_unit:
            return
        self._requested_oldest_unit = unit_id
        if self.on_summary_needed is not None:
            self.on_summary_needed(
                {
                    "unit_id": unit_id,
                    "start_sec": oldest.get("start_sec"),
                    "end_sec": oldest.get("end_sec"),
                }
            )

    def _episode_covering(self, oldest: Mapping[str, Any]) -> MemoryEpisode | None:
        start_sec = oldest.get("start_sec")
        end_sec = oldest.get("end_sec")
        if start_sec is None or end_sec is None:
            return None
        start = float(start_sec)
        end = float(end_sec)
        candidates = [
            episode
            for episode in self._episodes.values()
            if episode.start_sec <= start and episode.end_sec >= end
        ]
        return min(candidates, key=lambda item: (item.end_sec, item.start_sec, item.key)) if candidates else None

    def _drop_segment(
        self,
        entries: list[dict[str, Any]],
        *,
        episode: MemoryEpisode | None,
        capture_raw: bool = True,
    ) -> bool:
        if not entries:
            return False
        if episode is None:
            if capture_raw:
                text, tokens = self.decoder._extract_generated_text(entries)
                captured = text.strip()
                if not captured and tokens:
                    captured = self.decoder.tokenizer.decode(
                        tokens,
                        skip_special_tokens=True,
                    ).strip()
                if not captured:
                    return False
                start_sec = float(entries[0].get("start_sec", self._dropped_until_sec))
                end_sec = float(entries[-1].get("end_sec", start_sec))
                record = _PinnedRecord(
                    key=f"capture:{entries[0].get('unit_id')}:{entries[-1].get('unit_id')}",
                    start_sec=start_sec,
                    end_sec=end_sec,
                    text=captured,
                    raw_capture=True,
                )
                self._rolling.append(record)
                self._emergency_captures += 1
        else:
            record = _PinnedRecord(
                key=episode.key,
                start_sec=episode.start_sec,
                end_sec=episode.end_sec,
                text=episode.event_summary,
            )
            self._rolling = [item for item in self._rolling if item.key != record.key]
            self._rolling.append(record)
            self._episodes.pop(episode.key, None)

        unit_ids = {entry.get("unit_id") for entry in entries}
        self.decoder._unit_history = [
            entry
            for entry in getattr(self.decoder, "_unit_history", ())
            if entry.get("unit_id") not in unit_ids
        ]
        self._dropped_until_sec = max(
            self._dropped_until_sec,
            max(float(entry.get("end_sec", self._dropped_until_sec)) for entry in entries),
        )
        self._requested_oldest_unit = None
        self.refresh()
        self.decoder._sliding_event_count = int(
            getattr(self.decoder, "_sliding_event_count", 0)
        ) + 1
        self.decoder._total_dropped_units = int(
            getattr(self.decoder, "_total_dropped_units", 0)
        ) + len(unit_ids)
        self.decoder._total_dropped_tokens = int(
            getattr(self.decoder, "_total_dropped_tokens", 0)
        ) + sum(int(entry.get("length", 0)) for entry in entries)
        return True

    def _replace_captures_with_episode(self, episode: MemoryEpisode) -> None:
        self._rolling = [
            item
            for item in self._rolling
            if not (
                item.raw_capture
                and item.start_sec >= episode.start_sec
                and item.end_sec <= episode.end_sec
            )
            and item.key != episode.key
        ]
        self._rolling.append(
            _PinnedRecord(
                key=episode.key,
                start_sec=episode.start_sec,
                end_sec=episode.end_sec,
                text=episode.event_summary,
            )
        )
        self._episodes.pop(episode.key, None)

    def _render_tokens(self) -> tuple[list[int], str]:
        slate_text = self._render_slate(self._slate) if self._slate else ""
        slate_tokens = self._encode(slate_text)
        memory_header = "\n\n[MEMORY]\n"
        header_tokens = self._encode(memory_header)
        available = max(0, self.config.max_tokens - len(slate_tokens) - len(header_tokens))

        captures = [item for item in self._rolling if item.raw_capture]
        summaries = [item for item in self._rolling if not item.raw_capture]
        chosen: list[_PinnedRecord] = []
        used = 0
        for item in captures:
            size = len(self._encode(self._render_record(item)))
            chosen.append(item)
            used += size
        for item in reversed(summaries):
            size = len(self._encode(self._render_record(item)))
            if used + size > available:
                continue
            chosen.append(item)
            used += size
        chosen.sort(key=lambda item: (item.end_sec, item.start_sec, item.key))

        parts = [memory_header] if chosen else []
        parts.extend(self._render_record(item) for item in chosen)
        if slate_text:
            parts.append(slate_text)
        rendered = "".join(parts)
        return self._encode(rendered), rendered

    @staticmethod
    def _render_record(item: _PinnedRecord) -> str:
        kind = "RAW" if item.raw_capture else "EPISODE"
        return f"- {kind} {item.start_sec:.1f}-{item.end_sec:.1f}s: {item.text}\n"

    @staticmethod
    def _render_slate(slate: str) -> str:
        return render_slate_context(slate)

    def _encode(self, text: str) -> list[int]:
        return list(self.decoder.tokenizer.encode(text, add_special_tokens=False)) if text else []


def _install_pinned_context(
    decoder: Any,
    *,
    config: PinnedContextConfig,
    mode: str,
    allow_memory: bool,
    on_summary_needed: Callable[[dict[str, Any]], None] | None = None,
) -> PinnedContextController:
    existing = getattr(decoder, "_pinned_context_controller", None)
    if existing is not None:
        if existing.mode != mode:
            raise RuntimeError(
                f"pinned context already installed as {existing.mode}, requested {mode}"
            )
        return existing
    window_config = getattr(decoder, "_window_config", None)
    if getattr(window_config, "sliding_window_mode", None) != "context":
        raise ValueError("pinned context requires the decoder's internal context mode")
    required = ("_rebuild_cache_with_previous", "_extract_generated_text")
    missing = [name for name in required if not callable(getattr(decoder, name, None))]
    if missing:
        raise RuntimeError(f"StreamDecoder lacks pinned context hooks: {missing}")

    controller = PinnedContextController(
        decoder,
        config,
        mode=mode,
        allow_memory=allow_memory,
        on_summary_needed=on_summary_needed,
    )
    base_reset = decoder.reset
    base_stats = getattr(decoder, "get_window_stats", None)

    def enforce_window_with_context(self) -> bool:
        return controller.enforce()

    def reset(self) -> None:
        base_reset()
        controller.reset_runtime()

    def get_window_stats(self) -> dict[str, Any]:
        stats = dict(base_stats() if callable(base_stats) else {})
        config_value = stats.get("config")
        if isinstance(config_value, dict):
            config_value = {**config_value, "sliding_window_mode": mode}
        stats["config"] = config_value
        stats["pinned_context"] = controller.stats()
        return stats

    decoder.enforce_window_with_context = MethodType(enforce_window_with_context, decoder)
    decoder.reset = MethodType(reset, decoder)
    decoder.get_window_stats = MethodType(get_window_stats, decoder)
    decoder._reported_sliding_window_mode = mode
    decoder._pinned_context_controller = controller
    return controller


def install_context_memory(
    decoder: Any,
    *,
    config: PinnedContextConfig,
    on_summary_needed: Callable[[dict[str, Any]], None] | None = None,
) -> PinnedContextController:
    """Install the memory-plus-slate policy."""
    return _install_pinned_context(
        decoder,
        config=config,
        mode=CONTEXT_MEMORY,
        allow_memory=True,
        on_summary_needed=on_summary_needed,
    )


def install_context_slate(
    decoder: Any,
    *,
    config: PinnedContextConfig,
) -> PinnedContextController:
    """Pin the live task slate without converting evicted units into memory text."""
    return _install_pinned_context(
        decoder,
        config=config,
        mode=CONTEXT_SLATE,
        allow_memory=False,
    )


def _event_summary(value: Any) -> str:
    if isinstance(value, Mapping):
        return str(value.get("event_summary") or "")
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        parsed = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return text
    return str(parsed.get("event_summary") or "") if isinstance(parsed, Mapping) else ""
