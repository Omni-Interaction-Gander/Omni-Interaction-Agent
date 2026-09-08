from __future__ import annotations

from types import MethodType
from typing import Any


CONTEXT_NO_PREVIOUS = "context_no_previous"


def install_context_no_previous(decoder: Any) -> None:
    """Install no-previous absolute-RoPE eviction on a StreamDecoder.

    The context dispatcher remains unchanged. Eviction removes the oldest unit's KV
    slice, omits ``previous:`` tokens, and offsets new positions by removed-token count.
    """
    if getattr(decoder, "_context_no_previous_installed", False):
        return
    config = getattr(decoder, "_window_config", None)
    if getattr(config, "sliding_window_mode", None) != "context":
        raise ValueError(
            "context_no_previous requires the upstream decoder to be initialized in internal "
            "'context' mode"
        )

    decoder._drop_unit_with_context = MethodType(_drop_unit_without_rebuild, decoder)
    decoder.feed = MethodType(_feed_with_absolute_positions, decoder)
    decoder._reported_sliding_window_mode = CONTEXT_NO_PREVIOUS
    decoder._context_no_previous_installed = True

    get_stats = getattr(decoder, "get_window_stats", None)
    if callable(get_stats):
        decoder._context_no_previous_base_get_window_stats = get_stats
        decoder.get_window_stats = MethodType(_get_window_stats, decoder)


def _drop_unit_without_rebuild(
    decoder: Any,
    unit_id: int,
    max_previous_tokens: int,
) -> tuple[bool, str, list[int]]:
    del max_previous_tokens
    history = list(decoder._unit_history)
    matching_indices = [
        index for index, entry in enumerate(history) if entry.get("unit_id") == unit_id
    ]
    if not matching_indices:
        return False, "", []
    if matching_indices != list(range(matching_indices[0], matching_indices[-1] + 1)):
        raise RuntimeError(f"Non-contiguous cache entries for unit_id={unit_id}")

    entries = [history[index] for index in matching_indices]
    total_length = sum(int(entry.get("length", 0)) for entry in entries)
    if total_length <= 0:
        decoder._unit_history = [
            entry for entry in history if entry.get("unit_id") != unit_id
        ]
        return False, "", []

    first_index = matching_indices[0]
    start = int(decoder._system_preserve_length) + sum(
        int(entry.get("length", 0)) for entry in history[:first_index]
    )
    end = start + total_length
    cache_length = int(decoder.get_cache_length())
    if start < 0 or end > cache_length:
        raise RuntimeError(
            "Unit history/cache layout mismatch while evicting without RoPE rebuild: "
            f"unit_id={unit_id}, range=[{start}, {end}), cache_length={cache_length}"
        )

    decoder.cache = _remove_cache_range(decoder.cache, start, end)
    decoder._unit_history = [entry for entry in history if entry.get("unit_id") != unit_id]
    decoder._position_offset = int(decoder._position_offset) + total_length
    _clear_previous_state(decoder)
    return True, "", []


def _remove_cache_range(cache: Any, start: int, end: int) -> Any:
    import torch

    def remove(tensor):
        return torch.cat((tensor[:, :, :start, :], tensor[:, :, end:, :]), dim=2)

    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        cache.key_cache = [remove(key) for key in cache.key_cache]
        cache.value_cache = [remove(value) for value in cache.value_cache]
        new_length = cache.key_cache[0].shape[2] if cache.key_cache else 0
        if hasattr(cache, "_seen_tokens"):
            cache._seen_tokens = new_length
        return cache
    if isinstance(cache, (tuple, list)):
        layers = [(remove(layer[0]), remove(layer[1])) for layer in cache]
        return tuple(layers) if isinstance(cache, tuple) else layers
    raise TypeError(f"Unsupported decoder cache type: {type(cache)!r}")


def _clear_previous_state(decoder: Any) -> None:
    decoder._previous_content_length = 0
    decoder._has_previous = False
    decoder._previous_text = ""
    decoder._previous_token_ids = []


def _feed_with_absolute_positions(
    decoder: Any,
    embeds: Any,
    return_logits: bool = False,
):
    import torch

    with torch.no_grad():
        length = embeds.size(0)
        past_length = int(decoder.get_cache_length())
        absolute_start = past_length + int(decoder._position_offset)
        position_ids = torch.arange(
            absolute_start,
            absolute_start + length,
            device=embeds.device,
        ).unsqueeze(0)
        output = decoder.m(
            inputs_embeds=embeds.unsqueeze(0),
            position_ids=position_ids,
            past_key_values=decoder.cache,
            return_dict=True,
            output_hidden_states=True,
        )
        decoder.cache = output.past_key_values

        if return_logits:
            hidden = output.hidden_states[-1]
            logits = decoder.m.lm_head(hidden)[:, -1]
            return logits, hidden
    return None


def _get_window_stats(decoder: Any) -> dict[str, Any]:
    stats = dict(decoder._context_no_previous_base_get_window_stats())
    config = dict(stats.get("config") or {})
    config["sliding_window_mode"] = CONTEXT_NO_PREVIOUS
    stats["config"] = config
    stats["previous_content_length"] = 0
    stats["previous_text_length"] = 0
    stats["previous_token_count"] = 0
    return stats
