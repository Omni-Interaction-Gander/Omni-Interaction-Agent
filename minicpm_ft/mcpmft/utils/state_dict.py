from __future__ import annotations

from typing import Any


def strip_wrapper_prefix(name: str) -> str:
    """Strip the top-level OmniTrainWrapper prefix from a single state-dict key."""
    return name[len("model."):] if name.startswith("model.") else name


def add_wrapper_prefix(name: str) -> str:
    """Add the top-level OmniTrainWrapper prefix to a single bare MiniCPMO key."""
    return name if name.startswith("model.") else f"model.{name}"


def strip_wrapper_prefix_if_present(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Strip one ``model.`` prefix when all keys come from OmniTrainWrapper."""
    if not state_dict:
        return state_dict
    keys = list(state_dict)
    if all(k.startswith("model.") for k in keys):
        return {strip_wrapper_prefix(k): v for k, v in state_dict.items()}
    return state_dict


def add_wrapper_prefix_if_absent(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Add one ``model.`` prefix when all keys look like bare MiniCPMO keys.

    This makes checkpoints saved as canonical MiniCPMO weights (``llm.*``, ``tts.*``, ...)
    loadable back into ``OmniTrainWrapper`` during Trainer resume, while keeping already-wrapped
    checkpoints untouched.
    """
    if not state_dict:
        return state_dict
    keys = list(state_dict)
    if all(not k.startswith("model.") for k in keys):
        return {add_wrapper_prefix(k): v for k, v in state_dict.items()}
    return state_dict
