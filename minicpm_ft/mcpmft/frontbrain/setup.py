"""Tokenizer/model setup for the native task-tools front brain."""
from __future__ import annotations

from mcpmft.modeling.load import (
    add_native_frontbrain_tokens as ensure_native_frontbrain_tokens,
)

__all__ = [
    "ensure_native_frontbrain_tokens",
]
