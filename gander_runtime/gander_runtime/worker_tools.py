"""Provider-neutral tool contract for pull-context Gander workers."""

from __future__ import annotations

import asyncio
import inspect
import json
import math
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

WORKER_TOOL_NAMES = frozenset({"memory_search", "context_fetch", "share"})
CONTEXT_KINDS = frozenset(
    {"realtime", "turn", "task", "event", "artifact"}
)
REALTIME_CONTEXT_KINDS = frozenset(
    {
        "user_text",
        "frontbrain_reply",
        "audio_transcript",
        "video_frame",
        "image",
        "screen",
        "system",
        "tool_result",
    }
)
CONTEXT_ROLES = frozenset({"user", "frontbrain", "system", "tool"})
MEMORY_SCOPES = frozenset({"project", "owner"})
SHARE_KINDS = frozenset(
    {
        "important",
        "milestone",
        "correction",
    }
)

MEMORY_SEARCH_TOOL: dict[str, Any] = {
    "name": "memory_search",
    "description": (
        "Search durable multimodal MM-Mem evidence from earlier sessions. Write "
        "a focused retrieval query yourself; the memory backend returns evidence "
        "for you to interpret, not a final answer."
    ),
    "inputSchema": {
        "type": "object",
        "required": ["query"],
        "additionalProperties": False,
        "properties": {
            "query": {"type": "string", "minLength": 1, "maxLength": 1000},
            "scope": {
                "type": "string",
                "enum": sorted(MEMORY_SCOPES),
                "default": "owner",
            },
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 20,
                "default": 8,
            },
        },
    },
}

CONTEXT_FETCH_TOOL: dict[str, Any] = {
    "name": "context_fetch",
    "description": (
        "Fetch selected bounded context in one call. Use kinds for runtime record "
        "classes; for the frozen task-start timeline, narrow further with "
        "context_kinds, roles, last_ms, or query. Use refs for exact objects and "
        "include_media when selected screen/image content is needed."
    ),
    "inputSchema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "refs": {
                "type": "array",
                "maxItems": 32,
                "uniqueItems": True,
                "items": {"type": "string", "minLength": 1, "maxLength": 256},
            },
            "query": {"type": "string", "maxLength": 1000},
            "kinds": {
                "type": "array",
                "maxItems": len(CONTEXT_KINDS),
                "uniqueItems": True,
                "items": {"type": "string", "enum": sorted(CONTEXT_KINDS)},
                "description": (
                    "Optional record classes. realtime is the frozen task-start "
                    "timeline; turn/task/event/artifact describe the task lineage."
                ),
            },
            "context_kinds": {
                "type": "array",
                "maxItems": len(REALTIME_CONTEXT_KINDS),
                "uniqueItems": True,
                "items": {
                    "type": "string",
                    "enum": sorted(REALTIME_CONTEXT_KINDS),
                },
                "description": "Optional source types within realtime context.",
            },
            "roles": {
                "type": "array",
                "maxItems": len(CONTEXT_ROLES),
                "uniqueItems": True,
                "items": {"type": "string", "enum": sorted(CONTEXT_ROLES)},
            },
            "last_ms": {
                "type": "integer",
                "minimum": 1,
                "maximum": 3600000,
                "description": "Only realtime events this many ms before task start.",
            },
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 50,
                "default": 20,
            },
            "max_chars": {
                "type": "integer",
                "minimum": 256,
                "maximum": 48000,
                "default": 12000,
            },
            "cursor": {
                "type": "integer",
                "minimum": 0,
                "maximum": 100000,
                "default": 0,
            },
            "include_media": {
                "type": "boolean",
                "default": False,
                "description": (
                    "Return eligible image/screen references. Set true when the "
                    "instruction refers to what was visible on screen."
                ),
            },
        },
    },
}

SHARE_TOOL: dict[str, Any] = {
    "name": "share",
    "description": (
        "The only channel that delivers intermediate progress to the realtime "
        "frontbrain. Send a concise milestone, important finding, or correction; "
        "ordinary Codex commentary and plan updates are not delivered. Do not use "
        "this tool for questions, "
        "confirmations, permissions, or the final result; use Codex native "
        "interactions for questions and approvals, and return the final agent "
        "message normally."
    ),
    "inputSchema": {
        "type": "object",
        "required": ["kind", "text"],
        "additionalProperties": False,
        "properties": {
            "kind": {"type": "string", "enum": sorted(SHARE_KINDS)},
            "text": {"type": "string", "minLength": 1, "maxLength": 800},
            "state_patch": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "completed": {
                        "type": "array",
                        "maxItems": 64,
                        "items": {"type": "string", "maxLength": 512},
                    },
                    "artifacts": {
                        "type": "array",
                        "maxItems": 64,
                        "items": {"type": "object", "maxProperties": 32},
                    },
                    "decisions": {
                        "type": "array",
                        "maxItems": 64,
                        "items": {"type": "string", "maxLength": 512},
                    },
                    "next_step": {
                        "type": ["string", "null"],
                        "maxLength": 1000,
                    },
                },
            },
        },
    },
}

WORKER_TOOL_SCHEMAS = (MEMORY_SEARCH_TOOL, CONTEXT_FETCH_TOOL, SHARE_TOOL)


@dataclass(frozen=True)
class MemoryHit:
    """One provider-neutral memory result; text is always bounded by Gateway."""

    ref: str
    text: str
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.ref, str) or not self.ref.strip():
            raise ValueError("memory hit ref must be a non-empty string")
        if not isinstance(self.text, str):
            raise ValueError("memory hit text must be a string")
        if self.score is not None and (
            isinstance(self.score, bool)
            or not isinstance(self.score, (int, float))
            or not math.isfinite(float(self.score))
        ):
            raise ValueError("memory hit score must be a finite number")
        if not isinstance(self.metadata, dict):
            raise ValueError("memory hit metadata must be an object")


class MemoryProvider(Protocol):
    """Cross-session memory boundary supplied by the deployment."""

    async def search(
        self,
        *,
        owner_id: str,
        project_id: str,
        task_id: str,
        query: str,
        scope: Literal["project", "owner"],
        limit: int,
    ) -> Sequence[MemoryHit | Mapping[str, Any]]:
        ...


class WorkerToolControl(Protocol):
    async def memory_search(self, query: str, **kwargs: Any) -> dict[str, Any]:
        ...

    async def context_fetch(self, **kwargs: Any) -> dict[str, Any]:
        ...


ShareDispatch = Callable[
    [dict[str, Any], str], Awaitable[dict[str, Any]]
]


class WorkerToolSession:
    """Reusable dispatcher mounted by a WorkerProvider."""

    def __init__(
        self, control: WorkerToolControl, share: ShareDispatch
    ) -> None:
        self.control = control
        self._share = share

    async def dispatch(
        self, name: str, arguments: dict[str, Any], call_id: str
    ) -> dict[str, Any]:
        normalized = validate_tool_arguments(name, arguments)
        if name == "memory_search":
            return await self.control.memory_search(**normalized)
        if name == "context_fetch":
            return await self.control.context_fetch(**normalized)
        return await self._share(normalized, call_id)


MemorySearchCallable = Callable[..., Sequence[Any] | Awaitable[Sequence[Any]]]


class CallableMemoryProvider:
    """Callable memory-search provider."""

    def __init__(self, search: MemorySearchCallable) -> None:
        if not callable(search):
            raise TypeError("memory search provider must be callable")
        self._search = search

    async def search(self, **kwargs: Any) -> Sequence[Any]:
        if inspect.iscoroutinefunction(self._search):
            result = self._search(**kwargs)
        else:
            result = await asyncio.to_thread(self._search, **kwargs)
        if inspect.isawaitable(result):
            result = await result
        if isinstance(result, (str, bytes)) or not isinstance(result, Sequence):
            raise TypeError("memory provider must return a sequence of hits")
        return result


def validate_tool_arguments(name: str, arguments: Any) -> dict[str, Any]:
    """Validate a worker call without depending on a JSON-schema library."""

    if name not in WORKER_TOOL_NAMES:
        raise ValueError(f"unknown worker tool: {name}")
    if not isinstance(arguments, dict):
        raise TypeError(f"{name} arguments must be an object")
    if name == "memory_search":
        return _validate_memory_search(arguments)
    if name == "context_fetch":
        return _validate_context_fetch(arguments)
    return _validate_share(arguments)


def _validate_memory_search(arguments: dict[str, Any]) -> dict[str, Any]:
    _reject_unknown(arguments, {"query", "scope", "limit"}, "memory_search")
    query = _bounded_text(arguments.get("query"), "query", 1000)
    scope = arguments.get("scope", "owner")
    if scope not in MEMORY_SCOPES:
        raise ValueError(f"scope must be one of {sorted(MEMORY_SCOPES)}")
    limit = _bounded_int(arguments.get("limit", 8), "limit", 1, 20)
    return {"query": query, "scope": scope, "limit": limit}


def _validate_context_fetch(arguments: dict[str, Any]) -> dict[str, Any]:
    _reject_unknown(
        arguments,
        {
            "refs",
            "query",
            "kinds",
            "context_kinds",
            "roles",
            "last_ms",
            "limit",
            "max_chars",
            "cursor",
            "include_media",
        },
        "context_fetch",
    )
    raw_refs = arguments.get("refs", ())
    if not isinstance(raw_refs, (list, tuple)) or isinstance(raw_refs, (str, bytes)):
        raise TypeError("refs must be an array")
    if len(raw_refs) > 32:
        raise ValueError("refs exceeds 32 items")
    refs = tuple(_bounded_text(value, "ref", 256) for value in raw_refs)
    if len(refs) != len(set(refs)):
        raise ValueError("refs must not contain duplicates")
    query_value = arguments.get("query", "")
    if not isinstance(query_value, str):
        raise TypeError("query must be a string")
    query = query_value.strip()
    if len(query) > 1000:
        raise ValueError("query exceeds 1000 characters")
    raw_kinds = arguments.get("kinds", ())
    if not isinstance(raw_kinds, (list, tuple)) or isinstance(raw_kinds, (str, bytes)):
        raise TypeError("kinds must be an array")
    kinds = tuple(raw_kinds)
    if len(kinds) > len(CONTEXT_KINDS) or len(kinds) != len(set(kinds)):
        raise ValueError(
            f"kinds must contain at most {len(CONTEXT_KINDS)} unique values"
        )
    if not set(kinds) <= CONTEXT_KINDS:
        raise ValueError(f"kinds must be selected from {sorted(CONTEXT_KINDS)}")
    context_kinds = _unique_enum_values(
        arguments.get("context_kinds", ()),
        "context_kinds",
        REALTIME_CONTEXT_KINDS,
    )
    roles = _unique_enum_values(
        arguments.get("roles", ()), "roles", CONTEXT_ROLES
    )
    last_ms_value = arguments.get("last_ms")
    last_ms = (
        None
        if last_ms_value is None
        else _bounded_int(last_ms_value, "last_ms", 1, 3_600_000)
    )
    include_media = arguments.get("include_media", False)
    if not isinstance(include_media, bool):
        raise TypeError("include_media must be a boolean")
    return {
        "refs": refs,
        "query": query,
        "kinds": kinds,
        "context_kinds": context_kinds,
        "roles": roles,
        "last_ms": last_ms,
        "limit": _bounded_int(arguments.get("limit", 20), "limit", 1, 50),
        "max_chars": _bounded_int(
            arguments.get("max_chars", 12000), "max_chars", 256, 48000
        ),
        "cursor": _bounded_int(arguments.get("cursor", 0), "cursor", 0, 100000),
        "include_media": include_media,
    }


def _validate_share(arguments: dict[str, Any]) -> dict[str, Any]:
    _reject_unknown(arguments, {"kind", "text", "state_patch"}, "share")
    kind = arguments.get("kind")
    if kind not in SHARE_KINDS:
        raise ValueError(f"kind must be one of {sorted(SHARE_KINDS)}")
    text = _bounded_text(arguments.get("text"), "text", 800)
    patch = arguments.get("state_patch") or {}
    if not isinstance(patch, dict):
        raise TypeError("state_patch must be an object")
    _reject_unknown(
        patch, {"completed", "artifacts", "decisions", "next_step"}, "state_patch"
    )
    for key in ("completed", "artifacts", "decisions"):
        value = patch.get(key, ())
        if not isinstance(value, (list, tuple)) or isinstance(value, (str, bytes)):
            raise TypeError(f"state_patch.{key} must be an array")
        if len(value) > 64:
            raise ValueError(f"state_patch.{key} exceeds 64 items")
    for key in ("completed", "decisions"):
        for value in patch.get(key, ()):
            if not isinstance(value, str):
                raise TypeError(f"state_patch.{key} items must be strings")
            if len(value) > 512:
                raise ValueError(
                    f"state_patch.{key} items exceed 512 characters"
                )
    for value in patch.get("artifacts", ()):
        if not isinstance(value, dict):
            raise TypeError("state_patch.artifacts items must be objects")
        if len(value) > 32:
            raise ValueError(
                "state_patch.artifacts items exceed 32 properties"
            )
    next_step = patch.get("next_step")
    if next_step is not None and not isinstance(next_step, str):
        raise TypeError("state_patch.next_step must be a string or null")
    if isinstance(next_step, str) and len(next_step) > 1000:
        raise ValueError("state_patch.next_step exceeds 1000 characters")
    try:
        encoded = json.dumps(
            patch,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise TypeError("state_patch must be JSON-compatible") from exc
    if len(encoded) > 64 * 1024:
        raise ValueError("state_patch exceeds 65536 characters")
    normalized_patch = json.loads(encoded)
    assert isinstance(normalized_patch, dict)
    return {"kind": kind, "text": text, "state_patch": normalized_patch}


def _reject_unknown(arguments: Mapping[str, Any], allowed: set[str], name: str) -> None:
    unknown = set(arguments) - allowed
    if unknown:
        raise ValueError(f"unexpected {name} fields: {sorted(unknown)}")


def _unique_enum_values(
    value: Any, name: str, allowed: frozenset[str]
) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be an array")
    values = tuple(value)
    if len(values) > len(allowed) or len(values) != len(set(values)):
        raise ValueError(
            f"{name} must contain at most {len(allowed)} unique values"
        )
    if not set(values) <= allowed:
        raise ValueError(f"{name} must be selected from {sorted(allowed)}")
    return values


def _bounded_text(value: Any, name: str, maximum: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    normalized = value.strip()
    if len(normalized) > maximum:
        raise ValueError(f"{name} exceeds {maximum} characters")
    return normalized


def _bounded_int(value: Any, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value
