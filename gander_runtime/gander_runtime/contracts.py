from __future__ import annotations

import dataclasses
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

ContextKind = Literal[
    "user_text",
    "frontbrain_reply",
    "audio_transcript",
    "video_frame",
    "image",
    "screen",
    "system",
    "tool_result",
]
# Live perception is bounded by recency. ContextEvent and MediaRef use distinct
# frame-kind vocabularies, both listed here for consistent filtering.
EPHEMERAL_VISUAL_CONTEXT_KINDS: frozenset[str] = frozenset({"screen", "video_frame"})
EPHEMERAL_VISUAL_MEDIA_KINDS: frozenset[str] = frozenset({"screen", "frame"})

UpdateMode = Literal["additive", "replace", "cancel"]
ShareKind = Literal[
    "answer",
    "important",
    "milestone",
    "need_input",
    "confirm_action",
    "correction",
    "completed",
]
InteractionKind = Literal["approval", "user_input"]
TaskStatus = Literal[
    "running",
    "needs_input",
    "awaiting_confirmation",
    "completed",
    "cancelled",
    "partial",
    "failed",
]


def now_ms() -> int:
    return int(time.time() * 1000)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def stable_key(*parts: str) -> str:
    return uuid.uuid5(uuid.NAMESPACE_URL, "\0".join(parts)).hex


def stable_id(prefix: str, *parts: str) -> str:
    return f"{prefix}_{stable_key(*parts)}"


def storage_key(value: str, *, prefix_limit: int = 96) -> str:
    """Return a readable, collision-resistant cross-platform path component."""

    if not isinstance(value, str) or not value:
        raise ValueError("storage identifier must be a non-empty string")
    if prefix_limit < 1:
        raise ValueError("prefix_limit must be positive")
    prefix = "".join(
        char if char.isascii() and (char.isalnum() or char in "-_") else "_"
        for char in value
    ).strip("_")
    prefix = (prefix or "id")[:prefix_limit]
    suffix = stable_key(value)[:16]
    return f"{prefix}-{suffix}"


@dataclass(frozen=True)
class MediaRef:
    """A raw media reference. Only API-required conversion should create derivatives."""

    kind: Literal["audio", "video", "image", "frame", "screen"]
    path: str
    timestamp_ms: int | None = None
    mime_type: str | None = None
    source_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ContextEvent:
    """One raw event; ASR spans are session-relative and timestamp_ms is receive time."""

    session_id: str
    seq: int
    role: Literal["user", "frontbrain", "system", "tool"]
    kind: ContextKind
    text: str = ""
    media: tuple[MediaRef, ...] = ()
    start_ms: int | None = None
    end_ms: int | None = None
    timestamp_ms: int = field(default_factory=now_ms)
    event_id: str = field(default_factory=lambda: new_id("ctx"))
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate an optional session-relative media span."""

        if (self.start_ms is None) != (self.end_ms is None):
            raise ValueError("start_ms and end_ms must be provided together")
        if self.start_ms is None:
            return
        if self.start_ms < 0:
            raise ValueError("start_ms must be non-negative")
        if self.end_ms < self.start_ms:
            raise ValueError("end_ms must not be earlier than start_ms")


@dataclass
class WorkState:
    completed: list[str] = field(default_factory=list)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    next_step: str | None = None
    provider_state: dict[str, Any] = field(default_factory=dict)

    def apply_patch(self, patch: dict[str, Any] | None) -> None:
        if not patch:
            return
        for name in ("completed", "artifacts", "decisions"):
            values = patch.get(name)
            if not isinstance(values, list):
                continue
            target = getattr(self, name)
            for value in values:
                if value not in target:
                    target.append(value)
        if "next_step" in patch:
            self.next_step = patch.get("next_step")
        provider_state = patch.get("provider_state")
        if isinstance(provider_state, dict):
            self.provider_state.update(provider_state)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class ContextSnapshot:
    session_id: str
    events: tuple[ContextEvent, ...]
    created_at_ms: int = field(default_factory=now_ms)
    truncated_before_seq: int | None = None


@dataclass(frozen=True)
class TaskRequest:
    task_id: str
    session_id: str
    generation: int
    instruction: str
    context: ContextSnapshot
    work_state: WorkState
    trigger_event_id: str | None = None
    request_id: str = field(default_factory=lambda: new_id("req"))
    created_at_ms: int = field(default_factory=now_ms)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskUpdate:
    mode: UpdateMode
    event: ContextEvent
    instruction: str | None = None
    context: ContextSnapshot | None = None
    update_id: str = field(default_factory=lambda: new_id("upd"))
    created_at_ms: int = field(default_factory=now_ms)


@dataclass(frozen=True)
class TaskQuery:
    """One non-mutating question about an active task."""

    task_id: str
    session_id: str
    generation: int
    question: str
    context: ContextSnapshot
    request_id: str = field(default_factory=lambda: new_id("query"))
    created_at_ms: int = field(default_factory=now_ms)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InteractionOption:
    label: str
    description: str = ""


@dataclass(frozen=True)
class InteractionQuestion:
    question_id: str
    prompt: str
    header: str = ""
    options: tuple[InteractionOption, ...] = ()
    allows_other: bool = False
    is_secret: bool = False


@dataclass(frozen=True)
class TaskInteraction:
    """A provider request that pauses the active task until the user responds."""

    task_id: str
    session_id: str
    generation: int
    interaction_id: str
    kind: InteractionKind
    prompt: str
    choices: tuple[str, ...] = ()
    questions: tuple[InteractionQuestion, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at_ms: int = field(default_factory=now_ms)


@dataclass(frozen=True)
class TaskInteractionReply:
    """One user response routed back to the exact pending provider request."""

    task_id: str
    session_id: str
    generation: int
    interaction_id: str
    text: str = ""
    decision: str | None = None
    answers: dict[str, tuple[str, ...]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at_ms: int = field(default_factory=now_ms)


@dataclass(frozen=True)
class ShareEvent:
    task_id: str
    session_id: str
    generation: int
    kind: ShareKind
    text: str
    state_patch: dict[str, Any] = field(default_factory=dict)
    share_id: str = field(default_factory=lambda: new_id("share"))
    created_at_ms: int = field(default_factory=now_ms)


@dataclass(frozen=True)
class TaskResult:
    task_id: str
    session_id: str
    generation: int
    status: TaskStatus
    full_result: str
    artifacts: tuple[dict[str, Any], ...] = ()
    assumptions: tuple[str, ...] = ()
    unresolved: tuple[str, ...] = ()
    work_state: dict[str, Any] = field(default_factory=dict)
    provider_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProviderEvent:
    kind: Literal[
        "share",
        "interaction",
        "interaction_resolved",
        "result",
        "error",
    ]
    generation: int
    share: ShareEvent | None = None
    interaction: TaskInteraction | None = None
    interaction_id: str | None = None
    interaction_resolution: Literal["response", "external", "turn"] | None = None
    result: TaskResult | None = None
    error: str | None = None
