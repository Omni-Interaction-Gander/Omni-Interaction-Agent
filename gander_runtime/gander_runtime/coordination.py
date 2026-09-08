from __future__ import annotations

import dataclasses
import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from .contracts import MediaRef, new_id, now_ms
from .worker_tools import WORKER_TOOL_NAMES

DeliveryTiming = Literal["hold", "safe_pause", "interrupt"]
AuthorizationOutcome = Literal["allow", "require_permission", "deny"]
TaskStatus = Literal[
    "queued",
    "running",
    "finalizing",
    "cancelling",
    "completed",
    "partial",
    "failed",
    "cancelled",
]
RunStatus = Literal[
    "queued",
    "running",
    "waiting",
    "completed",
    "partial",
    "failed",
    "cancelled",
]
BlockingScope = Literal["action", "branch", "run"]
InteractionKind = Literal["question", "choice", "permission"]
InteractionState = Literal[
    "pending",
    "resolving",
    "resolved",
    "expired",
    "cancelled",
    "failed",
]
InteractionAudience = Literal["user", "coordinator"]
DeliveryState = Literal["pending", "claimed", "delivered", "failed", "cancelled"]
SteeringMode = Literal["native", "next_turn", "none"]
SideQueryMode = Literal[
    "native_fork",
    "independent_session",
    # Coordinator-era capability aliases.
    "native",
    "isolated_fork",
    "next_turn",
    "none",
]
AuthorityEnforcement = Literal["gateway", "backend_hook", "sandbox", "none"]
StructuredEvents = Literal["native", "injected_tool", "limited", "none"]
# Provider capability tiers.
ContextProvisioning = Literal["pull", "push_bounded"]
SessionModel = Literal["stateful", "stateless"]
CoordinatorCallStatus = Literal["completed", "failed", "timed_out"]
ReasoningProfile = Literal["fast", "balanced", "deep"]

# Native task tools bind the current TurnEnvelope in the runtime.
TaskLane = Literal["main", "fork"]
TaskKind = Literal["main", "side_query"]
TaskResolveAction = Literal[
    "cancel", "allow_once", "allow_session", "deny"
]
TASK_LANES = frozenset(TaskLane.__args__)
TASK_RESOLVE_ACTIONS = frozenset(TaskResolveAction.__args__)
# Task calls return a synchronous result for the front brain.
TaskControlStatus = Literal[
    "ok", "ambiguous", "no_such_task", "unsupported", "invalid_action"
]

_TERMINAL_TASK_STATUSES = frozenset({"completed", "partial", "failed", "cancelled"})
_STABLE_KEY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")


def _required(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _stable_key(value: str, name: str) -> None:
    _required(value, name)
    if not _STABLE_KEY.fullmatch(value):
        raise ValueError(f"{name} must be a stable ASCII key")


# Closed supervision vocabularies shared by coordinator and worker. Milestone
# keys remain descriptive.
ASK_REASONS = frozenset(
    {
        "critical_input.missing",  # a required input is missing
        "irreversible_ambiguity",  # irreversible / high-stakes ambiguity
        "source_conflict",  # conflicting sources/evidence that changes conclusions
        "formatting.detail",  # delegatable presentation detail
    }
)
AUTHORITY_ACTIONS = frozenset(
    {
        "read",
        "search",
        "draft",
        "edit_draft",  # default allow
        "send",
        "publish",
        "overwrite",
        "destructive_action",  # default require_permission
    }
)


def _ask_reason(value: str, name: str) -> None:
    _required(value, name)
    if value not in ASK_REASONS:
        raise ValueError(
            f"{name} must be one of {sorted(ASK_REASONS)}"
        )


def _authority_action(value: str, name: str) -> None:
    _required(value, name)
    if value not in AUTHORITY_ACTIONS:
        raise ValueError(
            f"{name} must be one of {sorted(AUTHORITY_ACTIONS)}"
        )



def _tuple(value: Any) -> tuple[Any, ...]:
    return value if isinstance(value, tuple) else tuple(value)


def _unique(values: tuple[str, ...], name: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{name} must not contain duplicates")


@dataclass(frozen=True)
class MilestoneCondition:
    field: str
    op: Literal["eq", "gte", "lte", "exists"] = "exists"
    value: Any = None

    def __post_init__(self) -> None:
        _stable_key(self.field, "condition field")
        if self.op not in {"eq", "gte", "lte", "exists"}:
            raise ValueError(f"unsupported milestone condition: {self.op}")
        if self.op != "exists" and self.value is None:
            raise ValueError(f"{self.op} requires a comparison value")


@dataclass(frozen=True)
class NotifyRule:
    key: str
    label: str
    condition: MilestoneCondition | None = None
    once: bool = True

    def __post_init__(self) -> None:
        _stable_key(self.key, "milestone key")
        _required(self.label, "milestone label")


@dataclass(frozen=True)
class NotifyPolicy:
    milestones: tuple[NotifyRule, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "milestones", _tuple(self.milestones))
        _unique(tuple(rule.key for rule in self.milestones), "milestone keys")


@dataclass(frozen=True)
class AskPolicy:
    must_ask: tuple[str, ...] = (
        "critical_input.missing",
        "irreversible_ambiguity",
    )
    delegate: tuple[str, ...] = ("formatting.detail",)
    when_exhausted: Literal["ask", "conservative", "fail"] = "conservative"
    max_clarification_rounds: int = 1
    max_questions_per_round: int = 3
    bundle_questions: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "must_ask", _tuple(self.must_ask))
        object.__setattr__(self, "delegate", _tuple(self.delegate))
        for name, values in (
            ("must_ask", self.must_ask),
            ("delegate", self.delegate),
        ):
            _unique(values, name)
            for value in values:
                _ask_reason(value, f"{name} reason")
        if set(self.must_ask) & set(self.delegate):
            raise ValueError("must_ask and delegate must be disjoint")
        if self.when_exhausted not in {"ask", "conservative", "fail"}:
            raise ValueError("invalid ask exhaustion policy")
        if self.max_clarification_rounds < 0:
            raise ValueError("max_clarification_rounds must be non-negative")
        if self.max_questions_per_round < 1:
            raise ValueError("max_questions_per_round must be positive")


@dataclass(frozen=True)
class AuthorityPolicy:
    allow: tuple[str, ...] = ("read", "search", "draft", "edit_draft")
    require_permission: tuple[str, ...] = (
        "send",
        "publish",
        "overwrite",
        "destructive_action",
    )
    deny: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("allow", "require_permission", "deny"):
            values = _tuple(getattr(self, name))
            object.__setattr__(self, name, values)
            _unique(values, name)
            for value in values:
                _authority_action(value, f"{name} action")
        sets = [set(self.allow), set(self.require_permission), set(self.deny)]
        overlaps = any(
            sets[left] & sets[right]
            for left in range(3)
            for right in range(left + 1, 3)
        )
        if overlaps:
            raise ValueError("authority action sets must be disjoint")


@dataclass(frozen=True)
class DeliveryPolicy:
    milestone: DeliveryTiming = "safe_pause"
    interaction: DeliveryTiming = "safe_pause"
    final: DeliveryTiming = "safe_pause"
    aggregate: DeliveryTiming = "safe_pause"
    high_risk: DeliveryTiming = "interrupt"
    aggregate_window_ms: int = 0

    def __post_init__(self) -> None:
        timings = {
            self.milestone,
            self.interaction,
            self.final,
            self.aggregate,
            self.high_risk,
        }
        if not timings <= {"hold", "safe_pause", "interrupt"}:
            raise ValueError("invalid delivery timing")
        if self.high_risk != "interrupt":
            raise ValueError("high-risk delivery must interrupt")
        if self.aggregate_window_ms < 0:
            raise ValueError("aggregate_window_ms must be non-negative")


@dataclass(frozen=True)
class ContractBody:
    notify_policy: NotifyPolicy = field(default_factory=NotifyPolicy)
    ask_policy: AskPolicy = field(default_factory=AskPolicy)
    authority_policy: AuthorityPolicy = field(default_factory=AuthorityPolicy)
    delivery_policy: DeliveryPolicy = field(default_factory=DeliveryPolicy)

    def __post_init__(self) -> None:
        if (
            self.delivery_policy.interaction == "hold"
            and (
                self.ask_policy.must_ask
                or self.ask_policy.when_exhausted == "ask"
                or self.authority_policy.require_permission
            )
        ):
            raise ValueError(
                "required questions and permissions cannot be held"
            )

    @classmethod
    def default(cls) -> ContractBody:
        return cls(
            notify_policy=NotifyPolicy(
                (NotifyRule("final.completed", "final result completed"),)
            )
        )


@dataclass(frozen=True)
class SupervisionContractRevision:
    task_id: str
    revision: int
    body: ContractBody
    source_turn_ids: tuple[str, ...]
    created_at_ms: int = field(default_factory=now_ms)

    def __post_init__(self) -> None:
        _required(self.task_id, "task_id")
        if self.revision < 1:
            raise ValueError("contract revision must be positive")
        object.__setattr__(self, "source_turn_ids", _tuple(self.source_turn_ids))
        if not self.source_turn_ids:
            raise ValueError("a contract revision requires source_turn_ids")


@dataclass(frozen=True)
class TurnEnvelope:
    owner_id: str
    voice_session_id: str
    turn_id: str
    final_asr: str
    media_refs: tuple[MediaRef, ...] = ()
    environment_ref: str = ""
    context_revision: int = 0
    timezone: str = "UTC"
    start_ms: int | None = None
    end_ms: int | None = None
    timestamp_ms: int | None = None
    # Routing metadata attached by the runtime after a native front-brain call.
    frontbrain_action: Literal["", "task_start"] = ""
    frontbrain_task_name: str = ""
    runtime_provider_name: str = ""
    created_at_ms: int = field(default_factory=now_ms)

    def __post_init__(self) -> None:
        for name in ("owner_id", "voice_session_id", "turn_id", "final_asr"):
            _required(getattr(self, name), name)
        object.__setattr__(self, "media_refs", _tuple(self.media_refs))
        if self.context_revision < 0:
            raise ValueError("context_revision must be non-negative")
        if (self.start_ms is None) != (self.end_ms is None):
            raise ValueError("turn start_ms and end_ms must be provided together")
        if self.start_ms is not None:
            if self.start_ms < 0:
                raise ValueError("turn start_ms must be non-negative")
            if self.end_ms < self.start_ms:
                raise ValueError("turn end_ms must not be earlier than start_ms")
        if self.timestamp_ms is not None and self.timestamp_ms < 0:
            raise ValueError("turn timestamp_ms must be non-negative")
        if self.frontbrain_action == "task_start":
            _required(self.frontbrain_task_name, "frontbrain_task_name")
        elif self.frontbrain_task_name or self.runtime_provider_name:
            raise ValueError(
                "frontbrain task metadata requires frontbrain_action=task_start"
            )

    @property
    def receipt_key(self) -> str:
        return f"{self.owner_id}\0{self.voice_session_id}\0{self.turn_id}\0assist"


@dataclass(frozen=True)
class ContextInventoryItem:
    kind: str
    count: int

    def __post_init__(self) -> None:
        _required(self.kind, "context inventory kind")
        if self.count < 1:
            raise ValueError("context inventory count must be positive")


@dataclass(frozen=True)
class ContextInventory:
    total: int = 0
    items: tuple[ContextInventoryItem, ...] = ()
    media_count: int = 0
    earliest_timestamp_ms: int | None = None
    latest_timestamp_ms: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "items", _tuple(self.items))
        if self.total < 0 or self.media_count < 0:
            raise ValueError("context inventory counts must be non-negative")
        if sum(item.count for item in self.items) != self.total:
            raise ValueError("context inventory item counts must equal total")
        if (self.earliest_timestamp_ms is None) != (
            self.latest_timestamp_ms is None
        ):
            raise ValueError("context inventory timestamps must be paired")
        if (
            self.earliest_timestamp_ms is not None
            and self.latest_timestamp_ms < self.earliest_timestamp_ms
        ):
            raise ValueError("context inventory timestamp range is invalid")


@dataclass(frozen=True)
class ContextPlan:
    event_refs: tuple[str, ...] = ()
    media_refs: tuple[str, ...] = ()
    artifact_refs: tuple[str, ...] = ()
    must_include_refs: tuple[str, ...] = ()
    brief: str = ""
    inventory: ContextInventory = field(default_factory=ContextInventory)

    def __post_init__(self) -> None:
        for name in (
            "event_refs",
            "media_refs",
            "artifact_refs",
            "must_include_refs",
        ):
            values = _tuple(getattr(self, name))
            object.__setattr__(self, name, values)
            _unique(values, name)


@dataclass(frozen=True)
class BackendCapabilities:
    steering: SteeringMode = "none"
    side_queries: SideQueryMode = "none"
    # Read-only queries may outlive the parent task.
    terminal_side_queries: SideQueryMode = "none"
    interactions: bool = False
    blocking_granularity: BlockingScope = "run"
    authority_enforcement: AuthorityEnforcement = "none"
    structured_events: StructuredEvents = "none"
    trusted_risk_signals: bool = False
    session_resume: bool = False
    modalities: frozenset[str] = frozenset({"text"})
    max_parallel_projects: int = 1
    # Pull requires the provider's context_fetch tool; other providers use bounded push.
    context_provisioning: ContextProvisioning = "push_bounded"
    session: SessionModel = "stateful"
    worker_tools: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if self.steering not in {"native", "next_turn", "none"}:
            raise ValueError("invalid steering capability")
        if self.side_queries not in {
            "native_fork",
            "independent_session",
            "native",
            "isolated_fork",
            "next_turn",
            "none",
        }:
            raise ValueError("invalid side query capability")
        if self.terminal_side_queries not in {
            "native_fork",
            "independent_session",
            "native",
            "isolated_fork",
            "none",
        }:
            raise ValueError("invalid terminal side query capability")
        if self.blocking_granularity not in {"action", "branch", "run"}:
            raise ValueError("invalid blocking granularity")
        if self.authority_enforcement not in {
            "gateway",
            "backend_hook",
            "sandbox",
            "none",
        }:
            raise ValueError("invalid authority enforcement")
        if self.structured_events not in {
            "native",
            "injected_tool",
            "limited",
            "none",
        }:
            raise ValueError("invalid structured event capability")
        object.__setattr__(self, "modalities", frozenset(self.modalities))
        if not self.modalities:
            raise ValueError("modalities must not be empty")
        if self.max_parallel_projects < 1:
            raise ValueError("max_parallel_projects must be positive")
        if self.context_provisioning not in {"pull", "push_bounded"}:
            raise ValueError("invalid context_provisioning")
        if self.session not in {"stateful", "stateless"}:
            raise ValueError("invalid session model")
        object.__setattr__(self, "worker_tools", frozenset(self.worker_tools))
        unsupported_tools = self.worker_tools - WORKER_TOOL_NAMES
        if unsupported_tools:
            raise ValueError(
                f"unsupported worker tools: {sorted(unsupported_tools)}"
            )
        if (
            self.context_provisioning == "pull"
            and "context_fetch" not in self.worker_tools
        ):
            raise ValueError(
                "pull context provisioning requires the context_fetch worker tool"
            )


@dataclass(frozen=True)
class WorkerPolicyView:
    contract_revision: int
    must_ask: tuple[str, ...]
    delegated_questions: tuple[str, ...]
    allowed_actions: tuple[str, ...]
    permission_actions: tuple[str, ...]
    denied_actions: tuple[str, ...]
    subscribed_milestones: tuple[str, ...]

    @classmethod
    def from_contract(
        cls, contract: SupervisionContractRevision
    ) -> WorkerPolicyView:
        body = contract.body
        return cls(
            contract_revision=contract.revision,
            must_ask=body.ask_policy.must_ask,
            delegated_questions=body.ask_policy.delegate,
            allowed_actions=body.authority_policy.allow,
            permission_actions=body.authority_policy.require_permission,
            denied_actions=body.authority_policy.deny,
            subscribed_milestones=tuple(
                rule.key for rule in body.notify_policy.milestones
            ),
        )


@dataclass(frozen=True)
class ProjectRecord:
    project_id: str
    owner_id: str
    label: str
    provider_name: str
    environment_ref: str = ""
    backend_session_id: str = ""
    status: Literal["active", "closed"] = "active"
    created_at_ms: int = field(default_factory=now_ms)
    updated_at_ms: int = field(default_factory=now_ms)

    def __post_init__(self) -> None:
        for name in ("project_id", "owner_id", "label", "provider_name"):
            _required(getattr(self, name), name)
        if self.status not in {"active", "closed"}:
            raise ValueError("invalid project status")


@dataclass(frozen=True)
class TaskRecord:
    task_id: str
    owner_id: str
    source_turn_id: str
    project_id: str
    title: str
    objective: str
    instruction: str
    provider_name: str
    context_plan: ContextPlan
    reasoning_profile: ReasoningProfile = "balanced"
    source_voice_session_id: str = ""
    status: TaskStatus = "queued"
    generation: int = 1
    contract_revision: int = 1
    active_run_id: str | None = None
    cancel_request_id: str | None = None
    blocked_actions: tuple[str, ...] = ()
    blocked_branches: tuple[str, ...] = ()
    run_blocked: bool = False
    assumptions: tuple[str, ...] = ()
    result: str = ""
    error: str = ""
    created_at_ms: int = field(default_factory=now_ms)
    updated_at_ms: int = field(default_factory=now_ms)
    # Human-facing name shared by the user, front brain, and provider.
    name: str = ""
    # Stable provider conversation identity across task_send continuations.
    lineage_id: str = ""
    # Side queries are persisted but omitted from the main task slate.
    kind: TaskKind = "main"
    parent_task_id: str = ""
    # Exact parent run; empty values support older ledger records.
    parent_run_id: str = ""

    def __post_init__(self) -> None:
        if not self.lineage_id:
            object.__setattr__(self, "lineage_id", self.task_id)
        for name in (
            "task_id",
            "lineage_id",
            "owner_id",
            "source_turn_id",
            "project_id",
            "title",
            "objective",
            "instruction",
            "provider_name",
        ):
            _required(getattr(self, name), name)
        for name in ("blocked_actions", "blocked_branches", "assumptions"):
            values = _tuple(getattr(self, name))
            object.__setattr__(self, name, values)
            _unique(values, name)
        if self.generation < 1 or self.contract_revision < 1:
            raise ValueError("task generation and contract revision must be positive")
        if self.kind not in {"main", "side_query"}:
            raise ValueError("invalid task kind")
        if self.kind == "side_query":
            if not self.parent_task_id:
                raise ValueError("side query requires parent_task_id")
        if self.kind == "main" and (self.parent_task_id or self.parent_run_id):
            raise ValueError("main task must not set a parent task or run")
        if self.reasoning_profile not in {"fast", "balanced", "deep"}:
            raise ValueError("invalid task reasoning profile")
        if self.status not in {
            "queued",
            "running",
            "finalizing",
            "cancelling",
            "completed",
            "partial",
            "failed",
            "cancelled",
        }:
            raise ValueError("invalid task status")

    @property
    def terminal(self) -> bool:
        return self.status in _TERMINAL_TASK_STATUSES


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    task_id: str
    project_id: str
    provider_name: str
    generation: int
    status: RunStatus = "queued"
    backend_session_id: str = ""
    started_at_ms: int | None = None
    ended_at_ms: int | None = None
    last_event_seq: int = 0
    error: str = ""

    def __post_init__(self) -> None:
        for name in ("run_id", "task_id", "project_id", "provider_name"):
            _required(getattr(self, name), name)
        if self.generation < 1 or self.last_event_seq < 0:
            raise ValueError("invalid run generation or sequence")
        if self.status not in {
            "queued",
            "running",
            "waiting",
            "completed",
            "partial",
            "failed",
            "cancelled",
        }:
            raise ValueError("invalid run status")


@dataclass(frozen=True)
class ArtifactEvidence:
    ref: str
    owner_id: str
    task_id: str
    facts: dict[str, Any] = field(default_factory=dict)
    created_at_ms: int = field(default_factory=now_ms)

    def __post_init__(self) -> None:
        _required(self.ref, "artifact ref")
        _required(self.owner_id, "artifact owner_id")
        _required(self.task_id, "artifact task_id")


@dataclass(frozen=True)
class PendingInteraction:
    interaction_id: str
    owner_id: str
    target_id: str
    task_id: str | None
    run_id: str | None
    source: Literal["worker", "coordinator"]
    audience: InteractionAudience
    kind: InteractionKind
    prompt: str
    reason_key: str
    source_event_id: str | None = None
    action_key: str | None = None
    choices: tuple[str, ...] = ()
    requested_scope: BlockingScope = "run"
    effective_scope: BlockingScope = "run"
    state: InteractionState = "pending"
    fingerprint: str = ""
    coordination_request_id: str = ""
    response: str = ""
    decision: str | None = None
    created_at_ms: int = field(default_factory=now_ms)
    expires_at_ms: int | None = None
    resolved_at_ms: int | None = None

    def __post_init__(self) -> None:
        _required(self.interaction_id, "interaction_id")
        _required(self.target_id, "interaction target_id")
        _required(self.prompt, "interaction prompt")
        _stable_key(self.reason_key, "interaction reason_key")
        object.__setattr__(self, "choices", _tuple(self.choices))
        _unique(self.choices, "interaction choices")
        if self.source not in {"worker", "coordinator"}:
            raise ValueError("invalid interaction source")
        if self.audience not in {"user", "coordinator"}:
            raise ValueError("invalid interaction audience")
        if self.kind not in {"question", "choice", "permission"}:
            raise ValueError("invalid interaction kind")
        if self.requested_scope not in {"action", "branch", "run"}:
            raise ValueError("invalid requested blocking scope")
        if self.effective_scope not in {"action", "branch", "run"}:
            raise ValueError("invalid effective blocking scope")
        if self.state not in {
            "pending",
            "resolving",
            "resolved",
            "expired",
            "cancelled",
            "failed",
        }:
            raise ValueError("invalid interaction state")


@dataclass(frozen=True)
class DeliveryRecord:
    delivery_id: str
    owner_id: str
    task_id: str | None
    event_id: str | None
    interaction_id: str | None
    timing: Literal["safe_pause", "interrupt"]
    topic: Literal["milestone", "interaction", "final", "risk", "aggregate"]
    speech_hint: str
    # Structured final outcome, phrased for the user by the front brain.
    status: str = ""
    state: DeliveryState = "pending"
    claim_token: str = ""
    claim_expires_at_ms: int | None = None
    attempts: int = 0
    created_at_ms: int = field(default_factory=now_ms)
    delivered_at_ms: int | None = None

    def __post_init__(self) -> None:
        _required(self.delivery_id, "delivery_id")
        _required(self.owner_id, "delivery owner_id")
        if not isinstance(self.speech_hint, str):
            raise ValueError("delivery speech_hint must be a string")
        if self.status not in {"", "completed", "partial", "failed", "cancelled"}:
            raise ValueError("invalid delivery status")
        if self.timing not in {"safe_pause", "interrupt"}:
            raise ValueError("invalid delivery timing")
        if self.topic not in {
            "milestone",
            "interaction",
            "final",
            "risk",
            "aggregate",
        }:
            raise ValueError("invalid delivery topic")
        if self.state not in {
            "pending",
            "claimed",
            "delivered",
            "failed",
            "cancelled",
        }:
            raise ValueError("invalid delivery state")


@dataclass(frozen=True)
class ActivityAggregate:
    aggregate_id: str
    owner_id: str
    task_id: str
    contract_revision: int
    event_ids: tuple[str, ...]
    summaries: tuple[str, ...]
    event_count: int
    due_at_ms: int
    state: Literal["open", "flushed"] = "open"
    created_at_ms: int = field(default_factory=now_ms)
    flushed_at_ms: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_ids", _tuple(self.event_ids))
        object.__setattr__(self, "summaries", _tuple(self.summaries))
        if not self.event_ids or not self.summaries:
            raise ValueError("activity aggregate must contain an event")
        if len(self.event_ids) != len(self.summaries):
            raise ValueError("aggregate event_ids and summaries must align")
        if self.event_count < len(self.event_ids):
            raise ValueError("aggregate event_count is inconsistent")


@dataclass(frozen=True)
class UpdatePayload:
    kind: Literal["activity", "milestone"]
    summary: str
    milestone: str | None = None
    evidence_refs: tuple[str, ...] = ()
    next_step: str = ""
    severity: Literal["normal", "high_risk"] = "normal"

    def __post_init__(self) -> None:
        _required(self.summary, "update summary")
        object.__setattr__(self, "evidence_refs", _tuple(self.evidence_refs))
        if self.kind not in {"activity", "milestone"}:
            raise ValueError("invalid update kind")
        if self.severity not in {"normal", "high_risk"}:
            raise ValueError("invalid update severity")
        if self.kind == "milestone":
            if self.milestone is None:
                raise ValueError("milestone update requires a milestone key")
            _stable_key(self.milestone, "milestone key")
            if not self.evidence_refs:
                raise ValueError("milestone update requires evidence_refs")
        elif self.milestone is not None:
            raise ValueError("activity update cannot carry a milestone key")


@dataclass(frozen=True)
class InteractionPayload:
    kind: InteractionKind
    prompt: str
    blocking_scope: BlockingScope = "run"
    choices: tuple[str, ...] = ()
    reason_key: str = "critical_input.missing"
    action_key: str | None = None

    def __post_init__(self) -> None:
        _required(self.prompt, "interaction prompt")
        _ask_reason(self.reason_key, "interaction reason_key")
        object.__setattr__(self, "choices", _tuple(self.choices))
        if self.kind not in {"question", "choice", "permission"}:
            raise ValueError("invalid interaction kind")
        if self.blocking_scope not in {"action", "branch", "run"}:
            raise ValueError("invalid blocking scope")
        if self.kind == "choice" and len(self.choices) < 2:
            raise ValueError("choice interaction requires at least two choices")
        if self.kind == "permission":
            if self.action_key is None:
                raise ValueError("permission interaction requires action_key")
            _authority_action(self.action_key, "permission action_key")


@dataclass(frozen=True)
class DonePayload:
    status: Literal["completed", "partial", "failed", "cancelled"]
    result: str = ""
    artifact_refs: tuple[str, ...] = ()
    unresolved: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifact_refs", _tuple(self.artifact_refs))
        object.__setattr__(self, "unresolved", _tuple(self.unresolved))
        if self.status not in {"completed", "partial", "failed", "cancelled"}:
            raise ValueError("invalid done status")


WorkerPayload = UpdatePayload | InteractionPayload | DonePayload


@dataclass(frozen=True)
class WorkerEvent:
    event_id: str
    owner_id: str
    task_id: str
    project_id: str
    run_id: str
    generation: int
    seq: int
    type: Literal["update", "interaction", "done"]
    payload: WorkerPayload
    created_at_ms: int = field(default_factory=now_ms)

    def __post_init__(self) -> None:
        for name in ("event_id", "owner_id", "task_id", "project_id", "run_id"):
            _required(getattr(self, name), name)
        if self.generation < 1 or self.seq < 1:
            raise ValueError("event generation and seq must be positive")
        expected = {
            "update": UpdatePayload,
            "interaction": InteractionPayload,
            "done": DonePayload,
        }.get(self.type)
        if expected is None or not isinstance(self.payload, expected):
            raise ValueError("worker event type does not match its payload")


@dataclass(frozen=True)
class QuestionProposal:
    target_id: str
    reason_key: str
    questions: tuple[str, ...]
    blocking_scope: BlockingScope = "run"
    default_if_unanswered: str | None = None

    def __post_init__(self) -> None:
        _required(self.target_id, "question target_id")
        _ask_reason(self.reason_key, "question reason_key")
        object.__setattr__(self, "questions", _tuple(self.questions))
        if not self.questions:
            raise ValueError("questions must not be empty")
        for question in self.questions:
            _required(question, "question")


@dataclass(frozen=True)
class CreateTaskCommand:
    title: str
    objective: str
    instruction: str
    provider_name: str
    reasoning_profile: ReasoningProfile = "balanced"
    contract_body: ContractBody = field(default_factory=ContractBody.default)
    context_plan: ContextPlan = field(default_factory=ContextPlan)
    project_id: str | None = None
    project_label: str = "default"

    def __post_init__(self) -> None:
        for name in ("title", "objective", "instruction", "provider_name"):
            _required(getattr(self, name), name)
        if self.reasoning_profile not in {"fast", "balanced", "deep"}:
            raise ValueError("invalid task reasoning profile")
        if self.project_id is None:
            _required(self.project_label, "project_label")


@dataclass(frozen=True)
class SendTaskCommand:
    task_id: str
    instruction: str
    mode: Literal["update", "query"] = "update"
    context_plan: ContextPlan | None = None
    preempt: bool = False  # interrupt the active turn

    def __post_init__(self) -> None:
        _required(self.task_id, "task_id")
        _required(self.instruction, "instruction")
        if self.mode not in {"update", "query"}:
            raise ValueError("invalid task message mode")


@dataclass(frozen=True)
class UpdateContractCommand:
    task_id: str
    body: ContractBody


@dataclass(frozen=True)
class CancelTaskCommand:
    task_id: str


@dataclass(frozen=True)
class ResolveInteractionCommand:
    interaction_id: str
    response: str
    decision: Literal["allow", "allow_once", "allow_session", "deny"] | None = None

    def __post_init__(self) -> None:
        _required(self.interaction_id, "interaction_id")
        _required(self.response, "interaction response")
        if self.decision not in {
            None,
            "allow",
            "allow_once",
            "allow_session",
            "deny",
        }:
            raise ValueError("decision must be an allow scope, deny, or null")


CoordinatorCommand = (
    CreateTaskCommand
    | SendTaskCommand
    | UpdateContractCommand
    | CancelTaskCommand
    | ResolveInteractionCommand
)


@dataclass(frozen=True)
class CoordinatorPlan:
    disposition: Literal["answered", "accepted", "clarify", "rejected"]
    speech: str = ""
    commands: tuple[CoordinatorCommand, ...] = ()
    question: QuestionProposal | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "commands", _tuple(self.commands))
        if self.disposition not in {
            "answered",
            "accepted",
            "clarify",
            "rejected",
        }:
            raise ValueError("invalid Coordinator disposition")
        if self.disposition == "clarify" and self.question is None:
            raise ValueError("clarify disposition requires a question")


@dataclass(frozen=True)
class CoordinatorContext:
    request_id: str
    reason: Literal["turn", "worker_interaction"]
    owner_id: str
    turn: TurnEnvelope | None
    tasks: tuple[TaskRecord, ...]
    projects: tuple[ProjectRecord, ...]
    interactions: tuple[PendingInteraction, ...]
    contracts: tuple[SupervisionContractRevision, ...]
    provider_capabilities: dict[str, BackendCapabilities]
    worker_event: WorkerEvent | None = None


@dataclass(frozen=True)
class CoordinatorContextPolicy:
    """Backend-neutral bounds for one authoritative Coordinator snapshot."""

    max_active_tasks: int = 64
    max_recent_tasks: int = 8
    max_projects: int = 32
    max_interactions: int = 32
    max_record_text_chars: int = 512
    max_snapshot_chars: int = 96_000

    def __post_init__(self) -> None:
        values = (
            self.max_active_tasks,
            self.max_recent_tasks,
            self.max_projects,
            self.max_interactions,
            self.max_record_text_chars,
            self.max_snapshot_chars,
        )
        if min(values) < 1:
            raise ValueError("Coordinator context limits must be positive")


@dataclass(frozen=True)
class CoordinatorCallMetric:
    """One model-independent observation of a Coordinator call."""

    request_id: str
    owner_id: str
    reason: Literal["turn", "worker_interaction"]
    backend: str
    status: CoordinatorCallStatus
    elapsed_ms: float
    snapshot_chars: int
    task_count: int
    project_count: int
    interaction_count: int
    command_count: int = 0
    error: str = ""


class CoordinatorObserver(Protocol):
    def __call__(self, metric: CoordinatorCallMetric) -> None:
        """Observe a completed call without blocking the control path."""


def coordinator_snapshot(
    context: CoordinatorContext,
    policy: CoordinatorContextPolicy | None = None,
) -> dict[str, Any]:
    """Return the compact, authoritative snapshot shared by control models."""

    policy = policy or CoordinatorContextPolicy()
    clip = lambda value: _snapshot_clip(value, policy.max_record_text_chars)
    contracts = {
        item.task_id: {
            "revision": item.revision,
            "notify_policy": _snapshot_value(item.body.notify_policy, clip),
            "ask_policy": _snapshot_value(item.body.ask_policy, clip),
            "authority_policy": _snapshot_value(
                item.body.authority_policy, clip
            ),
            "delivery_policy": _snapshot_value(
                item.body.delivery_policy, clip
            ),
        }
        for item in context.contracts
    }
    tasks = [
        {
            "task_id": item.task_id,
            "project_id": item.project_id,
            "title": clip(item.title),
            "objective": clip(item.objective),
            "instruction": clip(item.instruction),
            "provider_name": item.provider_name,
            "reasoning_profile": item.reasoning_profile,
            "status": item.status,
            "generation": item.generation,
            "contract_revision": item.contract_revision,
            "active_run_id": item.active_run_id,
            "blocked_actions": list(item.blocked_actions),
            "blocked_branches": list(item.blocked_branches),
            "run_blocked": item.run_blocked,
            "result": clip(item.result),
            "error": clip(item.error),
            "updated_at_ms": item.updated_at_ms,
        }
        for item in context.tasks
    ]
    return {
        "schema_version": 2,
        "request_id": context.request_id,
        "reason": context.reason,
        "owner_id": context.owner_id,
        "turn": _turn_snapshot(context.turn, clip),
        "tasks": tasks,
        "projects": [
            {
                "project_id": item.project_id,
                "label": clip(item.label),
                "provider_name": item.provider_name,
                "environment_ref": clip(item.environment_ref),
                "status": item.status,
            }
            for item in context.projects
        ],
        "interactions": [
            {
                "interaction_id": item.interaction_id,
                "target_id": item.target_id,
                "task_id": item.task_id,
                "source": item.source,
                "audience": item.audience,
                "kind": item.kind,
                "prompt": clip(item.prompt),
                "reason_key": item.reason_key,
                "action_key": item.action_key,
                "choices": [clip(value) for value in item.choices],
                "requested_scope": item.requested_scope,
                "effective_scope": item.effective_scope,
                "state": item.state,
            }
            for item in context.interactions
        ],
        "contracts_by_task": contracts,
        "provider_capabilities": _snapshot_value(
            context.provider_capabilities, clip
        ),
        "worker_event": _snapshot_value(context.worker_event, clip),
    }


def render_coordinator_snapshot(
    context: CoordinatorContext,
    policy: CoordinatorContextPolicy | None = None,
) -> str:
    """Serialize a compact Coordinator snapshot and enforce its hard bound."""

    policy = policy or CoordinatorContextPolicy()
    raw = json.dumps(
        coordinator_snapshot(context, policy),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    if len(raw) > policy.max_snapshot_chars:
        raise ValueError(
            "authoritative Coordinator snapshot exceeds configured limit "
            f"({len(raw)} > {policy.max_snapshot_chars})"
        )
    return raw


def _turn_snapshot(
    turn: TurnEnvelope | None, clip: Any
) -> dict[str, Any] | None:
    if turn is None:
        return None
    return {
        "turn_id": turn.turn_id,
        "voice_session_id": turn.voice_session_id,
        "final_asr": clip(turn.final_asr),
        "media_refs": _snapshot_value(turn.media_refs, clip),
        "environment_ref": clip(turn.environment_ref),
        "context_revision": turn.context_revision,
        "timezone": turn.timezone,
        "frontbrain_action": turn.frontbrain_action,
        "frontbrain_task_name": clip(turn.frontbrain_task_name),
        "runtime_provider_name": turn.runtime_provider_name,
    }


def _snapshot_value(value: Any, clip: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {
            item.name: _snapshot_value(getattr(value, item.name), clip)
            for item in dataclasses.fields(value)
        }
    if isinstance(value, dict):
        return {
            str(key): _snapshot_value(item, clip)
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [_snapshot_value(item, clip) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_snapshot_value(item, clip) for item in value)
    if isinstance(value, str):
        return clip(value)
    return value


def _snapshot_clip(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + "\n[runtime: text clipped]"


@dataclass(frozen=True)
class CoordinatorSessionRecord:
    """Persistent mapping from one logical owner lane to an Agent session."""

    session_id: str
    owner_id: str
    provider_name: str
    backend_session_id: str
    turn_count: int = 0
    created_at_ms: int = field(default_factory=now_ms)
    updated_at_ms: int = field(default_factory=now_ms)

    def __post_init__(self) -> None:
        for name in (
            "session_id",
            "owner_id",
            "provider_name",
            "backend_session_id",
        ):
            _required(getattr(self, name), name)
        if self.turn_count < 0:
            raise ValueError("coordinator turn_count must be non-negative")


class Coordinator(Protocol):
    async def coordinate(self, context: CoordinatorContext) -> CoordinatorPlan:
        """Return a short, declarative plan. The Gateway validates every command."""


@dataclass(frozen=True)
class WorkerRequest:
    task_id: str
    run_id: str
    project_id: str
    owner_id: str
    generation: int
    instruction: str
    context_plan: ContextPlan
    policy: WorkerPolicyView
    reasoning_profile: ReasoningProfile = "balanced"
    source_turn: TurnEnvelope | None = None
    # Provider-neutral conversation key, stable across task_send continuations.
    lineage_id: str = ""
    kind: TaskKind = "main"
    parent_task_id: str = ""
    parent_run_id: str = ""
    # Opaque provider session; empty for stateless providers.
    parent_backend_session_id: str = ""
    # Original user utterance, stored separately from the mutable task instruction.
    original_turn: str = ""

    def __post_init__(self) -> None:
        if not self.lineage_id:
            object.__setattr__(self, "lineage_id", self.task_id)
        if not self.original_turn:
            object.__setattr__(self, "original_turn", self.instruction)
        for name in (
            "task_id",
            "lineage_id",
            "run_id",
            "project_id",
            "owner_id",
            "instruction",
            "original_turn",
        ):
            _required(getattr(self, name), name)
        if self.generation < 1:
            raise ValueError("worker generation must be positive")
        if self.reasoning_profile not in {"fast", "balanced", "deep"}:
            raise ValueError("invalid worker reasoning profile")
        if self.kind not in {"main", "side_query"}:
            raise ValueError("invalid worker request kind")
        if self.kind == "side_query":
            if not self.parent_task_id:
                raise ValueError("side query request requires parent_task_id")
            if not self.parent_run_id:
                raise ValueError("side query request requires parent_run_id")
        if self.kind == "main" and (
            self.parent_task_id
            or self.parent_run_id
            or self.parent_backend_session_id
        ):
            raise ValueError("main worker request must not set parent metadata")


@dataclass(frozen=True)
class WorkerMessage:
    message_id: str
    instruction: str
    mode: Literal["update", "query", "interaction_reply", "policy"]
    context_plan: ContextPlan | None = None
    interaction_id: str | None = None
    source_event_id: str | None = None
    decision: str | None = None
    policy: WorkerPolicyView | None = None
    source_turn: TurnEnvelope | None = None
    preempt: bool = False  # interrupt the active turn

    def __post_init__(self) -> None:
        _required(self.message_id, "message_id")
        if self.mode not in {
            "update",
            "query",
            "interaction_reply",
            "policy",
        }:
            raise ValueError("invalid worker message mode")
        if self.mode == "interaction_reply" and not self.interaction_id:
            raise ValueError("interaction reply requires interaction_id")
        if self.mode == "policy" and self.policy is None:
            raise ValueError("policy message requires a policy")


@dataclass(frozen=True)
class ProviderCommandRecord:
    command_id: str
    owner_id: str
    task_id: str
    run_id: str
    message: WorkerMessage
    state: Literal["pending", "sending", "sent", "failed", "cancelled"] = "pending"
    attempts: int = 0
    error: str = ""
    created_at_ms: int = field(default_factory=now_ms)
    sent_at_ms: int | None = None

    def __post_init__(self) -> None:
        for name in ("command_id", "owner_id", "task_id", "run_id"):
            _required(getattr(self, name), name)
        if self.command_id != self.message.message_id:
            raise ValueError("command_id must match WorkerMessage.message_id")
        if self.attempts < 0:
            raise ValueError("command attempts must be non-negative")


@dataclass(frozen=True)
class AuthorizationDecision:
    outcome: AuthorizationOutcome
    action: str
    reason: str
    interaction_id: str | None = None


@dataclass(frozen=True)
class ArbitrationDecision:
    timing: DeliveryTiming
    accepted: bool = True
    interaction_route: Literal["none", "user", "coordinator"] = "none"
    reason: str = ""


@dataclass(frozen=True)
class TaskControlResult:
    request_id: str
    disposition: Literal["answered", "accepted", "clarify", "rejected"]
    speech: str = ""
    task_ids: tuple[str, ...] = ()
    interaction_id: str | None = None
    reason_key: str = ""
    # Synchronous task-tool outcome.
    status: TaskControlStatus = "ok"
    # Candidate task names when status is ambiguous.
    candidates: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_ids", _tuple(self.task_ids))
        object.__setattr__(self, "candidates", _tuple(self.candidates))
        if self.status not in TaskControlStatus.__args__:
            raise ValueError("invalid task-control status")
        if self.status == "ambiguous" and not self.candidates:
            raise ValueError("ambiguous task-control result requires candidates")
        if self.status != "ambiguous" and self.candidates:
            raise ValueError("task-control candidates require ambiguous status")


@dataclass(frozen=True)
class CoordinationJob:
    """The single current background Coordinator request for one owner."""

    owner_id: str
    request_id: str
    receipt_key: str
    turn_id: str
    voice_session_id: str
    version: int
    turn_receipt_keys: tuple[str, ...] = ()
    state: Literal["pending", "coordinating", "completed", "failed"] = "pending"
    result: TaskControlResult | None = None
    error: str = ""
    created_at_ms: int = field(default_factory=now_ms)
    updated_at_ms: int = field(default_factory=now_ms)

    def __post_init__(self) -> None:
        for name in (
            "owner_id",
            "request_id",
            "receipt_key",
            "turn_id",
            "voice_session_id",
        ):
            _required(getattr(self, name), name)
        if self.version < 1:
            raise ValueError("coordination job version must be positive")
        keys = _tuple(self.turn_receipt_keys) or (self.receipt_key,)
        object.__setattr__(self, "turn_receipt_keys", keys)
        _unique(keys, "coordination turn receipt keys")
        if keys[-1] != self.receipt_key:
            raise ValueError("latest coordination receipt key must be last")
        if self.state not in {
            "pending",
            "coordinating",
            "completed",
            "failed",
        }:
            raise ValueError("invalid coordination job state")
        if self.state in {"completed", "failed"} and self.result is None:
            raise ValueError("terminal coordination job requires a result")
        if self.state in {"pending", "coordinating"} and self.result is not None:
            raise ValueError("active coordination job cannot have a result")


@dataclass(frozen=True)
class TaskSlateEntry:
    """Task name and status shown to the front brain for reference resolution."""

    name: str
    status_line: str
    done: bool = False


def coarse_task_status(status: str) -> str:
    """Map task lifecycle state to running, finished, or failed."""
    if status in ("completed", "partial"):
        return "finished"
    if status in ("failed", "cancelled"):
        return "failed"
    return "running"


def render_task_slate(
    entries: tuple[TaskSlateEntry, ...], *, flat: bool = False
) -> str:
    """Render the live task slate for front-brain coreference.

    Flat mode emits one ``name: status`` line per task. Coordinator mode separates
    active and recently completed tasks.
    """

    if not entries:
        return "当前没有正在进行的后台任务。"
    if flat:
        return "\n".join(f"{e.name}: {e.status_line}" for e in entries)

    active = [e for e in entries if not e.done]
    done = [e for e in entries if e.done]
    if not active and not done:
        return "当前没有正在进行的后台任务。"
    blocks: list[str] = []
    if active:
        blocks.append(
            "当前进行中(可按名字引用):\n"
            + "\n".join(f"- {e.name}:{e.status_line}" for e in active)
        )
    if done:
        blocks.append(
            "最近完成(仍可按名字继续):\n"
            + "\n".join(f"- {e.name}:{e.status_line}" for e in done)
        )
    return "\n".join(blocks)


def new_worker_message(
    instruction: str,
    mode: Literal["update", "query", "interaction_reply", "policy"],
    *,
    message_id: str | None = None,
    **kwargs: Any,
) -> WorkerMessage:
    return WorkerMessage(
        message_id or new_id("message"), instruction, mode, **kwargs
    )
