from __future__ import annotations

import asyncio
import dataclasses
import inspect
import json
import logging
import re
import time
from collections.abc import AsyncIterator, Callable, Sequence
from typing import Any, Literal, Protocol

from .contracts import (
    EPHEMERAL_VISUAL_CONTEXT_KINDS,
    ContextEvent,
    new_id,
    now_ms,
    stable_key,
    stable_id,
)
from .lean_control import (
    ActiveTask,
    assign_frontbrain_task_name,
    normalize_frontbrain_task_name,
    resolve_target,
)
from .coordination import (
    ASK_REASONS,
    AUTHORITY_ACTIONS,
    ActivityAggregate,
    ArtifactEvidence,
    TaskControlResult,
    AuthorizationDecision,
    coarse_task_status,
    BackendCapabilities,
    CancelTaskCommand,
    ArbitrationDecision,
    Coordinator,
    CoordinatorCallMetric,
    CoordinatorContext,
    CoordinatorContextPolicy,
    CoordinationJob,
    CoordinatorObserver,
    CoordinatorPlan,
    ContextInventory,
    ContextInventoryItem,
    ContextPlan,
    ContractBody,
    CreateTaskCommand,
    DeliveryRecord,
    DonePayload,
    InteractionPayload,
    PendingInteraction,
    ProviderCommandRecord,
    ProjectRecord,
    ResolveInteractionCommand,
    RunRecord,
    SendTaskCommand,
    SupervisionContractRevision,
    TASK_LANES,
    TASK_RESOLVE_ACTIONS,
    TaskRecord,
    TaskLane,
    TaskResolveAction,
    TaskSlateEntry,
    TurnEnvelope,
    UpdateContractCommand,
    UpdatePayload,
    WorkerEvent,
    WorkerMessage,
    WorkerPolicyView,
    WorkerRequest,
    new_worker_message,
    render_coordinator_snapshot,
)
from .supervision import (
    DeliveryBroker,
    InteractionBroker,
    SupervisionEngine,
    TaskLedger,
    effective_scope,
)
from .worker_tools import (
    MemoryHit,
    MemoryProvider,
    validate_tool_arguments,
)

LOGGER = logging.getLogger(__name__)

REPORT_EVENT_TOOL_SCHEMA: dict[str, Any] = {
    "name": "report_event",
    "description": "Report structured worker progress, interaction, or completion.",
    "parameters": {
        "type": "object",
        "properties": {
            "report_id": {"type": "string"},
            "type": {"enum": ["update", "interaction", "done"]},
            "kind": {
                "enum": [
                    "activity",
                    "milestone",
                    "question",
                    "choice",
                    "permission",
                ]
            },
            "summary": {"type": "string"},
            "milestone": {"type": "string"},
            "evidence": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["ref", "facts"],
                    "properties": {
                        "ref": {"type": "string"},
                        "facts": {"type": "object"},
                    },
                    "additionalProperties": False,
                },
            },
            "next_step": {"type": "string"},
            "severity": {"enum": ["normal", "high_risk"]},
            "prompt": {"type": "string"},
            "blocking_scope": {"enum": ["action", "branch", "run"]},
            "choices": {
                "type": "array",
                "items": {"type": "string"},
            },
            "reason_key": {"type": "string", "enum": sorted(ASK_REASONS)},
            "action_key": {
                "type": "string",
                "enum": sorted(AUTHORITY_ACTIONS),
            },
            "status": {
                "enum": ["completed", "partial", "failed", "cancelled"]
            },
            "result": {"type": "string"},
            "artifact_refs": {
                "type": "array",
                "items": {"type": "string"},
            },
            "unresolved": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": ["report_id", "type"],
        "additionalProperties": False,
    },
}


class CoordinatorPlanError(ValueError):
    """The Coordinator proposed a command the authoritative state rejects."""

    def __init__(self, message: str, *, reason_key: str = "unroutable") -> None:
        super().__init__(message)
        self.reason_key = reason_key


class WorkerRun(Protocol):
    session_id: str

    def events(self) -> AsyncIterator[WorkerEvent]:
        ...

    async def send(self, message: WorkerMessage) -> bool:
        """Return False when a competing client already resolved the message."""

    async def cancel(self, request_id: str) -> bool:
        """Idempotently cancel; return True only when already terminal."""

    async def close(self) -> None:
        ...


class WorkerProject(Protocol):
    async def start(
        self, request: WorkerRequest, control: WorkerControl
    ) -> WorkerRun:
        ...

    async def close(self) -> None:
        ...


class WorkerProvider(Protocol):
    name: str
    capabilities: BackendCapabilities

    async def open_project(self, project: ProjectRecord) -> WorkerProject:
        ...

    async def close(self) -> None:
        ...


class ProjectResourceKeyProvider(Protocol):
    """Optional provider hook for serializing shared physical resources."""

    def project_resource_key(self, project: ProjectRecord) -> str:
        ...


class ProviderRegistry:
    """Validated provider lookup; scheduling remains a Gateway concern."""

    def __init__(self, providers: tuple[WorkerProvider, ...] = ()) -> None:
        self._providers: dict[str, WorkerProvider] = {}
        for provider in providers:
            self.register(provider)

    def register(self, provider: WorkerProvider) -> None:
        name = getattr(provider, "name", "")
        capabilities = getattr(provider, "capabilities", None)
        if not name:
            raise ValueError("worker provider requires a name")
        if not isinstance(capabilities, BackendCapabilities):
            raise TypeError("worker provider requires BackendCapabilities")
        if name in self._providers:
            raise ValueError(f"duplicate worker provider: {name}")
        self._providers[name] = provider

    def get(self, name: str) -> WorkerProvider:
        try:
            return self._providers[name]
        except KeyError:
            raise KeyError(f"unknown worker provider: {name}") from None

    def capabilities(self) -> dict[str, BackendCapabilities]:
        return {
            name: provider.capabilities
            for name, provider in self._providers.items()
        }

    def __contains__(self, name: str) -> bool:
        return name in self._providers

    def values(self) -> tuple[WorkerProvider, ...]:
        return tuple(self._providers.values())


class WorkerControl:
    """Provider hooks for authorization and evidence recording."""

    def __init__(
        self,
        gateway: GanderGateway,
        task_id: str,
        run_id: str,
        capabilities: BackendCapabilities,
    ) -> None:
        self._gateway = gateway
        self.task_id = task_id
        self.run_id = run_id
        self.capabilities = capabilities

    async def record_evidence(
        self, ref: str, facts: dict[str, Any]
    ) -> None:
        await self._gateway.record_evidence(
            task_id=self.task_id, ref=ref, facts=facts
        )

    async def fetch_artifact(self, ref: str) -> ArtifactEvidence:
        return await self._gateway.fetch_artifact(
            task_id=self.task_id, ref=ref
        )

    async def fetch_turn(
        self,
        turn_id: str,
        *,
        voice_session_id: str | None = None,
    ) -> TurnEnvelope:
        return await self._gateway.fetch_turn(
            task_id=self.task_id,
            turn_id=turn_id,
            voice_session_id=voice_session_id,
        )

    async def memory_search(
        self,
        query: str,
        *,
        scope: str = "owner",
        limit: int = 8,
    ) -> dict[str, Any]:
        arguments = validate_tool_arguments(
            "memory_search",
            {"query": query, "scope": scope, "limit": limit},
        )
        return await self._gateway.memory_search(
            task_id=self.task_id,
            run_id=self.run_id,
            **arguments,
        )

    async def context_fetch(
        self,
        *,
        refs: tuple[str, ...] = (),
        query: str = "",
        kinds: tuple[str, ...] = (),
        context_kinds: tuple[str, ...] = (),
        roles: tuple[str, ...] = (),
        last_ms: int | None = None,
        limit: int = 20,
        max_chars: int = 12_000,
        cursor: int = 0,
        include_media: bool = False,
    ) -> dict[str, Any]:
        arguments = validate_tool_arguments(
            "context_fetch",
            {
                "refs": list(refs),
                "query": query,
                "kinds": list(kinds),
                "context_kinds": list(context_kinds),
                "roles": list(roles),
                "last_ms": last_ms,
                "limit": limit,
                "max_chars": max_chars,
                "cursor": cursor,
                "include_media": include_media,
            },
        )
        return await self._gateway.context_fetch(
            task_id=self.task_id,
            run_id=self.run_id,
            **arguments,
        )

    async def authorize(
        self,
        action: str,
        *,
        request_key: str,
        prompt: str,
        blocking_scope: str = "action",
        side_effect: bool = True,
        high_risk: bool = False,
    ) -> AuthorizationDecision:
        """Return an idempotent, action-attempt-scoped authorization ticket."""

        if not request_key:
            raise ValueError("authorization request_key must be non-empty")
        return await self._gateway.authorize(
            task_id=self.task_id,
            run_id=self.run_id,
            action=action,
            request_key=request_key,
            prompt=prompt,
            blocking_scope=blocking_scope,
            side_effect=side_effect,
            high_risk=high_risk,
        )


def _build_worker_event(
    request: WorkerRequest,
    seq: int,
    payload: UpdatePayload | InteractionPayload | DonePayload,
    report_ref: str,
) -> WorkerEvent:
    event_type = (
        "update"
        if isinstance(payload, UpdatePayload)
        else "interaction"
        if isinstance(payload, InteractionPayload)
        else "done"
    )
    return WorkerEvent(
        event_id=stable_id("event", request.run_id, report_ref),
        owner_id=request.owner_id,
        task_id=request.task_id,
        project_id=request.project_id,
        run_id=request.run_id,
        generation=request.generation,
        seq=seq,
        type=event_type,
        payload=payload,
    )


class WorkerEventChannel:
    """Channel for native worker events or the structured report tool."""

    tool_schema = REPORT_EVENT_TOOL_SCHEMA

    def __init__(
        self,
        request: WorkerRequest,
        control: WorkerControl,
        *,
        max_pending_events: int = 256,
    ) -> None:
        if max_pending_events < 1:
            raise ValueError("max_pending_events must be positive")
        self.request = request
        self.control = control
        self._queue: asyncio.Queue[WorkerEvent] = asyncio.Queue(
            max_pending_events
        )
        self._reports: dict[str, WorkerEvent] = {}
        self._seq = 0
        self._terminal = False
        self._subscribed = False

    async def report(self, arguments: dict[str, Any]) -> WorkerEvent:
        """Validate one ``report_event`` tool call and enqueue it once."""

        if not isinstance(arguments, dict):
            raise TypeError("report_event arguments must be an object")
        report_id = arguments.get("report_id")
        event_type = arguments.get("type")
        if not isinstance(report_id, str) or not report_id:
            raise ValueError("report_id must be a non-empty string")
        if self._terminal and report_id not in self._reports:
            raise RuntimeError("worker event channel is already terminal")
        allowed = {
            "update": {
                "report_id",
                "type",
                "kind",
                "summary",
                "milestone",
                "evidence",
                "next_step",
                "severity",
            },
            "interaction": {
                "report_id",
                "type",
                "kind",
                "prompt",
                "blocking_scope",
                "choices",
                "reason_key",
                "action_key",
            },
            "done": {
                "report_id",
                "type",
                "status",
                "result",
                "artifact_refs",
                "unresolved",
            },
        }.get(event_type)
        if allowed is None:
            raise ValueError("report_event type must be update, interaction, or done")
        unknown = set(arguments) - allowed
        if unknown:
            raise ValueError(
                f"unexpected {event_type} fields: {sorted(unknown)}"
            )
        if event_type == "update":
            evidence_refs: list[str] = []
            for evidence in arguments.get("evidence", ()):
                if (
                    not isinstance(evidence, dict)
                    or set(evidence) != {"ref", "facts"}
                    or not isinstance(evidence["facts"], dict)
                ):
                    raise ValueError("invalid structured evidence")
                await self.control.record_evidence(
                    evidence["ref"], evidence["facts"]
                )
                evidence_refs.append(evidence["ref"])
            payload: Any = UpdatePayload(
                kind=arguments.get("kind"),
                summary=arguments.get("summary", ""),
                milestone=arguments.get("milestone"),
                evidence_refs=tuple(evidence_refs),
                next_step=arguments.get("next_step", ""),
                severity=arguments.get("severity", "normal"),
            )
        elif event_type == "interaction":
            payload = InteractionPayload(
                kind=arguments.get("kind"),
                prompt=arguments.get("prompt", ""),
                blocking_scope=arguments.get("blocking_scope", "run"),
                choices=tuple(arguments.get("choices", ())),
                reason_key=arguments.get(
                    "reason_key", "critical_input.missing"
                ),
                action_key=arguments.get("action_key"),
            )
        else:
            payload = DonePayload(
                status=arguments.get("status"),
                result=arguments.get("result", ""),
                artifact_refs=tuple(arguments.get("artifact_refs", ())),
                unresolved=tuple(arguments.get("unresolved", ())),
            )
        return await self.publish(payload, report_id=report_id)

    async def publish(
        self,
        payload: UpdatePayload | InteractionPayload | DonePayload,
        *,
        report_id: str | None = None,
    ) -> WorkerEvent:
        report_id = report_id or new_id("report")
        existing = self._reports.get(report_id)
        if existing is not None:
            if existing.payload != payload:
                raise ValueError(f"report_id collision: {report_id}")
            return existing
        if self._terminal:
            raise RuntimeError("worker event channel is already terminal")
        self._seq += 1
        event = _build_worker_event(
            self.request, self._seq, payload, report_id
        )
        self._reports[report_id] = event
        self._terminal = isinstance(payload, DonePayload)
        await self._queue.put(event)
        return event

    async def fail(self, error: str) -> WorkerEvent | None:
        if self._terminal:
            return None
        return await self.publish(DonePayload("failed", error))

    def events(self) -> AsyncIterator[WorkerEvent]:
        if self._subscribed:
            raise RuntimeError("worker event channel supports one subscriber")
        self._subscribed = True

        async def iterate() -> AsyncIterator[WorkerEvent]:
            while True:
                event = await self._queue.get()
                yield event
                if isinstance(event.payload, DonePayload):
                    return

        return iterate()


class WorkerRunChannel:
    def __init__(
        self,
        request: WorkerRequest,
        control: WorkerControl,
        provider_run: Any,
        capabilities: BackendCapabilities,
        result_callback: Callable[[Any], None] | None = None,
    ) -> None:
        self.request = request
        self.control = control
        self.provider_run = provider_run
        self.capabilities = capabilities
        self._result_callback = result_callback
        self.session_id = str(getattr(provider_run, "thread_id", "") or "")
        self._seq = 0
        self._context_seq = 1
        self._events_by_ref: dict[str, WorkerEvent] = {}
        self._native_interactions: dict[str, str] = {}
        self._terminal = False
        self._cancelled = asyncio.Event()

    def events(self) -> AsyncIterator[WorkerEvent]:
        async def iterate() -> AsyncIterator[WorkerEvent]:
            iterator = self.provider_run.events().__aiter__()
            while True:
                next_event = asyncio.create_task(anext(iterator))
                cancelled = asyncio.create_task(self._cancelled.wait())
                done, pending = await asyncio.wait(
                    (next_event, cancelled),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for job in pending:
                    job.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
                if cancelled in done and cancelled.result():
                    return
                try:
                    provider_event = next_event.result()
                except StopAsyncIteration:
                    break
                evidence_refs = await self._record_artifacts(provider_event)
                mapped = self._map_event(
                    provider_event, evidence_refs=evidence_refs
                )
                if mapped is None:
                    continue
                yield mapped
                if isinstance(mapped.payload, DonePayload):
                    self._terminal = True
                    return
            if not self._terminal:
                yield self._event(
                    DonePayload(
                        "failed",
                        "provider stream ended without a result",
                    ),
                    new_id("provider_stream_end"),
                )

        return iterate()

    async def _record_artifacts(
        self, provider_event: Any
    ) -> tuple[str, ...]:
        artifacts: Any = ()
        if provider_event.kind == "result" and provider_event.result is not None:
            artifacts = provider_event.result.artifacts
        elif provider_event.kind == "share" and provider_event.share is not None:
            patch = provider_event.share.state_patch
            artifacts = patch.get("artifacts", ()) if isinstance(patch, dict) else ()
        evidence_refs: list[str] = []
        for artifact in artifacts or ():
            if not isinstance(artifact, dict):
                continue
            ref = artifact.get("ref") or artifact.get("path")
            if isinstance(ref, str) and ref:
                facts = {**dict(artifact), "provider_ref": ref}
                canonical_facts = json.dumps(
                    facts,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    default=str,
                )
                fact_key = stable_key(canonical_facts)
                evidence_ref = (
                    f"worker-artifact:{self.request.task_id}:{fact_key}"
                )
                await self.control.record_evidence(evidence_ref, facts)
                evidence_refs.append(evidence_ref)
        return tuple(evidence_refs)

    async def send(self, message: WorkerMessage) -> bool:
        from .contracts import (
            ContextSnapshot,
            TaskInteractionReply,
            TaskQuery,
            TaskUpdate,
        )

        if message.mode == "policy":
            return True
        event = self._context_event(
            message.instruction, message.source_turn
        )
        snapshot = ContextSnapshot(
            session_id=self.request.lineage_id,
            events=(
                ()
                if self.capabilities.context_provisioning == "pull"
                else (event,)
            ),
        )
        if message.mode == "update":
            await self.provider_run.steer(
                TaskUpdate(
                    # Preemption replaces the active turn; soft steer appends to it.
                    mode="replace" if message.preempt else "additive",
                    event=event,
                    instruction=message.instruction,
                    context=snapshot,
                    update_id=message.message_id,
                )
            )
            return True
        if message.mode == "query":
            if self.capabilities.side_queries == "none":
                return False
            await self.provider_run.query(
                TaskQuery(
                    task_id=self.request.task_id,
                    session_id=self.request.lineage_id,
                    generation=self.request.generation,
                    question=message.instruction,
                    context=snapshot,
                    request_id=message.message_id,
                )
            )
            return True
        native_id = self._native_interactions.get(
            message.source_event_id or ""
        )
        if native_id is None:
            return False
        return await self.provider_run.respond(
            TaskInteractionReply(
                task_id=self.request.task_id,
                session_id=self.request.lineage_id,
                generation=self.request.generation,
                interaction_id=native_id,
                text=message.instruction,
                decision=message.decision,
            )
        )

    async def cancel(self, request_id: str) -> bool:
        del request_id
        await self.provider_run.cancel()
        self._terminal = True
        self._cancelled.set()
        return True

    async def close(self) -> None:
        self._cancelled.set()
        await self.provider_run.close()

    def _map_event(
        self,
        provider_event: Any,
        *,
        evidence_refs: tuple[str, ...] = (),
    ) -> WorkerEvent | None:
        if provider_event.generation != self.request.generation:
            return None
        if provider_event.kind == "share" and provider_event.share is not None:
            share = provider_event.share
            payload: Any = UpdatePayload(
                "activity",
                share.text,
                evidence_refs=evidence_refs,
                next_step=str(
                    share.state_patch.get("next_step", "")
                    if isinstance(share.state_patch, dict)
                    else ""
                ),
            )
            return self._event(payload, f"share:{share.share_id}")
        if (
            provider_event.kind == "interaction"
            and provider_event.interaction is not None
        ):
            interaction = provider_event.interaction
            choices = tuple(interaction.choices)
            questions = interaction.questions
            if (
                interaction.metadata.get("sequential_questions")
                and len(questions) > 1
            ):
                questions = questions[:1]
            for question in questions:
                choices += tuple(option.label for option in question.options)
            if interaction.kind == "approval":
                kind = "permission"
                requested_action = str(
                    interaction.metadata.get(
                        "action_key", "destructive_action"
                    )
                )
                action_key = (
                    requested_action
                    if requested_action in AUTHORITY_ACTIONS
                    else "destructive_action"
                )
                reason_key = "critical_input.missing"
            else:
                kind = "choice" if len(choices) >= 2 else "question"
                action_key = None
                reason_key = str(
                    interaction.metadata.get(
                        "reason_key", "critical_input.missing"
                    )
                )
            payload = InteractionPayload(
                kind,
                interaction.prompt,
                choices=choices,
                reason_key=reason_key,
                action_key=action_key,
            )
            event = self._event(
                payload, f"interaction:{interaction.interaction_id}"
            )
            self._native_interactions[event.event_id] = (
                interaction.interaction_id
            )
            return event
        if provider_event.kind == "result" and provider_event.result is not None:
            result = provider_event.result
            if self._result_callback is not None:
                self._result_callback(result)
            status = (
                result.status
                if result.status
                in {"completed", "partial", "failed", "cancelled"}
                else "failed"
            )
            artifacts = evidence_refs or tuple(
                str(item.get("ref") or item.get("path"))
                for item in result.artifacts
                if item.get("ref") or item.get("path")
            )
            return self._event(
                DonePayload(
                    status,
                    result.full_result,
                    artifact_refs=artifacts,
                    unresolved=tuple(result.unresolved),
                ),
                f"result:{result.generation}",
            )
        if provider_event.kind == "error":
            return self._event(
                DonePayload(
                    "failed", provider_event.error or "provider error"
                ),
                new_id("provider_error"),
            )
        if provider_event.kind == "interaction_resolved":
            return self._event(
                UpdatePayload("activity", "Worker interaction resolved"),
                (
                    "interaction-resolved:"
                    f"{provider_event.interaction_id or new_id('unknown')}"
                ),
            )
        return None

    def _event(self, payload: Any, report_ref: str) -> WorkerEvent:
        existing = self._events_by_ref.get(report_ref)
        if existing is not None:
            return existing
        self._seq += 1
        event = _build_worker_event(
            self.request, self._seq, payload, report_ref
        )
        self._events_by_ref[report_ref] = event
        return event

    def _context_event(
        self, text: str, turn: TurnEnvelope | None
    ) -> Any:
        from .contracts import ContextEvent

        self._context_seq += 1
        return ContextEvent(
            session_id=self.request.lineage_id,
            seq=self._context_seq,
            role="user",
            kind="user_text",
            text=turn.final_asr if turn is not None else text,
            media=turn.media_refs if turn is not None else (),
            metadata={"worker_message": True},
        )

class GanderGateway:
    """Persistent Coordinator, ledger, supervision, and project scheduler."""

    def __init__(
        self,
        coordinator: Coordinator | None,
        providers: ProviderRegistry,
        *,
        ledger: TaskLedger | None = None,
        memory_provider: MemoryProvider | None = None,
        memory_search_timeout_s: float = 10.0,
        max_memory_result_chars: int = 16_000,
        max_context_media_refs: int = 32,
        max_realtime_context_events: int = 512,
        screen_context_window_ms: int = 8_000,
        permission_timeout_ms: int = 300_000,
        max_queued_tasks_per_owner: int = 64,
        terminal_task_ttl_ms: int = 3_600_000,  # retain finished tasks for about 1 h
        max_terminal_tasks_per_owner: int = 20,
        recent_task_window_ms: int = 600_000,  # reference window for finished tasks
        coordinator_timeout_s: float = 20.0,
        provider_command_timeout_s: float = 30.0,
        provider_start_timeout_s: float = 60.0,
        context_policy: CoordinatorContextPolicy | None = None,
        coordinator_observer: CoordinatorObserver | None = None,
        mode: Literal["coordinator", "lean"] = "coordinator",
    ) -> None:
        if permission_timeout_ms < 1:
            raise ValueError("permission_timeout_ms must be positive")
        if mode not in {"coordinator", "lean"}:
            raise ValueError("mode must be 'coordinator' or 'lean'")
        if mode == "coordinator" and coordinator is None:
            raise ValueError("coordinator mode requires a coordinator (LLM middle-layer)")
        if max_queued_tasks_per_owner < 1:
            raise ValueError("max_queued_tasks_per_owner must be positive")
        if memory_search_timeout_s <= 0:
            raise ValueError("memory_search_timeout_s must be positive")
        if max_memory_result_chars < 256:
            raise ValueError("max_memory_result_chars must be at least 256")
        if max_context_media_refs < 0:
            raise ValueError("max_context_media_refs must be non-negative")
        if max_realtime_context_events < 1:
            raise ValueError("max_realtime_context_events must be positive")
        if screen_context_window_ms < 1:
            raise ValueError("screen_context_window_ms must be positive")
        if min(
            coordinator_timeout_s,
            provider_command_timeout_s,
            provider_start_timeout_s,
        ) <= 0:
            raise ValueError("Gateway timeouts must be positive")
        self.coordinator = coordinator
        self.providers = providers
        self.ledger = ledger or TaskLedger()
        self.memory_provider = memory_provider
        self.memory_search_timeout_s = memory_search_timeout_s
        self.max_memory_result_chars = max_memory_result_chars
        self.max_context_media_refs = max_context_media_refs
        self.max_realtime_context_events = max_realtime_context_events
        self.screen_context_window_ms = screen_context_window_ms
        self.permission_timeout_ms = permission_timeout_ms
        self.max_queued_tasks_per_owner = max_queued_tasks_per_owner
        self.mode = mode
        self.terminal_task_ttl_ms = terminal_task_ttl_ms
        self.max_terminal_tasks_per_owner = max_terminal_tasks_per_owner
        self.recent_task_window_ms = recent_task_window_ms
        self.coordinator_timeout_s = coordinator_timeout_s
        self.provider_command_timeout_s = provider_command_timeout_s
        self.provider_start_timeout_s = provider_start_timeout_s
        inferred_policy = getattr(coordinator, "context_policy", None)
        self.context_policy = (
            context_policy
            or inferred_policy
            or CoordinatorContextPolicy()
        )
        if not isinstance(self.context_policy, CoordinatorContextPolicy):
            raise TypeError(
                "context_policy must be a CoordinatorContextPolicy"
            )
        self.coordinator_observer = coordinator_observer
        self.supervision = SupervisionEngine(self.ledger)
        self.interactions = InteractionBroker(self.ledger)
        self.delivery = DeliveryBroker(self.ledger)
        self._owner_locks: dict[str, asyncio.Lock] = {}
        self._coordinator_model_locks: dict[str, asyncio.Lock] = {}
        self._task_locks: dict[str, asyncio.Lock] = {}
        self._resource_locks: dict[str, asyncio.Lock] = {}
        self._project_open_locks: dict[str, asyncio.Lock] = {}
        self._provider_slots = {
            provider.name: asyncio.Semaphore(
                provider.capabilities.max_parallel_projects
            )
            for provider in providers.values()
        }
        self._project_sessions: dict[str, WorkerProject] = {}
        self._runtime_runs: dict[str, WorkerRun] = {}
        # Session-scoped permission grants and delivery preferences.
        self._standing_allow: set[str] = set()
        self._muted: set[str] = set()
        self._interrupt_subscribed: set[str] = set()
        self._task_jobs: dict[str, asyncio.Task[None]] = {}
        self._coordination_jobs: dict[str, asyncio.Task[None]] = {}
        self._coordination_finished: dict[str, asyncio.Event] = {}
        self._auxiliary_jobs: set[asyncio.Task[Any]] = set()
        self._aggregate_jobs: dict[str, asyncio.Task[None]] = {}
        self._finished: dict[str, asyncio.Event] = {}
        self._start_lock = asyncio.Lock()
        self._instance_id = new_id("gateway")
        self._started = False
        self._closed = False

    async def start(self) -> None:
        """Replay committed events and recover schedulable work once."""

        async with self._start_lock:
            if self._started:
                return
            if self._closed:
                raise RuntimeError("Gateway is closed")
            warmup = getattr(self.coordinator, "warmup", None)
            if callable(warmup):
                result = warmup()
                if inspect.isawaitable(result):
                    await asyncio.wait_for(
                        result, self.provider_start_timeout_s
                    )
            self.ledger.reset_sending_provider_commands()
            self.ledger.recover_coordination_jobs()
            for interaction in self.ledger.expire_pending_permissions():
                self.ledger.cancel_provider_commands_for_interaction(
                    interaction.interaction_id
                )
                self.ledger.cancel_interaction_deliveries(
                    interaction.interaction_id
                )
                self._queue_expired_interaction(interaction)
                if interaction.task_id:
                    self._refresh_blocked(interaction.task_id)
            self._finish_sent_interactions()
            self._recover_resolving_interactions()
            for event in self.ledger.list_events(processed=False):
                await self._reduce_event(event, recovering=True)
                self.ledger.mark_event_processed(event.event_id)
            for task in self.ledger.list_tasks():
                if task.terminal:
                    self._finished_event(task.task_id).set()
                    continue
                if task.status == "queued":
                    self._schedule(task.task_id)
                    continue
                run = (
                    self.ledger.get_run(task.active_run_id)
                    if task.active_run_id
                    else None
                )
                capabilities = self.providers.capabilities().get(
                    task.provider_name
                )
                project = self.ledger.get_project(task.project_id)
                resumable = (
                    run is not None
                    and project is not None
                    and capabilities is not None
                    and capabilities.session_resume
                    and bool(run.backend_session_id)
                    and task.provider_name in self.providers
                )
                if resumable:
                    self._schedule(task.task_id, run)
                else:
                    await self._fail_task(
                        task,
                        run,
                        "Gateway restarted but the backend run cannot resume",
                    )
            for aggregate in self.ledger.list_activity_aggregates(
                state="open"
            ):
                self._schedule_aggregate(aggregate)
            self._started = True
            for job in self.ledger.list_coordination_jobs():
                if job.state == "pending":
                    self._coordination_finished.setdefault(
                        job.request_id, asyncio.Event()
                    )
                    self._schedule_coordination(job.owner_id)

    async def coordinate_turn(self, turn: TurnEnvelope) -> TaskControlResult:
        if self.mode == "lean":
            raise RuntimeError(
                "coordinate_turn() requires the LLM coordinator; lean mode is "
                "driven by task_start/task_send/task_resolve"
            )
        await self.start()
        await self.expire_interactions()
        lock = self._owner_locks.setdefault(turn.owner_id, asyncio.Lock())
        async with lock:
            request_id = stable_id(
                "coordination", turn.receipt_key
            )
            previous = self.ledger.get_coordination_job(turn.owner_id)
            receipt, job, created = self.ledger.submit_coordination(
                turn, request_id
            )
            if not created:
                return receipt
            if (
                previous is not None
                and previous.request_id != request_id
            ):
                self._signal_coordination(previous.request_id)
            self._coordination_event(request_id).clear()
            if job is not None:
                self._schedule_coordination(job.owner_id)
            return receipt

    async def coordinate_turn_and_wait(
        self, turn: TurnEnvelope, timeout: float | None = None
    ) -> TaskControlResult:
        """Submit one coordinator turn and wait for its bounded result."""

        receipt = await self.coordinate_turn(turn)
        if receipt.reason_key != "coordination.pending":
            return receipt
        return await self.wait_coordination(
            owner_id=turn.owner_id,
            request_id=receipt.request_id,
            timeout=timeout,
        )

    async def coordinate_task_start(
        self,
        *,
        owner_id: str,
        turn: TurnEnvelope,
        name: str,
        provider_name: str | None = None,
        timeout: float | None = None,
    ) -> TaskControlResult:
        """Compile task_start into a coordinator contract.

        The bound turn supplies the objective; name is its stable front-brain handle.
        """

        if self.mode != "coordinator":
            raise RuntimeError(
                "coordinate_task_start() requires coordinator mode"
            )
        self._validate_bound_turn(owner_id, turn)
        normalized_name = normalize_frontbrain_task_name(name)
        routed_turn = dataclasses.replace(
            turn,
            frontbrain_action="task_start",
            frontbrain_task_name=normalized_name,
            runtime_provider_name=(provider_name or "").strip(),
        )
        result = await self.coordinate_turn_and_wait(routed_turn, timeout)
        if result.disposition == "accepted" and len(result.task_ids) == 1:
            task = self.ledger.get_task(result.task_ids[0])
            if task is not None and task.owner_id == owner_id:
                return dataclasses.replace(
                    result,
                    status="ok",
                    speech=task.name or task.title,
                )
        return dataclasses.replace(
            result,
            status=(
                "unsupported"
                if result.disposition == "rejected"
                else "invalid_action"
            ),
            speech="",
        )

    async def wait_coordination(
        self,
        *,
        owner_id: str,
        request_id: str,
        timeout: float | None = None,
    ) -> TaskControlResult:
        job = self.ledger.get_coordination_job(owner_id)
        if job is not None and job.request_id == request_id:
            if job.result is not None:
                return job.result
        elif job is not None:
            return TaskControlResult(
                request_id=request_id,
                disposition="accepted",
                reason_key="coordination.superseded",
            )
        event = self._coordination_event(request_id)
        if timeout is None:
            await event.wait()
        else:
            await asyncio.wait_for(event.wait(), timeout)
        job = self.ledger.get_coordination_job(owner_id)
        if (
            job is not None
            and job.request_id == request_id
            and job.result is not None
        ):
            return job.result
        return TaskControlResult(
            request_id=request_id,
            disposition="accepted",
            reason_key="coordination.superseded",
        )

    def _coordination_event(self, request_id: str) -> asyncio.Event:
        return self._coordination_finished.setdefault(
            request_id, asyncio.Event()
        )

    def _signal_coordination(self, request_id: str) -> None:
        event = self._coordination_finished.pop(request_id, None)
        if event is not None:
            event.set()

    def _schedule_coordination(self, owner_id: str) -> None:
        current = self._coordination_jobs.get(owner_id)
        if current is not None and not current.done():
            return
        job = asyncio.create_task(
            self._run_coordination(owner_id),
            name=f"gander-coordination-{owner_id}",
        )
        self._coordination_jobs[owner_id] = job
        self._auxiliary_jobs.add(job)

        def finish(completed: asyncio.Task[None]) -> None:
            self._auxiliary_jobs.discard(completed)
            if self._coordination_jobs.get(owner_id) is completed:
                self._coordination_jobs.pop(owner_id, None)
            if not completed.cancelled() and completed.exception() is not None:
                LOGGER.warning(
                    "background coordination failed",
                    exc_info=(
                        type(completed.exception()),
                        completed.exception(),
                        completed.exception().__traceback__,
                    ),
                )
            pending = self.ledger.get_coordination_job(owner_id)
            if (
                not self._closed
                and pending is not None
                and pending.state == "pending"
            ):
                self._schedule_coordination(owner_id)

        job.add_done_callback(finish)

    async def _run_coordination(self, owner_id: str) -> None:
        while not self._closed:
            job = self.ledger.claim_coordination_job(owner_id)
            if job is None:
                return
            turn = self._coordination_turn(job)
            if turn is None:
                await self._finish_coordination_failure(
                    job, "persisted coordination turn is unavailable"
                )
                continue
            model_lock = self._coordinator_model_locks.setdefault(
                owner_id, asyncio.Lock()
            )
            try:
                async with model_lock:
                    if not self._coordination_is_current(job):
                        continue
                    context = self._coordinator_context(
                        request_id=job.request_id,
                        reason="turn",
                        owner_id=job.owner_id,
                        turn=turn,
                    )
                    plan = await self._coordinate(context)
                    if not isinstance(plan, CoordinatorPlan):
                        raise TypeError(
                            "Coordinator must return CoordinatorPlan"
                        )
                    owner_lock = self._owner_locks.setdefault(
                        owner_id, asyncio.Lock()
                    )
                    async with owner_lock:
                        if not self._coordination_is_current(job):
                            continue
                        try:
                            result = await self._apply_plan(
                                plan,
                                owner_id=job.owner_id,
                                request_id=job.request_id,
                                source_turn_id=job.turn_id,
                                source_turn=turn,
                            )
                        except CoordinatorPlanError as exc:
                            LOGGER.warning(
                                "Coordinator plan rejected: %s", exc
                            )
                            result = TaskControlResult(
                                request_id=job.request_id,
                                disposition="rejected",
                                speech=plan.speech,
                                reason_key=getattr(
                                    exc, "reason_key", "unroutable"
                                ),
                            )
                        finished = self.ledger.finish_coordination_job(
                            job, result
                        )
                if finished is not None:
                    self._signal_coordination(job.request_id)
                    self._deliver_coordination_result(
                        finished, result
                    )
            except asyncio.CancelledError:
                self.ledger.release_coordination_job(job)
                raise
            except Exception as exc:
                LOGGER.warning(
                    "Coordinator request failed", exc_info=True
                )
                await self._finish_coordination_failure(job, str(exc))

    def _coordination_turn(
        self, job: CoordinationJob
    ) -> TurnEnvelope | None:
        turns: list[TurnEnvelope] = []
        for receipt_key in job.turn_receipt_keys:
            turn = self.ledger.get_turn_by_receipt(receipt_key)
            if turn is None or turn.owner_id != job.owner_id:
                return None
            turns.append(turn)
        if not turns:
            return None
        latest = turns[-1]
        media_by_path = {
            media.path: media
            for turn in turns
            for media in turn.media_refs
        }
        text = "\n".join(
            turn.final_asr.strip()
            for turn in turns
            if turn.final_asr.strip()
        )
        return dataclasses.replace(
            latest,
            final_asr=text,
            media_refs=tuple(media_by_path.values()),
            context_revision=max(
                turn.context_revision for turn in turns
            ),
        )

    def _coordination_is_current(self, job: CoordinationJob) -> bool:
        current = self.ledger.get_coordination_job(job.owner_id)
        return (
            current is not None
            and current.request_id == job.request_id
            and current.version == job.version
            and current.state == "coordinating"
        )

    async def _finish_coordination_failure(
        self, job: CoordinationJob, error: str
    ) -> None:
        result = TaskControlResult(
            request_id=job.request_id,
            disposition="rejected",
            reason_key="coordinator.unavailable",
        )
        finished = self.ledger.finish_coordination_job(
            job, result, error=error
        )
        if finished is None:
            return
        self._signal_coordination(job.request_id)
        self._deliver_coordination_result(finished, result)

    def _deliver_coordination_result(
        self, job: CoordinationJob, result: TaskControlResult
    ) -> None:
        if result.disposition not in {"answered", "rejected"}:
            return
        speech = result.speech.strip()
        if not speech:
            speech = (
                "The request could not be prepared."
                if result.disposition == "rejected"
                else "The requested information is ready."
            )
        self.delivery.enqueue(
            owner_id=job.owner_id,
            task_id=result.task_ids[0] if result.task_ids else None,
            event_id=None,
            interaction_id=result.interaction_id,
            timing="safe_pause",
            topic="final",
            speech_hint=speech,
            dedupe_key=f"coordination:{job.request_id}",
        )

    async def record_evidence(
        self, *, task_id: str, ref: str, facts: dict[str, Any]
    ) -> None:
        task = self._owned_task(task_id)
        self.ledger.save_artifact(
            ArtifactEvidence(
                ref=ref,
                owner_id=task.owner_id,
                task_id=task_id,
                facts=facts,
            )
        )

    async def fetch_artifact(
        self, *, task_id: str, ref: str
    ) -> ArtifactEvidence:
        task = self._owned_task(task_id)
        artifact = self.ledger.get_artifact(ref)
        if artifact is None:
            raise KeyError(f"unknown artifact: {ref}")
        allowed = set(task.context_plan.artifact_refs) | set(
            task.context_plan.must_include_refs
        )
        if artifact.task_id != task_id and ref not in allowed:
            raise PermissionError("artifact was not selected for this task")
        source_task = self._owned_task(artifact.task_id)
        if (
            artifact.owner_id != task.owner_id
            or source_task.project_id != task.project_id
        ):
            raise PermissionError("artifact crosses owner or project boundary")
        return artifact

    async def fetch_turn(
        self,
        *,
        task_id: str,
        turn_id: str,
        voice_session_id: str | None = None,
    ) -> TurnEnvelope:
        task = self._owned_task(task_id)
        allowed = set(task.context_plan.event_refs) | set(
            task.context_plan.must_include_refs
        )
        if turn_id != task.source_turn_id and turn_id not in allowed:
            raise PermissionError("turn was not selected for this task")
        turn = self.ledger.get_turn(
            turn_id,
            owner_id=task.owner_id,
            voice_session_id=(
                voice_session_id
                or task.source_voice_session_id
                or None
            ),
        )
        if turn is None:
            raise KeyError(f"unknown turn: {turn_id}")
        return self._compile_source_turn(
            task,
            turn,
            self.providers.get(task.provider_name).capabilities,
        )

    async def memory_search(
        self,
        *,
        task_id: str,
        run_id: str,
        query: str,
        scope: str,
        limit: int,
    ) -> dict[str, Any]:
        """Call the deployment memory service through an active-run fence."""

        task = self._worker_tool_task(task_id, run_id, "memory_search")
        provider = self.memory_provider
        if provider is None:
            return {
                "status": "unavailable",
                "reason": "memory_provider_not_configured",
                "scope": scope,
                "results": [],
                "truncated": False,
            }
        try:
            pending = provider.search(
                owner_id=task.owner_id,
                project_id=task.project_id,
                task_id=task.task_id,
                query=query,
                scope=scope,
                limit=limit,
            )
            raw_hits = await asyncio.wait_for(
                pending if inspect.isawaitable(pending) else _ready(pending),
                timeout=self.memory_search_timeout_s,
            )
        except asyncio.TimeoutError:
            return {
                "status": "error",
                "reason": "memory_provider_timeout",
                "scope": scope,
                "results": [],
                "truncated": False,
            }
        except Exception as exc:
            LOGGER.warning("memory provider search failed", exc_info=True)
            return {
                "status": "error",
                "reason": "memory_provider_failed",
                "detail": str(exc)[:256],
                "scope": scope,
                "results": [],
                "truncated": False,
            }
        if isinstance(raw_hits, (str, bytes)) or not isinstance(
            raw_hits, Sequence
        ):
            raise TypeError("memory provider must return a sequence of hits")

        results: list[dict[str, Any]] = []
        used_chars = 0
        truncated = len(raw_hits) > limit
        for raw in raw_hits[:limit]:
            hit = _memory_hit(raw)
            remaining = self.max_memory_result_chars - used_chars
            if remaining <= 0:
                truncated = True
                break
            text = hit.text[: min(4000, remaining)]
            if len(text) < len(hit.text):
                truncated = True
            used_chars += len(text)
            item: dict[str, Any] = {
                "ref": hit.ref,
                "text": text,
                "metadata": _json_mapping(hit.metadata, "memory hit metadata"),
            }
            if hit.score is not None:
                item["score"] = float(hit.score)
            results.append(item)
        return {
            "status": "ok",
            "scope": scope,
            "query": query,
            "results": results,
            "content_chars": used_chars,
            "truncated": truncated,
        }

    async def context_fetch(
        self,
        *,
        task_id: str,
        run_id: str,
        refs: tuple[str, ...],
        query: str,
        kinds: tuple[str, ...],
        context_kinds: tuple[str, ...],
        roles: tuple[str, ...],
        last_ms: int | None,
        limit: int,
        max_chars: int,
        cursor: int,
        include_media: bool,
    ) -> dict[str, Any]:
        """Return a bounded projection for the current task lineage."""

        task = self._worker_tool_task(task_id, run_id, "context_fetch")
        capabilities = self.providers.get(task.provider_name).capabilities
        candidates = self._context_candidates(
            task, capabilities, include_media=include_media
        )
        available_kinds: dict[str, int] = {}
        for candidate in candidates:
            kind = str(candidate["kind"])
            available_kinds[kind] = available_kinds.get(kind, 0) + 1
        kind_filter = set(kinds)
        selection = "refs" if refs else "search" if query else "kinds"
        if not refs and not query and not kind_filter:
            # A zero-argument fetch returns the context captured at task creation.
            kind_filter = {"realtime"}
            selection = "task_start"
        if kind_filter:
            candidates = [
                candidate
                for candidate in candidates
                if candidate["kind"] in kind_filter
            ]
        if context_kinds or roles or last_ms is not None:
            context_kind_filter = set(context_kinds)
            role_filter = set(roles)
            trigger_ms = task.created_at_ms
            candidates = [
                candidate
                for candidate in candidates
                if candidate["kind"] == "realtime"
                and (
                    not context_kind_filter
                    or candidate.get("context_kind") in context_kind_filter
                )
                and (not role_filter or candidate.get("role") in role_filter)
                and (
                    last_ms is None
                    or int(candidate.get("created_at_ms", 0))
                    >= trigger_ms - last_ms
                )
            ]

        missing_refs: list[str] = []
        if refs:
            aliases: dict[str, list[dict[str, Any]]] = {}
            for candidate in candidates:
                for alias in candidate.pop("_aliases"):
                    aliases.setdefault(alias, []).append(candidate)
            selected: list[dict[str, Any]] = []
            selected_refs: set[str] = set()
            for ref in refs:
                matches = aliases.get(ref, ())
                if not matches:
                    missing_refs.append(ref)
                for candidate in matches:
                    if candidate["ref"] not in selected_refs:
                        selected.append(candidate)
                        selected_refs.add(candidate["ref"])
            candidates = selected
        else:
            if query:
                scored = [
                    (_context_score(query, candidate), candidate)
                    for candidate in candidates
                ]
                candidates = [
                    candidate
                    for score, candidate in sorted(
                        scored,
                        key=lambda item: (
                            item[0], item[1].get("created_at_ms", 0)
                        ),
                        reverse=True,
                    )
                    if score > 0
                ]
            else:
                candidates.sort(
                    key=lambda item: item.get("created_at_ms", 0),
                    reverse=kind_filter != {"realtime"},
                )
            for candidate in candidates:
                candidate.pop("_aliases", None)

        total = len(candidates)
        page = candidates[cursor : cursor + limit]
        results: list[dict[str, Any]] = []
        content_chars = 0
        text_truncated = False
        media_count = 0
        media_truncated = False
        for candidate in page:
            item = dict(candidate)
            item.pop("_search_text", None)
            text = str(item.get("text") or "")
            remaining = max_chars - content_chars
            if remaining <= 0:
                text_truncated = True
                break
            if len(text) > remaining:
                text = text[:remaining]
                text_truncated = True
                item["text_truncated"] = True
            item["text"] = text
            media = item.get("media")
            if isinstance(media, list):
                if item.get("media_truncated"):
                    media_truncated = True
                available = max(self.max_context_media_refs - media_count, 0)
                if len(media) > available:
                    item["media"] = media[:available]
                    item["media_truncated"] = True
                    media_truncated = True
                media_count += len(item["media"])
            content_chars += len(text)
            results.append(item)

        consumed = len(results)
        next_cursor = cursor + consumed
        has_more = text_truncated or next_cursor < total
        return {
            "status": "ok",
            "scope": {
                "owner_id": task.owner_id,
                "project_id": task.project_id,
                "task_id": task.task_id,
                "lineage_id": task.lineage_id,
            },
            "selection": selection,
            "available_kinds": available_kinds,
            "results": results,
            "missing_refs": missing_refs,
            "content_chars": content_chars,
            "media_refs": media_count,
            "media_truncated": media_truncated,
            "truncated": has_more,
            "next_cursor": next_cursor if has_more else None,
        }

    def _worker_tool_task(
        self, task_id: str, run_id: str, tool: str
    ) -> TaskRecord:
        task = self._owned_task(task_id)
        if task.active_run_id != run_id or task.terminal:
            raise PermissionError("worker tool call targets an inactive run")
        capabilities = self.providers.get(task.provider_name).capabilities
        if tool not in capabilities.worker_tools:
            raise PermissionError(f"provider did not declare worker tool: {tool}")
        return task

    def _context_candidates(
        self,
        task: TaskRecord,
        capabilities: BackendCapabilities,
        *,
        include_media: bool,
    ) -> list[dict[str, Any]]:
        lineage_tasks = tuple(
            candidate
            for candidate in self.ledger.list_tasks(task.owner_id)
            if candidate.project_id == task.project_id
            and candidate.lineage_id == task.lineage_id
        )
        lineage_task_ids = {candidate.task_id for candidate in lineage_tasks}

        turns: dict[str, TurnEnvelope] = {}
        realtime_events: dict[str, ContextEvent] = {}
        for lineage_task in lineage_tasks:
            source = self.ledger.get_turn(
                lineage_task.source_turn_id,
                owner_id=task.owner_id,
                voice_session_id=(
                    lineage_task.source_voice_session_id or None
                ),
            )
            if source is not None:
                turns[source.receipt_key] = source
            for ref in (
                *lineage_task.context_plan.event_refs,
                *lineage_task.context_plan.must_include_refs,
            ):
                if ref.startswith("realtime:"):
                    event = self.ledger.get_realtime_context(
                        ref.removeprefix("realtime:"), owner_id=task.owner_id
                    )
                    if (
                        event is not None
                        and event.session_id
                        == lineage_task.source_voice_session_id
                    ):
                        realtime_events[event.event_id] = event
                    continue
                selected = self.ledger.get_turn_by_receipt(ref)
                if selected is not None and selected.owner_id != task.owner_id:
                    selected = None
                if selected is None:
                    try:
                        selected = self.ledger.get_turn(
                            ref,
                            owner_id=task.owner_id,
                            voice_session_id=(
                                lineage_task.source_voice_session_id or None
                            ),
                        )
                    except ValueError:
                        selected = None
                if selected is not None:
                    turns[selected.receipt_key] = selected
        for command in self.ledger.list_provider_commands(task.owner_id):
            if command.task_id not in lineage_task_ids:
                continue
            source = command.message.source_turn
            if source is not None and source.owner_id == task.owner_id:
                turns[source.receipt_key] = source

        candidates: list[dict[str, Any]] = []
        for event in sorted(
            realtime_events.values(),
            key=lambda value: (value.seq, value.timestamp_ms, value.event_id),
        ):
            ref = f"realtime:{event.event_id}"
            item: dict[str, Any] = {
                "ref": ref,
                "kind": "realtime",
                "context_kind": event.kind,
                "role": event.role,
                "seq": event.seq,
                "created_at_ms": event.timestamp_ms,
                "timestamp_ms": event.timestamp_ms,
                "text": event.text,
                "voice_session_id": event.session_id,
                "metadata": _json_mapping(
                    event.metadata, "realtime context metadata"
                ),
                "_aliases": {event.event_id, ref},
                "_search_text": event.text,
            }
            if event.start_ms is not None:
                item["start_ms"] = event.start_ms
                item["end_ms"] = event.end_ms
            eligible_values = [
                media
                for media in event.media
                if _media_modality(media.kind) in capabilities.modalities
            ]
            item["media_summary"] = _media_summary(eligible_values)
            if include_media:
                eligible_media = [
                    {
                        "kind": media.kind,
                        "path": media.path,
                        "timestamp_ms": media.timestamp_ms,
                        "mime_type": media.mime_type,
                        "source_path": media.source_path,
                        "metadata": _json_mapping(
                            media.metadata, "realtime media metadata"
                        ),
                    }
                    for media in eligible_values
                ]
                item["media"] = eligible_media[:32]
                if len(eligible_media) > 32:
                    item["media_truncated"] = True
            candidates.append(item)

        turn_id_counts: dict[str, int] = {}
        for turn in turns.values():
            turn_id_counts[turn.turn_id] = turn_id_counts.get(turn.turn_id, 0) + 1
        for turn in turns.values():
            ref = f"turn:{turn.voice_session_id}:{turn.turn_id}"
            aliases = {ref, turn.receipt_key}
            if turn_id_counts[turn.turn_id] == 1:
                aliases.update({turn.turn_id, f"turn:{turn.turn_id}"})
            item: dict[str, Any] = {
                "ref": ref,
                "kind": "turn",
                "created_at_ms": turn.created_at_ms,
                "text": turn.final_asr,
                "voice_session_id": turn.voice_session_id,
                "context_revision": turn.context_revision,
                "timestamp_ms": turn.timestamp_ms or turn.created_at_ms,
                "_aliases": aliases,
                "_search_text": turn.final_asr,
            }
            if turn.start_ms is not None:
                item["start_ms"] = turn.start_ms
                item["end_ms"] = turn.end_ms
            eligible_values = [
                media
                for media in turn.media_refs
                if _media_modality(media.kind) in capabilities.modalities
            ]
            item["media_summary"] = _media_summary(eligible_values)
            if include_media:
                eligible_media = [
                    {
                        "kind": media.kind,
                        "path": media.path,
                        "timestamp_ms": media.timestamp_ms,
                        "mime_type": media.mime_type,
                    }
                    for media in eligible_values
                ]
                item["media"] = eligible_media[:32]
                if len(eligible_media) > 32:
                    item["media_truncated"] = True
            candidates.append(item)

        for lineage_task in lineage_tasks:
            text = "\n".join(
                value
                for value in (
                    f"title: {lineage_task.title}",
                    f"objective: {lineage_task.objective}",
                    f"status: {lineage_task.status}",
                    f"result: {lineage_task.result}" if lineage_task.result else "",
                    f"error: {lineage_task.error}" if lineage_task.error else "",
                )
                if value
            )
            candidates.append(
                {
                    "ref": f"task:{lineage_task.task_id}",
                    "kind": "task",
                    "created_at_ms": lineage_task.updated_at_ms,
                    "text": text,
                    "status": lineage_task.status,
                    "_aliases": {
                        lineage_task.task_id,
                        f"task:{lineage_task.task_id}",
                    },
                    "_search_text": text,
                }
            )

        for artifact in self.ledger.list_artifacts(task.owner_id):
            if artifact.task_id not in lineage_task_ids:
                continue
            text = json.dumps(
                _json_mapping(artifact.facts, "artifact facts"),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            canonical_ref = (
                artifact.ref
                if artifact.ref.startswith("artifact:")
                else f"artifact:{artifact.ref}"
            )
            candidates.append(
                {
                    "ref": canonical_ref,
                    "kind": "artifact",
                    "created_at_ms": artifact.created_at_ms,
                    "text": text,
                    "task_id": artifact.task_id,
                    "_aliases": {artifact.ref, canonical_ref},
                    "_search_text": text,
                }
            )

        for lineage_task in lineage_tasks:
            for event in self.ledger.list_events(task_id=lineage_task.task_id):
                text = _worker_event_text(event)
                candidates.append(
                    {
                        "ref": f"event:{event.event_id}",
                        "kind": "event",
                        "created_at_ms": event.created_at_ms,
                        "text": text,
                        "task_id": event.task_id,
                        "event_type": event.type,
                        "_aliases": {event.event_id, f"event:{event.event_id}"},
                        "_search_text": text,
                    }
                )
        return candidates

    async def authorize(
        self,
        *,
        task_id: str,
        run_id: str,
        action: str,
        request_key: str,
        prompt: str,
        blocking_scope: str = "action",
        side_effect: bool = True,
        high_risk: bool = False,
    ) -> AuthorizationDecision:
        task = self._owned_task(task_id)
        if task.active_run_id != run_id:
            return AuthorizationDecision("deny", action, "inactive_run")
        if task.kind == "side_query" and side_effect:
            return AuthorizationDecision("deny", action, "side_query_read_only")
        contract = self._required_contract(task)
        capabilities = self.providers.get(task.provider_name).capabilities
        decision = self.supervision.authorize(
            contract,
            capabilities,
            action,
            side_effect=side_effect,
            high_risk=high_risk,
        )
        if decision.outcome != "require_permission":
            return decision
        if task.owner_id in self._standing_allow:
            return AuthorizationDecision(
                "allow", action, "standing_session_grant"
            )
        fingerprint = stable_key(
            "permission",
            self._instance_id,
            task_id,
            run_id,
            action,
            request_key,
        )
        existing = self.ledger.find_interaction_by_fingerprint(fingerprint)
        if existing is not None:
            if existing.state != "resolved":
                return dataclasses.replace(
                    decision, interaction_id=existing.interaction_id
                )
            if _approved(existing.decision, existing.response):
                return AuthorizationDecision(
                    "allow", action, "permission_granted"
                )
            return AuthorizationDecision("deny", action, "permission_denied")
        interaction = PendingInteraction(
            interaction_id=new_id("interaction"),
            owner_id=task.owner_id,
            target_id=task_id,
            task_id=task_id,
            run_id=run_id,
            source_event_id=None,
            source="worker",
            audience="user",
            kind="permission",
            prompt=prompt,
            reason_key="authority.permission",
            action_key=action,
            requested_scope=blocking_scope,
            effective_scope=effective_scope(
                blocking_scope, capabilities.blocking_granularity
            ),
            fingerprint=fingerprint,
            expires_at_ms=now_ms() + self.permission_timeout_ms,
        )
        self.ledger.save_interaction(interaction)
        self._refresh_blocked(task_id)
        self.delivery.enqueue(
            owner_id=task.owner_id,
            task_id=task_id,
            event_id=None,
            interaction_id=interaction.interaction_id,
            timing=contract.body.delivery_policy.interaction,
            topic="interaction",
            speech_hint=prompt,
            dedupe_key=f"permission:{fingerprint}",
        )
        return dataclasses.replace(
            decision, interaction_id=interaction.interaction_id
        )

    async def expire_interactions(
        self, timestamp_ms: int | None = None
    ) -> tuple[PendingInteraction, ...]:
        expired = self.ledger.expire_due_interactions(timestamp_ms)
        for interaction in expired:
            self.ledger.cancel_provider_commands_for_interaction(
                interaction.interaction_id
            )
            self.ledger.cancel_interaction_deliveries(
                interaction.interaction_id
            )
            if interaction.task_id:
                self._refresh_blocked(interaction.task_id)
            run = self._runtime_runs.get(interaction.run_id or "")
            command = self._queue_expired_interaction(interaction)
            if run is not None and command is not None:
                await self._deliver_provider_command(run, command)
        return expired

    async def ingest_worker_event(self, event: WorkerEvent) -> bool:
        if self._closed:
            return False
        if not self.ledger.append_event(event):
            return False
        await self._reduce_event(event)
        self.ledger.mark_event_processed(event.event_id)
        return True

    async def wait_task(
        self, task_id: str, timeout: float | None = None
    ) -> TaskRecord:
        task = self._owned_task(task_id)
        if task.terminal:
            return task
        event = self._finished_event(task_id)
        if timeout is None:
            await event.wait()
        else:
            await asyncio.wait_for(event.wait(), timeout)
        return self._owned_task(task_id)

    def set_directive(self, owner_id: str, directive: str) -> bool:
        """Apply a session-scoped delivery or permission directive."""
        if directive == "revoke_session":
            self._standing_allow.discard(owner_id)
        elif directive == "mute":
            self._muted.add(owner_id)
        elif directive == "unmute":
            self._muted.discard(owner_id)
        elif directive == "subscribe_interrupt":
            self._interrupt_subscribed.add(owner_id)
        elif directive == "unsubscribe_interrupt":
            self._interrupt_subscribed.discard(owner_id)
        else:
            return False
        return True

    def pending_deliveries(self, owner_id: str) -> tuple[DeliveryRecord, ...]:
        if owner_id in self._muted:
            return ()
        self.ledger.requeue_expired_delivery_claims(owner_id)
        deliveries = [
            item
            for item in self.ledger.list_deliveries(owner_id)
            if item.state == "pending"
        ]
        # Subscribed interrupts preempt output; other deliveries pause. Lean mode
        # accepts the back brain's interrupt classification directly.
        if (
            self.mode == "coordinator"
            and owner_id not in self._interrupt_subscribed
        ):
            deliveries = [
                dataclasses.replace(item, timing="safe_pause")
                if item.timing == "interrupt"
                else item
                for item in deliveries
            ]
        return tuple(deliveries)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        runs = tuple(self._runtime_runs.values())
        if runs:
            await asyncio.gather(
                *(
                    asyncio.wait_for(
                        run.close(), self.provider_command_timeout_s
                    )
                    for run in runs
                ),
                return_exceptions=True,
            )
        jobs = tuple(self._task_jobs.values()) + tuple(self._auxiliary_jobs)
        if jobs:
            _, pending = await asyncio.wait(jobs, timeout=0.1)
            for job in pending:
                job.cancel()
            try:
                await asyncio.wait_for(
                    asyncio.gather(*pending, return_exceptions=True),
                    self.provider_command_timeout_s,
                )
            except asyncio.TimeoutError:
                LOGGER.warning("worker jobs did not stop before close timeout")
        for project in tuple(self._project_sessions.values()):
            try:
                await asyncio.wait_for(
                    project.close(), self.provider_command_timeout_s
                )
            except Exception:
                LOGGER.warning("worker project close failed", exc_info=True)
        for provider in self.providers.values():
            try:
                await asyncio.wait_for(
                    provider.close(), self.provider_command_timeout_s
                )
            except Exception:
                LOGGER.warning("worker provider close failed", exc_info=True)
        close_coordinator = getattr(self.coordinator, "close", None)
        if callable(close_coordinator):
            try:
                result = close_coordinator()
                if inspect.isawaitable(result):
                    await asyncio.wait_for(
                        result, self.provider_command_timeout_s
                    )
            except Exception:
                LOGGER.warning("Coordinator close failed", exc_info=True)

    def _coordinator_context(
        self,
        *,
        request_id: str,
        reason: str,
        owner_id: str,
        turn: TurnEnvelope | None,
        worker_event: WorkerEvent | None = None,
    ) -> CoordinatorContext:
        policy = self.context_policy
        all_tasks = tuple(
            task
            for task in self.ledger.list_tasks(owner_id)
            if task.kind == "main"
        )
        active = [task for task in all_tasks if not task.terminal][
            -policy.max_active_tasks:
        ]
        recent = [task for task in all_tasks if task.terminal][
            -policy.max_recent_tasks:
        ]
        tasks = tuple(active + recent)
        contracts = tuple(
            contract
            for task in tasks
            if (contract := self.ledger.get_contract(task.task_id)) is not None
        )
        interactions = tuple(
            item
            for item in self.ledger.list_interactions(owner_id)
            if item.state in {"pending", "resolving"}
        )[-policy.max_interactions:]
        projects = self.ledger.list_projects(owner_id)
        visible_project_ids = {task.project_id for task in tasks}
        visible_projects = tuple(
            project
            for project in projects
            if project.project_id in visible_project_ids
        )
        if len(visible_projects) < policy.max_projects:
            known = {project.project_id for project in visible_projects}
            visible_projects += tuple(
                project
                for project in projects[-policy.max_projects:]
                if project.project_id not in known
            )
        visible_projects = visible_projects[-policy.max_projects:]
        return CoordinatorContext(
            request_id=request_id,
            reason=reason,
            owner_id=owner_id,
            turn=turn,
            tasks=tasks,
            projects=tuple(
                dataclasses.replace(project, backend_session_id="")
                for project in visible_projects
            ),
            interactions=interactions,
            contracts=contracts,
            provider_capabilities=self.providers.capabilities(),
            worker_event=worker_event,
        )

    async def _coordinate(
        self, context: CoordinatorContext
    ) -> CoordinatorPlan:
        started = time.perf_counter()
        status = "completed"
        error = ""
        plan: CoordinatorPlan | None = None
        try:
            plan = await asyncio.wait_for(
                self.coordinator.coordinate(context),
                self.coordinator_timeout_s,
            )
            return plan
        except asyncio.TimeoutError as exc:
            status = "timed_out"
            error = type(exc).__name__
            raise
        except BaseException as exc:
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            observer = self.coordinator_observer
            if observer is not None:
                try:
                    snapshot_chars = len(
                        render_coordinator_snapshot(
                            context, self.context_policy
                        )
                    )
                except ValueError:
                    snapshot_chars = -1
                metric = CoordinatorCallMetric(
                    request_id=context.request_id,
                    owner_id=context.owner_id,
                    reason=context.reason,
                    backend=str(
                        getattr(
                            self.coordinator,
                            "name",
                            type(self.coordinator).__name__,
                        )
                    ),
                    status=status,
                    elapsed_ms=(
                        time.perf_counter() - started
                    ) * 1000,
                    snapshot_chars=snapshot_chars,
                    task_count=len(context.tasks),
                    project_count=len(context.projects),
                    interaction_count=len(context.interactions),
                    command_count=(
                        len(plan.commands)
                        if isinstance(plan, CoordinatorPlan)
                        else 0
                    ),
                    error=error,
                )
                try:
                    observer(metric)
                except Exception:
                    LOGGER.warning(
                        "Coordinator observer failed", exc_info=True
                    )

    async def _apply_plan(
        self,
        plan: CoordinatorPlan,
        *,
        owner_id: str,
        request_id: str,
        source_turn_id: str,
        source_turn: TurnEnvelope | None,
    ) -> TaskControlResult:
        self._validate_plan(plan, owner_id, request_id, source_turn)
        task_ids: list[str] = []
        resolved_ids: set[str] = set()
        for command_index, command in enumerate(plan.commands):
            operation_id = stable_id(
                "operation", request_id, str(command_index)
            )
            if isinstance(command, CreateTaskCommand):
                task_name = ""
                if (
                    source_turn is not None
                    and source_turn.frontbrain_action == "task_start"
                ):
                    active, recent = self._control_task_views(owner_id)
                    task_name = assign_frontbrain_task_name(
                        source_turn.frontbrain_task_name,
                        frozenset(
                            item.name for item in (*active, *recent)
                        ),
                    )
                task = self._create_task(
                    command,
                    owner_id,
                    source_turn_id,
                    source_turn,
                    task_id=stable_id(
                        "task", request_id, str(command_index)
                    ),
                    name=task_name,
                )
                task_ids.append(task.task_id)
                self._schedule(task.task_id)
            elif isinstance(command, SendTaskCommand):
                task_ids.append(command.task_id)
                await self._send_task(
                    command, source_turn, operation_id
                )
            elif isinstance(command, UpdateContractCommand):
                task_ids.append(command.task_id)
                await self._update_contract(
                    command, source_turn_id, operation_id
                )
            elif isinstance(command, CancelTaskCommand):
                task_ids.append(command.task_id)
                await self._cancel_task(command.task_id)
            elif isinstance(command, ResolveInteractionCommand):
                if await self._resolve_interaction(
                    command,
                    message_id=stable_id(
                        "message", operation_id
                    ),
                ):
                    resolved_ids.add(command.interaction_id)
            else:
                raise CoordinatorPlanError(
                    f"unsupported command: {type(command).__name__}"
                )

        interaction_id: str | None = None
        disposition = plan.disposition
        if plan.question is not None:
            target_contract = self.ledger.get_contract(plan.question.target_id)
            interaction, admission = self.interactions.admit_question(
                plan.question,
                owner_id=owner_id,
                coordination_request_id=request_id,
                contract=target_contract,
            )
            if interaction is not None:
                target_task = (
                    self.ledger.get_task(interaction.task_id)
                    if interaction.task_id
                    else None
                )
                if target_task is not None:
                    supported_scope = self.providers.get(
                        target_task.provider_name
                    ).capabilities.blocking_granularity
                    widened_scope = effective_scope(
                        interaction.requested_scope, supported_scope
                    )
                    if widened_scope != interaction.effective_scope:
                        interaction = dataclasses.replace(
                            interaction, effective_scope=widened_scope
                        )
                        self.ledger.save_interaction(interaction)
                interaction_id = interaction.interaction_id
                disposition = "clarify"
                if interaction.task_id:
                    self._refresh_blocked(interaction.task_id)
                self.delivery.enqueue(
                    owner_id=owner_id,
                    task_id=interaction.task_id,
                    event_id=None,
                    interaction_id=interaction.interaction_id,
                    timing=(
                        target_contract.body.delivery_policy.interaction
                        if target_contract
                        else "safe_pause"
                    ),
                    topic="interaction",
                    speech_hint=interaction.prompt,
                    dedupe_key=f"coordinator-question:{interaction.fingerprint}",
                )
            elif admission == "duplicate":
                disposition = "accepted"
            elif admission == "rejected":
                disposition = "rejected"
            else:
                disposition = "accepted"
                target_task = self.ledger.get_task(
                    plan.question.target_id
                )
                if target_task is not None:
                    assumption = plan.question.default_if_unanswered
                    self.ledger.save_task(
                        dataclasses.replace(
                            target_task,
                            assumptions=tuple(
                                dict.fromkeys(
                                    target_task.assumptions
                                    + (assumption,)
                                )
                            ),
                            updated_at_ms=now_ms(),
                        )
                    )
        return TaskControlResult(
            request_id=request_id,
            disposition=disposition,
            speech=plan.speech,
            task_ids=tuple(dict.fromkeys(task_ids)),
            interaction_id=interaction_id,
        )

    def _validate_plan(
        self,
        plan: CoordinatorPlan,
        owner_id: str,
        request_id: str,
        source_turn: TurnEnvelope | None,
    ) -> None:
        if (
            source_turn is not None
            and source_turn.frontbrain_action == "task_start"
        ):
            if (
                plan.question is not None
                or len(plan.commands) != 1
                or not isinstance(plan.commands[0], CreateTaskCommand)
            ):
                raise CoordinatorPlanError(
                    "task_start requires exactly one create command"
                )
            required_provider = source_turn.runtime_provider_name
            if (
                required_provider
                and plan.commands[0].provider_name != required_provider
            ):
                raise CoordinatorPlanError(
                    "task_start provider does not match deployment routing"
                )
        requested_tasks = sum(
            isinstance(command, CreateTaskCommand)
            for command in plan.commands
        )
        queued_tasks = sum(
            task.status == "queued" and task.kind == "main"
            for task in self.ledger.list_tasks(owner_id)
        )
        if (
            requested_tasks + queued_tasks
            > self.max_queued_tasks_per_owner
        ):
            raise CoordinatorPlanError("owner task queue limit exceeded")
        pending_permissions = tuple(
            item
            for item in self.ledger.list_interactions(owner_id)
            if item.state == "pending"
            and item.kind == "permission"
            and item.audience == "user"
        )
        resolves_permission = any(
            isinstance(command, ResolveInteractionCommand)
            and any(
                item.interaction_id == command.interaction_id
                for item in pending_permissions
            )
            for command in plan.commands
        )
        if (
            source_turn is not None
            and resolves_permission
            and len(pending_permissions) > 1
            and _generic_approval(source_turn.final_asr)
        ):
            raise CoordinatorPlanError(
                "ambiguous approval cannot resolve multiple permissions"
            )
        for command in plan.commands:
            if isinstance(command, CreateTaskCommand):
                if command.provider_name not in self.providers:
                    raise CoordinatorPlanError(
                        f"unknown provider: {command.provider_name}",
                        reason_key="no_eligible_worker",
                    )
                capabilities = self.providers.get(
                    command.provider_name
                ).capabilities
                precise_milestones = tuple(
                    rule
                    for rule in command.contract_body.notify_policy.milestones
                    if rule.key != "final.completed"
                )
                if (
                    precise_milestones
                    and capabilities.structured_events
                    not in {"native", "injected_tool"}
                ):
                    raise CoordinatorPlanError(
                        "backend cannot guarantee subscribed milestones",
                        reason_key="no_eligible_worker",
                    )
                if command.project_id is not None:
                    project = self.ledger.get_project(command.project_id)
                    if project is None:
                        raise CoordinatorPlanError(
                            f"unknown project: {command.project_id}"
                        )
                    if (
                        project.owner_id != owner_id
                        or project.provider_name != command.provider_name
                    ):
                        raise CoordinatorPlanError(
                            "project ownership or provider mismatch"
                        )
            elif isinstance(
                command,
                (
                    SendTaskCommand,
                    UpdateContractCommand,
                    CancelTaskCommand,
                ),
            ):
                task = self._owned_task(command.task_id, owner_id)
                if isinstance(command, SendTaskCommand):
                    if task.terminal:
                        raise CoordinatorPlanError(
                            f"cannot send to terminal task: {task.task_id}"
                        )
                    capabilities = self.providers.get(
                        task.provider_name
                    ).capabilities
                    if (
                        command.mode == "update"
                        and task.status != "queued"
                        and capabilities.steering == "none"
                    ):
                        raise CoordinatorPlanError(
                            "backend does not support task updates"
                        )
                    if (
                        command.mode == "query"
                        and capabilities.side_queries == "none"
                    ):
                        raise CoordinatorPlanError(
                            "backend does not support side queries"
                        )
                elif (
                    isinstance(command, UpdateContractCommand)
                    and task.terminal
                ):
                    raise CoordinatorPlanError(
                        "cannot revise a terminal task contract"
                    )
            elif isinstance(command, ResolveInteractionCommand):
                interaction = self.ledger.get_interaction(
                    command.interaction_id
                )
                if interaction is None or interaction.owner_id != owner_id:
                    raise CoordinatorPlanError(
                        f"unknown interaction: {command.interaction_id}"
                    )
                if (
                    interaction.kind == "permission"
                    and command.decision not in {"allow", "deny"}
                ):
                    raise CoordinatorPlanError(
                        "permission resolution requires allow or deny"
                    )
                if (
                    interaction.kind != "permission"
                    and command.decision is not None
                ):
                    raise CoordinatorPlanError(
                        "non-permission resolution must not set decision"
                    )
                if (
                    interaction.kind == "choice"
                    and command.response not in interaction.choices
                ):
                    raise CoordinatorPlanError(
                        "choice response must match an offered choice"
                    )
            else:
                raise CoordinatorPlanError(
                    f"unsupported command: {type(command).__name__}"
                )
        if plan.question is not None:
            target = self.ledger.get_task(plan.question.target_id)
            if (
                target is not None
                and target.owner_id != owner_id
                or target is None
                and plan.question.target_id != request_id
            ):
                raise CoordinatorPlanError("invalid question target")

    async def task_start(
        self,
        *,
        owner_id: str,
        turn: TurnEnvelope,
        name: str,
        provider_name: str | None = None,
    ) -> TaskControlResult:
        """Start a task with a display name and runtime-bound objective."""

        self._validate_bound_turn(owner_id, turn)
        normalized_name = normalize_frontbrain_task_name(name)
        await self.start()
        request_id = stable_id(
            "task_start", turn.receipt_key, normalized_name
        )
        lock = self._owner_locks.setdefault(owner_id, asyncio.Lock())
        async with lock:
            self._prune_control_tasks(owner_id)
            existing = self.ledger.get_receipt(request_id)
            if existing is not None:
                return existing
            self.ledger.put_turn(turn)
            active, recent = self._control_task_views(owner_id)
            result = self._start_task(
                owner_id,
                turn,
                provider_name,
                request_id,
                frozenset(item.name for item in (*active, *recent)),
                proposed_name=normalized_name,
            )
            return self.ledger.save_receipt(request_id, owner_id, result)

    def record_realtime_context(
        self, *, owner_id: str, event: ContextEvent
    ) -> bool:
        """Persist one pre-task realtime event for later pull context."""

        if not owner_id:
            raise ValueError("realtime context owner_id must not be empty")
        if not event.session_id:
            raise ValueError("realtime context session_id must not be empty")
        return self.ledger.put_realtime_context(owner_id, event)

    async def task_send(
        self,
        lane: TaskLane,
        *,
        owner_id: str,
        turn: TurnEnvelope,
        ref: str | None = None,
    ) -> TaskControlResult:
        """Send the bound turn to a task's main or read-only fork lane.

        Main resolves pending interaction or updates the active run. Fork creates an
        independent side query and requires native fork support.
        """

        if lane not in TASK_LANES:
            raise ValueError(f"task lane must be one of {sorted(TASK_LANES)}")
        self._validate_bound_turn(owner_id, turn)
        await self.start()
        request_id = stable_id(
            "task_send", turn.receipt_key, ref or "", lane
        )
        lock = self._owner_locks.setdefault(owner_id, asyncio.Lock())
        async with lock:
            self._prune_control_tasks(owner_id)
            existing = self.ledger.get_receipt(request_id)
            if existing is not None:
                return existing
            self.ledger.put_turn(turn)
            result = await self._apply_task_send(
                lane, owner_id, turn, ref, request_id
            )
            return self.ledger.save_receipt(request_id, owner_id, result)

    async def task_resolve(
        self,
        action: TaskResolveAction,
        *,
        owner_id: str,
        turn: TurnEnvelope,
        ref: str | None = None,
    ) -> TaskControlResult:
        """Apply one discrete decision enforced by the runtime."""

        if action not in TASK_RESOLVE_ACTIONS:
            raise ValueError(
                "task action must be one of "
                f"{sorted(TASK_RESOLVE_ACTIONS)}"
            )
        self._validate_bound_turn(owner_id, turn)
        await self.start()
        request_id = stable_id(
            "task_resolve", turn.receipt_key, ref or "", action
        )
        lock = self._owner_locks.setdefault(owner_id, asyncio.Lock())
        async with lock:
            self._prune_control_tasks(owner_id)
            existing = self.ledger.get_receipt(request_id)
            if existing is not None:
                return existing
            self.ledger.put_turn(turn)
            result = await self._apply_task_resolution(
                action, owner_id, turn, ref, request_id
            )
            return self.ledger.save_receipt(request_id, owner_id, result)

    def available_task_resolve_actions(
        self, owner_id: str
    ) -> tuple[TaskResolveAction, ...]:
        """Return the union of resolve actions available across the owner's tasks."""

        active, _ = self._control_task_views(owner_id)
        actions: set[TaskResolveAction] = set()
        if active:
            actions.add("cancel")
        active_ids = {item.task_id for item in active}
        has_permission = any(
            item.state == "pending"
            and item.kind == "permission"
            and item.task_id in active_ids
            for item in self.ledger.list_interactions(owner_id)
        )
        if has_permission:
            actions.update(("allow_once", "allow_session", "deny"))
        elif owner_id in self._standing_allow and active:
            actions.add("deny")
        order: tuple[TaskResolveAction, ...] = (
            "cancel",
            "allow_once",
            "allow_session",
            "deny",
        )
        return tuple(action for action in order if action in actions)

    async def _apply_task_send(
        self,
        lane: TaskLane,
        owner_id: str,
        turn: TurnEnvelope,
        ref: str | None,
        request_id: str,
    ) -> TaskControlResult:
        active, recent = self._control_task_views(owner_id)
        outcome = resolve_target("change", ref, active, recent)
        if outcome.kind == "no_such_task":
            return TaskControlResult(
                request_id,
                "rejected",
                status="no_such_task",
                reason_key="no_such_task",
            )
        if outcome.kind == "ambiguous":
            return TaskControlResult(
                request_id,
                "clarify",
                status="ambiguous",
                candidates=outcome.candidates,
            )
        if outcome.kind in {"start", "all"}:
            return TaskControlResult(
                request_id,
                "rejected",
                status="invalid_action",
                reason_key="task_send_all_unsupported",
            )

        task_id = outcome.task_id
        assert task_id is not None
        task = self._owned_task(task_id, owner_id)
        if outcome.terminal:
            if lane == "main":
                pending_side = []
                for interaction in self.ledger.list_interactions(owner_id):
                    if interaction.state != "pending" or not interaction.task_id:
                        continue
                    child = self.ledger.get_task(interaction.task_id)
                    if (
                        child is not None
                        and child.kind == "side_query"
                        and child.parent_task_id == task.task_id
                        and not child.terminal
                    ):
                        pending_side.append(interaction)
                if pending_side:
                    if any(item.kind == "permission" for item in pending_side):
                        return TaskControlResult(
                            request_id,
                            "rejected",
                            task_ids=(task_id,),
                            status="invalid_action",
                            reason_key="side_query_read_only",
                        )
                    child_ids = {item.task_id for item in pending_side}
                    if len(pending_side) > 1 or len(child_ids) > 1:
                        return TaskControlResult(
                            request_id,
                            "clarify",
                            status="ambiguous",
                            candidates=tuple(
                                item.prompt[:24] for item in pending_side
                            ),
                        )
                    child_id = next(iter(child_ids))
                    assert child_id is not None
                    reply = await self._control_reply(
                        owner_id, child_id, turn, request_id
                    )
                    return dataclasses.replace(reply, task_ids=(task_id,))
            if lane == "fork":
                capabilities = self.providers.get(task.provider_name).capabilities
                if capabilities.terminal_side_queries not in {
                    "native_fork",
                    "independent_session",
                    "native",
                    "isolated_fork",
                }:
                    return TaskControlResult(
                        request_id,
                        "rejected",
                        task_ids=(task_id,),
                        status="unsupported",
                        reason_key="terminal_side_queries_unsupported",
                    )
                parent_run = self._latest_run_for_task(task)
                if parent_run is None:
                    return TaskControlResult(
                        request_id,
                        "rejected",
                        task_ids=(task_id,),
                        status="unsupported",
                        reason_key="terminal_context_unavailable",
                    )
                side_task = self._create_terminal_side_query(
                    task, parent_run, turn, request_id, capabilities
                )
                self._schedule(side_task.task_id)
                return TaskControlResult(
                    request_id,
                    "accepted",
                    task_ids=(task_id,),
                    status="ok",
                    reason_key=(
                        "native_terminal_fork"
                        if capabilities.terminal_side_queries
                        in {"native_fork", "native"}
                        else "independent_terminal_session"
                    ),
                )
            return self._start_task(
                owner_id,
                turn,
                task.provider_name,
                request_id,
                frozenset(item.name for item in (*active, *recent)),
                project_id=task.project_id,
                continuation_task_id=task.task_id,
                proposed_name=task.name or task.title,
            )

        if lane == "main":
            pending = [
                item
                for item in self.ledger.list_interactions(owner_id)
                if item.state == "pending" and item.task_id == task_id
            ]
            if any(item.kind == "permission" for item in pending):
                return TaskControlResult(
                    request_id,
                    "rejected",
                    task_ids=(task_id,),
                    status="invalid_action",
                    reason_key="permission_requires_task_resolve",
                )
            if pending:
                return await self._control_reply(
                    owner_id, task_id, turn, request_id
                )
            capabilities = self.providers.get(task.provider_name).capabilities
            if task.status != "queued" and capabilities.steering == "none":
                return TaskControlResult(
                    request_id,
                    "rejected",
                    task_ids=(task_id,),
                    status="unsupported",
                    reason_key="steering_unsupported",
                )
            await self._send_task(
                SendTaskCommand(
                    task_id=task_id,
                    instruction=turn.final_asr,
                    mode="update",
                    preempt=capabilities.steering == "native",
                ),
                self._turn_for_provider(turn, capabilities),
                stable_id("op", request_id),
            )
            return TaskControlResult(
                request_id,
                "accepted",
                task_ids=(task_id,),
                status="ok",
                reason_key=(
                    "deferred_to_next_turn"
                    if capabilities.steering == "next_turn"
                    else ""
                ),
            )

        capabilities = self.providers.get(task.provider_name).capabilities
        if capabilities.side_queries not in {
            "native_fork",
            "independent_session",
            "native",
            "isolated_fork",
        }:
            return TaskControlResult(
                request_id,
                "rejected",
                task_ids=(task_id,),
                status="unsupported",
                reason_key="side_queries_unsupported",
            )
        if task.status == "queued" or task.active_run_id is None:
            return TaskControlResult(
                request_id,
                "rejected",
                task_ids=(task_id,),
                status="unsupported",
                reason_key="task_not_running",
            )
        await self._send_task(
            SendTaskCommand(
                task_id=task_id,
                instruction=turn.final_asr,
                mode="query",
            ),
            self._turn_for_provider(turn, capabilities),
            stable_id("op", request_id),
        )
        return TaskControlResult(
            request_id,
            "accepted",
            task_ids=(task_id,),
            status="ok",
            reason_key=(
                "native_fork"
                if capabilities.side_queries in {"native_fork", "native"}
                else "independent_session"
            ),
        )

    async def _apply_task_resolution(
        self,
        action: TaskResolveAction,
        owner_id: str,
        turn: TurnEnvelope,
        ref: str | None,
        request_id: str,
    ) -> TaskControlResult:
        active, recent = self._control_task_views(owner_id)
        if action == "cancel":
            outcome = resolve_target("stop", ref, active, recent)
            if outcome.kind == "no_such_task":
                return TaskControlResult(
                    request_id,
                    "rejected",
                    status="no_such_task",
                    reason_key="no_such_task",
                )
            if outcome.kind == "ambiguous":
                return TaskControlResult(
                    request_id,
                    "clarify",
                    status="ambiguous",
                    candidates=outcome.candidates,
                )
            if outcome.kind == "all":
                task_ids = tuple(item.task_id for item in active)
                for task_id in task_ids:
                    await self._cancel_task(task_id)
                return TaskControlResult(
                    request_id, "accepted", task_ids=task_ids, status="ok"
                )
            task_id = outcome.task_id
            assert task_id is not None
            if outcome.terminal:
                return TaskControlResult(
                    request_id, "answered", task_ids=(task_id,), status="ok"
                )
            await self._cancel_task(task_id)
            return TaskControlResult(
                request_id, "accepted", task_ids=(task_id,), status="ok"
            )

        outcome = resolve_target("reply", ref, active, recent)
        if outcome.kind == "no_such_task":
            return TaskControlResult(
                request_id,
                "rejected",
                status="no_such_task",
                reason_key="no_such_task",
            )
        if outcome.kind == "ambiguous":
            return TaskControlResult(
                request_id,
                "clarify",
                status="ambiguous",
                candidates=outcome.candidates,
            )
        if outcome.kind in {"start", "all"} or outcome.terminal:
            return TaskControlResult(
                request_id,
                "rejected",
                status="invalid_action",
                reason_key="permission_not_pending",
            )

        task_id = outcome.task_id
        assert task_id is not None
        permissions = [
            item
            for item in self.ledger.list_interactions(owner_id)
            if item.state == "pending"
            and item.task_id == task_id
            and item.kind == "permission"
        ]
        if not permissions:
            if action == "deny" and owner_id in self._standing_allow:
                self._standing_allow.discard(owner_id)
                return TaskControlResult(
                    request_id,
                    "accepted",
                    task_ids=(task_id,),
                    status="ok",
                    reason_key="standing_grant_revoked",
                )
            return TaskControlResult(
                request_id,
                "rejected",
                task_ids=(task_id,),
                status="invalid_action",
                reason_key="permission_not_pending",
            )
        if len(permissions) > 1:
            return TaskControlResult(
                request_id,
                "clarify",
                status="ambiguous",
                candidates=tuple(item.prompt[:24] for item in permissions),
            )

        interaction = permissions[0]
        # Preserve the user's permission utterance for later context retrieval.
        async with self._task_lock(task_id):
            self._link_context_turn(self._owned_task(task_id, owner_id), turn)
        # Retain one-shot versus session approval semantics for native providers.
        decision = action
        resolved = await self._resolve_interaction(
            ResolveInteractionCommand(
                interaction_id=interaction.interaction_id,
                response=turn.final_asr,
                decision=decision,
            ),
            message_id=stable_id("message", request_id),
        )
        if not resolved:
            return TaskControlResult(
                request_id,
                "rejected",
                task_ids=(task_id,),
                status="invalid_action",
                reason_key="interaction_resolution_failed",
            )
        if action == "allow_session":
            self._standing_allow.add(owner_id)
        elif action == "deny":
            self._standing_allow.discard(owner_id)
        return TaskControlResult(
            request_id,
            "accepted",
            task_ids=(task_id,),
            interaction_id=interaction.interaction_id,
            status="ok",
        )

    @staticmethod
    def _validate_bound_turn(owner_id: str, turn: TurnEnvelope) -> None:
        if turn.owner_id != owner_id:
            raise ValueError("bound turn owner does not match the runtime owner")

    def _prune_control_tasks(self, owner_id: str) -> None:
        self.ledger.prune_terminal_tasks(
            owner_id,
            ttl_ms=self.terminal_task_ttl_ms,
            max_terminal=self.max_terminal_tasks_per_owner,
        )

    def _control_task_views(
        self, owner_id: str
    ) -> tuple[tuple[ActiveTask, ...], tuple[ActiveTask, ...]]:
        active = tuple(
            ActiveTask(task_id=task.task_id, name=(task.name or task.title))
            for task in self.ledger.list_tasks(owner_id)
            if task.kind == "main" and not task.terminal
        )
        recent = tuple(
            ActiveTask(task_id=task.task_id, name=(task.name or task.title))
            for task in self.ledger.recent_terminal_tasks(
                owner_id,
                within_ms=self.recent_task_window_ms,
                limit=self.max_terminal_tasks_per_owner,
            )
        )
        return active, recent

    def _start_task(
        self,
        owner_id: str,
        turn: TurnEnvelope,
        provider_name: str | None,
        request_id: str,
        existing_names: frozenset[str],
        proposed_name: str,
        project_id: str | None = None,
        continuation_task_id: str | None = None,
    ) -> TaskControlResult:
        provider = self._default_provider(provider_name)
        if provider is None:
            return TaskControlResult(
                request_id,
                "rejected",
                status="no_such_task",
                reason_key="no_eligible_worker",
            )
        name = assign_frontbrain_task_name(proposed_name, existing_names)
        capabilities = self.providers.get(provider).capabilities
        media_refs = (
            tuple(
                dict.fromkeys(
                    media.path
                    for media in turn.media_refs
                    if _media_modality(media.kind) in capabilities.modalities
                )
            )
            if capabilities.context_provisioning == "push_bounded"
            else ()
        )
        realtime_refs = self._pre_task_realtime_refs(owner_id, turn)
        inventory = self._pre_task_context_inventory(owner_id, realtime_refs)
        command = CreateTaskCommand(
            title=name,
            objective=turn.final_asr,
            instruction=turn.final_asr,
            provider_name=provider,
            contract_body=ContractBody.default(),
            context_plan=ContextPlan(
                event_refs=realtime_refs,
                media_refs=media_refs,
                brief=self._pre_task_context_brief(inventory),
                inventory=inventory,
            ),
            project_label="default",
            project_id=project_id,
        )
        task = self._create_task(
            command,
            owner_id,
            turn.turn_id,
            turn,
            task_id=stable_id("task", request_id),
            name=name,
            continuation_task_id=continuation_task_id,
        )
        self._schedule(task.task_id)
        return TaskControlResult(
            request_id,
            "accepted",
            task_ids=(task.task_id,),
            status="ok",
            speech=name,
        )

    def _pre_task_realtime_refs(
        self, owner_id: str, turn: TurnEnvelope
    ) -> tuple[str, ...]:
        reference_ms = turn.timestamp_ms or turn.created_at_ms
        events = [
            event
            for event in self.ledger.list_realtime_context(owner_id)
            if event.session_id == turn.voice_session_id
        ]
        if not events:
            return ()
        events.sort(key=lambda event: (event.seq, event.timestamp_ms, event.event_id))
        events = events[-self.max_realtime_context_events :]
        screen_cutoff = reference_ms - self.screen_context_window_ms
        selected = [
            event
            for event in events
            if event.kind not in EPHEMERAL_VISUAL_CONTEXT_KINDS
            or event.timestamp_ms >= screen_cutoff
        ]
        return tuple(f"realtime:{event.event_id}" for event in selected)

    def _pre_task_context_inventory(
        self, owner_id: str, refs: tuple[str, ...]
    ) -> ContextInventory:
        events = [
            event
            for ref in refs
            if ref.startswith("realtime:")
            if (
                event := self.ledger.get_realtime_context(
                    ref.removeprefix("realtime:"), owner_id=owner_id
                )
            )
            is not None
        ]
        if not events:
            return ContextInventory()
        counts: dict[str, int] = {}
        media_count = 0
        for event in events:
            counts[event.kind] = counts.get(event.kind, 0) + 1
            media_count += len(event.media)
        timestamps = [event.timestamp_ms for event in events]
        return ContextInventory(
            total=len(events),
            items=tuple(
                ContextInventoryItem(kind, count)
                for kind, count in sorted(counts.items())
            ),
            media_count=media_count,
            earliest_timestamp_ms=min(timestamps),
            latest_timestamp_ms=max(timestamps),
        )

    @staticmethod
    def _pre_task_context_brief(inventory: ContextInventory) -> str:
        if not inventory.total:
            return ""
        kinds = ",".join(
            f"{item.kind}={item.count}" for item in inventory.items
        )
        return (
            f"pre_task_realtime_context:total={inventory.total};{kinds};"
            f"media={inventory.media_count};time="
            f"{inventory.earliest_timestamp_ms}..{inventory.latest_timestamp_ms}"
        )

    def _create_terminal_side_query(
        self,
        parent: TaskRecord,
        parent_run: RunRecord,
        turn: TurnEnvelope,
        request_id: str,
        capabilities: BackendCapabilities,
    ) -> TaskRecord:
        """Persist an invisible read-only child so execution and delivery survive races."""

        task_id = stable_id("side_query", request_id, parent.task_id)
        existing = self.ledger.get_task(task_id)
        if existing is not None:
            if (
                existing.kind != "side_query"
                or existing.parent_task_id != parent.task_id
                or existing.parent_run_id != parent_run.run_id
                or existing.instruction != turn.final_asr
            ):
                raise RuntimeError("idempotent side-query identity collision")
            return existing
        project = self.ledger.get_project(parent.project_id)
        if project is None:
            raise RuntimeError("parent task project is unavailable")
        context_plan = self._context_plan_with_turn(parent.context_plan, turn)
        if capabilities.context_provisioning == "push_bounded":
            media_refs = tuple(
                dict.fromkeys(
                    (
                        *context_plan.media_refs,
                        *(
                            media.path
                            for media in turn.media_refs
                            if _media_modality(media.kind) in capabilities.modalities
                        ),
                    )
                )
            )
            context_plan = dataclasses.replace(
                context_plan, media_refs=media_refs
            )
        child = TaskRecord(
            task_id=task_id,
            owner_id=parent.owner_id,
            source_turn_id=turn.turn_id,
            source_voice_session_id=turn.voice_session_id,
            project_id=parent.project_id,
            title=parent.title,
            objective=turn.final_asr,
            instruction=turn.final_asr,
            provider_name=parent.provider_name,
            context_plan=context_plan,
            reasoning_profile=parent.reasoning_profile,
            name=parent.name,
            lineage_id=parent.lineage_id,
            kind="side_query",
            parent_task_id=parent.task_id,
            parent_run_id=parent_run.run_id,
        )
        parent_contract = self._required_contract(parent)
        contract = SupervisionContractRevision(
            task_id=task_id,
            revision=1,
            body=parent_contract.body,
            source_turn_ids=(turn.turn_id,),
        )
        self.ledger.create_task(project, child, contract)
        return child

    def _latest_run_for_task(self, task: TaskRecord) -> RunRecord | None:
        runs = tuple(
            run
            for run in self.ledger.list_runs(task.owner_id)
            if run.task_id == task.task_id
        )
        if not runs:
            return None
        return max(
            runs,
            key=lambda run: (
                run.generation,
                run.ended_at_ms or run.started_at_ms or 0,
                run.run_id,
            ),
        )

    @staticmethod
    def _turn_for_provider(
        turn: TurnEnvelope,
        capabilities: BackendCapabilities,
    ) -> TurnEnvelope:
        """Keep trusted follow-up media only when the selected provider supports it."""

        media = tuple(
            item
            for item in turn.media_refs
            if _media_modality(item.kind) in capabilities.modalities
        )
        return turn if media == turn.media_refs else dataclasses.replace(
            turn, media_refs=media
        )

    async def _control_reply(
        self,
        owner_id: str,
        task_id: str,
        turn: TurnEnvelope,
        request_id: str,
        decision: str | None = None,
    ) -> TaskControlResult:
        pending = [
            item
            for item in self.ledger.list_interactions(owner_id)
            if item.state == "pending" and item.task_id == task_id
        ]
        if not pending:
            return TaskControlResult(
                request_id,
                "rejected",
                status="no_such_task",
                reason_key="no_pending_interaction",
            )
        if len(pending) > 1:
            return TaskControlResult(
                request_id,
                "clarify",
                status="ambiguous",
                candidates=tuple(item.prompt[:24] for item in pending),
            )
        interaction = pending[0]
        # Persist the reply association so pull providers can retrieve the turn.
        async with self._task_lock(task_id):
            self._link_context_turn(self._owned_task(task_id, owner_id), turn)
        # `allow_session` grants consent for subsequent actions in this session.
        norm = (decision or "").strip().lower()
        grant_session = False
        if interaction.kind == "permission":
            if norm == "allow_session":
                grant_session = True
                resolve_decision: str | None = "allow"
            elif norm in {"allow", "allow_once"}:
                resolve_decision = "allow"
            elif norm == "deny":
                resolve_decision = "deny"
            else:
                resolve_decision = decision
        else:
            resolve_decision = decision
        resolved = await self._resolve_interaction(
            ResolveInteractionCommand(
                interaction_id=interaction.interaction_id,
                response=turn.final_asr,
                decision=resolve_decision,
            ),
            message_id=stable_id("message", request_id),
        )
        if not resolved:
            return TaskControlResult(
                request_id,
                "rejected",
                task_ids=(task_id,),
                interaction_id=interaction.interaction_id,
                status="invalid_action",
                reason_key="interaction_resolution_failed",
            )
        if grant_session:
            self._standing_allow.add(owner_id)
        elif interaction.kind == "permission" and norm == "deny":
            self._standing_allow.discard(owner_id)
        return TaskControlResult(
            request_id,
            "accepted",
            task_ids=(task_id,),
            interaction_id=interaction.interaction_id,
            status="ok",
        )

    def _default_provider(self, provider_name: str | None) -> str | None:
        if provider_name and provider_name in self.providers:
            return provider_name
        names = [provider.name for provider in self.providers.values()]
        if len(names) == 1:
            return names[0]
        return None

    def task_slate(self, owner_id: str) -> tuple[TaskSlateEntry, ...]:
        """Return active task names and status lines for front-brain coreference."""
        entries: list[TaskSlateEntry] = []
        lean = self.mode == "lean"
        for task in self.ledger.list_tasks(owner_id):
            if task.kind != "main" or task.terminal:
                continue
            name = task.name or task.title
            if not name:
                continue
            status_line = (
                coarse_task_status(task.status)
                if lean
                else (task.result or f"状态:{task.status}")
            )
            entries.append(TaskSlateEntry(name=name, status_line=status_line))
        for task in self.ledger.recent_terminal_tasks(
            owner_id,
            within_ms=self.recent_task_window_ms,
            limit=self.max_terminal_tasks_per_owner,
        ):
            name = task.name or task.title
            if not name:
                continue
            status_line = (
                coarse_task_status(task.status)
                if lean
                else (task.result or f"状态:{task.status}")
            )
            entries.append(
                TaskSlateEntry(name=name, status_line=status_line, done=True)
            )
        return tuple(entries)

    def _create_task(
        self,
        command: CreateTaskCommand,
        owner_id: str,
        source_turn_id: str,
        source_turn: TurnEnvelope | None,
        *,
        task_id: str,
        name: str = "",
        continuation_task_id: str | None = None,
    ) -> TaskRecord:
        project = (
            self.ledger.get_project(command.project_id)
            if command.project_id
            else None
        )
        if project is None:
            project_id = _project_id(
                owner_id,
                command.provider_name,
                command.project_label,
                source_turn.environment_ref if source_turn else "",
            )
            project = self.ledger.get_project(project_id) or ProjectRecord(
                project_id=project_id,
                owner_id=owner_id,
                label=command.project_label,
                provider_name=command.provider_name,
                environment_ref=(
                    source_turn.environment_ref if source_turn else ""
                ),
            )
        continuation = (
            self.ledger.get_task(continuation_task_id)
            if continuation_task_id is not None
            else None
        )
        if continuation_task_id is not None and continuation is None:
            raise CoordinatorPlanError(
                f"missing continuation task: {continuation_task_id}"
            )
        if continuation is not None and (
            continuation.owner_id != owner_id
            or continuation.project_id != project.project_id
            or continuation.provider_name != command.provider_name
        ):
            raise CoordinatorPlanError(
                "continuation task does not belong to this owner/project/provider"
            )
        lineage_id = (
            continuation.lineage_id
            if continuation is not None
            else task_id
        )
        existing_task = self.ledger.get_task(task_id)
        if existing_task is not None:
            if (
                existing_task.owner_id != owner_id
                or existing_task.source_turn_id != source_turn_id
                or existing_task.provider_name != command.provider_name
                or existing_task.lineage_id != lineage_id
            ):
                raise CoordinatorPlanError(
                    f"idempotent task identity collision: {task_id}"
                )
            return existing_task
        task = TaskRecord(
            task_id=task_id,
            owner_id=owner_id,
            source_turn_id=source_turn_id,
            project_id=project.project_id,
            title=command.title,
            objective=command.objective,
            instruction=command.instruction,
            provider_name=command.provider_name,
            context_plan=command.context_plan,
            reasoning_profile=command.reasoning_profile,
            source_voice_session_id=(
                source_turn.voice_session_id if source_turn else ""
            ),
            name=name,
            lineage_id=lineage_id,
        )
        contract = SupervisionContractRevision(
            task_id=task_id,
            revision=1,
            body=command.contract_body,
            source_turn_ids=(source_turn_id,),
        )
        self.ledger.create_task(project, task, contract)
        return task

    async def _send_task(
        self,
        command: SendTaskCommand,
        source_turn: TurnEnvelope | None,
        operation_id: str,
    ) -> None:
        async with self._task_lock(command.task_id):
            task = self._owned_task(command.task_id)
            capabilities = self.providers.get(task.provider_name).capabilities
            if source_turn is not None:
                task = self._link_context_turn(task, source_turn)
            effective_context_plan = command.context_plan or task.context_plan
            if command.context_plan is not None:
                effective_context_plan = dataclasses.replace(
                    effective_context_plan,
                    event_refs=tuple(
                        dict.fromkeys(
                            (
                                *task.context_plan.event_refs,
                                *effective_context_plan.event_refs,
                            )
                        )
                    ),
                )
            effective_context_plan = self._context_plan_with_turn(
                effective_context_plan, source_turn
            )
            if task.status == "queued":
                if command.mode == "query":
                    raise CoordinatorPlanError(
                        "cannot fork a side query before the worker run starts"
                    )
                instruction = (
                    f"{task.instruction}\n\n{command.instruction}"
                    if command.instruction
                    else task.instruction
                )
                self.ledger.save_task_once(
                    operation_id,
                    dataclasses.replace(
                        task,
                        instruction=instruction,
                        context_plan=effective_context_plan,
                        updated_at_ms=now_ms(),
                    ),
                )
                return
            mode = "query" if command.mode == "query" else "update"
            message = new_worker_message(
                command.instruction,
                mode,
                message_id=stable_id("message", operation_id),
                context_plan=command.context_plan,
                source_turn=(
                    source_turn
                    if capabilities.context_provisioning == "push_bounded"
                    else None
                ),
                preempt=command.preempt,
            )
            provider_command = self._enqueue_provider_message(
                task, message
            )
            run = self._runtime_runs.get(task.active_run_id or "")
            if run is not None:
                accepted = await self._deliver_provider_command(
                    run, provider_command
                )
                if not accepted:
                    raise CoordinatorPlanError(
                        "worker rejected the task message"
                    )
            if command.mode == "update":
                current = self._owned_task(task.task_id)
                if not current.terminal:
                    self.ledger.save_task(
                        dataclasses.replace(
                            current,
                            instruction=command.instruction,
                            context_plan=effective_context_plan,
                            updated_at_ms=now_ms(),
                        )
                    )

    def _link_context_turn(
        self, task: TaskRecord, turn: TurnEnvelope
    ) -> TaskRecord:
        """Associate a trusted follow-up Turn without pushing it to a Worker."""

        context_plan = self._context_plan_with_turn(task.context_plan, turn)
        if context_plan is task.context_plan:
            return task
        linked = dataclasses.replace(
            task,
            context_plan=context_plan,
            updated_at_ms=now_ms(),
        )
        self.ledger.save_task(linked)
        return linked

    @staticmethod
    def _context_plan_with_turn(
        plan: ContextPlan, turn: TurnEnvelope | None
    ) -> ContextPlan:
        if turn is None or turn.receipt_key in plan.event_refs:
            return plan
        return dataclasses.replace(
            plan,
            event_refs=(*plan.event_refs, turn.receipt_key),
        )

    async def _update_contract(
        self,
        command: UpdateContractCommand,
        source_turn_id: str,
        operation_id: str,
    ) -> None:
        async with self._task_lock(command.task_id):
            contract = self.ledger.revise_contract(
                command.task_id, command.body, (source_turn_id,)
            )
            task = self._owned_task(command.task_id)
            if task.active_run_id is None:
                return
            message = new_worker_message(
                "",
                "policy",
                message_id=stable_id("message", operation_id),
                policy=WorkerPolicyView.from_contract(contract),
            )
            provider_command = self._enqueue_provider_message(
                task, message
            )
            run = self._runtime_runs.get(task.active_run_id)
            if run is None:
                return
            accepted = await self._deliver_provider_command(
                run, provider_command
            )
            if not accepted:
                LOGGER.warning(
                    "worker did not accept contract revision %s for %s",
                    contract.revision,
                    task.task_id,
                )

    async def _cancel_task(self, task_id: str) -> None:
        async with self._task_lock(task_id):
            task = self._owned_task(task_id)
            if task.terminal:
                return
            if task.status == "queued":
                cancelled = dataclasses.replace(
                    task,
                    status="cancelled",
                    result="Cancelled before the worker started",
                    updated_at_ms=now_ms(),
                )
                self.ledger.save_task(cancelled)
                self._close_task_interactions(task_id)
                self._finished_event(task_id).set()
                self.delivery.enqueue(
                    owner_id=task.owner_id,
                    task_id=task_id,
                    event_id=None,
                    interaction_id=None,
                    timing=(
                        self._required_contract(task)
                        .body.delivery_policy.final
                    ),
                    topic="final",
                    speech_hint="",
                    status="cancelled",
                    dedupe_key=f"terminal:{task_id}:{task.generation}",
                )
                return
            cancel_request_id = (
                task.cancel_request_id or new_id("cancel")
            )
            cancelling = dataclasses.replace(
                task,
                status="cancelling",
                cancel_request_id=cancel_request_id,
                updated_at_ms=now_ms(),
            )
            self.ledger.save_task(cancelling)
            run = self._runtime_runs.get(task.active_run_id or "")
            if run is None:
                return
            terminal = await asyncio.wait_for(
                run.cancel(cancel_request_id),
                self.provider_command_timeout_s,
            )
            if terminal:
                await self._terminal_from_cancel(task)

    async def _resolve_interaction(
        self,
        command: ResolveInteractionCommand,
        *,
        message_id: str | None = None,
    ) -> bool:
        resolving = self.ledger.begin_interaction_resolution(
            command.interaction_id, command.response, command.decision
        )
        if resolving is None:
            return False
        success = True
        if resolving.source == "worker":
            try:
                message = new_worker_message(
                    command.response,
                    "interaction_reply",
                    message_id=message_id
                    or stable_id(
                        "message",
                        resolving.interaction_id,
                        resolving.decision or "",
                        command.response,
                    ),
                    interaction_id=resolving.interaction_id,
                    source_event_id=resolving.source_event_id,
                    decision=resolving.decision,
                )
                provider_command = self._enqueue_provider_message(
                    self._owned_task(resolving.task_id or ""),
                    message,
                    run_id=resolving.run_id,
                )
                run = self._runtime_runs.get(resolving.run_id or "")
                if run is None:
                    self.ledger.cancel_interaction_deliveries(
                        resolving.interaction_id
                    )
                    if resolving.task_id:
                        self._refresh_blocked(resolving.task_id)
                    return True
                else:
                    success = await self._deliver_provider_command(
                        run, provider_command
                    )
            except Exception:
                success = False
                LOGGER.warning(
                    "worker interaction response failed", exc_info=True
                )
        self.ledger.finish_interaction_resolution(
            resolving.interaction_id, success=success
        )
        if success:
            self.ledger.cancel_interaction_deliveries(
                resolving.interaction_id
            )
        if resolving.task_id:
            self._refresh_blocked(resolving.task_id)
        return success

    def _schedule(
        self, task_id: str, run: RunRecord | None = None
    ) -> None:
        existing = self._task_jobs.get(task_id)
        if existing is not None and not existing.done():
            return
        job = asyncio.create_task(
            self._dispatch(task_id, run),
            name=f"gander-task-{task_id}",
        )
        self._task_jobs[task_id] = job

        def finish(completed: asyncio.Task[None]) -> None:
            if self._task_jobs.get(task_id) is completed:
                self._task_jobs.pop(task_id, None)
            if not completed.cancelled() and completed.exception() is not None:
                LOGGER.error(
                    "task dispatch failed",
                    exc_info=(
                        type(completed.exception()),
                        completed.exception(),
                        completed.exception().__traceback__,
                    ),
                )

        job.add_done_callback(finish)

    async def _dispatch(
        self, task_id: str, persisted_run: RunRecord | None
    ) -> None:
        task = self._owned_task(task_id)
        try:
            provider = self.providers.get(task.provider_name)
            slot = self._provider_slots[provider.name]
            project = self.ledger.get_project(task.project_id)
            if project is None:
                raise RuntimeError(
                    f"task project is unavailable: {task.project_id}"
                )
            resource_key = _provider_project_resource_key(provider, project)
        except Exception as exc:
            LOGGER.warning("worker resource resolution failed", exc_info=True)
            await self._fail_active_run_guarded(task_id, str(exc))
            return
        resource_lock = self._resource_locks.setdefault(
            f"{provider.name}\0{resource_key}", asyncio.Lock()
        )
        # Limit waiters per workspace so independent projects retain provider slots.
        async with resource_lock, slot:
            async with self._task_lock(task_id):
                if self._closed:
                    return
                task = self._owned_task(task_id)
                if task.terminal:
                    return
                run_record = persisted_run
                if run_record is None:
                    run_record = RunRecord(
                        run_id=new_id("run"),
                        task_id=task.task_id,
                        project_id=task.project_id,
                        provider_name=task.provider_name,
                        generation=task.generation,
                        status="running",
                        started_at_ms=now_ms(),
                    )
                    task = dataclasses.replace(
                        task,
                        status="running",
                        active_run_id=run_record.run_id,
                        updated_at_ms=now_ms(),
                    )
                    self.ledger.save_task(task)
                    self.ledger.save_run(run_record, task.owner_id)
            try:
                project_session = await self._open_project(task)
                control = WorkerControl(
                    self,
                    task.task_id,
                    run_record.run_id,
                    provider.capabilities,
                )
                source_turn = self.ledger.get_turn(
                    task.source_turn_id,
                    owner_id=task.owner_id,
                    voice_session_id=(
                        task.source_voice_session_id or None
                    ),
                )
                source_turn = (
                    self._compile_source_turn(
                        task, source_turn, provider.capabilities
                    )
                    if provider.capabilities.context_provisioning
                    == "push_bounded"
                    else None
                )
                parent_backend_session_id = ""
                resolved_parent_run_id = ""
                if task.kind == "side_query":
                    parent_run = (
                        self.ledger.get_run(task.parent_run_id)
                        if task.parent_run_id
                        else None
                    )
                    if parent_run is None and not task.parent_run_id:
                        parent_task = self.ledger.get_task(task.parent_task_id)
                        if parent_task is not None:
                            parent_run = self._latest_run_for_task(parent_task)
                    if (
                        parent_run is None
                        or parent_run.task_id != task.parent_task_id
                        or parent_run.provider_name != task.provider_name
                        or parent_run.project_id != task.project_id
                    ):
                        raise RuntimeError(
                            "side-query parent run is unavailable or mismatched"
                        )
                    parent_backend_session_id = (
                        parent_run.backend_session_id
                    )
                    resolved_parent_run_id = parent_run.run_id
                request = WorkerRequest(
                    task_id=task.task_id,
                    run_id=run_record.run_id,
                    project_id=task.project_id,
                    owner_id=task.owner_id,
                    generation=task.generation,
                    instruction=task.instruction,
                    context_plan=task.context_plan,
                    policy=WorkerPolicyView.from_contract(
                        self._required_contract(task)
                    ),
                    reasoning_profile=task.reasoning_profile,
                    source_turn=source_turn,
                    lineage_id=task.lineage_id,
                    kind=task.kind,
                    parent_task_id=task.parent_task_id,
                    parent_run_id=resolved_parent_run_id,
                    parent_backend_session_id=parent_backend_session_id,
                    original_turn=(
                        source_turn.final_asr
                        if source_turn is not None
                        else task.objective
                    ),
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                LOGGER.warning("worker project open failed", exc_info=True)
                if not self._closed:
                    await self._fail_active_run_guarded(task.task_id, str(exc))
                return
            run: WorkerRun | None = None
            terminal_seen = False
            try:
                if persisted_run is None:
                    run = await asyncio.wait_for(
                        project_session.start(request, control),
                        self.provider_start_timeout_s,
                    )
                else:
                    resume = getattr(project_session, "resume", None)
                    if resume is None:
                        raise RuntimeError(
                            "provider declared session_resume without resume()"
                        )
                    run = await asyncio.wait_for(
                        resume(
                            request,
                            control,
                            persisted_run.backend_session_id,
                        ),
                        self.provider_start_timeout_s,
                    )
                session_id = getattr(run, "session_id", "")
                if session_id != run_record.backend_session_id:
                    run_record = dataclasses.replace(
                        run_record, backend_session_id=session_id
                    )
                    self.ledger.save_run(run_record, task.owner_id)
                self._runtime_runs[run_record.run_id] = run
                await self._replay_provider_commands(
                    run_record.run_id, run
                )
                task = self._owned_task(task_id)
                if task.status == "cancelling":
                    cancel_request_id = (
                        task.cancel_request_id or new_id("cancel")
                    )
                    if task.cancel_request_id is None:
                        task = dataclasses.replace(
                            task,
                            cancel_request_id=cancel_request_id,
                            updated_at_ms=now_ms(),
                        )
                        self.ledger.save_task(task)
                    terminal_seen = await asyncio.wait_for(
                        run.cancel(cancel_request_id),
                        self.provider_command_timeout_s,
                    )
                    if terminal_seen:
                        await self._terminal_from_cancel(task)
                if not terminal_seen:
                    async for event in run.events():
                        accepted = await self.ingest_worker_event(event)
                        if accepted and isinstance(event.payload, DonePayload):
                            terminal_seen = True
                            break
                if not terminal_seen and not self._closed:
                    await self._fail_active_run_guarded(
                        task.task_id, "worker event stream ended without done"
                    )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                LOGGER.warning("worker run failed", exc_info=True)
                if not self._closed:
                    await self._fail_active_run_guarded(
                        task.task_id, str(exc)
                    )
            finally:
                self._runtime_runs.pop(run_record.run_id, None)
                if run is not None:
                    try:
                        await asyncio.wait_for(
                            run.close(),
                            self.provider_command_timeout_s,
                        )
                    except Exception:
                        LOGGER.warning("worker run close failed", exc_info=True)

    async def _open_project(self, task: TaskRecord) -> WorkerProject:
        existing = self._project_sessions.get(task.project_id)
        if existing is not None:
            return existing
        lock = self._project_open_locks.setdefault(
            task.project_id, asyncio.Lock()
        )
        async with lock:
            existing = self._project_sessions.get(task.project_id)
            if existing is not None:
                return existing
            project = self.ledger.get_project(task.project_id)
            if project is None:
                raise RuntimeError(f"missing project: {task.project_id}")
            session = await asyncio.wait_for(
                self.providers.get(task.provider_name).open_project(
                    project
                ),
                self.provider_start_timeout_s,
            )
            backend_session_id = getattr(session, "session_id", "")
            if (
                backend_session_id
                and backend_session_id != project.backend_session_id
            ):
                self.ledger.save_project(
                    dataclasses.replace(
                        project,
                        backend_session_id=backend_session_id,
                        updated_at_ms=now_ms(),
                    )
                )
            self._project_sessions[task.project_id] = session
            return session

    def _compile_source_turn(
        self,
        task: TaskRecord,
        turn: TurnEnvelope | None,
        capabilities: BackendCapabilities,
    ) -> TurnEnvelope | None:
        if turn is None:
            return None
        requested = set(task.context_plan.media_refs)
        if not requested:
            return dataclasses.replace(turn, media_refs=())
        by_path = {media.path: media for media in turn.media_refs}
        missing = requested - set(by_path)
        if missing:
            raise RuntimeError(
                f"selected media refs are unavailable: {sorted(missing)}"
            )
        selected = tuple(by_path[ref] for ref in task.context_plan.media_refs)
        unsupported = tuple(
            media
            for media in selected
            if _media_modality(media.kind) not in capabilities.modalities
        )
        if unsupported:
            kinds = sorted({media.kind for media in unsupported})
            raise RuntimeError(
                f"backend does not support selected media: {kinds}"
            )
        return dataclasses.replace(turn, media_refs=selected)

    def _enqueue_provider_message(
        self,
        task: TaskRecord,
        message: WorkerMessage,
        *,
        run_id: str | None = None,
    ) -> ProviderCommandRecord:
        target_run_id = run_id or task.active_run_id
        if not target_run_id:
            raise CoordinatorPlanError("task has no active provider run")
        return self.ledger.enqueue_provider_command(
            ProviderCommandRecord(
                command_id=message.message_id,
                owner_id=task.owner_id,
                task_id=task.task_id,
                run_id=target_run_id,
                message=message,
            )
        )

    async def _deliver_provider_command(
        self, run: WorkerRun, command: ProviderCommandRecord
    ) -> bool:
        current = self.ledger.get_provider_command(command.command_id)
        if current is None:
            raise RuntimeError("provider command disappeared from outbox")
        if current.state == "sent":
            return True
        if current.state in {"failed", "cancelled"}:
            return False
        self.ledger.set_provider_command_state(
            command.command_id, "sending"
        )
        try:
            accepted = await asyncio.wait_for(
                run.send(command.message),
                self.provider_command_timeout_s,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.ledger.set_provider_command_state(
                command.command_id, "failed", error=str(exc)
            )
            return False
        self.ledger.set_provider_command_state(
            command.command_id,
            "sent" if accepted else "failed",
            error="" if accepted else "worker rejected the command",
        )
        return accepted

    async def _replay_provider_commands(
        self, run_id: str, run: WorkerRun
    ) -> None:
        for command in self.ledger.list_provider_commands(run_id=run_id):
            if command.state not in {"pending", "sending"}:
                continue
            accepted = await self._deliver_provider_command(run, command)
            interaction_id = command.message.interaction_id
            if not accepted or interaction_id is None:
                continue
            interaction = self.ledger.get_interaction(interaction_id)
            if interaction is not None and interaction.state == "resolving":
                self.ledger.finish_interaction_resolution(
                    interaction_id, success=True
                )
                self.ledger.cancel_interaction_deliveries(interaction_id)
                if interaction.task_id:
                    self._refresh_blocked(interaction.task_id)

    def _queue_expired_interaction(
        self, interaction: PendingInteraction
    ) -> ProviderCommandRecord | None:
        if (
            interaction.source != "worker"
            or interaction.task_id is None
            or interaction.run_id is None
        ):
            return None
        task = self.ledger.get_task(interaction.task_id)
        if task is None:
            return None
        message = WorkerMessage(
            message_id=f"message_expired_{interaction.interaction_id}",
            instruction=interaction.response or "Interaction expired",
            mode="interaction_reply",
            interaction_id=interaction.interaction_id,
            source_event_id=interaction.source_event_id,
            decision=interaction.decision,
        )
        return self._enqueue_provider_message(
            task, message, run_id=interaction.run_id
        )

    def _finish_sent_interactions(self) -> None:
        sent_by_interaction = {
            command.message.interaction_id
            for command in self.ledger.list_provider_commands()
            if command.state == "sent"
            and command.message.interaction_id is not None
        }
        for interaction_id in sent_by_interaction:
            interaction = self.ledger.get_interaction(interaction_id)
            if interaction is None or interaction.state != "resolving":
                continue
            self.ledger.finish_interaction_resolution(
                interaction_id, success=True
            )
            self.ledger.cancel_interaction_deliveries(interaction_id)
            if interaction.task_id:
                self._refresh_blocked(interaction.task_id)

    def _recover_resolving_interactions(self) -> None:
        commands = self.ledger.list_provider_commands()
        for interaction in self.ledger.list_interactions():
            if interaction.state != "resolving":
                continue
            if interaction.source == "coordinator":
                self.ledger.finish_interaction_resolution(
                    interaction.interaction_id, success=True
                )
                self.ledger.cancel_interaction_deliveries(
                    interaction.interaction_id
                )
            elif not any(
                command.message.interaction_id
                == interaction.interaction_id
                and command.state in {"pending", "sending", "sent"}
                for command in commands
            ):
                self.ledger.finish_interaction_resolution(
                    interaction.interaction_id, success=False
                )
            if interaction.task_id:
                self._refresh_blocked(interaction.task_id)

    def _schedule_aggregate(self, aggregate: ActivityAggregate) -> None:
        current = self._aggregate_jobs.get(aggregate.aggregate_id)
        if current is not None and not current.done():
            return

        async def wait_and_flush() -> None:
            delay = max(0, aggregate.due_at_ms - now_ms()) / 1000
            if delay:
                await asyncio.sleep(delay)
            await self._flush_aggregate(aggregate.aggregate_id)

        job = asyncio.create_task(
            wait_and_flush(),
            name=f"gander-aggregate-{aggregate.aggregate_id}",
        )
        self._aggregate_jobs[aggregate.aggregate_id] = job
        self._auxiliary_jobs.add(job)

        def finish(completed: asyncio.Task[None]) -> None:
            self._auxiliary_jobs.discard(completed)
            if self._aggregate_jobs.get(aggregate.aggregate_id) is completed:
                self._aggregate_jobs.pop(aggregate.aggregate_id, None)
            if not completed.cancelled() and completed.exception() is not None:
                LOGGER.warning(
                    "activity aggregate flush failed",
                    exc_info=(
                        type(completed.exception()),
                        completed.exception(),
                        completed.exception().__traceback__,
                    ),
                )

        job.add_done_callback(finish)

    async def _flush_task_aggregates(self, task_id: str) -> None:
        for aggregate in self.ledger.list_activity_aggregates(
            task_id=task_id, state="open"
        ):
            await self._flush_aggregate(aggregate.aggregate_id)

    async def _flush_aggregate(self, aggregate_id: str) -> None:
        aggregate = self.ledger.flush_activity_aggregate(aggregate_id)
        if aggregate is None:
            return
        task = self.ledger.get_task(aggregate.task_id)
        contract = self.ledger.get_contract(
            aggregate.task_id, aggregate.contract_revision
        )
        if task is None or contract is None:
            return
        latest = aggregate.summaries[-1]
        speech = (
            f"{task.title}: {latest}"
            if aggregate.event_count == 1
            else (
                f"{task.title}: {aggregate.event_count} progress updates; "
                f"latest: {latest}"
            )
        )
        self.delivery.enqueue(
            owner_id=aggregate.owner_id,
            task_id=aggregate.task_id,
            event_id=aggregate.event_ids[-1],
            interaction_id=None,
            timing=contract.body.delivery_policy.aggregate,
            topic="aggregate",
            speech_hint=speech,
            dedupe_key=f"aggregate:{aggregate.aggregate_id}",
        )

    def _lean_arbitrate(
        self, event: WorkerEvent, capabilities: BackendCapabilities
    ) -> ArbitrationDecision:
        """Map lean-mode worker events directly to delivery and permission policy."""
        payload = event.payload
        if isinstance(payload, UpdatePayload):
            timing = (
                "interrupt" if payload.severity == "high_risk" else "safe_pause"
            )
            return ArbitrationDecision(timing, reason="lean_update")
        if isinstance(payload, InteractionPayload):
            if not capabilities.interactions:
                return ArbitrationDecision(
                    "safe_pause",
                    accepted=False,
                    reason="backend_did_not_declare_interactions",
                )
            if (
                payload.kind == "permission"
                and event.owner_id in self._standing_allow
            ):
                # Route standing grants through the existing auto-resolve path.
                return ArbitrationDecision(
                    "safe_pause",
                    interaction_route="coordinator",
                    reason="allowed_by_contract",
                )
            return ArbitrationDecision(
                "safe_pause", interaction_route="user", reason="lean_interaction"
            )
        if isinstance(payload, DonePayload):
            return ArbitrationDecision("safe_pause", reason="lean_terminal")
        raise TypeError(f"unknown worker payload: {type(payload)!r}")

    async def _reduce_event(
        self, event: WorkerEvent, *, recovering: bool = False
    ) -> None:
        task = self._owned_task(event.task_id, event.owner_id)
        contract_revision = self.ledger.event_contract_revision(event.event_id)
        contract = self.ledger.get_contract(
            task.task_id, contract_revision
        )
        if contract is None:
            raise RuntimeError(
                f"missing evaluated contract revision {contract_revision} "
                f"for {task.task_id}"
            )
        capabilities = self.providers.get(task.provider_name).capabilities
        if self.mode == "lean":
            decision = self._lean_arbitrate(event, capabilities)
        else:
            decision = self.supervision.arbitrate(
                event, contract, capabilities
            )
        payload = event.payload
        if isinstance(payload, UpdatePayload):
            if not decision.accepted:
                return
            if (
                self.mode == "coordinator"
                and payload.kind == "activity"
                and payload.severity == "normal"
                and contract.body.delivery_policy.aggregate_window_ms > 0
            ):
                aggregate = self.ledger.append_activity_aggregate(
                    event,
                    contract_revision=contract.revision,
                    window_ms=(
                        contract.body.delivery_policy.aggregate_window_ms
                    ),
                )
                self._schedule_aggregate(aggregate)
            if payload.kind == "milestone":
                await self._flush_task_aggregates(task.task_id)
            if decision.timing != "hold":
                rule = next(
                    (
                        item
                        for item in contract.body.notify_policy.milestones
                        if item.key == payload.milestone
                    ),
                    None,
                )
                dedupe = (
                    f"milestone:{task.task_id}:{payload.milestone}"
                    if rule is not None and rule.once
                    else f"event:{event.event_id}"
                )
                topic = "risk" if payload.severity == "high_risk" else "milestone"
                self.delivery.enqueue(
                    owner_id=task.owner_id,
                    task_id=task.task_id,
                    event_id=event.event_id,
                    interaction_id=None,
                    timing=decision.timing,
                    topic=topic,
                    speech_hint=payload.summary,
                    dedupe_key=dedupe,
                )
            return

        if isinstance(payload, InteractionPayload):
            if not decision.accepted:
                await self._fail_active_run(
                    task.task_id, decision.reason
                )
                return
            interaction = self.interactions.from_worker(
                event,
                capabilities,
                audience=(
                    "user"
                    if decision.interaction_route == "user"
                    else "coordinator"
                ),
            )
            if (
                payload.kind == "permission"
                and interaction.expires_at_ms is None
            ):
                interaction = dataclasses.replace(
                    interaction,
                    expires_at_ms=now_ms() + self.permission_timeout_ms,
                )
                self.ledger.save_interaction(interaction)
            self._refresh_blocked(task.task_id)
            if decision.interaction_route == "user" or recovering:
                if recovering and interaction.audience != "user":
                    interaction = dataclasses.replace(
                        interaction, audience="user"
                    )
                    self.ledger.save_interaction(interaction)
                self._deliver_interaction(task, contract, event, interaction)
                return
            if payload.kind == "permission" and decision.reason in {
                "allowed_by_contract",
                "denied_by_contract",
                "unknown_action",
                "backend_cannot_enforce_permission",
            }:
                allowed = decision.reason == "allowed_by_contract"
                await self._resolve_interaction(
                    ResolveInteractionCommand(
                        interaction.interaction_id,
                        "Allowed by supervision policy"
                        if allowed
                        else "Denied by supervision policy",
                        "allow" if allowed else "deny",
                    )
                )
                return
            await self._coordinate_worker_interaction(event, interaction)
            current = self.ledger.get_interaction(interaction.interaction_id)
            if current is not None and current.state == "pending":
                promoted = dataclasses.replace(current, audience="user")
                self.ledger.save_interaction(promoted)
                self._deliver_interaction(
                    task, contract, event, promoted
                )
            return

        if isinstance(payload, DonePayload):
            await self._flush_task_aggregates(task.task_id)
            run = self.ledger.get_run(event.run_id)
            if run is None:
                return
            updated_task = dataclasses.replace(
                task,
                status=payload.status,
                active_run_id=None,
                result=payload.result,
                error=payload.result if payload.status == "failed" else "",
                updated_at_ms=now_ms(),
            )
            updated_run = dataclasses.replace(
                run,
                status=payload.status,
                ended_at_ms=now_ms(),
                error=payload.result if payload.status == "failed" else "",
            )
            self.ledger.save_task(updated_task)
            self.ledger.save_run(updated_run, task.owner_id)
            self.ledger.fail_provider_commands_for_run(
                event.run_id, "run is terminal"
            )
            self._close_task_interactions(task.task_id)
            self._finished_event(task.task_id).set()
            self.delivery.enqueue(
                owner_id=task.owner_id,
                task_id=task.task_id,
                event_id=event.event_id,
                interaction_id=None,
                timing=decision.timing,
                topic="final",
                speech_hint=payload.result,
                status=payload.status,
                dedupe_key=f"terminal:{task.task_id}:{task.generation}",
            )

    async def _coordinate_worker_interaction(
        self, event: WorkerEvent, interaction: PendingInteraction
    ) -> None:
        model_lock = self._coordinator_model_locks.setdefault(
            event.owner_id, asyncio.Lock()
        )
        async with model_lock:
            request_id = stable_id(
                "coordination", event.event_id
            )
            context = self._coordinator_context(
                request_id=request_id,
                reason="worker_interaction",
                owner_id=event.owner_id,
                turn=None,
                worker_event=event,
            )
            try:
                plan = await self._coordinate(context)
                owner_lock = self._owner_locks.setdefault(
                    event.owner_id, asyncio.Lock()
                )
                async with owner_lock:
                    await self._apply_plan(
                        plan,
                        owner_id=event.owner_id,
                        request_id=request_id,
                        source_turn_id=event.event_id,
                        source_turn=None,
                    )
            except Exception:
                LOGGER.warning(
                    "Coordinator could not resolve worker interaction",
                    exc_info=True,
                )

    def _deliver_interaction(
        self,
        task: TaskRecord,
        contract: SupervisionContractRevision,
        event: WorkerEvent,
        interaction: PendingInteraction,
    ) -> None:
        self.delivery.enqueue(
            owner_id=task.owner_id,
            task_id=task.task_id,
            event_id=event.event_id,
            interaction_id=interaction.interaction_id,
            timing=contract.body.delivery_policy.interaction,
            topic="interaction",
            speech_hint=interaction.prompt,
            dedupe_key=f"interaction:{interaction.fingerprint}",
        )

    async def _terminal_from_cancel(self, task: TaskRecord) -> None:
        current = self._owned_task(task.task_id)
        if current.terminal:
            return
        run = (
            self.ledger.get_run(current.active_run_id)
            if current.active_run_id
            else None
        )
        cancelled = dataclasses.replace(
            current,
            status="cancelled",
            active_run_id=None,
            result="Cancellation confirmed by the worker",
            updated_at_ms=now_ms(),
        )
        self.ledger.save_task(cancelled)
        self._close_task_interactions(task.task_id)
        if run is not None:
            self.ledger.fail_provider_commands_for_run(
                run.run_id, "run cancelled"
            )
            self.ledger.save_run(
                dataclasses.replace(
                    run, status="cancelled", ended_at_ms=now_ms()
                ),
                current.owner_id,
            )
        self._finished_event(task.task_id).set()
        self.delivery.enqueue(
            owner_id=current.owner_id,
            task_id=current.task_id,
            event_id=None,
            interaction_id=None,
            timing=self._required_contract(
                current
            ).body.delivery_policy.final,
            topic="final",
            speech_hint="",
            status="cancelled",
            dedupe_key=f"terminal:{current.task_id}:{current.generation}",
        )

    async def _fail_active_run(self, task_id: str, error: str) -> None:
        task = self._owned_task(task_id)
        run = (
            self.ledger.get_run(task.active_run_id)
            if task.active_run_id
            else None
        )
        await self._fail_task(task, run, error)

    async def _fail_active_run_guarded(
        self, task_id: str, error: str
    ) -> None:
        async with self._task_lock(task_id):
            task = self._owned_task(task_id)
            if task.terminal:
                return
            run = (
                self.ledger.get_run(task.active_run_id)
                if task.active_run_id
                else None
            )
            await self._fail_task(task, run, error)

    async def _fail_task(
        self, task: TaskRecord, run: RunRecord | None, error: str
    ) -> None:
        current = self._owned_task(task.task_id)
        if current.terminal:
            return
        failed = dataclasses.replace(
            current,
            status="failed",
            active_run_id=None,
            error=error,
            result=error,
            updated_at_ms=now_ms(),
        )
        self.ledger.save_task(failed)
        self._close_task_interactions(task.task_id)
        if run is not None:
            self.ledger.fail_provider_commands_for_run(
                run.run_id, error
            )
            self.ledger.save_run(
                dataclasses.replace(
                    run,
                    status="failed",
                    ended_at_ms=now_ms(),
                    error=error,
                ),
                task.owner_id,
            )
        self._finished_event(task.task_id).set()
        contract = self._required_contract(task)
        self.delivery.enqueue(
            owner_id=task.owner_id,
            task_id=task.task_id,
            event_id=None,
            interaction_id=None,
            timing=contract.body.delivery_policy.final,
            topic="final",
            speech_hint=error,
            status="failed",
            dedupe_key=f"terminal:{task.task_id}:{task.generation}",
        )

    def _refresh_blocked(self, task_id: str) -> None:
        task = self.ledger.get_task(task_id)
        if task is None or task.terminal:
            return
        pending = tuple(
            interaction
            for interaction in self.ledger.list_interactions(task.owner_id)
            if interaction.task_id == task_id
            and interaction.state in {"pending", "resolving"}
        )
        actions = tuple(
            dict.fromkeys(
                interaction.action_key or interaction.interaction_id
                for interaction in pending
                if interaction.effective_scope == "action"
            )
        )
        branches = tuple(
            interaction.interaction_id
            for interaction in pending
            if interaction.effective_scope == "branch"
        )
        run_blocked = any(
            interaction.effective_scope == "run"
            for interaction in pending
        )
        if (
            actions,
            branches,
            run_blocked,
        ) == (
            task.blocked_actions,
            task.blocked_branches,
            task.run_blocked,
        ):
            return
        self.ledger.save_task(
            dataclasses.replace(
                task,
                blocked_actions=actions,
                blocked_branches=branches,
                run_blocked=run_blocked,
                updated_at_ms=now_ms(),
            )
        )

    def _close_task_interactions(self, task_id: str) -> None:
        task = self.ledger.get_task(task_id)
        if task is None:
            return
        for interaction in self.ledger.list_interactions(task.owner_id):
            if (
                interaction.task_id != task_id
                or interaction.state not in {"pending", "resolving"}
            ):
                continue
            self.ledger.save_interaction(
                dataclasses.replace(
                    interaction,
                    state="cancelled",
                    resolved_at_ms=now_ms(),
                )
            )
            self.ledger.cancel_provider_commands_for_interaction(
                interaction.interaction_id
            )
            self.ledger.cancel_interaction_deliveries(
                interaction.interaction_id
            )
        current = self.ledger.get_task(task_id)
        if current is not None and (
            current.blocked_actions
            or current.blocked_branches
            or current.run_blocked
        ):
            self.ledger.save_task(
                dataclasses.replace(
                    current,
                    blocked_actions=(),
                    blocked_branches=(),
                    run_blocked=False,
                    updated_at_ms=now_ms(),
                )
            )

    def _owned_task(
        self, task_id: str, owner_id: str | None = None
    ) -> TaskRecord:
        task = self.ledger.get_task(task_id)
        if task is None:
            raise CoordinatorPlanError(f"unknown task: {task_id}")
        if owner_id is not None and task.owner_id != owner_id:
            raise CoordinatorPlanError(f"task does not belong to owner: {task_id}")
        return task

    def _required_contract(
        self, task: TaskRecord
    ) -> SupervisionContractRevision:
        contract = self.ledger.get_contract(
            task.task_id, task.contract_revision
        )
        if contract is None:
            raise RuntimeError(f"missing task contract: {task.task_id}")
        return contract

    def _finished_event(self, task_id: str) -> asyncio.Event:
        return self._finished.setdefault(task_id, asyncio.Event())

    def _task_lock(self, task_id: str) -> asyncio.Lock:
        return self._task_locks.setdefault(task_id, asyncio.Lock())


async def _ready(value: Any) -> Any:
    return value


def _memory_hit(raw: Any) -> MemoryHit:
    if isinstance(raw, MemoryHit):
        return raw
    if not isinstance(raw, dict):
        raise TypeError("memory hit must be MemoryHit or an object")
    return MemoryHit(
        ref=raw.get("ref"),
        text=raw.get("text"),
        score=raw.get("score"),
        metadata=raw.get("metadata") or {},
    )


def _json_mapping(
    value: Any, name: str, *, max_chars: int = 4096
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be an object")
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be JSON-compatible") from exc
    if len(encoded) > max_chars:
        return {"truncated": True, "json_prefix": encoded[:max_chars]}
    decoded = json.loads(encoded)
    assert isinstance(decoded, dict)
    return decoded


def _context_score(query: str, candidate: dict[str, Any]) -> int:
    needle = query.casefold().strip()
    haystack = str(candidate.get("_search_text") or "").casefold()
    if not needle:
        return 1
    score = 100 if needle in haystack else 0
    terms = re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", needle)
    score += sum(3 for term in set(terms) if term in haystack)
    return score


def _worker_event_text(event: WorkerEvent) -> str:
    payload = event.payload
    if isinstance(payload, UpdatePayload):
        return "\n".join(
            value
            for value in (
                payload.summary,
                f"milestone: {payload.milestone}" if payload.milestone else "",
                f"next: {payload.next_step}" if payload.next_step else "",
            )
            if value
        )
    if isinstance(payload, InteractionPayload):
        choices = ", ".join(payload.choices)
        return (
            payload.prompt
            if not choices
            else f"{payload.prompt}\nchoices: {choices}"
        )
    assert isinstance(payload, DonePayload)
    unresolved = "\n".join(payload.unresolved)
    return "\n".join(
        value
        for value in (
            f"status: {payload.status}",
            payload.result,
            f"unresolved:\n{unresolved}" if unresolved else "",
        )
        if value
    )


def _project_id(
    owner_id: str, provider: str, label: str, environment_ref: str
) -> str:
    return stable_id(
        "project", owner_id, provider, label, environment_ref
    )


def _provider_project_resource_key(
    provider: WorkerProvider, project: ProjectRecord
) -> str:
    resolver = getattr(provider, "project_resource_key", None)
    value = resolver(project) if callable(resolver) else project.project_id
    if inspect.iscoroutine(value):
        value.close()
        raise TypeError("worker provider project resource key must be synchronous")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("worker provider project resource key must be non-empty")
    value = value.strip()
    if len(value) > 2048:
        raise ValueError("worker provider project resource key is too long")
    return value


def _approved(decision: str | None, response: str) -> bool:
    value = (decision or response).strip().lower()
    return value in {
        "allow",
        "allow_once",
        "allow_session",
        "accept",
        "acceptforsession",
        "approve",
        "approved",
        "yes",
        "ok",
        "同意",
        "允许",
    }


def _generic_approval(text: str) -> bool:
    return text.strip().lower() in {
        "yes",
        "y",
        "ok",
        "okay",
        "approve",
        "allow",
        "可以",
        "好",
        "好的",
        "同意",
        "允许",
        "确认",
    }


def _media_modality(kind: str) -> str:
    if kind == "audio":
        return "audio"
    if kind == "video":
        return "video"
    return "image"


def _media_summary(media: Sequence[Any]) -> dict[str, Any]:
    timestamps = [
        item.timestamp_ms
        for item in media
        if getattr(item, "timestamp_ms", None) is not None
    ]
    summary: dict[str, Any] = {
        "count": len(media),
        "kinds": sorted({str(item.kind) for item in media}),
    }
    if timestamps:
        summary["earliest_timestamp_ms"] = min(timestamps)
        summary["latest_timestamp_ms"] = max(timestamps)
    return summary
