from __future__ import annotations

import dataclasses
import json
import sqlite3
import threading
import types
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, TypeVar, Union, get_args, get_origin, get_type_hints

from .contracts import ContextEvent, new_id, now_ms, stable_key
from .coordination import (
    ActivityAggregate,
    ArbitrationDecision,
    ArtifactEvidence,
    AskPolicy,
    TaskControlResult,
    AuthorizationDecision,
    BackendCapabilities,
    ContractBody,
    CoordinationJob,
    CoordinatorSessionRecord,
    DeliveryRecord,
    DonePayload,
    InteractionPayload,
    MilestoneCondition,
    PendingInteraction,
    ProviderCommandRecord,
    ProjectRecord,
    QuestionProposal,
    RunRecord,
    SupervisionContractRevision,
    TaskRecord,
    TurnEnvelope,
    UpdatePayload,
    WorkerEvent,
)

T = TypeVar("T")


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {
            item.name: _jsonable(getattr(value, item.name))
            for item in dataclasses.fields(value)
        }
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_jsonable(item) for item in value]
    return value


def _dump(value: Any) -> str:
    return json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _load(payload: str) -> dict[str, Any]:
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError("stored payload must be a JSON object")
    return value


@lru_cache(maxsize=None)
def _type_hints(cls: type[Any]) -> dict[str, Any]:
    return get_type_hints(cls)


def _decode(annotation: Any, value: Any) -> Any:
    if annotation is Any:
        return value
    origin = get_origin(annotation)
    arguments = get_args(annotation)
    if origin is Literal:
        if value not in arguments:
            raise ValueError(f"{value!r} is not one of {arguments!r}")
        return value
    if origin in {Union, types.UnionType}:
        if value is None and type(None) in arguments:
            return None
        errors: list[Exception] = []
        for choice in arguments:
            if choice is type(None):
                continue
            try:
                return _decode(choice, value)
            except (TypeError, ValueError) as exc:
                errors.append(exc)
        raise ValueError(f"value does not match {annotation!r}: {errors!r}")
    if origin is tuple:
        item_type = arguments[0] if arguments else Any
        return tuple(_decode(item_type, item) for item in value)
    if origin is frozenset:
        item_type = arguments[0] if arguments else Any
        return frozenset(_decode(item_type, item) for item in value)
    if origin is dict:
        key_type, item_type = arguments or (Any, Any)
        return {
            _decode(key_type, key): _decode(item_type, item)
            for key, item in value.items()
        }
    if dataclasses.is_dataclass(annotation):
        if not isinstance(value, dict):
            raise TypeError(f"{annotation.__name__} must be an object")
        fields = {item.name for item in dataclasses.fields(annotation)}
        unknown = set(value) - fields
        if unknown:
            raise ValueError(
                f"unknown {annotation.__name__} fields: {sorted(unknown)}"
            )
        hints = _type_hints(annotation)
        return annotation(
            **{
                key: _decode(hints[key], item)
                for key, item in value.items()
            }
        )
    return value


_ENTITY_TYPES: dict[str, type[Any]] = {
    "turn": TurnEnvelope,
    "realtime_context": ContextEvent,
    "coordination_job": CoordinationJob,
    "coordinator_session": CoordinatorSessionRecord,
    "project": ProjectRecord,
    "task": TaskRecord,
    "run": RunRecord,
    "artifact": ArtifactEvidence,
    "interaction": PendingInteraction,
    "delivery": DeliveryRecord,
    "provider_command": ProviderCommandRecord,
    "aggregate": ActivityAggregate,
}


class TaskLedger:
    """Small SQLite source of truth for orchestration and delivery state."""

    def __init__(self, database_path: str | Path = ":memory:") -> None:
        self.database_path = str(database_path)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(
            self.database_path,
            check_same_thread=False,
            isolation_level="DEFERRED",
        )
        self._connection.row_factory = sqlite3.Row
        with self._lock:
            self._connection.execute("PRAGMA foreign_keys = ON")
            self._connection.execute("PRAGMA busy_timeout = 5000")
            if self.database_path != ":memory:":
                self._connection.execute("PRAGMA journal_mode = WAL")
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS entities (
                    kind TEXT NOT NULL,
                    entity_id TEXT NOT NULL,
                    owner_id TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    updated_at_ms INTEGER NOT NULL,
                    PRIMARY KEY (kind, entity_id)
                );
                CREATE INDEX IF NOT EXISTS entities_owner_kind
                    ON entities(owner_id, kind, updated_at_ms);

                CREATE TABLE IF NOT EXISTS contracts (
                    task_id TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    payload TEXT NOT NULL,
                    created_at_ms INTEGER NOT NULL,
                    PRIMARY KEY (task_id, revision)
                );

                CREATE TABLE IF NOT EXISTS worker_events (
                    event_id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    generation INTEGER NOT NULL,
                    seq INTEGER NOT NULL,
                    evaluated_contract_revision INTEGER NOT NULL,
                    payload TEXT NOT NULL,
                    processed INTEGER NOT NULL DEFAULT 0,
                    created_at_ms INTEGER NOT NULL,
                    UNIQUE (run_id, generation, seq)
                );
                CREATE INDEX IF NOT EXISTS worker_events_task
                    ON worker_events(task_id, created_at_ms);

                CREATE TABLE IF NOT EXISTS assist_receipts (
                    receipt_key TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at_ms INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS markers (
                    marker_key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    created_at_ms INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS ledger_entries (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    kind TEXT NOT NULL,
                    entity_id TEXT NOT NULL,
                    contract_revision INTEGER,
                    created_at_ms INTEGER NOT NULL
                );
                """
            )
            columns = {
                row["name"]
                for row in self._connection.execute(
                    "PRAGMA table_info(worker_events)"
                ).fetchall()
            }
            if "evaluated_contract_revision" not in columns:
                self._connection.execute(
                    "ALTER TABLE worker_events "
                    "ADD COLUMN evaluated_contract_revision INTEGER NOT NULL DEFAULT 1"
                )

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def clear(self) -> None:
        """Discard all state in this session-owned ledger."""

        with self._lock, self._connection:
            for table in (
                "entities",
                "contracts",
                "worker_events",
                "assist_receipts",
                "markers",
                "ledger_entries",
            ):
                self._connection.execute(f"DELETE FROM {table}")

    def _put(
        self,
        kind: str,
        entity_id: str,
        owner_id: str,
        value: Any,
        *,
        insert_only: bool = False,
    ) -> None:
        verb = "INSERT" if insert_only else "INSERT OR REPLACE"
        self._connection.execute(
            f"{verb} INTO entities "
            "(kind, entity_id, owner_id, payload, updated_at_ms) "
            "VALUES (?, ?, ?, ?, ?)",
            (kind, entity_id, owner_id, _dump(value), now_ms()),
        )

    def _get(self, kind: str, entity_id: str) -> Any | None:
        row = self._connection.execute(
            "SELECT payload FROM entities WHERE kind = ? AND entity_id = ?",
            (kind, entity_id),
        ).fetchone()
        if row is None:
            return None
        return _decode(_ENTITY_TYPES[kind], _load(row["payload"]))

    def _list(self, kind: str, owner_id: str | None = None) -> tuple[Any, ...]:
        if owner_id is None:
            rows = self._connection.execute(
                "SELECT payload FROM entities WHERE kind = ? ORDER BY updated_at_ms",
                (kind,),
            ).fetchall()
        else:
            rows = self._connection.execute(
                "SELECT payload FROM entities "
                "WHERE kind = ? AND owner_id = ? ORDER BY updated_at_ms",
                (kind, owner_id),
            ).fetchall()
        entity_type = _ENTITY_TYPES[kind]
        return tuple(
            _decode(entity_type, _load(row["payload"])) for row in rows
        )

    def put_turn(self, turn: TurnEnvelope) -> bool:
        with self._lock, self._connection:
            entity_id = turn.receipt_key
            try:
                self._put(
                    "turn",
                    entity_id,
                    turn.owner_id,
                    turn,
                    insert_only=True,
                )
            except sqlite3.IntegrityError:
                existing = self._get("turn", entity_id)
                if existing != turn:
                    raise ValueError(
                        "turn identity collision: "
                        f"{turn.owner_id}/{turn.voice_session_id}/{turn.turn_id}"
                    ) from None
                return False
            return True

    def get_turn(
        self,
        turn_id: str,
        *,
        owner_id: str | None = None,
        voice_session_id: str | None = None,
    ) -> TurnEnvelope | None:
        with self._lock:
            clauses = [
                "kind = 'turn'",
                "json_extract(payload, '$.turn_id') = ?",
            ]
            params: list[Any] = [turn_id]
            if owner_id is not None:
                clauses.append("owner_id = ?")
                params.append(owner_id)
            if voice_session_id is not None:
                clauses.append(
                    "json_extract(payload, '$.voice_session_id') = ?"
                )
                params.append(voice_session_id)
            rows = self._connection.execute(
                "SELECT payload FROM entities WHERE "
                + " AND ".join(clauses)
                + " ORDER BY updated_at_ms DESC LIMIT 2",
                params,
            ).fetchall()
            if not rows:
                return None
            if len(rows) > 1 and (
                owner_id is None or voice_session_id is None
            ):
                raise ValueError("turn lookup is ambiguous")
            return _decode(TurnEnvelope, _load(rows[0]["payload"]))

    def get_turn_by_receipt(
        self, receipt_key: str
    ) -> TurnEnvelope | None:
        with self._lock:
            return self._get("turn", receipt_key)

    def list_turns(
        self, owner_id: str | None = None
    ) -> tuple[TurnEnvelope, ...]:
        with self._lock:
            return self._list("turn", owner_id)

    def put_realtime_context(
        self, owner_id: str, event: ContextEvent
    ) -> bool:
        with self._lock, self._connection:
            existing = self._get("realtime_context", event.event_id)
            if existing is not None:
                if existing != event:
                    raise ValueError(
                        f"realtime context identity collision: {event.event_id}"
                    )
                return False
            self._put(
                "realtime_context",
                event.event_id,
                owner_id,
                event,
                insert_only=True,
            )
            return True

    def get_realtime_context(
        self, event_id: str, *, owner_id: str | None = None
    ) -> ContextEvent | None:
        with self._lock:
            if owner_id is not None:
                row = self._connection.execute(
                    "SELECT payload FROM entities WHERE kind = ? "
                    "AND entity_id = ? AND owner_id = ?",
                    ("realtime_context", event_id, owner_id),
                ).fetchone()
                return (
                    None
                    if row is None
                    else _decode(ContextEvent, _load(row["payload"]))
                )
            return self._get("realtime_context", event_id)

    def list_realtime_context(
        self, owner_id: str | None = None
    ) -> tuple[ContextEvent, ...]:
        with self._lock:
            return self._list("realtime_context", owner_id)

    def submit_coordination(
        self, turn: TurnEnvelope, request_id: str
    ) -> tuple[TaskControlResult, CoordinationJob | None, bool]:
        """Persist one turn, its immediate receipt, and the current owner job."""

        pending = TaskControlResult(
            request_id=request_id,
            disposition="accepted",
            reason_key="coordination.pending",
        )
        with self._lock, self._connection:
            receipt = self._connection.execute(
                "SELECT payload FROM assist_receipts WHERE receipt_key = ?",
                (turn.receipt_key,),
            ).fetchone()
            if receipt is not None:
                stored_turn = self._get("turn", turn.receipt_key)
                if stored_turn != turn:
                    raise ValueError(
                        "turn identity collision: "
                        f"{turn.owner_id}/{turn.voice_session_id}/{turn.turn_id}"
                    )
                existing_result = _decode(
                    TaskControlResult, _load(receipt["payload"])
                )
                current = self._get("coordination_job", turn.owner_id)
                if (
                    current is not None
                    and current.request_id != existing_result.request_id
                ):
                    current = None
                return existing_result, current, False

            stored_turn = self._get("turn", turn.receipt_key)
            if stored_turn is None:
                self._put(
                    "turn",
                    turn.receipt_key,
                    turn.owner_id,
                    turn,
                    insert_only=True,
                )
            elif stored_turn != turn:
                raise ValueError(
                    "turn identity collision: "
                    f"{turn.owner_id}/{turn.voice_session_id}/{turn.turn_id}"
                )

            current = self._get("coordination_job", turn.owner_id)
            timestamp = now_ms()
            turn_receipt_keys = (
                current.turn_receipt_keys + (turn.receipt_key,)
                if current is not None
                and current.state in {"pending", "coordinating"}
                else (turn.receipt_key,)
            )
            job = CoordinationJob(
                owner_id=turn.owner_id,
                request_id=request_id,
                receipt_key=turn.receipt_key,
                turn_id=turn.turn_id,
                voice_session_id=turn.voice_session_id,
                version=(current.version + 1 if current is not None else 1),
                turn_receipt_keys=turn_receipt_keys,
                created_at_ms=timestamp,
                updated_at_ms=timestamp,
            )
            self._put(
                "coordination_job",
                turn.owner_id,
                turn.owner_id,
                job,
            )
            self._connection.execute(
                "INSERT INTO assist_receipts "
                "(receipt_key, owner_id, payload, created_at_ms) "
                "VALUES (?, ?, ?, ?)",
                (
                    turn.receipt_key,
                    turn.owner_id,
                    _dump(pending),
                    timestamp,
                ),
            )
            return pending, job, True

    def get_coordination_job(
        self, owner_id: str
    ) -> CoordinationJob | None:
        with self._lock:
            return self._get("coordination_job", owner_id)

    def list_coordination_jobs(self) -> tuple[CoordinationJob, ...]:
        with self._lock:
            return self._list("coordination_job")

    def claim_coordination_job(
        self, owner_id: str
    ) -> CoordinationJob | None:
        with self._lock, self._connection:
            current = self._get("coordination_job", owner_id)
            if current is None or current.state != "pending":
                return None
            claimed = dataclasses.replace(
                current,
                state="coordinating",
                updated_at_ms=now_ms(),
            )
            self._put(
                "coordination_job",
                owner_id,
                owner_id,
                claimed,
            )
            return claimed

    def finish_coordination_job(
        self,
        job: CoordinationJob,
        result: TaskControlResult,
        *,
        error: str = "",
    ) -> CoordinationJob | None:
        with self._lock, self._connection:
            current = self._get("coordination_job", job.owner_id)
            if (
                current is None
                or current.request_id != job.request_id
                or current.version != job.version
                or current.state != "coordinating"
            ):
                return None
            finished = dataclasses.replace(
                current,
                state="failed" if error else "completed",
                result=result,
                error=error,
                updated_at_ms=now_ms(),
            )
            self._put(
                "coordination_job",
                job.owner_id,
                job.owner_id,
                finished,
            )
            return finished

    def release_coordination_job(self, job: CoordinationJob) -> bool:
        with self._lock, self._connection:
            current = self._get("coordination_job", job.owner_id)
            if (
                current is None
                or current.request_id != job.request_id
                or current.version != job.version
                or current.state != "coordinating"
            ):
                return False
            self._put(
                "coordination_job",
                job.owner_id,
                job.owner_id,
                dataclasses.replace(
                    current,
                    state="pending",
                    updated_at_ms=now_ms(),
                ),
            )
            return True

    def recover_coordination_jobs(self) -> None:
        with self._lock, self._connection:
            for job in self._list("coordination_job"):
                if job.state == "coordinating":
                    self._put(
                        "coordination_job",
                        job.owner_id,
                        job.owner_id,
                        dataclasses.replace(
                            job,
                            state="pending",
                            updated_at_ms=now_ms(),
                        ),
                    )

    def save_coordinator_session(
        self, session: CoordinatorSessionRecord
    ) -> None:
        with self._lock, self._connection:
            existing = self._get("coordinator_session", session.session_id)
            if existing is not None and (
                existing.owner_id != session.owner_id
                or existing.provider_name != session.provider_name
            ):
                raise ValueError(
                    f"coordinator session identity collision: {session.session_id}"
                )
            self._put(
                "coordinator_session",
                session.session_id,
                session.owner_id,
                session,
            )

    def get_coordinator_session(
        self, owner_id: str, provider_name: str
    ) -> CoordinatorSessionRecord | None:
        with self._lock:
            sessions = tuple(
                session
                for session in self._list("coordinator_session", owner_id)
                if session.provider_name == provider_name
            )
            if len(sessions) > 1:
                raise RuntimeError(
                    "multiple coordinator sessions exist for one owner/provider"
                )
            return sessions[0] if sessions else None

    def save_project(self, project: ProjectRecord) -> None:
        with self._lock, self._connection:
            self._put(
                "project", project.project_id, project.owner_id, project
            )

    def get_project(self, project_id: str) -> ProjectRecord | None:
        with self._lock:
            return self._get("project", project_id)

    def list_projects(self, owner_id: str | None = None) -> tuple[ProjectRecord, ...]:
        with self._lock:
            return self._list("project", owner_id)

    def save_task(self, task: TaskRecord) -> None:
        with self._lock, self._connection:
            current = self._get("task", task.task_id)
            if current is not None:
                _validate_task_transition(current, task)
            self._put("task", task.task_id, task.owner_id, task)

    def save_task_once(
        self, operation_id: str, task: TaskRecord
    ) -> TaskRecord:
        marker_key = f"operation:{operation_id}"
        with self._lock, self._connection:
            row = self._connection.execute(
                "SELECT value FROM markers WHERE marker_key = ?",
                (marker_key,),
            ).fetchone()
            if row is not None:
                existing = self._get("task", row["value"])
                if existing is None:
                    raise RuntimeError("operation marker points to a missing task")
                return existing
            current = self._get("task", task.task_id)
            if current is not None:
                _validate_task_transition(current, task)
            self._put("task", task.task_id, task.owner_id, task)
            self._connection.execute(
                "INSERT INTO markers (marker_key, value, created_at_ms) "
                "VALUES (?, ?, ?)",
                (marker_key, task.task_id, now_ms()),
            )
            return task

    def get_task(self, task_id: str) -> TaskRecord | None:
        with self._lock:
            return self._get("task", task_id)

    def list_tasks(self, owner_id: str | None = None) -> tuple[TaskRecord, ...]:
        with self._lock:
            return self._list("task", owner_id)

    def recent_terminal_tasks(
        self, owner_id: str, *, within_ms: int, limit: int
    ) -> tuple[TaskRecord, ...]:
        """Recently-finished tasks still worth showing for follow-up: terminal,
        finished within ``within_ms``, most-recent first, capped at ``limit``.
        Lets the front-brain reference "刚那个…" right after it completes.
        """

        cutoff = now_ms() - within_ms
        with self._lock:
            tasks = self._list("task", owner_id)
            active_side_parents = {
                task.parent_task_id
                for task in tasks
                if task.kind == "side_query" and not task.terminal
            }
            recent = [
                task
                for task in tasks
                if task.kind == "main"
                and task.terminal
                and (
                    task.updated_at_ms >= cutoff
                    or task.task_id in active_side_parents
                )
            ]
        recent.sort(key=lambda task: task.updated_at_ms, reverse=True)
        return tuple(recent[:limit])

    def prune_terminal_tasks(
        self,
        owner_id: str,
        *,
        ttl_ms: int,
        max_terminal: int,
    ) -> tuple[str, ...]:
        """Drop terminal tasks so the ledger does not grow without bound.

        A terminal task is removed when it is older than ``ttl_ms`` OR when the
        owner keeps more than ``max_terminal`` of its kind (oldest first). Hidden
        side queries therefore cannot evict user-visible main tasks. Tasks with a
        still-pending delivery are always kept, so a result is never dropped
        before it is spoken. Returns the removed task ids.
        """

        with self._lock, self._connection:
            tasks = self._list("task", owner_id)
            pending_task_ids = {
                delivery.task_id
                for delivery in self._list("delivery", owner_id)
                if delivery.state in {"pending", "claimed"}
                and delivery.task_id
            }
            protected_parent_ids = {
                task.parent_task_id
                for task in tasks
                if task.kind == "side_query" and not task.terminal
            }
            terminal = [
                task
                for task in tasks
                if task.terminal
                and task.task_id not in pending_task_ids
                and task.task_id not in protected_parent_ids
            ]
            # _list is ordered by updated_at_ms ascending → oldest first.
            now = now_ms()
            removed: list[str] = []
            keep_by_kind: dict[str, list[TaskRecord]] = {
                "main": [],
                "side_query": [],
            }
            for task in terminal:
                if now - task.updated_at_ms > ttl_ms:
                    removed.append(task.task_id)
                else:
                    keep_by_kind[task.kind].append(task)
            for keep in keep_by_kind.values():
                overflow = len(keep) - max_terminal
                if overflow > 0:
                    removed.extend(
                        task.task_id for task in keep[:overflow]
                    )
            for task_id in removed:
                self._connection.execute(
                    "DELETE FROM entities WHERE kind = 'task' AND entity_id = ?",
                    (task_id,),
                )
            return tuple(removed)

    def save_run(self, run: RunRecord, owner_id: str) -> None:
        with self._lock, self._connection:
            current = self._get("run", run.run_id)
            if current is not None:
                _validate_run_transition(current, run)
            self._put("run", run.run_id, owner_id, run)

    def get_run(self, run_id: str) -> RunRecord | None:
        with self._lock:
            return self._get("run", run_id)

    def list_runs(self, owner_id: str | None = None) -> tuple[RunRecord, ...]:
        with self._lock:
            return self._list("run", owner_id)

    def enqueue_provider_command(
        self, command: ProviderCommandRecord
    ) -> ProviderCommandRecord:
        with self._lock, self._connection:
            existing = self._get("provider_command", command.command_id)
            if existing is not None:
                if (
                    existing.owner_id != command.owner_id
                    or existing.task_id != command.task_id
                    or existing.run_id != command.run_id
                    or existing.message != command.message
                ):
                    raise ValueError(
                        f"provider command collision: {command.command_id}"
                    )
                return existing
            self._put(
                "provider_command",
                command.command_id,
                command.owner_id,
                command,
                insert_only=True,
            )
            return command

    def get_provider_command(
        self, command_id: str
    ) -> ProviderCommandRecord | None:
        with self._lock:
            return self._get("provider_command", command_id)

    def list_provider_commands(
        self,
        owner_id: str | None = None,
        *,
        run_id: str | None = None,
    ) -> tuple[ProviderCommandRecord, ...]:
        with self._lock:
            commands = self._list("provider_command", owner_id)
            if run_id is not None:
                commands = tuple(
                    command
                    for command in commands
                    if command.run_id == run_id
                )
            return commands

    def set_provider_command_state(
        self,
        command_id: str,
        state: str,
        *,
        error: str = "",
    ) -> ProviderCommandRecord:
        if state not in {
            "pending",
            "sending",
            "sent",
            "failed",
            "cancelled",
        }:
            raise ValueError("invalid provider command state")
        with self._lock, self._connection:
            current = self._get("provider_command", command_id)
            if current is None:
                raise KeyError(f"unknown provider command: {command_id}")
            updated = dataclasses.replace(
                current,
                state=state,
                attempts=(
                    current.attempts + 1
                    if state == "sending"
                    else current.attempts
                ),
                error=error,
                sent_at_ms=now_ms() if state == "sent" else current.sent_at_ms,
            )
            self._put(
                "provider_command",
                updated.command_id,
                updated.owner_id,
                updated,
            )
            return updated

    def reset_sending_provider_commands(self) -> None:
        with self._lock, self._connection:
            for command in self._list("provider_command"):
                if command.state == "sending":
                    self._put(
                        "provider_command",
                        command.command_id,
                        command.owner_id,
                        dataclasses.replace(command, state="pending"),
                    )

    def fail_provider_commands_for_run(
        self, run_id: str, error: str
    ) -> None:
        with self._lock, self._connection:
            for command in self._list("provider_command"):
                if (
                    command.run_id == run_id
                    and command.state in {"pending", "sending"}
                ):
                    self._put(
                        "provider_command",
                        command.command_id,
                        command.owner_id,
                        dataclasses.replace(
                            command, state="failed", error=error
                        ),
                    )

    def cancel_provider_commands_for_interaction(
        self, interaction_id: str
    ) -> None:
        with self._lock, self._connection:
            for command in self._list("provider_command"):
                if (
                    command.message.interaction_id == interaction_id
                    and command.state in {"pending", "sending"}
                ):
                    self._put(
                        "provider_command",
                        command.command_id,
                        command.owner_id,
                        dataclasses.replace(
                            command,
                            state="cancelled",
                            error="interaction invalidated",
                        ),
                    )

    def create_task(
        self,
        project: ProjectRecord,
        task: TaskRecord,
        contract: SupervisionContractRevision,
    ) -> None:
        if contract.task_id != task.task_id or contract.revision != 1:
            raise ValueError("new task must start with contract revision 1")
        with self._lock, self._connection:
            if self._get("task", task.task_id) is not None:
                raise ValueError(f"task already exists: {task.task_id}")
            existing_project = self._get("project", project.project_id)
            if existing_project is None:
                self._put(
                    "project",
                    project.project_id,
                    project.owner_id,
                    project,
                    insert_only=True,
                )
            elif existing_project.owner_id != task.owner_id:
                raise ValueError("project owner does not match task owner")
            self._put(
                "task", task.task_id, task.owner_id, task, insert_only=True
            )
            self._connection.execute(
                "INSERT INTO contracts "
                "(task_id, revision, payload, created_at_ms) VALUES (?, ?, ?, ?)",
                (
                    contract.task_id,
                    contract.revision,
                    _dump(contract),
                    contract.created_at_ms,
                ),
            )
            self._connection.execute(
                "INSERT INTO ledger_entries "
                "(kind, entity_id, contract_revision, created_at_ms) "
                "VALUES ('contract', ?, ?, ?)",
                (task.task_id, contract.revision, contract.created_at_ms),
            )

    def get_contract(
        self, task_id: str, revision: int | None = None
    ) -> SupervisionContractRevision | None:
        with self._lock:
            if revision is None:
                row = self._connection.execute(
                    "SELECT payload FROM contracts WHERE task_id = ? "
                    "ORDER BY revision DESC LIMIT 1",
                    (task_id,),
                ).fetchone()
            else:
                row = self._connection.execute(
                    "SELECT payload FROM contracts "
                    "WHERE task_id = ? AND revision = ?",
                    (task_id, revision),
                ).fetchone()
            return (
                _decode(
                    SupervisionContractRevision, _load(row["payload"])
                )
                if row
                else None
            )

    def list_contracts(
        self, task_ids: tuple[str, ...] | None = None
    ) -> tuple[SupervisionContractRevision, ...]:
        with self._lock:
            if task_ids is None:
                rows = self._connection.execute(
                    "SELECT payload FROM contracts ORDER BY created_at_ms"
                ).fetchall()
            elif not task_ids:
                return ()
            else:
                placeholders = ",".join("?" for _ in task_ids)
                rows = self._connection.execute(
                    "SELECT payload FROM contracts "
                    f"WHERE task_id IN ({placeholders}) ORDER BY created_at_ms",
                    task_ids,
                ).fetchall()
            return tuple(
                _decode(
                    SupervisionContractRevision, _load(row["payload"])
                )
                for row in rows
            )

    def revise_contract(
        self,
        task_id: str,
        body: ContractBody,
        source_turn_ids: tuple[str, ...],
    ) -> SupervisionContractRevision:
        with self._lock, self._connection:
            task = self._get("task", task_id)
            if task is None:
                raise KeyError(f"unknown task: {task_id}")
            current = self.get_contract(task_id, task.contract_revision)
            if (
                current is not None
                and current.body == body
                and current.source_turn_ids == source_turn_ids
            ):
                return current
            revision = task.contract_revision + 1
            contract = SupervisionContractRevision(
                task_id=task_id,
                revision=revision,
                body=body,
                source_turn_ids=source_turn_ids,
            )
            self._connection.execute(
                "INSERT INTO contracts "
                "(task_id, revision, payload, created_at_ms) VALUES (?, ?, ?, ?)",
                (task_id, revision, _dump(contract), contract.created_at_ms),
            )
            self._put(
                "task",
                task.task_id,
                task.owner_id,
                dataclasses.replace(
                    task, contract_revision=revision, updated_at_ms=now_ms()
                ),
            )
            self._connection.execute(
                "INSERT INTO ledger_entries "
                "(kind, entity_id, contract_revision, created_at_ms) "
                "VALUES ('contract', ?, ?, ?)",
                (task_id, revision, contract.created_at_ms),
            )
            return contract

    def save_artifact(self, artifact: ArtifactEvidence) -> None:
        task = self.get_task(artifact.task_id)
        if task is None or task.owner_id != artifact.owner_id:
            raise ValueError("artifact does not belong to the task owner")
        with self._lock, self._connection:
            existing = self._get("artifact", artifact.ref)
            if existing is not None:
                if (
                    existing.owner_id != artifact.owner_id
                    or existing.task_id != artifact.task_id
                    or existing.facts != artifact.facts
                ):
                    raise ValueError(
                        f"artifact evidence is immutable: {artifact.ref}"
                    )
                return
            self._put(
                "artifact",
                artifact.ref,
                artifact.owner_id,
                artifact,
                insert_only=True,
            )

    def get_artifact(self, ref: str) -> ArtifactEvidence | None:
        with self._lock:
            return self._get("artifact", ref)

    def list_artifacts(
        self, owner_id: str | None = None
    ) -> tuple[ArtifactEvidence, ...]:
        with self._lock:
            return self._list("artifact", owner_id)

    def append_activity_aggregate(
        self,
        event: WorkerEvent,
        *,
        contract_revision: int,
        window_ms: int,
    ) -> ActivityAggregate:
        if window_ms < 1:
            raise ValueError("aggregate window must be positive")
        payload = event.payload
        if not isinstance(payload, UpdatePayload) or payload.kind != "activity":
            raise TypeError("only activity updates can be aggregated")
        with self._lock, self._connection:
            row = self._connection.execute(
                "SELECT payload FROM entities "
                "WHERE kind = 'aggregate' AND owner_id = ? "
                "AND json_extract(payload, '$.task_id') = ? "
                "AND json_extract(payload, '$.contract_revision') = ? "
                "AND json_extract(payload, '$.state') = 'open' "
                "ORDER BY updated_at_ms DESC LIMIT 1",
                (event.owner_id, event.task_id, contract_revision),
            ).fetchone()
            if row is None:
                aggregate = ActivityAggregate(
                    aggregate_id=new_id("aggregate"),
                    owner_id=event.owner_id,
                    task_id=event.task_id,
                    contract_revision=contract_revision,
                    event_ids=(event.event_id,),
                    summaries=(payload.summary,),
                    event_count=1,
                    due_at_ms=now_ms() + window_ms,
                )
            else:
                current = _decode(
                    ActivityAggregate, _load(row["payload"])
                )
                if event.event_id in current.event_ids:
                    return current
                aggregate = dataclasses.replace(
                    current,
                    event_ids=(current.event_ids + (event.event_id,))[-32:],
                    summaries=(current.summaries + (payload.summary,))[-32:],
                    event_count=current.event_count + 1,
                )
            self._put(
                "aggregate",
                aggregate.aggregate_id,
                aggregate.owner_id,
                aggregate,
            )
            return aggregate

    def list_activity_aggregates(
        self,
        owner_id: str | None = None,
        *,
        task_id: str | None = None,
        state: str | None = None,
    ) -> tuple[ActivityAggregate, ...]:
        with self._lock:
            aggregates = self._list("aggregate", owner_id)
            if task_id is not None:
                aggregates = tuple(
                    item for item in aggregates if item.task_id == task_id
                )
            if state is not None:
                aggregates = tuple(
                    item for item in aggregates if item.state == state
                )
            return aggregates

    def flush_activity_aggregate(
        self, aggregate_id: str
    ) -> ActivityAggregate | None:
        with self._lock, self._connection:
            current = self._get("aggregate", aggregate_id)
            if current is None:
                raise KeyError(f"unknown activity aggregate: {aggregate_id}")
            if current.state != "open":
                return None
            flushed = dataclasses.replace(
                current, state="flushed", flushed_at_ms=now_ms()
            )
            self._put(
                "aggregate",
                flushed.aggregate_id,
                flushed.owner_id,
                flushed,
            )
            return flushed

    def append_event(self, event: WorkerEvent) -> bool:
        """Persist an event once and fence stale generations and sequences."""

        with self._lock, self._connection:
            existing = self._connection.execute(
                "SELECT payload FROM worker_events WHERE event_id = ?",
                (event.event_id,),
            ).fetchone()
            if existing is not None:
                if _decode(WorkerEvent, _load(existing["payload"])) != event:
                    raise ValueError(f"event_id collision: {event.event_id}")
                return False
            task = self._get("task", event.task_id)
            run = self._get("run", event.run_id)
            if (
                task is None
                or run is None
                or task.owner_id != event.owner_id
                or task.project_id != event.project_id
                or task.generation != event.generation
                or task.active_run_id != event.run_id
                or run.task_id != event.task_id
                or run.generation != event.generation
            ):
                return False
            if event.seq <= run.last_event_seq:
                raise ValueError(
                    f"non-monotonic event sequence for run {event.run_id}"
                )
            evaluated_revision = task.contract_revision
            try:
                self._connection.execute(
                    "INSERT INTO worker_events "
                    "(event_id, task_id, run_id, generation, seq, "
                    "evaluated_contract_revision, payload, processed, created_at_ms) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?)",
                    (
                        event.event_id,
                        event.task_id,
                        event.run_id,
                        event.generation,
                        event.seq,
                        evaluated_revision,
                        _dump(event),
                        event.created_at_ms,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError(
                    f"event sequence collision for run {event.run_id}"
                ) from exc
            self._put(
                "run",
                run.run_id,
                task.owner_id,
                dataclasses.replace(run, last_event_seq=event.seq),
            )
            self._connection.execute(
                "INSERT INTO ledger_entries "
                "(kind, entity_id, contract_revision, created_at_ms) "
                "VALUES ('worker_event', ?, ?, ?)",
                (event.event_id, evaluated_revision, event.created_at_ms),
            )
            return True

    def event_contract_revision(self, event_id: str) -> int:
        with self._lock:
            row = self._connection.execute(
                "SELECT evaluated_contract_revision FROM worker_events "
                "WHERE event_id = ?",
                (event_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown worker event: {event_id}")
            return int(row["evaluated_contract_revision"])

    def mark_event_processed(self, event_id: str) -> None:
        with self._lock, self._connection:
            cursor = self._connection.execute(
                "UPDATE worker_events SET processed = 1 WHERE event_id = ?",
                (event_id,),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"unknown worker event: {event_id}")

    def list_events(
        self,
        *,
        task_id: str | None = None,
        processed: bool | None = None,
    ) -> tuple[WorkerEvent, ...]:
        clauses: list[str] = []
        params: list[Any] = []
        if task_id is not None:
            clauses.append("events.task_id = ?")
            params.append(task_id)
        if processed is not None:
            clauses.append("events.processed = ?")
            params.append(int(processed))
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock:
            rows = self._connection.execute(
                "SELECT events.payload FROM worker_events AS events "
                "LEFT JOIN ledger_entries AS entries "
                "ON entries.kind = 'worker_event' "
                "AND entries.entity_id = events.event_id"
                + where
                + " ORDER BY COALESCE(entries.seq, events.rowid)",
                params,
            ).fetchall()
            return tuple(
                _decode(WorkerEvent, _load(row["payload"])) for row in rows
            )

    def save_interaction(self, interaction: PendingInteraction) -> None:
        with self._lock, self._connection:
            self._put(
                "interaction",
                interaction.interaction_id,
                interaction.owner_id,
                interaction,
            )

    def get_interaction(
        self, interaction_id: str
    ) -> PendingInteraction | None:
        with self._lock:
            return self._get("interaction", interaction_id)

    def list_interactions(
        self, owner_id: str | None = None
    ) -> tuple[PendingInteraction, ...]:
        with self._lock:
            return self._list("interaction", owner_id)

    def begin_interaction_resolution(
        self, interaction_id: str, response: str, decision: str | None
    ) -> PendingInteraction | None:
        with self._lock, self._connection:
            current = self._get("interaction", interaction_id)
            if current is None:
                raise KeyError(f"unknown interaction: {interaction_id}")
            if current.state != "pending":
                return None
            normalized_decision = decision
            if current.kind == "choice":
                # Choices are UI hints, not a restriction on natural-language
                # task_send replies. Only validate a provider-supplied discrete
                # decision; otherwise preserve the user's raw text for the Worker.
                if decision is not None and decision not in current.choices:
                    raise ValueError(
                        "choice decision must be one of the interaction choices"
                    )
                if decision is None and response in current.choices:
                    normalized_decision = response
            if current.kind == "permission" and decision not in {
                "allow",
                "allow_once",
                "allow_session",
                "approve",
                "deny",
            }:
                raise ValueError(
                    "permission decision must explicitly allow or deny"
                )
            resolving = dataclasses.replace(
                current,
                state="resolving",
                response=response,
                decision=normalized_decision,
            )
            cursor = self._connection.execute(
                "UPDATE entities SET payload = ?, updated_at_ms = ? "
                "WHERE kind = 'interaction' AND entity_id = ? "
                "AND json_extract(payload, '$.state') = 'pending'",
                (_dump(resolving), now_ms(), interaction_id),
            )
            return resolving if cursor.rowcount == 1 else None

    def finish_interaction_resolution(
        self, interaction_id: str, *, success: bool
    ) -> PendingInteraction:
        with self._lock, self._connection:
            current = self._get("interaction", interaction_id)
            if current is None:
                raise KeyError(f"unknown interaction: {interaction_id}")
            if current.state != "resolving":
                raise ValueError("interaction is not being resolved")
            updated = dataclasses.replace(
                current,
                state="resolved" if success else "pending",
                resolved_at_ms=now_ms() if success else None,
                response=current.response if success else "",
                decision=current.decision if success else None,
            )
            self._put(
                "interaction",
                updated.interaction_id,
                updated.owner_id,
                updated,
            )
            return updated

    def find_interaction_by_fingerprint(
        self, fingerprint: str
    ) -> PendingInteraction | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT payload FROM entities "
                "WHERE kind = 'interaction' "
                "AND json_extract(payload, '$.fingerprint') = ? "
                "ORDER BY updated_at_ms DESC LIMIT 1",
                (fingerprint,),
            ).fetchone()
            return (
                _decode(PendingInteraction, _load(row["payload"]))
                if row
                else None
            )

    def expire_due_interactions(
        self, timestamp_ms: int | None = None
    ) -> tuple[PendingInteraction, ...]:
        timestamp_ms = now_ms() if timestamp_ms is None else timestamp_ms
        expired: list[PendingInteraction] = []
        with self._lock, self._connection:
            rows = self._connection.execute(
                "SELECT payload FROM entities "
                "WHERE kind = 'interaction' "
                "AND json_extract(payload, '$.state') = 'pending' "
                "AND json_extract(payload, '$.expires_at_ms') IS NOT NULL "
                "AND json_extract(payload, '$.expires_at_ms') <= ?",
                (timestamp_ms,),
            ).fetchall()
            for row in rows:
                current = _decode(
                    PendingInteraction, _load(row["payload"])
                )
                updated = dataclasses.replace(
                    current,
                    state="expired",
                    decision="deny" if current.kind == "permission" else None,
                    response=(
                        "Permission expired"
                        if current.kind == "permission"
                        else current.response
                    ),
                    resolved_at_ms=timestamp_ms,
                )
                self._put(
                    "interaction",
                    updated.interaction_id,
                    updated.owner_id,
                    updated,
                )
                expired.append(updated)
        return tuple(expired)

    def expire_pending_permissions(self) -> tuple[PendingInteraction, ...]:
        expired: list[PendingInteraction] = []
        timestamp = now_ms()
        with self._lock, self._connection:
            rows = self._connection.execute(
                "SELECT payload FROM entities "
                "WHERE kind = 'interaction' "
                "AND json_extract(payload, '$.state') IN ('pending', 'resolving') "
                "AND json_extract(payload, '$.kind') = 'permission'"
            ).fetchall()
            for row in rows:
                current = _decode(
                    PendingInteraction, _load(row["payload"])
                )
                updated = dataclasses.replace(
                    current,
                    state="expired",
                    response="Permission invalidated by Gateway restart",
                    decision="deny",
                    resolved_at_ms=timestamp,
                )
                self._put(
                    "interaction",
                    updated.interaction_id,
                    updated.owner_id,
                    updated,
                )
                expired.append(updated)
        return tuple(expired)

    def create_delivery_once(
        self, marker_key: str, delivery: DeliveryRecord
    ) -> DeliveryRecord | None:
        with self._lock, self._connection:
            row = self._connection.execute(
                "SELECT value FROM markers WHERE marker_key = ?",
                (marker_key,),
            ).fetchone()
            if row is not None:
                return self._get("delivery", row["value"])
            self._put(
                "delivery",
                delivery.delivery_id,
                delivery.owner_id,
                delivery,
                insert_only=True,
            )
            self._connection.execute(
                "INSERT INTO markers (marker_key, value, created_at_ms) "
                "VALUES (?, ?, ?)",
                (marker_key, delivery.delivery_id, now_ms()),
            )
            return delivery

    def get_delivery(self, delivery_id: str) -> DeliveryRecord | None:
        with self._lock:
            return self._get("delivery", delivery_id)

    def list_deliveries(
        self, owner_id: str | None = None
    ) -> tuple[DeliveryRecord, ...]:
        with self._lock:
            return self._list("delivery", owner_id)

    def claim_delivery(
        self, owner_id: str, *, lease_ms: int
    ) -> DeliveryRecord | None:
        if lease_ms < 1:
            raise ValueError("delivery lease must be positive")
        self.requeue_expired_delivery_claims(owner_id)
        with self._lock, self._connection:
            timestamp = now_ms()
            row = self._connection.execute(
                "SELECT entity_id, payload FROM entities "
                "WHERE kind = 'delivery' AND owner_id = ? "
                "AND json_extract(payload, '$.state') = 'pending' "
                "ORDER BY CASE json_extract(payload, '$.timing') "
                "WHEN 'interrupt' THEN 0 ELSE 1 END, updated_at_ms LIMIT 1",
                (owner_id,),
            ).fetchone()
            if row is None:
                return None
            current = _decode(DeliveryRecord, _load(row["payload"]))
            claimed = dataclasses.replace(
                current,
                state="claimed",
                claim_token=new_id("claim"),
                claim_expires_at_ms=timestamp + lease_ms,
                attempts=current.attempts + 1,
            )
            cursor = self._connection.execute(
                "UPDATE entities SET payload = ?, updated_at_ms = ? "
                "WHERE kind = 'delivery' AND entity_id = ? "
                "AND json_extract(payload, '$.state') = 'pending'",
                (_dump(claimed), now_ms(), claimed.delivery_id),
            )
            return claimed if cursor.rowcount == 1 else None

    def requeue_expired_delivery_claims(self, owner_id: str) -> int:
        """Make abandoned claims visible before a consumer peeks at pending work."""

        with self._lock, self._connection:
            timestamp = now_ms()
            expired_rows = self._connection.execute(
                "SELECT payload FROM entities "
                "WHERE kind = 'delivery' AND owner_id = ? "
                "AND json_extract(payload, '$.state') = 'claimed' "
                "AND json_extract(payload, '$.claim_expires_at_ms') <= ?",
                (owner_id, timestamp),
            ).fetchall()
            for expired_row in expired_rows:
                expired = _decode(
                    DeliveryRecord, _load(expired_row["payload"])
                )
                self._put(
                    "delivery",
                    expired.delivery_id,
                    expired.owner_id,
                    dataclasses.replace(
                        expired,
                        state="pending",
                        claim_token="",
                        claim_expires_at_ms=None,
                    ),
                )
            return len(expired_rows)

    def finish_delivery(
        self, delivery_id: str, claim_token: str, *, delivered: bool
    ) -> DeliveryRecord:
        with self._lock, self._connection:
            current = self._get("delivery", delivery_id)
            if current is None:
                raise KeyError(f"unknown delivery: {delivery_id}")
            if current.state != "claimed" or current.claim_token != claim_token:
                raise ValueError("delivery claim does not match")
            updated = dataclasses.replace(
                current,
                state="delivered" if delivered else "pending",
                claim_token="",
                claim_expires_at_ms=None,
                delivered_at_ms=now_ms() if delivered else None,
            )
            self._put(
                "delivery", updated.delivery_id, updated.owner_id, updated
            )
            return updated

    def finish_expired_delivery(
        self,
        delivery_id: str,
        *,
        claim_attempt: int,
        delivered: bool,
    ) -> DeliveryRecord:
        """Commit an output whose lease expired while it waited for a model unit."""

        if claim_attempt < 1:
            raise ValueError("delivery claim attempt must be positive")
        with self._lock, self._connection:
            current = self._get("delivery", delivery_id)
            if current is None:
                raise KeyError(f"unknown delivery: {delivery_id}")
            if (
                current.state != "pending"
                or current.claim_token
                or current.attempts != claim_attempt
            ):
                raise ValueError("expired delivery claim does not match")
            updated = dataclasses.replace(
                current,
                state="delivered" if delivered else "pending",
                claim_token="",
                claim_expires_at_ms=None,
                delivered_at_ms=now_ms() if delivered else None,
            )
            self._put(
                "delivery", updated.delivery_id, updated.owner_id, updated
            )
            return updated

    def cancel_interaction_deliveries(self, interaction_id: str) -> None:
        with self._lock, self._connection:
            rows = self._connection.execute(
                "SELECT payload FROM entities "
                "WHERE kind = 'delivery' "
                "AND json_extract(payload, '$.interaction_id') = ? "
                "AND json_extract(payload, '$.state') IN ('pending', 'claimed')",
                (interaction_id,),
            ).fetchall()
            for row in rows:
                current = _decode(DeliveryRecord, _load(row["payload"]))
                self._put(
                    "delivery",
                    current.delivery_id,
                    current.owner_id,
                    dataclasses.replace(
                        current,
                        state="cancelled",
                        claim_token="",
                        claim_expires_at_ms=None,
                    ),
                )

    def get_receipt(self, receipt_key: str) -> TaskControlResult | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT payload FROM assist_receipts WHERE receipt_key = ?",
                (receipt_key,),
            ).fetchone()
            if row is None:
                return None
            return _decode(TaskControlResult, _load(row["payload"]))

    def save_receipt(
        self, receipt_key: str, owner_id: str, result: TaskControlResult
    ) -> TaskControlResult:
        with self._lock, self._connection:
            try:
                self._connection.execute(
                    "INSERT INTO assist_receipts "
                    "(receipt_key, owner_id, payload, created_at_ms) "
                    "VALUES (?, ?, ?, ?)",
                    (receipt_key, owner_id, _dump(result), now_ms()),
                )
                return result
            except sqlite3.IntegrityError:
                existing = self.get_receipt(receipt_key)
                if existing is None:
                    raise
                return existing


def _validate_task_transition(
    current: TaskRecord, updated: TaskRecord
) -> None:
    allowed = {
        "queued": {"queued", "running", "cancelled", "failed"},
        "running": {
            "running",
            "finalizing",
            "cancelling",
            "completed",
            "partial",
            "failed",
            "cancelled",
        },
        "finalizing": {
            "finalizing",
            "completed",
            "partial",
            "failed",
            "cancelling",
            "cancelled",
        },
        "cancelling": {
            "cancelling",
            "completed",
            "partial",
            "failed",
            "cancelled",
        },
        "completed": {"completed"},
        "partial": {"partial"},
        "failed": {"failed"},
        "cancelled": {"cancelled"},
    }
    if updated.status not in allowed[current.status]:
        raise ValueError(
            f"invalid task transition: {current.status} -> {updated.status}"
        )
    if updated.generation < current.generation:
        raise ValueError("task generation cannot move backwards")
    if updated.generation > current.generation + 1:
        raise ValueError("task generation cannot skip")
    if (
        current.kind,
        current.parent_task_id,
        current.parent_run_id,
        current.lineage_id,
        current.project_id,
        current.owner_id,
        current.provider_name,
    ) != (
        updated.kind,
        updated.parent_task_id,
        updated.parent_run_id,
        updated.lineage_id,
        updated.project_id,
        updated.owner_id,
        updated.provider_name,
    ):
        raise ValueError("task identity and parent metadata are immutable")
    if updated.status in {"queued", "completed", "partial", "failed", "cancelled"}:
        if updated.active_run_id is not None:
            raise ValueError(f"{updated.status} task cannot have an active run")
    elif updated.active_run_id is None:
        raise ValueError(f"{updated.status} task requires an active run")


def _validate_run_transition(current: RunRecord, updated: RunRecord) -> None:
    allowed = {
        "queued": {"queued", "running", "failed", "cancelled"},
        "running": {
            "running",
            "waiting",
            "completed",
            "partial",
            "failed",
            "cancelled",
        },
        "waiting": {
            "waiting",
            "running",
            "completed",
            "partial",
            "failed",
            "cancelled",
        },
        "completed": {"completed"},
        "partial": {"partial"},
        "failed": {"failed"},
        "cancelled": {"cancelled"},
    }
    if current.task_id != updated.task_id:
        raise ValueError("run cannot move to another task")
    if current.generation != updated.generation:
        raise ValueError("run generation is immutable")
    if updated.status not in allowed[current.status]:
        raise ValueError(
            f"invalid run transition: {current.status} -> {updated.status}"
        )
    if updated.last_event_seq < current.last_event_seq:
        raise ValueError("run event sequence cannot move backwards")


class SupervisionEngine:
    """Deterministic policy evaluation; no model calls and no side effects."""

    def __init__(self, ledger: TaskLedger) -> None:
        self.ledger = ledger

    def authorize(
        self,
        contract: SupervisionContractRevision,
        capabilities: BackendCapabilities,
        action: str,
        *,
        side_effect: bool = True,
        high_risk: bool = False,
    ) -> AuthorizationDecision:
        policy = contract.body.authority_policy
        if action in policy.deny:
            return AuthorizationDecision("deny", action, "denied_by_contract")
        if action in policy.allow:
            return AuthorizationDecision("allow", action, "allowed_by_contract")
        needs_permission = action in policy.require_permission
        if not needs_permission:
            if high_risk or not side_effect:
                return AuthorizationDecision("deny", action, "unknown_action")
            needs_permission = True
        if needs_permission and capabilities.authority_enforcement == "none":
            return AuthorizationDecision(
                "deny", action, "backend_cannot_enforce_permission"
            )
        return AuthorizationDecision(
            "require_permission",
            action,
            (
                "permission_required"
                if action in policy.require_permission
                else "unknown_side_effect"
            ),
        )

    def arbitrate(
        self,
        event: WorkerEvent,
        contract: SupervisionContractRevision,
        capabilities: BackendCapabilities,
    ) -> ArbitrationDecision:
        payload = event.payload
        delivery = contract.body.delivery_policy
        if isinstance(payload, UpdatePayload):
            if payload.severity == "high_risk":
                timing = (
                    delivery.high_risk
                    if capabilities.trusted_risk_signals
                    else "safe_pause"
                )
                return ArbitrationDecision(
                    timing,
                    reason=(
                        "verified_high_risk_update"
                        if capabilities.trusted_risk_signals
                        else "unverified_high_risk_update"
                    ),
                )
            if payload.kind == "activity":
                return ArbitrationDecision("hold", reason="ordinary_activity")
            rule = next(
                (
                    item
                    for item in contract.body.notify_policy.milestones
                    if item.key == payload.milestone
                ),
                None,
            )
            if rule is None:
                return ArbitrationDecision("hold", reason="unsubscribed_milestone")
            evidence = [
                self.ledger.get_artifact(ref) for ref in payload.evidence_refs
            ]
            if any(
                item is None
                or item.owner_id != event.owner_id
                or item.task_id != event.task_id
                for item in evidence
            ):
                return ArbitrationDecision(
                    "hold", accepted=False, reason="invalid_milestone_evidence"
                )
            if rule.condition is not None and not any(
                _matches(item.facts, rule.condition)
                for item in evidence
                if item is not None
            ):
                return ArbitrationDecision(
                    "hold", accepted=False, reason="milestone_condition_not_met"
                )
            return ArbitrationDecision(
                delivery.milestone, reason=f"milestone:{rule.key}"
            )

        if isinstance(payload, InteractionPayload):
            if not capabilities.interactions:
                return ArbitrationDecision(
                    "hold",
                    accepted=False,
                    reason="backend_did_not_declare_interactions",
                )
            if payload.kind == "permission":
                authorization = self.authorize(
                    contract, capabilities, payload.action_key or ""
                )
                if authorization.outcome == "require_permission":
                    return ArbitrationDecision(
                        delivery.interaction,
                        interaction_route="user",
                        reason=authorization.reason,
                    )
                return ArbitrationDecision(
                    "hold",
                    interaction_route="coordinator",
                    reason=authorization.reason,
                )
            ask = contract.body.ask_policy
            if payload.reason_key in ask.must_ask:
                route = "user"
            elif payload.reason_key in ask.delegate:
                route = "coordinator"
            elif ask.when_exhausted == "ask":
                route = "user"
            else:
                route = "coordinator"
            return ArbitrationDecision(
                delivery.interaction if route == "user" else "hold",
                interaction_route=route,
                reason=f"interaction_{route}",
            )

        if isinstance(payload, DonePayload):
            return ArbitrationDecision(delivery.final, reason="terminal_result")
        raise TypeError(f"unknown worker payload: {type(payload)!r}")


def _matches(facts: dict[str, Any], condition: MilestoneCondition) -> bool:
    current: Any = facts
    for part in condition.field.split("."):
        if not isinstance(current, dict) or part not in current:
            return False
        current = current[part]
    if condition.op == "exists":
        return current is not None
    if condition.op == "eq":
        return current == condition.value
    if condition.op == "gte":
        try:
            return current >= condition.value
        except TypeError:
            return False
    if condition.op == "lte":
        try:
            return current <= condition.value
        except TypeError:
            return False
    return False


def effective_scope(requested: str, supported: str) -> str:
    order = {"action": 0, "branch": 1, "run": 2}
    if requested not in order or supported not in order:
        raise ValueError("invalid blocking scope")
    return max((requested, supported), key=order.__getitem__)


class InteractionBroker:
    def __init__(self, ledger: TaskLedger) -> None:
        self.ledger = ledger

    def from_worker(
        self,
        event: WorkerEvent,
        capabilities: BackendCapabilities,
        *,
        audience: str,
    ) -> PendingInteraction:
        payload = event.payload
        if not isinstance(payload, InteractionPayload):
            raise TypeError("worker interaction requires InteractionPayload")
        fingerprint = stable_key("worker", event.event_id)
        existing = self.ledger.find_interaction_by_fingerprint(fingerprint)
        if existing is not None:
            return existing
        interaction = PendingInteraction(
            interaction_id=new_id("interaction"),
            owner_id=event.owner_id,
            target_id=event.task_id,
            task_id=event.task_id,
            run_id=event.run_id,
            source_event_id=event.event_id,
            source="worker",
            audience=audience,
            kind=payload.kind,
            prompt=payload.prompt,
            reason_key=payload.reason_key,
            action_key=payload.action_key,
            choices=payload.choices,
            requested_scope=payload.blocking_scope,
            effective_scope=effective_scope(
                payload.blocking_scope, capabilities.blocking_granularity
            ),
            fingerprint=fingerprint,
        )
        self.ledger.save_interaction(interaction)
        return interaction

    def admit_question(
        self,
        proposal: QuestionProposal,
        *,
        owner_id: str,
        coordination_request_id: str,
        contract: SupervisionContractRevision | None,
    ) -> tuple[
        PendingInteraction | None,
        Literal["admitted", "duplicate", "defaulted", "rejected"],
    ]:
        ask = contract.body.ask_policy if contract else AskPolicy()
        if len(proposal.questions) > ask.max_questions_per_round:
            raise ValueError("Coordinator exceeded max_questions_per_round")
        if not ask.bundle_questions and len(proposal.questions) > 1:
            raise ValueError("ask_policy does not allow bundled questions")
        revision = contract.revision if contract else 0
        fingerprint = stable_key(
            "coordinator",
            proposal.target_id,
            str(revision),
            proposal.reason_key,
        )
        existing = self.ledger.find_interaction_by_fingerprint(fingerprint)
        if existing is not None:
            if existing.state == "pending":
                return existing, "admitted"
            return None, "duplicate"
        mandatory = proposal.reason_key in ask.must_ask
        prior_rounds = sum(
            item.source == "coordinator"
            and item.coordination_request_id == coordination_request_id
            for item in self.ledger.list_interactions(owner_id)
        )
        if not mandatory and prior_rounds >= ask.max_clarification_rounds:
            if (
                ask.when_exhausted == "conservative"
                and proposal.default_if_unanswered is not None
            ):
                return None, "defaulted"
            return None, "rejected"
        prompt = (
            proposal.questions[0]
            if len(proposal.questions) == 1
            else "\n".join(
                f"{index}. {question}"
                for index, question in enumerate(proposal.questions, 1)
            )
        )
        task = self.ledger.get_task(proposal.target_id)
        interaction = PendingInteraction(
            interaction_id=new_id("interaction"),
            owner_id=owner_id,
            target_id=proposal.target_id,
            task_id=task.task_id if task else None,
            run_id=task.active_run_id if task else None,
            source_event_id=None,
            source="coordinator",
            audience="user",
            kind="question",
            prompt=prompt,
            reason_key=proposal.reason_key,
            requested_scope=proposal.blocking_scope,
            effective_scope=proposal.blocking_scope,
            fingerprint=fingerprint,
            coordination_request_id=coordination_request_id,
        )
        self.ledger.save_interaction(interaction)
        return interaction, "admitted"


class DeliveryBroker:
    def __init__(
        self, ledger: TaskLedger, *, claim_lease_ms: int = 30_000
    ) -> None:
        if claim_lease_ms < 1:
            raise ValueError("claim_lease_ms must be positive")
        self.ledger = ledger
        self.claim_lease_ms = claim_lease_ms

    def enqueue(
        self,
        *,
        owner_id: str,
        task_id: str | None,
        event_id: str | None,
        interaction_id: str | None,
        timing: str,
        topic: str,
        speech_hint: str,
        dedupe_key: str,
        status: str = "",
    ) -> DeliveryRecord | None:
        if timing == "hold":
            return None
        if timing not in {"safe_pause", "interrupt"}:
            raise ValueError("invalid deliverable timing")
        delivery = DeliveryRecord(
            delivery_id=new_id("delivery"),
            owner_id=owner_id,
            task_id=task_id,
            event_id=event_id,
            interaction_id=interaction_id,
            timing=timing,
            topic=topic,
            speech_hint=speech_hint,
            status=status,
        )
        return self.ledger.create_delivery_once(dedupe_key, delivery)

    def claim(self, owner_id: str) -> DeliveryRecord | None:
        return self.ledger.claim_delivery(
            owner_id, lease_ms=self.claim_lease_ms
        )

    def acknowledge(
        self, delivery_id: str, claim_token: str, *, delivered: bool = True
    ) -> DeliveryRecord:
        return self.ledger.finish_delivery(
            delivery_id, claim_token, delivered=delivered
        )

    def acknowledge_consumed(
        self,
        delivery_id: str,
        claim_token: str,
        *,
        claim_attempt: int,
        delivered: bool = True,
    ) -> DeliveryRecord:
        try:
            return self.acknowledge(
                delivery_id,
                claim_token,
                delivered=delivered,
            )
        except ValueError:
            return self.ledger.finish_expired_delivery(
                delivery_id,
                claim_attempt=claim_attempt,
                delivered=delivered,
            )
