"""Native front-brain contract for task_start, task_send, and task_resolve.

The module defines schemas, parses calls, dispatches Gateway operations, formats
tool responses, and renders the live task slate independently of the transport.
"""
from __future__ import annotations

from typing import Any, Mapping

from mcpmft.tool_protocol import (
    LEAN_TASK_TOOL_SCHEMAS,
    normalize_frontbrain_task_name,
)
from .coordination import (
    TASK_LANES,
    TASK_RESOLVE_ACTIONS,
    TaskControlResult,
    TaskLane,
    TaskResolveAction,
    TurnEnvelope,
    render_task_slate,
)

TASK_TOOL_SCHEMAS = LEAN_TASK_TOOL_SCHEMAS
TASK_START_SCHEMA, TASK_SEND_SCHEMA, TASK_RESOLVE_SCHEMA = TASK_TOOL_SCHEMAS

def _require_object(
    tool: str,
    arguments: dict[str, Any],
    *,
    allowed: frozenset[str],
    required: frozenset[str] = frozenset(),
) -> None:
    if not isinstance(arguments, dict):
        raise ValueError(f"{tool} arguments must be an object")
    unknown = set(arguments) - allowed
    if unknown:
        raise ValueError(f"{tool} has unexpected arguments: {sorted(unknown)}")
    missing = required - set(arguments)
    if missing:
        raise ValueError(f"{tool} requires arguments: {sorted(missing)}")


def _task_ref(arguments: dict[str, Any], tool: str) -> str | None:
    ref = arguments.get("task")
    if ref is None:
        return None
    if not isinstance(ref, str):
        raise ValueError(f"{tool} task must be a string or null")
    return ref.strip() or None


def parse_task_start_call(arguments: dict[str, Any]) -> str:
    """Validate and normalize the semantic name of a ``task_start`` call."""

    _require_object(
        "task_start",
        arguments,
        allowed=frozenset({"name"}),
        required=frozenset({"name"}),
    )
    return normalize_frontbrain_task_name(arguments.get("name"))


def parse_task_send_call(
    arguments: dict[str, Any],
) -> tuple[str | None, TaskLane]:
    """Validate a task target and the mandatory main/fork distinction."""

    _require_object(
        "task_send",
        arguments,
        allowed=frozenset({"task", "lane"}),
        required=frozenset({"lane"}),
    )
    lane = arguments.get("lane")
    if lane not in TASK_LANES:
        raise ValueError(f"task_send lane must be one of {sorted(TASK_LANES)}")
    return _task_ref(arguments, "task_send"), lane  # type: ignore[return-value]


def parse_task_resolve_call(
    arguments: dict[str, Any],
) -> tuple[str | None, TaskResolveAction]:
    """Validate a task target and one bounded runtime decision."""

    _require_object(
        "task_resolve",
        arguments,
        allowed=frozenset({"task", "action"}),
        required=frozenset({"action"}),
    )
    action = arguments.get("action")
    if action not in TASK_RESOLVE_ACTIONS:
        raise ValueError(
            "task_resolve action must be one of "
            f"{sorted(TASK_RESOLVE_ACTIONS)}"
        )
    return _task_ref(arguments, "task_resolve"), action  # type: ignore[return-value]


def control_tool_response(result: TaskControlResult) -> dict[str, Any]:
    """Format a bounded task result for front-brain phrasing.

    The payload carries status, task handles, candidates, and synchronous content.
    """

    payload: dict[str, Any] = {
        "status": result.status,
        "task_ids": list(result.task_ids),
    }
    if result.status == "ambiguous":
        payload["candidates"] = list(result.candidates)
    if result.reason_key:
        payload["reason"] = result.reason_key
    if result.speech:
        # Content for the front brain to convey, not a prescribed phrasing.
        payload["content"] = result.speech
    return payload


def worker_delivery_response(gateway: Any, delivery: Any) -> dict[str, Any]:
    """Project an internal delivery into the bounded front-brain contract."""

    task = (
        gateway.ledger.get_task(delivery.task_id)
        if delivery.task_id is not None
        else None
    )
    task_name = (
        (task.name or task.title).strip()
        if task is not None
        else "后台任务"
    )
    payload: dict[str, Any] = {
        "type": "worker_delivery",
        "task_name": task_name,
        "topic": delivery.topic,
        "content": delivery.speech_hint,
    }
    if delivery.topic == "final":
        if delivery.status not in {
            "completed",
            "partial",
            "failed",
            "cancelled",
        }:
            raise ValueError("final worker delivery requires a terminal status")
        payload["status"] = delivery.status
    elif delivery.status:
        raise ValueError("only final worker delivery may carry a status")
    if delivery.topic == "interaction":
        interaction = (
            gateway.ledger.get_interaction(delivery.interaction_id)
            if delivery.interaction_id
            else None
        )
        if interaction is None:
            raise ValueError("interaction delivery has no matching interaction")
        detail: dict[str, Any] = {"kind": interaction.kind}
        if interaction.kind == "choice":
            detail["choices"] = list(interaction.choices)
        payload["interaction"] = detail
    elif delivery.interaction_id is not None:
        raise ValueError("only interaction delivery may carry an interaction id")
    return payload


class TaskToolHandler:
    """Execute the front-brain's task tools for one Gateway owner.

    Session/transport-agnostic: online_duplex feeds it a tool call + the current
    turn, and gets back a tool_response payload; it also asks for the slate to
    keep the system prompt fresh.
    """

    def __init__(
        self,
        gateway: Any,
        owner_id: str,
        *,
        provider_name: str | None = None,
    ) -> None:
        self._gateway = gateway
        self._owner_id = owner_id
        # Provider routing is deployment policy rather than a model argument.
        self._provider_name = provider_name

    async def handle_task_start(
        self, arguments: dict[str, Any], turn: TurnEnvelope
    ) -> dict[str, Any]:
        name = parse_task_start_call(arguments)
        if self._gateway.mode == "coordinator":
            result = await self._gateway.coordinate_task_start(
                owner_id=self._owner_id,
                turn=turn,
                provider_name=self._provider_name,
                name=name,
            )
        else:
            result = await self._gateway.task_start(
                owner_id=self._owner_id,
                turn=turn,
                provider_name=self._provider_name,
                name=name,
            )
        return control_tool_response(result)

    async def handle_task_send(
        self, arguments: dict[str, Any], turn: TurnEnvelope
    ) -> dict[str, Any]:
        ref, lane = parse_task_send_call(arguments)
        result = await self._gateway.task_send(
            lane,
            owner_id=self._owner_id,
            turn=turn,
            ref=ref,
        )
        return control_tool_response(result)

    async def handle_task_resolve(
        self, arguments: dict[str, Any], turn: TurnEnvelope
    ) -> dict[str, Any]:
        ref, action = parse_task_resolve_call(arguments)
        result = await self._gateway.task_resolve(
            action,
            owner_id=self._owner_id,
            turn=turn,
            ref=ref,
        )
        return control_tool_response(result)

    async def handle_native_tool_call(
        self,
        name: str,
        arguments: dict[str, Any],
        turn: TurnEnvelope,
    ) -> dict[str, Any]:
        """Dispatch one of the three task tools."""

        handlers = {
            "task_start": self.handle_task_start,
            "task_send": self.handle_task_send,
            "task_resolve": self.handle_task_resolve,
        }
        try:
            handler = handlers[name]
        except KeyError:
            raise ValueError(f"unsupported lean control tool: {name!r}") from None
        return await handler(arguments, turn)

    @staticmethod
    def validate_native_tool_call(call: Mapping[str, Any]) -> dict[str, Any]:
        """Validate a task call without mutating Runtime state."""

        if not isinstance(call, Mapping):
            raise TypeError("tool call must be an object")
        name = str(call.get("name") or "")
        arguments = call.get("arguments")
        if not isinstance(arguments, dict):
            raise ValueError("tool arguments must be an object")
        parsers = {
            "task_start": parse_task_start_call,
            "task_send": parse_task_send_call,
            "task_resolve": parse_task_resolve_call,
        }
        try:
            parser = parsers[name]
        except KeyError:
            raise ValueError(f"unsupported lean control tool: {name!r}") from None
        parser(arguments)
        return {"name": name, "arguments": dict(arguments)}

    def task_tool_schemas(self) -> tuple[dict[str, Any], ...]:
        """Schemas with the resolve action enum narrowed to live state."""

        actions = tuple(
            self._gateway.available_task_resolve_actions(self._owner_id)
        )
        schemas: list[dict[str, Any]] = [TASK_START_SCHEMA, TASK_SEND_SCHEMA]
        if actions:
            resolve = {
                **TASK_RESOLVE_SCHEMA,
                "parameters": {
                    **TASK_RESOLVE_SCHEMA["parameters"],
                    "properties": {
                        **TASK_RESOLVE_SCHEMA["parameters"]["properties"],
                        "action": {
                            **TASK_RESOLVE_SCHEMA["parameters"]["properties"][
                                "action"
                            ],
                            "enum": list(actions),
                        },
                    },
                },
            }
            schemas.append(resolve)
        return tuple(schemas)

    def system_prompt_slate(self) -> str:
        """Render the current task slate for system-prompt reference resolution."""

        return render_task_slate(
            self._gateway.task_slate(self._owner_id),
            flat=self._gateway.mode == "lean",
        )
