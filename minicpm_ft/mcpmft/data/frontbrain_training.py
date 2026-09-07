from __future__ import annotations

from collections import Counter, deque
from typing import Any, Mapping

from mcpmft.data.sample import OmniSample
from mcpmft.tool_protocol import (
    LEAN_TASK_TOOL_SCHEMAS,
    MAX_TOOL_CALLS_PER_UNIT,
    assign_frontbrain_task_name,
    normalize_tool_schema,
    validate_tool_calls,
)


TASK_TOOLS_TRAINING_PROTOCOL = "task_tools_v1"
SUPPORTED_FRONTBRAIN_TRAINING_PROTOCOLS = frozenset(
    {"none", TASK_TOOLS_TRAINING_PROTOCOL}
)
TASK_TOOL_NAMES = frozenset({"task_start", "task_send", "task_resolve"})
VISIBLE_TASK_CONTEXT_CONTRACT = "visible_task_context_v2"
VISIBLE_TASK_LIFECYCLE_CONTRACT = "visible_task_lifecycle_v1"


def replay_visible_task_lifecycle(sample: OmniSample) -> dict[str, Any]:
    """Replay task identity from native calls and their actual responses.

    The response to ``task_start`` owns the assigned display name.  A differing
    name is valid only when Runtime deterministically suffixes a visible collision;
    every later send, resolve, receipt event, and Worker delivery must use that
    assigned name rather than the model's originally proposed name.
    """

    active: dict[str, set[str]] = {}
    recent: dict[str, set[str]] = {}
    pending: deque[tuple[int, Mapping[str, Any]]] = deque()
    issues: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()

    def issue(code: str, turn_index: int, **details: Any) -> None:
        issues.append({"code": code, "turn_index": turn_index, **details})
        counts[f"issue.{code}"] += 1

    def apply_response(turn_index: int, response: Mapping[str, Any]) -> None:
        if not pending:
            issue("synchronous_response_without_call", turn_index)
            return
        action_index, call = pending.popleft()
        name = str(call.get("name") or "")
        if name not in TASK_TOOL_NAMES:
            return
        counts[name] += 1
        arguments = call.get("arguments") or {}
        if not isinstance(arguments, Mapping):
            issue("task_call_or_response_not_object", turn_index, call=name)
            return
        status = str(response.get("status") or "")
        successful = status == "ok"
        response_task_ids = {
            str(task_id)
            for task_id in response.get("task_ids") or ()
            if str(task_id)
        }

        if name == "task_start":
            requested = str(arguments.get("name") or "").strip()
            if not requested:
                issue("task_start_missing_requested_name", action_index)
                return
            if not successful:
                events.append(
                    {
                        "kind": name,
                        "turn_index": action_index,
                        "requested_name": requested,
                        "status": status,
                    }
                )
                return
            assigned = str(response.get("content") or "").strip()
            if not assigned:
                issue("task_start_missing_assigned_name", turn_index, requested=requested)
                return
            expected = assign_frontbrain_task_name(
                requested, frozenset((*active, *recent))
            )
            collision_renamed = expected != requested
            counts[
                "task_start.collision_renamed"
                if collision_renamed
                else "task_start.name_unchanged"
            ] += 1
            if assigned != expected:
                issue(
                    "task_start_assigned_name_mismatch",
                    turn_index,
                    requested=requested,
                    assigned=assigned,
                    expected=expected,
                    visible_task_names=sorted((*active, *recent)),
                )
                return
            if assigned in active or assigned in recent:
                issue("task_start_duplicate_assigned_name", turn_index, assigned=assigned)
                return
            active[assigned] = response_task_ids
            events.append(
                {
                    "kind": name,
                    "turn_index": action_index,
                    "requested_name": requested,
                    "assigned_name": assigned,
                    "collision_renamed": collision_renamed,
                    "status": status,
                    "task_ids": sorted(response_task_ids),
                }
            )
            return

        task_value = arguments.get("task")
        terminal_reference = False
        if name == "task_resolve" and task_value == "all":
            resolved_names = sorted(active)
            visible = bool(resolved_names)
        elif task_value is None:
            resolved_names = list(active) if len(active) == 1 else []
            visible = len(resolved_names) == 1
        else:
            explicit_name = str(task_value).strip()
            if explicit_name in active:
                resolved_names = [explicit_name]
            elif explicit_name in recent:
                resolved_names = [explicit_name]
                terminal_reference = True
            else:
                resolved_names = []
            visible = bool(resolved_names)

        if successful and not visible:
            issue(
                "successful_task_reference_not_visible",
                action_index,
                call=name,
                task=task_value,
                visible_task_names=sorted((*active, *recent)),
            )
        elif not successful and visible and status == "no_such_task":
            issue(
                "visible_task_reference_returned_no_such_task",
                turn_index,
                call=name,
                task=task_value,
                resolved_names=resolved_names,
            )
        elif not successful and not visible and status not in {
            "no_such_task",
            "unsupported",
            "ambiguous",
        }:
            issue(
                "unknown_task_reference_has_invalid_status",
                turn_index,
                call=name,
                task=task_value,
                status=status,
            )

        known_ids = (
            set().union(
                *(
                    active[item] if item in active else recent[item]
                    for item in resolved_names
                )
            )
            if resolved_names
            else set()
        )
        terminal_continuation = (
            successful
            and name == "task_send"
            and arguments.get("lane") == "main"
            and terminal_reference
        )
        continuation_name: str | None = None
        if terminal_continuation:
            requested = resolved_names[0]
            continuation_name = str(response.get("content") or "").strip()
            expected = assign_frontbrain_task_name(
                requested, frozenset((*active, *recent))
            )
            if not continuation_name:
                issue(
                    "terminal_continuation_missing_assigned_name",
                    turn_index,
                    task=requested,
                )
            elif continuation_name != expected:
                issue(
                    "terminal_continuation_assigned_name_mismatch",
                    turn_index,
                    task=requested,
                    assigned=continuation_name,
                    expected=expected,
                    visible_task_names=sorted((*active, *recent)),
                )
            elif not response_task_ids:
                issue(
                    "terminal_continuation_missing_task_id",
                    turn_index,
                    task=requested,
                )
            else:
                active[continuation_name] = response_task_ids
        if (
            successful
            and known_ids
            and response_task_ids
            and not terminal_continuation
            and not response_task_ids.issubset(known_ids)
        ):
            issue(
                "task_response_ids_do_not_match_reference",
                turn_index,
                call=name,
                task_names=resolved_names,
                expected_task_ids=sorted(known_ids),
                response_task_ids=sorted(response_task_ids),
            )
        events.append(
            {
                "kind": name,
                "turn_index": action_index,
                "task_argument": task_value,
                "resolved_names": resolved_names,
                "terminal_reference": terminal_reference,
                **(
                    {"assigned_name": continuation_name}
                    if continuation_name
                    else {}
                ),
                "status": status,
                "task_ids": sorted(response_task_ids),
            }
        )
        if (
            successful
            and name == "task_resolve"
            and arguments.get("action") == "cancel"
        ):
            # Runtime acknowledges cancellation synchronously but the task
            # remains active/cancelling until its final worker delivery.
            pass

    for turn_index, turn in enumerate(sample.turns):
        if turn.tool_calls:
            if pending:
                issue(
                    "new_action_before_previous_responses",
                    turn_index,
                    pending_calls=len(pending),
                )
                pending.clear()
            pending.extend((turn_index, call) for call in turn.tool_calls)
            counts["tool_calls"] += len(turn.tool_calls)
            continue

        response = turn.tool_response
        if response is None:
            continue
        if isinstance(response, Mapping) and response.get("type") == "worker_delivery":
            task_name = str(response.get("task_name") or "").strip()
            terminal = response.get("topic") == "final"
            counts["worker_deliveries"] += 1
            if not task_name:
                issue("worker_delivery_missing_task_name", turn_index)
            elif task_name not in active and task_name not in recent:
                issue(
                    "worker_delivery_unknown_task_name",
                    turn_index,
                    task_name=task_name,
                    visible_task_names=sorted((*active, *recent)),
                )
            events.append(
                {
                    "kind": "worker_delivery",
                    "turn_index": turn_index,
                    "task_name": task_name,
                    "status": response.get("status"),
                    "topic": response.get("topic"),
                }
            )
            if terminal and task_name in active:
                recent[task_name] = active.pop(task_name)
            continue

        if not isinstance(response, Mapping):
            # Direct business tools may legally return any JSON value (for
            # example a number, string, or list).  This replay owns only the
            # task lifecycle, so consume a single pending non-task response
            # without imposing the task-control response object contract.
            if (
                len(pending) == 1
                and str(pending[0][1].get("name") or "") not in TASK_TOOL_NAMES
            ):
                pending.popleft()
                continue
            issue("task_call_or_response_not_object", turn_index)
            pending.clear()
            continue
        if response.get("status") == "batch":
            results = response.get("results")
            if (
                not isinstance(results, list)
                or len(results) != len(pending)
                or not all(isinstance(item, Mapping) for item in results)
            ):
                issue(
                    "batch_response_mismatch",
                    turn_index,
                    pending_calls=len(pending),
                    result_count=len(results) if isinstance(results, list) else None,
                )
                pending.clear()
                continue
            for item in results:
                apply_response(turn_index, item)
        else:
            apply_response(turn_index, response)

    return {
        "contract_version": VISIBLE_TASK_LIFECYCLE_CONTRACT,
        "issues": issues,
        "issue_codes": sorted({item["code"] for item in issues}),
        "events": events,
        "counts": dict(sorted(counts.items())),
        "remaining_visible_task_names": sorted((*active, *recent)),
        "remaining_active_task_names": sorted(active),
        "remaining_recent_terminal_task_names": sorted(recent),
        "pending_response_count": len(pending),
    }


def analyze_visible_task_context(sample: OmniSample) -> dict[str, Any]:
    """Describe whether the current task action is grounded in visible history.

    A static task slate is a valid runtime input, but some training releases deliberately disable
    it. In that mode every active task in the bound pre-action harness state must be recoverable
    from earlier native calls and their synchronous responses in the same sample, not only the
    task referenced by the current target. Natural-language acknowledgements are not task identity
    because the harness never parses them as state.
    """

    bound_task_names = [
        str(name).strip()
        for name in sample.meta.get("harness_active_task_names", [])
        if str(name).strip()
    ]
    bound_recent_names = [
        str(name).strip()
        for name in sample.meta.get("harness_recent_terminal_task_names", [])
        if str(name).strip()
    ]
    if (
        len(bound_task_names) != len(set(bound_task_names))
        or len(bound_recent_names) != len(set(bound_recent_names))
        or set(bound_task_names) & set(bound_recent_names)
    ):
        raise ValueError(
            f"sample {sample.id!r} contains duplicate harness task names"
        )

    action_indexes = [
        index for index, turn in enumerate(sample.turns) if turn.tool_calls
    ]
    marked_current = [
        index
        for index in action_indexes
        if str(sample.turns[index].meta.get("phase") or "") == "action"
    ]
    current_index = (
        marked_current[0]
        if len(marked_current) == 1
        else action_indexes[-1]
        if action_indexes
        else None
    )
    active_names: list[str] = []
    recent_names: list[str] = []
    pending_calls: list[Mapping[str, Any]] = []

    def apply_context_result(
        call: Mapping[str, Any],
        response: Mapping[str, Any],
    ) -> None:
        nonlocal active_names, recent_names
        call_name = str(call.get("name") or "")
        arguments = call.get("arguments") or {}
        if not isinstance(arguments, Mapping):
            return
        if response.get("status") == "ok" and call_name == "task_start":
            resolved_name = str(
                response.get("content") or arguments.get("name") or ""
            ).strip()
            if (
                resolved_name
                and resolved_name not in active_names
                and resolved_name not in recent_names
            ):
                active_names.append(resolved_name)
        elif (
            response.get("status") == "ok"
            and call_name == "task_send"
            and arguments.get("lane") == "main"
            and str(arguments.get("task") or "") in recent_names
        ):
            resolved_name = str(response.get("content") or "").strip()
            if (
                resolved_name
                and resolved_name not in active_names
                and resolved_name not in recent_names
            ):
                active_names.append(resolved_name)
        elif (
            response.get("status") == "ok"
            and call_name == "task_resolve"
            and arguments.get("action") == "cancel"
        ):
            # The receipt only confirms that cancellation was accepted.
            # Runtime keeps the task visible as active/cancelling until the
            # final worker delivery transitions it into recent terminal.
            pass

    context_end = current_index if current_index is not None else len(sample.turns)
    for turn in sample.turns[:context_end]:
        task_calls = [
            call
            for call in turn.tool_calls
            if str(call.get("name") or "") in TASK_TOOL_NAMES
        ]
        if task_calls:
            pending_calls = task_calls
            continue
        response = turn.tool_response
        if (
            isinstance(response, Mapping)
            and bool(turn.meta.get("runtime_event"))
            and response.get("type") == "worker_delivery"
            and response.get("topic") == "final"
        ):
            task_name = str(response.get("task_name") or "").strip()
            if task_name in active_names:
                active_names = [
                    name for name in active_names if name != task_name
                ]
                if task_name not in recent_names:
                    recent_names.append(task_name)
            continue
        if (
            not pending_calls
            or not isinstance(response, Mapping)
            or bool(turn.meta.get("runtime_event"))
        ):
            continue
        if response.get("status") == "batch":
            results = response.get("results")
            if isinstance(results, list) and len(results) == len(pending_calls):
                for call, result in zip(pending_calls, results):
                    if isinstance(result, Mapping):
                        apply_context_result(call, result)
        elif len(pending_calls) == 1:
            apply_context_result(pending_calls[0], response)
        pending_calls = []

    visible_names = [*active_names, *recent_names]
    hidden_task_names = [
        name
        for name in (*bound_task_names, *bound_recent_names)
        if name not in visible_names
    ]
    common = {
        "contract_version": VISIBLE_TASK_CONTEXT_CONTRACT,
        "harness_task_names": bound_task_names,
        "harness_recent_terminal_task_names": bound_recent_names,
        "visible_task_names": active_names,
        "visible_recent_terminal_task_names": recent_names,
        "hidden_task_names": hidden_task_names,
    }
    if current_index is None:
        return {
            "current_call": None,
            "current_task_ref": None,
            "control_status": None,
            **common,
            "context_source": (
                "pinned_context" if hidden_task_names else "not_applicable"
            ),
            "requires_pinned_context": bool(hidden_task_names),
        }

    current_task_calls = [
        call
        for call in sample.turns[current_index].tool_calls
        if str(call.get("name") or "") in TASK_TOOL_NAMES
    ]
    if not current_task_calls:
        return {
            "current_call": None,
            "current_task_ref": None,
            "control_status": None,
            **common,
            "context_source": (
                "pinned_context" if hidden_task_names else "not_applicable"
            ),
            "requires_pinned_context": bool(hidden_task_names),
        }

    current_call = current_task_calls[0]
    call_name = (
        str(current_call.get("name") or "")
        if len(current_task_calls) == 1
        else "batch"
    )
    arguments = current_call.get("arguments") or {}
    task_ref = (
        arguments.get("task")
        if len(current_task_calls) == 1 and isinstance(arguments, Mapping)
        else None
    )
    control_status: str | None = None
    current_results: list[Mapping[str, Any]] = []
    for turn in sample.turns[current_index + 1 :]:
        if turn.tool_calls:
            break
        response = turn.tool_response
        if (
            isinstance(response, Mapping)
            and not bool(turn.meta.get("runtime_event"))
        ):
            control_status = str(response.get("status") or "")
            if control_status == "batch":
                results = response.get("results")
                if isinstance(results, list):
                    current_results = [
                        item for item in results if isinstance(item, Mapping)
                    ]
            else:
                current_results = [response]
            break

    if hidden_task_names:
        context_source = "pinned_context"
        requires_pinned = True
    elif call_name == "batch":
        requires_pinned = False
        for index, call in enumerate(current_task_calls):
            name = str(call.get("name") or "")
            call_arguments = call.get("arguments") or {}
            if name not in {"task_send", "task_resolve"} or not isinstance(
                call_arguments, Mapping
            ):
                continue
            ref = call_arguments.get("task")
            if ref == "all":
                visible_match = bool(active_names)
            elif ref is not None:
                visible_match = (
                    str(ref) in active_names or str(ref) in recent_names
                )
            else:
                visible_match = len(active_names) == 1
            result_status = (
                str(current_results[index].get("status") or "")
                if index < len(current_results)
                else ""
            )
            if not visible_match and result_status not in {
                "no_such_task",
                "unsupported",
            }:
                requires_pinned = True
                break
        context_source = "pinned_context" if requires_pinned else "conversation"
    elif call_name not in {"task_send", "task_resolve"}:
        context_source = "not_applicable"
        requires_pinned = False
    else:
        if task_ref == "all":
            visible_match = bool(active_names)
        elif task_ref is not None:
            visible_match = (
                str(task_ref) in active_names or str(task_ref) in recent_names
            )
        else:
            visible_match = len(active_names) == 1
        if visible_match:
            context_source = "conversation"
            requires_pinned = False
        elif control_status in {"no_such_task", "unsupported"}:
            context_source = "conversation_absence"
            requires_pinned = False
        else:
            context_source = "pinned_context"
            requires_pinned = True

    return {
        "current_call": call_name,
        "current_task_ref": task_ref,
        "control_status": control_status,
        **common,
        "context_source": context_source,
        "requires_pinned_context": requires_pinned,
    }


def validate_frontbrain_training_sample(
    sample: OmniSample,
    *,
    protocol: str,
    use_sample_pinned_context: bool = True,
) -> Counter[str]:
    """Validate one row before native front-brain serialization.

    Raw ordinary audio-video Omni and speech rows may omit tools in their immutable manifests.
    The training collator materializes the complete task-tools face on every row before calling
    this validator. Any row that exposes tools must use that exact face, while optional bounded
    direct tools may be present beside it.
    """

    if protocol == "none":
        return Counter()
    if protocol not in SUPPORTED_FRONTBRAIN_TRAINING_PROTOCOLS:
        raise ValueError(f"unsupported front-brain training protocol: {protocol!r}")

    has_tool_content = bool(sample.tools) or any(
        turn.tool_calls or turn.role == "tool" or turn.tool_response is not None
        for turn in sample.turns
    )
    if not has_tool_content:
        return Counter()
    if not sample.caps.has_tools:
        raise ValueError(f"tool-bearing sample {sample.id!r} must set caps.has_tools=true")

    schemas = [normalize_tool_schema(tool) for tool in sample.tools]
    names = [schema["name"] for schema in schemas]
    if len(names) != len(set(names)):
        raise ValueError(f"sample {sample.id!r} exposes duplicate tool schemas")
    if "assist" in names:
        raise ValueError(
            f"sample {sample.id!r} mixes a removed control-tool face into task_tools_v1"
        )

    expected = {
        schema["name"]: normalize_tool_schema(schema)
        for schema in LEAN_TASK_TOOL_SCHEMAS
    }
    actual = {schema["name"]: schema for schema in schemas}
    mismatched = [
        name for name, schema in expected.items() if actual.get(name) != schema
    ]
    if mismatched:
        raise ValueError(
            f"sample {sample.id!r} does not expose the exact task_tools_v1 schemas: "
            f"{mismatched}"
        )

    call_counts: Counter[str] = Counter()
    for turn in sample.turns:
        if not turn.tool_calls:
            continue
        validation = validate_tool_calls(
            turn.tool_calls, schemas, max_calls=MAX_TOOL_CALLS_PER_UNIT
        )
        if not validation.valid:
            raise ValueError(
                f"sample {sample.id!r} contains an invalid native tool call: "
                f"{validation.error}"
            )
        call_counts.update(call["name"] for call in validation.calls)
    task_context = analyze_visible_task_context(sample)
    task_lifecycle = replay_visible_task_lifecycle(sample)
    # A task that is present only in a sample-pinned SLATE is valid while pinned
    # context is enabled.  Report the more actionable hidden-state error below
    # when the run deliberately disables that context instead of failing early
    # with the generic visible-lifecycle diagnostic.
    lifecycle_issues = list(task_lifecycle["issues"])
    hidden_lifecycle_codes = {
        "successful_task_reference_not_visible",
    }
    if (
        task_context["requires_pinned_context"]
        and all(
            str(item.get("code") or "") in hidden_lifecycle_codes
            for item in lifecycle_issues
        )
    ):
        lifecycle_issues = []
    if lifecycle_issues:
        raise ValueError(
            f"sample {sample.id!r} has inconsistent visible task identity: "
            f"{sorted({item['code'] for item in lifecycle_issues})}"
        )
    declared_requires_pinned = sample.meta.get("requires_pinned_context")
    if declared_requires_pinned is not None and (
        not isinstance(declared_requires_pinned, bool)
        or declared_requires_pinned != task_context["requires_pinned_context"]
    ):
        raise ValueError(
            f"sample {sample.id!r} has stale task-context metadata"
        )
    if (
        not use_sample_pinned_context
        and task_context["requires_pinned_context"]
    ):
        raise ValueError(
            f"sample {sample.id!r} targets {task_context['current_call']} for a task "
            "that exists only in hidden/static slate state; add visible prior native task "
            "events or enable sample pinned context"
        )
    return call_counts


def is_task_tools_sample(sample: OmniSample) -> bool:
    names = {normalize_tool_schema(schema)["name"] for schema in sample.tools}
    return {"task_start", "task_send", "task_resolve"}.issubset(names)
