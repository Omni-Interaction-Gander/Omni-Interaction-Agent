"""Deterministic task-reference resolution for the task-tools runtime.

The front-brain resolves references over the live task slate; the runtime then
binds the selected task here without a generative model on the control path:

  - ref matches one name  -> that task
  - no ref, 1 active task -> that task
  - no ref, N active      -> ambiguous, return candidate names to ask by name
  - ref matches nothing    -> no_such_task

``task_start({name})`` supplies the semantic display name; the Gateway normalizes
and deduplicates it without ever using it as the objective.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from mcpmft.tool_protocol import (
    TASK_NAME_MAX_CHARS,
    assign_frontbrain_task_name,
    normalize_frontbrain_task_name,
)

TaskReferenceAction = Literal["change", "stop", "reply"]
ResolveKind = Literal["target", "ambiguous", "no_such_task", "all"]


@dataclass(frozen=True)
class ActiveTask:
    """Minimal view the resolver needs: a task's id and its shared name."""

    task_id: str
    name: str


@dataclass(frozen=True)
class ResolveOutcome:
    kind: ResolveKind
    task_id: str | None = None
    candidates: tuple[str, ...] = field(default_factory=tuple)
    # True when the matched task is a recently-finished one (still in the slate's
    # "最近完成" window), so task_send can carry its explicit lineage forward.
    terminal: bool = False


def resolve_target(
    action: TaskReferenceAction,
    ref: str | None,
    active: tuple[ActiveTask, ...],
    recent: tuple[ActiveTask, ...] = (),
) -> ResolveOutcome:
    """Resolve which task a task-tool action targets. Pure and deterministic.

    ``active`` are running tasks; ``recent`` are recently-finished tasks still
    shown in the slate. A NAMED ref may match either (so "接着刚那个" works);
    unnamed / "all" only ever touch active tasks (we never implicitly operate on
    a finished one without the user naming it). Destructive/ambiguous cases never
    guess: they return ``ambiguous`` or ``no_such_task``.
    """

    if action not in {"change", "stop", "reply"}:
        raise ValueError(f"unsupported task-reference action: {action!r}")
    ref = (ref or "").strip()
    if ref in {"all", "全部", "所有", "都"}:
        if not active:
            return ResolveOutcome("no_such_task")
        return ResolveOutcome("all")
    if ref:
        matches = [t for t in active if t.name == ref]
        if len(matches) == 1:
            return ResolveOutcome("target", task_id=matches[0].task_id)
        if len(matches) > 1:
            return ResolveOutcome(
                "ambiguous", candidates=tuple(t.name for t in matches)
            )
        # Not active — try the recently-finished window (named reference only).
        recent_matches = [t for t in recent if t.name == ref]
        if len(recent_matches) == 1:
            return ResolveOutcome(
                "target", task_id=recent_matches[0].task_id, terminal=True
            )
        if len(recent_matches) > 1:
            return ResolveOutcome(
                "ambiguous", candidates=tuple(t.name for t in recent_matches)
            )
        return ResolveOutcome("no_such_task")

    if not active:
        return ResolveOutcome("no_such_task")
    if len(active) == 1:
        return ResolveOutcome("target", task_id=active[0].task_id)
    return ResolveOutcome("ambiguous", candidates=tuple(t.name for t in active))
