"""Deterministic task-reference resolution over the live task slate.

A named reference selects its matching task. An omitted reference selects the sole
active task or returns the active names as ambiguous. Unmatched names return
``no_such_task``. Task names are display handles rather than worker objectives.
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
    # Marks a task matched through the recent-completion window.
    terminal: bool = False


def resolve_target(
    action: TaskReferenceAction,
    ref: str | None,
    active: tuple[ActiveTask, ...],
    recent: tuple[ActiveTask, ...] = (),
) -> ResolveOutcome:
    """Resolve a task action against active and recently completed tasks.

    Named references may match either set. Unnamed and ``all`` references apply to
    active tasks; ambiguous or unmatched references return an explicit status.
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
        # Named references may resolve through the recent-completion window.
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
