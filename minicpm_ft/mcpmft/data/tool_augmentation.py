from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

from mcpmft.data.frontbrain_training import TASK_TOOLS_TRAINING_PROTOCOL
from mcpmft.data.sample import OmniSample
from mcpmft.tool_protocol import (
    LEAN_TASK_TOOL_SCHEMAS,
    ToolProtocolError,
    ensure_lean_task_tools,
    normalize_tool_schema,
    validate_realtime_tool_context,
)


TASK_TOOL_NAMES = frozenset(schema["name"] for schema in LEAN_TASK_TOOL_SCHEMAS)
TOOL_AUGMENTATION_META_FIELD = "training_tool_augmentation"
FORBIDDEN_BUSINESS_TOOLS_META_FIELD = "forbidden_business_tool_names"

def load_business_tool_augmentation_catalog(
    path: str | Path,
) -> tuple[tuple[dict[str, Any], ...], dict[str, tuple[str, ...]]]:
    """Load business schemas plus sample-specific counterfactual exclusions."""

    catalog_path = Path(path)
    payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    raw_tools = payload.get("tools") if isinstance(payload, Mapping) else payload
    if not isinstance(raw_tools, list):
        raise ValueError(
            f"business tool catalog {catalog_path} must be a JSON list or an object with tools"
        )

    tools: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw_tool in enumerate(raw_tools):
        if not isinstance(raw_tool, Mapping):
            raise ValueError(
                f"business tool catalog {catalog_path} entry {index} is not an object"
            )
        tool = normalize_tool_schema(raw_tool)
        name = tool["name"]
        if name in TASK_TOOL_NAMES:
            continue
        if name == "assist":
            raise ValueError(
                f"business tool catalog {catalog_path} contains removed tool 'assist'"
            )
        canonical = json.dumps(tool, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if canonical not in seen:
            tools.append(tool)
            seen.add(canonical)
    if not tools:
        raise ValueError(f"business tool catalog {catalog_path} has no usable business schemas")
    raw_forbidden = (
        payload.get("forbidden_tool_names_by_sample_id", {})
        if isinstance(payload, Mapping)
        else {}
    )
    if not isinstance(raw_forbidden, Mapping):
        raise ValueError(
            f"business tool catalog {catalog_path} has invalid counterfactual exclusions"
        )
    forbidden: dict[str, tuple[str, ...]] = {}
    for sample_id, raw_names in raw_forbidden.items():
        if not isinstance(sample_id, str) or not sample_id.strip():
            raise ValueError(f"business tool catalog {catalog_path} has an invalid sample id")
        if not isinstance(raw_names, list):
            raise ValueError(
                f"business tool catalog {catalog_path} exclusions for {sample_id!r} "
                "must be a list"
            )
        names = tuple(dict.fromkeys(str(name).strip() for name in raw_names))
        if any(not name or name in TASK_TOOL_NAMES or name == "assist" for name in names):
            raise ValueError(
                f"business tool catalog {catalog_path} has invalid exclusions for {sample_id!r}"
            )
        forbidden[sample_id] = names
    return tuple(tools), forbidden


def filter_business_tool_catalog_for_realtime(
    tools: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    *,
    max_tools: int = 6,
    max_schema_tokens: int = 1024,
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, str], ...]]:
    """Keep schemas that fit beside the mandatory task face under the runtime budget."""

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    for raw_tool in tools:
        tool = normalize_tool_schema(raw_tool)
        try:
            validate_realtime_tool_context(
                [tool, *LEAN_TASK_TOOL_SCHEMAS],
                tokenizer,
                max_tools=max_tools,
                max_schema_tokens=max_schema_tokens,
            )
        except ToolProtocolError as exc:
            rejected.append({"name": tool["name"], "reason": str(exc)})
        else:
            accepted.append(tool)
    return tuple(accepted), tuple(rejected)


def augment_frontbrain_tool_context(
    sample: OmniSample,
    *,
    protocol: str,
    business_tool_catalog: Sequence[Mapping[str, Any]] = (),
    business_tool_probability: float = 0.0,
    business_tool_min: int = 1,
    business_tool_max: int = 3,
    max_tools: int = 6,
    tokenizer: Any | None = None,
    max_schema_tokens: int = 1024,
    forbidden_tool_names: Sequence[str] = (),
    rng: random.Random,
) -> OmniSample:
    """Materialize the runtime-visible tool interface for a training sample.

    ``task_tools_v1`` always includes the three task schemas. Existing business schemas
    remain available, and sampled additions use the remaining capacity. A deep copy keeps
    epochs and collator workers isolated.
    """

    if protocol != TASK_TOOLS_TRAINING_PROTOCOL:
        return sample
    if not 0.0 <= float(business_tool_probability) <= 1.0:
        raise ValueError("business_tool_probability must be in [0, 1]")
    if business_tool_min < 0 or business_tool_max < business_tool_min:
        raise ValueError("business tool sample bounds must satisfy 0 <= min <= max")
    if max_tools < len(LEAN_TASK_TOOL_SCHEMAS):
        raise ValueError(
            f"max_tools must be at least {len(LEAN_TASK_TOOL_SCHEMAS)} for task_tools_v1"
        )

    augmented = copy.deepcopy(sample)
    try:
        tools = ensure_lean_task_tools(augmented.tools)
    except ToolProtocolError as exc:
        raise ValueError(f"sample {sample.id!r} has invalid tool schemas: {exc}") from exc
    if len(tools) > max_tools:
        raise ValueError(
            f"sample {sample.id!r} exposes {len(tools)} required/existing schemas, "
            f"exceeding max_tools={max_tools}; required schemas are never silently dropped"
        )

    added_names: list[str] = []
    rejected_by_schema_budget: list[str] = []
    requested = 0
    selected = False
    slots = max_tools - len(tools)
    meta_forbidden = augmented.meta.get(FORBIDDEN_BUSINESS_TOOLS_META_FIELD) or ()
    if isinstance(meta_forbidden, (str, bytes)) or not isinstance(meta_forbidden, Sequence):
        raise ValueError(
            f"sample {sample.id!r} has invalid {FORBIDDEN_BUSINESS_TOOLS_META_FIELD}"
        )
    forbidden_names = {
        str(name).strip()
        for name in [*forbidden_tool_names, *meta_forbidden]
        if str(name).strip()
    }
    if slots and business_tool_catalog and rng.random() < float(business_tool_probability):
        selected = True
        existing_names = {tool["name"] for tool in tools}
        requested = rng.randint(business_tool_min, business_tool_max)
        target = min(requested, slots)
        unique_candidates: list[dict[str, Any]] = []
        candidate_names: set[str] = set()

        def consider(raw_tool: Mapping[str, Any]) -> None:
            tool = normalize_tool_schema(raw_tool)
            name = tool["name"]
            if (
                len(unique_candidates) < target
                and name not in existing_names
                and name not in TASK_TOOL_NAMES
                and name != "assist"
                and name not in candidate_names
                and name not in forbidden_names
            ):
                proposed = [
                    *tools[: -len(LEAN_TASK_TOOL_SCHEMAS)],
                    *unique_candidates,
                    tool,
                    *tools[-len(LEAN_TASK_TOOL_SCHEMAS) :],
                ]
                if tokenizer is not None:
                    try:
                        validate_realtime_tool_context(
                            proposed,
                            tokenizer,
                            max_tools=max_tools,
                            max_schema_tokens=max_schema_tokens,
                        )
                    except ToolProtocolError:
                        rejected_by_schema_budget.append(name)
                        return
                unique_candidates.append(tool)
                candidate_names.add(name)

        # Probe a bounded random subset so per-row work is independent of catalog size.
        probe_count = min(len(business_tool_catalog), max(24, target * 16)) if target else 0
        for index in rng.sample(range(len(business_tool_catalog)), probe_count):
            consider(business_tool_catalog[index])
            if len(unique_candidates) >= target:
                break
        insert_at = max(0, len(tools) - len(LEAN_TASK_TOOL_SCHEMAS))
        for offset, tool in enumerate(unique_candidates):
            tools.insert(insert_at + offset, tool)
            added_names.append(tool["name"])

    augmented.tools = tools
    augmented.caps.has_tools = True
    augmented.meta = {
        **augmented.meta,
        TOOL_AUGMENTATION_META_FIELD: {
            "protocol": TASK_TOOLS_TRAINING_PROTOCOL,
            "task_tools_always_visible": True,
            "business_sampling_selected": selected,
            "business_tools_requested": requested,
            "business_tool_slots": slots,
            "business_tools_added": added_names,
            "business_tools_forbidden": sorted(forbidden_names),
            "business_tools_rejected_by_schema_budget": rejected_by_schema_budget,
            "visible_tool_names": [tool["name"] for tool in tools],
        },
    }
    return augmented
