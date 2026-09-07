from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


TOOL_CALL_START = "<tool_call>"
TOOL_CALL_END = "</tool_call>"
TOOL_RESPONSE_START = "<tool_response>"
TOOL_RESPONSE_END = "</tool_response>"
TOOL_SCHEMA_PLACEHOLDER = "{{运行时动态注入的 JSON Schema}}"

# Current native task protocol shared by training and runtime.
TASK_NAME_MAX_CHARS = 60
TASK_NAME_PATTERN = r"^[^\r\n:：]+$"
MAX_TOOL_CALLS_PER_UNIT = 4


def normalize_frontbrain_task_name(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("task_start name must be a string")
    if any(character in value for character in ("\r", "\n", ":", "：")):
        raise ValueError("task_start name contains a reserved character")
    normalized = " ".join(value.split())
    if not normalized:
        raise ValueError("task_start name must not be empty")
    if len(normalized) > TASK_NAME_MAX_CHARS:
        raise ValueError(
            f"task_start name exceeds {TASK_NAME_MAX_CHARS} characters"
        )
    return normalized


def assign_frontbrain_task_name(value: object, existing: frozenset[str]) -> str:
    stem = normalize_frontbrain_task_name(value)
    if stem not in existing:
        return stem
    index = 2
    while True:
        suffix = str(index)
        available = max(1, TASK_NAME_MAX_CHARS - len(suffix))
        candidate = f"{stem[:available].rstrip()}{suffix}"
        if candidate not in existing:
            return candidate
        index += 1

LEAN_TASK_START_SCHEMA: dict[str, Any] = {
    "name": "task_start",
    "description": (
        "用当前用户原话新建一个后台任务。name 是简短语义名称,"
        "仅用于任务列表展示和引用;完整任务内容由 Runtime 绑定当前用户 turn。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "minLength": 1,
                "maxLength": TASK_NAME_MAX_CHARS,
                "pattern": TASK_NAME_PATTERN,
                "description": "便于用户后续引用的简短语义任务名,"
                "不是用户原话或任务指令。",
            }
        },
        "required": ["name"],
        "additionalProperties": False,
    },
}

LEAN_TASK_SEND_SCHEMA: dict[str, Any] = {
    "name": "task_send",
    "description": (
        "把当前用户原话送给已有任务。main 会改变或回答主任务;fork 是只读侧问,"
        "不得改变或阻断主任务。"
    ),
    "parameters": {
        "type": "object",
        "additionalProperties": False,
        "required": ["lane"],
        "properties": {
            "task": {
                "type": ["string", "null"],
                "description": "当前任务列表中的任务名;仅一个活跃任务时可省略。",
            },
            "lane": {
                "type": "string",
                "enum": ["main", "fork"],
                "description": "main 改变主任务;fork 只读且不阻断。",
            },
        },
    },
}

LEAN_TASK_RESOLVE_SCHEMA: dict[str, Any] = {
    "name": "task_resolve",
    "description": "对已有任务执行由 Runtime 强制的取消或权限决定。",
    "parameters": {
        "type": "object",
        "additionalProperties": False,
        "required": ["action"],
        "properties": {
            "task": {
                "type": ["string", "null"],
                "description": (
                    "当前任务列表中的任务名;仅 cancel 可使用 all;"
                    "仅一个活跃任务时可省略。"
                ),
            },
            "action": {
                "type": "string",
                "enum": [
                    "cancel",
                    "allow_once",
                    "allow_session",
                    "deny",
                ],
            },
        },
    },
}

LEAN_TASK_TOOL_SCHEMAS = (
    LEAN_TASK_START_SCHEMA,
    LEAN_TASK_SEND_SCHEMA,
    LEAN_TASK_RESOLVE_SCHEMA,
)

_TOOL_CALL_PATTERN = re.compile(
    rf"{re.escape(TOOL_CALL_START)}\s*(.*?)\s*{re.escape(TOOL_CALL_END)}",
    re.DOTALL,
)


class ToolProtocolError(ValueError):
    pass


@dataclass(frozen=True)
class ToolValidationResult:
    calls: tuple[dict[str, Any], ...]
    error: str | None = None

    @property
    def valid(self) -> bool:
        return self.error is None


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=False)


def normalize_tool_schema(tool: Mapping[str, Any]) -> dict[str, Any]:
    value: Mapping[str, Any] = tool
    if tool.get("type") == "function" and isinstance(tool.get("function"), Mapping):
        value = tool["function"]
    name = value.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ToolProtocolError(f"tool schema has no valid name: {tool!r}")
    parameters = value.get("parameters") or {
        "type": "object",
        "properties": {},
        "required": [],
    }
    if not isinstance(parameters, Mapping):
        raise ToolProtocolError(f"tool {name!r} parameters must be an object")
    return {
        "name": name.strip(),
        "description": str(value.get("description") or ""),
        "parameters": _normalize_schema(dict(parameters)),
    }


def ensure_lean_task_tools(
    tools: Sequence[Mapping[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Ensure the complete three-tool protocol and reject removed control faces."""

    normalized = [normalize_tool_schema(tool) for tool in tools or ()]
    names = [tool["name"] for tool in normalized]
    if len(names) != len(set(names)):
        raise ToolProtocolError("tool schemas contain duplicate names")
    if "assist" in names:
        raise ToolProtocolError(
            "three-tool protocol cannot be mixed with a removed control face"
        )
    expected = [normalize_tool_schema(tool) for tool in LEAN_TASK_TOOL_SCHEMAS]
    by_name = {tool["name"]: tool for tool in normalized}
    for schema in expected:
        existing = by_name.get(schema["name"])
        if existing is not None and existing != schema:
            raise ToolProtocolError(
                f"existing {schema['name']} schema does not match the lean protocol"
            )
    # Keep source business-tool order, but always canonicalize the task face to one exact
    # trailing trio. The augmentation path reserves capacity from the tail, and runtime and
    # training must therefore agree even when an imported manifest listed task schemas first.
    expected_names = {schema["name"] for schema in expected}
    business_tools = [tool for tool in normalized if tool["name"] not in expected_names]
    return [*business_tools, *expected]


def render_tool_instructions(tools: Sequence[Mapping[str, Any]]) -> str:
    normalized = [normalize_tool_schema(tool) for tool in tools]
    if not normalized:
        return ""
    schemas = _escape_protocol_openers(compact_json(normalized))
    return f"\n\n<tools>\n{schemas}\n</tools>"


def system_prompt_with_tools(
    system_prompt: str,
    tools: Sequence[Mapping[str, Any]] | None,
) -> str:
    normalized = [normalize_tool_schema(tool) for tool in tools or ()]
    schema_payload = _escape_protocol_openers(compact_json(normalized))
    if TOOL_SCHEMA_PLACEHOLDER in system_prompt:
        return system_prompt.replace(TOOL_SCHEMA_PLACEHOLDER, schema_payload)
    return system_prompt + render_tool_instructions(normalized)


def validate_realtime_tool_context(
    tools: Sequence[Mapping[str, Any]] | None,
    tokenizer: Any,
    *,
    max_tools: int = 6,
    max_schema_tokens: int = 1024,
) -> list[dict[str, Any]]:
    """Validate the small schema set that is allowed on the latency-sensitive path."""

    if max_tools < 1 or max_schema_tokens < 1:
        raise ToolProtocolError("realtime tool limits must be positive")
    normalized = [normalize_tool_schema(tool) for tool in tools or ()]
    names = [tool["name"] for tool in normalized]
    if len(names) != len(set(names)):
        raise ToolProtocolError("realtime tool schemas contain duplicate names")
    if len(normalized) > max_tools:
        raise ToolProtocolError(
            f"realtime exposes {len(normalized)} tools; maximum is {max_tools}"
        )
    rendered = "\n".join(compact_json(tool) for tool in normalized)
    token_count = len(tokenizer.encode(rendered, add_special_tokens=False))
    if token_count > max_schema_tokens:
        raise ToolProtocolError(
            f"realtime tool schemas use {token_count} tokens; maximum is {max_schema_tokens}"
        )
    return normalized


def normalize_tool_call(call: Mapping[str, Any]) -> dict[str, Any]:
    value: Mapping[str, Any] = call
    if isinstance(call.get("function"), Mapping):
        value = call["function"]
    name = value.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ToolProtocolError(f"tool call has no valid name: {call!r}")
    arguments = value.get("arguments", {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise ToolProtocolError(f"tool {name!r} arguments are not valid JSON") from exc
    if not isinstance(arguments, Mapping):
        raise ToolProtocolError(f"tool {name!r} arguments must be a JSON object")
    return {"name": name.strip(), "arguments": dict(arguments)}


def format_tool_call(call: Mapping[str, Any]) -> str:
    normalized = normalize_tool_call(call)
    name_json = json.dumps(normalized["name"], ensure_ascii=False)
    arguments_json = json.dumps(normalized["arguments"], ensure_ascii=False)
    payload = _escape_protocol_openers(
        f'{{"name": {name_json}, "arguments": {arguments_json}}}'
    )
    return f"{TOOL_CALL_START}\n{payload}\n{TOOL_CALL_END}"


def format_tool_calls(calls: Iterable[Mapping[str, Any]]) -> str:
    normalized = [format_tool_call(call) for call in calls]
    if not normalized:
        raise ToolProtocolError("assistant tool action must contain at least one complete call")
    return "\n".join(normalized)


def parse_tool_calls(text: str) -> list[dict[str, Any]]:
    matches = list(_TOOL_CALL_PATTERN.finditer(text))
    if not matches:
        raise ToolProtocolError("no complete <tool_call>...</tool_call> span was generated")
    remainder = _TOOL_CALL_PATTERN.sub("", text).strip()
    if remainder:
        raise ToolProtocolError(f"unexpected text outside tool-call spans: {remainder[:80]!r}")

    calls: list[dict[str, Any]] = []
    for match in matches:
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError as exc:
            raise ToolProtocolError("tool-call payload is not valid JSON") from exc
        if not isinstance(payload, Mapping):
            raise ToolProtocolError("tool-call payload must be a JSON object")
        calls.append(normalize_tool_call(payload))
    return calls


def format_tool_response(value: Any) -> str:
    content = value if isinstance(value, str) else compact_json(value)
    content = _escape_protocol_openers(content)
    return f"{TOOL_RESPONSE_START}\n{content}\n{TOOL_RESPONSE_END}"


def validate_tool_calls(
    calls: Sequence[Mapping[str, Any]],
    tools: Sequence[Mapping[str, Any]] | None,
    *,
    max_calls: int = MAX_TOOL_CALLS_PER_UNIT,
) -> ToolValidationResult:
    try:
        normalized = tuple(normalize_tool_call(call) for call in calls)
        if not normalized:
            raise ToolProtocolError("empty tool-call sequence")
        if len(normalized) > int(max_calls):
            raise ToolProtocolError(
                f"realtime tool unit contains {len(normalized)} calls; maximum is {max_calls}"
            )
        schemas = {
            schema["name"]: schema
            for schema in (normalize_tool_schema(tool) for tool in tools or ())
        }
        for call in normalized:
            schema = schemas.get(call["name"])
            if schema is None:
                raise ToolProtocolError(f"tool {call['name']!r} is not available")
            _validate_schema_value(
                call["arguments"],
                schema.get("parameters") or {},
                path=f"{call['name']}.arguments",
            )
    except ToolProtocolError as exc:
        return ToolValidationResult(calls=(), error=str(exc))
    return ToolValidationResult(calls=normalized)


def validate_json_schema(value: Any, schema: Mapping[str, Any], *, path: str = "value") -> None:
    """Validate an arbitrary JSON value with the same subset used for tool arguments."""

    if not isinstance(schema, Mapping):
        raise ToolProtocolError(f"{path} schema must be an object")
    _validate_schema_value(value, _normalize_schema(schema), path=path)


def _validate_schema_value(value: Any, schema: Mapping[str, Any], *, path: str) -> None:
    for branch_name in ("allOf", "anyOf", "oneOf"):
        branches = schema.get(branch_name)
        if branches is None:
            continue
        if not isinstance(branches, list) or not all(
            isinstance(branch, Mapping) for branch in branches
        ):
            raise ToolProtocolError(f"{path} has malformed {branch_name}")
        matches = []
        for branch in branches:
            try:
                _validate_schema_value(value, branch, path=path)
            except ToolProtocolError:
                matches.append(False)
            else:
                matches.append(True)
        if branch_name == "allOf" and not all(matches):
            raise ToolProtocolError(f"{path} does not satisfy allOf")
        if branch_name == "anyOf" and not any(matches):
            raise ToolProtocolError(f"{path} does not satisfy anyOf")
        if branch_name == "oneOf" and sum(matches) != 1:
            raise ToolProtocolError(f"{path} does not satisfy exactly one oneOf branch")

    if "const" in schema and value != schema["const"]:
        raise ToolProtocolError(f"{path} does not match const")
    expected = schema.get("type")
    if isinstance(expected, list):
        if any(_matches_json_type(value, item) for item in expected):
            expected = None
        else:
            raise ToolProtocolError(f"{path} does not match any allowed JSON type")
    if isinstance(expected, str) and not _matches_json_type(value, expected):
        raise ToolProtocolError(f"{path} must be {expected}")
    if "enum" in schema and value not in schema["enum"]:
        raise ToolProtocolError(f"{path} is not one of the allowed values")

    if isinstance(value, Mapping):
        properties = schema.get("properties") or {}
        required = schema.get("required") or ()
        if not isinstance(properties, Mapping) or not isinstance(required, (list, tuple)):
            raise ToolProtocolError(f"{path} has malformed object schema")
        for name in required:
            if name not in value:
                raise ToolProtocolError(f"{path}.{name} is required")
        if schema.get("additionalProperties") is False:
            unexpected = sorted(set(value) - set(properties))
            if unexpected:
                raise ToolProtocolError(f"{path} has unexpected fields: {unexpected}")
        for name, child in value.items():
            child_schema = properties.get(name)
            if isinstance(child_schema, Mapping):
                _validate_schema_value(child, child_schema, path=f"{path}.{name}")
            elif isinstance(schema.get("additionalProperties"), Mapping):
                _validate_schema_value(
                    child,
                    schema["additionalProperties"],
                    path=f"{path}.{name}",
                )
    elif isinstance(value, list):
        min_items = schema.get("minItems")
        if min_items is not None and len(value) < int(min_items):
            raise ToolProtocolError(f"{path} is shorter than minItems={min_items}")
        max_items = schema.get("maxItems")
        if max_items is not None and len(value) > int(max_items):
            raise ToolProtocolError(f"{path} exceeds maxItems={max_items}")
        if schema.get("uniqueItems") and len(
            {
                json.dumps(
                    item,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                for item in value
            }
        ) != len(value):
            raise ToolProtocolError(f"{path} must contain unique items")
        items = schema.get("items")
        if isinstance(items, Mapping):
            for index, child in enumerate(value):
                _validate_schema_value(child, items, path=f"{path}[{index}]")
    elif isinstance(value, str):
        min_length = schema.get("minLength")
        if min_length is not None and len(value) < int(min_length):
            raise ToolProtocolError(f"{path} is shorter than minLength={min_length}")
        max_length = schema.get("maxLength")
        if max_length is not None and len(value) > int(max_length):
            raise ToolProtocolError(f"{path} exceeds maxLength={max_length}")
        pattern = schema.get("pattern")
        if pattern is not None:
            try:
                matched = re.search(str(pattern), value)
            except re.error as exc:
                raise ToolProtocolError(f"{path} has invalid schema pattern") from exc
            if matched is None:
                raise ToolProtocolError(f"{path} does not match the required pattern")
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        exclusive_minimum = schema.get("exclusiveMinimum")
        exclusive_maximum = schema.get("exclusiveMaximum")
        if minimum is not None and value < minimum:
            raise ToolProtocolError(f"{path} is below minimum={minimum}")
        if maximum is not None and value > maximum:
            raise ToolProtocolError(f"{path} exceeds maximum={maximum}")
        if exclusive_minimum is not None and value <= exclusive_minimum:
            raise ToolProtocolError(
                f"{path} must be greater than exclusiveMinimum={exclusive_minimum}"
            )
        if exclusive_maximum is not None and value >= exclusive_maximum:
            raise ToolProtocolError(
                f"{path} must be less than exclusiveMaximum={exclusive_maximum}"
            )


def _matches_json_type(value: Any, expected: str) -> bool:
    aliases = {"dict": "object", "int": "integer", "float": "number"}
    expected = aliases.get(expected, expected)
    if expected == "object":
        return isinstance(value, Mapping)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "null":
        return value is None
    raise ToolProtocolError(f"unsupported JSON schema type: {expected!r}")


def _normalize_schema(value: Any) -> Any:
    if isinstance(value, Mapping):
        normalized = {str(key): _normalize_schema(item) for key, item in value.items()}
        schema_type = normalized.get("type")
        aliases = {"dict": "object", "int": "integer", "float": "number"}
        if isinstance(schema_type, str):
            normalized["type"] = aliases.get(schema_type, schema_type)
        elif isinstance(schema_type, list):
            normalized["type"] = [
                aliases.get(item, item) if isinstance(item, str) else item
                for item in schema_type
            ]
        return normalized
    if isinstance(value, list):
        return [_normalize_schema(item) for item in value]
    return value


def _escape_protocol_openers(text: str) -> str:
    # JSON decoders restore ``\u003c`` inside tool-call arguments. For arbitrary tool-result text,
    # preserving the escaped spelling is preferable to allowing untrusted output to close the
    # structural span early.
    return text.replace("<", "\\u003c")
