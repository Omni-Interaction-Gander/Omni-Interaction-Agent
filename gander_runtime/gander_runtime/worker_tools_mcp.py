"""Dependency-free stdio MCP server for Gander's three worker tools."""

from __future__ import annotations

import ipaddress
import json
import os
import socket
import sys
from pathlib import Path
from typing import Any

from gander_runtime.contracts import stable_id
from gander_runtime.worker_tools import (
    WORKER_TOOL_NAMES,
    WORKER_TOOL_SCHEMAS,
    validate_tool_arguments,
)


def _reply(request_id: Any, result: Any = None, error: Any = None) -> None:
    message: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id}
    message["error" if error is not None else "result"] = (
        error if error is not None else result
    )
    sys.stdout.write(
        json.dumps(message, ensure_ascii=False, separators=(",", ":")) + "\n"
    )
    sys.stdout.flush()


def _allowed_tools() -> frozenset[str]:
    raw = os.environ.get("GANDER_WORKER_TOOL_ALLOW", "").strip()
    if not raw:
        raise RuntimeError("GANDER_WORKER_TOOL_ALLOW is not configured")
    allowed = frozenset(item.strip() for item in raw.split(",") if item.strip())
    if not allowed or not allowed <= WORKER_TOOL_NAMES:
        raise RuntimeError("GANDER_WORKER_TOOL_ALLOW contains unsupported tools")
    return allowed


def _call_worker_tool(
    name: str,
    arguments: dict[str, Any],
    *,
    call_id: str,
) -> dict[str, Any]:
    normalized = validate_tool_arguments(name, arguments)
    route_path = os.environ.get("GANDER_WORKER_TOOLS_ROUTE")
    if not route_path:
        raise RuntimeError("GANDER_WORKER_TOOLS_ROUTE is not configured")
    route = _load_route(route_path)
    if not route.get("active"):
        raise RuntimeError("no active Gander worker accepts tool calls")
    if name not in set(route.get("tools") or ()):
        raise RuntimeError(f"worker tool is not active: {name}")
    call = {
        "binding_id": route["binding_id"],
        "owner_id": route["owner_id"],
        "task_id": route["task_id"],
        "project_id": route["project_id"],
        "run_id": route["run_id"],
        "generation": int(route["generation"]),
        "tool": name,
        "call_id": call_id,
        "arguments": normalized,
    }
    acknowledgement = _router_request(route, {"call": call})
    result = acknowledgement.get("result")
    if not isinstance(result, dict):
        raise RuntimeError("worker tool router returned an invalid result")
    return result


def _mcp_call_id(name: str, request_id: Any, allowed: frozenset[str]) -> str:
    """Map a JSON-RPC request to a replay-stable, server-scoped call id."""

    if isinstance(request_id, bool) or not isinstance(request_id, (str, int)):
        raise ValueError("tools/call requires a string or integer request id")
    identity = json.dumps(
        {
            "server_tools": sorted(allowed),
            "tool": name,
            "request_id": request_id,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return stable_id("mcp", identity)


def _router_request(
    route: dict[str, Any], message: dict[str, Any]
) -> dict[str, Any]:
    host = str(route.get("host") or "")
    try:
        if not ipaddress.ip_address(host).is_loopback:
            raise ValueError
    except ValueError as exc:
        raise RuntimeError("worker tool router must use a loopback address") from exc
    try:
        port = int(route["port"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("worker tool router port is unavailable") from exc
    token = route.get("token")
    if not 0 < port < 65536 or not isinstance(token, str) or not token:
        raise RuntimeError("worker tool router endpoint is invalid")
    payload = json.dumps(
        {"token": token, **message},
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8") + b"\n"
    if len(payload) > 512 * 1024:
        raise ValueError("worker tool request exceeds limit")
    try:
        with socket.create_connection((host, port), timeout=5) as connection:
            timeout = float(os.environ["GANDER_WORKER_TOOL_TIMEOUT_SEC"])
            if timeout <= 0:
                raise ValueError("worker tool timeout must be positive")
            connection.settimeout(timeout)
            connection.sendall(payload)
            response = _read_line(connection, limit=512 * 1024)
    except (OSError, ValueError) as exc:
        raise RuntimeError("worker tool router is unavailable") from exc
    try:
        acknowledgement = json.loads(response)
    except json.JSONDecodeError as exc:
        raise RuntimeError("worker tool router returned an invalid response") from exc
    if not isinstance(acknowledgement, dict) or not acknowledgement.get("ok"):
        error = (
            acknowledgement.get("error")
            if isinstance(acknowledgement, dict)
            else None
        )
        raise RuntimeError(str(error or "worker tool router rejected the call"))
    return acknowledgement


def _load_route(path: str) -> dict[str, Any]:
    try:
        route = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("worker tool route is unavailable") from exc
    if not isinstance(route, dict):
        raise RuntimeError("worker tool route is invalid")
    return route


def _read_line(connection: socket.socket, *, limit: int) -> bytes:
    chunks: list[bytes] = []
    size = 0
    while True:
        chunk = connection.recv(min(4096, limit - size + 1))
        if not chunk:
            raise RuntimeError("worker tool router closed without a response")
        newline = chunk.find(b"\n")
        if newline >= 0:
            chunks.append(chunk[:newline])
            return b"".join(chunks)
        chunks.append(chunk)
        size += len(chunk)
        if size > limit:
            raise RuntimeError("worker tool router response exceeds limit")


def main() -> int:
    allowed = _allowed_tools()
    tools = [tool for tool in WORKER_TOOL_SCHEMAS if tool["name"] in allowed]
    for raw_line in sys.stdin.buffer:
        try:
            message = json.loads(raw_line)
        except json.JSONDecodeError:
            _reply(None, error={"code": -32700, "message": "Parse error"})
            continue
        method = message.get("method")
        request_id = message.get("id")
        if method == "initialize":
            requested = (message.get("params") or {}).get("protocolVersion")
            if not isinstance(requested, str) or not requested:
                _reply(
                    request_id,
                    error={"code": -32602, "message": "protocolVersion is required"},
                )
                continue
            _reply(
                request_id,
                {
                    "protocolVersion": requested,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": "gander-worker-tools", "version": "1.0.0"},
                    "instructions": (
                        "Pull memory/context only when needed and share only meaningful "
                        "user-facing updates."
                    ),
                },
            )
        elif method == "tools/list":
            _reply(request_id, {"tools": tools})
        elif method == "tools/call":
            params = message.get("params") or {}
            try:
                name = str(params.get("name") or "")
                if name not in allowed:
                    raise ValueError(f"unknown tool: {name}")
                result = _call_worker_tool(
                    name,
                    params.get("arguments") or {},
                    call_id=_mcp_call_id(name, request_id, allowed),
                )
                rendered = json.dumps(
                    result, ensure_ascii=False, separators=(",", ":")
                )
                payload = {
                    "content": [{"type": "text", "text": rendered}],
                    "structuredContent": result,
                    "isError": False,
                }
            except Exception as exc:
                payload = {
                    "content": [{"type": "text", "text": str(exc)}],
                    "isError": True,
                }
            _reply(request_id, payload)
        elif method in {"ping", "shutdown"}:
            _reply(request_id, {})
        elif request_id is not None:
            _reply(
                request_id,
                error={"code": -32601, "message": f"Method not found: {method}"},
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
