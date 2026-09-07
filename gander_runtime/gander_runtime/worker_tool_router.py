"""Authenticated run-scoped transport for provider-neutral worker tools."""

from __future__ import annotations

import asyncio
import hmac
import json
import os
import secrets
import tempfile
import uuid
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .worker_tools import WORKER_TOOL_NAMES, validate_tool_arguments

_LOOPBACK_HOST = "127.0.0.1"
ToolDispatch = Callable[[str, dict[str, Any], str], Awaitable[dict[str, Any]]]


@dataclass(frozen=True)
class _Route:
    binding_id: str
    owner_id: str
    task_id: str
    project_id: str
    run_id: str
    generation: int
    tools: frozenset[str]
    dispatch: ToolDispatch


class WorkerToolRouter:
    """Route MCP calls to the exact active Worker run with replay fencing."""

    def __init__(
        self,
        *,
        max_buffer_bytes: int = 512 * 1024,
        client_timeout_sec: float = 5.0,
        max_cached_calls: int = 1024,
    ) -> None:
        if max_buffer_bytes <= 0:
            raise ValueError("max_buffer_bytes must be positive")
        if client_timeout_sec <= 0:
            raise ValueError("client_timeout_sec must be positive")
        if max_cached_calls < 1:
            raise ValueError("max_cached_calls must be positive")
        self.max_buffer_bytes = max_buffer_bytes
        self.client_timeout_sec = client_timeout_sec
        self.max_cached_calls = max_cached_calls
        self.route_path: Path | None = None
        self._server: asyncio.Server | None = None
        self._token: str | None = None
        self._port: int | None = None
        self._route: _Route | None = None
        self._client_tasks: set[asyncio.Task[None]] = set()
        self._delivery_tasks: set[asyncio.Task[dict[str, Any]]] = set()
        self._calls: OrderedDict[
            tuple[str, str], tuple[str, asyncio.Task[dict[str, Any]]]
        ] = OrderedDict()

    async def start(self) -> None:
        if self._server is not None:
            return
        stem = f"gander-worker-tools-{os.getpid()}-{uuid.uuid4().hex[:12]}"
        self.route_path = Path(tempfile.gettempdir()) / f"{stem}.route.json"
        self._token = secrets.token_urlsafe(32)
        self._server = await asyncio.start_server(
            self._handle_client,
            host=_LOOPBACK_HOST,
            port=0,
            limit=self.max_buffer_bytes + 1,
        )
        sockets = self._server.sockets or ()
        if not sockets:
            await self.close()
            raise RuntimeError("worker tool router did not open a listening socket")
        self._port = int(sockets[0].getsockname()[1])
        self._write_route(None)

    def bind(
        self,
        *,
        owner_id: str,
        task_id: str,
        project_id: str,
        run_id: str,
        generation: int,
        tools: frozenset[str],
        dispatch: ToolDispatch,
    ) -> None:
        if self._server is None:
            raise RuntimeError("worker tool router is not started")
        normalized = frozenset(tools)
        if not normalized or not normalized <= WORKER_TOOL_NAMES:
            raise ValueError("worker tool route contains unsupported tools")
        route = _Route(
            binding_id=f"binding_{uuid.uuid4().hex}",
            owner_id=owner_id,
            task_id=task_id,
            project_id=project_id,
            run_id=run_id,
            generation=generation,
            tools=normalized,
            dispatch=dispatch,
        )
        self._route = route
        self._calls.clear()
        self._write_route(route)

    def unbind(
        self,
        *,
        task_id: str,
        run_id: str | None = None,
        generation: int | None = None,
    ) -> None:
        route = self._route
        if route is None or route.task_id != task_id:
            return
        if run_id is not None and route.run_id != run_id:
            return
        if generation is not None and route.generation != generation:
            return
        self._route = None
        self._calls.clear()
        self._write_route(None)

    async def close(self) -> None:
        self._route = None
        server = self._server
        self._server = None
        if server is not None:
            server.close()
            await server.wait_closed()
        current = asyncio.current_task()
        clients = [task for task in self._client_tasks if task is not current]
        for task in clients:
            task.cancel()
        if clients:
            await asyncio.gather(*clients, return_exceptions=True)
        deliveries = tuple(self._delivery_tasks)
        for task in deliveries:
            task.cancel()
        if deliveries:
            await asyncio.gather(*deliveries, return_exceptions=True)
        if self.route_path is not None:
            try:
                self.route_path.unlink()
            except FileNotFoundError:
                pass
        self.route_path = None
        self._token = None
        self._port = None
        self._client_tasks.clear()
        self._delivery_tasks.clear()
        self._calls.clear()

    async def _handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        task = asyncio.current_task()
        if task is not None:
            self._client_tasks.add(task)
        try:
            try:
                raw = await asyncio.wait_for(
                    reader.readline(), timeout=self.client_timeout_sec
                )
                if not raw or not raw.endswith(b"\n"):
                    raise ValueError("incomplete worker tool request")
                if len(raw) > self.max_buffer_bytes:
                    raise ValueError("worker tool request exceeds limit")
                envelope = json.loads(raw)
                if not isinstance(envelope, dict):
                    raise ValueError("invalid worker tool request")
                supplied_token = envelope.get("token")
                token = self._token
                if (
                    not isinstance(supplied_token, str)
                    or token is None
                    or not hmac.compare_digest(supplied_token, token)
                ):
                    raise PermissionError("worker tool router authentication failed")
                call = envelope.get("call")
                route = self._route
                if route is None or not _matches(call, route):
                    raise RuntimeError("worker tool route is no longer active")
                assert isinstance(call, dict)
                name = str(call.get("tool") or "")
                if name not in route.tools:
                    raise PermissionError(f"worker tool is not enabled: {name}")
                call_id = str(call.get("call_id") or "")
                if not call_id or len(call_id) > 256:
                    raise ValueError("worker tool call_id is invalid")
                arguments = validate_tool_arguments(name, call.get("arguments"))
                result = await self._dispatch_once(route, name, arguments, call_id)
                response = {"ok": True, "result": result}
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                response = {"ok": False, "error": str(exc)}
            try:
                encoded = json.dumps(
                    response, ensure_ascii=False, separators=(",", ":")
                ).encode("utf-8") + b"\n"
                if len(encoded) > self.max_buffer_bytes:
                    encoded = json.dumps(
                        {"ok": False, "error": "worker tool response exceeds limit"},
                        separators=(",", ":"),
                    ).encode("utf-8") + b"\n"
                writer.write(encoded)
                await writer.drain()
            except (ConnectionError, OSError):
                pass
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except (ConnectionError, OSError):
                pass
            if task is not None:
                self._client_tasks.discard(task)

    async def _dispatch_once(
        self,
        route: _Route,
        name: str,
        arguments: dict[str, Any],
        call_id: str,
    ) -> dict[str, Any]:
        key = (route.binding_id, call_id)
        fingerprint = json.dumps(
            {"name": name, "arguments": arguments},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        existing = self._calls.get(key)
        if existing is not None:
            old_fingerprint, delivery = existing
            if old_fingerprint != fingerprint:
                raise ValueError(f"worker tool call_id collision: {call_id}")
            self._calls.move_to_end(key)
            return await asyncio.shield(delivery)
        delivery = asyncio.create_task(route.dispatch(name, arguments, call_id))
        self._delivery_tasks.add(delivery)
        self._calls[key] = (fingerprint, delivery)
        while len(self._calls) > self.max_cached_calls:
            oldest_key, (_, oldest) = next(iter(self._calls.items()))
            if not oldest.done():
                break
            self._calls.pop(oldest_key, None)
        try:
            result = await asyncio.shield(delivery)
            if not isinstance(result, dict):
                raise TypeError("worker tool dispatcher must return an object")
            return result
        finally:
            self._delivery_tasks.discard(delivery)

    def _write_route(self, route: _Route | None) -> None:
        path = self.route_path
        token = self._token
        port = self._port
        if path is None or token is None or port is None:
            return
        payload = (
            {"version": 1, "active": False}
            if route is None
            else {
                "version": 1,
                "active": True,
                "binding_id": route.binding_id,
                "owner_id": route.owner_id,
                "task_id": route.task_id,
                "project_id": route.project_id,
                "run_id": route.run_id,
                "generation": route.generation,
                "tools": sorted(route.tools),
                "transport": "tcp",
                "host": _LOOPBACK_HOST,
                "port": port,
                "token": token,
            }
        )
        encoded = json.dumps(
            payload, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            fd = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            try:
                _write_all(fd, encoded)
            finally:
                os.close(fd)
            os.replace(temporary, path)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def _matches(call: Any, route: _Route) -> bool:
    return isinstance(call, dict) and (
        call.get("binding_id") == route.binding_id
        and call.get("owner_id") == route.owner_id
        and call.get("task_id") == route.task_id
        and call.get("project_id") == route.project_id
        and call.get("run_id") == route.run_id
        and call.get("generation") == route.generation
    )


def _write_all(fd: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise OSError("short write to worker tool route")
        view = view[written:]
