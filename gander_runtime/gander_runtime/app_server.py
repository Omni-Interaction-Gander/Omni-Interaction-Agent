from __future__ import annotations

import asyncio
import json
from collections import deque
from pathlib import Path
from typing import Any


class AppServerError(RuntimeError):
    def __init__(self, message: str, *, response_error: Any | None = None) -> None:
        super().__init__(message)
        self.response_error = response_error


class CodexAppServer:
    """JSON-RPC client for one private ``codex app-server --stdio`` process."""

    def __init__(
        self,
        command: list[str],
        *,
        cwd: str | Path,
        stderr_path: str | Path,
        env: dict[str, str] | None = None,
        answer_server_requests: bool = True,
    ) -> None:
        self.command = command
        self.cwd = str(Path(cwd).resolve())
        self.stderr_path = Path(stderr_path)
        self.env = env
        self.answer_server_requests = answer_server_requests
        self.process: asyncio.subprocess.Process | None = None
        self.notifications: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        self._pending: dict[int, asyncio.Future[Any]] = {}
        self._request_id = 0
        self._stdout_task: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._write_lock = asyncio.Lock()
        self.stderr_tail: deque[str] = deque(maxlen=80)

    async def start(self, *, timeout: float = 30.0) -> None:
        self.stderr_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.process = await asyncio.create_subprocess_exec(
                *self.command,
                cwd=self.cwd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self.env,
                limit=8 * 1024 * 1024,
            )
            self._stderr_task = asyncio.create_task(
                self._read_stderr(), name="codex-app-stderr"
            )
            self._stdout_task = asyncio.create_task(
                self._read_stdout(), name="codex-app-stdout"
            )
            await asyncio.wait_for(
                self.request(
                    "initialize",
                    {
                        "clientInfo": {"name": "gander-runtime", "version": "1.0.0"},
                        "capabilities": {"experimentalApi": True},
                    },
                ),
                timeout=timeout,
            )
            await self.notify("initialized", {})
        except BaseException:
            await self.close()
            raise

    async def request(
        self, method: str, params: dict[str, Any], *, timeout: float | None = 30.0
    ) -> Any:
        if not self.process or self.process.returncode is not None:
            raise AppServerError("Codex app-server is not running")
        self._request_id += 1
        request_id = self._request_id
        loop = asyncio.get_running_loop()
        future: asyncio.Future[Any] = loop.create_future()
        self._pending[request_id] = future
        await self._send(
            {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
        )
        try:
            if timeout is None:
                return await future
            return await asyncio.wait_for(future, timeout=timeout)
        finally:
            self._pending.pop(request_id, None)

    async def notify(self, method: str, params: dict[str, Any]) -> None:
        await self._send({"jsonrpc": "2.0", "method": method, "params": params})

    async def respond_server_request(
        self,
        request_id: str | int,
        *,
        result: dict[str, Any] | None = None,
        error: dict[str, Any] | None = None,
    ) -> None:
        """Return a response to one app-server initiated JSON-RPC request."""

        if isinstance(request_id, bool) or not isinstance(request_id, (str, int)):
            raise TypeError("server request id must be a string or integer")
        if (result is None) == (error is None):
            raise ValueError("exactly one of result or error is required")
        response: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id}
        if result is not None:
            response["result"] = result
        else:
            response["error"] = error
        await self._send(response)

    async def close(self) -> None:
        process = self.process
        if not process:
            return
        if process.returncode is None:
            if process.stdin:
                process.stdin.close()
            try:
                await asyncio.wait_for(process.wait(), timeout=3)
            except asyncio.TimeoutError:
                try:
                    process.terminate()
                except ProcessLookupError:
                    pass
                try:
                    await asyncio.wait_for(process.wait(), timeout=3)
                except asyncio.TimeoutError:
                    try:
                        process.kill()
                    except ProcessLookupError:
                        pass
                    else:
                        await process.wait()
        tasks = tuple(
            task for task in (self._stdout_task, self._stderr_task) if task is not None
        )
        for task in tasks:
            if task and not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._stdout_task = None
        self._stderr_task = None
        await self.notifications.put(None)
        self.process = None

    async def _send(self, message: dict[str, Any]) -> None:
        if not self.process or not self.process.stdin:
            raise AppServerError("Codex app-server stdin is unavailable")
        data = json.dumps(message, ensure_ascii=False, separators=(",", ":")).encode()
        async with self._write_lock:
            self.process.stdin.write(data + b"\n")
            await self.process.stdin.drain()

    async def _read_stdout(self) -> None:
        assert self.process and self.process.stdout
        failure: AppServerError | None = None
        try:
            while line := await self.process.stdout.readline():
                try:
                    message = json.loads(line)
                except json.JSONDecodeError as exc:
                    failure = AppServerError(
                        "Codex app-server emitted invalid JSON on stdout: "
                        + line.decode("utf-8", errors="replace").rstrip()[:500]
                    )
                    raise failure from exc
                await self._handle_message(message)
        finally:
            code = self.process.returncode if self.process else None
            detail = "\n".join(self.stderr_tail)
            error = failure or AppServerError(
                f"Codex app-server exited ({code}). {detail}".strip()
            )
            for future in list(self._pending.values()):
                if not future.done():
                    future.set_exception(error)
            await self.notifications.put(None)

    async def _read_stderr(self) -> None:
        assert self.process and self.process.stderr
        with self.stderr_path.open("ab") as handle:
            while line := await self.process.stderr.readline():
                handle.write(line)
                handle.flush()
                self.stderr_tail.append(line.decode("utf-8", errors="replace").rstrip())

    async def _handle_message(self, message: Any) -> None:
        if not isinstance(message, dict):
            return
        if "method" in message and "id" in message:
            if self.answer_server_requests:
                await self._handle_server_request(message)
            else:
                await self.notifications.put(message)
            return
        if "method" in message:
            await self.notifications.put(message)
            return
        if "id" not in message:
            return
        future = self._pending.get(message["id"])
        if not future or future.done():
            return
        if message.get("error") is not None:
            future.set_exception(
                AppServerError(
                    str(message["error"]),
                    response_error=message["error"],
                )
            )
        else:
            future.set_result(message.get("result"))

    async def _handle_server_request(self, message: dict[str, Any]) -> None:
        """Reject app-server approvals that are not routed through supervision."""

        method = str(message.get("method", ""))
        if "requestApproval" in method or "request_approval" in method:
            if method == "item/permissions/requestApproval":
                result: dict[str, Any] = {"permissions": {}, "scope": "turn"}
            else:
                result = {"decision": "decline"}
            response = {"jsonrpc": "2.0", "id": message["id"], "result": result}
        else:
            response = {
                "jsonrpc": "2.0",
                "id": message["id"],
                "error": {"code": -32601, "message": f"Unsupported client method: {method}"},
            }
        await self._send(response)
