"""OpenAI-compatible inference client for the OrcaRouter relay.

``https://api.orcarouter.ai/v1`` speaks the OpenAI wire format, so requests use
the standard chat-completions shape. The client holds one plain OrcaRouter API
key from the credential seam and never implements its own authentication.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

DEFAULT_API_BASE = "https://api.orcarouter.ai/v1"
CHAT_COMPLETIONS_PATH = "/chat/completions"
CLIENT_TIMEOUT_S = 120.0
CLIENT_MAX_BODY_BYTES = 16 * 1024 * 1024


class OrcaTransportError(RuntimeError):
    """A bounded transport/HTTP failure from the OrcaRouter relay."""

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class OrcaRouterClient:
    """One stateless OpenAI-compatible chat client for the relay."""

    def __init__(
        self,
        api_base: str,
        *,
        credential: Any,
        timeout_s: float = CLIENT_TIMEOUT_S,
        request_factory: Any | None = None,
    ) -> None:
        self.api_base = api_base.rstrip("/")
        self._credential = credential
        self._timeout_s = timeout_s
        self._request = request_factory

    def _bearer(self) -> str:
        key = getattr(self._credential, "key", None)
        if not isinstance(key, str) or not key:
            raise OrcaTransportError("OrcaRouter API key is unavailable")
        return f"Bearer {key}"

    async def chat_completions(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int = 512,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if temperature is not None:
            body["temperature"] = temperature
        url = f"{self.api_base}{CHAT_COMPLETIONS_PATH}"
        try:
            return await self._post_json(url, body)
        except OrcaTransportError as exc:
            raise exc

    async def _post_json(
        self, url: str, body: dict[str, Any]
    ) -> dict[str, Any]:
        if self._request is not None:
            result = self._request(url, body, self._bearer())
            if hasattr(result, "__await__"):
                result = await result
            return _require_object(result)
        encoded = json.dumps(body, separators=(",", ":")).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": self._bearer(),
            "Accept": "application/json",
        }
        request = urllib.request.Request(
            url, data=encoded, headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self._timeout_s
            ) as response:
                raw = response.read(CLIENT_MAX_BODY_BYTES + 1)
                if len(raw) > CLIENT_MAX_BODY_BYTES:
                    raise OrcaTransportError(
                        "OrcaRouter response exceeds the size limit"
                    )
                return _require_object(json.loads(raw))
        except urllib.error.HTTPError as exc:
            if exc.code == 401:
                raise OrcaTransportError(
                    "OrcaRouter rejected this API key (401); re-authorize or "
                    "paste a new key",
                    status=401,
                ) from exc
            if exc.code == 429:
                raise OrcaTransportError(
                    "OrcaRouter is rate-limiting this key (429); wait before "
                    "retrying",
                    status=429,
                ) from exc
            detail = exc.read(CLIENT_MAX_BODY_BYTES + 1)
            message = detail.decode("utf-8", errors="replace")[:500]
            raise OrcaTransportError(
                f"OrcaRouter inference failed with HTTP {exc.code}: {message}",
                status=exc.code,
            ) from exc
        except urllib.error.URLError as exc:
            raise OrcaTransportError(
                f"OrcaRouter inference is unreachable: {exc.reason}"
            ) from exc


def _require_object(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise OrcaTransportError("OrcaRouter response is not an object")
    return value
