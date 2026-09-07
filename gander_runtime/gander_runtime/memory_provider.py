"""Memory search providers for Gander workers."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from typing import Any
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


class HttpMemoryProvider:
    """POST memory searches to a deployment-owned JSON endpoint.

    Request body fields match ``MemoryProvider.search``. The endpoint returns
    ``{"results": [...]}``; Gateway performs the final hit and size validation.
    """

    def __init__(
        self,
        endpoint: str,
        *,
        bearer_token: str | None = None,
        timeout_s: float = 10.0,
        max_response_bytes: int = 2 * 1024 * 1024,
    ) -> None:
        parsed = urlsplit(endpoint)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("memory endpoint must be an absolute http(s) URL")
        if parsed.username or parsed.password or parsed.fragment:
            raise ValueError(
                "memory endpoint must not contain credentials or a fragment"
            )
        if timeout_s <= 0:
            raise ValueError("memory HTTP timeout must be positive")
        if max_response_bytes < 1024:
            raise ValueError("memory max_response_bytes must be at least 1024")
        self.endpoint = endpoint
        self.bearer_token = bearer_token
        self.timeout_s = timeout_s
        self.max_response_bytes = max_response_bytes

    async def search(self, **kwargs: Any) -> Sequence[dict[str, Any]]:
        return await asyncio.to_thread(self._search, kwargs)

    def _search(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        body = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "X-Gander-Memory-Version": "1",
        }
        if self.bearer_token:
            headers["Authorization"] = f"Bearer {self.bearer_token}"
        request = Request(
            self.endpoint,
            data=body,
            headers=headers,
            method="POST",
        )
        with urlopen(request, timeout=self.timeout_s) as response:
            raw = response.read(self.max_response_bytes + 1)
        if len(raw) > self.max_response_bytes:
            raise ValueError("memory HTTP response exceeds configured limit")
        try:
            decoded = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("memory HTTP response is not valid JSON") from exc
        if not isinstance(decoded, dict) or not isinstance(
            decoded.get("results"), list
        ):
            raise ValueError("memory HTTP response must contain a results array")
        results = decoded["results"]
        if any(not isinstance(hit, dict) for hit in results):
            raise ValueError("memory HTTP results must contain objects")
        return results
