"""Tests for the full OrcaRouter Flow B (out-of-band) PKCE connect flow.

These drive the real connect adapter with a local fake auth server and an
injected code reader: authorize -> paste code -> exchange -> persist. The fake
server proves the auth origin and exchange path, verifies the S256 challenge,
and rejects reused codes. Tests never touch a real browser or a real account.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import pytest

from gander_runtime.orcarouter.credentials import (
    AUTH_AUTHORIZE_PATH,
    AUTH_EXCHANGE_PATH,
    OrcaCredential,
    OrcaPkceAdapter,
    OrcaPkceStore,
    authorize_url,
)
from gander_runtime.orcarouter.pkce import generate_pkce_pair

FAKE_KEY = "sk-orca-test-fakekey1234"


class FakeAuthServer:
    """A tiny in-process stand-in for www.orcarouter.ai.

    It validates that the authorize URL carries an S256 challenge and a state,
    returns a one-time code at the redirect, then honors exactly one exchange
    at ``/api/v1/auth/keys`` before marking the code used.
    """

    def __init__(self) -> None:
        self.authorize_params: dict[str, list[str]] = {}
        self.code_verifier_received: str | None = None
        self.exchange_calls = 0
        self.minted_code: str | None = None
        self.stored_challenge: str | None = None
        self.stored_state: str | None = None

    def handle_authorize(self, url: str) -> dict[str, str]:
        parts = urlsplit(url)
        assert parts.path == AUTH_AUTHORIZE_PATH
        params = parse_qs(parts.query)
        self.authorize_params = params
        self.stored_challenge = params["code_challenge"][0]
        self.stored_state = params["state"][0]
        assert params["code_challenge_method"] == ["S256"]
        assert params["callback_url"] == ["oob"]
        assert params["scope"] == ["api"]
        # No verifier may ride on the authorize URL.
        assert "code_verifier" not in params
        self.minted_code = "one-time-auth-code-1234"
        return {"code": self.minted_code, "state": self.stored_state}

    def handle_exchange(self, url: str, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        self.exchange_calls += 1
        assert url.endswith(AUTH_EXCHANGE_PATH)
        code = body["code"]
        verifier = body["code_verifier"]
        self.code_verifier_received = verifier
        assert body["code_challenge_method"] == "S256"
        assert self.stored_challenge is not None
        # Server recomputes the S256 challenge from the verifier.
        computed = (
            hashlib.sha256(verifier.encode("utf-8")).digest()
            .__class__(verifier.encode("utf-8"))
        )
        digest = hashlib.sha256(verifier.encode("utf-8")).digest()
        import base64

        challenge = base64.urlsafe_b64encode(digest).decode().rstrip("=")
        if not hmac.compare_digest(challenge, self.stored_challenge):
            return (403, {"error": "invalid_grant"})
        if code != self.minted_code or self.exchange_calls > 1:
            return (403, {"error": "invalid_grant"})
        return (200, {"key": FAKE_KEY, "user_id": "user-1", "scope": "api"})


def test_full_connect_flow_b(tmp_path: Path) -> None:
    server = FakeAuthServer()

    def exchange_factory(url: str, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        return server.handle_exchange(url, body)

    captured_url: list[str] = []

    def open_browser(url: str) -> None:
        captured_url.append(url)

    def read_code(url: str) -> str:
        result = server.handle_authorize(url)
        return result["code"]

    store = OrcaPkceStore(tmp_path / "creds.json")

    async def run() -> None:
        adapter = OrcaPkceAdapter(
            store,
            auth_origin="https://www.orcarouter.ai",
            exchange_factory=exchange_factory,
            read_code_factory=read_code,
            open_browser=open_browser,
        )
        credential = await adapter.acquire()
        assert credential.key == FAKE_KEY
        assert credential.scope == "api"
        # persisted for restart reuse
        assert store.load() is not None

    asyncio.run(run())
    assert captured_url
    assert server.exchange_calls == 1
    assert server.code_verifier_received is not None
    # verifier was presented only at exchange, never in the URL
    assert server.code_verifier_received not in captured_url[0]


def test_connect_reuses_stored_key_without_new_authorization(tmp_path: Path) -> None:
    store = OrcaPkceStore(tmp_path / "creds.json")
    store.save(OrcaCredential(key=FAKE_KEY, user_id="user-1", scope="api"))

    authorizations = []

    def open_browser(url: str) -> None:
        authorizations.append(url)

    async def run() -> None:
        adapter = OrcaPkceAdapter(
            store,
            auth_origin="https://www.orcarouter.ai",
            read_code_factory=lambda url: "unused",
            open_browser=open_browser,
        )
        credential = await adapter.acquire()
        assert credential.key == FAKE_KEY
        assert credential == store.load()

    asyncio.run(run())
    # a stored durable key is reused; no new login is started
    assert authorizations == []
    assert store.load() is not None


def test_connect_denial_ends_cleanly(tmp_path: Path) -> None:
    server = FakeAuthServer()

    def read_code(url: str) -> str:
        params = server.handle_authorize(url)
        return params["code"]

    def exchange_factory(url: str, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        # The user denied; the auth endpoint answers access_denied.
        return (403, {"error": "access_denied"})

    store = OrcaPkceStore(tmp_path / "creds.json")

    async def run() -> None:
        adapter = OrcaPkceAdapter(
            store,
            auth_origin="https://www.orcarouter.ai",
            exchange_factory=exchange_factory,
            read_code_factory=read_code,
        )
        with pytest.raises(RuntimeError) as exc:
            await adapter.acquire()
        assert "refused" in str(exc.value)
        assert store.load() is None

    asyncio.run(run())


def test_connect_reused_code_rejected(tmp_path: Path) -> None:
    server = FakeAuthServer()

    def read_code(url: str) -> str:
        return server.handle_authorize(url)["code"]

    store = OrcaPkceStore(tmp_path / "creds.json")

    async def run() -> None:
        adapter = OrcaPkceAdapter(
            store,
            auth_origin="https://www.orcarouter.ai",
            exchange_factory=(
                lambda url, body: server.handle_exchange(url, body)
            ),
            read_code_factory=read_code,
        )
        first = await adapter.acquire()
        assert first.key == FAKE_KEY
        # second exchange reuses the same one-time code -> 403
        server.exchange_calls = 0  # a fresh adapter attempt still hits the used code
        with pytest.raises(RuntimeError):
            await adapter._exchange_code(server.minted_code, generate_pkce_pair())

    asyncio.run(run())


def test_authorize_url_only_carries_s256_challenge(tmp_path: Path) -> None:
    attempt = generate_pkce_pair()
    url = authorize_url(
        "https://www.orcarouter.ai", challenge=attempt, app_name="Gander"
    )
    params = parse_qs(urlsplit(url).query)
    assert params["code_challenge"][0] == attempt.challenge
    assert params["code_challenge_method"] == ["S256"]
    assert "code_verifier" not in params
    assert attempt.verifier not in url
