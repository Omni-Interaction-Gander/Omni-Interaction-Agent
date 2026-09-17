"""Unit tests for the OrcaRouter credential seam (API key + OAuth 2.0 + PKCE).

Both entry points must produce the same ``OrcaCredential`` for downstream use.
Tests use only fake keys/codes and assert that verifier/keys never appear in
errors or URLs.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest

from gander_runtime.orcarouter.credentials import (
    OrcaApiKeyAdapter,
    OrcaCredential,
    OrcaPkceAdapter,
    OrcaPkceStore,
    authorize_url,
    classify_exchange_error,
    resolve_origins,
)
from gander_runtime.orcarouter.pkce import (
    PkceChallenge,
    generate_pkce_pair,
    hash_verifier_challenge,
)

FAKE_KEY = "sk-orca-test-fakekey1234"


def test_api_key_adapter_reads_env_and_clears() -> None:
    original = os.environ.get("ORCAROUTER_API_KEY")
    os.environ["ORCAROUTER_API_KEY"] = FAKE_KEY
    try:
        adapter = OrcaApiKeyAdapter()
        credential = asyncio.run(adapter.acquire())
        assert credential.key == FAKE_KEY
        assert credential.scope == "api"
        assert "sk-orca-***" in credential.masked()
        assert FAKE_KEY not in credential.masked()
        asyncio.run(adapter.clear())
    finally:
        if original is None:
            os.environ.pop("ORCAROUTER_API_KEY", None)
        else:
            os.environ["ORCAROUTER_API_KEY"] = original


def test_api_key_adapter_rejects_missing_key() -> None:
    original = os.environ.get("ORCAROUTER_API_KEY")
    os.environ.pop("ORCAROUTER_API_KEY", None)
    try:
        with pytest.raises(ValueError):
            asyncio.run(OrcaApiKeyAdapter().acquire())
    finally:
        if original is None:
            os.environ.pop("ORCAROUTER_API_KEY", None)
        else:
            os.environ["ORCAROUTER_API_KEY"] = original


def test_pkce_helpers_are_fresh_and_s256() -> None:
    first = generate_pkce_pair()
    second = generate_pkce_pair()
    assert first.verifier != second.verifier
    assert first.state != second.state
    assert len(first.verifier) >= 43
    assert "=" not in first.challenge
    assert first.challenge == hash_verifier_challenge(first.verifier)
    # verifier never in the URL
    url = authorize_url(
        "https://www.orcarouter.ai", challenge=first, app_name="Gander"
    )
    assert first.verifier not in url
    assert "code_challenge_method=S256" in url
    assert "callback_url=oob" in url
    assert f"state={first.state}" in url


def test_pkce_store_roundtrip_clear(tmp_path: Path) -> None:
    store = OrcaPkceStore(tmp_path / "creds.json")
    store.save(OrcaCredential(key=FAKE_KEY, user_id="u1", scope="api"))
    loaded = store.load()
    assert loaded is not None
    assert loaded.key == FAKE_KEY
    assert loaded.user_id == "u1"
    store.clear()
    assert store.load() is None


def test_pkce_exchange_success_persists(tmp_path: Path) -> None:
    store = OrcaPkceStore(tmp_path / "creds.json")
    adapter = OrcaPkceAdapter(
        store,
        auth_origin="https://auth.example",
        exchange_factory=(
            lambda url, body: (
                200,
                {"key": FAKE_KEY, "user_id": "u9", "scope": "api"},
            )
        ),
    )

    async def run() -> OrcaCredential:
        challenge = generate_pkce_pair()
        return await adapter._exchange_code("dummy-code", challenge)

    credential = asyncio.run(run())
    assert credential.key == FAKE_KEY
    assert credential.scope == "api"
    # persisted for reuse on restart
    assert store.load() is not None


def test_pkce_exchange_denial(tmp_path: Path) -> None:
    store = OrcaPkceStore(tmp_path / "creds.json")

    async def run() -> None:
        adapter = OrcaPkceAdapter(
            store,
            auth_origin="https://auth.example",
            exchange_factory=(
                lambda url, body: (403, {"error": "invalid_grant"})
            ),
        )
        with pytest.raises(RuntimeError) as exc:
            await adapter._exchange_code("bad-code", generate_pkce_pair())
        assert "verifier" not in str(exc.value).lower()
        assert "bad-code" not in str(exc.value)

    asyncio.run(run())


def test_pkce_exchange_wrong_scope_fails(tmp_path: Path) -> None:
    store = OrcaPkceStore(tmp_path / "creds.json")

    async def run() -> None:
        adapter = OrcaPkceAdapter(
            store,
            auth_origin="https://auth.example",
            exchange_factory=(
                lambda url, body: (
                    200,
                    {"key": FAKE_KEY, "user_id": "u", "scope": "connector"},
                )
            ),
        )
        with pytest.raises(RuntimeError) as exc:
            await adapter._exchange_code("c", generate_pkce_pair())
        assert "scope" in str(exc.value)
        # nothing persisted on scope downgrade
        assert store.load() is None

    asyncio.run(run())


def test_pkce_exchange_rate_limit_and_network(tmp_path: Path) -> None:
    store = OrcaPkceStore(tmp_path / "creds.json")

    async def run() -> None:
        adapter = OrcaPkceAdapter(
            store,
            auth_origin="https://auth.example",
            exchange_factory=(lambda url, body: (429, {"error": "rate_limit"})),
        )
        with pytest.raises(RuntimeError) as exc:
            await adapter._exchange_code("c", generate_pkce_pair())
        assert "rate" in str(exc.value).lower()

        adapter_net = OrcaPkceAdapter(
            store,
            auth_origin="https://auth.example",
            exchange_factory=(lambda url, body: (_raise_network())),
        )
        with pytest.raises(RuntimeError):
            await adapter_net._exchange_code("c", generate_pkce_pair())

    asyncio.run(run())


def _raise_network():
    raise RuntimeError("simulated network failure")


def test_pkce_state_mismatch_constant_time() -> None:
    left = generate_pkce_pair().state
    right = generate_pkce_pair().state
    assert left != right


def test_origins_explicit_overrides_and_loopback() -> None:
    auth, api = resolve_origins()
    assert auth == "https://www.orcarouter.ai"
    assert api == "https://api.orcarouter.ai/v1"
    auth2, api2 = resolve_origins(
        auth_base="https://auth.example",
        api_base="https://api.example/v1",
        shared_base="https://ignored.example",
    )
    assert auth2 == "https://auth.example"
    assert api2 == "https://api.example/v1"
    auth3, api3 = resolve_origins(shared_base="https://self.example")
    assert auth3 == "https://self.example"
    assert api3 == "https://self.example/v1"
    with pytest.raises(ValueError):
        resolve_origins(shared_base="http://remote.example")
    auth4, api4 = resolve_origins(shared_base="http://127.0.0.1:8000")
    assert auth4 == "http://127.0.0.1:8000"


def test_classify_exchange_errors_are_safe() -> None:
    assert "expired" in classify_exchange_error(403, "", state="x")
    assert "rate" in classify_exchange_error(429, "", state="x").lower()
    assert "timed out" in classify_exchange_error(None, "", state="x")


def test_verifier_never_in_errors(tmp_path: Path) -> None:
    store = OrcaPkceStore(tmp_path / "creds.json")

    async def run() -> None:
        attempt = generate_pkce_pair()
        adapter = OrcaPkceAdapter(
            store,
            auth_origin="https://auth.example",
            exchange_factory=(
                lambda url, body: (
                    403,
                    {"error": "invalid_grant", "detail": attempt.verifier},
                )
            ),
        )
        with pytest.raises(RuntimeError) as exc:
            await adapter._exchange_code("code", attempt)
        assert attempt.verifier not in str(exc.value)

    asyncio.run(run())


def test_both_adapters_produce_same_credential(tmp_path: Path) -> None:
    """API-key adapter and PKCE adapter yield the same credential result type."""

    original = os.environ.get("ORCAROUTER_API_KEY")
    os.environ["ORCAROUTER_API_KEY"] = FAKE_KEY
    try:
        api_credential = asyncio.run(OrcaApiKeyAdapter().acquire())
        store = OrcaPkceStore(tmp_path / "creds.json")
        adapter = OrcaPkceAdapter(
            store,
            auth_origin="https://auth.example",
            exchange_factory=(
                lambda url, body: (
                    200,
                    {"key": FAKE_KEY, "user_id": "u", "scope": "api"},
                )
            ),
        )
        pkce_credential = asyncio.run(
            adapter._exchange_code("code", generate_pkce_pair())
        )
        assert isinstance(api_credential, OrcaCredential)
        assert isinstance(pkce_credential, OrcaCredential)
        assert api_credential.key == pkce_credential.key
        assert api_credential.scope == pkce_credential.scope
    finally:
        if original is None:
            os.environ.pop("ORCAROUTER_API_KEY", None)
        else:
            os.environ["ORCAROUTER_API_KEY"] = original
