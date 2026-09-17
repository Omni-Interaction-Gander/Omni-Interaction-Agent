"""Live regression tests through the implemented OrcaRouter provider path.

These tests are only run with a real ``ORCAROUTER_API_KEY``; otherwise they are
skipped. They prove the project's own code path (catalog client + provider
chat client) reaches the real OrcaRouter relay and model catalog.
"""

from __future__ import annotations

import asyncio
import os

import pytest

from gander_runtime.orcarouter.catalog import OrcaCatalog
from gander_runtime.orcarouter.client import OrcaRouterClient
from gander_runtime.orcarouter.credentials import (
    OrcaApiKeyAdapter,
    OrcaCredential,
)

pytestmark = pytest.mark.skipif(
    not os.environ.get("ORCAROUTER_API_KEY"),
    reason="requires a real ORCAROUTER_API_KEY",
)


@pytest.fixture(scope="module")
def live_key() -> str:
    return os.environ["ORCAROUTER_API_KEY"]


def test_live_model_catalog_via_implemented_path(live_key: str) -> None:
    async def run() -> None:
        catalog = OrcaCatalog(
            "https://api.orcarouter.ai/v1",
            credential=OrcaCredential(key=live_key),
        )
        models = await catalog.refresh(capability="chat")
        assert catalog.degraded is False
        assert models, "live chat catalog must not be empty"
        # orcarouter/auto is a first-class workspace model and must be present
        assert any(
            model.get("id") == "orcarouter/auto" for model in models
        ), "orcarouter/auto missing from live chat catalog"

    asyncio.run(run())


def test_live_inference_via_implemented_provider_client(live_key: str) -> None:
    async def run() -> None:
        client = OrcaRouterClient(
            "https://api.orcarouter.ai/v1",
            credential=OrcaCredential(key=live_key),
        )
        response = await client.chat_completions(
            model="orcarouter/auto",
            messages=[{"role": "user", "content": "Reply with the single word: orca"}],
            max_tokens=16,
        )
        choices = response.get("choices")
        assert isinstance(choices, list) and choices
        message = choices[0].get("message")
        assert isinstance(message, dict)
        content = str(message.get("content") or "")
        assert content

    asyncio.run(run())


def test_live_api_key_adapter_accepts_real_key(live_key: str) -> None:
    async def run() -> None:
        adapter = OrcaApiKeyAdapter()
        credential = await adapter.acquire()
        assert credential.key == live_key

    asyncio.run(run())
