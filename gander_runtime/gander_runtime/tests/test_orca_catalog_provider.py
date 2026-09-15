"""Tests for the OrcaRouter catalog, capability filters, and provider client.

Live discovery is authoritative; the verified seed is a bounded, metadata-intact
fallback. Every model capability entry point gets its own filter, and the
provider run path is exercised against a fake relay with a real Bearer header.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import pytest

from gander_runtime.orcarouter.catalog import (
    CATALOG_ENDPOINT_TYPES,
    VerifiedModelSeed,
    capability_filter,
    OrcaCatalog,
    VERIFIED_SEED_MODELS,
)
from gander_runtime.orcarouter.client import OrcaRouterClient, OrcaTransportError
from gander_runtime.orcarouter.credentials import OrcaCredential

FAKE_KEY = "sk-orca-test-fakekey1234"


def _model(
    model_id: str,
    *,
    endpoints: tuple[str, ...] = ("openai",),
    modalities: tuple[str, ...] = ("text",),
) -> dict[str, Any]:
    architecture = {"input_modalities": list(modalities)} if modalities else None
    return {
        "id": model_id,
        "supported_endpoint_types": list(endpoints),
        "architecture": architecture,
    }


def test_chat_filter_excludes_non_text_routes() -> None:
    models = [
        _model("chat-a"),
        _model("img", endpoints=("image-generation",)),
        _model("vid", endpoints=("openai-video",)),
        _model("rerank", endpoints=("jina-rerank",)),
        _model("emb", endpoints=("embeddings",)),
    ]
    chat = [m["id"] for m in capability_filter(models, capability="chat")]
    assert chat == ["chat-a"]


def test_multimodal_fail_closed() -> None:
    models = [
        _model("text-only"),
        _model("image-capable", modalities=("text", "image")),
        _model("undeclared"),
    ]
    image = [
        m["id"]
        for m in capability_filter(
            models, capability="chat", required_input_modalities=("image",)
        )
    ]
    assert image == ["image-capable"]


def test_embedding_image_video_rerank_filters() -> None:
    models = [
        _model("emb1", endpoints=("embeddings",)),
        _model("img1", endpoints=("image-generation",)),
        _model("vid1", endpoints=("openai-video",)),
        _model("rk1", endpoints=("jina-rerank",)),
    ]
    assert [m["id"] for m in capability_filter(models, capability="embedding")] == [
        "emb1"
    ]
    assert [m["id"] for m in capability_filter(models, capability="image")] == ["img1"]
    assert [m["id"] for m in capability_filter(models, capability="video")] == ["vid1"]
    assert [m["id"] for m in capability_filter(models, capability="rerank")] == ["rk1"]


def test_seed_metadata_preserved() -> None:
    seed = VerifiedModelSeed.from_raw(VERIFIED_SEED_MODELS[0])
    assert seed.id == "openai/gpt-5.5"
    assert seed.context_length == 1_000_000
    assert "image" in seed.input_modalities
    assert seed.reasoning == {"effort_ladder": ["low", "medium", "high", "xhigh"]}
    rendered = seed.to_dict()
    assert rendered["reasoning"]["effort_ladder"] == [
        "low",
        "medium",
        "high",
        "xhigh",
    ]


def test_catalog_degraded_falls_back_to_verified_seed() -> None:
    class BrokenCatalog(OrcaCatalog):
        async def _fetch_catalog(self, url: str) -> Any:
            raise RuntimeError("catalog down")

    async def run() -> None:
        catalog = BrokenCatalog("https://api.orcarouter.ai/v1")
        models = await catalog.refresh(capability="chat")
        assert catalog.degraded is True
        ids = [m["id"] for m in models]
        assert ids == [
            "openai/gpt-5.5",
            "anthropic/claude-opus-4.8",
            "google/gemini-3.5-flash",
            "deepseek/deepseek-v4-pro",
            "orcarouter/auto",
        ]

    asyncio.run(run())


def test_catalog_live_success_is_authoritative_and_bounded() -> None:
    raw = {
        "data": [
            {
                "id": f"model-{index}",
                "supported_endpoint_types": ["openai"],
            }
            for index in range(300)
        ]
    }

    async def run() -> None:
        catalog = OrcaCatalog("https://api.orcarouter.ai/v1", max_items=1024)
        catalog._fetch = lambda url: raw  # type: ignore[assignment]
        models = await catalog.refresh(capability="chat")
        assert catalog.degraded is False
        assert len(models) == 300

    asyncio.run(run())


def test_client_sends_bearer_and_parses_choices() -> None:
    seen_bearer: list[str] = []

    def fake_request(url: str, body: dict[str, Any], bearer: str) -> dict[str, Any]:
        seen_bearer.append(bearer)
        assert body["model"] == "orcarouter/auto"
        return {"choices": [{"message": {"content": "orca"}}]}

    async def run() -> None:
        client = OrcaRouterClient(
            "https://api.orcarouter.ai/v1",
            credential=OrcaCredential(key=FAKE_KEY),
            request_factory=fake_request,
        )
        out = await client.chat_completions(
            model="orcarouter/auto",
            messages=[{"role": "user", "content": "hi"}],
        )
        assert out["choices"][0]["message"]["content"] == "orca"

    asyncio.run(run())
    assert seen_bearer == [f"Bearer {FAKE_KEY}"]


def test_client_429_and_401_terminal() -> None:
    def fake_429(url: str, body: dict[str, Any], bearer: str) -> dict[str, Any]:
        raise OrcaTransportError("boom", status=429)

    def fake_401(url: str, body: dict[str, Any], bearer: str) -> dict[str, Any]:
        raise OrcaTransportError("unauthorized", status=401)

    async def run() -> None:
        c429 = OrcaRouterClient(
            "https://api.orcarouter.ai/v1",
            credential=OrcaCredential(key=FAKE_KEY),
            request_factory=fake_429,
        )
        with pytest.raises(OrcaTransportError) as exc:
            await c429.chat_completions(model="x", messages=[])
        assert exc.value.status == 429

        c401 = OrcaRouterClient(
            "https://api.orcarouter.ai/v1",
            credential=OrcaCredential(key=FAKE_KEY),
            request_factory=fake_401,
        )
        with pytest.raises(OrcaTransportError) as exc:
            await c401.chat_completions(model="x", messages=[])
        assert exc.value.status == 401

    asyncio.run(run())


def test_provider_roundtrip_through_gateway() -> None:
    """One task runs through the real provider code path with a fake relay."""

    from gander_runtime.orcarouter.provider import (
        OrcaRouterProvider,
        OrcaRouterProviderConfig,
    )

    requests: list[tuple[str, dict[str, Any], str]] = []

    def fake_client_factory(api_base: str, credential: Any) -> OrcaRouterClient:
        def fake_request(
            url: str, body: dict[str, Any], bearer: str
        ) -> dict[str, Any]:
            requests.append((url, body, bearer))
            return {"choices": [{"message": {"content": "done"}}]}

        return OrcaRouterClient(
            api_base,
            credential=credential,
            request_factory=fake_request,
        )

    async def run() -> None:
        from gander_runtime.coordination import DonePayload

        provider = OrcaRouterProvider(
            OrcaRouterProviderConfig(
                cwd=str(Path.cwd()),
                runtime_dir=str(Path.cwd() / "tmp-orca-provider-test"),
                model="orcarouter/auto",
                api_base="https://api.orcarouter.ai/v1",
                auth_base="https://www.orcarouter.ai",
                api_key_env="ORCAROUTER_API_KEY",
            ),
            credential_source=lambda: _FakeApiSource(),
            client_factory=fake_client_factory,
        )
        project = _make_project(provider.name)
        opened = await provider.open_project(project)
        request = _make_worker_request(project)
        run_channel = await opened.start(request, _FakeControl())
        events = []
        async for event in run_channel.events():
            events.append(event)
            if isinstance(event.payload, DonePayload):
                break
        await run_channel.close()

        done = next((e for e in events if isinstance(e.payload, DonePayload)), None)
        assert done is not None, "provider run must emit a terminal done event"
        assert done.payload.status == "completed"
        assert done.payload.result == "done"
        assert requests
        url, body, bearer = requests[0]
        assert url == "https://api.orcarouter.ai/v1/chat/completions"
        assert bearer == f"Bearer {FAKE_KEY}"
        assert body["model"] == "orcarouter/auto"

    asyncio.run(run())


class _FakeApiSource:
    source_name = "orcarouter-api-key"

    async def acquire(self) -> OrcaCredential:
        return OrcaCredential(key=FAKE_KEY)

    async def clear(self) -> None:
        return None


def _make_project(provider_name: str) -> Any:
    from gander_runtime.coordination import ProjectRecord

    return ProjectRecord(
        project_id="project-test-1",
        owner_id="owner-test-1",
        label="default",
        provider_name=provider_name,
    )


def _make_worker_request(project: Any) -> Any:
    from gander_runtime.coordination import (
        ContextPlan,
        WorkerPolicyView,
        WorkerRequest,
    )

    return WorkerRequest(
        task_id="task-test-1",
        run_id="run-test-1",
        project_id=project.project_id,
        owner_id=project.owner_id,
        generation=1,
        instruction="Write the word orca.",
        context_plan=ContextPlan(),
        policy=WorkerPolicyView(
            contract_revision=1,
            must_ask=(),
            delegated_questions=(),
            allowed_actions=("read", "search", "draft", "edit_draft"),
            permission_actions=(),
            denied_actions=(),
            subscribed_milestones=(),
        ),
    )


class _FakeControl:
    task_id = "task-test-1"
    run_id = "run-test-1"
    capabilities: Any = None

    async def record_evidence(self, ref: str, facts: dict[str, Any]) -> None:
        return None

    async def fetch_turn(self, *args: Any, **kwargs: Any) -> Any:
        return None

    async def memory_search(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"status": "unavailable", "results": []}

    async def context_fetch(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"status": "unavailable", "results": []}
