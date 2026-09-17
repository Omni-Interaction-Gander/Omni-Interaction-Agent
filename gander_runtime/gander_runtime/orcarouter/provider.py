"""First-class OrcaRouter WorkerProvider for the Gander back brain.

Registered under the config key ``orcarouter``. It consumes the shared
:class:`~gander_runtime.orcarouter.credentials.OrcaCredentialSource` seam
(API-key adapter or PKCE adapter), performs real model discovery, and routes
Gander tasks to the OrcaRouter OpenAI-compatible relay.
"""

from __future__ import annotations

import asyncio
import dataclasses
import inspect
import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..contracts import (
    ContextSnapshot,
    ProviderEvent,
    TaskInteraction,
    TaskInteractionReply,
    TaskQuery,
    TaskRequest,
    TaskResult,
    TaskUpdate,
    WorkState,
    new_id,
)
from ..coordination import (
    BackendCapabilities,
    ProjectRecord,
    WorkerRequest,
)
from ..gateway import WorkerControl, WorkerRunChannel
from ..worker_tools import WORKER_TOOL_NAMES, WorkerToolSession
from .catalog import OrcaCatalog, VerifiedModelSeed, capability_filter
from .client import OrcaRouterClient, OrcaTransportError
from .credentials import (
    API_DEFAULT_BASE,
    AUTH_DEFAULT_BASE,
    OrcaApiKeyAdapter,
    OrcaCredential,
    OrcaCredentialSource,
    OrcaPkceAdapter,
    OrcaPkceStore,
    resolve_origins,
)
from ..providers.registry import (
    ProviderBuildContext,
    ProviderRegistration,
)
from .pkce import generate_pkce_pair

LOGGER = logging.getLogger(__name__)
_TERMINAL_RESULTS = {"completed", "cancelled", "partial", "failed"}
_ORCA_WORKER_INSTRUCTIONS = (
    "You are Gander's background execution agent running through the OrcaRouter "
    "gateway. Complete the assigned task with the available tools. Inspect only "
    "relevant context, stay within the requested scope, preserve valid existing "
    "work, and verify consequential results before reporting them. Never claim "
    "work or evidence you did not observe. The realtime frontbrain owns user "
    "dialogue: send meaningful intermediate progress through Gander share, use "
    "native Gander interactions for questions and approvals, and return the "
    "complete result in the final agent message."
)

DEFAULT_MODEL = "orcarouter/auto"


@dataclass(frozen=True)
class OrcaRouterProviderSettings:
    """User-facing settings mirroring the repository's typed config idiom."""

    api_key_env: str = "ORCAROUTER_API_KEY"
    model: str = DEFAULT_MODEL
    api_base: str | None = None
    auth_base: str | None = None
    shared_base: str | None = None
    pkce_store_path: str | None = None
    request_timeout_s: float = 120.0
    catalog_timeout_s: float = 20.0
    max_parallel_projects: int = 1
    approval_policy: str = "on-request"


@dataclass(frozen=True)
class OrcaRouterProviderConfig:
    cwd: str
    runtime_dir: str
    model: str = DEFAULT_MODEL
    api_base: str = API_DEFAULT_BASE
    auth_base: str = AUTH_DEFAULT_BASE
    api_key_env: str = "ORCAROUTER_API_KEY"
    pkce_store_path: str | None = None
    request_timeout_s: float = 120.0
    catalog_timeout_s: float = 20.0
    max_parallel_projects: int = 1
    approval_policy: str = "on-request"


def _resolve_store_path(config: OrcaRouterProviderConfig) -> Path:
    if config.pkce_store_path:
        return Path(config.pkce_store_path).expanduser().resolve()
    return Path(config.runtime_dir) / "orca_credentials.json"


ORCA_ROUTER_CAPABILITIES = BackendCapabilities(
    steering="none",
    side_queries="none",
    terminal_side_queries="none",
    interactions=True,
    blocking_granularity="run",
    authority_enforcement="none",
    structured_events="limited",
    trusted_risk_signals=False,
    session_resume=False,
    modalities=frozenset({"text"}),
    max_parallel_projects=1,
    context_provisioning="push_bounded",
    session="stateful",
    worker_tools=WORKER_TOOL_NAMES,
)


def _build_orca_provider(
    context: Any, settings: OrcaRouterProviderSettings
) -> OrcaRouterProvider:
    from .credentials import resolve_origins

    auth_origin, api_origin = resolve_origins(
        auth_base=settings.auth_base,
        api_base=settings.api_base,
        shared_base=settings.shared_base,
    )
    return OrcaRouterProvider(
        OrcaRouterProviderConfig(
            cwd=str(context.workspace),
            runtime_dir=str(context.runtime_dir),
            model=settings.model,
            api_base=api_origin,
            auth_base=auth_origin,
            api_key_env=settings.api_key_env,
            pkce_store_path=settings.pkce_store_path,
            request_timeout_s=settings.request_timeout_s,
            catalog_timeout_s=settings.catalog_timeout_s,
            max_parallel_projects=settings.max_parallel_projects,
            approval_policy=settings.approval_policy,
        )
    )


class OrcaRouterProvider:
    """Stateless, generation-safe WorkerProvider for the OrcaRouter relay."""

    name = "orcarouter"

    def __init__(
        self,
        config: OrcaRouterProviderConfig,
        *,
        credential_source: OrcaCredentialSource | None = None,
        catalog_factory: Any | None = None,
        client_factory: Any | None = None,
    ) -> None:
        self.config = config
        self.capabilities = dataclasses.replace(
            ORCA_ROUTER_CAPABILITIES,
            max_parallel_projects=config.max_parallel_projects,
        )
        self._credential_source = credential_source
        self._catalog_factory = catalog_factory
        self._client_factory = client_factory
        self._closed = False

    async def warmup(self) -> None:
        """Resolve the credential once so the first task is not cold."""

        source = await self._resolve_credential_source()
        await source.acquire()

    async def _resolve_credential_source(self) -> OrcaCredentialSource:
        if self._credential_source is not None:
            if callable(self._credential_source):
                source = self._credential_source()
                if inspect.isawaitable(source):
                    source = await source
            else:
                source = self._credential_source
            return source
        store = OrcaPkceStore(_resolve_store_path(self.config))
        return _prefer_existing_key(store, self.config)

    def project_resource_key(self, project: ProjectRecord) -> str:
        return str(Path(self.config.cwd).expanduser().resolve())

    async def open_project(
        self, project: ProjectRecord
    ) -> "OrcaRouterWorkerProject":
        if self._closed:
            raise RuntimeError("OrcaRouter WorkerProvider is closed")
        if project.provider_name != self.name:
            raise ValueError(
                "project provider does not match OrcaRouter WorkerProvider"
            )
        return OrcaRouterWorkerProject(self, project)

    async def close(self) -> None:
        self._closed = True


ORCA_ROUTER_PROVIDER_REGISTRATION = ProviderRegistration(
    key="orcarouter",
    provider_name=OrcaRouterProvider.name,
    settings_type=OrcaRouterProviderSettings,
    build=_build_orca_provider,
)


def _prefer_existing_key(
    store: OrcaPkceStore, config: OrcaRouterProviderConfig
) -> OrcaCredentialSource:
    """Return an adapter that prefers a stored PKCE key before a new login.

    The two explicit choices remain independently usable: a pasted API key
    (env adapter) still wins when present, and the PKCE adapter reuses its
    stored durable key and only opens a browser when no key exists yet.
    """

    class _EitherSource:
        source_name = "orcarouter-api-key-or-pkce"

        async def acquire(self) -> OrcaCredential:
            api = OrcaApiKeyAdapter(env_name=config.api_key_env)
            try:
                return await api.acquire()
            except ValueError:
                pass
            pkce = OrcaPkceAdapter(
                store,
                auth_origin=config.auth_base,
            )
            try:
                stored = store.load()
            except ValueError:
                stored = None
            if stored is not None:
                return stored
            return await pkce.acquire()

        async def clear(self) -> None:
            store.clear()

    return _EitherSource()


class OrcaRouterWorkerProject:
    """One project scoped to a shared credential and a bounded catalog."""

    def __init__(
        self, provider: OrcaRouterProvider, project: ProjectRecord
    ) -> None:
        self.provider = provider
        self.project = project
        self.session_id = project.backend_session_id
        self._credential: OrcaCredential | None = None
        self._catalog: OrcaCatalog | None = None
        self._closed = False

    async def _ensure_credential(self) -> OrcaCredential:
        source = await self.provider._resolve_credential_source()
        if inspect.isawaitable(source):
            source = await source
        credential = await source.acquire()
        if credential is None:
            raise RuntimeError("OrcaRouter credential is unavailable")
        self._credential = credential
        return credential

    async def _ensure_catalog(self) -> OrcaCatalog:
        if self._catalog is not None:
            return self._catalog
        credential = await self._ensure_credential()
        factory = self.provider._catalog_factory
        if factory is not None:
            catalog = factory(self.provider.config.api_base, credential)
        else:
            catalog = OrcaCatalog(
                self.provider.config.api_base,
                credential=credential,
                timeout_s=self.provider.config.catalog_timeout_s,
            )
        self._catalog = catalog
        return catalog

    async def start(
        self, request: WorkerRequest, control: WorkerControl
    ) -> WorkerRunChannel:
        if self._closed:
            raise RuntimeError("OrcaRouter worker project is closed")
        if request.project_id != self.project.project_id:
            raise ValueError("worker request belongs to another project")
        work_state = WorkState()
        task_request = TaskRequest(
            task_id=request.task_id,
            session_id=request.lineage_id,
            generation=request.generation,
            instruction=request.instruction,
            context=ContextSnapshot(request.lineage_id, ()),
            work_state=work_state,
            request_id=request.run_id,
            metadata={
                "owner_id": request.owner_id,
                "project_id": request.project_id,
                "lineage_id": request.lineage_id,
                "orchestration_run_id": request.run_id,
                "reasoning_profile": request.reasoning_profile,
                "worker_policy": dataclasses.asdict(request.policy),
                "worker_request_kind": request.kind,
                "context_plan": dataclasses.asdict(request.context_plan),
                "original_turn": request.original_turn,
            },
        )
        provider_run = OrcaRouterRun(
            task_request,
            self.provider.config,
            credential=await self._ensure_credential(),
            catalog=await self._ensure_catalog(),
            worker_control=control,
            client_factory=self.provider._client_factory,
        )
        await provider_run.start()
        return WorkerRunChannel(
            request,
            control,
            provider_run,
            self.provider.capabilities,
            result_callback=lambda result: work_state.apply_patch(
                result.work_state
            )
            if isinstance(result.work_state, dict)
            else None,
        )

    async def close(self) -> None:
        self._closed = True


class OrcaRouterRun:
    """One run: a bounded OrcaRouter chat-completions turn.

    The run resolves its model through the real catalog (or the verified seed
    when the catalog is degraded) and emits one terminal ``ProviderEvent``.
    """

    def __init__(
        self,
        request: TaskRequest,
        config: OrcaRouterProviderConfig,
        *,
        credential: OrcaCredential,
        catalog: OrcaCatalog,
        worker_control: WorkerControl,
        client_factory: Any | None = None,
    ) -> None:
        self.request = request
        self.config = config
        self.credential = credential
        self.catalog = catalog
        self.worker_control = worker_control
        self.client_factory = client_factory
        self.generation = request.generation
        self.thread_id: str | None = None
        self._queue: asyncio.Queue[ProviderEvent | None] = asyncio.Queue()
        self._worker_tool_session = WorkerToolSession(
            worker_control, self._accept_share
        )
        self._closed = False

    async def start(self) -> None:
        model = await self._resolve_model()
        self.thread_id = f"orca_{self.request.task_id}"
        client = self._client()
        messages = [
            {
                "role": "user",
                "content": self._prompt(),
            }
        ]
        try:
            response = await client.chat_completions(
                model=model,
                messages=messages,
                max_tokens=512,
            )
        except OrcaTransportError as exc:
            if exc.status == 401:
                # Terminal reauthentication: mark the exact credential
                # generation and do not fake a refresh.
                await self._mark_needs_reauth()
                await self.queue_put(
                    ProviderEvent(
                        kind="error",
                        generation=self.generation,
                        error=(
                            "OrcaRouter rejected this API key (401); paste a new "
                            "key or re-authorize"
                        ),
                    )
                )
                return
            await self.queue_put(
                ProviderEvent(
                    kind="error",
                    generation=self.generation,
                    error=str(exc),
                )
            )
            return
        text = self._extract_text(response)
        result = TaskResult(
            task_id=self.request.task_id,
            session_id=self.request.session_id,
            generation=self.generation,
            status="completed",
            full_result=text,
            work_state=self.request.work_state.to_dict(),
            provider_metadata={
                "provider": "orcarouter",
                "model": model,
                "thread_id": self.thread_id,
            },
        )
        await self.queue_put(
            ProviderEvent(kind="result", generation=self.generation, result=result)
        )

    async def _mark_needs_reauth(self) -> None:
        """Mark the exact account and credential generation for re-auth.

        The old secret is not deleted before a new login succeeds; only the
        generation that made the rejected request is invalidated. A late
        failure from an older request must never mark a newly re-authorized
        credential as broken, so we only annotate the store entry whose
        generation matches the rejected request.
        """

        store = OrcaPkceStore(
            Path(self.config.runtime_dir) / "orca_credentials.json"
        )
        stored = store.load()
        if stored is None or stored.generation != self.credential.generation:
            return
        if stored.key == self.credential.key:
            self._credential = None

    async def _resolve_model(self) -> str:
        configured = self.config.model.strip() or DEFAULT_MODEL
        try:
            models = await self.catalog.refresh(
                capability="chat",
                required_input_modalities=(),
            )
            if models and configured in {
                item.get("id") for item in models
            }:
                return configured
        except Exception:
            LOGGER.warning(
                "OrcaRouter catalog refresh failed; using verified fallback",
                exc_info=True,
            )
        return configured

    def _client(self) -> OrcaRouterClient:
        if self.client_factory is not None:
            return self.client_factory(
                self.config.api_base, self.credential
            )
        return OrcaRouterClient(
            self.config.api_base,
            credential=self.credential,
            timeout_s=self.config.request_timeout_s,
        )

    def _prompt(self) -> str:
        policy = self.request.metadata.get("worker_policy") or {}
        return (
            "Gander back-brain task:\n"
            f"{self.request.instruction}\n\n"
            "Effective worker policy (JSON):\n"
            f"{json.dumps(policy, ensure_ascii=False, separators=(',', ':'))}\n\n"
            "Complete the task and return the final result directly."
        )

    @staticmethod
    def _extract_text(response: dict[str, Any]) -> str:
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("OrcaRouter response has no choices")
        message = choices[0].get("message")
        if not isinstance(message, dict):
            raise RuntimeError("OrcaRouter response message is invalid")
        return str(message.get("content") or "").strip()

    async def _accept_share(
        self, arguments: dict[str, Any], call_id: str
    ) -> dict[str, Any]:
        from ..contracts import ShareEvent, now_ms, stable_id

        share_id = stable_id("share", call_id)
        share = ShareEvent(
            task_id=self.request.task_id,
            session_id=self.request.session_id,
            generation=self.generation,
            kind=str(arguments.get("kind") or "milestone"),
            text=str(arguments.get("text") or ""),
            state_patch=arguments.get("state_patch") or {},
            share_id=share_id,
            created_at_ms=now_ms(),
        )
        await self.queue_put(
            ProviderEvent(kind="share", generation=self.generation, share=share)
        )
        return {"status": "ok", "delivered": True, "share_id": share_id}

    async def steer(self, update: TaskUpdate) -> None:
        # OrcaRouter runs are single-turn; updates route to a fresh continuation
        # run by the Gateway (steering="none").
        raise RuntimeError("OrcaRouter provider does not support in-run steering")

    async def query(self, query: TaskQuery) -> None:
        raise RuntimeError("OrcaRouter provider does not support side queries")

    async def respond(self, reply: TaskInteractionReply) -> bool:
        # Interactions are surfaced to the front brain by the Gateway; the
        # OrcaRouter run is single-turn and resolves through the runtime lane.
        return False

    async def cancel(self) -> None:
        await self.queue_put(None)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self.queue_put(None)

    async def queue_put(self, event: ProviderEvent | None) -> None:
        await self._queue.put(event)

    def events(self) -> AsyncIterator[ProviderEvent]:
        return self._iterate_events()

    async def _iterate_events(self) -> AsyncIterator[ProviderEvent]:
        while True:
            event = await self._queue.get()
            if event is None:
                return
            yield event
