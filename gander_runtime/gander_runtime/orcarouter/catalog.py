"""Real model discovery for OrcaRouter.

The single source of truth for the model list is ``GET {api_base}/models`` on
the configured inference origin. Live discovery is authoritative; when it fails,
a small, verified cold-start seed keeps a fresh installation usable with its
context / input-modality / reasoning metadata intact (never a free-text input).
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

CHAT_ENDPOINT_TYPES = ("openai", "openai-response", "anthropic", "gemini")
CATALOG_ENDPOINT_TYPES = CHAT_ENDPOINT_TYPES
# Bounds so a catalog response cannot consume unbounded memory or advertise
# routes the client cannot speak.
CATALOG_MAX_ITEMS = 1024
CATALOG_MAX_BODY_BYTES = 8 * 1024 * 1024
CATALOG_TIMEOUT_S = 20.0

# Verified against the live official catalog
# (https://api.orcarouter.ai/v1/models) on 2026-09-14/15. These are
# cold-start/outage fallbacks only; live discovery always takes precedence.
VERIFIED_SEED_MODELS: tuple[dict[str, Any], ...] = (
    {
        "id": "openai/gpt-5.5",
        "name": "OpenAI: GPT-5.5",
        "context_length": 1_000_000,
        "max_completion_tokens": 128_000,
        "supported_endpoint_types": ["openai", "openai-response"],
        "architecture": {"input_modalities": ["file", "image", "text"]},
        "reasoning": {"effort_ladder": ["low", "medium", "high", "xhigh"]},
    },
    {
        "id": "anthropic/claude-opus-4.8",
        "name": "Anthropic: Claude Opus 4.8",
        "context_length": 1_000_000,
        "max_completion_tokens": 128_000,
        "supported_endpoint_types": ["openai", "anthropic", "openai-response"],
        "architecture": {
            "input_modalities": ["text", "image", "file"],
            "output_modalities": ["text"],
        },
        "reasoning": {"effort_ladder": ["low", "medium", "high", "xhigh"]},
    },
    {
        "id": "google/gemini-3.5-flash",
        "name": "Gemini 3.5 Flash",
        "context_length": 1_048_576,
        "max_completion_tokens": 65_536,
        "supported_endpoint_types": ["openai", "gemini"],
        "architecture": {
            "input_modalities": ["text", "image", "video", "file", "audio"],
            "output_modalities": ["text"],
        },
        "reasoning": {"effort_ladder": ["low", "medium", "high", "xhigh"]},
    },
    {
        "id": "deepseek/deepseek-v4-pro",
        "name": "DeepSeek: DeepSeek V4 Pro",
        "context_length": 1_048_576,
        "max_completion_tokens": 384_000,
        "supported_endpoint_types": ["openai", "openai-response"],
        "architecture": {"input_modalities": ["text"]},
        "reasoning": {"effort_ladder": ["low", "medium", "high", "xhigh"]},
    },
    {
        "id": "orcarouter/auto",
        "name": "OrcaRouter: Auto",
        "context_length": None,
        "max_completion_tokens": None,
        "supported_endpoint_types": [
            "openai",
            "openai-response",
            "anthropic",
            "gemini",
        ],
        "architecture": None,
        "reasoning": None,
    },
)


@dataclass(frozen=True)
class VerifiedModelSeed:
    """One verified fallback model with its metadata intact."""

    id: str
    name: str = ""
    context_length: int | None = None
    max_completion_tokens: int | None = None
    endpoint_types: tuple[str, ...] = ()
    input_modalities: tuple[str, ...] = ()
    reasoning: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_raw(cls, raw: dict[str, Any]) -> "VerifiedModelSeed":
        architecture = raw.get("architecture") or {}
        endpoint_types = tuple(
            item
            for item in (raw.get("supported_endpoint_types") or ())
            if isinstance(item, str)
        )
        input_modalities = tuple(
            item
            for item in (architecture.get("input_modalities") or ())
            if isinstance(item, str)
        )
        reasoning = raw.get("reasoning")
        if not isinstance(reasoning, dict):
            reasoning = {}
        context_length = raw.get("context_length")
        max_completion_tokens = raw.get("max_completion_tokens")
        return cls(
            id=str(raw["id"]),
            name=str(raw.get("name") or raw["id"]),
            context_length=(
                int(context_length) if isinstance(context_length, int) else None
            ),
            max_completion_tokens=(
                int(max_completion_tokens)
                if isinstance(max_completion_tokens, int)
                else None
            ),
            endpoint_types=endpoint_types,
            input_modalities=input_modalities,
            reasoning=reasoning,
        )

    def to_dict(self) -> dict[str, Any]:
        item: dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "supported_endpoint_types": list(self.endpoint_types),
        }
        if self.context_length is not None:
            item["context_length"] = self.context_length
        if self.max_completion_tokens is not None:
            item["max_completion_tokens"] = self.max_completion_tokens
        architecture: dict[str, Any] = {}
        if self.input_modalities:
            architecture["input_modalities"] = list(self.input_modalities)
        if architecture:
            item["architecture"] = architecture
        if self.reasoning:
            item["reasoning"] = self.reasoning
        return item


def capability_filter(
    models: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    *,
    capability: str,
    required_input_modalities: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    """Filter a catalog for one AI entry point.

    - ``chat`` keeps models whose ``supported_endpoint_types`` intersect the
      OpenAI-compatible set and never advertises image-generation / video /
      rerank-only routes.
    - Additional ``required_input_modalities`` are enforced fail-closed: a model
      that does not explicitly declare the input modality is excluded.
    """

    output: list[dict[str, Any]] = []
    for model in models:
        if not isinstance(model, dict):
            continue
        model_id = model.get("id")
        if not isinstance(model_id, str) or not model_id:
            continue
        endpoint_types = set(
            item
            for item in (model.get("supported_endpoint_types") or ())
            if isinstance(item, str)
        )
        if capability == "embedding":
            if "embeddings" not in endpoint_types:
                continue
        elif capability == "image":
            if "image-generation" not in endpoint_types:
                continue
        elif capability == "video":
            if "openai-video" not in endpoint_types:
                continue
        elif capability == "rerank":
            if "jina-rerank" not in endpoint_types:
                continue
        else:  # chat and generic text
            if not endpoint_types & set(CHAT_ENDPOINT_TYPES):
                continue
            if endpoint_types & {"image-generation", "openai-video", "jina-rerank"}:
                continue
        if required_input_modalities:
            declared = set(
                item
                for item in ((model.get("architecture") or {}).get(
                    "input_modalities"
                ) or ())
                if isinstance(item, str)
            )
            if not declared >= set(required_input_modalities):
                # Fail closed: an un-declared modality is not assumed.
                continue
        output.append(model)
    return output


def _json_mapping(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("OrcaRouter catalog response must be an object")
    return value


class OrcaCatalog:
    """Bounded, live-first model catalog for the configured inference origin.

    ``refresh`` fetches ``GET {api_base}/models[?capability=…]`` and keeps a
    last-known-good copy. When the live request fails, the verified seed is
    returned with ``degraded=True`` so the UI can surface the fallback state.
    """

    def __init__(
        self,
        api_base: str,
        *,
        credential: OrcaCredential | None = None,
        fetch_factory: Any | None = None,
        timeout_s: float = CATALOG_TIMEOUT_S,
        max_items: int = CATALOG_MAX_ITEMS,
    ) -> None:
        if max_items < 1:
            raise ValueError("OrcaRouter catalog max_items must be positive")
        self.api_base = api_base.rstrip("/")
        self._credential = credential
        self._fetch = fetch_factory
        self._timeout_s = timeout_s
        self._max_items = max_items
        self._live: list[dict[str, Any]] = []
        self._degraded = False

    @property
    def degraded(self) -> bool:
        return self._degraded

    def last_known_good(self) -> list[dict[str, Any]]:
        return list(self._live)

    def seed(self) -> list[dict[str, Any]]:
        return [
            VerifiedModelSeed.from_raw(raw).to_dict()
            for raw in VERIFIED_SEED_MODELS
        ]

    async def refresh(
        self,
        *,
        capability: str = "chat",
        required_input_modalities: tuple[str, ...] = (),
    ) -> list[dict[str, Any]]:
        """Fetch the live catalog and return the filtered list.

        Live success is authoritative and stored as last-known-good. On failure
        the verified seed is returned with ``degraded=True``.
        """

        url = f"{self.api_base}/models"
        query: list[tuple[str, str]] = []
        if capability in {"chat", "embedding", "image", "video", "rerank"}:
            query.append(("capability", capability))
        if query:
            from urllib.parse import urlencode

            url = f"{url}?{urlencode(query)}"
        try:
            raw = await self._fetch_catalog(url)
        except Exception:
            self._degraded = True
            return self.seed()
        self._degraded = False
        parsed = _json_mapping(raw)
        items = parsed.get("data")
        if not isinstance(items, list):
            self._degraded = True
            return self.seed()
        bounded = items[: self._max_items]
        self._live = list(bounded)
        return capability_filter(
            bounded,
            capability=capability,
            required_input_modalities=required_input_modalities,
        )

    async def _fetch_catalog(self, url: str) -> Any:
        if self._fetch is not None:
            result = self._fetch(url)
            if hasattr(result, "__await__"):
                result = await result
            return result
        headers = {"Accept": "application/json"}
        if self._credential is not None:
            headers["Authorization"] = f"Bearer {self._credential.key}"
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(
                request, timeout=self._timeout_s
            ) as response:
                raw = response.read(CATALOG_MAX_BODY_BYTES + 1)
                if len(raw) > CATALOG_MAX_BODY_BYTES:
                    raise ValueError("OrcaRouter catalog response exceeds limit")
                return json.loads(raw)
        except urllib.error.HTTPError as exc:
            if exc.code == 401:
                raise RuntimeError(
                    "OrcaRouter model catalog requires a valid API key (401)"
                ) from exc
            raise RuntimeError(
                f"OrcaRouter model catalog failed with HTTP {exc.code}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"OrcaRouter model catalog is unavailable: {exc.reason}"
            ) from exc
