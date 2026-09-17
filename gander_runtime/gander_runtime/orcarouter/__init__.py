"""OrcaRouter: first-class AI gateway provider with API-key and PKCE auth.

OrcaRouter is an OpenAI-compatible AI gateway built for both models and agents.
This package provides a credential interface with two adapters (paste an
``sk-orca-…`` API key, or authorize an OrcaRouter account via OAuth 2.0 + PKCE),
real model discovery against ``https://api.orcarouter.ai/v1``, a verified
cold-start catalog, and a ``WorkerProvider`` that routes Gander back-brain tasks
to the OrcaRouter inference relay.

Auth and inference use separate public origins. Authentication (and the code
exchange) lives on ``https://www.orcarouter.ai``; inference and model discovery
live on ``https://api.orcarouter.ai/v1``. The two must never be conflated.
"""

from .credentials import (
    OrcaApiKeyAdapter,
    OrcaCredential,
    OrcaCredentialSource,
    OrcaPkceAdapter,
    OrcaPkceStore,
)
from .catalog import (
    CATALOG_ENDPOINT_TYPES,
    CATALOG_MAX_ITEMS,
    OrcaCatalog,
    VerifiedModelSeed,
    capability_filter,
)
from .client import OrcaRouterClient, OrcaTransportError
from .pkce import PkceChallenge, generate_pkce_pair, hash_verifier_challenge
from .provider import OrcaRouterProvider, OrcaRouterProviderSettings

__all__ = [
    "OrcaApiKeyAdapter",
    "OrcaCredential",
    "OrcaCredentialSource",
    "OrcaPkceAdapter",
    "OrcaPkceStore",
    "CATALOG_ENDPOINT_TYPES",
    "CATALOG_MAX_ITEMS",
    "OrcaCatalog",
    "VerifiedModelSeed",
    "capability_filter",
    "OrcaRouterClient",
    "OrcaTransportError",
    "PkceChallenge",
    "generate_pkce_pair",
    "hash_verifier_challenge",
    "OrcaRouterProvider",
    "OrcaRouterProviderSettings",
]
