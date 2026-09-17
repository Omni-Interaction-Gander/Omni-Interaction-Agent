"""Worker provider implementations and their configuration registry."""

from .registry import (
    ProviderBuildContext,
    ProviderFactoryRegistry,
    ProviderRegistration,
)


def builtin_provider_registry() -> ProviderFactoryRegistry:
    from .codex import CODEX_PROVIDER_REGISTRATION
    from ..orcarouter.provider import ORCA_ROUTER_PROVIDER_REGISTRATION

    registry = ProviderFactoryRegistry()
    registry.register(CODEX_PROVIDER_REGISTRATION)
    registry.register(ORCA_ROUTER_PROVIDER_REGISTRATION)
    return registry


__all__ = [
    "ProviderBuildContext",
    "ProviderFactoryRegistry",
    "ProviderRegistration",
    "builtin_provider_registry",
]
