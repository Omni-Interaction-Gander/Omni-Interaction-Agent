from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class ProviderBuildContext:
    workspace: Path
    runtime_dir: Path
    runtime_profile: str


ProviderBuilder = Callable[[ProviderBuildContext, Any], Any]


@dataclass(frozen=True)
class ProviderRegistration:
    key: str
    provider_name: str
    settings_type: type[Any]
    build: ProviderBuilder


@dataclass(frozen=True)
class ConfiguredProviderFactory:
    registration: ProviderRegistration
    settings: Any

    @property
    def provider_name(self) -> str:
        return self.registration.provider_name

    def create(self, context: ProviderBuildContext) -> Any:
        provider = self.registration.build(context, self.settings)
        if getattr(provider, "name", None) != self.provider_name:
            raise TypeError(
                f"provider factory {self.registration.key!r} returned "
                f"{getattr(provider, 'name', None)!r}, expected {self.provider_name!r}"
            )
        return provider


class ProviderFactoryRegistry:
    """Maps public configuration keys to typed WorkerProvider factories."""

    def __init__(self) -> None:
        self._registrations: dict[str, ProviderRegistration] = {}

    def register(self, registration: ProviderRegistration) -> None:
        if not registration.key:
            raise ValueError("provider registration requires a key")
        if registration.key in self._registrations:
            raise ValueError(f"duplicate provider registration: {registration.key}")
        if not is_dataclass(registration.settings_type):
            raise TypeError("provider settings_type must be a dataclass")
        self._registrations[registration.key] = registration

    def configure(
        self, key: str, values: dict[str, Any]
    ) -> ConfiguredProviderFactory:
        try:
            registration = self._registrations[key]
        except KeyError:
            available = ", ".join(sorted(self._registrations)) or "none"
            raise ValueError(
                f"unknown worker.provider {key!r}; available: {available}"
            ) from None

        known = {item.name for item in fields(registration.settings_type) if item.init}
        unknown = sorted(set(values) - known)
        if unknown:
            raise ValueError(
                "Unknown setting(s): "
                + ", ".join(f"worker.settings.{name}" for name in unknown)
            )
        settings = registration.settings_type(**values)
        return ConfiguredProviderFactory(registration, settings)

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(sorted(self._registrations))
