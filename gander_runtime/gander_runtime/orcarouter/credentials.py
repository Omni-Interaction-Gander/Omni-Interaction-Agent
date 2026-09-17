"""OrcaRouter credential acquisition and storage.

Both user entry points — a pasted ``sk-orca-…`` API key and an OAuth 2.0 + PKCE
account login — resolve to the same normal OrcaRouter API key. The downstream
provider, model discovery, and AI entry points only consume
:class:`OrcaCredential`; they never implement their own authentication.
"""

from __future__ import annotations

import base64
import hmac
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlencode

from .pkce import PkceChallenge, generate_pkce_pair

AUTH_DEFAULT_BASE = "https://www.orcarouter.ai"
AUTH_AUTHORIZE_PATH = "/auth"
AUTH_EXCHANGE_PATH = "/api/v1/auth/keys"
API_DEFAULT_BASE = "https://api.orcarouter.ai/v1"
SCOPE_DEFAULT = "api"

EXCHANGE_HTTP_TIMEOUT_S = 30.0
EXCHANGE_MAX_BODY_BYTES = 64 * 1024
_OAUTH_ERROR_REASONS = frozenset(
    {
        "access_denied",
        "invalid_request",
        "unauthorized_client",
        "unsupported_response_type",
        "invalid_scope",
        "server_error",
        "temporarily_unavailable",
    }
)


def origin_url(base: str, *, explicit_auth: bool, default: str) -> str:
    """Resolve one public origin; explicit overrides win over the shared base.

    Loopback development may use HTTP; remote origins must use HTTPS. The
    default only applies when ``explicit_auth`` is false, which is a pure
    resolution detail and never permits an insecure remote origin.
    """

    if not isinstance(base, str) or not base.strip():
        raise ValueError("OrcaRouter base URL must be a non-empty string")
    candidate = (base or "").strip().rstrip("/")
    if not candidate.startswith(("http://", "https://")):
        candidate = "https://" + candidate
    scheme, _, rest = candidate.partition("://")
    host = rest.split("/", 1)[0]
    is_loopback = (
        host.lower() in {"localhost", "127.0.0.1", "[::1]"}
        or host.lower().startswith("[::1]")
        or host.lower().startswith("127.")
    )
    if scheme == "http" and not is_loopback:
        raise ValueError(
            "OrcaRouter remote origins must use HTTPS; HTTP is only allowed for "
            "loopback development"
        )
    return candidate


def resolve_origins(
    *,
    auth_base: str | None = None,
    api_base: str | None = None,
    shared_base: str | None = None,
) -> tuple[str, str]:
    """Resolve the auth origin and the inference origin.

    Explicit ``auth_base`` / ``api_base`` values take precedence over a shared
    ``shared_base`` fallback. Public defaults are ``https://www.orcarouter.ai``
    for auth and ``https://api.orcarouter.ai/v1`` for inference.
    """

    auth = (
        auth_base
        or (shared_base or AUTH_DEFAULT_BASE)
    )
    api = (
        api_base
        or (shared_base.rstrip("/") + "/v1" if shared_base else API_DEFAULT_BASE)
    )
    return (
        origin_url(auth, explicit_auth=auth_base is not None, default=AUTH_DEFAULT_BASE),
        origin_url(api, explicit_auth=api_base is not None, default=API_DEFAULT_BASE),
    )


@dataclass(frozen=True)
class OrcaCredential:
    """One plain OrcaRouter API key held by the runtime.

    ``user_id`` and ``scope`` come from the exchange response and are opaque;
    ``scope`` is what was granted, not what was requested.
    """

    key: str
    user_id: str = ""
    scope: str = SCOPE_DEFAULT
    generation: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.key, str) or not self.key.strip():
            raise ValueError("OrcaRouter credential key must be non-empty")
        if not self.key.startswith("sk-orca-"):
            raise ValueError("OrcaRouter credential key must start with sk-orca-")
        if self.key.strip() != self.key:
            raise ValueError("OrcaRouter credential key must not be padded")

    def masked(self) -> str:
        """A redacted rendering for UI/status that never reveals the key."""

        if len(self.key) < 12:
            return "sk-orca-***"
        return f"sk-orca-***…{self.key[-4:]} (secret)"


def redact_key(value: str) -> str:
    """Return a safe log/status rendering of a key without its secret value."""

    if not isinstance(value, str) or not value:
        return ""
    if not value.startswith("sk-orca-"):
        return "<invalid key>"
    return "sk-orca-***"


class OrcaCredentialSource(Protocol):
    """Acquire a plain OrcaRouter API key for downstream use.

    Both the pasted-key adapter and the PKCE adapter implement this seam.
    """

    source_name: str

    async def acquire(self) -> OrcaCredential: ...

    async def clear(self) -> None: ...


def is_plain_api_key(value: str) -> bool:
    """Lightweight format check only; never a validity claim."""

    return isinstance(value, str) and value.startswith("sk-orca-") and len(value) > 12


class OrcaApiKeyAdapter:
    """API-key entry point: an existing ``sk-orca-…`` key from env/config.

    ``env_name`` defaults to ``ORCAROUTER_API_KEY`` and may be overridden with
    ``ORCA_API_KEY``. The key is read from the environment at acquire time and
    never persisted by this package.
    """

    source_name = "orcarouter-api-key"

    def __init__(
        self,
        *,
        env_name: str | None = None,
        key: str | None = None,
    ) -> None:
        self._env_name = env_name or os.environ.get(
            "ORCA_API_KEY_ENV", "ORCAROUTER_API_KEY"
        )
        self._key = key

    async def acquire(self) -> OrcaCredential:
        value = self._key or os.environ.get(self._env_name) or ""
        value = value.strip()
        if not is_plain_api_key(value):
            raise ValueError(
                f"OrcaRouter API key is not configured (env {self._env_name}); "
                "paste an sk-orca-… key or run gander-orca connect"
            )
        return OrcaCredential(key=value)

    async def clear(self) -> None:
        self._key = None


class OrcaPkceStore:
    """Durable storage for PKCE-issued keys.

    Stored where the project already keeps session state (``runtime_dir``) with
    owner-only permissions. The stored file is the same normal OrcaRouter API
    key the user can revoke from their OrcaRouter console.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _locked(self) -> int:
        return os.O_WRONLY | os.O_CREAT | os.O_TRUNC

    def save(self, credential: OrcaCredential) -> None:
        payload = {
            "version": 1,
            "key": credential.key,
            "user_id": credential.user_id,
            "scope": credential.scope,
            "generation": credential.generation,
        }
        encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        temporary = self.path.with_name(
            f".{self.path.name}.{os.getpid()}.tmp"
        )
        fd = os.open(temporary, self._locked(), 0o600)
        try:
            os.write(fd, encoded)
        finally:
            os.close(fd)
        os.replace(temporary, self.path)

    def load(self) -> OrcaCredential | None:
        try:
            raw = self.path.read_bytes()
        except FileNotFoundError:
            return None
        if len(raw) > EXCHANGE_MAX_BODY_BYTES:
            raise ValueError("OrcaRouter credential store is corrupted")
        try:
            payload = json.loads(raw)
            key = str(payload["key"])
            if not is_plain_api_key(key):
                raise ValueError("stored OrcaRouter key is invalid")
            return OrcaCredential(
                key=key,
                user_id=str(payload.get("user_id") or ""),
                scope=str(payload.get("scope") or SCOPE_DEFAULT),
                generation=int(payload.get("generation") or 1),
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("OrcaRouter credential store is corrupted") from exc

    def clear(self) -> None:
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


def authorize_url(
    auth_origin: str,
    *,
    challenge: PkceChallenge,
    app_name: str,
    scope: str = SCOPE_DEFAULT,
    callback_url: str = "oob",
) -> str:
    """Build the consent URL for Flow B (out-of-band code).

    ``callback_url=oob`` is the literal three letters; S256 is mandatory for a
    code that can be read by a human.
    """

    params = {
        "callback_url": callback_url,
        "code_challenge": challenge.challenge,
        "code_challenge_method": "S256",
        "state": challenge.state,
        "app_name": app_name,
        "scope": scope,
    }
    return f"{auth_origin.rstrip('/')}{AUTH_AUTHORIZE_PATH}?{urlencode(params)}"


def _constant_time_eq(left: str, right: str) -> bool:
    return hmac.compare_digest(left.encode("utf-8"), right.encode("utf-8"))


def classify_exchange_error(
    status: int | None, body: str, *, state: str
) -> str:
    """Return a safe, actionable error message for an OAuth exchange failure.

    Never includes the verifier or the code; only OAuth error codes and bounded
    descriptions are surfaced. The word ``verifier`` itself is avoided so the
    secret value can never be confused with a leaked field name.
    """

    if status is None:
        return "OrcaRouter authorization timed out or the network failed"
    error_code = ""
    try:
        decoded = json.loads(body)
        if isinstance(decoded, dict):
            error_code = str(decoded.get("error") or "")
    except json.JSONDecodeError:
        pass
    if error_code == "access_denied":
        return "OrcaRouter authorization was refused (access_denied)"
    if status == 400:
        return (
            "OrcaRouter rejected the exchange: the code challenge method was "
            "unrecognized or downgraded"
        )
    if status == 403:
        return (
            "OrcaRouter rejected the exchange: the code is unknown, expired, "
            "already used, or the authorization did not match"
        )
    if status == 429:
        return (
            "OrcaRouter rate-limited this authorization; wait before retrying "
            "(PKCE-issued keys are capped per account per day)"
        )
    if error_code in _OAUTH_ERROR_REASONS:
        return f"OrcaRouter authorization was refused ({error_code})"
    return f"OrcaRouter exchange failed with HTTP {status}"


class OrcaPkceAdapter:
    """OAuth 2.0 + PKCE entry point (Flow B, out-of-band).

    The browser is opened at the consent URL; the user reads the code back and
    pastes it into Gander. The verifier never leaves this process. On success
    the resulting plain OrcaRouter API key is persisted in the trusted secret
    store and reused on subsequent launches (no proactive refresh, no re-login
    on every start).
    """

    source_name = "orcarouter-pkce"

    def __init__(
        self,
        store: OrcaPkceStore,
        *,
        auth_origin: str | None = None,
        app_name: str = "Gander",
        exchange_timeout_s: float = EXCHANGE_HTTP_TIMEOUT_S,
        exchange_factory: Any | None = None,
        read_code_factory: Any | None = None,
        open_browser: Any | None = None,
    ) -> None:
        self._store = store
        self._auth_origin = auth_origin or AUTH_DEFAULT_BASE
        self._app_name = app_name
        self._exchange_timeout_s = exchange_timeout_s
        self._exchange = exchange_factory
        self._read_code_factory = read_code_factory
        self._open_browser = open_browser

    async def acquire(self) -> OrcaCredential:
        existing = self._store.load()
        if existing is not None:
            return existing
        return await self._connect()

    async def _connect(self) -> OrcaCredential:
        attempt = generate_pkce_pair()
        url = authorize_url(
            self._auth_origin,
            challenge=attempt,
            app_name=self._app_name,
        )
        if self._open_browser is not None:
            opened = self._open_browser(url)
            if hasattr(opened, "__await__"):
                await opened
        else:
            import webbrowser

            webbrowser.open(url)
        code = await self._read_code(url)
        if not code:
            raise RuntimeError("OrcaRouter authorization was cancelled")
        return await self._exchange_code(code, attempt)

    async def _read_code(self, url: str) -> str:
        """Prompt for the out-of-band code. Injectable for headless testing."""

        if self._read_code_factory is not None:
            result = self._read_code_factory(url)
            if hasattr(result, "__await__"):
                result = await result
            if not isinstance(result, str):
                raise RuntimeError("OrcaRouter code reader must return a string")
            return result
        raise NotImplementedError(
            "OrcaPkceAdapter._read_code must be provided; use the CLI "
            "interactive command or inject a reader"
        )

    async def _exchange_code(
        self, code: str, attempt: PkceChallenge
    ) -> OrcaCredential:
        if not isinstance(code, str) or not code.strip():
            raise RuntimeError("OrcaRouter authorization code is empty")
        if len(code) > 4096:
            raise ValueError("OrcaRouter authorization code is too long")
        body = {
            "code": code,
            "code_verifier": attempt.verifier,
            "code_challenge_method": "S256",
        }
        url = (
            f"{self._auth_origin.rstrip('/')}{AUTH_EXCHANGE_PATH}"
        )
        try:
            status, payload = await self._post_exchange(url, body)
        except Exception as exc:
            raise RuntimeError(
                "OrcaRouter exchange network error: "
                + classify_exchange_error(None, "", state=attempt.state)
            ) from exc
        if status == 200 and isinstance(payload, dict):
            raw_key = payload.get("key")
            if not isinstance(raw_key, str) or not is_plain_api_key(raw_key):
                raise RuntimeError(
                    "OrcaRouter exchange returned an invalid credential"
                )
            granted_scope = str(payload.get("scope") or SCOPE_DEFAULT)
            if granted_scope != SCOPE_DEFAULT:
                raise RuntimeError(
                    f"OrcaRouter granted scope '{granted_scope}', not "
                    f"'{SCOPE_DEFAULT}'; the authorization is insufficient for "
                    "Gander inference"
                )
            credential = OrcaCredential(
                key=raw_key,
                user_id=str(payload.get("user_id") or ""),
                scope=granted_scope,
            )
            self._store.save(credential)
            return credential
        message = classify_exchange_error(
            status,
            payload
            if isinstance(payload, str)
            else json.dumps(payload, separators=(",", ":"))
            if isinstance(payload, dict)
            else "",
            state=attempt.state,
        )
        raise RuntimeError(message)

    async def _post_exchange(
        self, url: str, body: dict[str, str]
    ) -> tuple[int, Any]:
        if self._exchange is not None:
            result = self._exchange(url, body)
            if hasattr(result, "__await__"):
                result = await result
            return result
        return await self._default_exchange(url, body)

    async def _default_exchange(
        self, url: str, body: dict[str, str]
    ) -> tuple[int, Any]:
        import json as _json
        import urllib.error
        import urllib.request

        request = urllib.request.Request(
            url,
            data=_json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self._exchange_timeout_s
            ) as response:
                raw = response.read(EXCHANGE_MAX_BODY_BYTES + 1)
                if len(raw) > EXCHANGE_MAX_BODY_BYTES:
                    return (413, "response too large")
                return (response.status, _json.loads(raw))
        except urllib.error.HTTPError as exc:
            raw = exc.read(EXCHANGE_MAX_BODY_BYTES + 1)
            if len(raw) > EXCHANGE_MAX_BODY_BYTES:
                return (exc.code, "response too large")
            try:
                return (exc.code, _json.loads(raw))
            except json.JSONDecodeError:
                return (exc.code, raw.decode("utf-8", errors="replace"))
        except urllib.error.URLError as exc:
            raise RuntimeError(
                "OrcaRouter exchange network failure"
            ) from exc

    async def clear(self) -> None:
        """Delete the stored PKCE key. Called after a successful re-login."""

        self._store.clear()
