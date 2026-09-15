"""PKCE helpers (RFC 7636) for the OrcaRouter OAuth 2.0 out-of-band flow.

The verifier is high-entropy random data that must never leave this process.
Only its unpadded ``base64url(sha256(verifier))`` challenge is sent to the
authorize endpoint; the verifier is presented at exchange time.
"""

from __future__ import annotations

import base64
import hashlib
import secrets
from dataclasses import dataclass

VERIFIER_BYTES = 32
STATE_BYTES = 16
CHALLENGE_METHOD = "S256"


def base64url_unpadded(raw: bytes) -> str:
    """RFC 4648 §5 base64url without padding."""

    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def hash_verifier_challenge(verifier: str) -> str:
    """Return ``base64url(sha256(verifier))`` with no padding."""

    digest = hashlib.sha256(verifier.encode("utf-8")).digest()
    return base64url_unpadded(digest)


def generate_pkce_pair() -> "PkceChallenge":
    """Create one fresh PKCE attempt: verifier + state from a crypto RNG."""

    verifier = base64url_unpadded(secrets.token_bytes(VERIFIER_BYTES))
    state = base64url_unpadded(secrets.token_bytes(STATE_BYTES))
    if not verifier or not state:
        raise RuntimeError("PKCE randomness generation failed")
    return PkceChallenge(
        verifier=verifier,
        state=state,
        challenge=hash_verifier_challenge(verifier),
    )


@dataclass(frozen=True)
class PkceChallenge:
    """One PKCE attempt. The verifier is process-private until exchange."""

    verifier: str
    state: str
    challenge: str

    def __post_init__(self) -> None:
        if not self.verifier or not self.state or not self.challenge:
            raise ValueError("PKCE attempt requires verifier, state, and challenge")
        if len(self.verifier) < 43:
            raise ValueError("PKCE verifier is too short")
        if self.challenge != hash_verifier_challenge(self.verifier):
            raise ValueError("PKCE challenge does not match the verifier")
