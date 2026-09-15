"""CLI commands for the OrcaRouter provider.

- ``gander-orca connect`` — Flow B out-of-band PKCE login. Opens the browser,
  accepts the pasted code, exchanges it, and persists the durable key.
- ``gander-orca models`` — live model discovery with capability filtering.
- ``gander-orca check`` — validate the configured credential and origins.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from .credentials import (
    AUTH_DEFAULT_BASE,
    OrcaApiKeyAdapter,
    OrcaPkceAdapter,
    OrcaPkceStore,
    resolve_origins,
)
from .catalog import OrcaCatalog, VerifiedModelSeed
from .pkce import generate_pkce_pair


async def _run_connect(
    *,
    auth_base: str | None,
    shared_base: str | None,
    store_path: str | None,
    app_name: str,
    code: str | None,
) -> int:
    auth_origin, _ = resolve_origins(
        auth_base=auth_base, shared_base=shared_base
    )
    store = OrcaPkceStore(
        Path(store_path) if store_path else Path.cwd() / "orca_credentials.json"
    )
    if code:
        attempt = generate_pkce_pair()
        try:
            adapter = OrcaPkceAdapter(
                store,
                auth_origin=auth_origin,
                app_name=app_name,
                read_code_factory=_prompt_code,
            )
            credential = await adapter._exchange_code(code, attempt)
        except RuntimeError as exc:
            print(f"OrcaRouter connect failed: {exc}", file=sys.stderr)
            return 1
        print(f"Connected as OrcaRouter user {credential.user_id}.")
        print(f"Stored a durable API key at {store.path}")
        return 0
    try:
        adapter = OrcaPkceAdapter(
            store,
            auth_origin=auth_origin,
            app_name=app_name,
            read_code_factory=_prompt_code,
        )
        credential = await adapter.acquire()
    except RuntimeError as exc:
        print(f"OrcaRouter connect failed: {exc}", file=sys.stderr)
        return 1
    print(f"Connected as OrcaRouter user {credential.user_id}.")
    print(f"Stored a durable API key at {store.path}")
    return 0


def _prompt_code(url: str) -> str:
    """Read the out-of-band code from the terminal after opening the browser."""

    print("Authorize in your browser, then paste the code back here:", flush=True)
    print(url, flush=True)
    import sys as _sys

    value = _sys.stdin.readline()
    return value.strip()


async def _run_models(
    *,
    api_base: str | None,
    shared_base: str | None,
    capability: str,
    api_key_env: str,
    json_output: bool,
) -> int:
    _, api_origin = resolve_origins(api_base=api_base, shared_base=shared_base)
    credential = None
    try:
        credential = await OrcaApiKeyAdapter(env_name=api_key_env).acquire()
    except ValueError:
        credential = None
    catalog = OrcaCatalog(api_origin, credential=credential)
    models = await catalog.refresh(capability=capability)
    degraded = catalog.degraded
    if json_output:
        payload = {
            "origin": api_origin,
            "capability": capability,
            "degraded": degraded,
            "catalog_source": f"{api_origin}/models?capability={capability}",
            "models": models,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    if degraded:
        print(
            f"live catalog unavailable for {capability}; showing verified "
            "fallback models (degraded)",
            file=sys.stderr,
        )
    for model in models:
        print(model.get("id"))
    return 0


async def _run_check(
    *,
    auth_base: str | None,
    api_base: str | None,
    shared_base: str | None,
    api_key_env: str,
) -> int:
    auth_origin, api_origin = resolve_origins(
        auth_base=auth_base, api_base=api_base, shared_base=shared_base
    )
    print(f"auth origin:      {auth_origin}")
    print(f"inference origin: {api_origin}")
    try:
        await OrcaApiKeyAdapter(env_name=api_key_env).acquire()
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"API key configured via {api_key_env}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="gander-orca",
        description="OrcaRouter provider commands for Gander.",
    )
    parser.add_argument(
        "--auth-base",
        default=None,
        help=f"auth origin override (default {AUTH_DEFAULT_BASE})",
    )
    parser.add_argument("--api-base", default=None, help="inference origin override")
    parser.add_argument(
        "--shared-base", default=None, help="shared self-hosted base override"
    )
    parser.add_argument(
        "--api-key-env", default="ORCAROUTER_API_KEY", help="API key env name"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    connect = sub.add_parser("connect", help="OAuth 2.0 + PKCE login")
    connect.add_argument("--store-path", default=None, help="credential store path")
    connect.add_argument("--app-name", default="Gander", help="consent screen name")
    connect.add_argument("--code", default=None, help="paste the out-of-band code")

    models = sub.add_parser("models", help="list models from the live catalog")
    models.add_argument(
        "--capability",
        default="chat",
        choices=["chat", "embedding", "image", "video", "rerank"],
    )
    models.add_argument("--json", action="store_true", dest="json_output")

    check = sub.add_parser("check", help="validate credential and origins")

    args = parser.parse_args(argv)
    if args.command == "connect":
        return asyncio.run(
            _run_connect(
                auth_base=args.auth_base,
                shared_base=args.shared_base,
                store_path=args.store_path,
                app_name=args.app_name,
                code=args.code,
            )
        )
    if args.command == "models":
        return asyncio.run(
            _run_models(
                api_base=args.api_base,
                shared_base=args.shared_base,
                capability=args.capability,
                api_key_env=args.api_key_env,
                json_output=args.json_output,
            )
        )
    if args.command == "check":
        return asyncio.run(
            _run_check(
                auth_base=args.auth_base,
                api_base=args.api_base,
                shared_base=args.shared_base,
                api_key_env=args.api_key_env,
            )
        )
    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
