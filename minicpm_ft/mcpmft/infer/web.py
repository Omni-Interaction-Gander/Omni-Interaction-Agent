from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

STATIC_DIR = Path(__file__).with_name("static")


def request_json(
    url: str,
    *,
    data: bytes | None = None,
    headers: dict[str, str] | None = None,
    timeout: float,
) -> dict[str, Any]:
    request = urllib.request.Request(url, data=data, headers=headers or {})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"ASR upstream returned HTTP {exc.code}: {detail[:300]}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"ASR upstream is unavailable: {exc.reason}") from exc
