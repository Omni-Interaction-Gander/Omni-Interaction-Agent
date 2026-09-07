from __future__ import annotations

import json
import logging
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from types import TracebackType
from typing import Literal
from urllib.parse import urlparse

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class AsrConfig:
    mode: Literal["managed", "external", "disabled"] = "disabled"
    url: str | None = None
    model_path: str | None = None
    host: str = "127.0.0.1"
    port: int = 8995
    cuda_visible_devices: str | None = None
    device: Literal["auto", "cuda", "cpu"] = "cuda"
    device_index: int = 0
    compute_type: str = "float16"
    beam_size: int = 3
    vad_threshold: float = 0.6
    min_frame_rms: float = 0.0035
    min_peak: float = 0.01
    min_active_speech_ms: int = 210
    startup_timeout_sec: float = 300.0
    request_timeout_sec: float = 120.0
    log_level: str = "info"


def asr_base_url(config: AsrConfig) -> str | None:
    if config.mode == "disabled":
        return None
    if config.mode == "external":
        assert config.url is not None
        return config.url.rstrip("/")
    return f"http://{config.host}:{config.port}"


def validate_asr_config(config: AsrConfig) -> None:
    if config.mode not in {"managed", "external", "disabled"}:
        raise ValueError("asr.mode must be 'managed', 'external', or 'disabled'")
    if config.startup_timeout_sec <= 0 or config.request_timeout_sec <= 0:
        raise ValueError("ASR timeouts must be positive")
    if config.mode == "disabled":
        if config.url:
            raise ValueError("asr.url is invalid when asr.mode is disabled")
        return
    if config.mode == "external":
        if not config.url:
            raise ValueError("asr.url is required when asr.mode is external")
        parsed = urlparse(config.url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("asr.url must be an absolute HTTP(S) URL")
        return
    if config.url:
        raise ValueError("managed ASR derives its URL from asr.host and asr.port")
    if config.host not in {"127.0.0.1", "localhost"}:
        raise ValueError("managed ASR must bind to localhost")
    if not 1 <= config.port <= 65535:
        raise ValueError("asr.port must be between 1 and 65535")
    if not config.model_path:
        raise ValueError("asr.model_path is required when asr.mode is managed")
    if config.device_index < 0:
        raise ValueError("asr.device_index must not be negative")
    if config.beam_size < 1:
        raise ValueError("asr.beam_size must be positive")
    if not 0 < config.vad_threshold < 1:
        raise ValueError("asr.vad_threshold must be between zero and one")
    if config.min_frame_rms < 0 or config.min_peak < 0:
        raise ValueError("ASR audio thresholds must not be negative")
    if config.min_active_speech_ms < 1:
        raise ValueError("asr.min_active_speech_ms must be positive")


class AsrService:
    def __init__(self, config: AsrConfig) -> None:
        validate_asr_config(config)
        self.config = config
        self.process: subprocess.Popen[bytes] | None = None
        self._closing = threading.Event()
        self._watcher: threading.Thread | None = None

    def __enter__(self) -> "AsrService":
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc, traceback
        self.close()

    def start(self) -> None:
        if self.config.mode == "disabled":
            LOGGER.info("ASR is disabled")
            return
        if self.config.mode == "external":
            self._wait_ready()
            LOGGER.info("External ASR is ready at %s", asr_base_url(self.config))
            return
        if _port_is_open(self.config.host, self.config.port):
            raise RuntimeError(
                f"managed ASR address {self.config.host}:{self.config.port} is already "
                "in use; stop that process or select asr.mode=external explicitly"
            )
        env = os.environ.copy()
        if self.config.cuda_visible_devices is not None:
            env["CUDA_VISIBLE_DEVICES"] = self.config.cuda_visible_devices
        command = _managed_command(self.config)
        LOGGER.info("Starting managed ASR on %s", asr_base_url(self.config))
        self.process = subprocess.Popen(command, env=env)
        try:
            self._wait_ready()
        except BaseException:
            self.close()
            raise
        self._watcher = threading.Thread(
            target=self._watch_process,
            name="gander-asr-watch",
            daemon=True,
        )
        self._watcher.start()
        LOGGER.info("Managed ASR is ready at %s", asr_base_url(self.config))

    def close(self) -> None:
        self._closing.set()
        process = self.process
        self.process = None
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()

    def _wait_ready(self) -> None:
        url = f"{asr_base_url(self.config)}/health"
        deadline = time.monotonic() + self.config.startup_timeout_sec
        last_error = "not ready"
        while time.monotonic() < deadline:
            process = self.process
            if process is not None and process.poll() is not None:
                raise RuntimeError(
                    f"managed ASR exited during startup with code {process.returncode}"
                )
            try:
                payload = _request_health(url)
                if payload.get("status") == "ok":
                    return
                last_error = f"unexpected health payload: {payload!r}"
            except RuntimeError as exc:
                last_error = str(exc)
            time.sleep(0.5)
        raise RuntimeError(f"ASR did not become ready at {url}: {last_error}")

    def _watch_process(self) -> None:
        process = self.process
        if process is None:
            return
        return_code = process.wait()
        if self._closing.is_set():
            return
        LOGGER.critical("Managed ASR exited unexpectedly with code %s", return_code)
        os.kill(os.getpid(), signal.SIGTERM)


def _managed_command(config: AsrConfig) -> list[str]:
    assert config.model_path is not None
    return [
        sys.executable,
        "-m",
        "mcpmft.infer.asr",
        "--model",
        config.model_path,
        "--parent-pid",
        str(os.getpid()),
        "--host",
        config.host,
        "--port",
        str(config.port),
        "--device",
        config.device,
        "--device-index",
        str(config.device_index),
        "--compute-type",
        config.compute_type,
        "--beam-size",
        str(config.beam_size),
        "--vad-threshold",
        str(config.vad_threshold),
        "--min-frame-rms",
        str(config.min_frame_rms),
        "--min-peak",
        str(config.min_peak),
        "--min-active-speech-ms",
        str(config.min_active_speech_ms),
        "--log-level",
        config.log_level,
    ]


def _port_is_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.2):
            return True
    except OSError:
        return False


def _request_health(url: str) -> dict[str, object]:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(url, timeout=2.0) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail[:300]}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(str(exc)) from exc
    if not isinstance(payload, dict):
        raise RuntimeError("health response is not a JSON object")
    return payload
