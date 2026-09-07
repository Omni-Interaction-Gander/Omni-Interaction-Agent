from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse

LOGGER = logging.getLogger(__name__)

SILENCE_HALLUCINATIONS = {
    "thankyou",
    "thanks",
    "thanksforwatching",
    "pleasesubscribe",
    "subtitlesbytheamaraorgcommunity",
    "谢谢",
    "谢谢观看",
    "感谢观看",
}


@dataclass(frozen=True)
class AsrSettings:
    model_path: str
    device: str = "cuda"
    device_index: int = 0
    compute_type: str = "float16"
    sample_rate: int = 16000
    beam_size: int = 3
    vad_threshold: float = 0.6
    min_frame_rms: float = 0.0035
    min_peak: float = 0.01
    min_active_speech_ms: int = 210


def analyze_audio(pcm: bytes, settings: AsrSettings) -> dict[str, Any]:
    audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
    frame_samples = max(1, int(settings.sample_rate * 0.03))
    frame_count = max(1, int(np.ceil(len(audio) / frame_samples)))
    padded = np.pad(audio, (0, frame_count * frame_samples - len(audio)))
    frames = padded.reshape(frame_count, frame_samples)
    frame_rms = np.sqrt(np.mean(np.square(frames), axis=1, dtype=np.float64))
    noise_floor = float(np.quantile(frame_rms, 0.2))
    adaptive_threshold = max(
        settings.min_frame_rms,
        min(noise_floor * 2.5, settings.min_frame_rms * 3.0),
    )
    active_frames = int(np.count_nonzero(frame_rms >= adaptive_threshold))
    active_ms = active_frames * frame_samples * 1000.0 / settings.sample_rate
    overall_rms = float(np.sqrt(np.mean(np.square(audio), dtype=np.float64)))
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    return {
        "duration": round(len(audio) / settings.sample_rate, 3),
        "rms": round(overall_rms, 6),
        "peak": round(peak, 6),
        "noise_floor_rms": round(noise_floor, 6),
        "speech_threshold_rms": round(adaptive_threshold, 6),
        "active_speech_ms": round(active_ms),
        "has_speech": bool(
            peak >= settings.min_peak
            and active_ms >= settings.min_active_speech_ms
        ),
    }


def _normalized_text(text: str) -> str:
    return "".join(character for character in text.casefold() if character.isalnum())


def _empty_result(
    *,
    audio: dict[str, Any],
    language: str | None,
    reason: str,
) -> dict[str, Any]:
    return {
        "text": "",
        "language": language,
        "language_probability": 0.0,
        "confidence": 0.0,
        "duration": audio["duration"],
        "segments": [],
        "words": [],
        "audio": audio,
        "discarded_reason": reason,
    }


def filter_transcription(
    result: dict[str, Any],
    *,
    audio: dict[str, Any],
    requested_language: str | None,
) -> dict[str, Any]:
    result = dict(result)
    result["audio"] = audio
    text = str(result.get("text") or "").strip()
    if not text:
        result.setdefault("discarded_reason", "no-speech")
        return result

    confidence = float(result.get("confidence", 1.0))
    language_probability = float(result.get("language_probability", 1.0))
    weak_signals = sum(
        (
            audio["active_speech_ms"] < 450,
            confidence < 0.45,
            not requested_language and language_probability < 0.55,
        )
    )
    if _normalized_text(text) in SILENCE_HALLUCINATIONS and weak_signals >= 2:
        return _empty_result(
            audio=audio,
            language=result.get("language"),
            reason="low-confidence-silence-hallucination",
        )
    return result


class FasterWhisperTranscriber:
    def __init__(self, settings: AsrSettings) -> None:
        from faster_whisper import WhisperModel

        self.settings = settings
        self.model = WhisperModel(
            settings.model_path,
            device=settings.device,
            device_index=settings.device_index,
            compute_type=settings.compute_type,
            cpu_threads=4,
            num_workers=1,
            local_files_only=True,
        )

    def transcribe(
        self,
        pcm: bytes,
        *,
        start_ms: int,
        language: str | None,
    ) -> dict[str, Any]:
        audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        segments_iter, info = self.model.transcribe(
            audio,
            language=language or None,
            beam_size=self.settings.beam_size,
            best_of=1,
            temperature=0.0,
            condition_on_previous_text=False,
            word_timestamps=False,
            vad_filter=True,
            vad_parameters={
                "threshold": self.settings.vad_threshold,
                "min_speech_duration_ms": self.settings.min_active_speech_ms,
                "min_silence_duration_ms": 500,
                "speech_pad_ms": 160,
            },
        )
        offset = start_ms / 1000.0
        segments = []
        words = []
        discarded_segments = 0
        for segment in segments_iter:
            text = segment.text.strip()
            if not text:
                continue
            no_speech_probability = float(
                getattr(segment, "no_speech_prob", 0.0) or 0.0
            )
            average_log_probability = float(
                getattr(segment, "avg_logprob", 0.0) or 0.0
            )
            if average_log_probability < -1.2 or (
                no_speech_probability > 0.8 and average_log_probability < -0.35
            ):
                discarded_segments += 1
                continue
            segment_words = []
            for word in segment.words or []:
                if word.start is None or word.end is None:
                    continue
                item = {
                    "start": round(offset + float(word.start), 3),
                    "end": round(offset + float(word.end), 3),
                    "text": word.word,
                    "probability": round(float(word.probability or 0.0), 4),
                }
                segment_words.append(item)
                words.append(item)
            segments.append(
                {
                    "start": round(offset + float(segment.start), 3),
                    "end": round(offset + float(segment.end), 3),
                    "text": text,
                    "words": segment_words,
                    "no_speech_probability": round(no_speech_probability, 4),
                    "average_log_probability": round(average_log_probability, 4),
                }
            )

        word_probabilities = [word["probability"] for word in words]
        segment_log_probabilities = [
            segment["average_log_probability"] for segment in segments
        ]
        if word_probabilities:
            confidence = float(np.mean(word_probabilities))
        elif segment_log_probabilities:
            confidence = float(np.exp(np.mean(segment_log_probabilities)))
        else:
            confidence = 0.0
        return {
            "text": " ".join(segment["text"] for segment in segments),
            "language": info.language,
            "language_probability": round(float(info.language_probability), 4),
            "confidence": round(confidence, 4),
            "duration": round(len(audio) / self.settings.sample_rate, 3),
            "segments": segments,
            "words": words,
            "discarded_segments": discarded_segments,
        }


def create_app(
    settings: AsrSettings,
    transcriber: FasterWhisperTranscriber,
) -> FastAPI:
    app = FastAPI(title="MiniCPM-o ASR Sidecar", version="1.0.0")
    lock = asyncio.Lock()

    @app.get("/health")
    async def health() -> JSONResponse:
        return JSONResponse(
            {
                "status": "ok",
                "model": settings.model_path,
                "device": settings.device,
                "sample_rate": settings.sample_rate,
                "vad_threshold": settings.vad_threshold,
                "min_active_speech_ms": settings.min_active_speech_ms,
            }
        )

    @app.post("/transcribe")
    async def transcribe(
        request: Request,
        start_ms: int = Query(default=0, ge=0),
        sample_rate: int = Query(default=16000),
        language: str = Query(default=""),
    ) -> JSONResponse:
        if sample_rate != settings.sample_rate:
            return JSONResponse(
                {
                    "type": "error",
                    "message": f"sample_rate must be {settings.sample_rate}",
                },
                status_code=400,
            )
        pcm = await request.body()
        if not pcm or len(pcm) % 2:
            return JSONResponse(
                {"type": "error", "message": "input must be non-empty PCM16"},
                status_code=400,
            )
        if len(pcm) > settings.sample_rate * 2 * 120:
            return JSONResponse(
                {"type": "error", "message": "input exceeds 120 seconds"},
                status_code=413,
            )

        audio = analyze_audio(pcm, settings)
        if not audio["has_speech"]:
            return JSONResponse(
                _empty_result(
                    audio=audio,
                    language=language or None,
                    reason="silence-or-background-noise",
                )
            )

        async with lock:
            result = await asyncio.to_thread(
                transcriber.transcribe,
                pcm,
                start_ms=start_ms,
                language=language or None,
            )
        return JSONResponse(
            filter_transcription(
                result,
                audio=audio,
                requested_language=language or None,
            )
        )

    return app


def watch_parent(parent_pid: int) -> None:
    if parent_pid <= 1:
        raise ValueError("parent_pid must identify a live process")

    def run() -> None:
        while os.getppid() == parent_pid:
            time.sleep(0.5)
        LOGGER.info("Gander parent exited; stopping managed ASR")
        os.kill(os.getpid(), signal.SIGTERM)

    threading.Thread(target=run, name="gander-parent-watch", daemon=True).start()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve local faster-whisper ASR with timestamps"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--parent-pid", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="cuda")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--compute-type", default="float16")
    parser.add_argument("--beam-size", type=int, default=3)
    parser.add_argument("--vad-threshold", type=float, default=0.6)
    parser.add_argument("--min-frame-rms", type=float, default=0.0035)
    parser.add_argument("--min-peak", type=float, default=0.01)
    parser.add_argument("--min-active-speech-ms", type=int, default=210)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8995)
    parser.add_argument("--log-level", default="info")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.parent_pid is not None:
        watch_parent(args.parent_pid)
    settings = AsrSettings(
        model_path=args.model,
        device=args.device,
        device_index=args.device_index,
        compute_type=args.compute_type,
        beam_size=args.beam_size,
        vad_threshold=args.vad_threshold,
        min_frame_rms=args.min_frame_rms,
        min_peak=args.min_peak,
        min_active_speech_ms=args.min_active_speech_ms,
    )
    LOGGER.info("Loading faster-whisper from %s", settings.model_path)
    transcriber = FasterWhisperTranscriber(settings)
    app = create_app(settings, transcriber)

    import uvicorn

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level.lower(),
    )


if __name__ == "__main__":
    main()
