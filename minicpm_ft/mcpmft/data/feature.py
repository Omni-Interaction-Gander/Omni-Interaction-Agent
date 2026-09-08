from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

from mcpmft.tokenizer_tools import AUDIO_END, AUDIO_START


@lru_cache(maxsize=1)
def _read_audio_sequentially(path: str):
    """Decode one shared-filesystem audio file without random seeks."""

    import soundfile as sf

    with sf.SoundFile(path) as handle:
        sample_rate = int(handle.samplerate)
        values = handle.read(dtype="float32", always_2d=True)
    return values, sample_rate


@lru_cache(maxsize=1)
def _read_video_audio_16k(path: str):
    """Decode a published source video's audio track once for adjacent unit slices."""

    try:
        import av
    except ImportError as exc:
        raise RuntimeError("Public video training requires PyAV") from exc
    import numpy as np

    chunks = []
    with av.open(path) as container:
        if not container.streams.audio:
            raise ValueError(f"public source video has no audio track: {path}")
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=16000)
        for frame in container.decode(stream):
            for output in resampler.resample(frame):
                chunks.append(output.to_ndarray().reshape(-1))
        for output in resampler.resample(None):
            chunks.append(output.to_ndarray().reshape(-1))
    if not chunks:
        raise ValueError(f"public source video has an empty audio track: {path}")
    return np.concatenate(chunks).astype(np.float32, copy=False)[:, None], 16000

@dataclass(frozen=True)
class AudioGeometry:
    sample_rate: int = 16000
    hop_length: int = 160
    pool_step: int = 5


def audio_placeholder_len(num_samples: int, geometry: AudioGeometry | None = None) -> int:
    """Match MiniCPM-o processor audio placeholder geometry.

    MiniCPM-o processor formula:
    N = (((ceil(samples / hop) - 1) // 2 + 1) - pool_step) // pool_step + 1
    """
    geometry = geometry or AudioGeometry()
    frames = math.ceil(num_samples / geometry.hop_length)
    whisper_steps = (max(frames - 1, 0) // 2) + 1
    pooled = ((whisper_steps - geometry.pool_step) // geometry.pool_step) + 1
    return max(int(pooled), 1)


def audio_placeholder_len_raw(num_samples: int, geometry: AudioGeometry | None = None) -> int:
    """Return the unclamped processor placeholder length.

    Very short residual windows may produce a non-positive value and are omitted to
    keep placeholder and feature counts aligned.
    """
    geometry = geometry or AudioGeometry()
    frames = math.ceil(num_samples / geometry.hop_length)
    whisper_steps = (max(frames - 1, 0) // 2) + 1
    return ((whisper_steps - geometry.pool_step) // geometry.pool_step) + 1


def audio_placeholder_for_duration_ms(duration_ms: int, geometry: AudioGeometry | None = None) -> list[int]:
    geometry = geometry or AudioGeometry()
    num_samples = int(duration_ms * geometry.sample_rate / 1000)
    n_audio_tokens = audio_placeholder_len(num_samples, geometry)
    # Turn serialization uses audio wrappers; duplex streaming uses explicit bounds.
    return [AUDIO_START.token_id] + [0] * n_audio_tokens + [AUDIO_END.token_id]


def find_audio_bounds(input_ids: list[int]) -> list[tuple[int, int]]:
    bounds: list[tuple[int, int]] = []
    idx = 0
    while idx < len(input_ids):
        if input_ids[idx] != AUDIO_START.token_id:
            idx += 1
            continue
        end = idx + 1
        while end < len(input_ids) and input_ids[end] != AUDIO_END.token_id:
            end += 1
        if end >= len(input_ids):
            raise ValueError("Unclosed audio placeholder")
        bounds.append((idx + 1, end))
        idx = end + 1
    return bounds


def load_audio_16k_mono(path: str, *, start_ms: int | None = None, end_ms: int | None = None):
    try:
        import librosa
    except ImportError as exc:
        raise RuntimeError("Audio loading requires librosa") from exc
    offset = 0.0 if start_ms is None else start_ms / 1000.0
    duration = None if end_ms is None else max((end_ms - (start_ms or 0)) / 1000.0, 0.0)
    audio, sr = librosa.load(path, sr=16000, mono=True, offset=offset, duration=duration)
    return audio, sr


def load_audio_ref_waveform(audio_ref) -> "np.ndarray":
    """Load a sliced 16 kHz mono waveform from files or embedded audio.

    Multichannel sources use audio_ref.channel to preserve full-duplex channel roles.
    """
    import numpy as np

    source = getattr(audio_ref, "source", None) or {}
    if source.get("kind") == "block_mix":
        from mcpmft.data.sample import AudioRef

        start_ms = getattr(audio_ref, "start_ms", None) or 0
        end_ms = getattr(audio_ref, "end_ms", None)
        dur_ms = int(source.get("duration_ms") or ((end_ms - start_ms) if end_ms is not None else 1000))
        total = max(int(dur_ms * 16000 / 1000), 1)
        mixed = np.zeros(total, dtype=np.float32)
        for segment in source.get("segments", []):
            ref = segment.get("ref")
            if isinstance(ref, dict):
                ref = AudioRef(**ref)
            if ref is None:
                continue
            wav = load_audio_ref_waveform(ref)
            offset = max(int(segment.get("offset_ms", 0) * 16000 / 1000), 0)
            if offset >= total:
                continue
            take = min(len(wav), total - offset)
            if take > 0:
                mixed[offset : offset + take] += wav[:take]
        return mixed
    if source.get("kind") == "silence":
        # Always-on microphone blocks include their silent duration.
        start_ms = getattr(audio_ref, "start_ms", None) or 0
        end_ms = getattr(audio_ref, "end_ms", None)
        dur_ms = (end_ms - start_ms) if end_ms is not None else 1000
        n = max(int(dur_ms * 16000 / 1000), 1)
        return np.zeros(n, dtype=np.float32)
    already_sliced = False
    if source.get("kind") == "indexed_audio":
        import soundfile as sf

        from mcpmft.data.audio_augment import NoiseArchiveCache, NoiseEntry

        entry_value = source.get("noise_entry")
        if not isinstance(entry_value, dict):
            raise ValueError("indexed_audio requires source.noise_entry")
        entry = NoiseEntry.from_dict(entry_value)
        cache_dir = source.get("archive_cache_dir") or "data/noise/archive_cache"

        def read_indexed():
            materialized_path = NoiseArchiveCache(cache_dir).resolve(entry)
            with sf.SoundFile(materialized_path) as handle:
                sample_rate = int(handle.samplerate)
                ref_start_ms = getattr(audio_ref, "start_ms", None) or 0
                ref_end_ms = getattr(audio_ref, "end_ms", None)
                start_frame = max(0, int(ref_start_ms * sample_rate / 1000))
                end_frame = len(handle) if ref_end_ms is None else min(
                    len(handle),
                    int(ref_end_ms * sample_rate / 1000),
                )
                handle.seek(min(start_frame, len(handle)))
                values = handle.read(
                    max(0, end_frame - start_frame),
                    dtype="float32",
                    always_2d=True,
                )
            return values, sample_rate

        wav, sr = read_indexed()
        start_ms = getattr(audio_ref, "start_ms", None) or 0
        end_ms = getattr(audio_ref, "end_ms", None)
        channel = getattr(audio_ref, "channel", None)
        if channel is not None and 0 <= channel < wav.shape[1]:
            wav = wav[:, channel]
        else:
            wav = wav.mean(axis=1)
        wav = np.asarray(wav, dtype=np.float32)
        wanted = (
            max(int((end_ms - start_ms) * 16000 / 1000), 1)
            if end_ms is not None
            else max(int(len(wav) * 16000 / max(sr, 1)), 1)
        )
        if sr != 16000 and len(wav) >= 2:
            from scipy.signal import resample_poly

            divisor = math.gcd(sr, 16000)
            wav = resample_poly(
                wav,
                up=16000 // divisor,
                down=sr // divisor,
            ).astype(np.float32)
        if len(wav) < wanted:
            wav = np.pad(wav, (0, wanted - len(wav)))
        return np.asarray(wav[:wanted], dtype=np.float32)
    elif source.get("kind") == "parquet_audio":
        import io

        import pyarrow.parquet as pq
        import soundfile as sf

        col = source.get("audio_column", "audio")
        table = pq.read_table(source["parquet"], columns=[col])
        cell = table.to_pylist()[int(source["row"])][col]
        if isinstance(cell, dict) and cell.get("array") is not None:
            # Hugging Face Audio decoded form.
            wav = np.asarray(cell["array"], dtype=np.float32)
            sr = int(cell.get("sampling_rate") or audio_ref.sample_rate or 16000)
        else:
            # Encoded audio bytes.
            audio_bytes = cell["bytes"] if isinstance(cell, dict) else cell
            wav, sr = sf.read(io.BytesIO(audio_bytes))
    else:
        start_ms = getattr(audio_ref, "start_ms", None)
        end_ms = getattr(audio_ref, "end_ms", None)
        canonical_path = getattr(audio_ref, "path", None)
        local_path = audio_ref.local_path()
        if not local_path:
            raise ValueError("audio reference has no path")
        from mcpmft.data.media import (
            is_public_video_reference,
            parse_public_video_reference,
        )

        if is_public_video_reference(canonical_path):
            reference = parse_public_video_reference(canonical_path)
            if reference.kind != "audio":
                raise ValueError(f"audio reference points to a video frame: {canonical_path}")
            all_values, sr = _read_video_audio_16k(str(local_path))
        else:
            all_values, sr = _read_audio_sequentially(str(local_path))
        start_frame = max(0, int((start_ms or 0) * sr / 1000))
        end_frame = len(all_values) if end_ms is None else min(
            len(all_values), int(end_ms * sr / 1000)
        )
        wav = all_values[start_frame:end_frame].copy()
        already_sliced = True
    wav = np.asarray(wav, dtype=np.float32)

    # Select a named channel or mix to mono.
    if wav.ndim > 1:
        ch = getattr(audio_ref, "channel", None)
        if ch is not None and ch < wav.shape[1]:
            wav = wav[:, ch]
        else:
            wav = wav.mean(axis=1)

    # Slice at the native sample rate before resampling.
    start_ms = getattr(audio_ref, "start_ms", None)
    end_ms = getattr(audio_ref, "end_ms", None)
    if not already_sliced and (start_ms is not None or end_ms is not None):
        s = int((start_ms or 0) * sr / 1000)
        e = None if end_ms is None else int(end_ms * sr / 1000)
        wav = wav[s:e]

    # Represent empty or one-sample windows as duration-matched silence.
    if end_ms is not None:
        want_16k = max(int((end_ms - (start_ms or 0)) * 16000 / 1000), 1)
    else:
        want_16k = max(int(len(wav) * 16000 / max(sr, 1)), 1)
    if wav.size == 0 or (sr != 16000 and len(wav) < 2):
        return np.zeros(want_16k, dtype=np.float32)

    if sr != 16000:
        from scipy.signal import resample_poly

        divisor = math.gcd(int(sr), 16000)
        wav = resample_poly(
            wav,
            up=16000 // divisor,
            down=int(sr) // divisor,
        ).astype(np.float32)

    # Pad partial overhang with trailing silence to match the placeholder duration.
    if end_ms is not None and len(wav) < want_16k:
        wav = np.pad(wav, (0, want_16k - len(wav)))
    return wav


def block_mic_audio(turns, block_start_ms, block_end_ms, geometry=None):
    """Return one full-block always-on mic AudioRef and placeholder length.

    The waveform is exactly the block duration: user overlaps are mixed at their timeline offsets,
    and non-overlap regions are zero. This keeps the audio encoder's grouped rows aligned to the
    1s unit grid, so a 30s encoder group can be split back into per-unit placeholders exactly.
    """
    from dataclasses import replace

    from mcpmft.data.sample import AudioRef

    geometry = geometry or AudioGeometry()
    dur_ms = max(block_end_ms - block_start_ms, 1)
    segments = []
    for turn in turns:
        if turn.audio_in is None:
            continue
        t_start = turn.start_ms if turn.start_ms is not None else block_start_ms
        t_end = turn.end_ms if turn.end_ms is not None else block_end_ms
        win_start = max(t_start, block_start_ms)
        win_end = min(t_end, block_end_ms)
        if win_end <= win_start:
            continue
        base = turn.audio_in.start_ms or 0
        ref = replace(
            turn.audio_in,
            start_ms=base + (win_start - t_start),
            end_ms=base + (win_end - t_start),
        )
        segment = {
            "ref": ref,
            "offset_ms": win_start - block_start_ms,
            "duration_ms": win_end - win_start,
        }
        speaker_id = turn.meta.get("speaker_id")
        if speaker_id is not None:
            segment["speaker_id"] = str(speaker_id)
        segments.append(segment)

    num_samples = int(dur_ms * geometry.sample_rate / 1000)
    n = audio_placeholder_len_raw(num_samples, geometry)
    if n <= 0:
        return None, 0
    if not segments:
        return silence_block_audio(block_start_ms, block_end_ms, geometry)
    ref = AudioRef(
        path=None,
        start_ms=0,
        end_ms=dur_ms,
        source={"kind": "block_mix", "duration_ms": dur_ms, "segments": segments},
    )
    return ref, n



def block_env_audio(turn, block_start_ms, block_end_ms, geometry=None):
    """Slice a turn's env-audio to one 1s block window; return (audio_ref, placeholder_len).

    Full-duplex env-audio is per-block: each chunk only perceives ITS second of audio. The
    returned AudioRef carries start_ms/end_ms = intersection of the turn span with
    [block_start_ms, block_end_ms), so the collator (load_audio_ref_waveform honours start/end_ms)
    encodes exactly that slice, and the placeholder length is computed from the SAME window —
    guaranteeing placeholder count == whisper feature frames (no scatter shape mismatch).

    Returns (None, 0) if the turn does not overlap the block or has no audio.
    """
    from dataclasses import replace

    geometry = geometry or AudioGeometry()
    if turn.audio_in is None:
        return None, 0
    t_start = turn.start_ms if turn.start_ms is not None else block_start_ms
    t_end = turn.end_ms if turn.end_ms is not None else block_end_ms
    win_start = max(t_start, block_start_ms)
    win_end = min(t_end, block_end_ms)
    if win_end <= win_start:
        return None, 0
    base = turn.audio_in.start_ms or 0
    ref = replace(turn.audio_in, start_ms=base + (win_start - t_start), end_ms=base + (win_end - t_start))
    num_samples = int((win_end - win_start) * geometry.sample_rate / 1000)
    n = audio_placeholder_len_raw(num_samples, geometry)
    if n <= 0:
        return None, 0
    return ref, n


def silence_block_audio(block_start_ms, block_end_ms, geometry=None):
    """Return (silence AudioRef, placeholder_len) for a full-duplex block with no user speech.

    Always-on mic: every 1s block feeds an audio_embed; when the user is silent this is ~1s of
    silence. The AudioRef uses source kind='silence' (load_audio_ref_waveform synthesizes zeros of
    the matching duration), and the placeholder length is computed from the SAME window so the
    placeholder count == whisper feature frames (no scatter shape mismatch), exactly like
    block_env_audio.
    """
    from mcpmft.data.sample import AudioRef

    geometry = geometry or AudioGeometry()
    dur_ms = max(block_end_ms - block_start_ms, 1)
    num_samples = int(dur_ms * geometry.sample_rate / 1000)
    n = audio_placeholder_len_raw(num_samples, geometry)
    if n <= 0:
        return None, 0
    ref = AudioRef(path=None, start_ms=0, end_ms=dur_ms, source={"kind": "silence"})
    return ref, n
