from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


STREAMING_SAMPLE_RATE = 16000
STREAMING_CHUNK_MS = 1000
STREAMING_HOP_LENGTH = 160
STREAMING_AUDIO_POOL_STEP = 5
STREAMING_FIRST_CHUNK_MS = 1035
STREAMING_CNN_REDUNDANCY_MS = 20
STREAMING_SLIDE_TRIGGER_SECONDS = 30.0
STREAMING_SLIDE_STRIDE_SECONDS = 10.0
STREAMING_MAX_GROUP_UNITS = 29


@dataclass(frozen=True)
class StreamingMelBatch:
    """Mel windows matching MiniCPM-o's one-second exact streaming frontend."""

    features: np.ndarray
    first_unit_mask: np.ndarray


def build_streaming_mel_batch(
    waveforms: Sequence[np.ndarray],
    feature_extractor: Any,
    *,
    sample_rate: int = STREAMING_SAMPLE_RATE,
    chunk_ms: int = STREAMING_CHUNK_MS,
    first_chunk_ms: int = STREAMING_FIRST_CHUNK_MS,
    cnn_redundancy_ms: int = STREAMING_CNN_REDUNDANCY_MS,
    slide_trigger_seconds: float = STREAMING_SLIDE_TRIGGER_SECONDS,
    slide_stride_seconds: float = STREAMING_SLIDE_STRIDE_SECONDS,
) -> StreamingMelBatch:
    """Vectorize the runtime mel contract without repeatedly running the APM.

    The vendor runtime receives one second at a time, prepends 35 ms before its 1030 ms first
    consumption, emits two stable suffix mel frames for the CNN, and subsequently emits two prefix
    plus two suffix frames. It also changes log-mel normalization from a fixed floor to an
    eight-decade rolling-window floor after five seconds.

    This function computes one raw log-mel spectrogram for the complete waveform, then slices and
    normalizes each unit exactly as the streaming frontend would. The returned windows can be
    processed by the CNN as one batch and by the Whisper Transformer as long chunk-masked groups;
    no per-second APM forward is required.
    """

    if not waveforms:
        return StreamingMelBatch(
            features=np.zeros((0, 0, 0), dtype=np.float32),
            first_unit_mask=np.zeros((0,), dtype=np.bool_),
        )
    if sample_rate <= 0 or chunk_ms <= 0:
        raise ValueError("sample_rate and chunk_ms must be positive")

    chunk_samples = int(round(sample_rate * chunk_ms / 1000))
    if int(feature_extractor.sampling_rate) != sample_rate:
        raise ValueError(
            "feature extractor sample rate does not match the streaming contract: "
            f"extractor={feature_extractor.sampling_rate}, requested={sample_rate}"
        )
    arrays = [np.asarray(value, dtype=np.float32).reshape(-1) for value in waveforms]
    invalid = [index for index, value in enumerate(arrays) if len(value) != chunk_samples]
    if invalid:
        raise ValueError(
            "streaming-exact audio frontend requires one fixed-size waveform per unit: "
            f"expected_samples={chunk_samples}, invalid_indices={invalid[:8]}"
        )

    hop_length = int(feature_extractor.hop_length)
    n_fft = int(feature_extractor.n_fft)
    if int(getattr(feature_extractor, "dither", 0)) != 0:
        raise ValueError("streaming-exact audio frontend requires dither=0")
    redundancy_samples = int(round(cnn_redundancy_ms * sample_rate / 1000))
    if redundancy_samples % hop_length:
        raise ValueError(
            "cnn_redundancy_ms must align to the mel hop: "
            f"samples={redundancy_samples}, hop_length={hop_length}"
        )
    redundancy_frames = redundancy_samples // hop_length
    if redundancy_frames != 2:
        raise ValueError(
            "MiniCPM-o's current two-layer CNN contract requires exactly two redundant "
            f"mel frames per side, got {redundancy_frames}"
        )

    requested_first_samples = int(first_chunk_ms * sample_rate / 1000)
    first_consumed_samples = max(
        hop_length,
        (requested_first_samples // hop_length) * hop_length,
    )
    initial_left_pad = max(0, requested_first_samples - chunk_samples)
    shifted_waveform = np.concatenate(
        [
            np.zeros(initial_left_pad, dtype=np.float32),
            np.concatenate(arrays),
        ]
    )
    final_consumed = first_consumed_samples + (len(arrays) - 1) * chunk_samples
    if final_consumed > len(shifted_waveform):
        raise ValueError(
            "streaming first-chunk geometry consumes unavailable samples: "
            f"consumed={final_consumed}, available={len(shifted_waveform)}"
        )

    raw_log_mel = _raw_log_mel(shifted_waveform, feature_extractor)
    frames_per_unit = chunk_samples // hop_length
    target_window_frames = frames_per_unit + 2 * redundancy_frames
    trigger_samples = int(round(slide_trigger_seconds * sample_rate))
    stride_samples = int(round(slide_stride_seconds * sample_rate))
    if trigger_samples <= 0 or stride_samples <= 0:
        raise ValueError("streaming mel slide trigger and stride must be positive")
    stride_samples = (stride_samples // hop_length) * hop_length

    base_samples = 0
    windows: list[np.ndarray] = []
    first_mask = np.zeros((len(arrays),), dtype=np.bool_)
    first_mask[0] = True
    for unit_index in range(len(arrays)):
        consumed_samples = first_consumed_samples + unit_index * chunk_samples
        buffer_length = consumed_samples - base_samples
        if buffer_length >= trigger_samples:
            minimum_keep = min(sample_rate, buffer_length - stride_samples)
            maximum_drop = max(0, buffer_length - minimum_keep)
            drop = min(stride_samples, maximum_drop)
            drop = (drop // hop_length) * hop_length
            base_samples += drop
            buffer_length -= drop

        core_start = unit_index * frames_per_unit
        start = max(0, core_start - redundancy_frames)
        end = (unit_index + 1) * frames_per_unit + redundancy_frames
        window = raw_log_mel[:, start:end].copy()
        expected = (
            frames_per_unit + redundancy_frames
            if unit_index == 0
            else target_window_frames
        )
        if window.shape[-1] != expected:
            raise ValueError(
                "streaming mel window has unexpected length: "
                f"unit={unit_index}, expected={expected}, actual={window.shape[-1]}"
            )

        threshold = _runtime_log_floor(
            shifted_waveform,
            raw_log_mel,
            feature_extractor,
            base_samples=base_samples,
            consumed_samples=consumed_samples,
            buffer_length=buffer_length,
        )
        window = (np.maximum(window, threshold) + 4.0) / 4.0
        if window.shape[-1] < target_window_frames:
            window = np.pad(
                window,
                ((0, 0), (0, target_window_frames - window.shape[-1])),
                mode="constant",
            )
        windows.append(window.astype(np.float32, copy=False))

    return StreamingMelBatch(
        features=np.stack(windows),
        first_unit_mask=first_mask,
    )


def _raw_log_mel(waveform: np.ndarray, feature_extractor: Any) -> np.ndarray:
    from transformers.audio_utils import spectrogram, window_function

    values = spectrogram(
        waveform,
        window_function(int(feature_extractor.n_fft), "hann"),
        frame_length=int(feature_extractor.n_fft),
        hop_length=int(feature_extractor.hop_length),
        power=2.0,
        dither=float(feature_extractor.dither),
        mel_filters=feature_extractor.mel_filters,
        log_mel="log10",
    )
    # MiniCPMAAudioProcessor removes the final centered frame.
    return np.asarray(values[:, :-1], dtype=np.float32)


def _runtime_log_floor(
    shifted_waveform: np.ndarray,
    raw_log_mel: np.ndarray,
    feature_extractor: Any,
    *,
    base_samples: int,
    consumed_samples: int,
    buffer_length: int,
) -> float:
    sample_rate = int(feature_extractor.sampling_rate)
    if buffer_length < 5 * sample_rate:
        # StreamingMelProcessorExact intentionally uses a fixed floor during startup.
        return -10.0

    hop_length = int(feature_extractor.hop_length)
    n_fft = int(feature_extractor.n_fft)
    half_window = n_fft // 2
    base_frame = base_samples // hop_length
    frame_count = buffer_length // hop_length
    stable_count = max(0, (buffer_length - half_window) // hop_length + 1)
    left_boundary_count = min(math.ceil(half_window / hop_length), frame_count)

    maxima: list[float] = []
    interior_start = base_frame + left_boundary_count
    interior_end = base_frame + stable_count
    if interior_end > interior_start:
        maxima.append(float(raw_log_mel[:, interior_start:interior_end].max()))

    # Center padding changes only the first two and the final unstable mel frames of a rolling
    # buffer. Recompute those few frames exactly instead of recalculating the complete prefix.
    local_waveform = shifted_waveform[base_samples:consumed_samples]
    boundary_indices = list(range(left_boundary_count))
    boundary_indices.extend(range(stable_count, frame_count))
    if boundary_indices:
        maxima.append(
            float(
                _selected_raw_log_mel_frames(
                    local_waveform,
                    feature_extractor,
                    boundary_indices,
                ).max()
            )
        )
    if not maxima:
        raise ValueError("streaming dynamic normalization has no mel frames")
    # StreamingMelProcessorExact explicitly switches to dynamic_range_db=8 after startup.
    return max(maxima) - 8.0


def _selected_raw_log_mel_frames(
    waveform: np.ndarray,
    feature_extractor: Any,
    frame_indices: Sequence[int],
) -> np.ndarray:
    n_fft = int(feature_extractor.n_fft)
    hop_length = int(feature_extractor.hop_length)
    half_window = n_fft // 2
    padded = np.pad(
        np.asarray(waveform, dtype=np.float32),
        (half_window, half_window),
        mode="reflect",
    ).astype(np.float64)
    from transformers.audio_utils import window_function

    window = window_function(n_fft, "hann").astype(np.float64)
    values = []
    for frame_index in frame_indices:
        start = int(frame_index) * hop_length
        frame = padded[start : start + n_fft] * window
        spectrum = np.fft.rfft(frame)
        power = np.abs(spectrum, dtype=np.float64) ** 2.0
        mel = np.maximum(
            1e-10,
            np.dot(feature_extractor.mel_filters.T, power),
        )
        values.append(np.log10(mel).astype(np.float32))
    return np.stack(values, axis=1)
