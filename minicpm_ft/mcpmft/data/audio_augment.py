from __future__ import annotations

import bisect
import hashlib
import json
import logging
import math
import os
import random
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from mcpmft.data.augmentation_profile import AugmentationProfile, ProfileResolver

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class NoiseEntry:
    id: str
    source: str
    category: str
    sample_rate: int
    channels: int
    frames: int
    duration_seconds: float
    virtual_windows: int = 1
    path: str | None = None
    archive_path: str | None = None
    archive_member: str | None = None
    size_bytes: int | None = None

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "NoiseEntry":
        entry = cls(
            id=str(value["id"]),
            source=str(value["source"]),
            category=str(value.get("category") or "general"),
            sample_rate=int(value["sample_rate"]),
            channels=int(value["channels"]),
            frames=int(value["frames"]),
            duration_seconds=float(value["duration_seconds"]),
            virtual_windows=max(1, int(value.get("virtual_windows") or 1)),
            path=value.get("path"),
            archive_path=value.get("archive_path"),
            archive_member=value.get("archive_member"),
            size_bytes=(
                int(value["size_bytes"]) if value.get("size_bytes") is not None else None
            ),
        )
        if entry.sample_rate <= 0 or entry.channels <= 0 or entry.frames <= 0:
            raise ValueError(f"Invalid noise entry geometry: {entry.id}")
        if bool(entry.path) == bool(entry.archive_path and entry.archive_member):
            raise ValueError(
                f"Noise entry must have exactly one file or archive source: {entry.id}"
            )
        return entry


class _WeightedBucket:
    def __init__(self, entries: Sequence[NoiseEntry]) -> None:
        self.entries = list(entries)
        self.cumulative: list[float] = []
        total = 0.0
        for entry in self.entries:
            # One long recording represents many independently sampled virtual windows. Weighting
            # by that count lets a multi-hour recording contribute more than a single ten-second
            # clip without physically duplicating either waveform or index rows.
            total += float(max(1, entry.virtual_windows))
            self.cumulative.append(total)
        self.total = total

    def pick(
        self,
        rng: random.Random,
        *,
        exclude_id: str | None = None,
        exclude_ids: set[str] | None = None,
    ) -> NoiseEntry:
        if not self.entries:
            raise LookupError("Cannot sample an empty noise bucket")
        excluded = set(exclude_ids or ())
        if exclude_id is not None:
            excluded.add(exclude_id)
        if len(excluded) >= len(self.entries) and all(
            entry.id in excluded for entry in self.entries
        ):
            raise LookupError("No usable entries remain in noise bucket")
        for _ in range(min(16, max(4, len(self.entries)))):
            point = rng.random() * self.total
            index = min(bisect.bisect_right(self.cumulative, point), len(self.entries) - 1)
            selected = self.entries[index]
            if selected.id not in excluded:
                return selected
        # This guarantees progress when a heavily weighted long recording was quarantined.
        return next(entry for entry in self.entries if entry.id not in excluded)


class NoiseCatalog:
    def __init__(self, entries: Sequence[NoiseEntry]) -> None:
        if not entries:
            raise ValueError("Noise catalog is empty")
        grouped: dict[tuple[str, str], list[NoiseEntry]] = {}
        for entry in entries:
            grouped.setdefault((entry.source, entry.category), []).append(entry)
        self.buckets = {key: _WeightedBucket(values) for key, values in grouped.items()}

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "NoiseCatalog":
        entries: list[NoiseEntry] = []
        with Path(path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    entries.append(NoiseEntry.from_dict(json.loads(line)))
                except Exception as exc:
                    raise ValueError(f"Invalid noise index row {path}:{line_number}") from exc
        return cls(entries)

    def choose_pair(
        self,
        rng: random.Random,
        *,
        source_weights: Mapping[str, float],
        category_weights: Mapping[str, float],
    ) -> tuple[str, str]:
        categories_by_source: dict[str, list[tuple[str, float]]] = {}
        for source, category in self.buckets:
            source_weight = max(float(source_weights.get(source, 0.0)), 0.0)
            category_weight = max(float(category_weights.get(category, 0.0)), 0.0)
            if source_weight > 0 and category_weight > 0:
                categories_by_source.setdefault(source, []).append((category, category_weight))
        if not categories_by_source:
            available = ", ".join(f"{source}/{category}" for source, category in self.buckets)
            requested = ", ".join(
                key for key, weight in category_weights.items() if float(weight) > 0
            )
            raise LookupError(
                f"Noise index has no source/category bucket for requested categories "
                f"[{requested}]; available=[{available}]"
            )
        source = _weighted_choice(
            [
                (name, float(source_weights[name]))
                for name in categories_by_source
            ],
            rng,
        )
        category = _weighted_choice(categories_by_source[source], rng)
        return source, category

    def pick_entry(
        self,
        pair: tuple[str, str],
        rng: random.Random,
        *,
        exclude_id: str | None = None,
        exclude_ids: set[str] | None = None,
    ) -> NoiseEntry:
        return self.buckets[pair].pick(
            rng,
            exclude_id=exclude_id,
            exclude_ids=exclude_ids,
        )


class NoiseReadError(LookupError):
    """Optional augmentation noise could not be decoded after bounded failover."""


class NoiseArchiveCache:
    """Materialize compressed archive members once, safely across ranks and workers."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def resolve(self, entry: NoiseEntry) -> Path:
        if entry.path:
            path = Path(entry.path)
            if not path.is_file():
                raise FileNotFoundError(f"Noise file not found: {path}")
            return path
        if not entry.archive_path or not entry.archive_member:
            raise ValueError(f"Archive noise entry is incomplete: {entry.id}")

        digest = hashlib.sha1(
            f"{entry.archive_path}\0{entry.archive_member}".encode("utf-8")
        ).hexdigest()
        suffix = Path(entry.archive_member).suffix or ".wav"
        target = self.root / digest[:2] / f"{digest}{suffix}"
        expected_size = entry.size_bytes
        if _complete_cached_file(target, expected_size):
            return target

        target.parent.mkdir(parents=True, exist_ok=True)
        lock_path = target.with_suffix(target.suffix + ".lock")
        with lock_path.open("a+b") as lock_handle:
            try:
                import fcntl

                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            except ImportError:
                pass
            if _complete_cached_file(target, expected_size):
                return target
            temporary = target.with_name(f"{target.name}.tmp.{os.getpid()}")
            try:
                with zipfile.ZipFile(entry.archive_path) as archive:
                    with archive.open(entry.archive_member) as source, temporary.open("wb") as out:
                        shutil.copyfileobj(source, out, length=8 * 1024 * 1024)
                if expected_size is not None and temporary.stat().st_size != expected_size:
                    raise IOError(
                        f"Extracted noise member has wrong size: {entry.archive_member}"
                    )
                os.replace(temporary, target)
            finally:
                if temporary.exists():
                    temporary.unlink()
        return target


class NoiseStreamSampler:
    def __init__(
        self,
        catalog: NoiseCatalog,
        *,
        archive_cache_dir: str | Path,
        sample_rate: int = 16000,
        crossfade_ms: int = 100,
        max_segment_seconds: float = 30.0,
    ) -> None:
        self.catalog = catalog
        self.archive_cache = NoiseArchiveCache(archive_cache_dir)
        self.sample_rate = int(sample_rate)
        self.crossfade_samples = max(0, int(crossfade_ms * self.sample_rate / 1000))
        self.max_segment_samples = max(1, int(max_segment_seconds * self.sample_rate))
        # Each DataLoader worker owns its sampler, so this quarantine is process-local and cheap.
        self._quarantined_entry_ids: set[str] = set()

    def sample_stream(
        self,
        num_samples: int,
        rng: random.Random,
        *,
        source_weights: Mapping[str, float],
        category_weights: Mapping[str, float],
    ) -> np.ndarray:
        waveform, _ = self.sample_stream_with_pair(
            num_samples,
            rng,
            source_weights=source_weights,
            category_weights=category_weights,
        )
        return waveform

    def sample_stream_with_pair(
        self,
        num_samples: int,
        rng: random.Random,
        *,
        source_weights: Mapping[str, float],
        category_weights: Mapping[str, float],
    ) -> tuple[np.ndarray, tuple[str, str]]:
        waveform, trace = self.sample_stream_with_trace(
            num_samples,
            rng,
            source_weights=source_weights,
            category_weights=category_weights,
        )
        return waveform, (str(trace["source"]), str(trace["category"]))

    def sample_stream_with_trace(
        self,
        num_samples: int,
        rng: random.Random,
        *,
        source_weights: Mapping[str, float],
        category_weights: Mapping[str, float],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if num_samples <= 0:
            return np.zeros(0, dtype=np.float32), {
                "source": "",
                "category": "",
                "entries": [],
            }
        pair = self.catalog.choose_pair(
            rng,
            source_weights=source_weights,
            category_weights=category_weights,
        )
        failed_entries: list[dict[str, str]] = []
        entry: NoiseEntry | None = None
        output: np.ndarray | None = None
        stream_trace: dict[str, Any] = {}
        bucket_size = len(self.catalog.buckets[pair].entries)
        for _ in range(min(4, bucket_size)):
            try:
                entry = self.catalog.pick_entry(
                    pair,
                    rng,
                    exclude_ids=self._quarantined_entry_ids,
                )
            except LookupError:
                break
            try:
                output, stream_trace = self._read_coherent_stream(entry, num_samples, rng)
                if output.size == 0:
                    raise RuntimeError("decoded waveform is empty")
                break
            except (OSError, RuntimeError, EOFError, zipfile.BadZipFile, KeyError) as exc:
                self._quarantined_entry_ids.add(entry.id)
                LOGGER.warning(
                    "Quarantined augmentation noise entry id=%s path=%s after decode failure: %s",
                    entry.id,
                    entry.path or entry.archive_path,
                    _safe_exception_text(exc),
                )
                failed_entries.append(
                    {
                        "id": entry.id,
                        "path": str(entry.path or entry.archive_path or ""),
                        "error": _safe_exception_text(exc),
                    }
                )
                entry = None
                output = None
        if entry is None or output is None:
            detail = json.dumps(failed_entries, ensure_ascii=True)
            raise NoiseReadError(
                f"Could not decode augmentation noise from bucket {pair}; "
                f"attempts={len(failed_entries)} failures={detail}"
            )
        output = output.astype(np.float32, copy=False)
        output -= float(np.mean(output))
        return output, {
            "source": pair[0],
            "category": pair[1],
            "coherent_single_recording": True,
            "failed_entries": failed_entries,
            "entries": [
                {
                    "id": entry.id,
                    "path": entry.path,
                    "archive_path": entry.archive_path,
                    "archive_member": entry.archive_member,
                    **stream_trace,
                }
            ],
        }

    def _read_coherent_stream(
        self,
        entry: NoiseEntry,
        wanted_samples: int,
        rng: random.Random,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        try:
            import soundfile as sf
        except ImportError as exc:
            raise RuntimeError("Audio augmentation requires soundfile") from exc

        path = self.archive_cache.resolve(entry)
        with sf.SoundFile(path) as handle:
            source_rate = int(handle.samplerate)
            source_frames = int(len(handle))
            if source_frames <= 0:
                return np.zeros(0, dtype=np.float32), {}
            channel = rng.randrange(int(handle.channels)) if handle.channels > 1 else 0
            native_wanted = max(
                1,
                int(math.ceil(wanted_samples * source_rate / self.sample_rate)) + 4,
            )
            native_limit = max(
                1,
                int(self.max_segment_samples * source_rate / self.sample_rate),
            )
            native_crossfade = max(
                0,
                int(self.crossfade_samples * source_rate / self.sample_rate),
            )
            initial_frame = rng.randrange(source_frames) if source_frames > 1 else 0
            cursor = initial_frame
            wraps = 0
            reads = 0
            crossfade_next = False
            waveform = np.zeros(0, dtype=np.float32)
            while len(waveform) < native_wanted:
                overlap = (
                    min(native_crossfade, len(waveform))
                    if crossfade_next
                    else 0
                )
                needed = min(
                    native_limit,
                    native_wanted - len(waveform) + overlap,
                )
                available = source_frames - cursor
                if available <= 0:
                    cursor = 0
                    wraps += 1
                    crossfade_next = True
                    continue
                handle.seek(cursor)
                values = handle.read(
                    min(needed, available),
                    dtype="float32",
                    always_2d=True,
                )
                if values.size == 0:
                    cursor = 0
                    wraps += 1
                    continue
                part = np.asarray(values[:, channel], dtype=np.float32)
                # Always keep at least half of a short recording so tiny clips cannot make a
                # wrapping stream grow one sample at a time.
                actual_overlap = min(overlap, len(waveform), len(part) // 2)
                if actual_overlap:
                    phase = np.linspace(
                        0.0,
                        math.pi / 2.0,
                        actual_overlap,
                        dtype=np.float32,
                    )
                    waveform[-actual_overlap:] = (
                        waveform[-actual_overlap:] * np.cos(phase)
                        + part[:actual_overlap] * np.sin(phase)
                    )
                waveform = np.concatenate([waveform, part[actual_overlap:]])
                cursor += len(part)
                reads += 1
                crossfade_next = False
                if cursor >= source_frames:
                    cursor = 0
                    wraps += 1
                    crossfade_next = True
        if source_rate != self.sample_rate:
            try:
                from scipy.signal import resample_poly
            except ImportError as exc:
                raise RuntimeError("Resampling augmentation noise requires scipy") from exc
            divisor = math.gcd(source_rate, self.sample_rate)
            waveform = resample_poly(
                waveform,
                up=self.sample_rate // divisor,
                down=source_rate // divisor,
            )
        waveform = np.asarray(waveform, dtype=np.float32)
        if len(waveform) < wanted_samples:
            waveform = np.pad(
                waveform,
                (0, wanted_samples - len(waveform)),
                mode="wrap" if waveform.size else "constant",
            )
        return waveform[:wanted_samples], {
            "source_start_frame": initial_frame,
            "source_channel": channel,
            "source_wraps": wraps,
            "read_segments": reads,
        }


class AudioAugmenter:
    """Profile-driven, sample-continuous microphone scene augmentation."""

    def __init__(self, args: Any, *, sample_rate: int = 16000) -> None:
        self.enabled = bool(args.enabled)
        self.index_path = args.noise_index_path
        self.archive_cache_dir = args.archive_cache_dir
        self.sample_rate = int(sample_rate)
        self.source_weights = dict(args.source_weights)
        self.profile_resolver = ProfileResolver(
            args.profiles,
            args.profile_rules,
            default_profile=args.default_profile,
        )
        self.category_snr_offsets_db = dict(args.category_snr_offsets_db)
        self.transient_category_weights = dict(args.transient_category_weights)
        self.transient_sir_bands = list(args.transient_sir_bands)
        self.transient_min_seconds = float(args.transient_min_seconds)
        self.transient_max_seconds = float(args.transient_max_seconds)
        self.rir_index_path = args.rir_index_path
        self.rir_max_seconds = float(args.rir_max_seconds)
        self.room_rt60_seconds = list(args.room_rt60_seconds)
        self.room_predelay_ms = list(args.room_predelay_ms)
        self.room_wet = list(args.room_wet)
        self.room_hf_damping = list(args.room_hf_damping)
        self.room_presets = {
            str(name): dict(value) for name, value in args.room_presets.items()
        }
        self.speaker_distance_presets = {
            str(name): dict(value)
            for name, value in args.speaker_distance_presets.items()
        }
        self.device_gain_db = list(args.device_gain_db)
        self.device_low_cut_hz = list(args.device_low_cut_hz)
        self.device_high_cut_hz = list(args.device_high_cut_hz)
        self.device_compression = list(args.device_compression)
        self.echo_delay_ms = list(args.echo_delay_ms)
        self.echo_erl_db = list(args.echo_erl_db)
        self.echo_room_mix = list(args.echo_room_mix)
        self.silence_dbfs_min = float(args.silence_dbfs_min)
        self.silence_dbfs_max = float(args.silence_dbfs_max)
        self.crossfade_ms = int(args.crossfade_ms)
        self.max_noise_segment_seconds = float(args.max_noise_segment_seconds)
        self.peak_limit = float(args.peak_limit)
        self._sampler: NoiseStreamSampler | None = None
        self._rir_rows: list[dict[str, Any]] | None = None
        self._warned: set[str] = set()

    def augment(
        self,
        waveforms: Sequence[np.ndarray],
        refs: Sequence[Any],
        sample_meta: Mapping[str, Any] | None,
        rng: random.Random,
    ) -> list[np.ndarray]:
        waveforms, _ = self.augment_with_trace(waveforms, refs, sample_meta, rng)
        return waveforms

    def augment_with_trace(
        self,
        waveforms: Sequence[np.ndarray],
        refs: Sequence[Any],
        sample_meta: Mapping[str, Any] | None,
        rng: random.Random,
        *,
        force_actions: Mapping[str, Any] | None = None,
    ) -> tuple[list[np.ndarray], dict[str, Any]]:
        clean = [np.asarray(waveform, dtype=np.float32) for waveform in waveforms]
        meta = sample_meta or {}
        profile = self.resolve_profile(meta)
        trace: dict[str, Any] = {
            "profile": profile.name,
            "actions": {},
        }
        if not self.enabled or not clean:
            trace["disabled"] = not self.enabled
            return clean, trace
        if len(clean) != len(refs):
            raise ValueError("Audio augmenter requires one AudioRef per waveform")

        forced = dict(force_actions or {})
        combined_scene = _select_action(
            profile.combined_probability,
            rng,
            forced.get("combined"),
        )
        trace["combined_scene"] = {
            "selected": combined_scene,
            "configured_probability": profile.combined_probability,
        }
        background_forced = forced.get("background")
        if background_forced is not None:
            augment_background = bool(background_forced)
        else:
            augment_background = (
                combined_scene and profile.background_probability > 0.0
            ) or _select_action(profile.background_probability, rng, None)
        floor_forced = forced.get("floor")
        augment_floor = (
            not augment_background
            and (
                bool(floor_forced)
                if floor_forced is not None
                else _select_action(
                    profile.floor_probability,
                    rng,
                    None,
                )
            )
        )

        lengths = [len(waveform) for waveform in clean]
        total_samples = sum(lengths)
        if total_samples <= 0:
            return clean, trace
        merged = np.concatenate(clean).astype(np.float32, copy=False)

        room_forced = forced.get("room")
        selected_room = (
            bool(room_forced)
            if room_forced is not None
            else (
                (combined_scene and profile.room_probability > 0.0)
                or _select_action(profile.room_probability, rng, None)
            )
        )
        spatial_forced = forced.get("speaker_spatial")
        selected_speaker_spatial = (
            bool(spatial_forced)
            if spatial_forced is not None
            else (
                (combined_scene and profile.speaker_spatial_probability > 0.0)
                or _select_action(profile.speaker_spatial_probability, rng, None)
            )
        )
        room_impulse: np.ndarray | None = None
        spatial_applied = False
        if selected_speaker_spatial:
            merged, spatial_trace = self._apply_multiparty_spatial_response(
                merged,
                refs,
                lengths,
                rng,
                room_preset_weights=profile.room_preset_weights,
                distance_weights=profile.speaker_distance_weights,
                forced_room_preset=forced.get("room_preset"),
            )
            spatial_applied = bool(spatial_trace.get("applied"))
            if spatial_applied:
                trace["actions"]["room"] = spatial_trace
        if selected_room and not spatial_applied:
            room_impulse, room_trace = self._sample_room_response(
                rng,
                preset_weights=profile.room_preset_weights,
                forced_preset=forced.get("room_preset"),
            )
            merged, applied_trace = self._apply_room_response(
                merged,
                room_impulse,
                rng,
                wet_range=room_trace.get("wet_range"),
            )
            trace["actions"]["room"] = {**room_trace, **applied_trace}
        elif not spatial_applied:
            trace["actions"]["room"] = {"selected": False, "applied": False}
            if selected_speaker_spatial:
                trace["actions"]["room"].update(
                    {
                        "selected": True,
                        "reason": spatial_trace.get(
                            "reason",
                            "multi-speaker source metadata is unavailable",
                        ),
                    }
                )

        processed_clean: list[np.ndarray] = []
        cursor = 0
        for length in lengths:
            processed_clean.append(merged[cursor : cursor + length])
            cursor += length
        active = [
            waveform
            for waveform, ref in zip(processed_clean, refs)
            if _is_active_ref(ref)
        ]
        signal_rms = _robust_signal_rms(active)

        background_rms = 0.0
        if augment_background or augment_floor:
            tier = "scene" if augment_background else "floor"
            weights = (
                profile.category_weights
                if augment_background
                else profile.floor_category_weights
            )
            snr_bands = (
                profile.snr_bands
                if augment_background
                else profile.floor_snr_bands
            )
            forced_category = forced.get("background_category")
            if forced_category:
                weights = {str(forced_category): 1.0}
            try:
                background, background_trace = self._get_sampler().sample_stream_with_trace(
                    total_samples,
                    rng,
                    source_weights=self.source_weights,
                    category_weights=weights,
                )
            except LookupError as exc:
                self._warn_once(f"base:{profile.name}", str(exc))
                background = np.zeros(total_samples, dtype=np.float32)
                background_trace = {
                    "selected": True,
                    "applied": False,
                    "reason": str(exc),
                }
            background_rms = _rms(background)
            if background_rms > 1e-8:
                if signal_rms is not None:
                    snr_db = _sample_band(snr_bands, rng) + float(
                        self.category_snr_offsets_db.get(
                            background_trace.get("category"), 0.0
                        )
                    )
                    background_gain = signal_rms / (
                        background_rms * (10.0 ** (snr_db / 20.0))
                    )
                else:
                    snr_db = None
                    target_dbfs = rng.uniform(
                        self.silence_dbfs_min,
                        self.silence_dbfs_max,
                    )
                    background_gain = (10.0 ** (target_dbfs / 20.0)) / background_rms
                scaled_background = background * float(background_gain)
                merged = merged + scaled_background
                background_rms = _rms(scaled_background)
                background_trace.update(
                    {
                        "selected": True,
                        "applied": True,
                        "tier": tier,
                        "scope": "full_stream",
                        "snr_db": snr_db,
                        "gain": float(background_gain),
                    }
                )
            else:
                background_trace.update(
                    {
                        "selected": True,
                        "applied": False,
                        "reason": "sampled noise was silent",
                    }
                )
            trace["actions"]["background"] = background_trace
        else:
            trace["actions"]["background"] = {"selected": False, "applied": False}

        selected_transient = _select_action(
            profile.transient_probability,
            rng,
            forced.get("transient"),
        )
        if selected_transient:
            merged, transient_trace = self._add_transient(
                merged,
                signal_rms=signal_rms,
                background_rms=background_rms,
                rng=rng,
                forced_category=forced.get("transient_category"),
            )
            trace["actions"]["transient"] = transient_trace
        else:
            trace["actions"]["transient"] = {"selected": False, "applied": False}

        echo_forced = forced.get("echo")
        selected_echo = (
            bool(echo_forced)
            if echo_forced is not None
            else (
                (combined_scene and profile.echo_probability > 0.0)
                or _select_action(profile.echo_probability, rng, None)
            )
        )
        if selected_echo:
            echo, echo_trace = self._build_causal_echo(
                refs,
                lengths,
                meta,
                rng=rng,
                room_impulse=room_impulse,
                room_preset_weights=profile.room_preset_weights,
            )
            if echo is not None:
                merged = merged + echo
            trace["actions"]["echo"] = echo_trace
        else:
            trace["actions"]["echo"] = {"selected": False, "applied": False}

        device_forced = forced.get("device")
        selected_device = (
            bool(device_forced)
            if device_forced is not None
            else (
                (combined_scene and profile.device_probability > 0.0)
                or _select_action(profile.device_probability, rng, None)
            )
        )
        if selected_device:
            merged, device_trace = self._apply_device_response(merged, rng)
            trace["actions"]["device"] = device_trace
        else:
            trace["actions"]["device"] = {"selected": False, "applied": False}

        peak = float(np.max(np.abs(merged))) if merged.size else 0.0
        if peak > self.peak_limit:
            trace["peak_limiter_gain"] = self.peak_limit / peak
            merged = merged * (self.peak_limit / peak)
        result: list[np.ndarray] = []
        cursor = 0
        for length in lengths:
            result.append(np.asarray(merged[cursor : cursor + length], dtype=np.float32).copy())
            cursor += length
        return result, trace

    def resolve_profile(self, meta: Mapping[str, Any]) -> AugmentationProfile:
        return self.profile_resolver.resolve(meta)

    def _add_transient(
        self,
        waveform: np.ndarray,
        *,
        signal_rms: float | None,
        background_rms: float,
        rng: random.Random,
        forced_category: str | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        duration = rng.uniform(self.transient_min_seconds, self.transient_max_seconds)
        event_samples = min(len(waveform), max(1, int(duration * self.sample_rate)))
        weights = (
            {str(forced_category): 1.0}
            if forced_category
            else self.transient_category_weights
        )
        try:
            event, event_trace = self._get_sampler().sample_stream_with_trace(
                event_samples,
                rng,
                source_weights=self.source_weights,
                category_weights=weights,
            )
        except LookupError as exc:
            self._warn_once("transient", str(exc))
            return waveform, {
                "selected": True,
                "applied": False,
                "reason": str(exc),
            }
        event_rms = _rms(event)
        if event_rms <= 1e-8:
            return waveform, {
                "selected": True,
                "applied": False,
                "reason": "sampled event was silent",
            }
        reference_rms = signal_rms or max(background_rms, 10.0 ** (-30.0 / 20.0))
        sir_db = _sample_band(self.transient_sir_bands, rng)
        event = event * (reference_rms / (event_rms * (10.0 ** (sir_db / 20.0))))

        fade = min(int(0.02 * self.sample_rate), len(event) // 2)
        if fade:
            envelope = np.ones(len(event), dtype=np.float32)
            ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
            envelope[:fade] = ramp
            envelope[-fade:] = ramp[::-1]
            event *= envelope
        start = rng.randrange(len(waveform) - len(event) + 1) if len(event) < len(waveform) else 0
        result = waveform.copy()
        result[start : start + len(event)] += event
        event_trace.update(
            {
                "selected": True,
                "applied": True,
                "start_ms": round(start * 1000.0 / self.sample_rate, 3),
                "duration_ms": round(len(event) * 1000.0 / self.sample_rate, 3),
                "sir_db": sir_db,
            }
        )
        return result, event_trace

    def _sample_room_response(
        self,
        rng: random.Random,
        *,
        preset_weights: Mapping[str, float] | None = None,
        forced_preset: str | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        rows = self._get_rir_rows()
        if not rows or forced_preset is not None:
            return self._sample_parametric_room_response(
                rng,
                preset_weights=preset_weights,
                forced_preset=forced_preset,
            )
        row = dict(rng.choice(rows))
        path = Path(str(row["path"]))
        try:
            import soundfile as sf

            values, sample_rate = sf.read(path, dtype="float32", always_2d=True)
            channel = rng.randrange(values.shape[1])
            impulse = np.asarray(values[:, channel], dtype=np.float32)
            if int(sample_rate) != self.sample_rate:
                from scipy.signal import resample_poly

                divisor = math.gcd(int(sample_rate), self.sample_rate)
                impulse = resample_poly(
                    impulse,
                    up=self.sample_rate // divisor,
                    down=int(sample_rate) // divisor,
                ).astype(np.float32)
            limit = max(1, int(self.rir_max_seconds * self.sample_rate))
            impulse = impulse[:limit]
            peak = float(np.max(np.abs(impulse))) if impulse.size else 0.0
            if peak <= 1e-8:
                raise ValueError("RIR is silent")
            onset = int(np.flatnonzero(np.abs(impulse) >= peak * 0.05)[0])
            impulse = impulse[max(0, onset - 2) :]
            energy = float(np.sqrt(np.sum(np.square(impulse, dtype=np.float64))))
            impulse = np.asarray(impulse / max(energy, 1e-8), dtype=np.float32)
        except Exception as exc:
            self._warn_once(f"rir:{path}", f"Could not load RIR {path}: {exc}")
            impulse, trace = self._sample_parametric_room_response(
                rng,
                preset_weights=preset_weights,
            )
            trace["measured_rir_error"] = {"path": str(path), "reason": str(exc)}
            return impulse, trace
        return impulse, {
            "selected": True,
            "mode": "measured_rir",
            "path": str(path),
            "samples": len(impulse),
            "wet_range": list(self.room_wet),
        }

    def _sample_parametric_room_response(
        self,
        rng: random.Random,
        *,
        preset_weights: Mapping[str, float] | None = None,
        forced_preset: str | None = None,
        room_context: Mapping[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        from scipy.signal import lfilter

        context = dict(
            room_context
            or self._sample_room_context(
                rng,
                preset_weights=preset_weights,
                forced_preset=forced_preset,
            )
        )
        rt60 = float(context["rt60_seconds"])
        damping = float(context["hf_damping"])
        predelay_range = context["predelay_ms"]
        predelay_ms = rng.uniform(
            *_ordered_pair(predelay_range, "room preset predelay_ms")
        )
        predelay = max(1, int(predelay_ms * self.sample_rate / 1000.0))
        response_seconds = min(
            self.rir_max_seconds,
            max(predelay_ms / 1000.0 + 0.12, rt60 * 1.15),
        )
        response_samples = max(predelay + 2, int(response_seconds * self.sample_rate))
        impulse = np.zeros(response_samples, dtype=np.float32)

        early_count_low, early_count_high = _ordered_pair(
            context.get("early_reflection_count", [6, 11]),
            "room preset early_reflection_count",
        )
        if (
            early_count_low < 0
            or not early_count_low.is_integer()
            or not early_count_high.is_integer()
        ):
            raise ValueError(
                "room preset early_reflection_count must contain non-negative integers"
            )
        early_count = rng.randint(int(early_count_low), int(early_count_high))
        early_limit_ms = min(95.0, max(35.0, rt60 * 160.0))
        early_delays_ms: list[float] = []
        for _ in range(early_count):
            delay_ms = predelay_ms + rng.uniform(4.0, early_limit_ms)
            delay = min(response_samples - 1, int(delay_ms * self.sample_rate / 1000.0))
            decay = math.exp(-6.9078 * (delay / self.sample_rate) / max(rt60, 1e-3))
            sign = -1.0 if rng.random() < 0.35 else 1.0
            impulse[delay] += sign * rng.uniform(0.25, 0.75) * decay
            early_delays_ms.append(delay_ms)

        late_start = min(
            response_samples - 1,
            predelay + int(rng.uniform(20.0, 55.0) * self.sample_rate / 1000.0),
        )
        late_length = response_samples - late_start
        np_rng = np.random.default_rng(rng.getrandbits(64))
        diffuse = np_rng.standard_normal(late_length).astype(np.float32)
        diffuse = lfilter([1.0 - damping], [1.0, -damping], diffuse).astype(np.float32)
        time = np.arange(late_length, dtype=np.float32) / self.sample_rate
        envelope = np.exp(-6.9078 * time / max(rt60, 1e-3)).astype(np.float32)
        impulse[late_start:] += diffuse * envelope * 0.035

        energy = float(np.sqrt(np.sum(np.square(impulse, dtype=np.float64))))
        if energy <= 1e-8:
            impulse[predelay] = 1.0
            energy = 1.0
        impulse = np.asarray(impulse / energy, dtype=np.float32)
        return impulse, {
            "selected": True,
            "mode": "parametric_room",
            "preset": context.get("preset"),
            "rt60_seconds": rt60,
            "predelay_ms": predelay_ms,
            "hf_damping": damping,
            "wet_range": list(context["wet"]),
            "early_reflection_count": early_count,
            "early_reflections_ms": early_delays_ms,
            "samples": len(impulse),
        }

    def _sample_room_context(
        self,
        rng: random.Random,
        *,
        preset_weights: Mapping[str, float] | None = None,
        forced_preset: str | None = None,
    ) -> dict[str, Any]:
        preset_name: str | None = None
        preset: Mapping[str, Any] = {}
        if forced_preset is not None:
            preset_name = str(forced_preset)
            if preset_name not in self.room_presets:
                raise ValueError(f"Unknown room preset: {preset_name!r}")
            preset = self.room_presets[preset_name]
        elif self.room_presets:
            weights = dict(preset_weights or {})
            choices = [
                (
                    name,
                    float(
                        weights.get(name, 0.0)
                        if weights
                        else value.get("weight", 1.0)
                    ),
                )
                for name, value in self.room_presets.items()
            ]
            preset_name = str(_weighted_choice(choices, rng))
            preset = self.room_presets[preset_name]

        rt60_range = preset.get("rt60_seconds", self.room_rt60_seconds)
        predelay_range = preset.get("predelay_ms", self.room_predelay_ms)
        wet_range = preset.get("wet", self.room_wet)
        damping_range = preset.get("hf_damping", self.room_hf_damping)
        early_reflection_count = preset.get("early_reflection_count", [6, 11])
        return {
            "preset": preset_name,
            "rt60_seconds": rng.uniform(
                *_ordered_pair(rt60_range, "room preset rt60_seconds")
            ),
            "predelay_ms": list(
                _ordered_pair(predelay_range, "room preset predelay_ms")
            ),
            "wet": list(_ordered_pair(wet_range, "room preset wet")),
            "hf_damping": rng.uniform(
                *_ordered_pair(damping_range, "room preset hf_damping")
            ),
            "early_reflection_count": list(
                _ordered_pair(
                    early_reflection_count,
                    "room preset early_reflection_count",
                )
            ),
        }

    def _apply_room_response(
        self,
        waveform: np.ndarray,
        impulse: np.ndarray,
        rng: random.Random,
        *,
        wet_range: Sequence[float] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        dry_rms = _rms(waveform)
        if dry_rms <= 1e-8:
            return waveform, {
                "applied": False,
                "reason": "microphone stream is silent",
            }
        reflected = _convolve_causal_same_length(waveform, impulse)
        reflected_rms = _rms(reflected)
        if reflected_rms <= 1e-8:
            return waveform, {
                "applied": False,
                "reason": "room response produced no reflected energy",
            }
        wet = rng.uniform(
            *_ordered_pair(wet_range or self.room_wet, "room_wet")
        )
        reflected *= float(dry_rms / reflected_rms)
        result = waveform * (1.0 - wet) + reflected * wet
        return np.asarray(result, dtype=np.float32), {
            "applied": True,
            "wet": wet,
        }

    def _apply_multiparty_spatial_response(
        self,
        waveform: np.ndarray,
        refs: Sequence[Any],
        lengths: Sequence[int],
        rng: random.Random,
        *,
        room_preset_weights: Mapping[str, float],
        distance_weights: Mapping[str, float],
        forced_room_preset: str | None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        masks = _speaker_activity_masks(
            refs,
            lengths,
            sample_rate=self.sample_rate,
        )
        if len(masks) < 2:
            return waveform, {
                "selected": True,
                "applied": False,
                "mode": "multiparty_spatial_room",
                "reason": "fewer than two identifiable active speakers",
            }
        if not self.speaker_distance_presets:
            return waveform, {
                "selected": True,
                "applied": False,
                "mode": "multiparty_spatial_room",
                "reason": "speaker_distance_presets is empty",
            }

        room_context = self._sample_room_context(
            rng,
            preset_weights=room_preset_weights,
            forced_preset=forced_room_preset,
        )
        speaker_ids = sorted(masks)
        assignments = _sample_weighted_assignments(
            speaker_ids,
            self.speaker_distance_presets,
            distance_weights,
            rng,
        )
        occupancy = np.zeros(len(waveform), dtype=np.int16)
        for mask in masks.values():
            occupancy += mask.astype(np.int16)
        result = np.asarray(waveform, dtype=np.float32).copy()
        result[occupancy == 1] = 0.0
        speaker_traces: list[dict[str, Any]] = []

        from scipy.signal import butter, sosfilt

        for speaker_id in speaker_ids:
            unique_mask = masks[speaker_id] & (occupancy == 1)
            active_samples = int(np.count_nonzero(unique_mask))
            if active_samples == 0:
                continue
            distance_name = assignments[speaker_id]
            distance = self.speaker_distance_presets[distance_name]
            stem = np.zeros_like(waveform, dtype=np.float32)
            stem[unique_mask] = waveform[unique_mask]
            impulse, room_trace = self._sample_parametric_room_response(
                rng,
                room_context=room_context,
            )
            wet_scale = rng.uniform(
                *_ordered_pair(
                    distance.get("wet_scale", [1.0, 1.0]),
                    f"speaker distance {distance_name} wet_scale",
                )
            )
            wet_range = [
                min(max(float(value) * wet_scale, 0.0), 0.72)
                for value in room_context["wet"]
            ]
            stem, applied_trace = self._apply_room_response(
                stem,
                impulse,
                rng,
                wet_range=wet_range,
            )
            lowpass_hz = rng.uniform(
                *_ordered_pair(
                    distance.get(
                        "lowpass_hz",
                        [self.sample_rate * 0.45, self.sample_rate * 0.49],
                    ),
                    f"speaker distance {distance_name} lowpass_hz",
                )
            )
            lowpass_hz = min(max(lowpass_hz, 1000.0), self.sample_rate * 0.49)
            stem = sosfilt(
                butter(
                    2,
                    lowpass_hz,
                    btype="lowpass",
                    fs=self.sample_rate,
                    output="sos",
                ),
                stem,
            ).astype(np.float32)
            gain_db = rng.uniform(
                *_ordered_pair(
                    distance.get("gain_db", [0.0, 0.0]),
                    f"speaker distance {distance_name} gain_db",
                )
            )
            stem *= float(10.0 ** (gain_db / 20.0))
            result += stem
            distance_m = rng.uniform(
                *_ordered_pair(
                    distance.get("distance_m", [1.0, 1.0]),
                    f"speaker distance {distance_name} distance_m",
                )
            )
            speaker_traces.append(
                {
                    "speaker_id": speaker_id,
                    "distance_preset": distance_name,
                    "distance_m": distance_m,
                    "gain_db": gain_db,
                    "lowpass_hz": lowpass_hz,
                    "active_samples": active_samples,
                    **room_trace,
                    **applied_trace,
                }
            )

        if len(speaker_traces) < 2:
            return waveform, {
                "selected": True,
                "applied": False,
                "mode": "multiparty_spatial_room",
                "reason": "fewer than two non-overlapped speaker stems",
            }
        return np.asarray(result, dtype=np.float32), {
            "selected": True,
            "applied": True,
            "mode": "multiparty_spatial_room",
            "preset": room_context.get("preset"),
            "rt60_seconds": room_context["rt60_seconds"],
            "hf_damping": room_context["hf_damping"],
            "wet": float(
                sum(float(item["wet"]) for item in speaker_traces)
                / len(speaker_traces)
            ),
            "speakers": speaker_traces,
            "overlap_samples_left_unmodified": int(np.count_nonzero(occupancy > 1)),
        }

    def _get_rir_rows(self) -> list[dict[str, Any]]:
        if self._rir_rows is not None:
            return self._rir_rows
        self._rir_rows = []
        if not self.rir_index_path:
            return self._rir_rows
        with Path(self.rir_index_path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                path = row.get("path")
                if not path:
                    raise ValueError(
                        f"RIR index row {self.rir_index_path}:{line_number} has no path"
                    )
                self._rir_rows.append(dict(row))
        return self._rir_rows

    def _apply_device_response(
        self,
        waveform: np.ndarray,
        rng: random.Random,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        from scipy.signal import butter, sosfilt

        gain_db = rng.uniform(*_ordered_pair(self.device_gain_db, "device_gain_db"))
        low_hz = rng.uniform(*_ordered_pair(self.device_low_cut_hz, "device_low_cut_hz"))
        high_hz = rng.uniform(*_ordered_pair(self.device_high_cut_hz, "device_high_cut_hz"))
        high_hz = min(high_hz, self.sample_rate * 0.49)
        low_hz = min(low_hz, high_hz * 0.5)
        compression = rng.uniform(
            *_ordered_pair(self.device_compression, "device_compression")
        )
        filtered = sosfilt(
            butter(
                2,
                [low_hz, high_hz],
                btype="bandpass",
                fs=self.sample_rate,
                output="sos",
            ),
            waveform,
        ).astype(np.float32)
        if compression > 1.0001:
            filtered = np.tanh(filtered * compression) / math.tanh(compression)
        filtered *= float(10.0 ** (gain_db / 20.0))
        return filtered, {
            "selected": True,
            "applied": True,
            "gain_db": gain_db,
            "low_cut_hz": low_hz,
            "high_cut_hz": high_hz,
            "compression": compression,
        }

    def _build_causal_echo(
        self,
        refs: Sequence[Any],
        lengths: Sequence[int],
        sample_meta: Mapping[str, Any],
        *,
        rng: random.Random,
        room_impulse: np.ndarray | None,
        room_preset_weights: Mapping[str, float],
    ) -> tuple[np.ndarray | None, dict[str, Any]]:
        playback_segments = sample_meta.get("assistant_playback_segments") or []
        if not isinstance(playback_segments, list) or not playback_segments:
            return None, {
                "selected": True,
                "applied": False,
                "reason": "sample has no isolated assistant playback stem",
            }

        ref_ranges = [_ref_timeline_range(ref, length, self.sample_rate) for ref, length in zip(refs, lengths)]
        timeline_end_ms = max((end for _, end in ref_ranges), default=0.0)
        timeline_samples = max(1, int(math.ceil(timeline_end_ms * self.sample_rate / 1000.0)))
        playback = np.zeros(timeline_samples, dtype=np.float32)
        used: list[dict[str, Any]] = []
        from mcpmft.data.feature import load_audio_ref_waveform
        from mcpmft.data.sample import AudioRef

        for item in playback_segments:
            if not isinstance(item, Mapping) or not item.get("path"):
                continue
            timeline_start_ms = float(item.get("timeline_start_ms") or 0.0)
            timeline_end = item.get("timeline_end_ms")
            source_start_ms = item.get("source_start_ms")
            source_end_ms = item.get("source_end_ms")
            ref = AudioRef(
                path=str(item["path"]),
                start_ms=int(source_start_ms) if source_start_ms is not None else None,
                end_ms=int(source_end_ms) if source_end_ms is not None else None,
                channel=item.get("channel"),
            )
            try:
                values = np.asarray(load_audio_ref_waveform(ref), dtype=np.float32)
            except Exception as exc:
                self._warn_once(
                    f"echo:{item['path']}",
                    f"Could not load assistant playback stem {item['path']}: {exc}",
                )
                continue
            if timeline_end is not None:
                wanted = max(
                    0,
                    int((float(timeline_end) - timeline_start_ms) * self.sample_rate / 1000.0),
                )
                values = values[:wanted]
            start = max(0, int(timeline_start_ms * self.sample_rate / 1000.0))
            take = min(len(values), max(0, timeline_samples - start))
            if take <= 0:
                continue
            playback[start : start + take] += values[:take]
            used.append(
                {
                    "path": str(item["path"]),
                    "timeline_start_ms": timeline_start_ms,
                    "used_samples": take,
                }
            )
        playback_rms = _rms(playback)
        if playback_rms <= 1e-8:
            return None, {
                "selected": True,
                "applied": False,
                "reason": "assistant playback stems were unreadable or silent",
            }

        delay_ms = rng.uniform(*_ordered_pair(self.echo_delay_ms, "echo_delay_ms"))
        delay_samples = max(0, int(delay_ms * self.sample_rate / 1000.0))
        leaked = np.zeros_like(playback)
        if delay_samples < len(playback):
            leaked[delay_samples:] = playback[: len(playback) - delay_samples]
        direct_rms = _rms(leaked)
        if room_impulse is None:
            echo_room, echo_room_trace = self._sample_room_response(
                rng,
                preset_weights=room_preset_weights,
            )
        else:
            echo_room = room_impulse
            echo_room_trace = {"reused_microphone_room_response": True}
        reflected = _convolve_causal_same_length(leaked, echo_room)
        reflected_rms = _rms(reflected)
        room_mix = rng.uniform(*_ordered_pair(self.echo_room_mix, "echo_room_mix"))
        if direct_rms > 1e-8 and reflected_rms > 1e-8:
            reflected *= float(direct_rms / reflected_rms)
            leaked = leaked + reflected * room_mix
        leaked = np.tanh(leaked * 1.15).astype(np.float32)
        erl_db = rng.uniform(*_ordered_pair(self.echo_erl_db, "echo_erl_db"))
        leaked *= float(playback_rms / max(_rms(leaked), 1e-8) * 10.0 ** (-erl_db / 20.0))
        playback_onsets = np.flatnonzero(np.abs(playback) > 1e-8)
        if playback_onsets.size:
            causal_onset = min(len(leaked), int(playback_onsets[0]) + delay_samples)
            leaked[:causal_onset] = 0.0
        leaked[np.abs(leaked) < 1e-8] = 0.0

        pieces: list[np.ndarray] = []
        for (start_ms, end_ms), length in zip(ref_ranges, lengths):
            start = int(start_ms * self.sample_rate / 1000.0)
            end = start + length
            piece = leaked[start:end]
            if len(piece) < length:
                piece = np.pad(piece, (0, length - len(piece)))
            pieces.append(np.asarray(piece, dtype=np.float32))
        echo = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)
        return echo, {
            "selected": True,
            "applied": True,
            "delay_ms": delay_ms,
            "erl_db": erl_db,
            "playback_segments": used,
            "room_mix": room_mix,
            "room": echo_room_trace,
        }

    def _get_sampler(self) -> NoiseStreamSampler:
        if self._sampler is None:
            if not self.index_path:
                raise ValueError(
                    "audio_augment.noise_index_path is required when augmentation is enabled"
                )
            catalog = NoiseCatalog.from_jsonl(self.index_path)
            self._sampler = NoiseStreamSampler(
                catalog,
                archive_cache_dir=self.archive_cache_dir,
                sample_rate=self.sample_rate,
                crossfade_ms=self.crossfade_ms,
                max_segment_seconds=self.max_noise_segment_seconds,
            )
        return self._sampler

    def _warn_once(self, key: str, message: str) -> None:
        if key not in self._warned:
            LOGGER.warning("Audio augmentation: %s", message)
            self._warned.add(key)


def _select_action(
    probability: float,
    rng: random.Random,
    forced: Any,
) -> bool:
    if forced is not None:
        return bool(forced)
    return probability > 0.0 and rng.random() < probability


def _ordered_pair(values: Sequence[float], field_name: str) -> tuple[float, float]:
    if len(values) != 2:
        raise ValueError(f"{field_name} must contain exactly [min, max]")
    low, high = float(values[0]), float(values[1])
    if not math.isfinite(low) or not math.isfinite(high) or high < low:
        raise ValueError(f"Invalid {field_name}: {values!r}")
    return low, high


def _ref_timeline_range(
    ref: Any,
    length: int,
    sample_rate: int,
) -> tuple[float, float]:
    source = getattr(ref, "source", None) or {}
    start = source.get("timeline_start_ms")
    end = source.get("timeline_end_ms")
    if start is None:
        start = source.get("block_index", 0) * 1000.0
    start = float(start)
    if end is None:
        end = start + length * 1000.0 / sample_rate
    return start, float(end)


def _speaker_activity_masks(
    refs: Sequence[Any],
    lengths: Sequence[int],
    *,
    sample_rate: int,
) -> dict[str, np.ndarray]:
    total_samples = sum(int(length) for length in lengths)
    masks: dict[str, np.ndarray] = {}
    cursor = 0
    for ref, length in zip(refs, lengths):
        length = int(length)
        source = getattr(ref, "source", None) or {}
        segments = source.get("segments") if source.get("kind") == "block_mix" else None
        if isinstance(segments, list):
            for segment in segments:
                if not isinstance(segment, Mapping):
                    continue
                speaker_id = segment.get("speaker_id")
                nested_ref = segment.get("ref")
                nested_source = getattr(nested_ref, "source", None) or {}
                if speaker_id is None and isinstance(nested_ref, Mapping):
                    nested_source = nested_ref.get("source") or {}
                if speaker_id is None:
                    speaker_id = nested_source.get("speaker_id")
                if speaker_id is None:
                    continue
                start = cursor + max(
                    0,
                    int(float(segment.get("offset_ms") or 0.0) * sample_rate / 1000.0),
                )
                duration = max(
                    0,
                    int(
                        float(segment.get("duration_ms") or 0.0)
                        * sample_rate
                        / 1000.0
                    ),
                )
                end = min(cursor + length, start + duration)
                if end <= start:
                    continue
                key = str(speaker_id)
                masks.setdefault(key, np.zeros(total_samples, dtype=bool))[start:end] = True
        else:
            speaker_id = source.get("speaker_id")
            if speaker_id is not None:
                masks.setdefault(
                    str(speaker_id),
                    np.zeros(total_samples, dtype=bool),
                )[cursor : cursor + length] = True
        cursor += length
    return masks


def _sample_weighted_assignments(
    keys: Sequence[str],
    presets: Mapping[str, Mapping[str, Any]],
    configured_weights: Mapping[str, float],
    rng: random.Random,
) -> dict[str, str]:
    explicit = dict(configured_weights)
    base = [
        (
            name,
            float(explicit.get(name, 0.0) if explicit else value.get("weight", 1.0)),
        )
        for name, value in presets.items()
    ]
    if not any(weight > 0.0 for _, weight in base):
        raise ValueError("speaker distance presets require a positive total weight")
    available = list(base)
    result: dict[str, str] = {}
    for key in keys:
        if not any(weight > 0.0 for _, weight in available):
            available = list(base)
        selected = str(_weighted_choice(available, rng))
        result[str(key)] = selected
        available = [
            (name, weight) for name, weight in available if name != selected
        ]
    return result


def _convolve_causal_same_length(
    waveform: np.ndarray,
    impulse: np.ndarray,
) -> np.ndarray:
    from scipy.signal import fftconvolve

    if waveform.size == 0 or impulse.size == 0:
        return np.asarray(waveform, dtype=np.float32)
    result = fftconvolve(waveform, impulse, mode="full")[: len(waveform)]
    # FFT round-off can leave ~1e-10 energy before a mathematically causal onset. Clamp it so
    # serialized far-end echo has a strict no-future-leak contract, not just an approximate one.
    result[np.abs(result) < 1e-8] = 0.0
    return np.asarray(result, dtype=np.float32)


def _complete_cached_file(path: Path, expected_size: int | None) -> bool:
    if not path.is_file():
        return False
    return expected_size is None or path.stat().st_size == expected_size


def _weighted_choice(
    values: Sequence[tuple[Any, float]],
    rng: random.Random,
) -> Any:
    total = sum(max(float(weight), 0.0) for _, weight in values)
    if total <= 0:
        raise ValueError("Weighted choice requires a positive total weight")
    point = rng.random() * total
    cumulative = 0.0
    for value, weight in values:
        cumulative += max(float(weight), 0.0)
        if point < cumulative:
            return value
    return values[-1][0]


def _safe_exception_text(exc: BaseException) -> str:
    """Format third-party exceptions without trusting their occasionally broken ``__str__``."""

    try:
        message = str(exc)
    except BaseException:
        message = "<exception str() failed>"
    return f"{type(exc).__name__}: {message}"


def _sample_band(bands: Sequence[Sequence[float]], rng: random.Random) -> float:
    choices: list[tuple[tuple[float, float], float]] = []
    for band in bands:
        if len(band) != 3:
            raise ValueError(f"SNR/SIR band must be [min, max, weight], got {band!r}")
        low, high, weight = (float(value) for value in band)
        if high < low or weight < 0:
            raise ValueError(f"Invalid SNR/SIR band: {band!r}")
        choices.append(((low, high), weight))
    low, high = _weighted_choice(choices, rng)
    return rng.uniform(low, high)


def _is_active_ref(ref: Any) -> bool:
    source = getattr(ref, "source", None) or {}
    kind = source.get("kind")
    if kind == "silence":
        return False
    if kind == "block_mix":
        return bool(source.get("segments"))
    return True


def _robust_signal_rms(waveforms: Sequence[np.ndarray]) -> float | None:
    frame_size = 320
    frame_values: list[np.ndarray] = []
    for waveform in waveforms:
        values = np.asarray(waveform, dtype=np.float32)
        if values.size == 0:
            continue
        usable = len(values) - (len(values) % frame_size)
        if usable:
            frames = values[:usable].reshape(-1, frame_size)
            rms = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1))
        else:
            rms = np.asarray([_rms(values)], dtype=np.float64)
        frame_values.append(rms[rms > 1e-5])
    nonempty = [values for values in frame_values if values.size]
    if not nonempty:
        return None
    return float(np.quantile(np.concatenate(nonempty), 0.70))


def _rms(waveform: np.ndarray) -> float:
    values = np.asarray(waveform, dtype=np.float64)
    if values.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(values))))
