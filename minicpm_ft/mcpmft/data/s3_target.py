from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable

import numpy as np

from mcpmft.data.sample import AudioRef


S3_READY_FILENAME = "READY.json"


def manifest_identities(
    manifest_paths: Iterable[str | Path],
    release_root: str | Path | None,
) -> list[str]:
    root = Path(release_root).resolve() if release_root else None
    identities = []
    for value in manifest_paths:
        path = Path(value).resolve()
        if root is not None:
            try:
                identities.append(path.relative_to(root).as_posix())
                continue
            except ValueError:
                pass
        identities.append(str(path))
    return identities


def require_complete_s3_cache(
    cache_dir: str | Path,
    *,
    manifest_paths: Iterable[str | Path],
    release_root: str | Path | None,
) -> dict:
    marker = Path(cache_dir) / S3_READY_FILENAME
    if not marker.is_file():
        raise FileNotFoundError(
            f"Talker S3 cache is not complete: {marker}. Run ./prepare_data.sh s3 "
            "<release>/talker/train_config.yaml before training."
        )
    value = json.loads(marker.read_text(encoding="utf-8"))
    expected = manifest_identities(manifest_paths, release_root)
    if value.get("status") != "ready" or value.get("manifests") != expected:
        raise ValueError(f"Talker S3 cache marker does not match this training view: {marker}")
    return value


class S3TokenCache:
    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, audio_ref: AudioRef) -> Path:
        return self.path_for_id(audio_ref.id())

    def path_for_id(self, audio_ref_id: str) -> Path:
        digest = hashlib.sha1(audio_ref_id.encode("utf-8")).hexdigest()
        # Partition large caches by one hash-prefix level.
        return self.cache_dir / digest[:2] / f"{digest[2:24]}.npy"

    def get(self, audio_ref: AudioRef) -> list[int] | None:
        return self._load_path(self.path_for(audio_ref))

    def get_by_id(self, audio_ref_id: str) -> list[int] | None:
        return self._load_path(self.path_for_id(audio_ref_id))

    def contains(self, audio_ref: AudioRef) -> bool:
        return self.path_for(audio_ref).is_file()

    def length(self, audio_ref: AudioRef) -> int | None:
        """Read only the array shape when only the S3 count is needed."""
        return self._load_length(self.path_for(audio_ref))

    def _load_path(self, path: Path) -> list[int] | None:
        if not path.exists():
            return None
        return np.load(path).astype(np.int64).tolist()

    @staticmethod
    def _load_length(path: Path) -> int | None:
        if not path.exists():
            return None
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        try:
            return int(values.size)
        finally:
            # Release each mmap before opening the next cache file.
            mmap = getattr(values, "_mmap", None)
            if mmap is not None:
                mmap.close()

    def put(self, audio_ref: AudioRef, codes: Iterable[int]) -> Path:
        path = self.path_for(audio_ref)
        path.parent.mkdir(parents=True, exist_ok=True)
        arr = np.asarray(list(codes), dtype=np.int16)
        tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
        try:
            with tmp.open("wb") as handle:
                np.save(handle, arr)
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)
        return path


class S3TokenizerExtractor:
    LONG_AUDIO_MEL_FRAMES = 3000
    MAX_PADDED_MEL_FRAMES_PER_BATCH = 24000

    def __init__(
        self,
        model_name_or_path: str = "speech_tokenizer_v2_25hz",
        *,
        download_root: str | None = None,
        device: str | None = None,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.download_root = download_root
        self.device = device
        self._model = None
        self._s3 = None

    def _load(self):
        if self._model is not None:
            return self._s3, self._model
        try:
            import s3tokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Talker target generation requires minicpmo-utils[tts]"
            ) from exc
        kwargs = {}
        if self.download_root:
            kwargs["download_root"] = self.download_root
        model = s3tokenizer.load_model(self.model_name_or_path, **kwargs)
        if self.device is None:
            import torch

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device != "cpu":
            model = model.to(self.device)
        model.eval()
        self._model = model
        self._s3 = s3tokenizer
        return self._s3, self._model

    @staticmethod
    def _load_waveform(audio_ref: AudioRef):
        from mcpmft.data.feature import load_audio_ref_waveform

        return load_audio_ref_waveform(audio_ref)

    def extract_many(
        self,
        audio_refs: list[AudioRef],
        *,
        audio_workers: int = 1,
    ) -> list[list[int]]:
        if not audio_refs:
            return []
        import torch

        s3, model = self._load()
        if audio_workers > 1 and len(audio_refs) > 1:
            with ThreadPoolExecutor(max_workers=audio_workers) as pool:
                waveforms = list(pool.map(self._load_waveform, audio_refs))
        else:
            waveforms = [self._load_waveform(ref) for ref in audio_refs]
        mels = [
            s3.log_mel_spectrogram(
                waveform if torch.is_tensor(waveform) else torch.from_numpy(waveform)
            )
            for waveform in waveforms
        ]
        device = next(model.parameters()).device
        results: list[list[int] | None] = [None] * len(mels)

        def quantize(indices: list[int]) -> None:
            padded, lengths = s3.padding([mels[index] for index in indices])
            with torch.inference_mode():
                codes, code_lengths = model.quantize(
                    padded.to(device), lengths.to(device)
                )
            values = codes.detach().cpu().numpy()
            for local_index, original_index in enumerate(indices):
                results[original_index] = values[
                    local_index, : int(code_lengths[local_index])
                ].astype(np.int64).tolist()

        long_indices = [
            index
            for index, mel in enumerate(mels)
            if int(mel.shape[-1]) > self.LONG_AUDIO_MEL_FRAMES
        ]
        long_set = set(long_indices)
        short_indices = sorted(
            (index for index in range(len(mels)) if index not in long_set),
            key=lambda index: int(mels[index].shape[-1]),
        )
        batch: list[int] = []
        batch_max_frames = 0
        for index in short_indices:
            frames = int(mels[index].shape[-1])
            next_max = max(batch_max_frames, frames)
            if batch and next_max * (len(batch) + 1) > self.MAX_PADDED_MEL_FRAMES_PER_BATCH:
                quantize(batch)
                batch = []
                batch_max_frames = 0
            batch.append(index)
            batch_max_frames = max(batch_max_frames, frames)
        if batch:
            quantize(batch)
        for index in long_indices:
            quantize([index])

        if any(value is None for value in results):
            raise RuntimeError("S3 extraction did not produce every target")
        return [value for value in results if value is not None]
