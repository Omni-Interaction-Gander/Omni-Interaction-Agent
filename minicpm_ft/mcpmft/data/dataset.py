from __future__ import annotations

import hashlib
import itertools
import json
import random
from pathlib import Path
from typing import Iterable, Sequence

from mcpmft.data.media import ReleaseMediaResolver
from mcpmft.data.sample import OmniSample
from mcpmft.utils.dist import get_dist_info
from mcpmft.utils.io import read_jsonl


def _iterable_dataset_base():
    from torch.utils.data import IterableDataset

    return IterableDataset


_IterableBase = _iterable_dataset_base()


class ManifestDataset(_IterableBase):
    def __init__(
        self,
        manifest_path: str | Path | Sequence[str | Path],
        *,
        shard_by_rank: bool = True,
        shard_by_worker: bool = True,
        max_audio_seconds: float | None = None,
        mix_strategy: str = "concat",
        mix_weights: Sequence[float] | None = None,
        manifest_row_counts: Sequence[int] | None = None,
        manifest_sample_counts: Sequence[int] | None = None,
        shuffle: bool = False,
        seed: int = 42,
        shuffle_block_bytes: int = 256 * 1024,
        media_resolver: ReleaseMediaResolver | None = None,
    ) -> None:
        super().__init__()
        if isinstance(manifest_path, (str, Path)):
            self.manifest_paths = [Path(manifest_path)]
        else:
            self.manifest_paths = [Path(path) for path in manifest_path]
        if not self.manifest_paths:
            raise ValueError("ManifestDataset requires at least one manifest")
        missing = [path for path in self.manifest_paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Manifest files do not exist: {missing}")
        if mix_strategy not in {"concat", "smooth"}:
            raise ValueError(f"Unsupported manifest mix strategy: {mix_strategy}")
        self.mix_strategy = mix_strategy
        self.mix_weights = list(mix_weights or [])
        if self.mix_weights and len(self.mix_weights) != len(self.manifest_paths):
            raise ValueError("manifest mix weights must match manifest paths")
        if any(weight <= 0 for weight in self.mix_weights):
            raise ValueError("manifest mix weights must be positive")
        row_counts = list(manifest_row_counts or [])
        if row_counts and len(row_counts) != len(self.manifest_paths):
            raise ValueError("manifest row counts must match manifest paths")
        if any(count < 0 for count in row_counts):
            raise ValueError("manifest row counts must be non-negative")
        sample_counts = list(manifest_sample_counts or [])
        if sample_counts and len(sample_counts) != len(self.manifest_paths):
            raise ValueError("manifest sample counts must match manifest paths")
        if any(count < 0 for count in sample_counts):
            raise ValueError("manifest sample counts must be non-negative")
        if row_counts and sample_counts:
            oversized = [
                (index, sampled, available)
                for index, (sampled, available) in enumerate(zip(sample_counts, row_counts))
                if sampled > available
            ]
            if oversized:
                raise ValueError(
                    "manifest sample counts cannot exceed audited row counts: "
                    f"{oversized}"
                )
        if sample_counts and max_audio_seconds is not None:
            raise ValueError(
                "manifest sample counts require max_audio_seconds=null so every source emits "
                "the configured exact count"
            )
        if shuffle_block_bytes <= 0:
            raise ValueError("shuffle_block_bytes must be positive")
        self.shard_by_rank = shard_by_rank
        self.shard_by_worker = shard_by_worker
        self.max_audio_seconds = max_audio_seconds
        self.shuffle = shuffle
        self.seed = seed
        self.shuffle_block_bytes = shuffle_block_bytes
        self.media_resolver = media_resolver
        self.epoch = 0
        self.dist = get_dist_info()
        self._unsharded_length: int | None = None
        self._source_lengths: list[int] | None = row_counts or None
        self._sample_counts: list[int] = sample_counts

    def __len__(self) -> int:
        if self._unsharded_length is None:
            self._unsharded_length = self._count_filtered_rows()
        total = self._unsharded_length
        if not self.shard_by_rank:
            return total
        dist = get_dist_info()
        world_size = max(dist.world_size, 1)
        if dist.rank >= total:
            return 0
        return (total - dist.rank + world_size - 1) // world_size

    def _count_filtered_rows(self) -> int:
        if self.max_audio_seconds is None:
            # Length discovery runs once on every rank. Avoid parsing potentially hundreds of MB
            # of JSON when no content-based duration filter is requested.
            return sum(self._get_effective_source_lengths())

        from mcpmft.data.sample import sample_total_audio_ms

        cap_ms = self.max_audio_seconds * 1000
        return sum(
            1
            for path in self.manifest_paths
            for row in read_jsonl(path)
            if sample_total_audio_ms(OmniSample.from_dict(row)) <= cap_ms
        )

    def _get_source_lengths(self) -> list[int]:
        if self._source_lengths is None:
            lengths = []
            for path in self.manifest_paths:
                lengths.append(sum(1 for _ in read_jsonl(path)))
            self._source_lengths = lengths
        return self._source_lengths

    def _get_effective_source_lengths(self) -> list[int]:
        lengths = self._get_source_lengths()
        if not self._sample_counts:
            return lengths
        oversized = [
            (index, sampled, available)
            for index, (sampled, available) in enumerate(zip(self._sample_counts, lengths))
            if sampled > available
        ]
        if oversized:
            raise ValueError(
                "manifest sample counts cannot exceed source row counts: "
                f"{oversized}"
            )
        return list(self._sample_counts)

    def set_epoch(self, epoch: int) -> None:
        """Select a deterministic permutation for this epoch.

        Accelerate forwards ``DataLoader.set_epoch`` to iterable datasets. Keeping the seed a
        pure function of epoch makes Trainer's normal data-skip resume reproduce the exact order.
        """
        self.epoch = int(epoch)

    def _iter_raw_rows(self) -> Iterable[dict]:
        if len(self.manifest_paths) == 1 or self.mix_strategy == "concat":
            for source_index, path in enumerate(self.manifest_paths):
                yield from self._iter_sampled_source_rows(path, source_index)
            return

        lengths = self._get_effective_source_lengths()
        weights = self.mix_weights or [float(length) for length in lengths]
        iterators = [
            iter(self._iter_sampled_source_rows(path, source_index))
            for source_index, path in enumerate(self.manifest_paths)
        ]
        emitted = [0] * len(iterators)
        active = {index for index, length in enumerate(lengths) if length > 0}
        while active:
            # Weighted fair queuing: pick the source with the least normalized progress.  With
            # default weights=row counts, all sources finish together and stay evenly interleaved.
            index = min(active, key=lambda idx: (emitted[idx] / weights[idx], idx))
            try:
                row = next(iterators[index])
            except StopIteration:
                active.remove(index)
                continue
            emitted[index] += 1
            yield row

    def _iter_sampled_source_rows(
        self, path: Path, source_index: int = 0
    ) -> Iterable[dict]:
        rows = self._iter_source_rows(path, source_index)
        if not self._sample_counts:
            yield from rows
            return
        yield from itertools.islice(rows, self._sample_counts[source_index])

    def _iter_source_rows(self, path: Path, source_index: int = 0) -> Iterable[dict]:
        if not self.shuffle:
            yield from read_jsonl(path)
            return
        source_seed = _stable_source_seed(self.seed, self.epoch, source_index, path)
        yield from _iter_shuffled_jsonl_blocks(
            path,
            seed=source_seed,
            block_bytes=self.shuffle_block_bytes,
        )

    def __iter__(self) -> Iterable[OmniSample]:
        from mcpmft.data.sample import sample_total_audio_ms

        dist = get_dist_info()
        worker_id, num_workers = _worker_shard()
        rank_world = max(dist.world_size, 1) if self.shard_by_rank else 1
        rank_id = dist.rank if self.shard_by_rank else 0
        worker_world = num_workers if self.shard_by_worker else 1
        worker_id = worker_id if self.shard_by_worker else 0
        total_shards = rank_world * worker_world
        shard_id = rank_id * worker_world + worker_id
        cap_ms = None if self.max_audio_seconds is None else self.max_audio_seconds * 1000
        kept = 0  # index AFTER filtering, so rank shards stay balanced (counts differ by <=1)
        for row in self._iter_raw_rows():
            sample = OmniSample.from_dict(row)
            if self.media_resolver is not None:
                self.media_resolver.resolve_sample(sample)
            if cap_ms is not None and sample_total_audio_ms(sample) > cap_ms:
                continue  # drop over-long sample (bounds OOM / per-rank length variance)
            if total_shards > 1 and kept % total_shards != shard_id:
                kept += 1
                continue
            kept += 1
            yield sample


def _stable_source_seed(seed: int, epoch: int, source_index: int, path: Path) -> int:
    identity = f"{source_index}\0{path}".encode("utf-8")
    path_bits = int.from_bytes(hashlib.blake2b(identity, digest_size=8).digest(), "little")
    return (int(seed) + int(epoch) * 0x9E3779B97F4A7C15 + path_bits) & ((1 << 64) - 1)


def _iter_shuffled_jsonl_blocks(
    path: Path,
    *,
    seed: int,
    block_bytes: int,
) -> Iterable[dict]:
    """Read every JSONL row once in a deterministic, bounded-memory shuffled order.

    Equal byte ranges partition the file by each row's starting offset. Byte-range order and
    rows within each range are shuffled. This samples across the whole file immediately without
    writing an index or modifying the source manifest.
    """
    file_size = path.stat().st_size
    if file_size == 0:
        return
    rng = random.Random(seed)
    block_ids = list(range((file_size + block_bytes - 1) // block_bytes))
    rng.shuffle(block_ids)

    with path.open("rb") as handle:
        for block_id in block_ids:
            start = block_id * block_bytes
            end = min(start + block_bytes, file_size)
            handle.seek(start)
            if start:
                handle.seek(start - 1)
                if handle.read(1) != b"\n":
                    handle.readline()

            rows: list[dict] = []
            while handle.tell() < end:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise ValueError(f"Invalid JSONL at {path} byte offset {offset}: {exc}") from exc
                if not isinstance(value, dict):
                    raise ValueError(
                        f"JSONL row must be an object at {path} byte offset {offset}"
                    )
                rows.append(value)
            rng.shuffle(rows)
            yield from rows


def validate_training_schedule(dataset, *, max_steps: int, dry_run: bool = False) -> None:
    """Fail before model loading when epoch training has no knowable dataset length."""
    if dry_run or max_steps > 0:
        return
    try:
        sample_count = len(dataset)
    except TypeError as exc:
        raise ValueError(
            "Epoch-based training requires finite manifest row counts."
        ) from exc
    if sample_count <= 0:
        raise ValueError("Epoch-based training cannot use an empty manifest dataset.")


def _worker_shard() -> tuple[int, int]:
    try:
        from torch.utils.data import get_worker_info
    except ImportError:
        return 0, 1
    info = get_worker_info()
    if info is None:
        return 0, 1
    return int(info.id), max(int(info.num_workers), 1)
