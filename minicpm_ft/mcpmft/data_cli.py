from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

from mcpmft.args import ProjectConfig, load_project_document
from mcpmft.data.media import ReleaseMediaResolver
from mcpmft.data.s3_target import (
    S3_READY_FILENAME,
    S3TokenCache,
    S3TokenizerExtractor,
    manifest_identities,
    require_complete_s3_cache,
)
from mcpmft.data.sample import OmniSample
from mcpmft.utils.io import read_jsonl


def _project(paths: list[str]) -> ProjectConfig:
    return ProjectConfig.from_dict(load_project_document(paths))


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _iter_jsonl_byte_shard(path: Path, *, num_shards: int, shard_index: int):
    if num_shards == 1:
        yield from read_jsonl(path)
        return
    size = path.stat().st_size
    start = size * shard_index // num_shards
    end = size * (shard_index + 1) // num_shards
    with path.open("rb") as handle:
        if start:
            handle.seek(start - 1)
            if handle.read(1) != b"\n":
                handle.readline()
        while True:
            offset = handle.tell()
            if shard_index < num_shards - 1 and offset >= end:
                break
            raw = handle.readline()
            if not raw:
                break
            if not raw.strip():
                continue
            try:
                value = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid JSONL at {path} byte offset {offset}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row must be an object at {path} byte offset {offset}")
            yield value


def _check(config: ProjectConfig) -> dict:
    data = config.data
    missing = [path for path in data.manifest_paths if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"Training manifests not found: {missing[:10]}")
    if data.release_root and not Path(data.release_root).is_dir():
        raise FileNotFoundError(f"Release root not found: {data.release_root}")
    if data.public_video_root and not Path(data.public_video_root).is_dir():
        raise FileNotFoundError(
            f"Public video root not found: {data.public_video_root}. Arrange each source at "
            "<root>/<public-namespace>/<relative-video-path>."
        )
    if (
        data.frontbrain_business_tool_catalog_path
        and not Path(data.frontbrain_business_tool_catalog_path).is_file()
    ):
        raise FileNotFoundError(
            f"Business tool catalog not found: {data.frontbrain_business_tool_catalog_path}"
        )
    if config.train.audio_loss_weight > 0 and data.s3_cache_dir:
        require_complete_s3_cache(
            data.s3_cache_dir,
            manifest_paths=data.manifest_paths,
            release_root=data.release_root,
        )

    resolver = ReleaseMediaResolver(
        release_root=data.release_root,
        public_video_root=data.public_video_root,
    )
    sampled = 0
    missing_media: list[str] = []
    for manifest in data.manifest_paths:
        first = next(iter(read_jsonl(manifest)), None)
        if first is not None:
            sample = resolver.resolve_sample(OmniSample.from_dict(first))
            for turn in sample.turns:
                for audio_ref in (turn.audio_in, turn.speech_out):
                    if audio_ref is not None and audio_ref.local_path():
                        path = Path(audio_ref.local_path())
                        if not path.is_file():
                            missing_media.append(str(path))
                for image_ref in turn.images:
                    path = Path(image_ref.local_path())
                    if not path.is_file():
                        missing_media.append(str(path))
            sampled += 1
    if missing_media:
        raise FileNotFoundError(
            "Media paths from the first row of each manifest are missing: "
            f"{missing_media[:20]}"
        )
    return {
        "status": "ready",
        "mode": config.train.mode,
        "release_root": data.release_root,
        "manifests": len(data.manifest_paths),
        "sampled_manifests": sampled,
        "public_video_root": data.public_video_root,
        "s3_cache_dir": data.s3_cache_dir,
    }


def _write_s3_ready(cache_dir: Path, report: dict) -> bool:
    num_shards = int(report["num_shards"])
    if num_shards == 1:
        _atomic_json(
            cache_dir / S3_READY_FILENAME,
            {**report, "schema_version": "mcpmft_s3_cache_v1", "status": "ready"},
        )
        return True

    parts_dir = cache_dir / ".parts"
    part_path = parts_dir / f"{num_shards:05d}-{int(report['shard_index']):05d}.json"
    _atomic_json(part_path, report)
    paths = [parts_dir / f"{num_shards:05d}-{index:05d}.json" for index in range(num_shards)]
    if not all(path.is_file() for path in paths):
        return False
    parts = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    expected = report["manifests"]
    if any(
        part.get("num_shards") != num_shards
        or part.get("manifests") != expected
        or part.get("model") != report["model"]
        for part in parts
    ):
        raise ValueError("S3 shard reports do not describe the same training view")
    totals = Counter()
    for part in parts:
        totals.update(part.get("counts", {}))
    ready = {
        "schema_version": "mcpmft_s3_cache_v1",
        "status": "ready",
        "model": report["model"],
        "manifests": expected,
        "num_shards": num_shards,
        "counts": dict(totals),
    }
    _atomic_json(cache_dir / S3_READY_FILENAME, ready)
    return True


def _extract_s3(args, config: ProjectConfig) -> dict:
    if config.train.audio_loss_weight <= 0:
        raise ValueError("S3 target generation requires a Talker training view")
    if not config.data.s3_cache_dir:
        raise ValueError("Talker release config must set data.s3_cache_dir")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard index must satisfy 0 <= index < num-shards")
    if args.batch_size < 1 or args.audio_workers < 1 or args.limit < 0:
        raise ValueError("batch size and audio workers must be positive; limit cannot be negative")
    if args.torch_cpu_threads < 0:
        raise ValueError("torch CPU threads cannot be negative")
    if args.torch_cpu_threads:
        import torch

        torch.set_num_threads(args.torch_cpu_threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

    cache_dir = Path(config.data.s3_cache_dir)
    cache = S3TokenCache(cache_dir)
    extractor = S3TokenizerExtractor(
        args.model,
        download_root=args.download_root,
        device=args.device,
    )
    resolver = ReleaseMediaResolver(
        release_root=config.data.release_root,
        public_video_root=config.data.public_video_root,
    )
    counts: Counter[str] = Counter()
    pending: dict[str, object] = {}

    def flush() -> None:
        if not pending:
            return
        refs = list(pending.values())
        codes = extractor.extract_many(refs, audio_workers=args.audio_workers)
        for ref, values in zip(refs, codes):
            cache.put(ref, values)
            counts["extracted_targets"] += 1
            counts["extracted_codes"] += len(values)
        pending.clear()

    stop = False
    for manifest in config.data.manifest_paths:
        rows = _iter_jsonl_byte_shard(
            Path(manifest),
            num_shards=args.num_shards,
            shard_index=args.shard_index,
        )
        for row in rows:
            if args.limit and counts["rows"] >= args.limit:
                stop = True
                break
            counts["rows"] += 1
            sample = resolver.resolve_sample(OmniSample.from_dict(row))
            for turn in sample.turns:
                if turn.role != "assistant" or turn.speech_out is None:
                    continue
                counts["speech_targets"] += 1
                inline = turn.meta.get("s3_codes") if isinstance(turn.meta, dict) else None
                if isinstance(inline, list) and inline:
                    counts["inline_targets"] += 1
                    continue
                if cache.contains(turn.speech_out):
                    counts["cached_targets"] += 1
                    continue
                pending.setdefault(turn.speech_out.id(), turn.speech_out)
                if len(pending) >= args.batch_size:
                    flush()
            if args.progress_every and counts["rows"] % args.progress_every == 0:
                print(
                    f"rows={counts['rows']} extracted={counts['extracted_targets']} "
                    f"cached={counts['cached_targets']}",
                    flush=True,
                )
        if stop:
            break
    flush()

    report = {
        "model": args.model,
        "manifests": manifest_identities(
            config.data.manifest_paths,
            config.data.release_root,
        ),
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "counts": dict(counts),
    }
    report["ready"] = False if args.limit else _write_s3_ready(cache_dir, report)
    return report


def _add_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", action="append", required=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare and validate published training data")
    commands = parser.add_subparsers(dest="command", required=True)

    check = commands.add_parser("check", help="validate one release training view")
    _add_config_argument(check)

    s3 = commands.add_parser("s3", help="generate Talker S3 supervision")
    _add_config_argument(s3)
    s3.add_argument("--model", default="speech_tokenizer_v2_25hz")
    s3.add_argument("--download-root")
    s3.add_argument("--device")
    s3.add_argument("--batch-size", type=int, default=32)
    s3.add_argument("--audio-workers", type=int, default=4)
    s3.add_argument("--torch-cpu-threads", type=int, default=0)
    s3.add_argument("--num-shards", type=int, default=1)
    s3.add_argument("--shard-index", type=int, default=0)
    s3.add_argument("--limit", type=int, default=0)
    s3.add_argument("--progress-every", type=int, default=1000)

    args = parser.parse_args(argv)
    config = _project(args.config)
    result = _check(config) if args.command == "check" else _extract_s3(args, config)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
