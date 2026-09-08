from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

from mcpmft.args import ModelArguments
from mcpmft.utils.state_dict import strip_wrapper_prefix_if_present

LOGGER = logging.getLogger(__name__)


def torch_dtype_from_string(name: str):
    import torch

    mapping = {
        "auto": "auto",
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    try:
        return mapping[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported torch dtype: {name}") from exc


def load_tokenizer_and_processor(
    args: ModelArguments,
    *,
    tokenizer_path: str | Path | None = None,
):
    from transformers import AutoProcessor, AutoTokenizer

    processor_path = args.processor_name_or_path or args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path or args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
    )
    processor = AutoProcessor.from_pretrained(
        processor_path,
        trust_remote_code=args.trust_remote_code,
    )
    return tokenizer, processor


def load_composed_minicpmo_model(
    args: ModelArguments,
    checkpoint_path: str | Path,
    *,
    tokenizer_size: int,
    init_token2wav: bool = True,
    strict: bool = False,
):
    """Load trained weights from a checkpoint and frozen weights from the base model.

    Trainer checkpoints intentionally omit frozen parameters. Loading the complete base and then
    overlaying the checkpoint reads the large trained LLM twice. Instead, construct the model from
    the checkpoint and fetch only its missing parameters from the base safetensors.
    """

    from transformers import AutoConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    from transformers.utils import logging as transformers_logging

    checkpoint = Path(checkpoint_path)
    if not checkpoint.is_dir():
        raise ValueError("Inference checkpoint must be a Hugging Face checkpoint directory")

    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
        init_vision=args.init_vision,
        init_audio=args.init_audio,
        init_tts=args.init_tts,
        attn_implementation=args.attn_implementation,
    )
    config.vocab_size = tokenizer_size
    model_class = get_class_from_dynamic_module(
        config.auto_map["AutoModel"],
        args.model_name_or_path,
    )
    kwargs: dict[str, Any] = {
        "config": config,
        "torch_dtype": torch_dtype_from_string(args.torch_dtype),
        "low_cpu_mem_usage": args.low_cpu_mem_usage,
        "output_loading_info": True,
    }
    if args.device_map:
        kwargs["device_map"] = args.device_map

    previous_verbosity = transformers_logging.get_verbosity()
    transformers_logging.set_verbosity_error()
    try:
        model, loading_info = model_class.from_pretrained(checkpoint, **kwargs)
    finally:
        transformers_logging.set_verbosity(previous_verbosity)

    mismatched = list(loading_info.get("mismatched_keys") or ())
    errors = list(loading_info.get("error_msgs") or ())
    unexpected = list(loading_info.get("unexpected_keys") or ())
    if mismatched or errors or (strict and unexpected):
        raise RuntimeError(
            "Checkpoint is incompatible with the base architecture: "
            f"mismatched={mismatched[:3]} unexpected={unexpected[:3]} errors={errors[:3]}"
        )

    missing = list(loading_info.get("missing_keys") or ())
    _load_safetensor_keys(model, Path(args.model_name_or_path), missing)
    model.name_or_path = args.model_name_or_path
    model.config._name_or_path = args.model_name_or_path
    LOGGER.info(
        "Loaded inference checkpoint %s and %d frozen base tensors",
        checkpoint,
        len(missing),
    )
    if init_token2wav and args.init_tts and args.token2wav_dir:
        maybe_init_tts(model, args)
    return model


def _load_safetensor_keys(model, model_path: Path, keys: list[str]) -> None:
    if not keys:
        return

    from safetensors import safe_open

    index_path = model_path / "model.safetensors.index.json"
    if index_path.is_file():
        weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
    else:
        state_path = model_path / "model.safetensors"
        if not state_path.is_file():
            raise FileNotFoundError(f"Base model has no safetensors weights: {model_path}")
        with safe_open(str(state_path), framework="pt", device="cpu") as handle:
            weight_map = {key: state_path.name for key in handle.keys()}

    absent = sorted(set(keys) - set(weight_map))
    if absent:
        raise RuntimeError(
            f"Base model does not provide {len(absent)} frozen checkpoint parameters: "
            f"{absent[:5]}"
        )

    by_shard: dict[str, list[str]] = defaultdict(list)
    for key in keys:
        by_shard[weight_map[key]].append(key)
    for shard_name, shard_keys in by_shard.items():
        with safe_open(
            str(model_path / shard_name), framework="pt", device="cpu"
        ) as handle:
            state = {key: handle.get_tensor(key) for key in shard_keys}
        result = model.load_state_dict(state, strict=False)
        unexpected = sorted(set(result.unexpected_keys) & set(state))
        if unexpected:
            raise RuntimeError(f"Frozen base tensors did not match the model: {unexpected[:5]}")


def load_minicpmo_model(args: ModelArguments, *, init_token2wav: bool = True):
    from transformers import AutoModel

    kwargs: dict[str, Any] = {
        "trust_remote_code": args.trust_remote_code,
        "torch_dtype": torch_dtype_from_string(args.torch_dtype),
        "low_cpu_mem_usage": args.low_cpu_mem_usage,
        "init_vision": args.init_vision,
        "init_audio": args.init_audio,
        "init_tts": args.init_tts,
    }
    if args.attn_implementation:
        kwargs["attn_implementation"] = args.attn_implementation
    if args.device_map:
        kwargs["device_map"] = args.device_map
    LOGGER.info("Loading MiniCPM-o model from %s", args.model_name_or_path)
    model = AutoModel.from_pretrained(args.model_name_or_path, **kwargs)
    # Batched training uses the non-streaming audio scatter path.
    if args.train_disable_stream_input and model.config.stream_input:
        model.config.stream_input = False
        LOGGER.info("Set config.stream_input=False for batched training")
    if init_token2wav and args.init_tts and args.token2wav_dir:
        maybe_init_tts(model, args)
    return model


def maybe_init_tts(model, args: ModelArguments) -> None:
    token2wav_dir = Path(args.token2wav_dir)
    if not token2wav_dir.is_dir():
        raise FileNotFoundError(f"token2wav_dir not found: {token2wav_dir}")
    model.init_tts(
        model_dir=str(token2wav_dir),
        n_timesteps=args.token2wav_n_timesteps,
        enable_float16=args.token2wav_enable_float16,
    )


def load_partial_state_dict(model, checkpoint_path: str | Path, *, strict: bool = False) -> tuple[list[str], list[str]]:
    state_dict = _load_checkpoint_state_dict(Path(checkpoint_path))
    result = model.load_state_dict(state_dict, strict=strict)
    return list(result.missing_keys), list(result.unexpected_keys)


def load_selected_state_dict(
    model,
    checkpoint_path: str | Path,
    *,
    prefixes: list[str] | tuple[str, ...],
) -> int:
    """Warm-start only explicitly owned model components from a checkpoint.

    This is intentionally separate from Trainer resume: it overlays model tensors before
    freezing while leaving base-model components outside ``prefixes`` untouched. For example, a
    thinker checkpoint can overlay ``llm.`` and ``audio_projection_layer.`` without replacing the
    base Talker under ``tts.``.
    """
    normalized = tuple(str(prefix) for prefix in prefixes if str(prefix))
    if not normalized:
        raise ValueError("Checkpoint warm-start requires at least one non-empty prefix")
    model_keys = set(model.state_dict())
    expected_by_prefix = {
        prefix: {key for key in model_keys if key.startswith(prefix)}
        for prefix in normalized
    }
    unknown_prefixes = [
        prefix for prefix, keys in expected_by_prefix.items() if not keys
    ]
    if unknown_prefixes:
        raise ValueError(
            "Checkpoint warm-start prefix(es) do not name model components: "
            f"{unknown_prefixes}"
        )

    loaded = 0
    loaded_keys: set[str] = set()
    for state_path in _resolve_state_paths(Path(checkpoint_path)):
        shard = strip_wrapper_prefix_if_present(_load_state_file(state_path))
        subset = {
            key: value
            for key, value in shard.items()
            if any(key.startswith(prefix) for prefix in normalized)
        }
        if not subset:
            continue
        result = model.load_state_dict(subset, strict=False)
        unexpected = [key for key in result.unexpected_keys if key in subset]
        if unexpected:
            raise RuntimeError(
                f"{len(unexpected)} selected checkpoint tensors did not match the model, "
                f"e.g. {unexpected[:3]}"
            )
        loaded += len(subset)
        loaded_keys.update(subset)
        del subset
        del shard
    expected_keys = set().union(*expected_by_prefix.values())
    missing = sorted(expected_keys - loaded_keys)
    if missing:
        raise RuntimeError(
            "Checkpoint warm-start is incomplete for the explicitly owned components: "
            f"checkpoint={checkpoint_path}, prefixes={list(normalized)}, "
            f"missing={missing[:5]}, missing_count={len(missing)}"
        )
    return loaded


def load_prefixed_state_dict(model, checkpoint_path: str | Path, *, prefix: str) -> int:
    """Load only the tensors whose key starts with `prefix` from a checkpoint into `model`.

    Used to overlay one component (e.g. the talker, prefix="tts.") from a separately-trained
    checkpoint onto a model that already holds another component (e.g. the thinker). Returns the
    number of tensors copied. Non-matching keys and model params without a checkpoint match are
    left untouched.
    """
    state_dict = _load_checkpoint_state_dict(Path(checkpoint_path))
    subset = {k: v for k, v in state_dict.items() if k.startswith(prefix)}
    if not subset:
        raise RuntimeError(
            f"Checkpoint {checkpoint_path} has no tensors with required prefix {prefix!r}"
        )
    missing, unexpected = model.load_state_dict(subset, strict=False)
    # With strict=False, only unexpected subset keys indicate a prefix mismatch.
    stray = [k for k in unexpected if k.startswith(prefix)]
    if stray:
        raise RuntimeError(f"{len(stray)} '{prefix}' tensors did not match model params, e.g. {stray[:3]}")
    return len(subset)


def load_prefixed_submodule_state_dict(
    module,
    checkpoint_path: str | Path,
    *,
    prefix: str,
    strict: bool = True,
) -> int:
    """Load a prefixed checkpoint component into its standalone submodule.

    Unlike :func:`load_prefixed_state_dict`, checkpoint keys have ``prefix`` removed before
    loading. Shards are consumed one at a time so a standalone Talker can be materialized without
    first constructing, or holding in CPU memory, the full MiniCPM-o model state.
    """
    expected = set(module.state_dict())
    loaded: set[str] = set()
    count = 0
    for state_path in _resolve_state_paths(Path(checkpoint_path)):
        shard = strip_wrapper_prefix_if_present(_load_state_file(state_path))
        subset = {
            key[len(prefix) :]: value
            for key, value in shard.items()
            if key.startswith(prefix)
        }
        del shard
        if not subset:
            continue
        result = module.load_state_dict(subset, strict=False)
        unexpected = sorted(set(result.unexpected_keys) & set(subset))
        if unexpected:
            raise RuntimeError(
                f"{len(unexpected)} stripped '{prefix}' tensors did not match the standalone "
                f"module, e.g. {unexpected[:3]}"
            )
        loaded.update(subset)
        count += len(subset)
        del subset

    if not count:
        raise RuntimeError(
            f"Checkpoint {checkpoint_path} has no tensors with required prefix {prefix!r}"
        )
    if strict:
        missing = sorted(expected - loaded)
        extra = sorted(loaded - expected)
        if missing or extra:
            raise RuntimeError(
                "Standalone component checkpoint is incomplete or incompatible: "
                f"checkpoint={checkpoint_path}, prefix={prefix!r}, "
                f"missing={missing[:5]} ({len(missing)}), extra={extra[:5]} ({len(extra)})"
            )
    return count


def _load_checkpoint_state_dict(checkpoint_path: Path) -> dict[str, Any]:
    state_dict: dict[str, Any] = {}
    for state_path in _resolve_state_paths(checkpoint_path):
        state_dict.update(_load_state_file(state_path))
    return strip_wrapper_prefix_if_present(state_dict)


def _load_state_file(state_path: Path) -> dict[str, Any]:
    if state_path.suffix == ".safetensors":
        from safetensors.torch import load_file

        shard = load_file(str(state_path))
    else:
        import torch

        shard = torch.load(state_path, map_location="cpu")
        if isinstance(shard, dict) and "state_dict" in shard:
            shard = shard["state_dict"]
    if not isinstance(shard, dict):
        raise TypeError(f"Checkpoint weight file is not a state dict: {state_path}")
    return shard


def _resolve_state_paths(checkpoint_path: Path) -> list[Path]:
    """Return model-weight files from either a single file or an HF checkpoint directory."""
    if checkpoint_path.is_file():
        return [checkpoint_path]
    if not checkpoint_path.is_dir():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    for name in ["model.safetensors", "pytorch_model.bin"]:
        candidate = checkpoint_path / name
        if candidate.exists():
            return [candidate]

    for index_name in ["model.safetensors.index.json", "pytorch_model.bin.index.json"]:
        index_path = checkpoint_path / index_name
        if not index_path.exists():
            continue
        with index_path.open("r", encoding="utf-8") as handle:
            index = json.load(handle)
        shard_names = sorted(set(index.get("weight_map", {}).values()))
        paths = [checkpoint_path / name for name in shard_names]
        missing = [str(path) for path in paths if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Checkpoint index references missing shard(s): {missing}")
        return paths

    raise FileNotFoundError(
        f"No model weights found in {checkpoint_path}; expected model.safetensors, "
        "pytorch_model.bin, or a HuggingFace sharded index."
    )


def describe_model_args(args: ModelArguments) -> dict[str, Any]:
    return asdict(args)


def add_native_frontbrain_tokens(model, tokenizer) -> dict[str, int]:
    """Add interaction controls absent from the MiniCPM-o vocabulary."""

    from mcpmft.tokenizer_tools import NATIVE_FRONTBRAIN_TOKENS

    return _add_special_tokens(model, tokenizer, NATIVE_FRONTBRAIN_TOKENS)


def _add_special_tokens(model, tokenizer, token_texts: list[str]) -> dict[str, int]:
    missing = [
        text
        for text in token_texts
        if tokenizer.convert_tokens_to_ids(text) == tokenizer.unk_token_id
    ]
    if missing:
        tokenizer.add_special_tokens({"additional_special_tokens": missing})
        target = model.llm
        old = target.get_input_embeddings().weight.shape[0]
        # New rows receive the existing embedding mean below.
        target.resize_token_embeddings(len(tokenizer), mean_resizing=False)
        _mean_init_new_rows(target, old, len(tokenizer))
    return {text: int(tokenizer.convert_tokens_to_ids(text)) for text in token_texts}


def _mean_init_new_rows(target, old_size: int, new_size: int) -> None:
    import torch

    if new_size <= old_size:
        return
    inp = target.get_input_embeddings().weight
    with torch.no_grad():
        mean_in = inp[:old_size].mean(dim=0)
        inp[old_size:new_size] = mean_in
        out = target.get_output_embeddings()
        if out is not None and out.weight.shape[0] >= new_size and out.weight is not inp:
            mean_out = out.weight[:old_size].mean(dim=0)
            out.weight[old_size:new_size] = mean_out
