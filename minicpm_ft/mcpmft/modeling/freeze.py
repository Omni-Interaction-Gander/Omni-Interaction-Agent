from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from mcpmft.args import FreezeArguments


MODULE_PATHS = {
    "vision": ["vpm"],
    "resampler": ["resampler"],
    "audio_encoder": ["apm"],
    "audio_proj": ["audio_projection_layer", "audio_avg_pooler"],
    "llm": ["llm"],
    # projector_spk is used only to encode a reference speaker at inference; the current
    # teacher-forced Talker loss has no speaker-embedding target/path, so leaving it trainable
    # creates an unused DDP parameter and saves an unchanged tensor in every checkpoint.
    "tts_proj": ["tts.projector_semantic", "tts.emb_text"],
    "tts_decoder": ["tts.model", "tts.emb_code", "tts.head_code"],
}


@dataclass(frozen=True)
class ParamStats:
    total: int
    trainable: int

    @property
    def trainable_ratio(self) -> float:
        return 0.0 if self.total == 0 else self.trainable / self.total


def resolve_attr(root, dotted: str):
    obj = root
    for part in dotted.split("."):
        if not hasattr(obj, part):
            return None
        obj = getattr(obj, part)
    return obj


def set_module_trainable(module, trainable: bool) -> int:
    if module is None:
        return 0
    count = 0
    for param in module.parameters(recurse=True):
        param.requires_grad = trainable
        count += param.numel()
    return count


def apply_freeze(model, args: FreezeArguments) -> dict[str, int]:
    core = getattr(model, "model", model)
    for param in model.parameters():
        param.requires_grad = False

    plan = {
        "vision": args.tune_vision,
        "resampler": args.tune_resampler,
        "audio_encoder": args.tune_audio_encoder,
        "audio_proj": args.tune_audio_proj,
        "llm": args.tune_llm,
        "tts_proj": args.tune_tts_proj,
        "tts_decoder": args.tune_tts_decoder,
    }
    touched: dict[str, int] = {}
    for group, trainable in plan.items():
        group_count = 0
        for dotted in MODULE_PATHS[group]:
            group_count += set_module_trainable(resolve_attr(core, dotted), trainable)
        touched[group] = group_count

    # MiniCPMTTS always supplies ``inputs_embeds`` to its internal LlamaModel. Its vocabulary
    # embedding is therefore never called by either teacher forcing or vendor generation. Keep it
    # frozen even when the decoder is tuned so DDP has no needlessly trainable unused parameter.
    unused_decoder_embedding = set_module_trainable(
        resolve_attr(core, "tts.model.embed_tokens"),
        False,
    )
    if unused_decoder_embedding:
        if args.tune_tts_decoder:
            touched["tts_decoder"] -= unused_decoder_embedding
        touched["tts_decoder_input_embedding_frozen"] = unused_decoder_embedding
    return touched


def param_stats(model) -> ParamStats:
    total = 0
    trainable = 0
    for param in model.parameters():
        numel = param.numel()
        total += numel
        if param.requires_grad:
            trainable += numel
    return ParamStats(total=total, trainable=trainable)


def iter_trainable_named_parameters(model) -> Iterable[tuple[str, object]]:
    for name, param in model.named_parameters():
        if param.requires_grad:
            yield name, param


def group_named_parameters(
    model,
    *,
    base_lr: float,
    projector_lr: float | None,
    tts_lr: float | None,
    weight_decay: float = 0.0,
    decay_parameter_names: Iterable[str] | None = None,
):
    projector_lr = base_lr if projector_lr is None else projector_lr
    tts_lr = base_lr if tts_lr is None else tts_lr
    decay_names = set(decay_parameter_names) if decay_parameter_names is not None else None
    learning_rates = {"base": base_lr, "projector": projector_lr, "tts": tts_lr}
    groups: dict[tuple[str, bool], dict] = {}
    for name, param in iter_trainable_named_parameters(model):
        if any(
            key in name
            for key in [
                "audio_projection_layer",
                "projector_semantic",
                "projector_spk",
                "tts.emb_text",
                "resampler",
            ]
        ):
            bucket = "projector"
        elif ".tts." in name or name.startswith("tts."):
            bucket = "tts"
        else:
            bucket = "base"
        should_decay = name in decay_names if decay_names is not None else _default_should_decay(name, param)
        group = groups.setdefault(
            (bucket, should_decay),
            {
                "params": [],
                "lr": learning_rates[bucket],
                "weight_decay": weight_decay if should_decay else 0.0,
                "name": f"{bucket}_{'decay' if should_decay else 'no_decay'}",
            },
        )
        group["params"].append(param)
    return [
        group
        for group in groups.values()
        if group["params"]
    ]


def _default_should_decay(name: str, param) -> bool:
    lowered = name.lower()
    return param.ndim >= 2 and not name.endswith(".bias") and not any(
        marker in lowered for marker in ("layernorm", "layer_norm", "rmsnorm", "rms_norm")
    )
