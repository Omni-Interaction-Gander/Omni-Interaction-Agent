from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from mcpmft.args import ModelArguments
from mcpmft.modeling.load import (
    load_composed_minicpmo_model,
    load_minicpmo_model,
    load_tokenizer_and_processor,
)


@dataclass
class InferBundle:
    model: Any
    tokenizer: Any
    processor: Any


def load_for_infer(
    model_args: ModelArguments,
    *,
    checkpoint: str | None = None,
    talker_checkpoint: str | None = None,
    strict: bool = False,
    init_token2wav: bool = True,
    load_processor: bool = True,
) -> InferBundle:
    # Composed loading fills frozen tensors from the base model.
    for label, value in (
        ("checkpoint", checkpoint),
        ("talker checkpoint", talker_checkpoint),
    ):
        if value and not Path(value).exists():
            raise FileNotFoundError(f"{label.capitalize()} not found: {value}")
    # Inference retains the model's streaming embedding path.
    inference_args = replace(model_args, train_disable_stream_input=False)
    if load_processor:
        tokenizer, processor = load_tokenizer_and_processor(
            inference_args,
            tokenizer_path=checkpoint,
        )
    else:
        # Offline duplex replay can use the native streaming audio processor directly.
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint or inference_args.model_name_or_path,
            trust_remote_code=inference_args.trust_remote_code,
        )
        processor = None
    if checkpoint:
        model = load_composed_minicpmo_model(
            inference_args,
            checkpoint,
            tokenizer_size=len(tokenizer),
            init_token2wav=init_token2wav,
            strict=strict,
        )
    else:
        model = load_minicpmo_model(
            inference_args,
            init_token2wav=init_token2wav,
        )
    from mcpmft.modeling.load import add_native_frontbrain_tokens

    add_native_frontbrain_tokens(model, tokenizer)
    if processor is not None:
        processor.tokenizer = tokenizer
        model.processor = processor
    model_processor = getattr(model, "processor", None)
    if model_processor is not None:
        model_processor.tokenizer = tokenizer
    if talker_checkpoint:
        # Overlay tts.* weights while retaining Thinker weights from checkpoint.
        from mcpmft.modeling.load import load_prefixed_state_dict

        n = load_prefixed_state_dict(model, talker_checkpoint, prefix="tts.")
        print(f"[infer] overlaid {n} talker (tts.*) tensors from {talker_checkpoint}")
    model.eval()
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Gander inference requires CUDA")
    if inference_args.device_map is None:
        model.to("cuda")
    return InferBundle(model=model, tokenizer=tokenizer, processor=processor)


def load_ref_audio(path: str | Path, sample_rate: int = 16000):
    import librosa

    audio, _ = librosa.load(path, sr=sample_rate, mono=True)
    return audio


def save_wav(path: str | Path, waveform, sample_rate: int = 24000) -> None:
    import soundfile as sf

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, waveform, sample_rate)
