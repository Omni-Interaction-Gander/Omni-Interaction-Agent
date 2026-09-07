from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from mcpmft.infer.common import InferBundle, load_ref_audio, save_wav


class OfflineRunner:
    """Run one turn-style multimodal MiniCPM conversation."""

    def __init__(self, bundle: InferBundle) -> None:
        self.bundle = bundle

    def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        generate_audio: bool = False,
        output_audio_path: str | None = None,
        ref_audio_path: str | None = None,
        enable_thinking: bool = False,
        omni_mode: bool = False,
        sys_mode: str = "audio_assistant",
        language: str = "zh",
        **generation_kwargs: Any,
    ) -> dict[str, Any]:
        if not messages:
            raise ValueError("Offline inference requires at least one message")
        if generate_audio:
            if not ref_audio_path:
                raise ValueError("Audio generation requires inference.input.ref_audio")
            system_message = self.bundle.model.get_sys_prompt(
                ref_audio=ref_audio_path,
                mode=sys_mode,
                language=language,
            )
            if messages[0].get("role") != "system":
                messages = [system_message, *messages]

        result = self.bundle.model.chat(
            msgs=messages,
            tokenizer=self.bundle.tokenizer,
            processor=self.bundle.processor,
            stream=generate_audio,
            generate_audio=generate_audio,
            use_tts_template=generate_audio,
            enable_thinking=enable_thinking,
            omni_mode=omni_mode,
            **generation_kwargs,
        )
        normalized = _consume_result(result)
        if generate_audio and normalized.get("audio") is None:
            raise RuntimeError("MiniCPM-o completed without producing an audio waveform")
        if output_audio_path:
            save_wav(output_audio_path, normalized["audio"])
        return normalized


def build_user_messages(
    *,
    text: str | None = None,
    audio_path: str | None = None,
    image_path: str | None = None,
) -> list[dict[str, Any]]:
    content: list[Any] = []
    if image_path:
        from PIL import Image

        with Image.open(Path(image_path)) as image:
            content.append(image.convert("RGB"))
    if audio_path:
        content.append(load_ref_audio(audio_path))
    if text:
        content.append(text)
    if not content:
        raise ValueError("Set at least one of inference.input.text/audio/image")
    return [
        {
            "role": "user",
            "content": content[0] if len(content) == 1 else content,
        }
    ]


def _consume_result(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        return dict(result)
    if isinstance(result, str):
        return {"text": result, "audio": None}
    if not hasattr(result, "__iter__") or isinstance(result, (bytes, bytearray)):
        raise TypeError(f"Unsupported MiniCPM-o inference result: {type(result).__name__}")

    texts: list[str] = []
    audio_chunks: list[Any] = []
    for item in result:
        if isinstance(item, str):
            texts.append(item)
            continue
        if not isinstance(item, tuple) or len(item) < 2:
            raise TypeError(
                f"Unsupported MiniCPM-o stream item: {type(item).__name__}"
            )
        waveform, text = item[0], item[1]
        if waveform is not None:
            audio_chunks.append(waveform)
        if text:
            texts.append(str(text))
    return {
        "text": "".join(texts),
        "audio": _concat_audio(audio_chunks) if audio_chunks else None,
    }


def _concat_audio(chunks: list[Any]) -> np.ndarray:
    arrays = [
        chunk.detach().cpu().float().reshape(-1).numpy()
        if torch.is_tensor(chunk)
        else np.asarray(chunk, dtype=np.float32).reshape(-1)
        for chunk in chunks
    ]
    return np.concatenate(arrays)
