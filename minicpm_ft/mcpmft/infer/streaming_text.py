from __future__ import annotations

import codecs
from types import MethodType
from typing import Any, Iterable


def _byte_decoder() -> dict[str, int]:
    byte_values = list(range(ord("!"), ord("~") + 1))
    byte_values += list(range(161, 173))
    byte_values += list(range(174, 256))
    codepoints = list(byte_values)
    extra = 0
    for value in range(256):
        if value not in byte_values:
            byte_values.append(value)
            codepoints.append(256 + extra)
            extra += 1
    return {chr(codepoint): value for value, codepoint in zip(byte_values, codepoints)}


_BYTE_DECODER = _byte_decoder()


class IncrementalByteLevelDecoder:
    """Decode ByteLevel token IDs without breaking UTF-8 at unit boundaries."""

    def __init__(self, tokenizer: Any) -> None:
        backend = getattr(tokenizer, "backend_tokenizer", None)
        decoder = getattr(backend, "decoder", None)
        if type(decoder).__name__ != "ByteLevel":
            raise TypeError("duplex streaming requires a ByteLevel tokenizer decoder")
        self.tokenizer = tokenizer
        added_special_ids = {
            int(token_id)
            for token_id, token in tokenizer.added_tokens_decoder.items()
            if token.special
        }
        self.special_token_ids = frozenset(
            {*map(int, tokenizer.all_special_ids), *added_special_ids}
        )
        self.reset()

    def reset(self) -> None:
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

    def decode(self, token_ids: Iterable[int], *, final: bool = False) -> str:
        lexical_ids = [
            int(value)
            for value in token_ids
            if int(value) not in self.special_token_ids
        ]
        tokens = self.tokenizer.convert_ids_to_tokens(lexical_ids)
        data = bytes(
            _BYTE_DECODER[character]
            for token in tokens
            for character in token
        )
        text = self._decoder.decode(data, final=final)
        if final:
            self.reset()
        return text


class DuplexStreamingText:
    """Project generated token units into stable text before context registration."""

    def __init__(self, duplex: Any, tokenizer: Any) -> None:
        self._stream = IncrementalByteLevelDecoder(tokenizer)
        self._turn_eos_token_id = int(duplex.turn_eos_token_id)
        self._last_unit_text: str | None = None

        decoder = duplex.decoder
        register_unit_end = decoder.register_unit_end

        def register(
            _decoder: Any,
            input_type: str,
            generated_tokens: list[int] | None = None,
            is_listen: bool = False,
            generated_text: str | None = None,
        ) -> Any:
            token_ids = [int(value) for value in generated_tokens or ()]
            if is_listen:
                self.reset()
            else:
                generated_text = self._stream.decode(
                    token_ids,
                    final=self._turn_eos_token_id in token_ids,
                )
                self._last_unit_text = generated_text
            return register_unit_end(
                input_type=input_type,
                generated_tokens=generated_tokens,
                is_listen=is_listen,
                generated_text=generated_text,
            )

        decoder.register_unit_end = MethodType(register, decoder)

    def begin_unit(self) -> None:
        self._last_unit_text = None

    def take_unit_text(self) -> str | None:
        text = self._last_unit_text
        self._last_unit_text = None
        return text

    def reset(self) -> None:
        self._stream.reset()
        self._last_unit_text = None
