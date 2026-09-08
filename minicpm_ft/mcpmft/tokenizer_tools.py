from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SpecialToken:
    text: str
    token_id: int


AUDIO_START = SpecialToken("<|audio_start|>", 151697)
AUDIO_END = SpecialToken("<|audio_end|>", 151699)
IMAGE_START = SpecialToken("<image>", 151669)
IMAGE_END = SpecialToken("</image>", 151670)
SLICE_START = SpecialToken("<slice>", 151679)
SLICE_END = SpecialToken("</slice>", 151680)
# MiniCPM-o 4.5 emits 64 positions for each unsliced duplex frame.
IMAGE_FEATURE_SIZE = 64
UNIT_START = SpecialToken("<unit>", 151683)
UNIT_END = SpecialToken("</unit>", 151684)
CHUNK_EOS = SpecialToken("<|chunk_eos|>", 151718)
TURN_EOS = SpecialToken("<|turn_eos|>", 151717)
LISTEN = SpecialToken("<|listen|>", 151705)
SPEAK = SpecialToken("<|speak|>", 151706)
INTERRUPT = SpecialToken("<|interrupt|>", 151707)
TOOL_CALL_START = SpecialToken("<tool_call>", 151657)
TOOL_CALL_END = SpecialToken("</tool_call>", 151658)
TOOL_RESPONSE_START = SpecialToken("<tool_response>", 151665)
TOOL_RESPONSE_END = SpecialToken("</tool_response>", 151666)
SPK_BOS = SpecialToken("<|spk_bos|>", 151700)
SPK_EOS = SpecialToken("<|spk_eos|>", 151702)
TTS_BOS = SpecialToken("<|tts_bos|>", 151703)
TTS_EOS = SpecialToken("<|tts_eos|>", 151704)
# Talker embedding-table IDs; they are unrelated to the LLM tokenizer vocabulary.
TEXT_EOS_ID = 151692
AUDIO_BOS_ID = 151687
S3_EOS_ID = 6561
S3_NUM_AUDIO_TOKENS = 6562

# Backchannel extends MiniCPM-o's native listen/speak/interrupt decisions.
NATIVE_FRONTBRAIN_TOKENS = ["<|backchannel|>"]


def _resolved_or_none(tokenizer, text: str) -> int | None:
    """Resolve a token ID, treating unknown aliases as absent.

    MiniCPM-o returns unk_token_id for unknown spellings; the literal unknown token
    remains a valid vocabulary entry.
    """
    rid = tokenizer.convert_tokens_to_ids(text)
    if rid is None or rid < 0:
        return None
    unk = getattr(tokenizer, "unk_token_id", None)
    unk_text = getattr(tokenizer, "unk_token", None)
    if unk is not None and rid == unk and text != unk_text:
        return None
    return int(rid)

CONTROL_TOKENS = {
    "listen": LISTEN,
    "speak": SPEAK,
    "interrupt": INTERRUPT,
}

# Generation-control tokens excluded from visible text and speech.
PUBLIC_BAN_TOKENS = [
    LISTEN.text,
    SPEAK.text,
    INTERRUPT.text,
    UNIT_START.text,
    UNIT_END.text,
    CHUNK_EOS.text,
    "<|chunk_bos|>",
    "<|chunk_tts_bos|>",
    "<|chunk_tts_eos|>",
    TURN_EOS.text,
    "<|turn_bos|>",
    TTS_BOS.text,
    TTS_EOS.text,
    AUDIO_START.text,
    AUDIO_END.text,
    IMAGE_START.text,
    IMAGE_END.text,
    SLICE_START.text,
    SLICE_END.text,
    "<|audio|>",
    SPK_BOS.text,
    SPK_EOS.text,
    "<|spk|>",
    "<|tts_pad|>",
    "<|vad_start|>",
    "<|vad_end|>",
]


def assert_minicpmo_tokenizer(tokenizer) -> None:
    mismatches: list[str] = []
    required = [
        AUDIO_START,
        AUDIO_END,
        IMAGE_START,
        IMAGE_END,
        SLICE_START,
        SLICE_END,
        UNIT_START,
        UNIT_END,
        CHUNK_EOS,
        TURN_EOS,
        LISTEN,
        SPEAK,
        INTERRUPT,
        TOOL_CALL_START,
        TOOL_CALL_END,
        TOOL_RESPONSE_START,
        TOOL_RESPONSE_END,
        SPK_BOS,
        SPK_EOS,
        TTS_BOS,
        TTS_EOS,
    ]
    for token in required:
        resolved = _resolved_or_none(tokenizer, token.text)
        if resolved is None:
            mismatches.append(f"{token.text}: missing (expected id {token.token_id})")
        elif resolved != token.token_id:
            mismatches.append(f"{token.text}: expected {token.token_id}, got {resolved}")
    if mismatches:
        raise ValueError("MiniCPM-o special token id mismatch: " + "; ".join(mismatches))


def forbidden_token_ids(tokenizer) -> list[int]:
    """Return deduplicated IDs for available public-output control tokens."""
    ids: list[int] = []
    seen: set[int] = set()
    for text in PUBLIC_BAN_TOKENS:
        rid = _resolved_or_none(tokenizer, text)
        if rid is not None and rid not in seen:
            seen.add(rid)
            ids.append(rid)
    return ids
