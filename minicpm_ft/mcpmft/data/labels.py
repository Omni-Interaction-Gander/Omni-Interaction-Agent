from __future__ import annotations

IGNORE_INDEX = -100


def labels_from_assistant_spans(
    input_ids: list[int],
    assistant_spans: list[tuple[int, int]],
    *,
    ignore_index: int = IGNORE_INDEX,
) -> list[int]:
    """Create unshifted labels; causal shift is applied only in the loss."""
    labels = [ignore_index] * len(input_ids)
    for start, end in assistant_spans:
        if start < 0 or end > len(input_ids) or start > end:
            raise ValueError(f"Invalid assistant span {(start, end)} for sequence length {len(input_ids)}")
        labels[start:end] = input_ids[start:end]
    return labels


def mask_labels(labels: list[int], mask_spans: list[tuple[int, int]], *, ignore_index: int = IGNORE_INDEX) -> list[int]:
    labels = list(labels)
    for start, end in mask_spans:
        labels[start:end] = [ignore_index] * max(end - start, 0)
    return labels


def trim_to_length(values: list[int], max_length: int, *, left: bool = False) -> list[int]:
    if len(values) <= max_length:
        return values
    return values[-max_length:] if left else values[:max_length]

