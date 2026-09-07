from __future__ import annotations

from typing import Any

from mcpmft.data.streaming_audio import STREAMING_MAX_GROUP_UNITS


def encode_streaming_audio_windows(core: Any, batch: dict[str, Any]):
    """Encode runtime-equivalent per-unit mel windows with one batched APM pass.

    CNN runs over all unit windows in parallel. Their 50-frame cores are then concatenated into
    the same 29-unit reset groups used by realtime APM KV caching, and the Whisper Transformer is
    evaluated once with a one-second chunk-causal mask.
    """

    import torch
    import torch.nn.functional as F

    apm = core.apm
    features = batch["audio_features"].to(
        device=apm.conv1.weight.device,
        dtype=apm.conv1.weight.dtype,
    )
    first_mask = batch["audio_streaming_first_unit_mask"].to(
        device=features.device,
        dtype=torch.bool,
    )
    if features.ndim != 3 or features.size(0) != first_mask.numel():
        raise ValueError(
            "streaming audio feature/mask mismatch: "
            f"features={tuple(features.shape)}, first_mask={tuple(first_mask.shape)}"
        )

    cnn = F.gelu(apm.conv1(features))
    cnn = F.gelu(apm.conv2(cnn))
    if cnn.size(-1) < 52:
        raise ValueError(
            "streaming audio CNN windows must produce at least 52 frames, "
            f"got {cnn.size(-1)}"
        )
    # Runtime first unit has no prefix redundancy and removes one suffix CNN frame. Subsequent
    # units remove one CNN frame from each side. Padding the first mel window to the common 104
    # frames does not affect its retained first 50 outputs.
    first_core = cnn[:, :, :50]
    regular_core = cnn[:, :, 1:51]
    unit_cnn = torch.where(
        first_mask.view(-1, 1, 1),
        first_core,
        regular_core,
    ).transpose(1, 2)

    group_counts = batch["audio_streaming_group_unit_counts"]
    if len(group_counts) != len(batch["audio_group_bounds"]):
        raise ValueError(
            "streaming audio group metadata batch mismatch: "
            f"counts={len(group_counts)}, bounds={len(batch['audio_group_bounds'])}"
        )
    flat_counts = [int(count) for sample in group_counts for count in sample]
    if sum(flat_counts) != unit_cnn.size(0):
        raise ValueError(
            "streaming audio group layout mismatch: "
            f"grouped_units={sum(flat_counts)}, cnn_units={unit_cnn.size(0)}"
        )
    if any(count > STREAMING_MAX_GROUP_UNITS for count in flat_counts):
        raise ValueError(
            "streaming audio APM groups must contain at most "
            f"{STREAMING_MAX_GROUP_UNITS} one-second units"
        )
    unit_cursor = 0
    for sample_index, sample_counts in enumerate(group_counts):
        sample_units = sum(int(count) for count in sample_counts)
        sample_mask = first_mask[unit_cursor : unit_cursor + sample_units]
        expected = torch.zeros_like(sample_mask)
        if sample_units:
            expected[0] = True
        if not torch.equal(sample_mask, expected):
            raise ValueError(
                "streaming first-unit mask does not match sample boundaries: "
                f"sample={sample_index}, units={sample_units}"
            )
        unit_cursor += sample_units
    group_sequences = []
    cursor = 0
    for count in flat_counts:
        if count <= 0:
            raise ValueError(f"streaming audio group has invalid unit count: {count}")
        group_sequences.append(
            unit_cnn[cursor : cursor + count].reshape(
                count * 50,
                unit_cnn.size(-1),
            )
        )
        cursor += count

    lengths = torch.tensor(
        [sequence.size(0) for sequence in group_sequences],
        device=features.device,
        dtype=torch.long,
    )
    hidden = torch.nn.utils.rnn.pad_sequence(
        group_sequences,
        batch_first=True,
    )
    maximum = hidden.size(1)
    if maximum > apm.embed_positions.weight.size(0):
        raise ValueError(
            "streaming audio APM group exceeds positional capacity: "
            f"frames={maximum}, capacity={apm.embed_positions.weight.size(0)}"
        )
    hidden = hidden + apm.embed_positions.weight[:maximum].unsqueeze(0)
    hidden = F.dropout(hidden, p=apm.dropout, training=apm.training)

    chunk_mask = core.subsequent_chunk_mask(
        size=maximum,
        chunk_size=50,
        num_left_chunks=-1,
        device=features.device,
    )
    key_padding = (
        torch.arange(maximum, device=features.device).unsqueeze(0)
        >= lengths.unsqueeze(1)
    )
    blocked = torch.logical_or(
        key_padding[:, None, None, :],
        torch.logical_not(chunk_mask)[None, None, :, :],
    )
    attention_mask = torch.zeros(
        blocked.shape,
        device=features.device,
        dtype=apm.conv1.weight.dtype,
    )
    attention_mask.masked_fill_(blocked, float("-inf"))

    selected_layer = int(getattr(core, "audio_encoder_layer", -1))
    hidden_state_count = len(apm.layers) + 1
    selected_hidden_index = (
        selected_layer
        if selected_layer >= 0
        else hidden_state_count + selected_layer
    )
    if not 0 <= selected_hidden_index < hidden_state_count:
        raise ValueError(
            f"Unsupported audio_encoder_layer={selected_layer} for {len(apm.layers)} layers"
        )
    selected_hidden = None
    for layer_index, layer in enumerate(apm.layers):
        if selected_hidden_index == layer_index:
            selected_hidden = hidden
        to_drop = False
        if apm.training and float(getattr(apm, "layerdrop", 0.0)) > 0.0:
            to_drop = bool(
                torch.rand((), device=hidden.device)
                < float(apm.layerdrop)
            )
        if to_drop:
            continue
        if apm.gradient_checkpointing and apm.training:
            output = apm._gradient_checkpointing_func(
                layer.__call__,
                hidden,
                attention_mask,
                None,
                False,
                None,
                False,
            )
        else:
            output = layer(
                hidden,
                attention_mask,
                layer_head_mask=None,
                output_attentions=False,
                past_key_values=None,
                use_cache=False,
            )
        hidden = output[0]
    hidden = apm.layer_norm(hidden)
    if selected_hidden_index == len(apm.layers):
        selected_hidden = hidden
    if selected_hidden is None:
        raise AssertionError("selected audio encoder hidden state was not captured")

    projected = core.audio_projection_layer(selected_hidden)
    projected = core.audio_avg_pooler(projected.transpose(1, 2)).transpose(1, 2)

    nested = []
    group_index = 0
    pool_step = int(core.config.audio_pool_step)
    if pool_step <= 0 or 50 % pool_step:
        raise ValueError(
            "streaming audio batching requires audio_pool_step to divide the "
            f"50 CNN frames in each unit, got {pool_step}"
        )
    tokens_per_unit = (50 - pool_step) // pool_step + 1
    for sample_counts in group_counts:
        sample_groups = []
        for count in sample_counts:
            output_tokens = int(count) * tokens_per_unit
            sample_groups.append(projected[group_index, :output_tokens])
            group_index += 1
        nested.append(sample_groups)
    return nested
