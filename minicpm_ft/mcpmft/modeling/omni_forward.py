from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch.nn as nn

from mcpmft.args import FreezeArguments, TalkerArguments
from mcpmft.data.labels import IGNORE_INDEX
from mcpmft.modeling.freeze import MODULE_PATHS, resolve_attr
from mcpmft.utils.state_dict import add_wrapper_prefix_if_absent


@dataclass
class OmniForwardOutput:
    loss: Any
    text_loss: Any | None
    audio_loss: Any | None
    logits: Any | None = None

class OmniTrainWrapper(nn.Module):
    def __init__(
        self,
        model,
        *,
        tokenizer,
        talker_args: TalkerArguments | None = None,
        text_loss_weight: float = 1.0,
        audio_loss_weight: float = 1.0,
        control_loss_weight: float = 1.0,
        tune_llm: bool = True,
        audio_chunk_length: int = 1,
        freeze_args: FreezeArguments | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.talker_args = talker_args or TalkerArguments()
        self.text_loss_weight = text_loss_weight
        self.audio_loss_weight = audio_loss_weight
        self.control_loss_weight = control_loss_weight
        self.tune_llm = tune_llm
        self.audio_chunk_length = audio_chunk_length
        self._frozen_module_paths = _frozen_module_paths(freeze_args)
        self._ctrl_ids_cache = None

    @property
    def config(self):
        # DeepSpeed / HF Trainer read .config on the top-level module for hidden_size, etc.
        return self.model.config

    @property
    def generation_config(self):
        return self.model.generation_config

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        state_dict = add_wrapper_prefix_if_absent(state_dict)
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def gradient_checkpointing_enable(self, **kwargs):
        self.model.gradient_checkpointing_enable(**kwargs)

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            for dotted in self._frozen_module_paths:
                module = resolve_attr(self.model, dotted)
                if module is not None:
                    module.eval()
        return self

    def forward(self, **batch):
        import torch

        need_text = self.text_loss_weight != 0 and "labels" in batch
        need_audio = self.audio_loss_weight != 0 and "speech_segments" in batch
        output_hidden_states = need_audio

        inputs_embeds = self._build_inputs_embeds(batch)
        llm = self._llm()
        attention_mask = batch.get("attention_mask")
        # A 4D additive KV-delete mask (built float32 by the collator) must match the compute
        # dtype, or SDPA raises "invalid dtype for bias". 2D padding masks stay long.
        if attention_mask is not None and attention_mask.dim() == 4 and attention_mask.is_floating_point():
            attention_mask = attention_mask.to(inputs_embeds.dtype)
        llm_kwargs = {
            "input_ids": None,
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "position_ids": batch.get("position_ids"),
            "output_hidden_states": output_hidden_states,
            # Training is full-sequence teacher forcing; retaining incremental KV tensors only
            # increases memory and is not consumed by either loss path.
            "use_cache": False,
        }
        if not need_text:
            # The audio loss needs all hidden states but no vocabulary projection. Keep one logits
            # position for the output contract instead of materializing BxLxV.
            llm_kwargs["logits_to_keep"] = 1
        out = llm(**llm_kwargs)
        logits = out.logits

        text_loss = None
        if need_text:
            labels = batch["labels"]
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            if self.control_loss_weight != 1.0:
                # Per-token weighted CE: upweight the full-duplex control tokens (listen/speak/
                # interrupt/chunk_eos/turn_eos/backchannel) so the model learns turn-taking timing
                # harder (fixes premature-speak). reduction='none' → weight → normalize by sum.
                flat_logits = shift_logits.view(-1, shift_logits.size(-1))
                flat_labels = shift_labels.view(-1)
                per_tok = _cross_entropy_fp32(
                    flat_logits,
                    flat_labels,
                    ignore_index=IGNORE_INDEX,
                    reduction="none",
                )
                weights = torch.ones_like(per_tok)
                ctrl_ids = self._control_token_ids(labels.device)
                is_ctrl = torch.isin(flat_labels, ctrl_ids)
                weights = torch.where(is_ctrl, weights * self.control_loss_weight, weights)
                # zero the ignored positions so they don't count toward the normalizer
                valid = flat_labels != IGNORE_INDEX
                weights = weights * valid
                text_loss = self._normalize_loss_sum((per_tok * weights).sum(), weights.sum())
            else:
                flat_logits = shift_logits.view(-1, shift_logits.size(-1))
                flat_labels = shift_labels.view(-1)
                text_sum = _cross_entropy_fp32(
                    flat_logits,
                    flat_labels,
                    ignore_index=IGNORE_INDEX,
                    reduction="sum",
                )
                text_loss = self._normalize_loss_sum(text_sum, (flat_labels != IGNORE_INDEX).sum())

        audio_loss = None
        if need_audio:
            hidden_states = out.hidden_states[self.talker_args.tts_proj_layer]
            if self._detach_llm_for_tts():
                hidden_states = hidden_states.detach()
            audio_loss = self.build_talker_loss(hidden_states, batch["speech_segments"])

        loss = None
        if text_loss is not None:
            loss = text_loss * self.text_loss_weight
        if audio_loss is not None:
            weighted_audio = audio_loss * self.audio_loss_weight
            loss = weighted_audio if loss is None else loss + weighted_audio
        if loss is None:
            loss = logits.sum() * 0.0

        return OmniForwardOutput(loss=loss, text_loss=text_loss, audio_loss=audio_loss, logits=logits)

    def _build_inputs_embeds(self, batch: dict[str, Any]):
        if "inputs_embeds" in batch:
            return batch["inputs_embeds"]
        has_vision = "pixel_values" in batch and "tgt_sizes" in batch and "image_bound" in batch
        has_audio = "audio_features" in batch and "audio_bounds" in batch

        # get_vllm_embedding hard-requires pixel_values/tgt_sizes/image_bound even for text-only
        # batches; only use it when the vision keys are present. Otherwise embed text directly.
        if has_vision:
            inputs_embeds, _ = self.model.get_vllm_embedding(batch)
        else:
            inputs_embeds = self._llm().get_input_embeddings()(batch["input_ids"])

        if has_audio:
            if "audio_group_bounds" in batch:
                inputs_embeds = self._scatter_grouped_audio_embeddings(batch, inputs_embeds)
            else:
                inputs_embeds = self.model.get_omni_embedding(
                    batch,
                    inputs_embeds,
                    chunk_length=self.audio_chunk_length,
                )
        return inputs_embeds

    def _scatter_grouped_audio_embeddings(self, batch: dict[str, Any], inputs_embeds):
        """Encode audio groups and scatter their embeddings into unit placeholders."""
        import torch

        if "audio_streaming_group_unit_counts" in batch:
            from mcpmft.modeling.streaming_audio import encode_streaming_audio_windows

            audio_embeddings = encode_streaming_audio_windows(self.model, batch)
        else:
            audio_embeddings = self.model.get_audio_embedding(
                batch,
                chunk_length=self.audio_chunk_length,
            )
        if len(audio_embeddings) != inputs_embeds.size(0):
            raise ValueError(
                "Grouped audio embedding batch mismatch: "
                f"got {len(audio_embeddings)} sample groups for batch size {inputs_embeds.size(0)}"
            )

        group_bounds_batch = batch["audio_group_bounds"]
        if len(group_bounds_batch) != inputs_embeds.size(0):
            raise ValueError(
                "audio_group_bounds batch mismatch: "
                f"got {len(group_bounds_batch)} for batch size {inputs_embeds.size(0)}"
            )

        for batch_idx, (groups, group_bounds) in enumerate(zip(audio_embeddings, group_bounds_batch)):
            if len(groups) != len(group_bounds):
                raise ValueError(
                    "Audio group count mismatch for sample "
                    f"{batch_idx}: embeddings={len(groups)} bounds={len(group_bounds)}"
                )
            for group_idx, (group_embeds, bounds) in enumerate(zip(groups, group_bounds)):
                cursor = 0
                expected = sum(int(end) - int(start) for start, end in bounds)
                if group_embeds.size(0) != expected:
                    raise ValueError(
                        "Grouped audio embedding length mismatch for sample "
                        f"{batch_idx} group {group_idx}: encoder produced {group_embeds.size(0)} "
                        f"tokens, but target bounds require {expected}"
                    )
                for start, end in bounds:
                    start = int(start)
                    end = int(end)
                    length = end - start
                    if length <= 0:
                        continue
                    audio_indices = torch.arange(start, end, dtype=torch.long, device=inputs_embeds.device)
                    inputs_embeds[batch_idx, audio_indices] = group_embeds[
                        cursor : cursor + length
                    ].to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
                    cursor += length
        return inputs_embeds

    def _llm(self):
        return self.model.llm

    def _control_token_ids(self, device):
        """Cached tensor of full-duplex control token ids to upweight in the text CE."""
        cached = self._ctrl_ids_cache
        if cached is not None and cached.device == device:
            return cached
        import torch

        from mcpmft.tokenizer_tools import (
            CHUNK_EOS,
            INTERRUPT,
            LISTEN,
            SPEAK,
            TOOL_CALL_START,
            TURN_EOS,
        )

        ids = [
            LISTEN.token_id,
            SPEAK.token_id,
            INTERRUPT.token_id,
            TOOL_CALL_START.token_id,
            CHUNK_EOS.token_id,
            TURN_EOS.token_id,
        ]
        bc = self.tokenizer.convert_tokens_to_ids("<|backchannel|>")
        ids.append(int(bc))
        cached = torch.tensor(sorted(set(ids)), device=device, dtype=torch.long)
        self._ctrl_ids_cache = cached
        return cached

    def _tts(self):
        tts = self.model.tts
        if tts is None:
            raise RuntimeError("T2S loss requested but model.tts is not initialized")
        return tts

    def _detach_llm_for_tts(self) -> bool:
        mode = self.talker_args.detach_llm_for_tts
        if mode == "always":
            return True
        if mode == "never":
            return False
        return not self.tune_llm

    def build_talker_loss(self, llm_hidden, speech_segments: list[dict[str, Any]]):
        return self.build_duplex_unit_talker_loss(llm_hidden, speech_segments)

    def build_duplex_unit_talker_loss(self, llm_hidden, speech_segments: list[dict[str, Any]]):
        """Teacher-forced approximation of MiniCPMODuplex TTS `generate_chunk`.

        Layout per speak unit:
          [unit text/turn_eos conditions] audio_bos [this unit S3 codes]

        No talker text_eos is inserted. Segments sharing (batch_index, turn_group) are packed into
        one continuous TTS sequence so the talker KV/positions are continuous across speak units of
        the same assistant turn; different turn groups reset positions.
        """
        import torch
        import torch.nn.functional as F

        tts = self._tts()
        packed = []
        labels = []
        lengths = []

        groups: list[list[dict[str, Any]]] = []
        index: dict[tuple, int] = {}
        missing_codes: list[str] = []
        for segment in speech_segments:
            codes = segment.get("s3_codes")
            if codes is None:
                missing_codes.append(str(segment.get("audio_ref_id", "<unknown>")))
                continue
            key = (int(segment["batch_index"]), int(segment.get("turn_group", 0)))
            if key not in index:
                index[key] = len(groups)
                groups.append([])
            groups[index[key]].append(segment)
        if missing_codes:
            preview = ", ".join(missing_codes[:3])
            suffix = "..." if len(missing_codes) > 3 else ""
            raise RuntimeError(
                "Missing S3 codes for speech segment(s) while audio_loss_weight > 0: "
                f"{preview}{suffix}. Materialize them or provide turn.meta['s3_codes']."
            )

        max_talker = self.talker_args.max_talker_tokens
        audio_bos = torch.tensor([self.talker_args.audio_bos_id], device=llm_hidden.device, dtype=torch.long)
        audio_bos_emb = tts.emb_text(audio_bos)
        audio_eos_id = self.talker_args.audio_eos_id

        for group in groups:
            group = sorted(group, key=lambda s: (
                int(s.get("unit_index", 0) if s.get("unit_index") is not None else 0),
                int(s["text_token_positions"][0]) if s.get("text_token_positions") else 0,
            ))
            batch_index = int(group[0]["batch_index"])
            emb_parts: list = []
            tgt_parts: list = []
            for segment in group:
                if len(segment["text_token_positions"]) != len(segment["text_token_ids"]):
                    raise ValueError(
                        "Talker condition positions/token ids have different lengths: "
                        f"audio_ref_id={segment.get('audio_ref_id', '<unknown>')}"
                    )
                positions = torch.tensor(segment["text_token_positions"], device=llm_hidden.device, dtype=torch.long)
                if positions.numel() == 0:
                    raise ValueError(
                        "Talker supervision requires at least one thinker-token condition: "
                        f"audio_ref_id={segment.get('audio_ref_id', '<unknown>')}"
                    )
                text_ids = torch.tensor(segment["text_token_ids"], device=llm_hidden.device, dtype=torch.long)
                hidden = llm_hidden[batch_index].index_select(0, positions)
                projected = tts.projector_semantic(hidden)
                if _normalize_projected_hidden(tts):
                    projected = F.normalize(projected, p=2, dim=-1)
                cond = tts.emb_text(text_ids) + projected

                code_ids = torch.tensor(segment.get("s3_codes") or [], device=llm_hidden.device, dtype=torch.long)
                while code_ids.numel() and int(code_ids[-1].item()) == audio_eos_id:
                    code_ids = code_ids[:-1]
                if code_ids.numel() and (
                    int(code_ids.min().item()) < 0
                    or int(code_ids.max().item()) >= audio_eos_id
                ):
                    raise ValueError(
                        "S3 target contains an out-of-range or non-terminal EOS code: "
                        f"audio_ref_id={segment.get('audio_ref_id', '<unknown>')} "
                        f"valid_range=[0,{audio_eos_id - 1}]"
                    )
                predict_eos = bool(segment.get("should_predict_audio_eos", segment.get("is_turn_final", False)))

                emb_parts.append(cond)
                tgt_parts.append(torch.full((cond.size(0),), IGNORE_INDEX, device=llm_hidden.device, dtype=torch.long))

                first_code_tgt = (
                    int(code_ids[0].item())
                    if code_ids.numel()
                    else (audio_eos_id if predict_eos else IGNORE_INDEX)
                )
                emb_parts.append(audio_bos_emb)
                tgt_parts.append(torch.tensor([first_code_tgt], device=llm_hidden.device, dtype=torch.long))

                if code_ids.numel():
                    emb_parts.append(tts.emb_code[0](code_ids))
                    nxt = code_ids[1:]
                    tail = (
                        torch.tensor([audio_eos_id], device=llm_hidden.device, dtype=torch.long)
                        if predict_eos
                        else torch.tensor([IGNORE_INDEX], device=llm_hidden.device, dtype=torch.long)
                    )
                    code_tgt = torch.cat([nxt, tail])
                    tgt_parts.append(code_tgt[: code_ids.numel()])

            if not emb_parts:
                continue
            embeds = torch.cat(emb_parts, dim=0)
            target = torch.cat(tgt_parts, dim=0)
            if embeds.size(0) > max_talker:
                raise ValueError(
                    "Talker target exceeds its decoder context and would be truncated: "
                    f"batch_index={batch_index} turn_group={group[0].get('turn_group', 0)} "
                    f"packed_tokens={embeds.size(0)} max_talker_tokens={max_talker}. "
                    "Reduce the packed target length; silent S3 truncation is disabled."
                )
            packed.append(embeds)
            labels.append(target)
            lengths.append(embeds.size(0))

        if not packed:
            if self.text_loss_weight == 0:
                raise RuntimeError(
                    "Talker-only training batch has no valid assistant speech supervision. "
                    "Every selected row must serialize at least one speech target."
                )
            return self._normalize_loss_sum(llm_hidden.sum() * 0.0, llm_hidden.new_zeros(()))

        return self._run_tts_ce(llm_hidden, packed, labels, lengths)

    def _run_tts_ce(self, llm_hidden, packed, labels, lengths):
        import torch

        tts = self._tts()
        max_len = max(lengths)
        hidden_size = packed[0].size(-1)
        batch_size = len(packed)
        inputs = llm_hidden.new_zeros((batch_size, max_len, hidden_size))
        label_tensor = torch.full((batch_size, max_len), IGNORE_INDEX, device=llm_hidden.device, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), device=llm_hidden.device, dtype=torch.long)
        position_ids = torch.arange(max_len, device=llm_hidden.device, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)

        for idx, (embeds, target, length) in enumerate(zip(packed, labels, lengths)):
            inputs[idx, :length] = embeds
            label_tensor[idx, :length] = target
            attention_mask[idx, :length] = 1

        tts_out = tts.model(
            inputs_embeds=inputs,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        )
        hidden = tts_out.last_hidden_state
        logits = tts.head_code[0](hidden)
        audio_sum = _cross_entropy_fp32(
            logits.view(-1, logits.size(-1)),
            label_tensor.view(-1),
            ignore_index=IGNORE_INDEX,
            reduction="sum",
        )
        return self._normalize_loss_sum(audio_sum, (label_tensor != IGNORE_INDEX).sum())

    def _normalize_loss_sum(self, loss_sum, denom):
        """Normalize summed CE by the global valid-token count under DDP/DeepSpeed.

        DDP averages gradients across ranks. Returning ``local_sum * world_size / global_denom`` on
        each rank therefore gives the same gradient as a single process averaging over all valid
        tokens. Without this, each rank's local mean is weighted equally even when sequence lengths
        or control-token weights differ substantially.
        """
        import torch

        denom = torch.as_tensor(denom, device=loss_sum.device, dtype=loss_sum.dtype)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            global_denom = denom.detach().clone()
            torch.distributed.all_reduce(global_denom, op=torch.distributed.ReduceOp.SUM)
            world_size = torch.distributed.get_world_size()
            return loss_sum * world_size / global_denom.clamp_min(1.0)
        return loss_sum / denom.clamp_min(1.0)


def _normalize_projected_hidden(tts) -> bool:
    return bool(tts.config.normalize_projected_hidden)


def _cross_entropy_fp32(logits, labels, **kwargs):
    """Keep CE reduction and its normalization denominator in float32 under BF16 training."""
    import torch
    import torch.nn.functional as F

    use_autocast = logits.dtype in (torch.float16, torch.bfloat16) and (
        logits.device.type == "cuda"
        or (logits.device.type == "cpu" and logits.dtype == torch.bfloat16)
    )
    if use_autocast:
        # Cross entropy is an fp32 autocast op. This avoids the severe rounding of a BF16
        # reduction=sum without explicitly materializing another full fp32 logits tensor.
        with torch.autocast(
            device_type=logits.device.type,
            dtype=logits.dtype,
        ):
            return F.cross_entropy(logits, labels, **kwargs)
    return F.cross_entropy(logits, labels, **kwargs)


def _frozen_module_paths(args: FreezeArguments | None) -> tuple[str, ...]:
    if args is None:
        return ()
    plan = {
        "vision": args.tune_vision,
        "resampler": args.tune_resampler,
        "audio_encoder": args.tune_audio_encoder,
        "audio_proj": args.tune_audio_proj,
        "llm": args.tune_llm,
        "tts_proj": args.tune_tts_proj,
        "tts_decoder": args.tune_tts_decoder,
    }
    return tuple(
        dotted
        for group, trainable in plan.items()
        if not trainable
        for dotted in MODULE_PATHS[group]
    )
