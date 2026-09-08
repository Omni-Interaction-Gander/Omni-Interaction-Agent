from __future__ import annotations

import inspect
import json
import os
from pathlib import Path

from mcpmft.modeling.freeze import group_named_parameters
from mcpmft.modeling.load import load_partial_state_dict
from mcpmft.train.run_state import write_project_config
from mcpmft.utils.state_dict import strip_wrapper_prefix


def _import_trainer():
    try:
        from transformers import Trainer
    except ImportError as exc:
        raise RuntimeError(
            "Training requires the dependencies declared by the mcpmft package."
        ) from exc
    return Trainer


class CPMTrainer(_import_trainer()):
    def __init__(
        self,
        *args,
        projector_learning_rate: float | None = None,
        tts_learning_rate: float | None = None,
        project_config: dict | None = None,
        **kwargs,
    ) -> None:
        self.projector_learning_rate = projector_learning_rate
        self.tts_learning_rate = tts_learning_rate
        self.project_config = project_config
        self._last_checkpoint_step: int | None = None
        self._component_loss_sums: dict[str, object | None] = {
            "text_loss": None,
            "audio_loss": None,
        }
        self._component_loss_counts = {"text_loss": 0, "audio_loss": 0}
        self._run_start_global_step = 0
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        loss = outputs.loss
        if self.is_in_train and model.training:
            for name in self._component_loss_sums:
                value = getattr(outputs, name, None)
                if value is None:
                    continue
                detached = value.detach().float()
                current = self._component_loss_sums[name]
                self._component_loss_sums[name] = (
                    detached if current is None else current + detached
                )
                self._component_loss_counts[name] += 1
        return (loss, outputs) if return_outputs else loss

    def log(self, logs: dict[str, float], start_time=None) -> None:
        if "loss" in logs:
            logs.update(self._consume_component_logs(total_loss=float(logs["loss"])))
        if "train_loss" in logs:
            completed_steps = int(self.state.global_step) - self._run_start_global_step
            if completed_steps > 0:
                # Report loss over steps completed by this trainer invocation.
                logs["train_loss"] = float(self._total_loss_scalar) / completed_steps
        super().log(logs, start_time)

    def _consume_component_logs(self, *, total_loss: float) -> dict[str, float]:
        import torch

        names = tuple(self._component_loss_sums)
        values = []
        for name in names:
            value = self._component_loss_sums[name]
            if value is None:
                value = torch.zeros((), device=self.args.device, dtype=torch.float32)
            values.extend(
                [
                    value.to(device=self.args.device, dtype=torch.float32),
                    torch.tensor(
                        float(self._component_loss_counts[name]),
                        device=self.args.device,
                        dtype=torch.float32,
                    ),
                ]
            )
        local = torch.stack(values)
        gathered = self._nested_gather(local).reshape(-1, len(values))
        result = {"component/loss": total_loss}
        for index, name in enumerate(names):
            summed = gathered[:, 2 * index].sum()
            count = gathered[:, 2 * index + 1].sum()
            result[f"component/{name}"] = float(
                (summed / count.clamp_min(1.0)).item()
            )
            self._component_loss_sums[name] = None
            self._component_loss_counts[name] = 0
        return result

    def train(self, resume_from_checkpoint=None, *args, **kwargs):
        self._run_start_global_step = _checkpoint_global_step(resume_from_checkpoint)
        self._last_checkpoint_step = (
            self._run_start_global_step if resume_from_checkpoint else None
        )
        if (
            resume_from_checkpoint
            and self.is_deepspeed_enabled
            and getattr(self, "save_trainable_only", False)
        ):
            # Load trainable weights before DeepSpeed restores compact optimizer shards.
            self._load_from_checkpoint(resume_from_checkpoint)
        result = super().train(resume_from_checkpoint, *args, **kwargs)
        corrected = result.metrics.get("train_loss")
        if corrected is not None and float(corrected) != float(result.training_loss):
            result = type(result)(result.global_step, float(corrected), result.metrics)
        return result

    def _inner_training_loop(self, *args, **kwargs):
        if not (
            self.is_deepspeed_enabled and getattr(self, "save_trainable_only", False)
        ):
            return super()._inner_training_loop(*args, **kwargs)

        # Compact checkpoints require non-strict DeepSpeed restore for this invocation.
        import transformers.trainer as trainer_module

        original_loader = trainer_module.deepspeed_load_checkpoint

        def load_trainable_checkpoint(engine, checkpoint_path, load_module_strict=True):
            return original_loader(
                engine,
                checkpoint_path,
                load_module_strict=False,
            )

        trainer_module.deepspeed_load_checkpoint = load_trainable_checkpoint
        try:
            return super()._inner_training_loop(*args, **kwargs)
        finally:
            trainer_module.deepspeed_load_checkpoint = original_loader

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer
        import torch

        lr = self.args.learning_rate
        decay_parameter_names = self.get_decay_parameter_names(self.model)
        param_groups = group_named_parameters(
            self.model,
            base_lr=lr,
            projector_lr=self.projector_learning_rate,
            tts_lr=self.tts_learning_rate,
            weight_decay=self.args.weight_decay,
            decay_parameter_names=decay_parameter_names,
        )
        self.optimizer = torch.optim.AdamW(
            param_groups,
            lr=lr,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            eps=self.args.adam_epsilon,
            # Each group carries its own weight decay.
            weight_decay=0.0,
        )
        return self.optimizer

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        """Restore canonical MiniCPMO weights inside OmniTrainWrapper.

        Checkpoints store inference-facing keys such as ``tts.model.*``. Resume loads them
        into the wrapped MiniCPMO and validates the trainable subset separately.
        """
        candidate = self.model if model is None else model
        wrapped_module = getattr(candidate, "module", None)
        if candidate is self.model or wrapped_module is self.model:
            checkpoint_model = getattr(self.model, "model", self.model)
        else:
            checkpoint_model = candidate

        if getattr(self, "save_trainable_only", False):
            missing, unexpected = load_partial_state_dict(
                checkpoint_model,
                resume_from_checkpoint,
                strict=False,
            )
            trainable_names = {
                name for name, parameter in checkpoint_model.named_parameters()
                if parameter.requires_grad
            }
            missing_trainable = sorted(trainable_names.intersection(missing))
            if unexpected or missing_trainable:
                raise RuntimeError(
                    "Trainable-only checkpoint is incompatible: "
                    f"unexpected={unexpected[:5]}, missing_trainable={missing_trainable[:5]}"
                )
            return None

        return super()._load_from_checkpoint(
            resume_from_checkpoint,
            model=checkpoint_model,
        )

    def _save(self, output_dir=None, state_dict=None):
        model_to_save = getattr(self.model, "model", self.model)
        if getattr(self, "save_trainable_only", False):
            state_dict = {
                strip_wrapper_prefix(name): param.detach().cpu()
                for name, param in self.model.named_parameters()
                if param.requires_grad
            }
        elif state_dict is None:
            state_dict = model_to_save.state_dict()
        else:
            state_dict = {strip_wrapper_prefix(k): v for k, v in state_dict.items()}

        original_model = self.model
        try:
            # Save MiniCPMO with canonical llm.*, tts.*, and apm.* keys.
            self.model = model_to_save
            result = super()._save(output_dir=output_dir, state_dict=state_dict)
        finally:
            self.model = original_model
        if self.project_config is not None and self.args.should_save:
            write_project_config(
                self.project_config,
                output_dir or self.args.output_dir,
            )
        return result

    def save_model(self, output_dir=None, _internal_call=False):
        if not (
            self.is_deepspeed_enabled and getattr(self, "save_trainable_only", False)
        ):
            return super().save_model(
                output_dir=output_dir,
                _internal_call=_internal_call,
            )

        # ZeRO-2 can export replicated trainable Talker tensors directly.
        output_dir = output_dir or self.args.output_dir
        if self.args.should_save:
            self._save(output_dir=output_dir)
        if self.args.push_to_hub and not _internal_call:
            self.push_to_hub(commit_message="Model save")

    def _save_optimizer_and_scheduler(self, output_dir):
        if not (
            self.is_deepspeed_enabled and getattr(self, "save_trainable_only", False)
        ):
            return super()._save_optimizer_and_scheduler(output_dir)

        save_checkpoint = self.model_wrapped.save_checkpoint
        if "exclude_frozen_parameters" not in inspect.signature(save_checkpoint).parameters:
            raise RuntimeError(
                "Installed DeepSpeed cannot omit frozen parameters from exact-resume checkpoints"
            )
        # All ranks save optimizer state; frozen Thinker tensors come from init_checkpoint.
        save_checkpoint(output_dir, exclude_frozen_parameters=True)
        if self.args.should_save:
            import torch

            torch.save(
                self.lr_scheduler.state_dict(),
                os.path.join(output_dir, "scheduler.pt"),
            )

    def save_final_checkpoint(self) -> None:
        """Persist an exact-resume checkpoint at the final global step.

        ``Trainer.save_model`` writes serving weights only. DeepSpeed optimizer, scheduler, RNG,
        and engine shards are written by ``_save_checkpoint`` and otherwise may lag the final model
        by up to ``save_steps - 1`` updates.
        """
        if self._last_checkpoint_step != self.state.global_step:
            self._save_checkpoint(self.model, trial=None)

    def _save_checkpoint(self, model, trial):
        result = super()._save_checkpoint(model, trial)
        self._last_checkpoint_step = int(self.state.global_step)
        return result


def _checkpoint_global_step(resume_from_checkpoint) -> int:
    if not resume_from_checkpoint or isinstance(resume_from_checkpoint, bool):
        return 0
    state_path = Path(resume_from_checkpoint) / "trainer_state.json"
    if not state_path.is_file():
        return 0
    with state_path.open("r", encoding="utf-8") as handle:
        return int(json.load(handle).get("global_step", 0))
