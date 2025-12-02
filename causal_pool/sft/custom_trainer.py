from functools import partial
from typing import Callable, Optional, Union, Any

import contextlib
import torch
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel
from torch.utils.data import DataLoader, Dataset

from transformers.trainer_utils import seed_worker
from transformers.utils import (
    is_accelerate_available,
    is_datasets_available,
    is_sagemaker_mp_enabled,
    is_torch_hpu_available,
    is_torch_mlu_available,
    is_torch_mps_available,
    is_torch_musa_available,
    is_torch_npu_available,
    is_torch_xpu_available,
    logging,
)
from transformers import Seq2SeqTrainer
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.integrations.fsdp import is_fsdp_managed_module
from transformers.training_args import OptimizerNames

if is_datasets_available():
    import datasets

logger = logging.get_logger(__name__)

if is_sagemaker_mp_enabled():
    from transformers.trainer_pt_utils import smp_forward_backward

if is_accelerate_available():
    from accelerate.utils import DistributedType

class CausalPoolTrainer(Seq2SeqTrainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Track input lengths for logging
        self._total_input_length = 0.0
        self._num_batches_logged = 0

    def _training_step(
        self,
        model: nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        num_items_in_batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        # Prepare buffers for context parallelism

        cp_context, inputs = self._prepare_context_parallel_inputs(model, inputs)

        # Context manager is no-op if CP isn't enabled
        with cp_context():
            model.train()
            if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
                self.optimizer.train()

            torch.cuda.reset_peak_memory_stats(self.args.device)

            inputs = self._prepare_inputs(inputs)
            
            base = torch.cuda.max_memory_allocated(self.args.device)
            
            if is_sagemaker_mp_enabled():
                loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
                return loss_mb.reduce_mean().detach().to(self.args.device)

            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)

            torch.cuda.synchronize()
            after_fwd = torch.cuda.max_memory_allocated(self.args.device)

            del inputs
            if (
                self.args.torch_empty_cache_steps is not None
                and self.state.global_step % self.args.torch_empty_cache_steps == 0
            ):
                if is_torch_xpu_available():
                    torch.xpu.empty_cache()
                elif is_torch_mlu_available():
                    torch.mlu.empty_cache()
                elif is_torch_musa_available():
                    torch.musa.empty_cache()
                elif is_torch_npu_available():
                    torch.npu.empty_cache()
                elif is_torch_mps_available():
                    torch.mps.empty_cache()
                elif is_torch_hpu_available():
                    logger.warning(
                        "`torch_empty_cache_steps` is set but HPU device/backend does not support empty_cache()."
                    )
                else:
                    torch.cuda.empty_cache()

            kwargs = {}

            # For LOMO optimizers you need to explicitly use the learning rate
            if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
                kwargs["learning_rate"] = self._get_learning_rate()

            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            if self.use_apex:
                from apex import amp

                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                # Finally we need to normalize the loss for reporting if GA loss bug is not fixed during compute loss
                if (
                    not self.model_accepts_loss_kwargs or num_items_in_batch is None
                ) and self.compute_loss_func is None:
                    # If the model does not accept loss kwargs, we need to normalize the loss by the number of gradient accumulation steps
                    loss = loss / self.current_gradient_accumulation_steps

                # Turning off loss scaling w.r.t. gradient accumulation when DeepSpeed is enabled
                # https://github.com/huggingface/transformers/pull/35808
                if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
                    kwargs["scale_wrt_gas"] = False

                self.accelerator.backward(loss, **kwargs)

            torch.cuda.synchronize()
            after_bwd = torch.cuda.max_memory_allocated(self.args.device)

            print(
                f"[step {self.state.global_step}] "
                f"base={base/1e9:.2f} GB, "
                f"after_fwd={after_fwd/1e9:.2f} GB, "
                f"after_bwd={after_bwd/1e9:.2f} GB"
            )

            return loss.detach()

    def training_step(
        self,
        model: nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        num_items_in_batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        seq_lengths = inputs["attention_mask"].sum(dim=1).float()
        avg_len = seq_lengths.mean().item()
        
        self._total_input_length += avg_len
        self._num_batches_logged += 1

        return self._training_step(model, inputs, num_items_in_batch)

    def _maybe_log_save_evaluate(
        self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time, learning_rate=None
    ):
        avg_input_len = None
        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
            if self._num_batches_logged > 0:
                avg_input_len = self._total_input_length / self._num_batches_logged
                self._total_input_length = 0.0
                self._num_batches_logged = 0
        
        super()._maybe_log_save_evaluate(tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time, learning_rate)
        
        if avg_input_len is not None:
            self.log({"avg_input_len": avg_input_len}, start_time)

    def prediction_step(
        self,
        model: nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
        **gen_kwargs,
    ) -> tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            gen_kwargs:
                Additional `generate` specific kwargs.

        Return:
            tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """

        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        # Priority (handled in generate):
        # non-`None` gen_kwargs > model.generation_config > default GenerationConfig()
        if len(gen_kwargs) == 0 and hasattr(self, "_gen_kwargs"):
            gen_kwargs = self._gen_kwargs.copy()
        if "num_beams" in gen_kwargs and gen_kwargs["num_beams"] is None:
            gen_kwargs.pop("num_beams")
        if "max_length" in gen_kwargs and gen_kwargs["max_length"] is None:
            gen_kwargs.pop("max_length")

        default_synced_gpus = is_deepspeed_zero3_enabled() or is_fsdp_managed_module(self.model)
        gen_kwargs["synced_gpus"] = gen_kwargs.get("synced_gpus", default_synced_gpus)

        # --- scott: modification starts here ---
        # generation_inputs = inputs.pop("generation_inputs")
        generation_inputs = inputs.copy()
        generation_inputs.pop("labels")
        # TODO: this is a hack to remove the label and the turn end token, assuming single token label and 2 token turn end
        generation_inputs["input_ids"] = generation_inputs["input_ids"][:, :-3]
        generation_inputs["attention_mask"] = generation_inputs["attention_mask"][:, :-3]
        # If the `decoder_input_ids` was created from `labels`, evict the former, so that the model can freely generate
        # (otherwise, it would continue generating from the padded `decoder_input_ids`)
        # if (
        #     "labels" in generation_inputs
        #     and "decoder_input_ids" in generation_inputs
        #     and generation_inputs["labels"].shape == generation_inputs["decoder_input_ids"].shape
        # ):
        #     generation_inputs = {
        #         k: v for k, v in inputs.items() if k not in ("decoder_input_ids", "decoder_attention_mask")
        #     }        
        # --- scott: modification ends here ---

        summon_full_params_context = (
            FullyShardedDataParallel.summon_full_params(self.model)
            if isinstance(self.model, FullyShardedDataParallel)
            else contextlib.nullcontext()
        )

        with summon_full_params_context:
            generated_tokens = self.model.generate(**generation_inputs, **gen_kwargs)
            
        # --- scott: modification starts here ---
        prompt_len = generation_inputs["attention_mask"].size(1)
        generated_tokens = generated_tokens[:, prompt_len:]
        # --- scott: modification ends here ---

        # Temporary hack to ensure the generation config is not initialized for each iteration of the evaluation loop
        # TODO: remove this hack when the legacy code that initializes generation_config from a model config is
        # removed in https://github.com/huggingface/transformers/blob/98d88b23f54e5a23e741833f1e973fdf600cc2c5/src/transformers/generation/utils.py#L1183
        if self.model.generation_config._from_model_config:
            self.model.generation_config._from_model_config = False

        # Retrieves GenerationConfig from model.generation_config
        gen_config = self.model.generation_config
        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_config.max_length:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_config.max_length)
        elif gen_config.max_new_tokens is not None and generated_tokens.shape[-1] < gen_config.max_new_tokens + 1:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_config.max_new_tokens + 1)

        with torch.no_grad():
            if has_labels:
                with self.compute_loss_context_manager():
                    outputs = model(**inputs)
                if self.label_smoother is not None:
                    loss = self.label_smoother(outputs, inputs["labels"]).detach().mean()
                else:
                    loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).detach().mean()
            else:
                loss = None

        if self.args.prediction_loss_only:
            return loss, None, None

        if has_labels:
            labels = inputs["labels"]
            if labels.shape[-1] < gen_config.max_length:
                labels = self._pad_tensors_to_max_len(labels, gen_config.max_length)
            elif gen_config.max_new_tokens is not None and labels.shape[-1] < gen_config.max_new_tokens + 1:
                labels = self._pad_tensors_to_max_len(labels, gen_config.max_new_tokens + 1)
        else:
            labels = None

        return loss, generated_tokens, labels