# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Implement a multiprocess PPOTranslator
"""

import logging
import os
import re

import torch
import torch.distributed
from torch import nn, optim
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.utils.device import get_device_id, get_device_name, is_cuda_available, is_npu_available
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_functional import masked_mean, logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad_and_slice_inputs
from verl.workers.translator import BasePPOTranslator
import verl.utils.torch_functional as verl_F



if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOTranslator(BasePPOTranslator):
    def __init__(self, config, translator_module: nn.Module, translator_optimizer: optim.Optimizer, translator_tokenizer):
        super().__init__(config=config)
        self.translator_module = translator_module
        self.translator_optimizer = translator_optimizer
        self.use_remove_padding = self.config.model.get("use_remove_padding", False)
        print(f"Translator use_remove_padding={self.use_remove_padding}")

        self.ulysses_sequence_parallel_size = self.config.get("ulysses_sequence_parallel_size", 1)
        self.device_name = get_device_name()
        self.tokenizer = translator_tokenizer

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.translator_module, FSDP):
            grad_norm = self.translator_module.clip_grad_norm_(self.config.grad_clip)
        elif isinstance(self.translator_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.translator_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.translator_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: grad_norm is not finite: {grad_norm}")
            self.translator_optimizer.zero_grad()
        else:
            self.translator_optimizer.step()
        return grad_norm

    def _process_micro_batch(self, micro_batch, temperature, is_train=False, calculate_entropy=False):
        reasoning_ids_list = []
        reasoning_mask_list = []
        for response_str, prompt_str in zip(micro_batch["raw_responses"], micro_batch["raw_prompts"]):
            parts = response_str.split("</think>")
            reasoning = parts[0]
            prompt_chat_str = prompt_str + reasoning + "</think>\nYou are given a question and a reasoning process. Do NOT re-derive the solution. Based only on the reasoning above, output the final answer:"
            #prompt_chat = [{"content": "Question: "+prompt_str+"\nReasoning: "+response_str+"</think>\nYou are given a question and a reasoning process. Do NOT re-derive the solution. Based only on the reasoning above, output the final answer:", "role": "user"}]
            #prompt_chat_str = self.tokenizer.apply_chat_template(
            #    prompt_chat, add_generation_prompt=True, tokenize=False, enable_reasoning=False
            #)
            print(f"Prompt chat str: {prompt_chat_str}")
            #match = re.search(r"<think>(.*?)</think>", response_str, re.DOTALL)
            #if match:
            #    reasoning = match.group(1)
            #else:
            #    reasoning = response_str
            
            # Tokenize the conditional prompt
            r_input_ids, r_attention_mask = verl_F.tokenize_and_postprocess_data(
                prompt=prompt_chat_str,
                tokenizer=self.tokenizer,
                max_length=4608,
                pad_token_id=self.tokenizer.pad_token_id,
                left_pad=True,
                truncation='right'
            )
            reasoning_ids_list.append(r_input_ids[0])
            reasoning_mask_list.append(r_attention_mask[0])

        M_reasoning_ids = torch.stack(reasoning_ids_list, dim=0).to(get_device_id())
        M_reasoning_mask = torch.stack(reasoning_mask_list, dim=0).to(get_device_id())

        targets = micro_batch["target_ids"]
        target_length = targets.size(-1)
        targets_attention_mask = micro_batch["target_attention_mask"]

        input_ids = torch.cat((M_reasoning_ids, targets), dim=1)
        attention_mask = torch.cat((M_reasoning_mask, targets_attention_mask), dim=1)

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            entropy = None
            output = self.translator_module(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )  # prevent model thinks we are generating

            logits = output.logits
            logits.div_(temperature)
            
            # Extract logits for target tokens (at the end of the sequence)
            target_logits = logits[:, -target_length-1:-1, :]  # (bsz, target_length, vocab_size)
            if is_train:
                return target_logits, targets, targets_attention_mask
            
            # compute log probabilities for the target tokens
            log_probs = logprobs_from_logits(target_logits, targets)
            
            if calculate_entropy:
                entropy = torch.distributions.Categorical(logits=target_logits).entropy()  # (bsz, target_length)

        return entropy, log_probs

    @GPUMemoryLogger(role="dp translator", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> tuple[torch.Tensor, torch.Tensor]:
        self.translator_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error

        select_keys = ["target_ids", "target_attention_mask"]
        non_tensor_select_keys = ["raw_responses", "raw_prompts"]
        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        micro_batches = data.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs = self._process_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                )
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)

        return log_probs, entropys

    @GPUMemoryLogger(role="dp translator", logger=logger)
    def update_translator(self, data: DataProto):
        # make sure we are in training mode
        self.translator_module.train()
        metrics = {}

        if "advantages" in data.batch:
            # RL Training
            select_keys = ["target_ids", "target_attention_mask", "old_log_probs", "advantages"]
            non_tensor_select_keys = ["raw_responses", "raw_prompts"]
        else:
            # SFT Training
            select_keys = ["target_ids", "target_attention_mask"] # "prompts", "responses", "attention_mask"
            non_tensor_select_keys = ["raw_responses", "raw_prompts"]
        
        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Split to make minibatch iterator for updating the translator
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                self.gradient_accumulation = (
                    self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                )
                micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.translator_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}

                    if "advantages" in micro_batch.batch:
                        # RL Update
                        entropy, log_probs = self._process_micro_batch(
                            model_inputs, temperature=1.0, is_train=False, calculate_entropy=True
                        )
                        old_log_probs = micro_batch.batch["old_log_probs"]
                        advantages = micro_batch.batch["advantages"]
                        response_mask = micro_batch.batch["target_attention_mask"]

                        pg_loss, clipfrac, approx_kl, _ = core_algos.compute_policy_loss(
                             old_log_prob=old_log_probs,
                             log_prob=log_probs,
                             advantages=advantages,
                             response_mask=response_mask,
                             loss_agg_mode="token-mean", # Use default or config
                             config=self.config
                        )
                         
                        loss = pg_loss - self.config.entropy_coef * masked_mean(entropy, response_mask)
                        
                        loss_scale_factor = 1 / self.gradient_accumulation
                        loss = loss * loss_scale_factor
                        loss.backward()
                         
                        micro_batch_metrics.update({
                             "translator/pg_loss": pg_loss.item(),
                             "translator/entropy": masked_mean(entropy, response_mask).item(),
                             "translator/approx_kl": approx_kl.item()
                        })

                    else:
                        # SFT Update
                        logits, targets, loss_mask = self._process_micro_batch(model_inputs, temperature=1.0, is_train=True)
                        
                        logits = logits.contiguous()
                        targets = targets.contiguous()
                        loss_mask = loss_mask.contiguous().view(-1)

                        # calculate cross-entropy loss
                        loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
                        loss = loss_fn(
                            logits.view(-1, logits.size(-1)), 
                            targets.view(-1)
                        )
                        loss = loss * loss_mask
                        loss = torch.sum(loss) / torch.sum(loss_mask)
                        loss_scale_factor = 1 / self.gradient_accumulation
                        loss = loss * loss_scale_factor
                        
                        loss.backward()

                        micro_batch_metrics.update(
                            {
                                "translator/loss": loss.detach().item(),
                                #"translator/logits_mean": masked_mean(logits, response_mask).detach().item(),
                            }
                        )

                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"translator/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
        self.translator_optimizer.zero_grad()
        return metrics


    @GPUMemoryLogger(role="dp translator", logger=logger)
    def generate(self, data: DataProto):
        self.translator_module.eval()

        temperature = data.meta_info.get("temperature", 1.0)
        do_sample = data.meta_info.get("do_sample", False)
        max_new_tokens = data.meta_info.get("max_new_tokens", 1024)

        prompts = []
        raw_prompts = data.non_tensor_batch["raw_prompts"]
        raw_responses = data.non_tensor_batch["raw_responses"]
        
        for response_str, prompt_str in zip(raw_responses, raw_prompts):
            if "</think>" in response_str:
                parts = response_str.split("</think>")
                reasoning = parts[0] + "</think>"
            else:
                reasoning = response_str 
            
            prompt_chat_str = prompt_str + reasoning + "\nYou are given a question and a reasoning process. Do NOT re-derive the solution. Based only on the reasoning above, output the final answer:"
            prompts.append(prompt_chat_str)

        input_ids_list = []
        attention_mask_list = []
        
        for p in prompts:
             ids, mask = verl_F.tokenize_and_postprocess_data(
                prompt=p,
                tokenizer=self.tokenizer,
                max_length=4096, 
                pad_token_id=self.tokenizer.pad_token_id,
                left_pad=True,
                truncation='left'
            )
             input_ids_list.append(ids[0])
             attention_mask_list.append(mask[0])
             
        input_ids = torch.stack(input_ids_list).to(get_device_id())
        attention_mask = torch.stack(attention_mask_list).to(get_device_id())
        
        with torch.no_grad():
             output_ids = self.translator_module.generate(
                 input_ids=input_ids,
                 attention_mask=attention_mask,
                 max_new_tokens=max_new_tokens,
                 do_sample=do_sample,
                 temperature=temperature,
                 pad_token_id=self.tokenizer.pad_token_id,
                 eos_token_id=self.tokenizer.eos_token_id,
             )
             
        # Extract generated part
        generated_sequences = output_ids[:, input_ids.shape[1]:]
        
        generated_texts = []
        for seq in generated_sequences:
            text = self.tokenizer.decode(seq, skip_special_tokens=True)
            generated_texts.append(text)
            
        return DataProto.from_dict(
            tensors={"m_responses": generated_sequences},
            non_tensors={"m_raw_responses": generated_texts}
        )


    def _forward_micro_batch_for_log_prob(self, micro_batch, temperature, calculate_entropy=False):
        """
        Forward pass for computing log probabilities of targets given responses.
        Similar to actor's _forward_micro_batch.
        
        Args:
            micro_batch: Dictionary containing input_ids, attention_mask, position_ids, responses, and target
            temperature: Temperature for scaling logits
            calculate_entropy: Whether to calculate entropy
            
        Returns:
            entropy: # (bs, target_len) or None
            log_probs: # (bs, target_len) - log probabilities of target tokens given responses
        """
        response_length = micro_batch["responses"].size(-1)
        target_length = micro_batch["target"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            for key in micro_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat(
                    [inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0
                )

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                # for compute the log_prob - use target tokens instead of rolled input_ids
                # We need to extract target tokens from the full sequence
                # target tokens should be at the end of the sequence after responses
                total_nnz = input_ids_rmpad.size(1)
                target_start_idx = total_nnz - target_length * batch_size
                target_rmpad = input_ids_rmpad[:, target_start_idx:]  # (1, target_length * batch_size)

                # pad and slice the inputs if sp > 1
                if self.ulysses_sequence_parallel_size > 1:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad, position_ids_rmpad, sp_size=self.ulysses_sequence_parallel_size
                    )
                    target_rmpad, _, _ = ulysses_pad_and_slice_inputs(
                        target_rmpad,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                target_rmpad = target_rmpad.squeeze(0)  # ((target_length * batch_size / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                output = self.translator_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                )  # prevent model thinks we are generating

                logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                logits_rmpad.div_(temperature)

                # Extract logits for target tokens (at the end of the sequence)
                target_logits_rmpad = logits_rmpad[-target_rmpad.size(0):]  # (target_length * batch_size / sp, vocab_size)

                # compute log probabilities for target tokens
                inplace_backward = True
                if calculate_entropy:
                    inplace_backward = False
                log_probs = logprobs_from_logits(
                    logits=target_logits_rmpad,
                    labels=target_rmpad,
                    inplace_backward=inplace_backward,
                )

                # compute entropy if requested
                if calculate_entropy:
                    entropy_rmpad = torch.distributions.Categorical(logits=target_logits_rmpad).entropy()

                # gather log_prob if sp > 1
                if self.ulysses_sequence_parallel_size > 1:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )

                # pad back to (bsz, target_length)
                if calculate_entropy:
                    entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=target_length,
                    ).squeeze(-1)  # (bsz, target_length)
                log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=target_length,
                ).squeeze(-1)  # (bsz, target_length)

            else:  # not using rmpad and no ulysses sp
                output = self.translator_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                )  # prevent model thinks we are generating

                logits = output.logits
                logits.div_(temperature)
                # Extract logits for target tokens (at the end of the sequence)
                target_logits = logits[:, -target_length:, :]  # (bsz, target_length, vocab_size)
                
                # compute log probabilities for the target tokens
                log_probs = logprobs_from_logits(target_logits, micro_batch["target"])
                
                if calculate_entropy:
                    entropy = torch.distributions.Categorical(logits=target_logits).entropy()  # (bsz, target_length)

            return entropy, log_probs