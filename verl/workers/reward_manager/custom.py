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
Custom Reward Manager for LLM Bottleneck Training.

Implements the reward for the actor as:

    reward_actor = r_Mm - r_m - lambda * L / B

where:
    - r_Mm: reward from the translator conditioned on the actor output
    - r_m:  reward from the translator conditioned only on the original prompt
    - L:    actor response length (valid tokens)
    - B:    predefined bound (1000)
    - lambda: adaptive factor in [0, 1], updated after every batch:
        lambda <- clip_0_1(lambda + eta * (E[L]/B - 1)), eta=0.01

The translator reward r_m is expected to be pre-computed and stored in
`batch["m_prompt_rewards"]`, while r_Mm is stored in `batch["m_rewards"]`.
"""

from collections import defaultdict
from operator import is_
from typing import Any

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager


@register("custom")
class CustomRewardManager(AbstractRewardManager):
    """
    Custom Reward Manager that implements the actor reward with translator
    baselines and adaptive length penalty.
    """
    
    def __init__(self, tokenizer, m_tokenizer, num_examine, compute_score=None, reward_fn_key="data_source"):
        """
        Initialize the Custom Reward Manager.
        
        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: The number of batches of decoded responses to print to the console for debugging purpose.
            compute_score: A function to compute the reward score. If None, `default_compute_score` will be used.
            reward_fn_key: The key used to access the data source in the non-tensor batch data. Defaults to
                "data_source".
        """
        self.tokenizer = tokenizer  # Store the tokenizer for decoding token IDs
        self.m_tokenizer = m_tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key  # Store the key for accessing the data source

        # Adaptive length penalty parameters
        self.length_bound = 1000.0
        self.lambda_factor = 0.0
        self.lambda_eta = 0.01
    
    def __call__(self, data: DataProto, return_dict: bool = False, is_train: bool = True) -> torch.Tensor | dict[str, Any]:
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        # Determine whether we should treat the batch as coming from the translator
        # or from the actor. Validation without translator should use the actor tokenizer.
        use_translator = None
        if getattr(data, "meta_info", None) is not None:
            use_translator = data.meta_info.get("use_translator")

        if use_translator is False:
            m_flag = False
            _tokenizer = self.tokenizer
        elif "m_rewards" in data.batch.keys():
            m_flag = False
            _tokenizer = self.tokenizer
        else:
            m_flag = True
            _tokenizer = self.m_tokenizer

        #print(f"m_flag: {m_flag}")

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}
        response_lengths: list[float] = []

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = _tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = _tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
            extra_info["num_turns"] = num_turns

            if not m_flag and "m_rewards" in data.batch.keys():
                # r_Mm: translator reward conditioned on actor output
                r_mm = data_item.batch["m_rewards"]
                # r_m: translator reward conditioned only on prompt
                r_m = data_item.batch.get("m_prompt_rewards", 0.0)

                r_mm_val = r_mm.item() if isinstance(r_mm, torch.Tensor) else float(r_mm)
                r_m_val = r_m.item() if isinstance(r_m, torch.Tensor) else float(r_m)

                reward = r_mm_val - r_m_val
            else:
                score = self.compute_score(
                    data_source=data_source,
                    solution_str=response_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                )

                if isinstance(score, dict):
                    reward = score["score"]
                    # Store the information including original reward
                    for key, value in score.items():
                        reward_extra_info[key].append(value)
                else:
                    reward = score
            
            if i == 0 and is_train and m_flag:
                print("[prompt]", prompt_str)
                print("[m response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)
            
            # apply adaptive length penalty
            if is_train and not m_flag:
                penalty = self.lambda_factor * (valid_response_length / self.length_bound)
                reward = reward - penalty
                response_lengths.append(float(valid_response_length))
            
            reward_tensor[i, valid_response_length - 1] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[m response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

        # Update lambda using batch mean length
        if is_train and response_lengths:
            mean_len = sum(response_lengths) / len(response_lengths)
            self.lambda_factor += self.lambda_eta * (mean_len / self.length_bound - 1.0)
            self.lambda_factor = max(0.0, min(1.0, self.lambda_factor))
        
        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor