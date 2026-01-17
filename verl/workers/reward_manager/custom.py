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

This reward manager implements the custom reward formula:
reward = 1/(length of the response) - logp of the response

Where:
- response_length: The number of tokens in the generated response
- logp: Log probability computed by the translator worker and stored in batch.batch["m_logp"]
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
    Custom Reward Manager that implements the formula: reward = 1/(response_length) - logp
    
    This reward manager can access batch data directly and compute rewards based on
    log probabilities stored in batch.batch["m_logp"] by the TranslatorWorker.
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

        self.alpha = 1 / 2000 # length penalty
    
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
                score = data_item.batch["m_rewards"]
                reward = score
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
            
            # apply length penalty
            if is_train and not m_flag:
                reward = reward - self.alpha * valid_response_length
            
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

        #if is_train:
        #    self.alpha *= 0.9
        
        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor