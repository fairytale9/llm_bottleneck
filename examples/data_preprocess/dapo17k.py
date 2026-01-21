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
Preprocess the MATH-lighteval dataset to parquet format
"""

import argparse
import json
import os

import datasets

from verl.utils.hdfs_io import copy, makedirs
from verl.utils.reward_score.math import last_boxed_only_string, remove_boxed


def extract_solution(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/dapo17k")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    # 'lighteval/MATH' is no longer available on huggingface.
    # Use mirror repo: DigitalLearningGmbH/MATH-lighteval
    data_source = "math_dapo"
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    dataset = datasets.load_dataset("open-r1/DAPO-Math-17k-Processed", "en", trust_remote_code=True)

    train_dataset = dataset["train"]

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            raw_question = example.pop("prompt")
            prompt = raw_question + " Let's think step by step and output the final answer within \\boxed{}."
            data = {
                "data_source": example.pop("data_source"),
                "prompt": [{"role": "user", "content": prompt}],
                "ability": "math",
                "reward_model": example.pop("reward_model"),
                "extra_info": example.pop("extra_info"),
                "raw_question": raw_question,
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)

    local_dir = os.path.expanduser(args.local_dir)
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    # Save one example as JSON for reference
    example = train_dataset[0]
    with open(os.path.join(local_dir, "train_example.json"), "w") as f:
        json.dump(example, f, indent=2)
    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
