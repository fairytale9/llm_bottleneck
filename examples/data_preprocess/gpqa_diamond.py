# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
Preprocess the Idavidrein/gpqa (GPQA Diamond) dataset
"""

import argparse
import os
import random

import datasets

from verl.utils.hdfs_io import copy, makedirs

GPQA_QUERY_TEMPLATE = (
    "Answer the following multiple choice question. The last line of your response should be of the following "
    "format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before "
    "answering.\n\n{Question}\n\nA) {A}\nB) {B}\nC) {C}\nD) {D}"
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/concise_prompt/gpqa_diamond")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    data_source = "Idavidrein/gpqa"
    test_dataset = datasets.load_dataset(data_source, "gpqa_diamond", split="train")

    def make_map_fn(split):
        def process_fn(example, idx):
            question = example["Question"]
            choices = [example["Incorrect Answer 1"], example["Incorrect Answer 2"], example["Incorrect Answer 3"]]
            random.shuffle(choices)
            gold_index = random.randint(0, 3)
            choices.insert(gold_index, example["Correct Answer"])
            prompt = GPQA_QUERY_TEMPLATE.format(
                Question=question, A=choices[0], B=choices[1], C=choices[2], D=choices[3]
            )
            gold_choice = "ABCD"[gold_index]

            data = {
                "data_source": data_source,
                "prompt": [{"role": "user", "content": prompt}],
                "ability": "science",
                "reward_model": {"style": "rule", "ground_truth": gold_choice},
                "extra_info": {"split": split, "index": idx, "question": question},
                "raw_question": prompt,
            }
            return data

        return process_fn

    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)
    hdfs_dir = args.hdfs_dir

    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
