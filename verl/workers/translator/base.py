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
Base class for a translator
"""

from abc import ABC, abstractmethod

import torch

from verl import DataProto

__all__ = ["BasePPOTranslator"]


class BasePPOTranslator(ABC):
    def __init__(self, config):
        super().__init__()
        self.config = config

    @abstractmethod
    def update_translator(self, data: DataProto):
        """Update the translator"""
        pass

    @abstractmethod
    def compute_log_prob(self, data: DataProto, calculate_entropy=False):
        """Compute log probabilities of translations"""
        pass
