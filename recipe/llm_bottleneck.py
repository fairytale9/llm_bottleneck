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
LLM Bottleneck Trainer with Ray-based single controller.
This trainer extends RayPPOTrainer to include our custom TranslatorWorker for policy updates.
"""

import os
import uuid
import json
from collections import defaultdict
from copy import deepcopy
from pprint import pprint

import numpy as np
import torch
from tqdm import tqdm

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    reduce_metrics,
)
from verl.trainer.ppo.ray_trainer import (
    AdvantageEstimator,
    RayPPOTrainer,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.utils.profiler import marked_timer
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.checkpoint.checkpoint_manager import should_save_ckpt_esi
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
import ray


class RayLLMBottleneckTrainer(RayPPOTrainer):
    """
    LLM Bottleneck Trainer that extends RayPPOTrainer to include TranslatorWorker.
    
    This trainer adds support for our custom TranslatorWorker which can:
    - Compute log probabilities of output sequences given input sequences
    - Perform supervised fine-tuning (SFT) to update the policy model
    - Integrate seamlessly with the existing VERL distributed training infrastructure
    """

    def __init__(self, *args, **kwargs):
        """Initialize the LLM Bottleneck Trainer."""
        super().__init__(*args, **kwargs)
        
        # Initialize async rollout mode attribute
        self.async_rollout_mode = False
        if hasattr(self.config.actor_rollout_ref.rollout, 'mode'):
            self.async_rollout_mode = self.config.actor_rollout_ref.rollout.mode == "async"

    def fit(self):
        """
        The training loop of PPO with LLM Bottleneck capabilities.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )

                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # add uid to batch
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                gen_batch = self._get_gen_batch(batch)

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        else:
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        if self.reward_fn is None:
                            raise ValueError("A reward_fn is required for REMAX advantage estimation.")

                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            if not self.async_rollout_mode:
                                gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                            else:
                                gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)



                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(data=batch, reward_fn=self.reward_fn)
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    # recompute old_log_probs
                    with marked_timer("old_log_prob", timing_raw, color="blue"):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                        if "rollout_log_probs" in batch.batch.keys():
                            # TODO: we may want to add diff of probs too.
                            from verl.utils.debug.metrics import calculate_debug_metrics

                            metrics.update(calculate_debug_metrics(batch))

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer("ref", timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # compute advantages, executed on the driver process

                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # === LLM Bottleneck: Update Translator Worker ===
                    # This is where we can add custom logic for our TranslatorWorker
                    # For example, we could call the translator worker to perform policy updates
                    # or compute additional metrics
                    if hasattr(self, 'translator_wg') and self.translator_wg is not None:
                        with marked_timer("translator_update", timing_raw, color="orange"):
                            try:
                                # === Step 1: Compute Log Probabilities ===
                                # Compute log probabilities using our translator worker
                                translator_logp = self.translator_wg.compute_log_prob(batch)
                                
                                # Store the log probabilities in the batch
                                if "log_probs" in translator_logp.batch:
                                    batch.batch["m_logp"] = translator_logp.batch["log_probs"]
                                    print(f"✓ TranslatorWorker computed log probabilities, shape: {batch.batch['m_logp'].shape}")
                                else:
                                    print("⚠ TranslatorWorker log_probs not found in output")
                                    # Create a placeholder tensor if computation fails
                                    batch_size = len(batch.batch["input_ids"])
                                    seq_len = batch.batch["input_ids"].shape[1]
                                    batch.batch["m_logp"] = torch.zeros(batch_size, seq_len, device=batch.batch["input_ids"].device)
                                
                                # === Step 2: Update Translator Worker ===
                                # Call the translator worker's update function to perform policy updates
                                if hasattr(self.translator_wg, 'update_policy'):
                                    update_result = self.translator_wg.update_policy(batch)
                                    print(f"✓ TranslatorWorker policy update completed")
                                    
                                    # Add any metrics from the update if available
                                    if hasattr(update_result, 'meta_info') and 'metrics' in update_result.meta_info:
                                        translator_metrics = update_result.meta_info['metrics']
                                        metrics.update({f"translator/{k}": v for k, v in translator_metrics.items()})
                                else:
                                    print("ℹ TranslatorWorker update_policy method not available")
                                
                                # === Step 3: Log Statistics ===
                                # Log some statistics about the computed log probabilities
                                if "m_logp" in batch.batch:
                                    m_logp = batch.batch["m_logp"]
                                    avg_logp = m_logp.mean().item()
                                    std_logp = m_logp.std().item()
                                    
                                    # Add metrics for monitoring
                                    metrics.update({
                                        "translator/avg_logp": avg_logp,
                                        "translator/std_logp": std_logp,
                                        "translator/logp_shape": list(m_logp.shape)
                                    })
                                    
                                    print(f"✓ TranslatorWorker log probabilities - Avg: {avg_logp:.4f}, Std: {std_logp:.4f}")
                                
                                # You can add more custom logic here, such as:
                                # - Computing additional metrics
                                # - Performing additional policy updates based on log probabilities
                                
                            except Exception as e:
                                print(f"Warning: Translator worker update failed: {e}")
                                # Create a placeholder tensor if computation fails
                                try:
                                    batch_size = len(batch.batch["input_ids"])
                                    seq_len = batch.batch["input_ids"].shape[1]
                                    batch.batch["m_logp"] = torch.zeros(batch_size, seq_len, device=batch.batch["input_ids"].device)
                                except:
                                    pass

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            sample_gts = [
                                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None)
                                for item in batch
                            ]

                            if "request_id" in batch.non_tensor_batch:
                                reward_extra_infos_dict.setdefault(
                                    "request_id",
                                    batch.non_tensor_batch["request_id"].tolist(),
                                )

                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                gts=sample_gts,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                esi_close_to_expiration = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
                # Check if the conditions for saving a checkpoint are met.
                # The conditions include a mandatory condition (1) and
                # one of the following optional conditions (2/3/4):
                # 1. The save frequency is set to a positive value.
                # 2. It's the last training step.
                # 3. The current step number is a multiple of the save frequency.
                # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
                ):
                    if esi_close_to_expiration:
                        print("Force saving checkpoint: ESI instance expiration approaching.")
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        """Extract generation batch from the main batch."""
        reward_model_keys = set({"data_source", "reward_model", "extra_info", "uid"}) & batch.non_tensor_batch.keys()

        # pop those keys for generation
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_model_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        # For agent loop, we need reward model keys to compute score.
        if hasattr(self, 'async_rollout_mode') and self.async_rollout_mode:
            gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        return gen_batch

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.
        
        This method extends the parent class to also initialize our TranslatorWorker.
        """
        # Call parent method to initialize standard workers
        super().init_workers()
        
        # Initialize our custom TranslatorWorker if it exists in the role mapping
        if hasattr(self, 'role_worker_mapping') and 'Translator' in self.role_worker_mapping:
            try:
                # Get the resource pool for the translator worker
                from verl.trainer.ppo.ray_trainer import Role
                translator_role = getattr(Role, 'Translator', None)
                
                if translator_role is not None:
                    resource_pool = self.resource_pool_manager.get_resource_pool(translator_role)
                    
                    # Create the translator worker group
                    from verl.single_controller.ray import RayClassWithInitArgs
                    translator_cls = RayClassWithInitArgs(
                        cls=self.role_worker_mapping[translator_role],
                        config=self.config.get('translator', {}),  # Use translator config if available
                        role="translator"
                    )
                    
                    # Add to resource pool mapping
                    if resource_pool not in self.resource_pool_to_cls:
                        self.resource_pool_to_cls[resource_pool] = {}
                    self.resource_pool_to_cls[resource_pool]["translator"] = translator_cls
                    
                    # Create and initialize the translator worker group
                    from verl.single_controller.ray.base import create_colocated_worker_cls
                    worker_dict_cls = create_colocated_worker_cls(
                        class_dict=self.resource_pool_to_cls[resource_pool]
                    )
                    
                    wg_kwargs = {}
                    if hasattr(self.config.trainer, 'ray_wait_register_center_timeout'):
                        wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
                    if hasattr(self.config.global_profiler, 'steps') and self.config.global_profiler.steps is not None:
                        wg_kwargs["profile_steps"] = self.config.global_profiler.steps
                    wg_kwargs["device_name"] = getattr(self, 'device_name', 'cuda')
                    
                    translator_wg = self.ray_worker_group_cls(
                        resource_pool=resource_pool,
                        ray_cls_with_init=worker_dict_cls,
                        **wg_kwargs
                    )
                    
                    # Spawn the translator worker group
                    spawn_wg = translator_wg.spawn(prefix_set=["translator"])
                    self.translator_wg = spawn_wg.get("translator")
                    
                    if self.translator_wg:
                        self.translator_wg.init_model()
                        print("✓ TranslatorWorker initialized successfully")
                    else:
                        print("⚠ TranslatorWorker initialization failed")
                        
            except Exception as e:
                print(f"Warning: Failed to initialize TranslatorWorker: {e}")
                self.translator_wg = None
        else:
            self.translator_wg = None
            print("ℹ TranslatorWorker not configured in role mapping")

    def _save_checkpoint(self):
        """Save checkpoint including translator worker state."""
        # Call parent method to save standard checkpoints
        super()._save_checkpoint()
        
        # Save translator worker checkpoint if it exists
        if hasattr(self, 'translator_wg') and self.translator_wg is not None:
            try:
                local_global_step_folder = os.path.join(
                    self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
                )
                translator_local_path = os.path.join(local_global_step_folder, "translator")
                
                translator_remote_path = (
                    None
                    if self.config.trainer.default_hdfs_dir is None
                    else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "translator")
                )
                
                # Save translator checkpoint
                self.translator_wg.save_checkpoint(
                    translator_local_path, 
                    translator_remote_path, 
                    self.global_steps
                )
                print(f"✓ TranslatorWorker checkpoint saved to {translator_local_path}")
                
            except Exception as e:
                print(f"Warning: Failed to save TranslatorWorker checkpoint: {e}")

    def _load_checkpoint(self):
        """Load checkpoint including translator worker state."""
        # Call parent method to load standard checkpoints
        super()._load_checkpoint()
        
        # Load translator worker checkpoint if it exists
        if hasattr(self, 'translator_wg') and self.translator_wg is not None:
            try:
                checkpoint_folder = self.config.trainer.default_local_dir
                if not os.path.isabs(checkpoint_folder):
                    working_dir = os.getcwd()
                    checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
                
                # Find the latest checkpoint
                from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
                global_step_folder = find_latest_ckpt_path(checkpoint_folder)
                
                if global_step_folder:
                    translator_path = os.path.join(global_step_folder, "translator")
                    if os.path.exists(translator_path):
                        self.translator_wg.load_checkpoint(
                            translator_path, 
                            del_local_after_load=self.config.trainer.get('del_local_ckpt_after_load', False)
                        )
                        print(f"✓ TranslatorWorker checkpoint loaded from {translator_path}")
                    else:
                        print("ℹ No TranslatorWorker checkpoint found, starting from scratch")
                        
            except Exception as e:
                print(f"Warning: Failed to load TranslatorWorker checkpoint: {e}")
