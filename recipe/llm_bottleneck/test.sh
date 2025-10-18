set -x

#translator.ppo_mini_batch_size=256 \
#translator.ppo_micro_batch_size_per_gpu=32 \

# if use lora
#actor_rollout_ref.rollout.load_format=safetensors \
#actor_rollout_ref.rollout.layered_summon=True \

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/math/train.parquet \
    data.val_files=$HOME/data/math/test.parquet \
    data.train_batch_size=128 \
    data.max_prompt_length=512 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" \
    actor_rollout_ref.model.lora_rank=0 \
    actor_rollout_ref.model.lora_alpha=16 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    +translator.enable=True \
    translator.model.path="Qwen/Qwen3-0.6B"\
    translator.model.tokenizer_path="Qwen/Qwen3-0.6B" \
    translator.ppo_epochs=1 \
    translator.optim.lr=1e-4 \
    translator.model.use_remove_padding=False \
    translator.model.enable_gradient_checkpointing=False \
    translator.ppo_micro_batch_size_per_gpu=4 \
    translator.model.fsdp_config.param_offload=False \
    translator.model.fsdp_config.optimizer_offload=False \
    reward_model.reward_manager='custom' \
    algorithm.use_kl_in_reward=False \
    trainer.val_before_train=False \
    trainer.val_only=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='llm-bottleneck' \
    trainer.experiment_name='bs_128_length_4096_full_training_distill_1.5b_no_prompt_to_m' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=200 \
    trainer.test_freq=0 \
    trainer.total_epochs=15 $@