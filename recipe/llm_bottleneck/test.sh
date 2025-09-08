export VLLM_ATTENTION_BACKEND=XFORMERS

set -x

gsm8k_train_path=$HOME/data/gsm8k/train.parquet
gsm8k_test_path=$HOME/data/gsm8k/test.parquet
#math_train_path=$HOME/data/math/train.parquet
#math_test_path=$HOME/data/math/test.parquet

#translator.ppo_mini_batch_size=256 \
#translator.ppo_micro_batch_size_per_gpu=32 \

#train_files="['$gsm8k_train_path', '$math_train_path']"
#test_files="['$gsm8k_test_path', '$math_test_path']"

train_files="['$gsm8k_train_path']"
test_files="['$gsm8k_test_path']"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=256 \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=/projectnb/noc-lab/ylchen/model/qwen_math_1.5b \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
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
    reward_model.reward_manager='naive' \
    translator.enable=true \
    translator.model.path=/projectnb/noc-lab/ylchen/model/qwen_0.6b \
    translator.ppo_epochs=1 \
    translator.optim.lr=1e-5 \
    translator.model.use_remove_padding=False \
    translator.model.enable_gradient_checkpointing=False \
    translator.ppo_micro_batch_size_per_gpu=4 \
    translator.model.fsdp_config.param_offload=False \
    translator.model.fsdp_config.optimizer_offload=False \
    algorithm.use_kl_in_reward=False \
    trainer.val_before_train=False \
    trainer.val_only=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='llm-bottleneck' \
    trainer.experiment_name='debug' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=0 \
    trainer.test_freq=10 \
    trainer.total_epochs=15 $@