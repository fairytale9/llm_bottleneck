set -x

export RAY_ENABLE_DASHBOARD=0
########################################################
# specify configs
########################################################

# training and test datasets
dapo_train_path=$HOME/data/dapo17k/train.parquet
math_train_path=$HOME/data/math_strategy/train.parquet
limo_train_path=$HOME/data/limo/train.parquet
deepscaler_train_path=$HOME/data/deepscaler/train.parquet
deepmath_train_path=$HOME/data/deepmath/train.parquet
train_files="['$deepscaler_train_path']"

math_test_path=$HOME/data/math/test.parquet
aime_2024_test_path=$HOME/data/aime2024/test.parquet
test_files="['$math_test_path', '$aime_2024_test_path']"

# M model config
M_model_path="Qwen/Qwen3-4B"
M_prompt_length=512
M_response_length=8192

# m model config
m_enable=True
train_m=True
m_model_path="Qwen/Qwen3-0.6B"
m_response_length=1024

# reward / length penalty config
length_bound=1000                    # B in the penalty term L/B
length_penalty_clip=True             # clip lambda to [0, 1]
length_penalty_schedule="adaptive"   # "adaptive" or "constant"
length_penalty_lambda=0.0            # fixed lambda (constant) or initial lambda (adaptive)
length_penalty_eta=0.01              # learning rate for adaptive lambda updates
use_marginal_utility=True            # True: r_Mm - r_m; False: r_Mm

# experiment
train_batch_size=64
val_batch_size=128
project_name="DUET" # wandb
experiment_name="bs-${train_batch_size}-qwen-4b-0.6b-Lout${M_response_length}-${m_response_length}-trained_on_deepscaler" # wandb


python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=$train_batch_size \
    data.val_batch_size=$val_batch_size \
    data.max_prompt_length=$M_prompt_length \
    data.max_response_length=$M_response_length \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$M_model_path \
    actor_rollout_ref.model.lora_rank=0 \
    actor_rollout_ref.model.lora_alpha=16 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.max_num_batched_tokens=20000 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    +translator.enable=$m_enable \
    +translator.train=$train_m \
    translator.model.path=$m_model_path \
    translator.model.lora_rank=0 \
    translator.model.lora_alpha=16 \
    translator.model.use_remove_padding=True \
    translator.model.enable_gradient_checkpointing=True \
    translator.actor.optim.lr=1e-6 \
    translator.actor.ppo_mini_batch_size=32 \
    translator.actor.ppo_micro_batch_size_per_gpu=4 \
    translator.actor.use_kl_loss=False \
    translator.actor.kl_loss_coef=0.001 \
    translator.actor.kl_loss_type=low_var_kl \
    translator.actor.entropy_coeff=0 \
    translator.actor.fsdp_config.param_offload=False \
    translator.actor.fsdp_config.optimizer_offload=False \
    translator.rollout.log_prob_micro_batch_size_per_gpu=8 \
    translator.rollout.prompt_length=$(($M_prompt_length+$M_response_length)) \
    translator.rollout.response_length=$m_response_length \
    translator.rollout.tensor_model_parallel_size=1 \
    translator.rollout.name=vllm \
    translator.rollout.gpu_memory_utilization=0.25 \
    translator.rollout.free_cache_engine=False \
    translator.rollout.enforce_eager=True \
    translator.rollout.n=4 \
    translator.rollout.max_num_batched_tokens=20000 \
    reward_model.reward_manager='custom' \
    +reward_model.reward_kwargs.length_bound=$length_bound \
    +reward_model.reward_kwargs.length_penalty_clip=$length_penalty_clip \
    +reward_model.reward_kwargs.length_penalty_schedule=$length_penalty_schedule \
    +reward_model.reward_kwargs.length_penalty_lambda=$length_penalty_lambda \
    +reward_model.reward_kwargs.length_penalty_eta=$length_penalty_eta \
    +reward_model.reward_kwargs.use_marginal_utility=$use_marginal_utility \
    algorithm.use_kl_in_reward=True \
    trainer.val_before_train=False \
    trainer.val_only=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=4 \
    trainer.translator_n_gpus_per_node=0 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=10 \
    trainer.total_epochs=3 $@