set -x

deepscaler_train_path=$HOME/data/deepscaler/train.parquet
dapo_train_path=$HOME/data/dapo17k/train.parquet
limo_train_path=$HOME/data/limo/train.parquet
# specify configs
math_test_path=$HOME/data/math/test.parquet
aime_2024_test_path=$HOME/data/aime2024/test.parquet
aime_2025_test_path=$HOME/data/aime2025/test.parquet
amc23_test_path=$HOME/data/amc23/test.parquet
gpqa_diamond_test_path=$HOME/data/gpqa_diamond/test.parquet
gpqa_main_test_path=$HOME/data/gpqa_main/test.parquet

test_files="['$math_test_path', '$aime_2024_test_path', '$aime_2025_test_path', '$amc23_test_path', '$gpqa_diamond_test_path', '$gpqa_main_test_path']"

M_model_path="/projectnb/noc-lab/ylchen/llm_bottleneck/checkpoints/rl-M-m/bs-64-qwen3-4b-0.6b-Lout4096-1024-math/global_step_50/hf_actor"
M_prompt_length=1024
M_response_length=16384

m_enable=True
train_m=False
m_model_path="/projectnb/noc-lab/ylchen/llm_bottleneck/checkpoints/rl-M-m/bs-64-qwen3-4b-0.6b-Lout4096-1024-math/global_step_50/hf_translator"
m_response_length=2048

project_name="eval-for-paper"
experiment_name="Mm-Lout-${M_response_length}-${m_response_length}-trained-on-math-ckp50"


if [[ "${m_enable,,}" == "true" ]]; then
    translator_n_gpus_per_node=2
    trainer_n_gpus_per_node=2
else
    translator_n_gpus_per_node=0
    trainer_n_gpus_per_node=4
fi

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/math/train.parquet \
    data.val_files="$test_files" \
    data.train_batch_size=128 \
    data.val_batch_size=128 \
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
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.max_num_batched_tokens=20000 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    +translator.enable=$m_enable \
    +translator.train=$train_m \
    translator.model.path=$m_model_path \
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
    translator.rollout.gpu_memory_utilization=0.6 \
    translator.rollout.n=1 \
    translator.rollout.max_num_batched_tokens=20000 \
    reward_model.reward_manager='custom' \
    algorithm.use_kl_in_reward=False \
    trainer.val_before_train=True \
    trainer.val_only=True \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=$trainer_n_gpus_per_node \
    trainer.translator_n_gpus_per_node=$translator_n_gpus_per_node  \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=10 \
    trainer.total_epochs=3 $@