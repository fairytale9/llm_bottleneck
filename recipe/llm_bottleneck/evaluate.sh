set -x

data_path=$HOME/data/math/test.parquet
save_path=$HOME/data/math/eval.parquet
model_path=/projectnb/noc-lab/ylchen/llm_bottleneck/checkpoints/llm-bottleneck/bs_128_length_4096_full_training_distill_1.5b/global_step_400/actor

python3 -m verl.trainer.main_generation \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=4 \
    data.path=$data_path \
    data.prompt_key=prompt \
    data.n_samples=1 \
    data.batch_size=4 \
    data.output_path=$save_path \
    model.path=$model_path \
    +model.trust_remote_code=True \
    rollout.temperature=1.0 \
    rollout.top_k=50 \
    rollout.top_p=0.7 \
    rollout.prompt_length=512 \
    rollout.response_length=2048 \
    rollout.tensor_model_parallel_size=2 \
    rollout.gpu_memory_utilization=0.8