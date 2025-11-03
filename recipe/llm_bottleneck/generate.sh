python3 -m verl.trainer.main_generation \
    trainer.n_gpus_per_node=4 \
    data.batch_size=128 \
    data.path=$HOME/data/M_math/deepseek-distill-1.5b.parquet \
    data.prompt_key=prompt \
    data.n_samples=1 \
    data.output_path=$HOME/data/m_math/qwen-0.6b-with-prompt.parquet \
    model.path=Qwen/Qwen3-0.6B \
    rollout.temperature=0 \
    rollout.top_k=-1 \
    rollout.top_p=1 \
    rollout.prompt_length=4608 \
    rollout.response_length=512