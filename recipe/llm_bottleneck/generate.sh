#+data.apply_chat_template_kwargs.enable_thinking=False \

python3 -m verl.trainer.main_generation \
    trainer.n_gpus_per_node=4 \
    data.batch_size=128 \
    data.path=$HOME/data/m_math/distill-qwen-1.5b-length-500-step200.parquet \
    data.prompt_key=prompt \
    data.n_samples=1 \
    data.output_path=$HOME/data/m_math/distill-qwen-1.5b-length-500-step200.parquet \
    model.path=Qwen/Qwen2.5-1.5B-Instruct \
    rollout.temperature=0.6 \
    rollout.top_k=-1 \
    rollout.top_p=1 \
    rollout.prompt_length=4096 \
    rollout.response_length=1024 \
    +M_reasoning=True \
    +output_key=m_responses