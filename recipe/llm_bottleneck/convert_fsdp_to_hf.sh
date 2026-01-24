python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir /projectnb/noc-lab/ylchen/llm_bottleneck/checkpoints/rl-M-m/bs64-qwen3-4b-0.6b-Lout8192-2048-deepscaler/global_step_150/translator \
    --target_dir /projectnb/noc-lab/ylchen/llm_bottleneck/checkpoints/rl-M-m/bs64-qwen3-4b-0.6b-Lout8192-2048-deepscaler/global_step_150/hf_translator