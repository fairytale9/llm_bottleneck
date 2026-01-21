python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir /projectnb/noc-lab/ylchen/llm_bottleneck/checkpoints/rl-M-m/qwen3-4b-0.6b-Lout8096-2048-deepscaler/global_step_50/translator \
    --target_dir /projectnb/noc-lab/ylchen/llm_bottleneck/checkpoints/rl-M-m/qwen3-4b-0.6b-Lout8096-2048-deepscaler/global_step_50/hf_translator