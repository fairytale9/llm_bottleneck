python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir /projectnb/noc-lab/ylchen/llm_bottleneck/checkpoints/rl-M-m/qwen3-4b-0.6b-Lout4096-2048/global_step_100/translator \
    --target_dir /projectnb/noc-lab/ylchen/llm_bottleneck/checkpoints/rl-M-m/qwen3-4b-0.6b-Lout4096-2048/global_step_100/hf_translator