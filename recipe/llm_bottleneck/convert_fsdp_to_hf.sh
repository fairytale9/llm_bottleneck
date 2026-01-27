python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir /projectnb/noc-lab/ylchen/llm_bottleneck/checkpoints/rl-M-m/bs-64-qwen3-4b-0.6b-Lout4096-1024-math/global_step_50/translator \
    --target_dir /projectnb/noc-lab/ylchen/llm_bottleneck/checkpoints/rl-M-m/bs-64-qwen3-4b-0.6b-Lout4096-1024-math/global_step_50/hf_translator