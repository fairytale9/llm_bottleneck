python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir /projectnb/noc-lab/ylchen/llm_bottleneck/checkpoints/rl-M-m/bs64-qwen3-4b-qwen-2.5-0.5b-instruct-Lout8192-1024-deepscaler/global_step_50/translator \
    --target_dir /projectnb/noc-lab/ylchen/llm_bottleneck/checkpoints/rl-M-m/bs64-qwen3-4b-qwen-2.5-0.5b-instruct-Lout8192-1024-deepscaler/global_step_50/hf_translator