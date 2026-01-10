python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir /projectnb/noc-lab/ylchen/llm_bottleneck/checkpoints/llm-bottleneck/bs_128_length_4096_abs_penalty/global_step_200/actor \
    --target_dir /projectnb/noc-lab/ylchen/llm_bottleneck/checkpoints/llm-bottleneck/bs_128_length_4096_abs_penalty/global_step_200/hf_actor
