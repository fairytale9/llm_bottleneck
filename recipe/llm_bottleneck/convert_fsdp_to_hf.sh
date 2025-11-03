python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir /projectnb/noc-lab/ylchen/llm_bottleneck/checkpoints/llm-bottleneck/bs_128_length_4096_full_training_distill_1.5b_no_prompt_to_m/global_step_400/translator \
    --target_dir /projectnb/noc-lab/ylchen/llm_bottleneck/checkpoints/llm-bottleneck/bs_128_length_4096_full_training_distill_1.5b_no_prompt_to_m/global_step_400/hf_translator
