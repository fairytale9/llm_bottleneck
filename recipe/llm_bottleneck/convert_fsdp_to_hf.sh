python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir /opt/app-root/src/llm_bottleneck/checkpoints/rl-M-m/LoutM3072m2048_v128_onlytrainM_nolengthpenalty/global_step_50/translator \
    --target_dir /opt/app-root/src/llm_bottleneck/checkpoints/rl-M-m/LoutM3072m2048_v128_onlytrainM_nolengthpenalty/global_step_50/hf_translator