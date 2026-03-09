python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir $fsdp_model_path \
    --target_dir $target_model_path