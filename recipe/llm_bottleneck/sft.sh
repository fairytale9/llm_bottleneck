# Tested with 2 & 4 GPUs

set -x

if [ "$#" -lt 2 ]; then
    echo "Usage: run_gemma_2b.sh <nproc_per_node> <save_path> [other_configs...]"
    exit 1
fi

nproc_per_node=$1
save_path=$2

# Shift the arguments so $@ refers to the rest
shift 2

#+data.apply_chat_template_kwargs.enable_thinking=False \

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$HOME/data/M_math/train-distill-qwen-1.5b.parquet \
    data.val_files=$HOME/data/math/test.parquet \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    data.prompt_dict_keys=['question'] \
    data.response_dict_keys=['answer'] \
    data.max_length=8000 \
    data.micro_batch_size_per_gpu=4 \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    model.partial_pretrain=Qwen/Qwen3-0.6B \
    optim.lr=1e-5 \
    trainer.default_local_dir=$save_path \
    trainer.project_name=Math-sft \
    trainer.experiment_name=Qwen3-0.6B-sft-M-dataset \
    trainer.total_epochs=2 \
    trainer.logger='["console","wandb"]' $@