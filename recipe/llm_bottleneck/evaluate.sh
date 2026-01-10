data_path=$HOME/data/m_math/distill-qwen-1.5b-length-500-step200.parquet

python3 -m verl.trainer.main_eval_old \
    data.path=$data_path \
    data.response_key=m_responses \