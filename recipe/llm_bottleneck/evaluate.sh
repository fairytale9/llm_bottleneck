data_path=$HOME/data/m_math/qwen-0.6b-with-prompt.parquet

python3 -m verl.trainer.main_eval \
    data.path=$data_path \
    data.response_key=m_responses \