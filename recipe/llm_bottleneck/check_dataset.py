import pandas as pd
from pathlib import Path
'''
df = pd.read_parquet("/projectnb/noc-lab/ylchen/data/gpqa_diamond/test.parquet")

# Add raw_question from prompt
# If prompt is already a string column
# df["raw_question"] = df["prompt"].apply(
#     lambda p: p[0]["content"]
# )
print(df.iloc[0]['data_source'])
df["data_source"] = "Idavidrein/gpqa_diamond"

print(df.iloc[0]['data_source'])
# Save back to parquet
df.to_parquet("/projectnb/noc-lab/ylchen/data/gpqa_diamond/test.parquet", index=False)
'''

path = Path("/projectnb/noc-lab/ylchen/data/gpqa_diamond/test.parquet")
df = pd.read_parquet(path)
print(len(df))
# Show a few samples with prompt and raw_question
print(df.iloc[4]['data_source'])
print(df.iloc[4]['prompt'])
print(df.iloc[4]['raw_question'])
print(df.iloc[4]['extra_info'])


