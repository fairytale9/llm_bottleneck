import pandas as pd
from pathlib import Path
'''
df = pd.read_parquet("/projectnb/noc-lab/ylchen/data/limo/train.parquet")

# Add raw_question from prompt
# If prompt is already a string column
df["raw_question"] = df["prompt"].apply(
    lambda p: p[0]["content"]
)

print(df.iloc[0]['prompt'])
print(df.iloc[0]['raw_question'])
# Save back to parquet
df.to_parquet("/projectnb/noc-lab/ylchen/data/limo/train.parquet", index=False)
'''

path = Path("/projectnb/noc-lab/ylchen/data/aime2024/test.parquet")
df = pd.read_parquet(path)
print(len(df))
# Show a few samples with prompt and raw_question
print(df.iloc[0]['prompt'])
print(df.iloc[0]['raw_question'])
print(df.iloc[0]['extra_info'])



