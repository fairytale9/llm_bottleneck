# When Less is Enough: Efficient Inference via Collaborative Reasoning

## Overview

This repository implements **DUET** (**D**ual-model **E**fficient **T**wo-stage inference), a collaborative inference framework in which a capable model **M** and a lightweight model **m** work together to solve a task. DUET decomposes inference into two stages: **M** produces a reasoning signal, and **m** interprets this signal to generate the final answer — keeping reasoning-intensive computation on the capable model while delegating non–reasoning-intensive work to the lightweight model.

- **Joint RL training**: M and m are co-trained end-to-end with shared reward signals
- **Marginal utility length penalty**: Penalizes M's verbosity relative to m's standalone performance — rewarding compression only when it provides benefit

This codebase extends the [verl](https://github.com/volcengine/verl) (Volcano Engine RL for LLMs) framework.

---

## Installation

Follow the official verl installation guide: https://github.com/volcengine/verl

---

## Data Preparation

Preprocessing scripts are provided for all supported datasets:

```bash
python examples/data_preprocess/<dataset>.py
```

**Supported datasets:**

| Dataset | Script | Output path |
|---------|--------|-------------|
| DeepScaleR | `deepscaler.py` | `~/data/deepscaler/` |
| DeepMath | `deepmath.py` | `~/data/deepmath/` |
| DAPO 17k | `dapo17k.py` | `~/data/dapo17k/` |
| LIMO | `limo.py` | `~/data/limo/` |
| MATH | `math_dataset.py` | `~/data/math/` |
| AIME 2024 | `aime2024.py` | `~/data/aime2024/` |
| AIME 2025 | `aime2025.py` | `~/data/aime2025/` |
| AMC 23 | `amc23.py` | `~/data/amc23/` |
| GPQA Diamond | `gpqa_diamond.py` | `~/data/gpqa_diamond/` |

Each script outputs `train.parquet` and/or `test.parquet` under the corresponding directory.

---

## Training

### 1. Edit the training script

Open `recipe/llm_bottleneck/train.sh` and set your model paths and data paths:

```bash
# Models
M_model_path="Qwen/Qwen3-4B"        # Large reasoning model (M)
m_model_path="Qwen/Qwen3-0.6B"      # Small translator model (m)

# Data
train_files="['$deepscaler_train_path']"
test_files="['$math_test_path', '$aime_2024_test_path']"
```

### 2. Key configuration variables

| Variable | Default | Description |
|----------|---------|-------------|
| `M_model_path` | `Qwen/Qwen3-8B` | Path or HuggingFace ID for model M |
| `m_model_path` | `Qwen/Qwen3-0.6B` | Path or HuggingFace ID for model m |
| `train_files` | deepscaler | Training data parquet file(s) |
| `M_prompt_length` | `512` | Max prompt tokens for M |
| `M_response_length` | `8192` | Max response tokens for M |
| `m_response_length` | `1024` | Max response tokens for m |
| `length_bound` | `1000` | Target length bound B in penalty term L/B |
| `length_penalty_schedule` | `adaptive` | `"adaptive"` (auto-tune λ) or `"constant"` |
| `length_penalty_lambda` | `0.0` | Fixed λ (constant) or initial λ (adaptive) |
| `length_penalty_eta` | `0.01` | Learning rate for adaptive λ updates |
| `use_marginal_utility` | `True` | If `True`, reward = r(M,m) − r(m); if `False`, reward = r(M,m) |
| `train_batch_size` | `64` | Global training batch size |
| `trainer_n_gpus_per_node` | `4` | Number of GPUs per node |


### 3. Run training

```bash
bash recipe/llm_bottleneck/train.sh
```

---

## Evaluation

### 1. Edit the eval script

Open `recipe/llm_bottleneck/eval.sh` and set your checkpoint path and desired model configuration. The eval script runs in `val_only=True` mode — no training occurs.

### 2. Run evaluation

```bash
bash recipe/llm_bottleneck/eval.sh
```

**Evaluation benchmarks:**
- MATH (test split)
- AIME 2024
- AIME 2025
- AMC 23
- GPQA Diamond
