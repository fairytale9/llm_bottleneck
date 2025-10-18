# LLM Bottleneck Inference Guide

This guide explains how to run inference with the trained LLM bottleneck model.

## Model Checkpoint Structure

The model checkpoints are stored in FSDP (Fully Sharded Data Parallel) format, which needs to be converted to HuggingFace format for inference with vLLM.

```
checkpoints/llm-bottleneck/bs_128_length_4096_full_training_distill_1.5b/global_step_400/
├── actor/                          # FSDP checkpoint directory
│   ├── huggingface/                # Tokenizer and config files
│   │   ├── config.json
│   │   ├── tokenizer.json
│   │   └── ...
│   ├── model_world_size_4_rank_*.pt  # Sharded model weights
│   ├── optim_world_size_4_rank_*.pt  # Optimizer states
│   └── fsdp_config.json
├── translator/                     # Translator model (if used)
└── data.pt                        # Dataloader state
```

## Quick Start

### Option 1: Convert Checkpoint First (Recommended)

1. **Convert the FSDP checkpoint to HuggingFace format:**
   ```bash
   cd /projectnb/noc-lab/ylchen/llm_bottleneck
   python -m verl.model_merger merge \
       --backend fsdp \
       --local_dir checkpoints/llm-bottleneck/bs_128_length_4096_full_training_distill_1.5b/global_step_400/actor \
       --target_dir checkpoints/llm-bottleneck/bs_128_length_4096_full_training_distill_1.5b/global_step_400/actor_hf
   ```

2. **Run the inference notebook:**
   ```bash
   jupyter notebook recipe/llm_bottleneck/inference.ipynb
   ```

### Option 2: Use the Conversion Script

1. **Run the provided conversion script:**
   ```bash
   cd /projectnb/noc-lab/ylchen/llm_bottleneck
   python convert_checkpoint.py
   ```

2. **Or use the shell script:**
   ```bash
   ./run_conversion.sh
   ```

## Understanding the Checkpoint Loading Process

### How VERL Saves Checkpoints

The VERL training framework saves checkpoints in FSDP format, which:
- Shards model parameters across multiple GPUs
- Saves optimizer states separately
- Includes configuration files for reconstruction

### Why Conversion is Needed

- **vLLM** expects HuggingFace format models
- **FSDP checkpoints** are distributed and need reconstruction
- **Tokenizers** are saved separately in the `huggingface/` subdirectory

### Conversion Process

The conversion process:
1. Loads all sharded model weights from all ranks
2. Reconstructs the full model state dictionary
3. Saves it in HuggingFace format with proper file structure
4. Preserves tokenizer and configuration files

## Troubleshooting

### Common Issues

1. **"Model not found" error:**
   - Ensure the checkpoint path is correct
   - Check if the conversion completed successfully

2. **"Tokenizer not found" error:**
   - The tokenizer files are in the `huggingface/` subdirectory
   - Make sure the path points to the correct location

3. **"CUDA out of memory" error:**
   - The model is large (1.5B parameters)
   - Consider using CPU inference or reducing batch size

### Alternative Inference Methods

If vLLM doesn't work, you can try:

1. **Direct HuggingFace loading:**
   ```python
   from transformers import AutoModelForCausalLM, AutoTokenizer
   
   model = AutoModelForCausalLM.from_pretrained(converted_model_path)
   tokenizer = AutoTokenizer.from_pretrained(converted_model_path)
   ```

2. **Using the VERL framework directly:**
   - Requires distributed setup
   - More complex but handles FSDP natively

## Model Information

- **Architecture:** Qwen2ForCausalLM
- **Size:** 1.5B parameters
- **Training:** Full training with distillation
- **Checkpoint:** global_step_400 (latest)

## File Structure After Conversion

```
checkpoints/llm-bottleneck/bs_128_length_4096_full_training_distill_1.5b/global_step_400/
├── actor_hf/                       # Converted HuggingFace model
│   ├── config.json
│   ├── pytorch_model.bin           # Full model weights
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── ...
└── actor/                          # Original FSDP checkpoint
    └── ...
```
