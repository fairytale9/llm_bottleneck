import os
import torch
from omegaconf import OmegaConf
from verl import DataProto
from verl.workers.fsdp_workers import ActorRolloutRefWorker, TranslatorWorker
from verl.utils.model import compute_position_id_with_mask

def test_rollout():
    # Environment setup
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "12345"
    
    # Common config
    model_path = "Qwen/Qwen2.5-0.5B-Instruct" 
    
    # Basic configuration string
    config_str = f"""
    rollout_n: 1
    ppo_mini_batch_size: 2
    ppo_micro_batch_size: 1
    
    model:
      path: {model_path}
      trust_remote_code: false
      model_dtype: fp16
      fsdp_config:
        fsdp_size: -1
        param_offload: false
        optimizer_offload: false
        wrap_policy:
            min_num_params: 0
      
    rollout:
      name: vllm
      n: 1
      mode: sync
      tensor_model_parallel_size: 1
      gpu_memory_utilization: 0.4 
      ignore_eos: false
      max_num_batched_tokens: 8192
      disable_log_stats: true
      
    actor:
      strategy: fsdp
      ppo_mini_batch_size: 2
      ppo_micro_batch_size: 1
      fsdp_config:
        fsdp_size: -1
        param_offload: false
        optimizer_offload: false
        
    """
    
    config = OmegaConf.create(config_str)
    
    # Create dummy prompts
    prompts_txt = ["What is 1+1?", "Write a haiku about coding."]
    # We need a tokenizer to tokenize these for the input
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    input_ids_list = []
    attention_mask_list = []
    
    for txt in prompts_txt:
        enc = tokenizer(txt, return_tensors="pt")
        input_ids_list.append(enc.input_ids[0])
        attention_mask_list.append(enc.attention_mask[0])
        
    # Pad them
    max_len = max(len(ids) for ids in input_ids_list)
    padded_input_ids = []
    padded_attention_mask = []
    
    for ids, mask in zip(input_ids_list, attention_mask_list):
        padding = max_len - len(ids)
        padded_input_ids.append(torch.cat([torch.tensor([tokenizer.pad_token_id] * padding), ids]))
        padded_attention_mask.append(torch.cat([torch.tensor([0] * padding), mask]))
        
    input_ids = torch.stack(padded_input_ids).long()
    attention_mask = torch.stack(padded_attention_mask).long()
    position_ids = compute_position_id_with_mask(attention_mask)
    
    prompts = DataProto.from_dict(tensors={
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids
    })
    prompts.meta_info = {
        "max_new_tokens": 16,
        "temperature": 0.7,
        "top_p": 0.9,
    }
    
    # 1. Test ActorRolloutRefWorker
    print("\n" + "="*50)
    print("Testing ActorRolloutRefWorker...")
    print("="*50)
    actor_worker = ActorRolloutRefWorker(config, role="actor_rollout")
    actor_worker.init_model()
    
    output_actor = actor_worker.generate_sequences(prompts)
    print("Actor Output (input_ids shape):", output_actor.batch["input_ids"].shape)
    
    # Decode output
    # output input_ids contains both prompt and response
    for i in range(len(output_actor.batch["input_ids"])):
        seq = output_actor.batch["input_ids"][i]
        text = tokenizer.decode(seq, skip_special_tokens=True)
        print(f"Actor seq {i}: {text}")

    # 2. Test TranslatorWorker
    print("\n" + "="*50)
    print("Testing TranslatorWorker...")
    print("="*50)
    
    # We need to make sure TranslatorWorker uses the config correctly. 
    # It expects 'model', 'rollout', 'checkpoint' etc. in config.
    
    translator_worker = TranslatorWorker(config)
    translator_worker.init_model()
    
    output_translator = translator_worker.generate_sequences(prompts)
    print("Translator Output (input_ids shape):", output_translator.batch["input_ids"].shape)
    
    for i in range(len(output_translator.batch["input_ids"])):
        seq = output_translator.batch["input_ids"][i]
        text = tokenizer.decode(seq, skip_special_tokens=True)
        print(f"Translator seq {i}: {text}")

if __name__ == "__main__":
    test_rollout()

