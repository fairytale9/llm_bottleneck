from transformers import AutoTokenizer, AutoModelForCausalLM

repo_id = "fairytale-49/M-qwen3-4b-math"
tok = AutoTokenizer.from_pretrained(repo_id, token=True)   # uses HF_TOKEN if set
model = AutoModelForCausalLM.from_pretrained(repo_id, token=True)