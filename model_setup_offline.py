import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- CONFIGURATION ---

MODEL_ID = "google/gemma-2-2b-it"



print(f"\n---  DOWNLOADING MODEL: {MODEL_ID} ---")
# This downloads the model to ~/.cache/huggingface/hub
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, 
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)
print(f"✅ Model downloaded to cache.")

print("\n--- SETUP COMPLETE. YOU ARE READY TO SUBMIT. ---")
