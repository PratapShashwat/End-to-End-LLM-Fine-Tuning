import os
from datasets import load_dataset

# --- CONFIG ---
DATASET_ID = "Exploration-Lab/IL-TUR"
CONFIG_NAME = "summ"   # <--- CHANGED to lowercase (Critical Fix)
SAVE_PATH = "./il_tur_data"

# PASTE YOUR TOKEN HERE
MY_TOKEN = "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" 

print(f"--- DOWNLOADING {DATASET_ID} (Config: {CONFIG_NAME}) ---")

try:
    # 1. Login & Download
    # Removed 'trust_remote_code' as requested by the error
    dataset = load_dataset(DATASET_ID, CONFIG_NAME, split=None, token=MY_TOKEN)
    
    print(f"✅ Downloaded successfully!")
    print(f"   Splits found: {list(dataset.keys())}")
    
    # 2. Save to a folder
    dataset.save_to_disk(SAVE_PATH)
    print(f"✅ Saved to folder: {SAVE_PATH}")
    print("   Now zip this folder and upload it!")

except Exception as e:
    print(f"❌ Error: {e}")