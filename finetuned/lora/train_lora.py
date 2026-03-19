import os
import torch
import glob
import pyarrow as pa
import pyarrow.ipc
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    BitsAndBytesConfig
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer

# ================= CONFIGURATION =================
MODEL_ID = "google/gemma-2-2b-it"
DATASET_PATH = "./il_tur_data"
OUTPUT_DIR = "./results/gemma_legal_finetuned"

# MEMORY OPTIMIZATION SETTINGS
MAX_SEQ_LENGTH = 1024   # Reduced from 2048 to fit in 16GB VRAM
NUM_EPOCHS = 1         
BATCH_SIZE = 1         
GRAD_ACCUMULATION = 4  
LEARNING_RATE = 2e-4
# =================================================

def load_custom_dataset():
    print(f"--- 1. LOADING TRAINING DATA SHARDS ---")
    arrow_files = glob.glob(f"{DATASET_PATH}/**/*train*/*.arrow", recursive=True)
    if not arrow_files: 
        arrow_files = glob.glob(f"{DATASET_PATH}/**/*.arrow", recursive=True)
    
    full_data = []
    # Use first 50 files to prevent RAM overflow during loading
    for i, file_path in enumerate(arrow_files): 
        try:
            with pa.memory_map(file_path, 'r') as source:
                try: reader = pa.ipc.open_stream(source)
                except: 
                    source.seek(0)
                    reader = pa.ipc.open_file(source)
                shard = reader.read_all().to_pylist()
                full_data.extend(shard)
        except Exception as e:
            continue
            
    print(f"✅ Total Training Samples Loaded: {len(full_data)}")
    return full_data

def format_instruction(sample):
    doc = " ".join(sample['document']) if isinstance(sample['document'], list) else sample['document']
    summary = " ".join(sample['summary']) if isinstance(sample['summary'], list) else sample['summary']
    # Truncate input text to fit context
    text = f"<start_of_turn>user\nSummarize this legal document:\n\n{doc[:8000]}<end_of_turn>\n<start_of_turn>model\n{summary}<end_of_turn>"
    return {"text": text}

def main():
    # 1. PREPARE DATASET
    raw_data = load_custom_dataset()
    formatted_data = [format_instruction(item) for item in raw_data]
    train_dataset = Dataset.from_list(formatted_data)
    
    # 2. LOAD MODEL (4-bit Quantization)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        local_files_only=True,
        use_cache=False # Disable cache for gradient checkpointing
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    tokenizer.padding_side = 'right'

    # 3. CONFIGURE LORA (Rank 8 for Memory Safety)
    peft_config = LoraConfig(
        r=8,       # Reduced from 16 to 8 to save VRAM
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'] 
    )

    # 4. TRAINER SETUP (With Gradient Checkpointing)
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        logging_steps=5,
        save_steps=50,
        num_train_epochs=NUM_EPOCHS,
        fp16=True,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True, # <--- THE CRITICAL FIX
        report_to="none"
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        args=training_args,
        packing=False,
    )

    print("--- 🚀 STARTING FINE-TUNING (Memory Optimized) ---")
    trainer.train()

    print(f"--- ✅ SAVING MODEL TO {OUTPUT_DIR} ---")
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()

#run results...'train_runtime': 4529.2643, 'train_samples_per_second': 1.552, 'train_steps_per_second': 0.388, 'train_loss': 1.761719117872947, 'epoch': 1.0