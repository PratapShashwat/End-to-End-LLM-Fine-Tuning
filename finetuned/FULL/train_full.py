import os
import torch
import glob
import time
import pyarrow as pa
import pyarrow.ipc
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    TrainerCallback
)
from trl import SFTTrainer

# ================= CONFIGURATION =================
MODEL_ID = "google/gemma-2-2b-it"
DATASET_PATH = "../data/il_tur_data"
OUTPUT_DIR = "../finetuned/full_fsdp"

# FSDP SETTINGS
MAX_SEQ_LENGTH = 1024   
NUM_EPOCHS = 1          
BATCH_SIZE = 1          # 1 per GPU * 4 GPUs = Effective Batch 4
GRAD_ACCUMULATION = 4   # 4 * 4 = Effective Batch 16 (Matches previous runs)
LEARNING_RATE = 2e-5    
# =================================================

class ResourceMonitorCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        # Only print on the main GPU (Rank 0) to avoid messy logs
        if args.process_index == 0:
            print("?? Distributed Training Started...")

    def on_train_end(self, args, state, control, **kwargs):
        if args.process_index == 0:
            total_time = time.time() - self.start_time
            print(f"\n\n?? --- FSDP BENCHMARK RESULTS ---")
            print(f"??  Total Training Time: {total_time/60:.2f} minutes")
            print(f"--------------------------------------\n")

def load_custom_dataset():
    # Only Rank 0 prints logs
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(f"--- 1. LOADING TRAINING DATA SHARDS ---")
    
    arrow_files = glob.glob(f"{DATASET_PATH}/**/*train*/*.arrow", recursive=True)
    if not arrow_files:
        arrow_files = glob.glob(f"{DATASET_PATH}/**/*.arrow", recursive=True)

    full_data = []
    for file_path in arrow_files:
        try:
            with pa.memory_map(file_path, 'r') as source:
                try: reader = pa.ipc.open_stream(source)
                except:
                    source.seek(0)
                    reader = pa.ipc.open_file(source)
                full_data.extend(reader.read_all().to_pylist())
        except Exception:
            continue
    return full_data

def format_instruction(sample):
    doc = " ".join(sample['document']) if isinstance(sample['document'], list) else sample['document']
    summary = " ".join(sample['summary']) if isinstance(sample['summary'], list) else sample['summary']
    text = f"<start_of_turn>user\nSummarize this legal document:\n\n{doc[:8000]}<end_of_turn>\n<start_of_turn>model\n{summary}<end_of_turn>"
    return {"text": text}

def main():
    raw_data = load_custom_dataset()
    formatted_data = [format_instruction(item) for item in raw_data]
    
    full_dataset = Dataset.from_list(formatted_data)
    dataset_dict = full_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset_dict['train']
    eval_dataset = dataset_dict['test']

    # --- 2. LOAD MODEL ---
    # FSDP requires the model to be loaded on CPU first, then it moves shards to GPU
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print("--- ?? Loading Model for FSDP (FP32) ---")
        
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32, 
        # device_map="auto",  <-- REMOVE THIS! FSDP handles device placement manually
        local_files_only=True,
        use_cache=False
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    tokenizer.padding_side = 'right'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- 3. TRAINER SETUP WITH FSDP ---
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE, 
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION,
        optim="adamw_torch",
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE, 
        weight_decay=0.01,
        
        # FSDP CONFIGURATION
        fsdp="full_shard auto_wrap", # Shard params, grads, and optimizer states
        fsdp_config={
            "min_num_params": 1000, # Wrap layers larger than this
        },
        
        # Keep FP32 (Safe Mode)
        fp16=False,                 
        bf16=False,                 
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        
        logging_steps=10,
        save_strategy="no", 
        evaluation_strategy="epoch",
        report_to="none",
        ddp_find_unused_parameters=False
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        tokenizer=tokenizer,
        args=training_args,
        packing=False,
        callbacks=[ResourceMonitorCallback()]
    )

    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print("--- ?? STARTING DISTRIBUTED FULL FINE-TUNING ---")
        
    trainer.train()

    # Save only on main process
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(f"--- ? SAVING MODEL TO {OUTPUT_DIR} ---")
        trainer.save_model(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()