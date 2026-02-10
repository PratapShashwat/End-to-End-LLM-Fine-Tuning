import os
import json
import torch
import numpy as np
import sys
import glob
import traceback
from transformers import AutoTokenizer, AutoModelForCausalLM
import pyarrow as pa
import pyarrow.ipc

# --- LIBRARY SAFETY ---
try:
    from rouge import Rouge
    rouge_available = True
except ImportError:
    print("⚠️ 'rouge' library not found. Summaries will be generated but not scored.")
    rouge_available = False

# ==========================================
#              CONFIGURATION
# ==========================================
MODEL_ID = "google/gemma-2-2b-it"
DATASET_PATH = "./il_tur_data"
CHECKPOINT_FILE = "./results/ckpt_il_tur.jsonl"
OUTPUT_FILE = "./results/final_il_tur.json"

MAX_INPUT_CHARS = 25000
MAX_NEW_TOKENS = 1024
DO_SAMPLE = False 

def main():
    if not os.path.exists("./results"): os.makedirs("./results")
    
    # --- 1. Load Model ---
    print(f"--- LOADING MODEL: {MODEL_ID} ---")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, torch_dtype=torch.float16, device_map="auto", local_files_only=True
        )
    except Exception as e:
        print(f"❌ Model Error: {e}")
        sys.exit(1)

    # --- 2. Load Dataset (VALIDATION MODE + MERGE) ---
    print(f"--- SEARCHING FOR DATA ---")
    try:
        # PRIORITY: Look for VALIDATION data (Contains answers -> Real Scores)
        arrow_files = glob.glob(f"{DATASET_PATH}/**/*val*/*.arrow", recursive=True)
        split_name = "VALIDATION"
        
        # Fallback to anything else if missing
        if not arrow_files:
            print("⚠️ No Validation set found. Switching to generic search.")
            arrow_files = glob.glob(f"{DATASET_PATH}/**/*.arrow", recursive=True)
            split_name = "GENERIC"

        if not arrow_files:
            raise FileNotFoundError("No .arrow files found anywhere!")

        print(f"👉 Selected Split: {split_name}")
        print(f"👉 Found {len(arrow_files)} file shards. Merging...")
        
        # MERGE ALL SHARDS (This fixes the "Only 100 files" bug)
        dataset = []
        for file_path in arrow_files:
            try:
                with pa.memory_map(file_path, 'r') as source:
                    try: reader = pa.ipc.open_stream(source)
                    except: 
                        source.seek(0)
                        reader = pa.ipc.open_file(source)
                    shard_data = reader.read_all().to_pylist()
                    dataset.extend(shard_data)
            except Exception as e:
                print(f"⚠️ Warning: Skipped broken shard {file_path}")
                continue

        print(f"✅ FINAL DATASET SIZE: {len(dataset)} documents.")

    except Exception as e:
        print(f"❌ Dataset Error: {e}")
        sys.exit(1)

    # --- 3. Setup Scorer ---
    if rouge_available:
        rouge = Rouge()
    all_scores = {'r1': [], 'r2': [], 'rL': []}
    
    # Resume Logic
    processed_ids = set()
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            for line in f:
                try: processed_ids.add(json.loads(line)['id'])
                except: continue
        print(f"Resuming... {len(processed_ids)} already done.")

    # --- 4. Main Loop ---
    results = []
    print(f"--- STARTING GENERATION ---")
    
    for i, record in enumerate(dataset):
        doc_id = record.get('id', str(i))
        if doc_id in processed_ids: continue

        # Data Adapter
        raw_doc = record.get('document') or record.get('text') or []
        raw_sum = record.get('summary') or record.get('abstract') or []
        text = " ".join(raw_doc) if isinstance(raw_doc, list) else str(raw_doc)
        ref = " ".join(raw_sum) if isinstance(raw_sum, list) else str(raw_sum)

        if len(text) < 50: continue

        try:
            # GENERATE
            chat = [{"role": "user", "content": f"Write a detailed summary of this legal document:\n\n{text[:MAX_INPUT_CHARS]}"}]
            prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            
            outputs = model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=DO_SAMPLE, return_dict_in_generate=True
            )
            
            input_len = inputs.input_ids.shape[1]
            gen_seq = outputs.sequences[0] if hasattr(outputs, "sequences") else outputs[0]
            gen_text = tokenizer.decode(gen_seq[input_len:], skip_special_tokens=True).strip()
            
            # SCORING
            r1, r2, rL = 0.0, 0.0, 0.0
            try:
                # Check if we have a valid reference to score against
                if rouge_available and len(ref) > 10 and len(gen_text) > 0:
                    scores = rouge.get_scores([gen_text], [ref])
                    if isinstance(scores, list): s = scores[0]
                    else: s = scores
                    r1, r2, rL = s['rouge-1']['f'], s['rouge-2']['f'], s['rouge-l']['f']
            except: pass

            if r1 > 0:
                all_scores['r1'].append(r1)
                all_scores['r2'].append(r2)
                all_scores['rL'].append(rL)

            # SAVE
            entry = {
                "id": doc_id, 
                "reference": ref, 
                "generated": gen_text, 
                "scores": {"rouge1": r1, "rouge2": r2, "rougeL": rL}
            }
            with open(CHECKPOINT_FILE, 'a') as f: f.write(json.dumps(entry) + "\n")
            results.append(entry)

            if len(results) % 10 == 0:
                avg_r1 = np.mean(all_scores['r1']) if all_scores['r1'] else 0.0
                print(f"Processed {len(results)} files | Avg ROUGE-1: {avg_r1:.4f}")

        except Exception as e:
            print(f"⚠️ Error on file {doc_id}: {e}")
            continue

    # --- 5. Final Report ---
    if results:
        final_stats = {
            "avg_r1": np.mean(all_scores['r1']) if all_scores['r1'] else 0,
            "avg_r2": np.mean(all_scores['r2']) if all_scores['r2'] else 0,
            "avg_rL": np.mean(all_scores['rL']) if all_scores['rL'] else 0
        }
        with open(OUTPUT_FILE, 'w') as f:
            json.dump({"stats": final_stats, "data": results}, f, indent=4)
        print(f"\n✅ JOB COMPLETE. Final Stats: {final_stats}")

if __name__ == "__main__": main()
