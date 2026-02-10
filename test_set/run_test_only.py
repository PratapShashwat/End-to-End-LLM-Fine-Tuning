import os
import json
import torch
import glob
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM
import pyarrow as pa
import pyarrow.ipc

# CONFIG
MODEL_ID = "google/gemma-2-2b-it"
DATASET_PATH = "./il_tur_data"
OUTPUT_FILE = "./results/test_set_summaries.json"
MAX_NEW_TOKENS = 1024

def main():
    if not os.path.exists("./results"): os.makedirs("./results")

    # 1. LOAD MODEL
    print(f"--- LOADING MODEL ---")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="auto", local_files_only=True
    )

    # 2. LOAD TEST DATA ONLY
    print(f"--- LOADING TEST DATA ---")
    # We explicitly look for "test" in the filename
    arrow_files = glob.glob(f"{DATASET_PATH}/**/*test*/*.arrow", recursive=True)
    
    if not arrow_files:
        print("❌ No TEST files found. Checking generic...")
        arrow_files = glob.glob(f"{DATASET_PATH}/**/*.arrow", recursive=True)

    dataset = []
    for file_path in arrow_files:
        with pa.memory_map(file_path, 'r') as source:
            try: reader = pa.ipc.open_stream(source)
            except: 
                source.seek(0)
                reader = pa.ipc.open_file(source)
            dataset.extend(reader.read_all().to_pylist())
    
    print(f"✅ Loaded {len(dataset)} Test Files.")

    # 3. GENERATE
    results = []
    print("--- STARTING GENERATION ---")
    
    for i, record in enumerate(dataset):
        # Extract Text
        raw_doc = record.get('document') or record.get('text') or []
        text = " ".join(raw_doc) if isinstance(raw_doc, list) else str(raw_doc)
        doc_id = record.get('id', str(i))

        # Generate
        chat = [{"role": "user", "content": f"Write a detailed summary of this legal document:\n\n{text[:25000]}"}]
        prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        gen_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

        # Save (No Score)
        entry = {"id": doc_id, "generated_summary": gen_text}
        results.append(entry)

        if i % 10 == 0: print(f"Processed {i} files...")

    # 4. SAVE FINAL
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=4)
    print("✅ TEST RUN COMPLETE.")

if __name__ == "__main__": main()
