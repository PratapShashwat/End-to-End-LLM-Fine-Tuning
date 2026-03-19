import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json
import glob
import pyarrow as pa
import pyarrow.ipc

# CONFIG
BASE_MODEL_ID = "google/gemma-2-2b-it"
ADAPTER_PATH = "./results/gemma_legal_finetuned" # <--- Your new model
DATASET_PATH = "./il_tur_data"
OUTPUT_FILE = "./results/finetuned_test_summaries.json"

def main():
    print("--- 1. LOADING BASE MODEL ---")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, local_files_only=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, 
        torch_dtype=torch.float16, 
        device_map="auto", 
        local_files_only=True
    )

    print(f"--- 2. LOADING LORA ADAPTER FROM {ADAPTER_PATH} ---")
    # This merges your fine-tuning training into the base model
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    
    print("--- 3. LOADING TEST DATA ---")
    arrow_files = glob.glob(f"{DATASET_PATH}/**/*test*/*.arrow", recursive=True)
    if not arrow_files: arrow_files = glob.glob(f"{DATASET_PATH}/**/*.arrow", recursive=True)
    
    dataset = []
    # Just load the first file to demonstrate
    for file_path in arrow_files:
        with pa.memory_map(file_path, 'r') as source:
            try: reader = pa.ipc.open_stream(source)
            except: 
                source.seek(0)
                reader = pa.ipc.open_file(source)
            dataset.extend(reader.read_all().to_pylist())

    print(f"Loaded {len(dataset)} test files.")
    
    results = []
    print("--- 4. GENERATING SUMMARIES (Fine-Tuned) ---")
    
    # Process files
    for i, record in enumerate(dataset):
        doc_id = record.get('id', str(i))
        raw_doc = record.get('document') or record.get('text') or []
        text = " ".join(raw_doc) if isinstance(raw_doc, list) else str(raw_doc)
        
        # Use the same format as training
        prompt = f"<start_of_turn>user\nSummarize this legal document:\n\n{text[:10000]}<end_of_turn>\n<start_of_turn>model\n"
        
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=512, do_sample=False)
            
        gen_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        
        print(f"Generated Summary for ID {doc_id}")
        results.append({
            "id": doc_id,
            "finetuned_summary": gen_text
        })

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"✅ Saved fine-tuned results to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
