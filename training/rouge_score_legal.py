import json
import numpy as np

# ---------------- CONFIGURATION ----------------
INPUT_FILE = "final_il_tur.json"       # Your file from the supercomputer
OUTPUT_FILE = "final_il_tur_scored.json"
# -----------------------------------------------

# IMPORT GOOGLE'S ROUGE SCORER
try:
    from rouge_score import rouge_scorer
except ImportError:
    print("❌ Error: Library missing.")
    print("👉 Run: pip install rouge-score nltk")
    exit()

print(f"--- 🛠️ FIXING SCORES USING GOOGLE ROUGE-SCORE ---")

# 1. Load Data
try:
    with open(INPUT_FILE, 'r') as f:
        full_data = json.load(f)
    # Handle list vs dict format
    data_entries = full_data if isinstance(full_data, list) else full_data.get("data", [])
    print(f"✅ Loaded {len(data_entries)} summaries.")
except Exception as e:
    print(f"❌ Error reading file: {e}")
    exit()

# 2. Initialize Scorer (use_stemmer=True matches 'charged' with 'charge')
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
all_scores = {'r1': [], 'r2': [], 'rL': []}

print("--- CALCULATING SCORES ---")

for i, entry in enumerate(data_entries):
    ref = entry.get('reference', '')
    gen = entry.get('generated', '')
    
    # Check if texts exist
    if isinstance(ref, str) and len(ref) > 10 and isinstance(gen, str) and len(gen) > 10:
        try:
            # CALCULATE
            scores = scorer.score(ref, gen)
            
            # Extract F-Measure (Standard metric)
            r1 = scores['rouge1'].fmeasure
            r2 = scores['rouge2'].fmeasure
            rL = scores['rougeL'].fmeasure
            
            # Save to entry
            entry['scores'] = {
                "rouge1": round(r1, 4),
                "rouge2": round(r2, 4),
                "rougeL": round(rL, 4)
            }
            
            all_scores['r1'].append(r1)
            all_scores['r2'].append(r2)
            all_scores['rL'].append(rL)
            
        except Exception as e:
            print(f"⚠️ Error on entry {i}: {e}")
            entry['scores'] = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    else:
        # Zero out if text is missing
        entry['scores'] = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    if i % 100 == 0 and i > 0:
        print(f"   Processed {i} entries...")

# 3. Calculate Final Averages
avg_r1 = np.mean(all_scores['r1']) if all_scores['r1'] else 0.0
avg_r2 = np.mean(all_scores['r2']) if all_scores['r2'] else 0.0
avg_rL = np.mean(all_scores['rL']) if all_scores['rL'] else 0.0

final_structure = {
    "stats": {
        "avg_r1": round(avg_r1, 4),
        "avg_r2": round(avg_r2, 4),
        "avg_rL": round(avg_rL, 4)
    },
    "data": data_entries
}

# 4. Save
with open(OUTPUT_FILE, 'w') as f:
    json.dump(final_structure, f, indent=4)

print(f"\n✅ SUCCESS! Saved to: {OUTPUT_FILE}")
print(f"🏆 FINAL STATS: R1: {avg_r1:.4f} | R2: {avg_r2:.4f} | RL: {avg_rL:.4f}")