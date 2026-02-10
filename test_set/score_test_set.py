import json
import numpy as np

# CONFIG
INPUT_FILE = "test_set_full_data.json"  # The file we just downloaded
OUTPUT_FILE = "Final_Test_Set_Scored.json"

try:
    from rouge_score import rouge_scorer
except ImportError:
    print("❌ Please run: pip install rouge-score")
    exit()

print("--- CALCULATING TEST SET SCORES ---")

with open(INPUT_FILE, 'r') as f:
    data = json.load(f)

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
all_scores = {'r1': [], 'r2': [], 'rL': []}
scored_count = 0

for entry in data:
    ref = entry.get('reference', '')
    gen = entry.get('generated', '')
    
    # Check if we actually found a reference text (it might be empty)
    if len(ref) > 10:
        s = scorer.score(ref, gen)
        
        # Save score to entry
        entry['scores'] = {
            "rouge1": s['rouge1'].fmeasure,
            "rouge2": s['rouge2'].fmeasure,
            "rougeL": s['rougeL'].fmeasure
        }
        
        all_scores['r1'].append(s['rouge1'].fmeasure)
        all_scores['r2'].append(s['rouge2'].fmeasure)
        all_scores['rL'].append(s['rougeL'].fmeasure)
        scored_count += 1
    else:
        entry['scores'] = "NO REFERENCE AVAILABLE"

# Calculate Averages
if scored_count > 0:
    avg_r1 = np.mean(all_scores['r1'])
    avg_r2 = np.mean(all_scores['r2'])
    avg_rL = np.mean(all_scores['rL'])
    
    print(f"\n✅ SCORED {scored_count} DOCUMENTS")
    print(f"🏆 ROUGE-1: {avg_r1:.4f}")
    print(f"🏆 ROUGE-2: {avg_r2:.4f}")
    print(f"🏆 ROUGE-L: {avg_rL:.4f}")
else:
    print("\n⚠️ NO REFERENCES FOUND. This is likely a 'Blind Test' dataset.")

# Save final
with open(OUTPUT_FILE, 'w') as f:
    json.dump(data, f, indent=4)