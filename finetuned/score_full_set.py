import json
import numpy as np
try:
    from rouge_score import rouge_scorer
except ImportError:
    print("❌ Error: Library missing. Run: pip install rouge-score")
    exit()

# CONFIG
INPUT_FILE = "finetuned_test_ready_to_score.json"
OUTPUT_FILE = "FINAL_FINETUNED_RESULTS.json"

print(f"--- 📊 SCORING FULL TEST SET ---")

# 1. Load
with open(INPUT_FILE, 'r') as f:
    data = json.load(f)

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
all_scores = {'r1': [], 'r2': [], 'rL': []}

# 2. Score Each Document
for entry in data:
    ref = entry.get('reference', '')
    gen = entry.get('generated', '')
    
    if len(ref) > 0 and len(gen) > 0:
        s = scorer.score(ref, gen)
        r1 = s['rouge1'].fmeasure
        r2 = s['rouge2'].fmeasure
        rL = s['rougeL'].fmeasure
        
        # Add score to entry
        entry['scores'] = {
            "rouge1": round(r1, 4),
            "rouge2": round(r2, 4),
            "rougeL": round(rL, 4)
        }
        
        all_scores['r1'].append(r1)
        all_scores['r2'].append(r2)
        all_scores['rL'].append(rL)

# 3. Calculate Final Means
stats = {
    "mean_rouge1": round(np.mean(all_scores['r1']), 4),
    "mean_rouge2": round(np.mean(all_scores['r2']), 4),
    "mean_rougeL": round(np.mean(all_scores['rL']), 4),
    "total_documents": len(all_scores['r1'])
}

# 4. Save Final Output
final_output = {
    "overall_stats": stats,
    "data": data
}

with open(OUTPUT_FILE, 'w') as f:
    json.dump(final_output, f, indent=4)

print(f"\n✅ SUCCESS! Final file saved to: {OUTPUT_FILE}")
print(f"🏆 FINAL METRICS:")
print(f"   ROUGE-1: {stats['mean_rouge1']}")
print(f"   ROUGE-2: {stats['mean_rouge2']}")
print(f"   ROUGE-L: {stats['mean_rougeL']}")