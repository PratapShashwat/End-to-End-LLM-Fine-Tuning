import json
import re
import matplotlib.pyplot as plt

def load_rouge_scores(filepath):
    """Loads the 3 ROUGE scores from the finetuned results file."""
    rouge_dict = {}
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if "data" in data:
                for item in data["data"]:
                    doc_id = str(item.get("id", ""))
                    scores = item.get("scores", {})
                    rouge_dict[doc_id] = {
                        "rouge1": scores.get("rouge1", 0.0),
                        "rouge2": scores.get("rouge2", 0.0),
                        "rougeL": scores.get("rougeL", 0.0)
                    }
        print(f"📖 Loaded {len(rouge_dict)} ROUGE scores.")
    except Exception as e:
        print(f"Error loading ROUGE file: {e}")
    return rouge_dict

def load_eval_metrics(eval_data, is_text=False):
    """Dynamically parses the 5 LLM evaluation metrics (JSON or Regex fallback)."""
    metrics_dict = {}
    

    if is_text:
        pattern = r'"id"\s*:\s*"(\d+)".*?"faithfulness"\s*:\s*([\d.]+).*?"coverage"\s*:\s*([\d.]+).*?"fluency"\s*:\s*([\d.]+).*?"legal_reasoning"\s*:\s*([\d.]+).*?"overall"\s*:\s*([\d.]+)'
        matches = re.findall(pattern, eval_data, re.DOTALL | re.IGNORECASE)
        if matches:
            for match in matches:
                metrics_dict[str(match[0])] = {
                    "faithfulness": float(match[1]),
                    "coverage": float(match[2]),
                    "fluency": float(match[3]),
                    "legal_reasoning": float(match[4]),
                    "average": float(match[5])
                }
        return metrics_dict

    if isinstance(eval_data, list):
        for item in eval_data:
            doc_id = str(item.get("id", ""))
            metrics_dict[doc_id] = {
                "faithfulness": item.get("faithfulness", 0.0),
                "coverage": item.get("coverage", 0.0),
                "fluency": item.get("fluency", 0.0),
                "legal_reasoning": item.get("legal_reasoning", 0.0),
                "average": item.get("overall", 0.0)
            }
            
    elif isinstance(eval_data, dict):
        if "data" in eval_data and isinstance(eval_data["data"], list):
            for item in eval_data["data"]:
                doc_id = str(item.get("id", ""))
                if "base_judgement" in item:
                    scores = item["base_judgement"]
                    metrics_dict[doc_id] = {
                        "faithfulness": scores.get("faithfulness", 0.0),
                        "coverage": scores.get("coverage", 0.0),
                        "fluency": scores.get("fluency", 0.0),
                        "legal_reasoning": scores.get("legal_reasoning", 0.0),
                        "average": scores.get("individual_aggregate", 0.0)
                    }
                else:
                    metrics_dict[doc_id] = {
                        "faithfulness": item.get("faithfulness", 0.0),
                        "coverage": item.get("coverage", 0.0),
                        "fluency": item.get("fluency", 0.0),
                        "legal_reasoning": item.get("legal_reasoning", 0.0),
                        "average": item.get("overall", item.get("document_average", 0.0))
                    }
                    
        elif "document_results" in eval_data and isinstance(eval_data["document_results"], list):
            for item in eval_data["document_results"]:
                doc_id = str(item.get("id", ""))
                scores = item.get("scores", {})
                metrics_dict[doc_id] = {
                    "faithfulness": scores.get("faithfulness", 0.0),
                    "coverage": scores.get("coverage", 0.0),
                    "fluency": scores.get("fluency", 0.0),
                    "legal_reasoning": scores.get("legal_reasoning", 0.0),
                    "average": item.get("document_average", 0.0)
                }

    return metrics_dict

def main():
    
    ROUGE_FILE = "../finetuned/lora/FINAL_FINETUNED_RESULTS.json" 
    EVAL_FILE = "G_evaluation_results_onQ.txt"  # <-- Change to L, G, or Q file here!
    
    
    rouge_dict = load_rouge_scores(ROUGE_FILE)
    if not rouge_dict:
        return

    
    try:
        with open(EVAL_FILE, 'r', encoding='utf-8') as f:
            try:
                eval_data = json.load(f)
                is_text_file = False
            except json.JSONDecodeError:
                f.seek(0)
                eval_data = f.read()
                is_text_file = True
    except FileNotFoundError:
        print(f"Error: Could not find {EVAL_FILE}")
        return

    eval_dict = load_eval_metrics(eval_data, is_text=is_text_file)
    print(f"📖 Loaded {len(eval_dict)} LLM Evaluation scores.")

    
    merged_data = []
    for doc_id, eval_scores in eval_dict.items():
        if doc_id in rouge_dict:
            r_scores = rouge_dict[doc_id]
            merged_data.append({
                "id": doc_id,
                "faithfulness": eval_scores["faithfulness"],
                "coverage": eval_scores["coverage"],
                "fluency": eval_scores["fluency"],
                "legal_reasoning": eval_scores["legal_reasoning"],
                "average": eval_scores["average"],
                "rouge1": r_scores["rouge1"],
                "rouge2": r_scores["rouge2"],
                "rougeL": r_scores["rougeL"]
            })

    if not merged_data:
        print("Error: Could not match any IDs between the two files! Check your file paths.")
        return
        
    print(f"✅ Successfully merged {len(merged_data)} documents.")

    
    merged_data.sort(key=lambda x: x["average"])

    ids = [item["id"] for item in merged_data]
    faithfulness = [item["faithfulness"] for item in merged_data]
    coverage = [item["coverage"] for item in merged_data]
    fluency = [item["fluency"] for item in merged_data]
    legal_reasoning = [item["legal_reasoning"] for item in merged_data]
    average = [item["average"] for item in merged_data]
    
    rouge1 = [item["rouge1"] for item in merged_data]
    rouge2 = [item["rouge2"] for item in merged_data]
    rougeL = [item["rougeL"] for item in merged_data]

    
    plt.figure(figsize=(16, 9))

    
    plt.plot(ids, fluency, label='LLM: Fluency', color='green', alpha=0.5, linewidth=1.5)
    plt.plot(ids, faithfulness, label='LLM: Faithfulness', color='blue', alpha=0.5, linewidth=1.5)
    plt.plot(ids, coverage, label='LLM: Coverage', color='orange', alpha=0.5, linewidth=1.5)
    plt.plot(ids, legal_reasoning, label='LLM: Legal Reasoning', color='red', alpha=0.5, linewidth=1.5)
    
    
    plt.plot(ids, average, label='LLM: OVERALL AVERAGE', color='black', linewidth=3, linestyle='--')


    plt.plot(ids, rouge1, label='ROUGE-1', color='purple', linestyle=':', linewidth=2.5, alpha=0.8)
    plt.plot(ids, rougeL, label='ROUGE-L', color='brown', linestyle=':', linewidth=2.5, alpha=0.8)
    plt.plot(ids, rouge2, label='ROUGE-2', color='magenta', linestyle=':', linewidth=2.5, alpha=0.8)


    plt.title(f"LLM vs ROUGE Performance Across Criteria ({EVAL_FILE})\n*Sorted by LLM Overall Average Score*", fontsize=15, fontweight='bold')
    plt.xlabel("Document IDs (Sorted Lowest to Highest LLM Perfomance)", fontsize=12)
    plt.ylabel("Score (0.0 to 1.0)", fontsize=12)
    plt.ylim(-0.05, 1.05) 
    
    if len(ids) > 20:
        plt.xticks([]) 
    else:
        plt.xticks(rotation=90)

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), framealpha=0.9, fontsize=11)
    plt.tight_layout() 
    plt.show()

if __name__ == "__main__":
    main()