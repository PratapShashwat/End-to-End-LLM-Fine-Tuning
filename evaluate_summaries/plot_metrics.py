import json
import matplotlib.pyplot as plt

def load_all_metrics(eval_data):
    """Dynamically parses the evaluation JSON to extract all criteria scores."""
    metrics_data = []
    
    if isinstance(eval_data, list):
        for item in eval_data:
            doc_id = str(item.get("id", ""))
            metrics_data.append({
                "id": doc_id,
                "faithfulness": item.get("faithfulness", 0.0),
                "coverage": item.get("coverage", 0.0),
                "fluency": item.get("fluency", 0.0),
                "legal_reasoning": item.get("legal_reasoning", 0.0),
                "average": item.get("overall", 0.0)
            })
            
    elif isinstance(eval_data, dict):
        if "data" in eval_data and isinstance(eval_data["data"], list):
            for item in eval_data["data"]:
                doc_id = str(item.get("id", ""))
                if "base_judgement" in item:
                    scores = item["base_judgement"]
                    metrics_data.append({
                        "id": doc_id,
                        "faithfulness": scores.get("faithfulness", 0.0),
                        "coverage": scores.get("coverage", 0.0),
                        "fluency": scores.get("fluency", 0.0),
                        "legal_reasoning": scores.get("legal_reasoning", 0.0),
                        "average": scores.get("individual_aggregate", 0.0)
                    })
                else:
                    metrics_data.append({
                        "id": doc_id,
                        "faithfulness": item.get("faithfulness", 0.0),
                        "coverage": item.get("coverage", 0.0),
                        "fluency": item.get("fluency", 0.0),
                        "legal_reasoning": item.get("legal_reasoning", 0.0),
                        "average": item.get("overall", item.get("document_average", 0.0))
                    })
                    
        elif "document_results" in eval_data and isinstance(eval_data["document_results"], list):
            for item in eval_data["document_results"]:
                doc_id = str(item.get("id", ""))
                scores = item.get("scores", {})
                metrics_data.append({
                    "id": doc_id,
                    "faithfulness": scores.get("faithfulness", 0.0),
                    "coverage": scores.get("coverage", 0.0),
                    "fluency": scores.get("fluency", 0.0),
                    "legal_reasoning": scores.get("legal_reasoning", 0.0),
                    "average": item.get("document_average", 0.0)
                })

    return metrics_data

def main():
    #input files
    EVAL_FILE = "G_evaluation_results_onQ.txt"  
    
    try:
        with open(EVAL_FILE, 'r', encoding='utf-8') as f:
            eval_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find {EVAL_FILE}")
        return
    except json.JSONDecodeError:
        print(f"Error: {EVAL_FILE} is not a valid JSON file.")
        return

    metrics_data = load_all_metrics(eval_data)

    if not metrics_data:
        print("Could not extract metrics. Check the file structure.")
        return

    metrics_data.sort(key=lambda x: x["average"])

    ids = [item["id"] for item in metrics_data]
    faithfulness = [item["faithfulness"] for item in metrics_data]
    coverage = [item["coverage"] for item in metrics_data]
    fluency = [item["fluency"] for item in metrics_data]
    legal_reasoning = [item["legal_reasoning"] for item in metrics_data]
    average = [item["average"] for item in metrics_data]

    plt.figure(figsize=(14, 8))

    
    plt.plot(ids, fluency, label='Fluency', color='green', alpha=0.6, linewidth=1.5)
    plt.plot(ids, faithfulness, label='Faithfulness', color='blue', alpha=0.6, linewidth=1.5)
    plt.plot(ids, coverage, label='Coverage', color='orange', alpha=0.6, linewidth=1.5)
    plt.plot(ids, legal_reasoning, label='Legal Reasoning', color='red', alpha=0.6, linewidth=1.5)
    
    
    plt.plot(ids, average, label='OVERALL AVERAGE', color='black', linewidth=3, linestyle='--')

    plt.title(f"LLM Performance Across Criteria ({EVAL_FILE})\n*Sorted by Overall Average Score*", fontsize=14, fontweight='bold')
    plt.xlabel("Document IDs (Sorted Lowest to Highest Perfomance)", fontsize=12)
    plt.ylabel("Score (0.0 to 1.0)", fontsize=12)
    plt.ylim(-0.05, 1.05)
    
    if len(ids) > 20:
        plt.xticks([]) 
    else:
        plt.xticks(rotation=90)

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(loc='lower right', framealpha=0.9)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()