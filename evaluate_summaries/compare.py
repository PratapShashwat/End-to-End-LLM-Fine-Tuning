import json
import random
import re

def calculate_group_averages(group):
    """Calculates the average ROUGE and average Eval score for a list of documents."""
    if not group: return 0.0, 0.0
    avg_r = sum(item["avg_rouge"] for item in group) / len(group)
    avg_e = sum(item["eval_score"] for item in group) / len(group)
    return avg_r, avg_e

def load_eval_scores(eval_data, is_text=False):
    """Dynamically parses the evaluation data, handling multiple JSON structures OR plain text."""
    eval_scores = {}
    
    # --- TEXT FILE PARSER ---
    if is_text:
        # This regex looks for: "id": "4963" ... followed eventually by ... "overall": 0.584
        # It handles newlines and spaces perfectly
        pattern = r'"id"\s*:\s*"(\d+)".*?"overall"\s*:\s*([\d.]+)'
        matches = re.findall(pattern, eval_data, re.DOTALL | re.IGNORECASE)
        
        if matches:
            for doc_id, score in matches:
                eval_scores[str(doc_id)] = float(score)
            print(f"📄 Text file detected. Successfully extracted {len(eval_scores)} scores via Regex.")
        else:
            print("⚠️ Warning: Could not find matching ID/Score patterns in the text file.")
        return eval_scores

    # --- JSON PARSERS ---
    # Structure 2: It's a direct list of dictionaries (G_evaluation_results_on!.json)
    if isinstance(eval_data, list):
        for item in eval_data:
            doc_id = str(item.get("id"))
            eval_scores[doc_id] = item.get("overall", 0.0)
            
    # Structure 1 & 3: It's a dictionary containing a list
    elif isinstance(eval_data, dict):
        # Structure 1: Uses "data" and "base_judgement" -> "individual_aggregate"
        if "data" in eval_data and isinstance(eval_data["data"], list):
            for item in eval_data["data"]:
                doc_id = str(item.get("id"))
                if "base_judgement" in item:
                    eval_scores[doc_id] = item["base_judgement"].get("individual_aggregate", 0.0)
                else:
                    eval_scores[doc_id] = item.get("overall", item.get("document_average", 0.0))
                    
        # Structure 3: Uses "document_results" and "document_average"
        elif "document_results" in eval_data and isinstance(eval_data["document_results"], list):
            for item in eval_data["document_results"]:
                doc_id = str(item.get("id"))
                eval_scores[doc_id] = item.get("document_average", 0.0)

    print(f"📋 JSON file detected. Successfully extracted {len(eval_scores)} scores.")
    return eval_scores

def main():
    # --- CONFIGURATION ---
    ROUGE_FILE = "../finetuned/lora/FINAL_FINETUNED_RESULTS.json" 
    EVAL_FILE = "G_evaluation_results_onQ.txt"  # <-- Put your .txt or .json file here!
    OUTPUT_FILE = "GonQ_rouge_vs_eval_report.json"
    
    # 1. Load the Data
    try:
        with open(ROUGE_FILE, 'r', encoding='utf-8') as f:
            rouge_data = json.load(f)
            
        # SMART LOADING: Always try JSON first, regardless of file extension!
        with open(EVAL_FILE, 'r', encoding='utf-8') as f:
            try:
                eval_data = json.load(f)
                is_text_file = False
            except json.JSONDecodeError:
                # If it fails, rewind and read as a raw text string
                f.seek(0)
                eval_data = f.read()
                is_text_file = True
                
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return

    # 2. Extract Evaluation Scores dynamically
    eval_scores = load_eval_scores(eval_data, is_text=is_text_file)

    if not eval_scores:
        print("Stopping execution: No evaluation scores were extracted.")
        return

    # 3. Extract ROUGE Scores and Merge
    combined_data = []
    rouge_docs = rouge_data.get("data", [])
    
    for doc in rouge_docs:
        doc_id = str(doc.get("id"))
        
        if doc_id in eval_scores:
            r_scores = doc.get("scores", {})
            r1 = r_scores.get("rouge1", 0.0)
            r2 = r_scores.get("rouge2", 0.0)
            rL = r_scores.get("rougeL", 0.0)
            
            avg_rouge = (r1 + r2 + rL) / 3.0
            
            combined_data.append({
                "doc_id": doc_id,
                "avg_rouge": avg_rouge,
                "eval_score": eval_scores[doc_id]
            })

    if not combined_data:
        print("No matching documents found between the two files. Make sure the IDs match!")
        return

    # 4. Sort by ROUGE score (Highest to Lowest)
    combined_data.sort(key=lambda x: x["avg_rouge"], reverse=True)

    # 5. Extract Groups
    top_10 = combined_data[:10]
    bottom_10 = combined_data[-10:]
    middle_pool = combined_data[10:-10] if len(combined_data) > 20 else combined_data
    middle_10 = random.sample(middle_pool, min(10, len(middle_pool)))
    
    # Calculate group insights
    top_r, top_e = calculate_group_averages(top_10)
    mid_r, mid_e = calculate_group_averages(middle_10)
    bot_r, bot_e = calculate_group_averages(bottom_10)

    # 6. Print Console Summary
    print("\n" + "="*55)
    print(f" 📊 ROUGE vs LLM EVALUATION CORRELATION REPORT")
    print(f"    Testing File: {EVAL_FILE}")
    print("="*55)
    print(f"Total matched documents: {len(combined_data)}\n")
    
    print(f"🏆 TOP 10 ROUGE DOCUMENTS:")
    print(f"   -> Average ROUGE Score: {top_r:.4f}")
    print(f"   -> Average LLM Eval Score:  {top_e:.4f}")
    
    print(f"\n⚖️  MIDDLE 10 ROUGE DOCUMENTS (Random Sample):")
    print(f"   -> Average ROUGE Score: {mid_r:.4f}")
    print(f"   -> Average LLM Eval Score:  {mid_e:.4f}")
    
    print(f"\n📉 BOTTOM 10 ROUGE DOCUMENTS:")
    print(f"   -> Average ROUGE Score: {bot_r:.4f}")
    print(f"   -> Average LLM Eval Score:  {bot_e:.4f}")
    print("="*55 + "\n")

    # 7. Save Detailed JSON Report
    report = {
        "tested_eval_file": EVAL_FILE,
        "insights": {
            "top_10_avg_rouge": top_r,
            "top_10_avg_eval": top_e,
            "mid_10_avg_rouge": mid_r,
            "mid_10_avg_eval": mid_e,
            "bot_10_avg_rouge": bot_r,
            "bot_10_avg_eval": bot_e
        },
        "top_10_documents": top_10,
        "middle_10_documents": middle_10,
        "bottom_10_documents": bottom_10
    }

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=4)
        
    print(f"✅ Detailed data saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()