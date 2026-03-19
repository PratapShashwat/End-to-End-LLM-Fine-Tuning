import json
import argparse
import re
import torch
import gc  # Added Garbage Collector
from transformers import AutoTokenizer, AutoModelForCausalLM

def parse_json_output(response_text):
    """Extracts and parses JSON from the model's text generation."""
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        # Fallback to finding anything that resembles a JSON object
        match = re.search(r'\{.*?\}', response_text, re.DOTALL)
        json_str = match.group(0) if match else response_text
        
    try:
        data = json.loads(json_str)
        return {
            "faithfulness": float(data.get("faithfulness", 0.0)),
            "coverage": float(data.get("coverage", 0.0)),
            "fluency": float(data.get("fluency", 0.0)),
            "legal_reasoning": float(data.get("legal_reasoning", 0.0))
        }
    except Exception as e:
        print(f"Failed to parse model output: {response_text}\nError: {e}")
        # Return zeros on failure to prevent crashing the loop
        return {"faithfulness": 0.0, "coverage": 0.0, "fluency": 0.0, "legal_reasoning": 0.0}

def main():
    parser = argparse.ArgumentParser(description="Evaluate legal summaries using local Gemma model")
    parser.add_argument("--input", type=str, required=True, help="Input JSON file path")
    parser.add_argument("--output", type=str, required=True, help="Output JSON file path")
    parser.add_argument("--model_path", type=str, required=True, help="Local path to the downloaded Gemma model")
    args = parser.parse_args()

    # 1. Load the input JSON file
    with open(args.input, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    # --- FORMAT NORMALIZATION ---
    if isinstance(raw_data, dict):
        for key in ["data", "results", "documents", "document_results"]:
            if key in raw_data and isinstance(raw_data[key], list):
                data = raw_data[key]
                break
        else:
            data = list(raw_data.values()) if not ("id" in raw_data and "reference" in raw_data) else [raw_data]
    else:
        data = raw_data

    # 2. Initialize Gemma Model and Tokenizer (Allocating automatically to available GPUs)
    print("Loading model into VRAM...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    # Using float16 instead of bfloat16 to save a bit of VRAM
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, 
        device_map="auto", 
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )

    results = []
    totals = {"faithfulness": 0.0, "coverage": 0.0, "fluency": 0.0, "legal_reasoning": 0.0, "overall_average": 0.0}

    # 3. Process each document
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            continue

        doc_id = item.get("id", "unknown")
        reference = item.get("reference", "")
        generated = item.get("generated", "")

        if not reference or not generated:
            continue

        prompt_text = f"""You are an expert legal evaluator. Evaluate the Generated summary against the Reference text.
Score the following 4 criteria strictly from 0.0 to 1.0:
1) faithfulness: Accuracy to the reference.
2) coverage: Inclusion of all key legal details.
3) fluency: Readability and grammar.
4) legal_reasoning: Correct application of legal logic.

Reference:
{reference}

Generated:
{generated}

Output ONLY a valid JSON object with the keys "faithfulness", "coverage", "fluency", and "legal_reasoning" mapped to their float scores. Do not output any conversational text."""

        messages = [{"role": "user", "content": prompt_text}]
        
        # We truncate the input context slightly if the document is excessively long
        input_ids = tokenizer.apply_chat_template(
            messages, 
            tokenize=True, 
            add_generation_prompt=True, 
            return_tensors="pt",
            max_length=4000, 
            truncation=True
        ).to(model.device)
        
        # 4. Generate the evaluation with no_grad
        with torch.no_grad():
            outputs = model.generate(
                input_ids, 
                max_new_tokens=150, 
                temperature=0.1, 
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
        scores = parse_json_output(response)
        
        doc_avg = sum(scores.values()) / 4.0
        
        item_result = {
            "id": doc_id,
            "reference": reference,
            "generated": generated,
            "scores": scores,
            "document_average": doc_avg
        }
        results.append(item_result)
        
        for key in ["faithfulness", "coverage", "fluency", "legal_reasoning"]:
            totals[key] += scores[key]
        totals["overall_average"] += doc_avg
        
        print(f"Processed ID: {doc_id} | Doc Avg: {doc_avg:.3f} | Progress: {idx+1}/{len(data)}")

        # --- EXPLICIT MEMORY CLEANUP ---
        del input_ids
        del outputs
        gc.collect()
        torch.cuda.empty_cache()

    # 6. Calculate criteria-wise and grand totals
    num_docs = len(results)
    grand_averages = {k: (v / num_docs if num_docs > 0 else 0) for k, v in totals.items()}

    final_output = {
        "grand_averages": grand_averages,
        "document_results": results
    }

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=4)
        
    print(f"\nEvaluation complete. Results saved to {args.output}")

if __name__ == "__main__":
    main()