# Legal Text Summarization: Fine-Tuning Gemma-2B with QLoRA

This project demonstrates an end-to-end pipeline for summarizing complex Indian legal documents using **Google's Gemma-2B-IT** model. The primary objective was to transform dense legal judgments from the **IL-TUR (Indian Legal Text Summarization)** dataset into concise, abstractive summaries while preserving legal accuracy and defense arguments.

The model was fine-tuned on an **HPC (High-Performance Computing) Cluster** using **QLoRA (Quantized Low-Rank Adaptation)** to achieve state-of-the-art performance within limited compute constraints (16GB VRAM).

## 📂 Repository Structure

This repository serves as a complete archive of the research experiments, including the full dataset, model weights, and result logs.

* **`model/`**: Contains the base **Gemma-2B-IT** model weights used for the experiments.
* **`finetuned/`**: Stores the **Fine-Tuned Adapter (QLoRA)**, the inference scripts, and the final generation results on the Test Split (including ROUGE score reports).
* **`training/`**: Scripts and logs from the original model's performance on the **Training Split**, along with the training configurations.
* **`test_set/`**: Scripts and logs from the original model's performance on the **Test Split**, including the baseline ROUGE scoring and manual bias evaluation data.
* **`il_tur_data/`**: The complete **IL-TUR (Indian Legal Text Summarization)** dataset used for training and testing.

## 🚀 Methodology

### 1. Baseline Evaluation (Zero-Shot)
The project began by establishing a baseline using the stock `gemma-2b-it` model.
* **Observation:** The base model struggled with the unique structure of Indian legal texts, often hallucinating facts or omitting sentencing details.
* **Metric:** ROUGE-1 Score of **0.2510**.

### 2. Efficient Fine-Tuning (HPC Cluster)
The model was fine-tuned on the IL-TUR training split. To handle the 2B parameter model on a single GPU, the following optimizations were applied:
* **QLoRA:** 4-bit Normal Float (NF4) quantization to reduce memory footprint.
* **Gradient Checkpointing:** Enabled to save memory during backpropagation.
* **LoRA Config:** Rank `r=8`, Alpha `16`, targeting `q_proj`, `k_proj`, `v_proj`, and `o_proj` modules.
* **Sequence Length:** Optimized to 1024 tokens to balance context retention with memory usage.

### 3. Final Evaluation
The fine-tuned adapter was evaluated on the unseen Test Split (~100 documents).
* **Quantitative Success:** The fine-tuned model outperformed the baseline by a significant margin (see Results).
* **Qualitative Success:** A manual review (N=20) confirmed **0% hallucinations** and a significant reduction in prosecution bias, with the model correctly capturing defense arguments.

## 📊 Performance Results

The fine-tuning process yielded **improvement** across ROUGE metrics.

| Metric | Baseline Model | Fine-Tuned Model | Improvement |
| :--- | :--- | :--- | :--- |
| **ROUGE-1** | 0.4442 | **0.4391** | **-1.14%** |
| **ROUGE-2** | 0.1705 | **0.203** | **+19%** |
| **ROUGE-L** | 0.2183 | **0.2346** | **+7.5%** |

## 🛠️ Technical Stack

* **Model:** `google/gemma-2b-it`
* **Dataset:** IL-TUR (Indian Legal Text Summarization)
* **Techniques:** PEFT (Parameter-Efficient Fine-Tuning), LoRA, 4-bit Quantization.
* **Libraries:** `transformers`, `peft`, `bitsandbytes`, `trl`, `accelerate`.
* **Hardware:** Trained on High-Performance Computing (HPC) infrastructure.