# Hybrid RAG & Fine-Tuning for Customer Support

This project is from my upGrad program. The idea is to build a small customer support bot that does two things well. First, it figures out what the customer actually wants, even when the message is angry, sarcastic or vague. Second, it answers only from the company's own SOPs (Standard Operating Procedures) and doesn't make up refund windows, delivery times or phone numbers.

We built it in three steps so we could measure what each part adds:

| Version | What it does |
|---|---|
| **Baseline** | Plain `Qwen2.5-1.5B-Instruct`, no company info given to it |
| **V1 - Naive RAG** | Embeds the raw customer message, pulls the closest SOP from ChromaDB and puts it in the prompt |
| **V2 - Hybrid RAG** | A LoRA fine-tuned router first turns the message into `{"intent", "category"}` JSON. The intent picks the SOP, and the same generator answers from it |

```
customer message -> [LoRA router] -> {"intent","category"} -> intent -> SOP search string
                 -> [ChromaDB top-1] -> SOP -> [Qwen2.5 generator] -> answer
```

## Results

Everything is scored on the same 390-query held-out test split. None of these queries appear in train or valid.

| Metric | Baseline | V1 Naive RAG | V2 Hybrid RAG |
|---|---|---|---|
| Intent accuracy, test | - | - | **96.4%** (base model 56.4%) |
| Intent accuracy, adversarial (38 rows) | - | - | **97.4%** (base model 50.0%) |
| Retrieval Hit@1 | - | 44.1% | **99.0%** |
| ROUGE-L vs SOP | 0.065 | 0.079 | **0.085** |
| BLEU vs SOP | 0.0005 | 0.0025 | **0.0032** |
| Hallucination frequency | 4.1% | 3.1% | 3.1% |
| Format adherence / consistency | 100% | 100% | 100% |

A few things we noticed:
- The fine-tuned router made the biggest difference. Better intents meant the right SOP almost every time (214 queries fixed, 0 broken compared to V1).
- Adding an SOP cut hallucinations by about a quarter. V2 didn't cut them any further though. What's left comes from the 1.5B generator filling gaps with placeholders like `[phone number]`.
- We tried a stricter prompt ("never make up a phone number or link") and it actually made things worse (7.4%), so we went back to the simpler prompt.

The full write-up is in [Comparative_Analysis_Report.pdf](Comparative_Analysis_Report.pdf), and the plan and metric definitions are in [Project_Proposal_and_Methodology.pdf](Project_Proposal_and_Methodology.pdf).

## Data

- **Queries:** [Bitext customer support dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset) (26,872 rows, 27 intents, 11 categories). We used a 4,000-row sample, which came to 3,896 rows after cleaning and dedup, split 80/10/10 stratified on intent.
- **SOPs:** 13 markdown policies in `Dataset.zip` (refund policy, shipping delays, password reset, escalation matrix etc.)

## Notebooks

The notebooks in this repo are the executed versions, so you can see all outputs without running anything.

| # | Notebook | What happens there |
|---|---|---|
| 1 | `Data_Understanding_and_EDA.ipynb` | Load data and SOPs, quality checks, intent distribution, TF-IDF similarity between SOPs |
| 2 | `Data_Preparation.ipynb` | Cleaning, ChatML router prompts, tokenisation with label masking, intent-to-SOP map, train/valid/test split |
| 3 | `Baseline_Model_Evaluation.ipynb` | Base model with no context, hallucination probes |
| 4 | `RAG_Implementation.ipynb` | MiniLM embeddings, ChromaDB index, naive RAG |
| 5 | `Solution_V1_RAG_Evaluation.ipynb` | Baseline vs V1 on the full test split |
| 6 | `Fine_Tuning_Pipeline.ipynb` | QLoRA training of the intent router with early stopping |
| 7 | `Solution_V2_FineTuned_RAG_Evaluation.ipynb` | Router eval (test + adversarial), V2 eval, final comparison of all three |

`Run_All_in_Colab.ipynb` is a helper that runs all 7 in order.

## Running it

It's set up for Google Colab on the free T4 GPU, with everything kept in Google Drive.

1. Create `MyDrive/hybrid_rag/` in Google Drive and put `Dataset.zip` and the 7 notebooks in it.
2. Open `Run_All_in_Colab.ipynb` in Colab and set Runtime -> Change runtime type -> **T4 GPU**.
3. Run the setup cell, then the 7 notebook cells top to bottom.

Executed copies land in `MyDrive/hybrid_rag/executed/`. Your original notebooks are not overwritten. The whole thing takes roughly 1.5 hours, and the fine-tuning part is about 16-18 min. Everything else (cleaned data, ChromaDB, LoRA adapter, metric CSVs, charts) gets regenerated in Drive, so none of it is kept in this repo.

No Hugging Face token is needed, since all models are public.

## Stack

`Qwen/Qwen2.5-1.5B-Instruct` (4-bit, bitsandbytes) · LoRA via `peft` (r=16 on q_proj/v_proj, ~0.14% params trainable) · `sentence-transformers/all-MiniLM-L6-v2` · ChromaDB + LangChain · `rouge-score`, `nltk` BLEU · PyTorch, transformers

## Limitations

- ROUGE/BLEU are measured against the SOP text, not a human-written answer, so they only make sense for comparing the three versions.
- The hallucination check is rule-based (made-up numbers, placeholders, fake contacts, asking for passwords), so it probably undercounts.
- Small model, 13 SOPs and a 38-row adversarial set. See section 6 of the report for more.
