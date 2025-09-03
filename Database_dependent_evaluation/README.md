# Database-Dependent Evaluation: Clinical Trial QA

This directory contains the data for the database-dependent evaluation pipeline of our framework. The goal of this stage is to objectively measure the factuality of Large Language Model (LLM) responses against a structured knowledge base. The findings from this evaluation also provide the labeled data used to train our database-free classifier.

---

## Experimental Design

Our experiment was designed to assess LLM performance in a high-stakes question-answering (QA) task on oncology clinical trials.

### **1. Baseline Hallucination Rates (Low-Context)**

We first established a baseline for factuality across a diverse suite of state-of-the-art models.

* **Task:** Each model answered **15 questions** for **100 randomly selected clinical trials**.
* **Context:** Models were intentionally provided with **only the clinical trial title** as context. This low-context setting was designed to probe each model's tendency to hallucinate when faced with a lack of information.
* **Models Tested:**
    * DeepSeek-R1-Distill-Llama-70B
    * GPT-o3
    * GPT-5
    * Llama3.3-70B-Instruct

### **2. Influence of Context on a Fixed Model**

Next, we investigated how the quality of context impacts factual accuracy. For this test, we focused on a single powerful, open-source model (Llama3.3-70B-Instruct) to ensure reproducibility.

* **Task:** The same 15-question QA task for the 100 trials.
* **Context Conditions:** We evaluated the model's responses using three different types of input context:
    1.  **Title-Only:** The minimal context from the baseline test.
    2.  **Raw JSON Record:** The complete, unstructured trial data.
    3.  **Curated Summary:** A computationally generated, structured summary of the trial.

---

## Evaluation Method

All model-generated answers were evaluated using a **factual and counterfactual analysis** performed by an independent LLM judge. The judge cross-references each claim in an answer against the ground-truth clinical trial database to determine if it is supported or contradicted by the available evidence.


<img width="7057" height="1618" alt="fig-2" src="https://github.com/user-attachments/assets/c1cd9776-fd8d-498c-97b7-01cdbb8acd95" />


* A statement is considered **Supported** if the LLM judge can deduce it from a set of explicit statements within the database.
* A statement is deemed **Contradicted** if the LLM judge can deduce its negation from a set of explicit statements in the database.

Based on these criteria, the judge classifies the content into one of four categories:

1.  **Fact:** The response is Supported and not Contradicted.
2.  **Hallucination:** The response is Contradicted and not Supported.
3.  **Judgment Error:** The response is both Supported and Contradicted, implying a database inconsistency.
4.  **Coverage Gap:** The response is neither Supported nor Contradicted by the database.


<img width="991" height="617" alt="Screenshot 2025-09-03 at 5 57 28 PM" src="https://github.com/user-attachments/assets/35b953b3-dcba-47e1-a55f-863c3fb18341" />



For complete details on the methodology and results, please refer to the main paper.
```eof
