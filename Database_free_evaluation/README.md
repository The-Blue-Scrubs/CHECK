# Database-Free Evaluation: A Classifier-Based Approach

This directory contains the experiments that validate our database-free pipeline. The core of this approach is a model-agnostic classifier designed to detect hallucinations without relying on an external knowledge base.

The experiments here test our central hypothesis: that hallucinations exhibit a distinct and learnable **"phenotype"** detectable from a model's output probabilities alone. This approach is grounded in the principle that **truth is stable, while hallucinations are probabilistically unstable**.

---

## Core Methodology: Learning the Hallucination Phenotype

Instead of cross-referencing a database, we trained a stacking classifier to identify hallucinations based on statistical signals of uncertainty and inconsistency. The classifier uses a feature set grounded in information theory to capture two key distributional signals from an ensemble of models.

1.  **Distributional Sharpness (Uncertainty):** Factual, confident answers tend to produce sharply peaked, low-entropy probability distributions for the next token. In contrast, confused outputs often exhibit flatter, high-entropy distributions.
2.  **Cross-Ensemble Consistency (Variance):** Factual answers yield stable and similar next-token distributions, even when the input query is rephrased or different models are used. Hallucinations, particularly confabulations, result in high-variance distributions (measured by Kullback–Leibler divergence) across the ensemble.

This process of building features from an ensemble of model outputs is illustrated below.

<p align="center">
  <img src="https://github.com/user-attachments/assets/ed727ce6-141a-4dc8-b020-04436e73adb9" alt="Feature Building Process Diagram" width="900"/>
  <br>
  <em><b>Figure:</b> The process for extracting information-theoretic features from an ensemble of LLMs to detect hallucinations.</em>
</p>

---

## Benchmarks and Evaluation

To validate the performance of our classifier, we tested it on a held-out set of generated answers from a diverse suite of models across several standard medical tasks. The following subdirectories contain the different evaluation tasks:

* `Clinical_trials/`: Contains the primary experiments for training and validating the hallucination classifier on the oncology clinical trials QA task.
* `Clinical_note_summarization/`: Evaluates the classifier's ability to detect hallucinations and factual inconsistencies in the summarization of clinical notes.
* `MedQA/`: Assesses performance on the MedQA (Medical reasoning) dataset, a benchmark based on US Medical Licensing Exams.
* `HealthBench/`: Provides a broad evaluation of the classifier's robustness using phisician patient dialogues (HealthBench benchmark).

Please refer to the `README.md` file in each subdirectory for detailed instructions on how to replicate the respective experiments. For a complete discussion of the methodology and results, please see our main paper.
