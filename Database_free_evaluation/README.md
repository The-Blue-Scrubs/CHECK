# Database-Free Evaluation

This directory contains experiments for the database-free evaluation of large language models. Unlike the database-dependent evaluations, these tasks do not rely on a specific curated database. Instead, they assess the model's capabilities on more general medical NLP tasks.

The following subdirectories contain the different database-free evaluation benchmarks:

- `Clinical_note_summarization/`: Evaluation of clinical note summarization, focusing on factual consistency and hallucination detection.
- `Clinical_trials/`: An alternative evaluation on clinical trials using an ensemble of models and a classifier-based approach.
- `HealthBench/`: Evaluation using the HealthBench benchmark.
- `MedQA/`: Evaluation on the MedQA (Medical Question Answering) dataset.

Please refer to the `README.md` file in each subdirectory for detailed instructions on how to replicate the respective experiments.
