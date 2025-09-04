# Clinical Note Summarization Evaluation

This directory contains the experiment for evaluating large language models on the task of clinical note summarization. The focus of this evaluation is to assess the factual consistency of the generated summaries and to detect hallucinations.

## Experimental Workflow

The experiment follows a three-step process, from running the summarization model to analyzing the results.

### 1. Run CliSum Model (`1-Run_CliSum/`)

This step involves running the ensemble of model on the dataset. The experiments are conducted under two conditions:
- `Fact/`: Generation of factual summaries.
- `Hallucination/`: Generation of summaries with induced hallucinations.

The input data for this step is located in the `Data/` directory. You will need to download it from Zenodo as described in the main project `README.md`.

### 2. Process Ensemble Outputs (`2-Proccess_ensemble/`)

After generating the summaries, this step processes the outputs (probability distributions) extracted from the model ensemble. This involves several stages:
- `1-features_creation.ipynb`: Creates features from the generated summaries. The output features are stored in the `FEATURES/` directory.
- `2-KL_div.ipynb`: Calculates the KL divergence across the ensemble of models.
- `3-Reduce_and_concatenate_Paragraph.ipynb`: Processes and combines the results for final analysis.

### 3. Analysis (`3-Analysis/`)

This final step involves analyzing the processed results to evaluate the model's performance on summarization, factual consistency, and hallucination detection.
- `Paragraph_analisis-combined.ipynb`: The main Jupyter notebook for performing the analysis.
- The `.csv` files in this directory contain the training, testing, and aggregated results used in the analysis notebook.

### Supporting Files

- `Data/`: Contains the clinical note data in JSON format.
- `FEATURES/`: Stores the features extracted from the model outputs.
- `generate_hallucination.py`: A Python script used to generate hallucinated examples for the evaluation.
