# Clinical Trials Evaluation (Database-Free)

This directory contains a database-free approach to evaluating large language models on clinical trial information. This experiment uses an ensemble of different models and a classifier-based approach to assess performance, rather than relying on a curated database for question answering.

## Experimental Workflow

### 1. Run Model Ensemble (`1-Run_ensemble/`)

In this step, an ensemble of large language models is used to process the clinical trial data. Each subdirectory corresponds to a specific model run:
- `DeepSeek/`
- `Gpt5/`
- `GptO3/`
- `LLama3.3-70B/`

### 2. Process Ensemble Outputs (`2-Proccess_ensemble/`)

The raw outputs from each model in the ensemble are processed in this step. This involves feature extraction and preparation for the classifier. The generated features are stored in the `Features/` directory.

### 3. Classifier (`3-Classifier/`)

A classifier is trained to evaluate the quality of the model-generated content.
- `Paragraph_analisis-combined.ipynb`: Jupyter notebook for training the classifier and analyzing its performance.
- `Data/`: Contains the data used for training and testing the classifier.

### Additional Analyses

This project also includes further analyses:

- **Coverage Analysis (`Assign_probability_to_Coverage/`)**: This analysis assigns a probability score related to the "coverage" of the generated text. The main notebook for this is `Paragraph_analisis-combined-predict_COVERAGE.ipynb`.

- **Generalization to UMLS (`Generalization_to_UMLS/`)**: This sub-experiment evaluates the model's ability to generalize to the Unified Medical Language System (UMLS). It has its own mini-pipeline of running models, processing outputs, and analysis.

### Supporting Directories

- `Features/`: Contains the features extracted from the outputs of each model in the ensemble.
