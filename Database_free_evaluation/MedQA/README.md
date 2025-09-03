# MedQA Evaluation

This directory contains the experiment for evaluating large language models on the MedQA dataset, which is a benchmark for medical question answering.

## Experimental Workflow

### 1. Run Model Ensemble (`1-Run_ensemble/`)

An ensemble of models is run on the MedQA dataset. This is done for both the training and testing splits of the dataset.
- `Run_ensemble_train/`: Directory for running the ensemble on the training set.
- `Run_ensemble_test/`: Directory for running the ensemble on the test set.
- `Features/`: Stores features extracted during the run.

The MedQA data is located in the `Data/` directory and needs to be downloaded from Zenodo.

### 2. Process Ensemble Outputs (`2-Proccess_ensemble/`)

The outputs from the model ensemble are processed to prepare them for analysis. This directory contains the scripts and notebooks for this processing step.

### 3. Analysis (`3-Analysis/`)

The final step is to analyze the processed results to evaluate the performance of the models on the MedQA task. The analysis notebooks are located in this directory.

### Supporting Directories

- `Data/`: Contains the MedQA dataset files.
