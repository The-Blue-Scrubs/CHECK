# CHECK: Learning the Phenotype of Medical Hallucinations

<img width="1461" height="979" alt="Figure_1" src="https://github.com/user-attachments/assets/5a79f882-dd1a-4929-9a63-edea9ebc9998" />

This repository contains the code for the paper "Learning the Phenotype of Medical Hallucinations". The data required to run the analyses and notebooks is stored on Zenodo to ensure long-term availability and reproducibility.

## Installation

1.  Clone this repository to your local machine:
    ```bash
    git clone [https://github.com/The-Blue-Scrubs/CHECK.git](https://github.com/The-Blue-Scrubs/CHECK.git)
    cd CHECK
    ```

2.  Install the required Python dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Data Download and Setup

All data files (`.csv` and `.json`) are hosted on Zenodo. To download and place the data in the correct directories, please run the following command from the root of this repository:

```bash
python download_data.py
```

This script will read the file_manifest.txt, download each file from our Zenodo archive, and place it in the correct location within the project structure. The total download size is approximately 40 GB.

Dataset DOI: 10.5281/zenodo.17048677

Usage
Once the data has been downloaded, you can run the analysis notebooks located in the /notebooks directory or execute the main scripts as described below.

