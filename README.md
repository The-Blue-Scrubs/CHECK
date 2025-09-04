# CHECK: Learning the Phenotype of Medical Hallucinations

This repository contains the official code and implementation for the paper **"Learning the Phenotype of Medical Hallucinations"**. We introduce CHECK, a hybrid, self-improving framework designed to advance factual reliability in clinical language models. By combining a structured medical knowledge base with a model-agnostic classifier, CHECK can effectively detect and mitigate hallucinations in high-stakes medical applications.

<p align="center">
  <img src="https://github.com/user-attachments/assets/5a79f882-dd1a-4929-9a63-edea9ebc9998" alt="CHECK Framework Diagram" width="800"/>
  <br>
  <em><b>Figure 1:</b> The CHECK framework, illustrating the dual-pipeline approach for hallucination detection.</em>
</p>

---

## Repository Structure

* `/database_dependent_evaluation`: Contains the data and notebooks for our database-dependent pipeline (fig b). This includes experiments that (1) establish baseline hallucination rates across a suite of models and (2) measure the impact of context quality on a fixed model. The resulting labeled data is used to train our database-free classifier.
* `/database_free_evaluation`: Contains the experiments that validate the performance and applications of our database-free classifier (fig c) across a diverse set of medical tasks.
* `/notebooks`: Contains primary analysis and figure-generation notebooks.
* `/scripts`: Contains core Python scripts for data processing and model evaluation.
* `download_data.py`: Script to download the full dataset from Zenodo.
* `file_manifest.txt`: A list of all data files required for the project.

---
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

---
## Data Download and Setup

All datasets required to run the analyses (`.csv` and `.json` files) are hosted on Zenodo to ensure long-term availability and reproducibility.

1.  **Download the data:** Run the following command from the root of this repository. This will download and place all files in their correct directories according to the `file_manifest.txt`.
    ```bash
    python download_data.py
    ```
    * **Note:** The total download size is approximately **40 GB**. Please ensure you have sufficient disk space.

2.  **Zenodo Archive:** The complete dataset is archived on Zenodo and can be accessed directly at:
    * **DOI:** [`10.5281/zenodo.17048677`](https://doi.org/10.5281/zenodo.17048677)

---

## Usage
Once the data has been successfully downloaded and placed, you can reproduce the experiments and analyses.

---

## Citation

If you use this code or the associated datasets in your research, please cite our paper:

```bibtex
@misc{garciafernandez2025trustworthyaimedicinecontinuous,
      title={Trustworthy AI for Medicine: Continuous Hallucination Detection and Elimination with CHECK}, 
      author={Carlos Garcia-Fernandez and Luis Felipe and Monique Shotande and Muntasir Zitu and Elier Delgado and Ghulam Rasool and Issam El Naqa and Vivek Rudrapatna and Gilmer Valdes},
      year={2025},
      eprint={2506.11129},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
