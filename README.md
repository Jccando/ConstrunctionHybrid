# ConstructionHybrid

**Hybrid Machine Learning and Knowledge-Enhanced Learning for Construction Cost Estimation**

This repository contains the code and experimental resources for a study on machine learning-based construction cost estimation. The project investigates conventional machine learning models, hybrid optimization-based models, temporal generalization, model interpretability, and construction-domain knowledge enhancement across multiple real-world datasets.

---

## Overview

Construction cost estimation is a challenging regression problem due to the heterogeneous characteristics of construction projects, nonlinear relationships between project attributes and cost, temporal variations, and the availability of domain-specific knowledge.

This project evaluates whether hybrid machine learning and knowledge-enhanced approaches can improve construction cost estimation compared with conventional single-model approaches.

The repository includes experiments covering:

- Conventional machine learning regression
- Ensemble and stacking methods
- Optimization-based hybrid learning
- Temporal generalization
- SHAP-based model interpretation
- Construction-domain knowledge enhancement
- Retrieval-augmented cost estimation
- Cross-dataset evaluation

The experiments use several construction- and building-related datasets, including the UCI Residential Building Dataset, NYC School Construction Authority (SCA) project data, DDC/CWICR construction knowledge data, and ComStock.

---

## Repository Structure

```text
ConstrunctionHybrid/
│
├── code/
│   ├── baseline/
│   │   ├── e1_uci.py
│   │   ├── e2_nyc.py
│   │   ├── e3_temporal.py
│   │   ├── e4_shap.py
│   │   ├── e4b_nyc_shap.py
│   │   ├── e5_ddc.py
│   │   ├── e5b_rag.py
│   │   ├── e6_comstock.py
│   │   ├── analysis.py
│   │   ├── qa_audit.py
│   │   ├── supplement.py
│   │   └── visualization scripts
│   │
│   ├── utils/
│   │   ├── datasets.py
│   │   ├── metaheuristics.py
│   │   ├── metrics.py
│   │   └── __init__.py
│   │
│   ├── catboost_info/
│   └── experimentresult/
│
├── dataset/
│   └── raw/
│       ├── comstock/
│       ├── ddc_cwicr_zh/
│       ├── nyc_sca/
│       └── uci_437/
│
└── README.md
```

### Main Components

- `code/baseline/`: Experimental scripts for different datasets and research questions.
- `code/utils/datasets.py`: Dataset loading and preprocessing utilities.
- `code/utils/metaheuristics.py`: Metaheuristic optimization methods used by hybrid models.
- `code/utils/metrics.py`: Evaluation metrics and statistical analysis utilities.
- `dataset/raw/`: Raw datasets required by the experiments.
- `experimentresult/`: Experimental outputs, tables, and figures.

---

## Environment

The experiments are implemented in **Python 3.10+**.

We recommend using a virtual environment.

### Create a virtual environment

```bash
python -m venv .venv
```

### Activate the environment

On Windows:

```bash
.venv\Scripts\activate
```

On Linux/macOS:

```bash
source .venv/bin/activate
```

### Install dependencies

Install the main scientific computing and machine learning packages:

```bash
pip install numpy pandas scipy scikit-learn
pip install xgboost lightgbm catboost ngboost
pip install shap matplotlib openpyxl pyarrow
```

---

## Dataset Preparation

The experiments use multiple datasets. Raw data should be placed under:

```text
dataset/raw/
```

The expected organization is:

```text
dataset/raw/
├── uci_437/
├── nyc_sca/
├── ddc_cwicr_zh/
└── comstock/
```

### 1. UCI Residential Building Dataset

The UCI Residential Building Dataset is used for the main methodological comparison.

Expected location:

```text
dataset/raw/uci_437/
└── Residential-Building-Data-Set.xlsx
```

The dataset is used as a regression problem for construction cost estimation.

### 2. NYC SCA Capital Projects

The NYC School Construction Authority (SCA) dataset is used to evaluate construction cost estimation on real-world public construction projects.

Expected location:

```text
dataset/raw/nyc_sca/
└── nyc_sca_capital_projects.csv
```

The experiments use project-level and phase-level information together with derived project characteristics.

### 3. DDC / CWICR Construction Knowledge Base

The DDC/CWICR dataset provides construction-domain knowledge used in the knowledge-enhanced and retrieval-augmented experiments.

Expected location:

```text
dataset/raw/ddc_cwicr_zh/
```

The knowledge base contains construction work-item and cost-related information that can be retrieved and incorporated into the prediction pipeline.

### 4. ComStock

The ComStock dataset is used as an additional evaluation dataset for building-related cost estimation.

Expected location:

```text
dataset/raw/comstock/
```

---

## Experiments

The main experimental scripts are located in:

```text
code/baseline/
```

Each experiment can be executed independently.

> **Note:** Before running the experiments, make sure that the dataset paths in `code/utils/datasets.py` match the directory structure of your local copy.

### E1 — UCI Residential Building Dataset

The E1 experiment provides the main comparison between conventional machine learning models and hybrid learning approaches.

The evaluated models include:

- Support Vector Regression (SVR)
- Random Forest (RF)
- LightGBM
- XGBoost
- CatBoost
- Artificial Neural Network (ANN)
- NGBoost
- Gaussian Process Regression (GPR)
- Stacking
- PSO-SVR
- GA-ANN

Run:

```bash
python code/baseline/e1_uci.py
```

The experiment evaluates prediction performance using metrics including:

- MAPE
- RMSE
- R²

### E2 — NYC SCA Construction Cost Estimation

The E2 experiment evaluates construction cost estimation using the NYC SCA capital project dataset.

The experiment compares conventional machine learning models with hybrid learning approaches for real-world construction projects.

Run:

```bash
python code/baseline/e2_nyc.py
```

### E3 — Temporal Generalization

Construction cost estimation models may experience performance changes when applied to projects from different time periods.

The E3 experiment evaluates the temporal generalization and robustness of the prediction models.

Run:

```bash
python code/baseline/e3_temporal.py
```

### E4 — SHAP-Based Model Interpretation

The E4 experiment investigates model interpretability using SHAP.

The analysis is used to identify important features contributing to construction cost predictions and to better understand the relationship between project characteristics and model outputs.

Run:

```bash
python code/baseline/e4_shap.py
```

For the NYC-specific SHAP analysis:

```bash
python code/baseline/e4b_nyc_shap.py
```

### E5 — DDC Knowledge-Enhanced Cost Estimation

The E5 experiment investigates whether construction-domain knowledge can improve machine learning-based cost estimation.

Information from the DDC/CWICR construction knowledge base is incorporated into the prediction process.

Run:

```bash
python code/baseline/e5_ddc.py
```

The experiment is designed to evaluate whether domain-specific construction knowledge provides additional predictive information beyond conventional project metadata.

### E5b — Retrieval-Augmented Cost Estimation

The E5b experiment investigates a retrieval-based approach for incorporating construction knowledge into cost estimation.

Instead of relying only on structured project metadata, relevant construction work items are retrieved from the DDC/CWICR knowledge base.

The retrieval component uses text similarity to identify construction work items relevant to each project and incorporates the retrieved information into the prediction features.

The experiment compares different information settings:

```text
M0: Metadata only

M1: Metadata + work-scope information

M2: Metadata + retrieval-augmented construction knowledge
```

Run:

```bash
python code/baseline/e5b_rag.py
```

This experiment provides a lightweight retrieval-augmented framework for integrating construction-domain knowledge into machine learning-based cost estimation.

### E6 — ComStock Evaluation

The E6 experiment evaluates the learning framework using the ComStock dataset.

Run:

```bash
python code/baseline/e6_comstock.py
```

---

## Evaluation Metrics

The main experiments evaluate regression performance using the following metrics.

### MAPE

Mean Absolute Percentage Error:

```text
MAPE = mean(|y - ŷ| / |y|)
```

MAPE measures the relative prediction error between the observed and predicted construction costs.

### RMSE

Root Mean Squared Error:

```text
RMSE = sqrt(mean((y - ŷ)²))
```

RMSE penalizes larger prediction errors more strongly and is used to measure the overall magnitude of prediction errors.

### R²

Coefficient of Determination:

```text
R² = 1 - SS_res / SS_tot
```

R² measures the proportion of variance in the target variable explained by the prediction model.

---

## Additional Analyses

In addition to model performance comparison, the repository includes scripts for:

- Statistical significance analysis
- Temporal analysis
- SHAP-based feature importance
- Prediction error analysis
- Knowledge-enhancement ablation studies
- Supplementary experiments
- Visualization and publication-quality figures

Supporting scripts are located under:

```text
code/baseline/
```

Examples include:

```text
analysis.py
qa_audit.py
supplement.py
viz2.py
viz_advanced.py
viz_final.py
viz_polish.py
viz_pub.py
viz_shap_inset.py
viz_tornado.py
```

These scripts can be executed after the corresponding main experiments have been completed.

---

## Reproducibility

To reproduce the experiments:

### Step 1 — Clone the repository

```bash
git clone https://github.com/Jccando/ConstrunctionHybrid.git
cd ConstrunctionHybrid
```

### Step 2 — Create the Python environment

```bash
python -m venv .venv
```

Activate the environment and install the required dependencies.

### Step 3 — Prepare the datasets

Place the required raw datasets under:

```text
dataset/raw/
```

### Step 4 — Check dataset paths

Before running an experiment, check:

```text
code/utils/datasets.py
```

and make sure that the paths correspond to your local repository structure.

### Step 5 — Run an experiment

For example:

```bash
python code/baseline/e1_uci.py
```

or:

```bash
python code/baseline/e2_nyc.py
```

### Step 6 — Inspect the results

Generated tables and figures can be found in the corresponding experiment result directories.

---

## Reproducibility Notes

For reproducible results:

- Use Python 3.10 or later.
- Keep the same dataset versions.
- Keep the random seeds unchanged.
- Use the same preprocessing procedure.
- Use the same train/test splitting strategy.
- Make sure all required dependencies are installed.
- Run the experiments using the provided scripts rather than modifying the experimental settings.

Some experiments use multiple random seeds and report aggregated performance statistics.

---

## Results

The experimental results include:

- Model performance comparisons
- Prediction error analysis
- Statistical comparisons
- Temporal generalization results
- SHAP feature importance
- Knowledge-enhancement comparisons
- Retrieval-augmented learning results
- Visualization figures

Generated experimental outputs are organized under the experiment result directories.

---

## Project Status

This repository is under active development.

The current version focuses on:

1. Hybrid machine learning for construction cost estimation.
2. Robust evaluation across multiple construction datasets.
3. Temporal generalization of cost estimation models.
4. Model interpretability using SHAP.
5. Construction-domain knowledge enhancement.
6. Retrieval-augmented construction cost estimation.

Future versions may include additional experiments, improved retrieval methods, and more advanced AI-based construction knowledge integration.

---

## Citation

If you use this repository, code, datasets, or experimental results in your research, please cite the corresponding paper.

```bibtex
@inproceedings{TODO,
  title     = {TODO},
  author    = {TODO},
  booktitle = {TODO},
  year      = {2026}
}
```

The citation information will be updated after the final publication details are available.

---

## Acknowledgements

This project makes use of publicly available datasets and open-source machine learning libraries.

We thank the original dataset creators and the developers of the open-source tools used in this research.

---

## License

Please see the repository license for details.
