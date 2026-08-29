# ConstructionHybrid

A hybrid machine learning framework for construction cost estimation.

This repository contains the code, datasets, and experimental scripts for
evaluating hybrid learning methods for construction cost estimation across
multiple real-world construction datasets.

The project investigates whether hybrid models and knowledge-enhanced
learning strategies can improve construction cost prediction compared with
conventional single-model approaches.

---

## Overview

Construction cost estimation is a challenging regression problem because
construction projects often involve heterogeneous features, nonlinear
relationships, temporal effects, and domain-specific knowledge.

This project evaluates a range of conventional machine learning models and
hybrid learning strategies, including:

- Support Vector Regression (SVR)
- Random Forest (RF)
- LightGBM
- XGBoost
- CatBoost
- Artificial Neural Networks (ANN)
- NGBoost
- Gaussian Process Regression (GPR)
- Stacking
- PSO-SVR
- GA-ANN
- Retrieval-Augmented / knowledge-enhanced cost estimation

The experiments are conducted on multiple construction-related datasets to
evaluate model effectiveness, temporal generalization, interpretability,
and the value of construction-domain knowledge.

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
│   │   └── ...
│   │
│   ├── utils/
│   │   ├── datasets.py
│   │   ├── metaheuristics.py
│   │   ├── metrics.py
│   │   └── __init__.py
│   │
│   └── catboost_info/
│
├── dataset/
│   └── raw/
│       ├── uci_437/
│       ├── nyc_sca/
│       ├── ddc_cwicr_zh/
│       └── comstock/
│
├── exprimentresult/
│   └── figures/
│
└── README.md
