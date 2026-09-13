# Retrieval-Augmented Hybrid Learning for Public-Building Cost Estimation

Reference implementation and datasets for the study **"Retrieval-Augmented Hybrid Learning for
Public-Building Cost Estimation"** (Lyu, S.), which proposes a framework that couples a stacking
ensemble of four diverse learners with a retrieval module over an open construction cost-knowledge
base, for design-stage cost estimation of public buildings.

---

## Description

Design-stage cost estimates commit the majority of a public project's total budget, yet machine
learning (ML) approaches to construction cost estimation predominantly target residential data, rely
on single algorithms, and make limited use of structured domain cost knowledge. This repository
contains the complete, reproducible implementation of a **retrieval-augmented hybrid learning
framework** that addresses those limitations.

The framework has four stages:

1. **Data sources** supply features `X`, labels `Y`, and an external cost-knowledge base.
2. **Retrieval-augmented generation (RAG)** retrieves cost knowledge `K` for each project and fuses
   it into an enhanced feature representation `Z = Φ(X, K)`.
3. **Stacking ensemble** of four base learners `f₁`–`f₄` (XGBoost, LightGBM, CatBoost, ANN) maps `Z`
   to a prediction matrix `P`, combined by a Ridge meta-learner `g(·)` into the estimate `ŷ`.
4. **Outputs** comprise the cost estimate, a prediction interval, and a SHAP explanation.

The repository contains the full experimental suite: method validation, real public-building cost
estimation, temporal generalization, large-scale design-stage estimation, knowledge-ablation ladders,
component/feature ablation, SHAP cost-driver analysis, uncertainty quantification, and a data-quality
(QA) audit that verifies the absence of data leakage and the reproducibility of the reported results.

### Reported headline results

| Finding | Value |
|---|---|
| Stacking ensemble — best average rank across the three datasets | 1.67 |
| Best `R²_log` on large-scale data (ComStock) | 0.962 |
| RAG knowledge injection (CatBoost, NYC SCA) | significant, `p = 0.0048` |
| Removing floor area from ComStock | `R²_log` collapses 0.962 → 0.505 |

---

## Dataset Information

Four data sources were used, each with a distinct role. **No dataset is redistributed here beyond
what its licence permits**; `dataset/raw/` ships the source files that are openly redistributable.

| Dataset | Role | `n` | Key attributes | Source & licence |
|---|---|---|---|---|
| **UCI #437** | Method validation | 372 | 107 features; real construction cost (V-10) | [DOI 10.24432/C5S896](https://doi.org/10.24432/C5S896) — CC BY 4.0 |
| **NYC SCA** | Real public-school cost | 6,783 | program type, phase, district, status, dates, free-text scope; USD | NYC Open Data dataset `2xh6-psuq` — NYC Open Data Terms of Use |
| **ComStock** (NREL) | Large-scale design features | 25,000 | floor area, building type, vintage, HVAC type, climate zone, stories | NREL ComStock 2024 Release 1 (AMY2018), OEDI data lake — BSD-3-Clause |
| **DDC CWICR** (Asia–China) | External cost-knowledge base | 2,681 | work items with CNY unit prices | [OpenConstructionEstimate-DDC-CWICR](https://github.com/datadrivenconstruction/OpenConstructionEstimate-DDC-CWICR) — CC BY-NC 4.0 (non-commercial) |

### Notes and honest limitations

- **UCI #437** — 372 real residential projects (Tehran, 1993–2008). All features and the target are
  real; this is the reproducible anchor of the study.
- **NYC SCA** — 14,077 public school capital-project records, of which 6,783 phases with actual
  spending ≥ \$10,000 were retained to remove near-zero administrative artifacts. These records
  **lack floor area**, which limits absolute accuracy; the consequence is quantified by the floor-area
  ablation.
- **ComStock** — 25,000 sampled public buildings with *real* design attributes but **no cost labels**.
  Labels were therefore **derived** by resource-based costing,
  `C = A · u_t · ν_v · η_h · κ_s · ξ_c · ε`, where `u_t` is a type-specific unit cost anchored to
  DDC/literature values, `ν_v / η_h / κ_s / ξ_c` are vintage/HVAC/story/climate multipliers, and
  `ε ~ LogN(0, 0.28)` is multiplicative lognormal noise. These are *not* market costs; this is
  disclosed explicitly in the manuscript.
- Although the DDC base is denominated in CNY and the NYC SCA labels in USD, the retrieval module
  emits **currency-invariant** features (price ratios and distributional statistics) rather than
  absolute prices, so no exchange-rate calibration is applied.

### Repository layout of the data

```
dataset/raw/
├── uci_437/Residential-Building-Data-Set.xlsx     # UCI #437, 107 features + cost/sales targets
├── nyc_sca/nyc_sca_capital_projects.csv           # NYC SCA capital projects (phase level)
├── comstock/baseline_metadata_only.parquet        # NREL ComStock baseline building metadata
├── comstock/data_dictionary.tsv                   # ComStock column dictionary
└── ddc_cwicr_zh/DDC_CWICR_ZH_CHINA_Catalog.csv    # DDC work items, CNY unit prices
```

The upstream DDC repository also publishes a full resource-level parquet
(`ZH_CHINA_workitems_costs_resources_DDC_CWICR.parquet`) in the 95-column CWICR master
schema. It is **not redistributed here** — obtain it from the source repository if you need the
resource-decomposed base. This study uses only the compact catalog above.

---

## Code Information

```
code/
├── utils/
│   ├── datasets.py         # Data loaders for all four sources; defines ROOT, TABLES, FIGURES
│   ├── metrics.py          # MAPE, RMSE, R², and the Wilcoxon signed-rank significance test
│   └── metaheuristics.py   # Dependency-free PSO (for SVR) and GA (for ANN) optimizers
├── baseline/
│   ├── e1_uci.py           # E1 · UCI #437 method validation (8 singles vs 3 hybrids, 5 seeds)
│   ├── e2_nyc.py           # E2 · NYC SCA real public-school cost (phase level)
│   ├── e3_temporal.py      # E3 · temporal generalization (train earlier → test recent projects)
│   ├── e4_shap.py          # E4 · UCI design-stage cost drivers (SHAP)
│   ├── e4b_nyc_shap.py     # E4b · NYC public-school real-cost drivers (SHAP)
│   ├── e5_ddc.py           # E5 · knowledge-augmentation ablation ladder M0 → M1 → M2 (DDC unit prices)
│   ├── e5b_rag.py          # E5b · retrieval-augmented ablation M0 → M1 → M2_RAG (the core innovation)
│   ├── e6_comstock.py      # E6 · large-scale design-stage cost (ComStock + DDC-anchored labels)
│   ├── supplement.py       # Uncertainty quantification (NGBoost intervals) + SHAP dependence
│   ├── analysis.py         # Stacking component ablation, cross-dataset ranks, Nemenyi CD, beeswarm
│   ├── qa_audit.py         # QA audit: data leakage, reproducibility, and honesty checks
│   └── viz*.py             # Figure generation (main comparison, heatmap, CD, ablations, SHAP, tornado…)
└── experimentresult/       # All outputs (created automatically): tables/*.csv, figures/*.png
```

**Model family covered.** Single learners: grid-search SVR, Random Forest, LightGBM, XGBoost,
CatBoost, ANN, NGBoost, Gaussian Process Regression. Hybrid learners: Stacking, PSO-SVR, GA-ANN.

**Evaluation protocol.** 5 random seeds (3 for the large-scale ComStock run), `mean ± std` reporting,
pooled Wilcoxon signed-rank tests on per-sample absolute errors, and probabilistic interval coverage
for the distributional models (NGBoost, GPR). Preprocessing (scalers, TF-IDF vectorizers) is fitted on
training folds **or** on the external DDC base only — never on test data.

---

## Usage Instructions

### 1. Requirements

Python **3.10+** is required. Install the dependencies:

```bash
pip install -r requirements.txt
```

The full list is given in the [Requirements](#requirements) section below.

### 2. Expected repository layout

Data is resolved relative to the repository root, e.g.
`dataset/raw/uci_437/Residential-Building-Data-Set.xlsx`. Keep the `dataset/` tree exactly as shipped.

Outputs are written to `code/experimentresult/tables/` (CSV/JSON) and
`code/experimentresult/figures/` (PNG). These directories are created automatically on import of
`utils.datasets`, and by the scripts that call `mkdir(exist_ok=True)`.

### 3. Running the experiments

**Run every script as a module from the `code/` directory** so that the `utils` package resolves:

```bash
cd code

# Main experiments (each is standalone and writes its own results CSV/PNG)
python -m baseline.e1_uci          # UCI #437 method validation      (~minutes)
python -m baseline.e2_nyc          # NYC SCA real public-school cost
python -m baseline.e3_temporal     # temporal generalization
python -m baseline.e4_shap         # UCI SHAP cost drivers
python -m baseline.e4b_nyc_shap    # NYC SHAP cost drivers
python -m baseline.e5_ddc          # knowledge-augmentation ladder M0/M1/M2
python -m baseline.e5b_rag         # retrieval-augmented ladder M0/M1/M2_RAG
python -m baseline.e6_comstock     # large-scale ComStock design-stage cost
python -m baseline.supplement      # uncertainty quantification + SHAP dependence
```

### 4. Analyses that depend on earlier results

These read the CSVs produced in step 3, so run them afterwards:

```bash
cd code
python -m baseline.analysis        # component ablation, cross-dataset ranks, Nemenyi CD, SHAP beeswarm
python -m baseline.viz_pub         # Figures 3, 5, 6, 8, 9, 10  (CSV-based, no training)
python -m baseline.viz_final       # alternative publication-grade versions of the same figures
python -m baseline.viz_polish      # 3-panel MAPE main figure + per-dataset versions
python -m baseline.viz_shap_inset  # standalone SHAP inset for the method diagram
python -m baseline.viz2            # Fig 4 parity + Fig 7 stability violin  (retrains models)
python -m baseline.viz_advanced    # waterfall / tornado / radar / cost-landscape (self-contained)
python -m baseline.viz_tornado     # clean tornado chart (self-contained)
```

### 5. Reproducibility audit

```bash
cd code
python -m baseline.qa_audit
```

This audit verifies three things and writes `experimentresult/tables/qa_audit.json`:

- **L1 — Data leakage:** the scaler is fitted only on the training set; features never contain the
  target cost; the RAG TF-IDF vectorizer is fitted on the external DDC base only.
- **L2 — Reproducibility:** fixed-seed re-runs of key configurations reproduce identical values, and
  across-seed results fall inside the reported `mean ± 2 std` band.
- **L3 — Honesty:** derived labels (E6) are disclosed, and metric definitions are correct
  (MAPE in original cost space, `R²_log` in log space).

### 6. Reading the code

Every module is self-documenting: each file opens with a docstring stating its purpose, inputs, and
outputs, and unit conventions (`MAPE` in %, `R²_log` in log space, costs in USD unless stated).
Figures are regenerated from `experimentresult/tables/*.csv`, so plot styling can be changed without
re-running the models.

---

## Requirements

Dependencies are listed in `requirements.txt`. Core:

| Package | Used for |
|---|---|
| `numpy`, `pandas` | Data handling and arrays |
| `scikit-learn` | SVR, RF, ANN, GPR, stacking, preprocessing, CV, metrics |
| `xgboost` | XGBoost base learner |
| `lightgbm` | LightGBM base learner |
| `catboost` | CatBoost base learner |
| `ngboost` | Probabilistic regression and uncertainty intervals |
| `shap` | SHapley Additive exPlanations cost-driver analysis |
| `scipy` | Wilcoxon signed-rank test |
| `matplotlib` | All figures |
| `openpyxl` | Reading the UCI `.xlsx` workbook |
| `pyarrow` | Reading the ComStock / DDC `.parquet` files |

**Computing infrastructure.** The study was run on a single desktop workstation with a 12–16-core CPU
and 32 GB of RAM, on 64-bit Windows 11, using Python 3.13. **No GPU is required** — every model
trains on the CPU. All model fitting is deliberately restricted to a single thread (`n_jobs=1`),
which suppresses the non-deterministic reduction order that parallel tree construction can introduce,
so repeated runs with identical seeds reproduce identical results. Reported runtimes are therefore
longer than they would be multi-threaded. The largest run (E6, 25,000 buildings × 3 seeds × 6 models)
is the most time-consuming; all remaining experiments complete within minutes.

---

## Methodology

### Feature construction and the knowledge-ablation ladder

Project metadata `X₀` and 24 binary work-scope indicators `X_s` (extracted from free-text
descriptions via the DDC taxonomy) are combined with retrieved knowledge. The ablation ladder is:

```
Z_M0 = X_0                          # metadata only
Z_M1 = [X_0 ; X_s]                  # + work-scope knowledge
Z_M2 = [X_0 ; X_s ; z_rag]          # + DDC unit-price knowledge  (E5)
Z_M2RAG = [X_0 ; X_s ; z_rag]       # + retrieval features        (E5b)
```

Each successive level is tested against the previous one with a pooled Wilcoxon signed-rank test.

### Retrieval-augmented cost-knowledge features

Each DDC work item `j` has text `t_j` and unit price `p_j`. TF-IDF vectors are fitted **on the DDC
corpus only** to prevent leakage. Each project description `d` is embedded as `e_d`, and the top-10
items are retrieved by cosine similarity:

```
sim(d, j) = (e_d · e_j) / (‖e_d‖ ‖e_j‖)
K = top-10_j sim(d, j)
```

The aggregated cost-knowledge vector captures central tendency, spread, and retrieval confidence:

```
z_rag = [ mean(p_K), median(p_K), max p, σ_p, mean(sim), |depts| ]
```

### Stacking ensemble

Four diverse base learners — XGBoost, LightGBM, CatBoost, and an ANN (with standardized targets and
early stopping) — are combined by a Ridge meta-learner over 3–5-fold cross-validated out-of-fold
predictions. Metaheuristic hybrids (PSO-SVR, GA-ANN) and distributional models (NGBoost, GPR) are
included as strong single baselines and for uncertainty quantification.

### Explainability

SHAP TreeExplainer values are computed on held-out test folds. Global importance ranks the cost
drivers; dependence plots characterize marginal effects (e.g. project duration, construction phase,
work scope). The manuscript reports that duration, construction phase, and work scopes dominate, and
that removing floor area collapses `R²_log` from 0.962 to 0.505 on ComStock.

---

## Citations

If you use this code or these derived datasets, please cite the paper:

```bibtex
@article{lyu2026retrieval,
  title  = {Retrieval-Augmented Hybrid Learning for Public-Building Cost Estimation},
  author = {Lyu, Shuojun},
  year   = {2026},
  note   = {School of Engineering, Shanwei Institute of Technology, Shanwei, China}
}
```

Please also cite the original data sources:

```bibtex
@article{rafiei2018novel,
  title   = {Novel Machine-Learning Model for Estimating Construction Costs Considering
             Economic Variables and Indexes},
  author  = {Rafiei, Mohammad Hossein and Adeli, Hojjat},
  journal = {Journal of Construction Engineering and Management},
  volume  = {144},
  number  = {12},
  pages   = {04018106},
  year    = {2018},
  doi     = {10.1061/(ASCE)CO.1943-7862.0001570}
}
```

- **UCI Residential Building Data Set (#437)** — Rafiei & Adeli (2018). DOI
  [10.24432/C5S896](https://doi.org/10.24432/C5S896). Licensed CC BY 4.0.
- **NYC School Construction Authority, "Capital Project Schedules and Budgets"** — NYC Open Data,
  dataset identifier `2xh6-psuq`. NYC Open Data Terms of Use.
- **NREL ComStock, 2024 Release 1 (AMY2018), baseline building metadata** — Open Energy Data
  Initiative data lake. Parent collection *End-Use Load Profiles for the U.S. Building Stock*,
  DOI [10.25984/1876417](https://doi.org/10.25984/1876417). Licensed BSD-3-Clause.
- **DataDrivenConstruction (DDC) Open Construction Estimate (CWICR), Asia–China catalog** —
  <https://github.com/datadrivenconstruction/OpenConstructionEstimate-DDC-CWICR>. Licensed
  CC BY-NC 4.0 (non-commercial) for the DDC compilation, free for research, teaching, evaluation,
  and non-profit use; commercial use requires a separate DDC commercial licence. Source: Beijing
  municipal construction quota (2012), valued under GB 50500.

Method references central to the implementation:

- Lundberg & Lee — SHapley Additive exPlanations (SHAP). DOI
  [10.48550/arXiv.1705.07874](https://doi.org/10.48550/arXiv.1705.07874)
- Breiman — Random Forests. DOI [10.1023/A:1010933404324](https://doi.org/10.1023/A:1010933404324)
- Duan et al. — NGBoost: natural gradient boosting for probabilistic prediction. DOI
  [10.48550/arXiv.1910.03225](https://doi.org/10.48550/arXiv.1910.03225)

---

## License & Contribution Guidelines

### License

The **code** in this repository is released, consistent with the manuscript's Data Availability
Statement, under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** licence — see
<https://creativecommons.org/licenses/by/4.0/>. You are free to share and adapt the material for any
purpose, provided appropriate credit is given.

**Dataset licensing is separate and follows each provider.** Redistribution in `dataset/raw/` is
limited to sources whose licences permit it (CC BY 4.0 for UCI #437; BSD-3-Clause for ComStock). NYC
SCA data are redistributed subject to the NYC Open Data Terms of Use. The DDC CWICR base is under
**CC BY-NC 4.0**, so the copy shipped here is restricted to non-commercial use — for any commercial
use of that data you must obtain a separate licence from DataDrivenConstruction. Please consult each
source's terms before further redistribution.

### Contribution guidelines

Contributions, bug reports, and reproducibility reports are welcome.

1. **Open an issue first** for anything beyond a trivial fix, describing the problem, the expected
   behaviour, and the command you ran.
2. **Fork and branch** — create a topic branch (`fix/…`, `feat/…`, `docs/…`) off the default branch.
3. **Preserve reproducibility.** Any change that alters numerical results must fix `random_state`
   and be reported alongside the before/after values. Do not remove or relax the seed handling.
4. **Keep outputs reproducible.** Figures must be regenerable from `experimentresult/tables/*.csv`.
5. **Report results honestly.** If a change weakens a reported result, state so explicitly in the
   pull request rather than adjusting the evaluation protocol.
6. **Write code comments and docstrings in English**, so that editors, reviewers, and readers can
   follow the implementation.
7. Open a pull request against the default branch with a clear description of the change and the
   commands used to verify it.

### Reporting problems

When reporting a problem, please include: the script you ran, the exact command, your Python and
package versions (`pip freeze`), and the full error output. For numerical discrepancies, include the
resulting value alongside the value you expected.
