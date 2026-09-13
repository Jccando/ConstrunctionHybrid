# Code

Source code for the study **"Retrieval-Augmented Hybrid Learning for Public-Building Cost
Estimation"**.

The complete documentation — project description, dataset information, code map, usage instructions,
requirements, methodology, citations, and licence and contribution guidelines — is in the repository's
root **[`README.md`](../README.md)**.

## Quick start

```bash
pip install -r ../requirements.txt

cd code
python -m baseline.e1_uci        # UCI #437 method validation
python -m baseline.e2_nyc        # NYC SCA real public-school cost
python -m baseline.e5b_rag       # retrieval-augmented knowledge ablation (core innovation)
python -m baseline.e6_comstock   # large-scale ComStock design-stage cost
python -m baseline.qa_audit      # leakage / reproducibility / honesty audit
```

Scripts must be run **as modules from this `code/` directory** so that the `utils` package resolves.
Data is read from `../dataset/raw/`; results are written to `experimentresult/tables/` and
`experimentresult/figures/`.

## Layout

```
code/
├── utils/              # datasets.py, metrics.py, metaheuristics.py
├── baseline/           # e1–e6 experiment scripts, qa_audit, supplement, analysis, viz*
└── experimentresult/   # generated tables (CSV/JSON) and figures (PNG)
```

All code comments and docstrings are written in English so that editors, reviewers, and readers can
follow the implementation.
