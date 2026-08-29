```
code/
├── baseline/      # UCI #437 baseline（Rafiei&Adeli 2018 + one modle compare）
├── models/        # hybrid learning（stacking / GA-ANN / PSO-SVR / CNN+tree）
├── utils/         # function
└── README.md
```

## envirment

- Python ≥ 3.10
- `scikit-learn`、`xgboost`、`lightgbm`、`ngboost`、`torch`、`shap`、`pandas`、`numpy`、`matplotlib`

## run's order

1. `utils/build_features.py` 
2. `baseline/run_baselines.py` 
3. `models/run_hybrid.py`
4. `utils/evaluate.py`
